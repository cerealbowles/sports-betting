import time
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, date as _date
from zoneinfo import ZoneInfo
import odds_api
import cfb_model
import football_total_model

_ET = ZoneInfo('America/New_York')

def _today_et():
    return datetime.now(_ET).strftime('%Y-%m-%d')

def _event_date_et(date_str):
    """Convert an ESPN event's UTC ISO timestamp to its ET calendar date.
    Late kickoffs (8pm+ ET) land past midnight UTC, so comparing the raw
    UTC string against an ET date string (e.g. via startswith) misses them."""
    if not date_str:
        return ''
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.astimezone(_ET).strftime('%Y-%m-%d')
    except ValueError:
        return ''

ESPN_CFB         = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
ESPN_CFB_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
# groups=80 = FBS (I-A) — confirmed live against ESPN's API. Without it the
# scoreboard mixes in FCS/D-II "buy game" opponents that have no meaningful
# season stats and would otherwise blow up team-stat/model-input quality.
_FBS_GROUP = 80

_cache = {}
# scoreboard TTL matches nfl_api.py's reasoning: 120s is fresh enough for a
# manual "↻ Refresh" during a live game to actually show something new.
# summary (weather only for CFB — no injuries block exists on this sport's
# ESPN summary payload, confirmed live) doesn't need that freshness.
_TTL = {'scoreboard': 120, 'season_log': 3600, 'summary': 600}


def _parse_linescores(comp, regulation_periods=4):
    """Quarter-by-quarter scoring from ESPN's per-competitor `linescores`
    array — only present once a game has started. Always includes all
    `regulation_periods` columns, padding not-yet-played quarters with None
    so the table doesn't jump around as the game progresses, plus an
    OT/OT2/... column for each overtime actually played. Values come back
    as floats from ESPN (e.g. 10.0) even though scores are always whole
    numbers — cast to int for display. Returns None pre-kickoff."""
    competitors = comp.get('competitors', [])
    home_ls = next((c.get('linescores') for c in competitors if c.get('homeAway') == 'home'), None) or []
    away_ls = next((c.get('linescores') for c in competitors if c.get('homeAway') == 'away'), None) or []
    if not home_ls and not away_ls:
        return None
    n = max(len(home_ls), len(away_ls), regulation_periods)

    def _val(arr, i):
        if i >= len(arr):
            return None
        v = arr[i].get('value')
        return int(v) if v is not None else None

    periods = []
    for i in range(n):
        if i < regulation_periods:
            label = str(i + 1)
        else:
            ot_num = i - regulation_periods + 1
            label = 'OT' if ot_num == 1 else f'OT{ot_num}'
        periods.append({'label': label, 'away': _val(away_ls, i), 'home': _val(home_ls, i)})
    return periods


def _cached_get(url, params, key, ttl):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < ttl:
            return data
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return _cache[key][0] if key in _cache else None
    _cache[key] = (data, now)
    return data


def _parse_record(records, name):
    for rec in records:
        if rec.get('name', '').lower() == name.lower():
            return rec.get('summary', '0-0')
    return '—'


def _record_to_wl(summary):
    """'6-1' → (6, 1)"""
    try:
        parts = summary.split('-')
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def _parse_rank(competitor):
    """ESPN's curatedRank.current: 1-25 for ranked teams, 99 for unranked."""
    try:
        return int((competitor.get('curatedRank') or {}).get('current', 99))
    except (TypeError, ValueError):
        return 99


# ESPN's `team.conferenceId` → FBS conference name. Hardcoded rather than
# fetched live: confirmed against the standings endpoint
# (site.api.espn.com/.../standings?group=80) and conference realignment is
# an offseason event, not something that needs a live lookup on every page
# load. Non-FBS opponents (buy games) carry conference ids outside this map
# — those fall back to 'FCS' in _conference_name below.
_CFB_CONFERENCES = {
    '151': 'American',
    '1':   'ACC',
    '4':   'Big 12',
    '5':   'Big Ten',
    '12':  'C-USA',
    '18':  'Independent',
    '15':  'MAC',
    '17':  'Mountain West',
    '9':   'Pac-12',
    '8':   'SEC',
    '37':  'Sun Belt',
}


def _conference_name(conference_id):
    return _CFB_CONFERENCES.get(str(conference_id), 'FCS')


def _get_cfb_season():
    """CFB season year currently in progress or most recently completed.
    Regular season kicks off in late August, so anything from August on
    belongs to that calendar year's season."""
    today = datetime.now(_ET).date()
    return today.year if today.month >= 8 else today.year - 1


# ── Season game log ────────────────────────────────────────────────────────────

def _get_season_game_log(season):
    """
    Fetch and cache all completed FBS regular-season games for `season`.
    Makes up to 15 weekly API calls; each week cached for 1 hour.
    Returns list of {game_date, home_name, away_name, home_score, away_score, home_won}.
    """
    key = f'cfb_season_log_{season}'
    now_ts = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now_ts - ts < _TTL['season_log']:
            return data

    games = []
    for week in range(1, 16):
        week_data = _cached_get(
            ESPN_CFB,
            # `dates` (not `season`) is what actually selects the year on
            # this endpoint — same quirk nfl_bootstrap.py documented for the
            # NFL scoreboard. limit=400 covers every FBS-involved game in a
            # single week (confirmed live: a full week tops out under 90).
            {'seasontype': 2, 'week': week, 'season': season, 'dates': season,
             'groups': _FBS_GROUP, 'limit': 400},
            f'cfb_week_{season}_{week}',
            _TTL['season_log'],
        )
        if not week_data:
            continue
        found_completed = False
        for event in week_data.get('events', []):
            comp  = event.get('competitions', [{}])[0]
            state = comp.get('status', {}).get('type', {}).get('state', 'pre')
            if state != 'post':
                continue
            tmap   = {c['homeAway']: c for c in comp.get('competitors', [])}
            home_c = tmap.get('home', {})
            away_c = tmap.get('away', {})
            try:
                h_score = int(home_c.get('score', 0) or 0)
                a_score = int(away_c.get('score', 0) or 0)
                h_name  = home_c.get('team', {}).get('displayName', '')
                a_name  = away_c.get('team', {}).get('displayName', '')
                if not h_name or not a_name:
                    continue
                h_rank = _parse_rank(home_c)
                a_rank = _parse_rank(away_c)
                games.append({
                    'game_date':  event.get('date', '')[:10],
                    'home_name':  h_name,
                    'away_name':  a_name,
                    'home_score': h_score,
                    'away_score': a_score,
                    'home_won':   h_score > a_score,
                    # Opponent context at the time this game was played — used
                    # to discount PPG/PPG-allowed for cupcake/G5 non-conference
                    # wins vs. a P4 or ranked opponent (see _opponent_weight).
                    'home_conf':  _conference_name(home_c.get('team', {}).get('conferenceId')),
                    'away_conf':  _conference_name(away_c.get('team', {}).get('conferenceId')),
                    'home_rank':  h_rank if h_rank <= 25 else None,
                    'away_rank':  a_rank if a_rank <= 25 else None,
                })
                found_completed = True
            except Exception:
                pass
        # Stop early if a week returned no completed games and we're past week 3
        # (season not started yet, or season has ended and weeks are empty)
        if not found_completed and week > 3:
            break

    _cache[key] = (games, now_ts)
    return games


# A team with fewer than this many games played this season gets its
# win%/split%/ppg/ppg-allowed blended with last season's final numbers —
# otherwise every team starts week 1 at an identical 0-0 (50%) record and
# factors #3-6 in cfb_model.predict() contribute exactly 0 for the whole
# non-conference slate. Same approach and threshold as nfl_api.py.
EARLY_SEASON_GAMES = 4


def _get_prior_season_team_stats(season):
    """Final wins/losses, home/road split, and ppg/ppg_allowed for every team
    from `season` (the season *before* the one currently in progress). Used
    to blend in prior-season signal for early-current-season games — see
    EARLY_SEASON_GAMES."""
    key = f'cfb_prior_stats_{season}'
    now_ts = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now_ts - ts < _TTL['season_log']:
            return data

    games = _get_season_game_log(season)
    wl = defaultdict(lambda: {'wins': 0, 'losses': 0, 'split_w': 0, 'split_l': 0})
    for g in games:
        h, a, h_won = g['home_name'], g['away_name'], g['home_won']
        wl[h]['wins']    += 1 if h_won else 0
        wl[h]['losses']  += 0 if h_won else 1
        wl[h]['split_w'] += 1 if h_won else 0
        wl[h]['split_l'] += 0 if h_won else 1
        wl[a]['wins']    += 0 if h_won else 1
        wl[a]['losses']  += 1 if h_won else 0
        wl[a]['split_w'] += 0 if h_won else 1
        wl[a]['split_l'] += 1 if h_won else 0

    ppg = _compute_team_season_stats(games)
    result = {
        t: {**v, 'ppg': ppg.get(t, {}).get('ppg'), 'ppg_allowed': ppg.get(t, {}).get('ppg_allowed')}
        for t, v in wl.items()
    }
    _cache[key] = (result, now_ts)
    return result


def _apply_prior_season_blend(team, prior_stats):
    """When `team` has played fewer than EARLY_SEASON_GAMES games this
    season, blend its current-season win%/split%/ppg/ppg_allowed with last
    season's final numbers, weighted toward current-season data as more of
    it accumulates. Sets 'blend_*' keys consumed by cfb_model.predict();
    leaves the raw wins/losses/ppg fields untouched so the game card still
    displays the team's actual current-season record."""
    cur_games = team.get('wins', 0) + team.get('losses', 0)
    if cur_games >= EARLY_SEASON_GAMES:
        return
    prior = prior_stats.get(team.get('name', ''))
    if not prior:
        return

    w = cur_games / EARLY_SEASON_GAMES  # weight given to current-season data

    cur_win_pct = team['wins'] / cur_games if cur_games else 0.5
    prior_total = prior['wins'] + prior['losses']
    prior_win_pct = prior['wins'] / prior_total if prior_total else 0.5
    team['blend_win_pct'] = w * cur_win_pct + (1 - w) * prior_win_pct

    cur_split_total = team.get('split_w', 0) + team.get('split_l', 0)
    cur_split_pct = team['split_w'] / cur_split_total if cur_split_total else 0.5
    prior_split_total = prior['split_w'] + prior['split_l']
    prior_split_pct = prior['split_w'] / prior_split_total if prior_split_total else 0.5
    team['blend_split_pct'] = w * cur_split_pct + (1 - w) * prior_split_pct

    cur_ppg = team.get('ppg')
    if prior.get('ppg') is not None:
        team['blend_ppg'] = w * cur_ppg + (1 - w) * prior['ppg'] if cur_ppg is not None else prior['ppg']
    cur_ppga = team.get('ppg_allowed')
    if prior.get('ppg_allowed') is not None:
        team['blend_ppg_allowed'] = (
            w * cur_ppga + (1 - w) * prior['ppg_allowed'] if cur_ppga is not None else prior['ppg_allowed']
        )


# Power-4 opponents count more toward PPG/PPG-allowed; Group of 5 counts less;
# FCS ("buy game") opponents barely count at all. Ranked opponents (regardless
# of conference) get an extra bump — beating/scoring on a ranked team is more
# informative than the same score against an unranked team in the same league.
# This targets the "48 ppg but it was against three cupcakes" distortion that
# shows up in the first 1-4 weeks of every CFB season.
_POWER_CONFS = {'ACC', 'Big 12', 'Big Ten', 'Pac-12', 'SEC'}
_G5_CONFS    = {'American', 'C-USA', 'MAC', 'Mountain West', 'Sun Belt'}


def _opponent_weight(conference, rank):
    if conference in _POWER_CONFS:
        weight = 1.15
    elif conference in _G5_CONFS:
        weight = 0.75
    elif conference == 'FCS':
        weight = 0.3
    else:  # Independent (Notre Dame, UMass, UConn, Army...) — mixed schedules
        weight = 1.0
    if rank is not None:
        weight *= 1.35
    return weight


def _compute_team_season_stats(games):
    """Compute {team_name: {ppg, ppg_allowed}} from season game scores,
    weighting each game by the opponent's tier (see _opponent_weight) so a
    stat line padded against weak non-conference opponents doesn't look the
    same as one built against real competition."""
    pts = defaultdict(lambda: {'for': 0.0, 'against': 0.0, 'w': 0.0, 'g': 0})
    for g in games:
        hw = _opponent_weight(g.get('away_conf'), g.get('away_rank'))
        aw = _opponent_weight(g.get('home_conf'), g.get('home_rank'))
        pts[g['home_name']]['for']     += g['home_score'] * hw
        pts[g['home_name']]['against'] += g['away_score'] * hw
        pts[g['home_name']]['w']       += hw
        pts[g['home_name']]['g']       += 1
        pts[g['away_name']]['for']     += g['away_score'] * aw
        pts[g['away_name']]['against'] += g['home_score'] * aw
        pts[g['away_name']]['w']       += aw
        pts[g['away_name']]['g']       += 1
    return {
        t: {
            'ppg':         round(v['for']     / v['w'], 1),
            'ppg_allowed': round(v['against'] / v['w'], 1),
        }
        for t, v in pts.items() if v['w'] > 0
    }


def _team_recent_form(games, team_name, n=3):
    """Last n W/L results for team_name from game_log, newest first.
    Display-only — see _team_recent_form_score for the opponent-weighted
    version the model actually uses."""
    tg = []
    for g in games:
        if g['home_name'] == team_name:
            tg.append((g['game_date'], 'W' if g['home_won'] else 'L'))
        elif g['away_name'] == team_name:
            tg.append((g['game_date'], 'W' if not g['home_won'] else 'L'))
    tg.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in tg[:n]]


def _team_recent_form_score(games, team_name, n=3):
    """Opponent-weighted win rate over the last n games (same tier weights as
    _compute_team_season_stats) — a 3-0 start padded with FCS/G5 cupcakes
    scores lower than a 2-1 start that includes a ranked win. Returns None
    if the team hasn't played yet."""
    tg = []
    for g in games:
        if g['home_name'] == team_name:
            won, opp_conf, opp_rank = g['home_won'], g.get('away_conf'), g.get('away_rank')
        elif g['away_name'] == team_name:
            won, opp_conf, opp_rank = not g['home_won'], g.get('home_conf'), g.get('home_rank')
        else:
            continue
        tg.append((g['game_date'], won, _opponent_weight(opp_conf, opp_rank)))
    tg.sort(key=lambda x: x[0], reverse=True)
    recent = tg[:n]
    total_w = sum(w for _, _, w in recent)
    if not recent or total_w == 0:
        return None
    return sum(w for _, won, w in recent if won) / total_w


def _team_rest_days(games, team_name, game_date):
    """Days between team_name's last completed game and game_date."""
    past = [g['game_date'] for g in games
            if (g['home_name'] == team_name or g['away_name'] == team_name)
            and g['game_date'] < game_date]
    if not past:
        return None
    try:
        last = datetime.strptime(max(past), '%Y-%m-%d').date()
        curr = datetime.strptime(game_date, '%Y-%m-%d').date()
        return (curr - last).days
    except Exception:
        return None


# ── Per-game detail: weather ──────────────────────────────────────────────────
# Not present on the scoreboard payload — one extra cached request per game.
# Unlike NFL, ESPN's CFB summary payload has no `injuries` key at all
# (confirmed live), so there's no injury report to show for this sport.

def _get_game_summary(event_id):
    if not event_id:
        return None
    return _cached_get(ESPN_CFB_SUMMARY, {'event': event_id}, f'cfb_summary_{event_id}', _TTL['summary'])


def _prefetch_game_summaries(events):
    """Warms _get_game_summary's cache for every event in parallel — a full
    FBS week runs 60-90 games (vs. the NFL's ~16), so fetching summaries
    one at a time on the week page would be the dominant cost of the load.
    Same fix nfl_api.py applies for the same reason."""
    event_ids = [e.get('id') for e in events if e.get('id')]
    if not event_ids:
        return
    with ThreadPoolExecutor(max_workers=min(len(event_ids), 8)) as pool:
        list(pool.map(_get_game_summary, event_ids))


def _parse_weather(summary):
    """None for domes/retractable roofs — ESPN omits the weather block entirely for those."""
    if not summary:
        return None
    wx = (summary.get('gameInfo') or {}).get('weather')
    if not wx or wx.get('temperature') is None:
        return None
    return {
        'temp':       wx.get('temperature'),
        'gust_mph':   wx.get('gust'),
        'precip_pct': wx.get('precipitation') or None,
    }


# ── Per-game detail: team stats (live/final only) ───────────────────────────
# ESPN's summary payload carries a boxscore.teams[] array (one entry per side,
# each a flat list of {name, displayValue} stats) once a game has kicked off.
# Rendered as the "Team Stats" panel in the details sheet, replacing the model
# factors panel while the game is live/final.
_TEAM_STAT_ORDER = [
    ('totalYards',          'Total Yards'),
    ('turnovers',           'Turnovers'),
    ('firstDowns',          '1st Downs'),
    ('totalPenaltiesYards', 'Penalties'),
    ('thirdDownEff',        '3rd Down'),
    ('fourthDownEff',       '4th Down'),
    ('possessionTime',      'Possession'),
]


def _stat_magnitude(val):
    """Reduce an ESPN stat's displayValue ('9-75', '2/8', '14:25', '98') to a
    single comparable number for sizing the away/home split bar."""
    if val is None:
        return 0.0
    s = str(val).strip()
    try:
        if '-' in s and s.count('-') == 1 and not s.startswith('-'):
            return float(s.split('-')[-1])
        if '/' in s:
            num, den = s.split('/')
            den = float(den)
            return (float(num) / den) if den else 0.0
        if ':' in s:
            m, sec = s.split(':')
            return float(m) * 60 + float(sec)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_team_stats(summary, home_id, away_id):
    """Returns a list of {label, away, home, away_pct, home_pct} rows for the
    live/final team-stats panel, or None pre-kickoff / if ESPN hasn't
    populated the boxscore yet."""
    if not summary or not home_id or not away_id:
        return None
    teams = (summary.get('boxscore') or {}).get('teams') or []
    if len(teams) < 2:
        return None
    stat_maps = {}
    for t in teams:
        tid = str((t.get('team') or {}).get('id') or '')
        stat_maps[tid] = {s.get('name'): s.get('displayValue') for s in (t.get('statistics') or [])}
    home_stats = stat_maps.get(str(home_id))
    away_stats = stat_maps.get(str(away_id))
    if not home_stats or not away_stats:
        return None
    rows = []
    for key, label in _TEAM_STAT_ORDER:
        away_val = away_stats.get(key)
        home_val = home_stats.get(key)
        if away_val is None and home_val is None:
            continue
        a_mag = _stat_magnitude(away_val)
        h_mag = _stat_magnitude(home_val)
        total = a_mag + h_mag
        away_pct = (a_mag / total * 100) if total > 0 else 50.0
        home_pct = 100 - away_pct
        rows.append({
            'label': label, 'away': away_val, 'home': home_val,
            'away_pct': round(away_pct, 1), 'home_pct': round(home_pct, 1),
        })
    return rows or None


# ── Live scores ────────────────────────────────────────────────────────────────

def get_live_scores(date_str=None):
    """
    Returns {game_key: score_dict} for CFB games on `date_str` (YYYY-MM-DD ET,
    defaults to today) with a 2-min cache. Passing a past date lets an open bet
    from a prior day keep showing its final score until the bet is closed.
    """
    from odds_api import _normalize
    target_str = date_str or _today_et()
    params = {'groups': _FBS_GROUP, 'limit': 400}
    if date_str:
        params['dates'] = target_str.replace('-', '')
    data = _cached_get(ESPN_CFB, params, f'cfb_scores_{target_str}', 120)
    scores = {}
    if not data:
        return scores
    for event in data.get('events', []):
        if _event_date_et(event.get('date', '')) != target_str:
            continue
        comp      = event.get('competitions', [{}])[0]
        status_obj = comp.get('status', {}).get('type', {})
        state     = status_obj.get('state', 'pre')
        if state == 'in':
            status = 'Live'
        elif state == 'post':
            status = 'Final'
        else:
            status = 'Preview'
        teams = {}
        for competitor in comp.get('competitors', []):
            side   = competitor.get('homeAway', 'home')
            team   = competitor.get('team', {})
            teams[side] = {
                'name':  team.get('displayName', ''),
                'abbr':  team.get('abbreviation', ''),
                'id':    team.get('id'),
                'score': competitor.get('score'),
                'id':    team.get('id'),
            }
        home = teams.get('home', {})
        away = teams.get('away', {})
        if not home or not away:
            continue
        gk = f"{_normalize(home['name'])}_{_normalize(away['name'])}"
        period = None
        if status == 'Live':
            period = status_obj.get('detail', '')
        elif status == 'Final':
            period = 'Final'
        scores[gk] = {
            'status':     status,
            'away_abbr':  away.get('abbr', ''),
            'home_abbr':  home.get('abbr', ''),
            'away_logo':  f"https://a.espncdn.com/i/teamlogos/ncaa/500/{away.get('id')}.png" if away.get('id') else '',
            'home_logo':  f"https://a.espncdn.com/i/teamlogos/ncaa/500/{home.get('id')}.png" if home.get('id') else '',
            'away_score': away.get('score'),
            'home_score': home.get('score'),
            'period':     period,
            'away_logo':  f"https://a.espncdn.com/i/teamlogos/ncaa/500/{away.get('id')}.png" if away.get('id') else None,
            'home_logo':  f"https://a.espncdn.com/i/teamlogos/ncaa/500/{home.get('id')}.png" if home.get('id') else None,
        }
    return scores


def get_live_game_states(date_str=None):
    """
    Returns {game_key: state_dict} for CFB games on `date_str` (defaults to
    today, ET) — status, score, and a clock/period text — matching what
    templates/_cfb_game_card.html renders server-side. Reuses
    get_live_scores()'s 2-min cache (same ESPN call) so polling the /cfb
    schedule page adds no extra API traffic.
    """
    from odds_api import _normalize
    target_str = date_str or _today_et()
    params = {'groups': _FBS_GROUP, 'limit': 400}
    if date_str:
        params['dates'] = target_str.replace('-', '')
    data = _cached_get(ESPN_CFB, params, f'cfb_scores_{target_str}', 120)
    states = {}
    if not data:
        return states
    for event in data.get('events', []):
        if _event_date_et(event.get('date', '')) != target_str:
            continue
        comp       = event.get('competitions', [{}])[0]
        status_obj = comp.get('status', {}).get('type', {})
        state      = status_obj.get('state', 'pre')
        if state == 'in':
            status = 'Live'
        elif state == 'post':
            status = 'Final'
        else:
            status = 'Preview'
        teams = {}
        for competitor in comp.get('competitors', []):
            side = competitor.get('homeAway', 'home')
            team = competitor.get('team', {})
            teams[side] = {'id': team.get('id'), 'name': team.get('displayName', ''), 'score': competitor.get('score')}
        home = teams.get('home', {})
        away = teams.get('away', {})
        if not home or not away:
            continue
        gk = f"{_normalize(home['name'])}_{_normalize(away['name'])}"

        live_state = None
        if state == 'in':
            live_status = comp.get('status', {}) or {}
            situation   = comp.get('situation', {}) or {}
            poss_id     = situation.get('possession')
            live_state = {
                'period':          live_status.get('period'),
                'display_clock':   live_status.get('displayClock'),
                'clock_text':      status_obj.get('shortDetail') or status_obj.get('detail'),
                'down_distance':   situation.get('shortDownDistanceText') or situation.get('downDistanceText'),
                'field_pos':       situation.get('possessionText'),
                'is_redzone':      bool(situation.get('isRedZone')),
                'home_timeouts':   situation.get('homeTimeouts'),
                'away_timeouts':   situation.get('awayTimeouts'),
                'possession_home': bool(poss_id) and poss_id == home.get('id'),
                'possession_away': bool(poss_id) and poss_id == away.get('id'),
            }

        states[gk] = {
            'status':     status,
            'away_score': away.get('score'),
            'home_score': home.get('score'),
            'live_state': live_state,
        }
    return states


# ── Schedule context ───────────────────────────────────────────────────────────

def _build_game(event, team_stats, game_log, cfb_odds_map, prior_stats=None):
    """Builds one game's full display dict — teams, model, odds, weather.
    Shared by build_schedule_context() (the daily-digest cache warmer) and
    build_week_schedule_context() (the /cfb page) so both render off the
    same data shape, same split as nfl_api.py's _build_game."""
    from odds_api import _normalize

    comp = event.get('competitions', [{}])[0]
    venue_obj = comp.get('venue', {})
    venue = venue_obj.get('fullName', '')
    if venue_obj.get('city'):
        venue += f", {venue_obj['city']}"

    status_obj = comp.get('status', {}).get('type', {})
    state = status_obj.get('state', 'pre')
    if state == 'in':
        status = 'Live'
    elif state == 'post':
        status = 'Final'
    else:
        status = 'Preview'

    teams = {}
    for competitor in comp.get('competitors', []):
        side = competitor.get('homeAway', 'home')
        team = competitor.get('team', {})
        records = competitor.get('records', [])
        overall = _parse_record(records, 'overall')
        home_r  = _parse_record(records, 'Home')
        away_r  = _parse_record(records, 'Road')
        w, l = _record_to_wl(overall)

        if side == 'home':
            split_summary = home_r
            split_label   = 'Home'
        else:
            split_summary = away_r
            split_label   = 'Away'

        split_w, split_l = _record_to_wl(split_summary)

        team_id = team.get('id')
        rank    = _parse_rank(competitor)
        teams[side] = {
            'id':          team_id,
            'name':        team.get('displayName', ''),
            'abbrev':      team.get('abbreviation', ''),
            'abbr':        team.get('abbreviation', ''),   # alias used in model template
            # NCAA teams use ESPN's numeric team id in the logo CDN path,
            # not abbreviation — abbreviations aren't unique/clean across
            # 130+ FBS schools the way they are for the NFL's 32 teams.
            'logo_url':    f"https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png" if team_id else '',
            'wins':        w,
            'losses':      l,
            'ot_losses':   None,
            'split_w':     split_w,
            'split_l':     split_l,
            'split_label': split_label,
            'side':        side,
            'form':        [],
            'form_score':  None,
            'score':       competitor.get('score'),
            'ppg':         None,
            'ppg_allowed': None,
            'rest_days':   None,
            'rank':        rank,
            'rank_display': rank if rank <= 25 else None,
            'conference':  _conference_name(team.get('conferenceId')),
        }

    away = teams.get('away', {})
    home = teams.get('home', {})

    summary = _get_game_summary(event.get('id'))
    weather = _parse_weather(summary)

    # Live game state — quarter, clock, down/distance, possession, field
    # position, red zone, timeouts. Same ESPN `situation` schema as
    # nfl_api.py's _build_game(); only present while state == 'in'.
    live_state = None
    if state == 'in':
        live_status = comp.get('status', {}) or {}
        situation   = comp.get('situation', {}) or {}
        poss_id     = situation.get('possession')
        live_state = {
            'period':          live_status.get('period'),
            'display_clock':   live_status.get('displayClock'),
            'clock_text':      status_obj.get('shortDetail') or status_obj.get('detail'),
            'down_distance':   situation.get('shortDownDistanceText') or situation.get('downDistanceText'),
            'field_pos':       situation.get('possessionText'),
            'is_redzone':      bool(situation.get('isRedZone')),
            'home_timeouts':   situation.get('homeTimeouts'),
            'away_timeouts':   situation.get('awayTimeouts'),
            'possession_home': bool(poss_id) and poss_id == home.get('id'),
            'possession_away': bool(poss_id) and poss_id == away.get('id'),
        }

    # Enrich with season stats, recent form, and rest days entering *this*
    # game's date — not "today", since a week view renders games that
    # haven't happened yet (or already have, for past weeks).
    game_date_et = _event_date_et(event.get('date', ''))
    for team in (home, away):
        name = team.get('name', '')
        ts   = team_stats.get(name, {})
        team['ppg']         = ts.get('ppg')
        team['ppg_allowed'] = ts.get('ppg_allowed')
        team['form']        = _team_recent_form(game_log, name)
        team['form_score']  = _team_recent_form_score(game_log, name)
        team['rest_days']   = _team_rest_days(game_log, name, game_date_et)
        if prior_stats:
            _apply_prior_season_blend(team, prior_stats)

    game_odds = odds_api.lookup_game_odds(cfb_odds_map, home.get('name', ''), away.get('name', ''),
                                           game_date=event.get('date', '')[:13] or None)

    # Run the model
    try:
        market_home_prob = game_odds.get('home_implied') if game_odds else None
        model = cfb_model.predict(home, away, game_time_utc=event.get('date', ''),
                                   market_home_prob=market_home_prob)
        model['factors'].sort(key=lambda f: abs(f[1]), reverse=True)
    except Exception:
        model = None

    # Pace-adjusted total (O/U) projection — see football_total_model.py.
    # total_inputs is stashed alongside the projection so a future
    # total-model recalibration can replay this exact game.
    total_inputs = {
        'home_id': home.get('id'), 'home_ppg': home.get('ppg'), 'home_ppg_allowed': home.get('ppg_allowed'),
        'away_id': away.get('id'), 'away_ppg': away.get('ppg'), 'away_ppg_allowed': away.get('ppg_allowed'),
    }
    try:
        total_model = football_total_model.predict_total(
            'CFB', home.get('id'), home.get('ppg'), home.get('ppg_allowed'),
            away.get('id'), away.get('ppg'), away.get('ppg_allowed'))
    except Exception:
        total_model = None

    espn_odds = (comp.get('odds') or [{}])[0]
    odds_line  = espn_odds.get('details', '')
    over_under = espn_odds.get('overUnder')

    a_ab = away.get('abbrev', '')
    h_ab = home.get('abbrev', '')

    team_stats = _parse_team_stats(summary, home.get('id'), away.get('id')) if status in ('Live', 'Final') else None

    return {
        'game_id':       event.get('id'),
        'game_time_utc': event.get('date', ''),
        'status':        status,
        'venue':         venue,
        'away':          away,
        'home':          home,
        'odds_line':     odds_line,
        'over_under':    over_under,
        'sport':         'CFB',
        'odds':          game_odds,
        'model':         model,
        'total_model':   total_model,
        'total_inputs':  total_inputs,
        'weather':       weather,
        'team_stats':    team_stats,
        'bet_name':      f"{a_ab} @ {h_ab}",
        'game_key':      f"{_normalize(home['name'])}_{_normalize(away['name'])}",
        'is_top25':      bool(home.get('rank_display') or away.get('rank_display')),
        'live_state':    live_state,
        'linescore':     _parse_linescores(comp) if status != 'Preview' else None,
        # ESPN season.type: 1=preseason (spring games), 2=regular, 3=postseason.
        # _upsert_predictions uses this to skip writing preseason/bowl games into
        # game_predictions so they don't corrupt the Model Performance page's
        # regular-season accuracy/calibration tracking — bowls have opt-outs and
        # roster-churn dynamics the model has no signal for, same reasoning
        # nfl_api.py applies to preseason.
        'is_preseason':  event.get('season', {}).get('type') != 2,
    }


def get_today_game_count():
    """Cheap today's-game count for the sport-chip badge — shares the same
    cached scoreboard fetch/cache key as build_schedule_context() without
    paying for any of its team-stats/odds/model enrichment."""
    today_str = _today_et()
    data = _cached_get(ESPN_CFB, {'groups': _FBS_GROUP, 'limit': 400}, f'cfb_{today_str}', _TTL['scoreboard'])
    if not data:
        return 0
    events = data.get('events', [])
    return len([e for e in events if _event_date_et(e.get('date', '')) == today_str])


def build_schedule_context():
    """Returns today's FBS games from ESPN with model predictions. Empty list
    during offseason. Used by the daily-digest cache warmer, not the /cfb
    page itself (see build_week_schedule_context)."""
    today_str = _today_et()
    data = _cached_get(ESPN_CFB, {'groups': _FBS_GROUP, 'limit': 400}, f'cfb_{today_str}', _TTL['scoreboard'])
    if not data:
        return []

    events = data.get('events', [])
    today_events = [
        e for e in events
        if _event_date_et(e.get('date', '')) == today_str
    ]
    if not today_events:
        return []

    season       = _get_cfb_season()
    game_log     = _get_season_game_log(season)
    team_stats   = _compute_team_season_stats(game_log)
    prior_stats  = _get_prior_season_team_stats(season - 1)
    cfb_odds_map = odds_api.get_odds_map('cfb')

    _prefetch_game_summaries(today_events)
    games = [_build_game(event, team_stats, game_log, cfb_odds_map, prior_stats) for event in today_events]

    try:
        date_display = datetime.now(_ET).strftime('%a, %b %-d')
    except Exception:
        date_display = today_str

    return [{'date': today_str, 'date_display': date_display, 'games': games}]


# ── Week schedule context ─────────────────────────────────────────────────────

_REG_SEASON_WEEKS = 15   # matches _get_season_game_log's range(1, 16)


def get_current_week():
    """(season, week_number) for the week containing "now" — falls back to
    week 1 if ESPN's scoreboard doesn't return a week (e.g. deep offseason)."""
    season = _get_cfb_season()
    data = _cached_get(ESPN_CFB, {'groups': _FBS_GROUP, 'limit': 400}, f'cfb_{_today_et()}', _TTL['scoreboard'])
    week = (data or {}).get('week', {}).get('number') or 1
    return season, week


def build_week_schedule_context(week=None):
    """Returns a full FBS week's games grouped by day — same shape as
    build_schedule_context() (a list of {date, date_display, games}) under
    'days', so game-card rendering is identical to the old daily view.
    `week=None` uses the current week. Clamped to the 15-week regular season."""
    season, current_week = get_current_week()
    if week is None:
        week = current_week
    week = max(1, min(_REG_SEASON_WEEKS, week))

    data = _cached_get(
        ESPN_CFB,
        # Regular season only (seasontype=2) — matches _get_season_game_log's
        # convention; a "week" concept doesn't apply the same way to bowls.
        {'seasontype': 2, 'week': week, 'season': season, 'dates': season,
         'groups': _FBS_GROUP, 'limit': 400},
        f'cfb_week_sched_{season}_{week}',
        _TTL['scoreboard'],
    )
    events = (data or {}).get('events', [])

    game_log     = _get_season_game_log(season)
    team_stats   = _compute_team_season_stats(game_log)
    prior_stats  = _get_prior_season_team_stats(season - 1)
    cfb_odds_map = odds_api.get_odds_map('cfb')

    _prefetch_game_summaries(events)
    by_date = defaultdict(list)
    for event in events:
        by_date[_event_date_et(event.get('date', ''))].append(
            _build_game(event, team_stats, game_log, cfb_odds_map, prior_stats)
        )

    today_str = _today_et()
    days = []
    for d in sorted(by_date.keys()):
        try:
            date_display = datetime.strptime(d, '%Y-%m-%d').strftime('%a, %b %-d')
        except Exception:
            date_display = d
        days.append({'date': d, 'date_display': date_display, 'games': by_date[d], 'is_today': d == today_str})

    return {
        'week':            week,
        'season':          season,
        'is_current_week': week == current_week,
        'prev_week':       week - 1 if week > 1 else None,
        'next_week':       week + 1 if week < _REG_SEASON_WEEKS else None,
        'days':            days,
    }
