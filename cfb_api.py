import time
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, date as _date
from zoneinfo import ZoneInfo
import odds_api
import cfb_model

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
                games.append({
                    'game_date':  event.get('date', '')[:10],
                    'home_name':  h_name,
                    'away_name':  a_name,
                    'home_score': h_score,
                    'away_score': a_score,
                    'home_won':   h_score > a_score,
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


def _compute_team_season_stats(games):
    """Compute {team_name: {ppg, ppg_allowed}} from season game scores."""
    pts = defaultdict(lambda: {'for': 0, 'against': 0, 'g': 0})
    for g in games:
        pts[g['home_name']]['for']     += g['home_score']
        pts[g['home_name']]['against'] += g['away_score']
        pts[g['home_name']]['g']       += 1
        pts[g['away_name']]['for']     += g['away_score']
        pts[g['away_name']]['against'] += g['home_score']
        pts[g['away_name']]['g']       += 1
    return {
        t: {
            'ppg':         round(v['for']     / v['g'], 1),
            'ppg_allowed': round(v['against'] / v['g'], 1),
        }
        for t, v in pts.items() if v['g'] > 0
    }


def _team_recent_form(games, team_name, n=3):
    """Last n W/L results for team_name from game_log, newest first."""
    tg = []
    for g in games:
        if g['home_name'] == team_name:
            tg.append((g['game_date'], 'W' if g['home_won'] else 'L'))
        elif g['away_name'] == team_name:
            tg.append((g['game_date'], 'W' if not g['home_won'] else 'L'))
    tg.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in tg[:n]]


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


# ── Live scores ────────────────────────────────────────────────────────────────

def get_live_scores():
    """
    Returns {game_key: score_dict} for today's CFB games with a 2-min cache.
    """
    from odds_api import _normalize
    today_str = _today_et()
    data = _cached_get(ESPN_CFB, {'groups': _FBS_GROUP, 'limit': 400}, f'cfb_scores_{today_str}', 120)
    scores = {}
    if not data:
        return scores
    for event in data.get('events', []):
        if _event_date_et(event.get('date', '')) != today_str:
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
                'score': competitor.get('score'),
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
            'away_score': away.get('score'),
            'home_score': home.get('score'),
            'period':     period,
        }
    return scores


# ── Schedule context ───────────────────────────────────────────────────────────

def _build_game(event, team_stats, game_log, cfb_odds_map):
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
        team['rest_days']   = _team_rest_days(game_log, name, game_date_et)

    # Run the model
    try:
        model = cfb_model.predict(home, away, game_time_utc=event.get('date', ''))
    except Exception:
        model = None

    game_odds = odds_api.lookup_game_odds(cfb_odds_map, home.get('name', ''), away.get('name', ''),
                                           game_date=event.get('date', '')[:13] or None)

    espn_odds = (comp.get('odds') or [{}])[0]
    odds_line  = espn_odds.get('details', '')
    over_under = espn_odds.get('overUnder')

    a_ab = away.get('abbrev', '')
    h_ab = home.get('abbrev', '')

    return {
        'game_id':       event.get('id'),
        'game_time_utc': event.get('date', ''),
        'status':        status,
        'venue':         venue,
        'away':          away,
        'home':          home,
        'odds_line':     odds_line,
        'over_under':    over_under,
        'odds':          game_odds,
        'model':         model,
        'weather':       weather,
        'bet_name':      f"{a_ab} @ {h_ab}",
        'game_key':      f"{_normalize(home['name'])}_{_normalize(away['name'])}",
        'is_top25':      bool(home.get('rank_display') or away.get('rank_display')),
        # ESPN season.type: 1=preseason (spring games), 2=regular, 3=postseason.
        # _upsert_predictions uses this to skip writing preseason/bowl games into
        # game_predictions so they don't corrupt the Model Performance page's
        # regular-season accuracy/calibration tracking — bowls have opt-outs and
        # roster-churn dynamics the model has no signal for, same reasoning
        # nfl_api.py applies to preseason.
        'is_preseason':  event.get('season', {}).get('type') != 2,
    }


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
    cfb_odds_map = odds_api.get_odds_map('cfb')

    _prefetch_game_summaries(today_events)
    games = [_build_game(event, team_stats, game_log, cfb_odds_map) for event in today_events]

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
    cfb_odds_map = odds_api.get_odds_map('cfb')

    _prefetch_game_summaries(events)
    by_date = defaultdict(list)
    for event in events:
        by_date[_event_date_et(event.get('date', ''))].append(
            _build_game(event, team_stats, game_log, cfb_odds_map)
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
