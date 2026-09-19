import time
import requests
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import odds_api
import nba_model

_ET = ZoneInfo('America/New_York')

def _today_et():
    return datetime.now(_ET).strftime('%Y-%m-%d')

def _event_date_et(date_str):
    """Convert an ESPN event's UTC ISO timestamp to its ET calendar date.
    Late tip-offs (10pm+ ET) land past midnight UTC, so comparing the raw
    UTC string against an ET date string (e.g. via startswith) misses them."""
    if not date_str:
        return ''
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.astimezone(_ET).strftime('%Y-%m-%d')
    except ValueError:
        return ''

ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

_cache = {}
# scoreboard: 120s matches nfl_api.py/get_live_scores()'s TTL — live-game
# state (clock/period/score) needs to be fresh on manual refresh.
# season_log: 1 hour — team season stats don't need live-game freshness.
_TTL = {'scoreboard': 120, 'season_log': 3600}


def _parse_linescores(comp):
    """Quarter-by-quarter scoring from ESPN's per-competitor `linescores`
    array — only present once a game has started. Returns a list of
    {'label', 'away', 'home'} periods (Q1-Q4, then OT/OT2/... for however
    many overtimes were played), or None pre-tipoff."""
    competitors = comp.get('competitors', [])
    home_ls = next((c.get('linescores') for c in competitors if c.get('homeAway') == 'home'), None) or []
    away_ls = next((c.get('linescores') for c in competitors if c.get('homeAway') == 'away'), None) or []
    n = max(len(home_ls), len(away_ls))
    if n == 0:
        return None
    periods = []
    for i in range(n):
        label = str(i + 1) if i < 4 else ('OT' if n == 5 else f'OT{i - 3}')
        periods.append({
            'label': label,
            'away':  away_ls[i].get('value') if i < len(away_ls) else None,
            'home':  home_ls[i].get('value') if i < len(home_ls) else None,
        })
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


def _get_nba_season():
    """NBA season year (ESPN convention: `season=2024` means the 2024-25
    season) currently in progress or most recently completed. New season
    tips off in October."""
    today = datetime.now(_ET).date()
    return today.year if today.month >= 10 else today.year - 1


# ── Season game log ────────────────────────────────────────────────────────────

def _get_season_game_log(season):
    """
    Fetch and cache all completed NBA regular-season games for `season`
    (ESPN's `season` year, e.g. 2024 = 2024-25) up through today, by paging
    the scoreboard day-by-day from the season's Oct 1 start. Cached for 1
    hour — this call re-walks the whole season-to-date on every cache miss,
    which is cheap (ESPN scoreboard is a lightweight per-day payload) but
    not free, so build_schedule_context()/get_today_game_count() share this
    single cached game log rather than each re-fetching it.
    Returns list of {game_date, home_name, away_name, home_score, away_score, home_won}.
    """
    key = f'nba_season_log_{season}'
    now_ts = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now_ts - ts < _TTL['season_log']:
            return data

    games = []
    seen_ids = set()
    start = datetime(season, 10, 1, tzinfo=_ET).date()
    today = datetime.now(_ET).date()
    end = min(today, datetime(season + 1, 6, 30, tzinfo=_ET).date())
    day = start
    while day <= end:
        date_str = day.strftime('%Y%m%d')
        data = _cached_get(ESPN_NBA, {'dates': date_str, 'limit': 100},
                            f'nba_day_{date_str}', _TTL['season_log'])
        day += timedelta(days=1)
        if not data:
            continue
        for event in data.get('events', []):
            if event.get('id') in seen_ids:
                continue
            comp  = event.get('competitions', [{}])[0]
            state = comp.get('status', {}).get('type', {}).get('state', 'pre')
            if state != 'post':
                continue
            if event.get('season', {}).get('type') != 2:
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
                seen_ids.add(event.get('id'))
            except Exception:
                pass

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


def _team_recent_form(games, team_name, n=5):
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


# ── Live scores ────────────────────────────────────────────────────────────────

def get_live_scores(date_str=None):
    """
    Returns {game_key: score_dict} for NBA games on `date_str` (YYYY-MM-DD ET,
    defaults to today) with a 2-min cache. Passing a past date lets an open bet
    from a prior day keep showing its final score until the bet is closed.
    """
    from odds_api import _normalize
    target_str = date_str or _today_et()
    params = {'dates': target_str.replace('-', '')} if date_str else {}
    data = _cached_get(ESPN_NBA, params, f'nba_scores_{target_str}', 120)
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
            'away_logo':  f"https://a.espncdn.com/i/teamlogos/nba/500/{away.get('abbr', '').lower()}.png" if away.get('abbr') else '',
            'home_logo':  f"https://a.espncdn.com/i/teamlogos/nba/500/{home.get('abbr', '').lower()}.png" if home.get('abbr') else '',
            'away_score': away.get('score'),
            'home_score': home.get('score'),
            'period':     period,
            'away_logo':  f"https://a.espncdn.com/i/teamlogos/nba/500/{away.get('abbr', '').lower()}.png" if away.get('abbr') else None,
            'home_logo':  f"https://a.espncdn.com/i/teamlogos/nba/500/{home.get('abbr', '').lower()}.png" if home.get('abbr') else None,
        }
    return scores


def get_live_game_states(date_str=None):
    """
    Returns {game_key: state_dict} for NBA games on `date_str` (defaults to
    today, ET) — status, score, and a clock/period text — matching what the
    /nba schedule page's game cards render server-side. Reuses
    get_live_scores()'s 2-min cache (same ESPN call) so polling the page
    adds no extra API traffic.
    """
    from odds_api import _normalize
    target_str = date_str or _today_et()
    params = {'dates': target_str.replace('-', '')} if date_str else {}
    data = _cached_get(ESPN_NBA, params, f'nba_scores_{target_str}', 120)
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
            teams[side] = {'name': team.get('displayName', ''), 'score': competitor.get('score')}
        home = teams.get('home', {})
        away = teams.get('away', {})
        if not home or not away:
            continue
        gk = f"{_normalize(home['name'])}_{_normalize(away['name'])}"
        clock_text = status_obj.get('detail') if status == 'Live' else None
        states[gk] = {
            'status':     status,
            'away_score': away.get('score'),
            'home_score': home.get('score'),
            'live_state': {'clock_text': clock_text} if status == 'Live' else None,
        }
    return states


# ── Schedule context ───────────────────────────────────────────────────────────

def _build_game(event, team_stats, game_log, nba_odds_map):
    """Builds one game's full display dict — teams, live state, model, odds.
    Shared by build_schedule_context() (the /nba page and the daily-digest
    cache warmer)."""
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

        abbrev = team.get('abbreviation', '')
        teams[side] = {
            'id':          team.get('id'),
            'name':        team.get('displayName', ''),
            'abbrev':      abbrev,
            'abbr':        abbrev,
            'logo_url':    f"https://a.espncdn.com/i/teamlogos/nba/500/{abbrev.lower()}.png",
            'wins':        w,
            'losses':      l,
            'split_w':     split_w,
            'split_l':     split_l,
            'split_label': split_label,
            'side':        side,
            'form':        [],
            'score':       competitor.get('score'),
            'ppg':         None,
            'ppg_allowed': None,
            'rest_days':   None,
            'injuries':    [],
        }

    away = teams.get('away', {})
    home = teams.get('home', {})

    # Enrich with season stats, recent form, and rest days entering *this*
    # game's date.
    game_date_et = _event_date_et(event.get('date', ''))
    for team in (home, away):
        name = team.get('name', '')
        ts   = team_stats.get(name, {})
        team['ppg']         = ts.get('ppg')
        team['ppg_allowed'] = ts.get('ppg_allowed')
        team['form']        = _team_recent_form(game_log, name)
        team['rest_days']   = _team_rest_days(game_log, name, game_date_et)

    game_odds = odds_api.lookup_game_odds(nba_odds_map, home.get('name', ''), away.get('name', ''),
                                           game_date=event.get('date', '')[:13] or None)

    # Run the model
    try:
        market_home_prob = game_odds.get('home_implied') if game_odds else None
        model = nba_model.predict(home, away, game_time_utc=event.get('date', ''),
                                   market_home_prob=market_home_prob)
    except Exception:
        model = None

    espn_odds  = (comp.get('odds') or [{}])[0]
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
        'bet_name':      f"{a_ab} @ {h_ab}",
        'game_key':      f"{_normalize(home['name'])}_{_normalize(away['name'])}",
        'linescore':     _parse_linescores(comp) if status != 'Preview' else None,
        # ESPN season.type: 1=preseason, 2=regular, 3=postseason/play-in.
        # _upsert_predictions uses this to skip writing preseason games into
        # game_predictions — min-effort preseason results would otherwise
        # corrupt the Model Performance page's accuracy/calibration tracking.
        'is_preseason':  event.get('season', {}).get('type') == 1,
    }


def get_today_game_count():
    """Cheap today's-game count for the sport-chip badge — shares the same
    cached scoreboard fetch/cache key as build_schedule_context() without
    paying for any of its team-stats/odds/model enrichment."""
    today_str = _today_et()
    data = _cached_get(ESPN_NBA, {}, f'nba_{today_str}', _TTL['scoreboard'])
    if not data:
        return 0
    events = data.get('events', [])
    return len([e for e in events if _event_date_et(e.get('date', '')) == today_str])


def build_schedule_context():
    """Returns today's NBA games from ESPN with model predictions. Empty list during offseason."""
    today_str = _today_et()
    data = _cached_get(ESPN_NBA, {}, f'nba_{today_str}', _TTL['scoreboard'])
    if not data:
        return []

    events = data.get('events', [])
    today_events = [
        e for e in events
        if _event_date_et(e.get('date', '')) == today_str
    ]
    if not today_events:
        return []

    season       = _get_nba_season()
    game_log     = _get_season_game_log(season)
    team_stats   = _compute_team_season_stats(game_log)
    nba_odds_map = odds_api.get_odds_map('nba')

    games = [_build_game(event, team_stats, game_log, nba_odds_map) for event in today_events]

    try:
        date_display = datetime.now(_ET).strftime('%a, %b %-d')
    except Exception:
        date_display = today_str

    return [{'date': today_str, 'date_display': date_display, 'games': games}]
