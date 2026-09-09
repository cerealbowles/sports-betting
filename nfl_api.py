import time
import requests
from collections import defaultdict
from datetime import datetime, timezone, date as _date
from zoneinfo import ZoneInfo
import odds_api
import weather_api
import nfl_model

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

ESPN_NFL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

_cache = {}
# scoreboard was 1800s (30 min) — far too stale for the new live game-state
# fields (quarter/clock/down/distance/possession) to mean anything on a
# manual "↻ Refresh" during a live game, since that link just re-requests
# this same cached page. 120s matches get_live_scores()'s existing TTL for
# the open-bets ticker, which already polls this same ESPN endpoint that
# often with no issues.
_TTL = {'scoreboard': 120, 'season_log': 3600}


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


def _get_nfl_season():
    """NFL season year currently in progress or most recently completed."""
    today = datetime.now(_ET).date()
    return today.year if today.month >= 9 else today.year - 1


# ── Season game log ────────────────────────────────────────────────────────────

def _get_season_game_log(season):
    """
    Fetch and cache all completed NFL regular-season games for `season`.
    Makes up to 18 weekly API calls; each week cached for 1 hour.
    Returns list of {game_date, home_name, away_name, home_score, away_score, home_won}.
    """
    key = f'nfl_season_log_{season}'
    now_ts = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now_ts - ts < _TTL['season_log']:
            return data

    games = []
    for week in range(1, 19):
        week_data = _cached_get(
            ESPN_NFL,
            # `dates` (not `season`) is what actually selects the year on
            # this endpoint — see nfl_bootstrap.py's fetch_season_games for
            # the full explanation. Harmless here today since `season` is
            # always the current season anyway, but left implicit it would
            # silently break the moment this ever fetches a past season.
            {'seasontype': 2, 'week': week, 'season': season, 'dates': season, 'limit': 20},
            f'nfl_week_{season}_{week}',
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


# ── Live scores ────────────────────────────────────────────────────────────────

def get_live_scores():
    """
    Returns {game_key: score_dict} for today's NFL games with a 2-min cache.
    """
    from odds_api import _normalize
    today_str = _today_et()
    data = _cached_get(ESPN_NFL, {}, f'nfl_scores_{today_str}', 120)
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

def build_schedule_context():
    """Returns today's NFL games from ESPN with model predictions. Empty list during offseason."""
    from odds_api import _normalize
    today_str = _today_et()
    data = _cached_get(ESPN_NFL, {}, f'nfl_{today_str}', _TTL['scoreboard'])
    if not data:
        return []

    events = data.get('events', [])
    today_events = [
        e for e in events
        if _event_date_et(e.get('date', '')) == today_str
    ]
    if not today_events:
        return []

    # Fetch season-level stats once for all games
    season     = _get_nfl_season()
    game_log   = _get_season_game_log(season)
    team_stats = _compute_team_season_stats(game_log)

    games = []
    for event in today_events:
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
                'id':          team.get('id'),   # ESPN team id — needed to match situation.possession below
                'name':        team.get('displayName', ''),
                'abbrev':      abbrev,
                'abbr':        abbrev,   # alias used in model template
                'logo_url':    f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbrev.lower()}.png",
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
            }

        away = teams.get('away', {})
        home = teams.get('home', {})

        # ── Live game state — quarter, clock, down/distance, possession,
        # field position, red zone, timeouts. ESPN's `situation` object is
        # only present while state == 'in'; it goes away between plays'
        # snapshots occasionally (e.g. right at a quarter change), so every
        # field here is optional and the template only shows what's
        # present. Field names below are ESPN's standard (undocumented but
        # widely-used) football scoreboard schema — not yet verified
        # against a real live NFL game since this was built in the
        # off-season; if anything renders oddly once week 1 kicks off,
        # check the actual payload shape first before assuming the model
        # or template is wrong.
        live_state = None
        if state == 'in':
            live_status = comp.get('status', {}) or {}
            situation   = comp.get('situation', {}) or {}
            poss_id     = situation.get('possession')
            live_state = {
                'period':        live_status.get('period'),
                'display_clock': live_status.get('displayClock'),
                'clock_text':    status_obj.get('shortDetail') or status_obj.get('detail'),
                'down_distance': situation.get('shortDownDistanceText') or situation.get('downDistanceText'),
                'field_pos':     situation.get('possessionText'),
                'is_redzone':    bool(situation.get('isRedZone')),
                'home_timeouts': situation.get('homeTimeouts'),
                'away_timeouts': situation.get('awayTimeouts'),
                'possession_home': bool(poss_id) and poss_id == home.get('id'),
                'possession_away': bool(poss_id) and poss_id == away.get('id'),
            }

        # Enrich with season stats, recent form, and rest days
        for team, is_home in ((home, True), (away, False)):
            name = team.get('name', '')
            ts   = team_stats.get(name, {})
            team['ppg']         = ts.get('ppg')
            team['ppg_allowed'] = ts.get('ppg_allowed')
            team['form']        = _team_recent_form(game_log, name)
            team['rest_days']   = _team_rest_days(game_log, name, today_str)

        # Run the model
        try:
            model = nfl_model.predict(home, away, game_time_utc=event.get('date', ''))
        except Exception:
            model = None

        # Odds
        nfl_odds_map = odds_api.get_odds_map('nfl')
        game_odds    = odds_api.lookup_game_odds(nfl_odds_map, home.get('name', ''), away.get('name', ''),
                                                  game_date=event.get('date', '')[:13] or None)

        espn_odds = (comp.get('odds') or [{}])[0]
        odds_line  = espn_odds.get('details', '')
        over_under = espn_odds.get('overUnder')

        a_ab = away.get('abbrev', '')
        h_ab = home.get('abbrev', '')

        games.append({
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
            'weather':       weather_api.get_game_weather(venue_obj.get('fullName', ''), 'nfl'),
            'bet_name':      f"{a_ab} @ {h_ab}",
            'game_key':      f"{_normalize(home['name'])}_{_normalize(away['name'])}",
            'live_state':    live_state,
            # ESPN season.type: 1=preseason, 2=regular, 3=postseason. Still
            # shown live (useful to see today's score even in August), but
            # _upsert_predictions uses this to skip writing preseason games
            # into game_predictions — backup-heavy, min-effort preseason
            # results would otherwise corrupt the Model Performance page's
            # regular-season accuracy/calibration tracking.
            'is_preseason':  event.get('season', {}).get('type') == 1,
        })

    try:
        date_display = datetime.now(_ET).strftime('%a, %b %-d')
    except Exception:
        date_display = today_str

    return [{'date': today_str, 'date_display': date_display, 'games': games}]
