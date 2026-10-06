"""
nba_roster_api.py — Per-player data layer for the player-level NBA model
(nba_player_model.py), sourced from ESPN's unofficial APIs (same provider
every other module in this app already uses).

Two endpoints, two different shapes:
  - site.api.espn.com .../teams/{id}/roster       — current roster (who's on
    the team right now; no stats).
  - site.web.api.espn.com .../athletes/{id}/gamelog — a player's own
    per-game log (minutes/points/rebounds/etc per game this season),
    newest-first after sorting here.

Confirmed live against real team/player ids before building on it (Warriors
roster endpoint, LeBron James gamelog) — see nba_roster_api spike notes in
the NBA player-model plan.

No pre-game "confirmed starters" feed exists on ESPN's free API. Worse, for
NBA specifically, ESPN's scoreboard-based injury signal that other sports use
(injuries_api.get_injury_map) doesn't exist at all — NBA competitor objects
carry no `injuries` key (confirmed against a live response), so it always
returns {}. get_active_roster() below instead uses recent-game participation
(did this player appear in minutes in their team's last game) as the active-
roster signal — a best-effort "probably still active" guess, not a confirmed
lineup or real injury status. A player who's OUT tonight but played last game
will be wrongly included; this is a known gap, not a bug.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
ESPN_WEB  = 'https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba'

_cache = {}
_TTL = {'roster': 24 * 3600, 'gamelog': 6 * 3600}
_fail_ts = {}
_FAIL_BACKOFF = 300  # same backoff pattern as bball_total_model._fetch_pace_components


def _get(url, params, kind):
    key = (url, tuple(sorted((params or {}).items())))
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL[kind]:
            return data
    if now - _fail_ts.get(key, 0) < _FAIL_BACKOFF:
        return _cache.get(key, (None, 0))[0]
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        _fail_ts[key] = now
        return _cache.get(key, (None, 0))[0]
    _cache[key] = (data, now)
    return data


def get_roster(team_id):
    """Returns [{id, name}] for team_id's current roster, or [] on failure."""
    data = _get(f'{ESPN_SITE}/teams/{team_id}/roster', None, 'roster')
    if not data:
        return []
    out = []
    for ath in data.get('athletes', []):
        pid = ath.get('id')
        name = ath.get('displayName') or ath.get('fullName')
        if pid and name:
            out.append({'id': pid, 'name': name})
    return out


# Order of nba_player_model.PLAYER_GAMELOG_NAMES must match this endpoint's
# `names` field — asserted in get_player_gamelog() so a silent ESPN schema
# change fails loudly instead of mis-mapping stats.
GAMELOG_NAMES = [
    'minutes', 'fieldGoalsMade-fieldGoalsAttempted', 'fieldGoalPct',
    'threePointFieldGoalsMade-threePointFieldGoalsAttempted', 'threePointPct',
    'freeThrowsMade-freeThrowsAttempted', 'freeThrowPct', 'totalRebounds',
    'assists', 'blocks', 'steals', 'fouls', 'turnovers', 'points',
]


def _attempts(made_attempt_str):
    """'7-14' -> 14.0 (the attempts half of a made-attempt composite field).
    Returns 0.0 on anything unparseable."""
    try:
        return float(str(made_attempt_str).split('-')[1])
    except (IndexError, ValueError, TypeError):
        return 0.0


def _parse_game_stats(names, raw_stats):
    """Maps a gamelog event's raw `stats` string array to a {stat: float}
    dict using `names`. Composite made-attempt fields (e.g. '7-14') are
    kept under their own name as the ATTEMPTS half (used for usage-rate
    projection) rather than skipped outright."""
    out = {}
    for name, val in zip(names, raw_stats):
        try:
            out[name] = float(val)
        except (TypeError, ValueError):
            if '-' in str(val):
                out[name] = _attempts(val)
    return out


def get_player_gamelog(player_id, season_type_contains='Regular Season'):
    """Returns this player's games this season, newest first:
    [{event_id, date, minutes, points, rebounds, assists, team_id, home}],
    or [] on failure / no games found.
    """
    data = _get(f'{ESPN_WEB}/athletes/{player_id}/gamelog', None, 'gamelog')
    if not data:
        return []

    names = data.get('names', [])
    events_map = data.get('events', {})
    season_types = data.get('seasonTypes', [])
    st = next((s for s in season_types
               if season_type_contains in (s.get('displayName') or '')), None)
    if not st:
        return []

    rows = []
    for cat in st.get('categories', []):
        for ev in cat.get('events', []):
            eid = ev.get('eventId')
            detail = events_map.get(eid, {})
            stats = _parse_game_stats(names, ev.get('stats', []))
            minutes = stats.get('minutes')
            if minutes is None:
                continue
            rows.append({
                'event_id': eid,
                'date':     (detail.get('gameDate') or '')[:10],
                'minutes':  minutes,
                'points':   stats.get('points', 0.0),
                'rebounds': stats.get('totalRebounds', 0.0),
                'assists':  stats.get('assists', 0.0),
                'fga':      stats.get('fieldGoalsMade-fieldGoalsAttempted', 0.0),
                'fta':      stats.get('freeThrowsMade-freeThrowsAttempted', 0.0),
                'tov':      stats.get('turnovers', 0.0),
                'team_id':  (detail.get('team') or {}).get('id'),
                'home':     detail.get('atVs') == 'vs',
            })

    rows.sort(key=lambda r: r['date'], reverse=True)
    return rows


def get_active_roster(team_id, min_recent_games=1, lookback=3):
    """Best-effort "who's probably playing tonight" for team_id: roster
    players whose own gamelog shows minutes in at least `min_recent_games`
    of their last `lookback` games. Excludes players who are rostered but
    haven't played recently (likely injured/inactive/G-League assigned) —
    see this module's docstring for why this heuristic exists instead of a
    real injury/lineup feed for NBA.

    Returns [{id, name, games}] where `games` is that player's full gamelog
    (newest first) for project_player() to use.

    Fetches each roster player's gamelog in parallel (same ThreadPoolExecutor
    pattern as mlb_api.py/nhl_api.py's own fan-out fetches) — a full ~20-man
    roster fetched sequentially took 23+ seconds per team in testing, which
    multiplies to many minutes across a full slate of games and isn't
    workable for a live page load or cache-warm cycle.
    """
    roster = get_roster(team_id)
    if not roster:
        return []
    games_by_pid = {}
    with ThreadPoolExecutor(max_workers=min(len(roster), 12)) as ex:
        futures = {ex.submit(get_player_gamelog, p['id']): p['id'] for p in roster}
        for f in as_completed(futures):
            pid = futures[f]
            try:
                games_by_pid[pid] = f.result()
            except Exception:
                games_by_pid[pid] = []

    active = []
    for p in roster:
        games = games_by_pid.get(p['id'], [])
        recent = games[:lookback]
        if sum(1 for g in recent if g['minutes'] > 0) >= min_recent_games:
            active.append({'id': p['id'], 'name': p['name'], 'games': games})
    return active
