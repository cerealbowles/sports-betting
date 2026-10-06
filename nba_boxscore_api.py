"""
nba_boxscore_api.py — Per-game injury status and live/final boxscore data,
sourced from ESPN's free `summary?event={id}` endpoint.

This is a different endpoint than nba_roster_api.py's roster/gamelog calls:
it's keyed by game (event id), not by team or player, and it's the only
place this app has found real NBA injury statuses — contrary to
nba_roster_api.py's docstring, the scoreboard endpoint's per-competitor
`injuries` key is indeed always empty for NBA, but this summary endpoint's
top-level `injuries` key is populated (confirmed live: a Day-To-Day entry
for Peyton Watson on a game 2 days out). No confirmed-starters feed exists
pre-game though — `boxscore.players` is empty until the game actually tips
off, at which point each athlete entry carries `starter: true/false` and a
live, continuously-updating `stats` array.

Used by nba_api.py to:
  - filter the active-roster guess in nba_roster_api.get_active_roster()
    down using real OUT/day-to-day statuses instead of (or alongside) the
    recent-participation heuristic, pre-game.
  - once a game is Live or Final, show each projected player's actual
    line next to their pre-game projection.
"""
import time
import requests

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'

_cache = {}
# injuries can change right up to tip-off; boxscore needs to track a live
# game closely. Both short — matches nba_api.py's own 120s scoreboard TTL.
_TTL = 120
_fail_ts = {}
_FAIL_BACKOFF = 300  # same pattern as nba_roster_api._get


def _get(event_id):
    key = event_id
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
            return data
    if now - _fail_ts.get(key, 0) < _FAIL_BACKOFF:
        return _cache.get(key, (None, 0))[0]
    try:
        r = requests.get(f'{ESPN_SITE}/summary', params={'event': event_id}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        _fail_ts[key] = now
        return _cache.get(key, (None, 0))[0]
    _cache[key] = (data, now)
    return data


# Statuses that mean "not available tonight" — excluded from the active
# roster. Anything else (Day-To-Day, Probable, Questionable, Game-Time
# Decision) is left in, same philosophy as the recent-participation
# heuristic: when in doubt, project them rather than silently zero them out.
_OUT_STATUSES = {'out', 'injured reserve', 'suspension', 'suspended'}


def get_injury_statuses(event_id):
    """Returns {player_id: status_text} for both teams in this game, or {}
    on failure / no injury data. status_text is ESPN's own label, e.g.
    'Day-To-Day', 'Out'."""
    data = _get(event_id)
    if not data:
        return {}
    out = {}
    for team in data.get('injuries', []):
        for inj in team.get('injuries', []):
            ath = inj.get('athlete', {})
            pid = ath.get('id')
            status = inj.get('status')
            if pid and status:
                out[pid] = status
    return out


def get_unavailable_player_ids(event_id):
    """Subset of get_injury_statuses() whose status means the player is
    confirmed not playing tonight (see _OUT_STATUSES)."""
    statuses = get_injury_statuses(event_id)
    return {pid for pid, status in statuses.items() if status.lower() in _OUT_STATUSES}


def get_live_boxscore(event_id):
    """Returns {team_id: [{id, name, starter, minutes, points, rebounds,
    assists, did_not_play}]} once the game has started (Live or Final);
    {} pre-game or on failure. Player stats reflect the game's current
    state — call again (subject to the 120s cache TTL) to get updates
    during a live game."""
    data = _get(event_id)
    if not data:
        return {}
    players = (data.get('boxscore') or {}).get('players') or []
    if not players:
        return {}

    out = {}
    for team_block in players:
        team_id = (team_block.get('team') or {}).get('id')
        stat_groups = team_block.get('statistics') or []
        if not team_id or not stat_groups:
            continue
        labels = stat_groups[0].get('labels', [])
        idx = {label: i for i, label in enumerate(labels)}

        def _stat(stats, label, default=0.0):
            i = idx.get(label)
            if i is None or i >= len(stats):
                return default
            try:
                return float(stats[i])
            except (TypeError, ValueError):
                return default

        rows = []
        for ath_entry in stat_groups[0].get('athletes', []):
            athlete = ath_entry.get('athlete', {})
            pid = athlete.get('id')
            name = athlete.get('displayName')
            if not pid or not name:
                continue
            stats = ath_entry.get('stats', [])
            did_not_play = bool(ath_entry.get('didNotPlay'))
            rows.append({
                'id':            pid,
                'name':          name,
                'starter':       bool(ath_entry.get('starter')),
                'did_not_play':  did_not_play,
                'minutes':       0.0 if did_not_play else _stat(stats, 'MIN'),
                'points':        0.0 if did_not_play else _stat(stats, 'PTS'),
                'rebounds':      0.0 if did_not_play else _stat(stats, 'REB'),
                'assists':       0.0 if did_not_play else _stat(stats, 'AST'),
            })
        out[team_id] = rows
    return out
