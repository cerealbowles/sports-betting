"""
nhl_boxscore_api.py — Per-player box score parsing for NHL, sourced from
ESPN's free `summary?event={id}` endpoint (same endpoint nhl_api.py
fetches per game elsewhere in this app).

ESPN's hockey box score splits into 'forwards'/'defenses'/'goalies' stat
groups (a 'skaters' group also exists but comes back empty in practice —
forwards/defenses already cover every skater) — unlike football's ~10
scattered categories, this is a clean 1-athlete-1-stat-line shape per
group, closer to basketball's. Confirmed live: ESPN's hockey athlete
object DOES carry a real position (e.g. "Center", "Goalie"), unlike NBA's/
NFL's, which is why this module can reliably separate goalies from
skaters by ESPN's own stat-group name rather than needing a usage-based
guess the way nfl_player_model.identify_key_players() does for football.

Goalie play is this sport's single highest-leverage individual signal
(the hockey equivalent of a starting QB — who's in net matters enormously
more than any one skater) — see nhl_player_model.py for how that gets a
dedicated goalie-specific grade rather than being folded into one blind
team-skater aggregate the way nfl_player_model.py's non-QB positions are.

parse_player_boxscore() is a pure function (dict in, dict out) so
nhl_api._build_game() can reuse a `summary` it already has in hand —
ingestion costs zero extra API calls when wired that way.
get_live_boxscore() below is a thin fetch+parse wrapper for callers that
don't already have a summary (nhl_stats_backfill.py).
"""
import time
import requests

ESPN_NHL_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary"

_cache = {}
_TTL = 600
_fail_ts = {}
_FAIL_BACKOFF = 300


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
        r = requests.get(ESPN_NHL_SUMMARY, params={'event': event_id}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        _fail_ts[key] = now
        return _cache.get(key, (None, 0))[0]
    _cache[key] = (data, now)
    return data


# Statuses that mean "not available tonight" — same convention as every
# other sport's boxscore/stats_db module in this app.
_OUT_STATUSES = {'out', 'injured reserve', 'suspension', 'suspended'}


def get_injuries(event_id):
    """Returns {team_id: [{id, name, pos, status}, ...]} for both teams in
    this game, or {} on failure/no data. ESPN's hockey scoreboard
    competitor `injuries` key is always empty here (same gap as NBA/NFL) —
    this per-game summary endpoint's top-level `injuries` key is where the
    real data lives, confirmed live (Out/Injured Reserve statuses with
    real athlete ids and position labels)."""
    data = _get(event_id)
    if not data:
        return {}
    out = {}
    for team in data.get('injuries', []):
        team_id = (team.get('team') or {}).get('id')
        if not team_id:
            continue
        rows = []
        for inj in team.get('injuries', []):
            ath = inj.get('athlete', {})
            pid = ath.get('id')
            status = inj.get('status')
            if pid and status:
                rows.append({
                    'id': pid,
                    'name': ath.get('displayName', ''),
                    'pos': (ath.get('position') or {}).get('abbreviation', ''),
                    'status': status,
                })
        out[team_id] = rows
    return out


def get_injury_statuses(event_id):
    """Returns {player_id: status_text} for both teams in this game — thin
    flattened view over get_injuries(), for callers that just want a quick
    status lookup rather than the full per-team record list."""
    out = {}
    for rows in get_injuries(event_id).values():
        for r in rows:
            out[r['id']] = r['status']
    return out


def get_unavailable_player_ids(event_id):
    statuses = get_injury_statuses(event_id)
    return {pid for pid, status in statuses.items() if status.lower() in _OUT_STATUSES}


def parse_player_boxscore(summary):
    """summary: a full `summary?event=` response.

    Returns {team_id: [{id, name, position, is_goalie, stats: {key: value,
    ...}}]} — {} if summary has no boxscore yet (pre-game)."""
    if not summary:
        return {}
    team_blocks = (summary.get('boxscore') or {}).get('players') or []
    if not team_blocks:
        return {}

    out = {}
    for team_block in team_blocks:
        team_id = (team_block.get('team') or {}).get('id')
        if not team_id:
            continue
        rows = []
        for group in team_block.get('statistics') or []:
            group_name = group.get('name')
            if group_name not in ('forwards', 'defenses', 'goalies'):
                continue  # skip the always-empty 'skaters' duplicate group
            keys = group.get('keys', [])
            for ath_entry in group.get('athletes', []):
                athlete = ath_entry.get('athlete', {})
                pid = athlete.get('id')
                name = athlete.get('displayName')
                if not pid or not name:
                    continue
                stats = dict(zip(keys, ath_entry.get('stats', [])))
                rows.append({
                    'id': pid,
                    'name': name,
                    'position': (athlete.get('position') or {}).get('displayName'),
                    'is_goalie': group_name == 'goalies',
                    'stats': stats,
                })
        out[team_id] = rows
    return out


def get_live_boxscore(event_id):
    """Fetch+parse convenience for callers with no summary already in hand
    (e.g. nhl_stats_backfill.py). nhl_api.py itself should prefer
    parse_player_boxscore(summary) directly to avoid a redundant fetch."""
    return parse_player_boxscore(_get(event_id))
