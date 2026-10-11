"""
nfl_roster_api.py — Team roster + position data for NFL, sourced from
ESPN's free `teams/{id}/roster` endpoint (same provider every other module
in this app uses).

Unlike nfl_boxscore_api.py's per-game summary endpoint (confirmed live to
carry NO position field on its athlete objects — see that module's
docstring), this endpoint's athlete objects DO carry a real position
(confirmed live: {'name': 'Running Back', 'abbreviation': 'RB', ...}), plus
which roster GROUP the player is currently in — 'offense', 'defense',
'specialTeam', 'injuredReserveOrOut', 'suspended', 'practiceSquad'. That
group is a free, no-extra-cost "is this guy actually available to play"
signal on top of position, which is exactly what nfl_fantasy.py's player
pool needs (no sense pricing/projecting someone on IR or the practice
squad).

This is a CURRENT-roster snapshot only (no historical position-by-date) —
fine for nfl_fantasy.py's forward-looking lineup builder (always asking
"who's on this team right now"), but NOT a source nfl_fantasy_bootstrap.py
can lean on for historical backtesting (a 2024 game's participants may have
since been traded/cut/retired and no longer resolve on any current roster).
That script classifies position by box-score role instead — see
nfl_fantasy.classify_bucket().
"""
import time
import requests

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl'

_cache = {}
_TTL = 24 * 3600
_fail_ts = {}
_FAIL_BACKOFF = 300

# Skill positions FanDuel's classic NFL contest actually drafts as
# individuals (QB/RB/WR/TE/FLEX) — everyone else (OL/DL/LB/DB/K/P/LS) is
# either undraftable or, for defense, scored as the team DST unit instead
# of an individual (see nfl_fantasy.py). FB folds into RB — both occupy the
# RB/FLEX roster slots and FanDuel doesn't distinguish them.
SKILL_POSITIONS = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}

# Roster groups that mean "not actually available this week" — same OUT-
# equivalent convention nfl_stats_db._OUT_STATUSES uses for injury reports,
# just at the roster-group granularity ESPN exposes here.
_UNAVAILABLE_GROUPS = {'injuredReserveOrOut', 'suspended', 'practiceSquad'}


def _get(url, params, key):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
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
    """Returns [{id, name, position, group}] for team_id's full roster
    (every group, including IR/suspended/practice-squad — see
    get_skill_players() below for the filtered version nfl_fantasy.py
    actually wants). [] on failure."""
    data = _get(f'{ESPN_SITE}/teams/{team_id}/roster', None, f'roster_{team_id}')
    if not data:
        return []
    out = []
    for group in data.get('athletes', []):
        group_name = group.get('position')
        for item in group.get('items', []):
            pid = item.get('id')
            name = item.get('displayName') or item.get('fullName')
            if not pid or not name:
                continue
            pos = (item.get('position') or {}).get('abbreviation')
            out.append({'id': pid, 'name': name, 'position': pos, 'group': group_name})
    return out


def get_skill_players(team_id):
    """Returns [{id, name, position}] for team_id's currently-available
    QB/RB/WR/TE's only (FB relabeled 'RB') — IR/suspended/practice-squad
    players excluded. This is the pool nfl_fantasy.build_player_pool() reads
    per team; still needs narrowing further against nfl_stats_db's own
    injury-status snapshot (Questionable/Doubtful aren't excluded here,
    same as every other roster-availability heuristic in this app — see
    nba_roster_api.get_active_roster's docstring for why "project them
    rather than silently zero them out" is the house convention)."""
    out = []
    for p in get_roster(team_id):
        if p['group'] in _UNAVAILABLE_GROUPS:
            continue
        mapped = SKILL_POSITIONS.get(p['position'])
        if not mapped:
            continue
        out.append({'id': p['id'], 'name': p['name'], 'position': mapped})
    return out
