"""
cfb_roster_api.py — Team roster data layer for CFB, sourced from ESPN's
free `teams/{id}/roster` endpoint (same provider every other module in
this app uses). Built purely for cfb_news.py's player-name matching (see
news_ingest.build_indexes()) — no position/availability filtering like
nfl_roster_api.py's, since nothing here needs it yet.

Same grouped-by-position roster shape as nfl_roster_api.py's endpoint
(athletes: [{position, items: [...]}]), confirmed live, not the flat
athletes: [...] shape nba_roster_api.py/wnba_roster_api.py use.

~130 FBS teams (vs NFL's 32) means far more name collisions across
rosters — news_ingest.match_players()'s last-name-only fallback will be
ambiguous much more often here, by design (it skips rather than guesses).
"""
import time
import requests

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football'

_cache = {}
_TTL = 24 * 3600
_fail_ts = {}
_FAIL_BACKOFF = 300


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
    """Returns [{id, name}] for team_id's current roster, or [] on failure."""
    data = _get(f'{ESPN_SITE}/teams/{team_id}/roster', None, f'roster_{team_id}')
    if not data:
        return []
    out = []
    for group in data.get('athletes', []):
        for item in group.get('items', []):
            pid = item.get('id')
            name = item.get('displayName') or item.get('fullName')
            if pid and name:
                out.append({'id': pid, 'name': name})
    return out


def get_fbs_team_ids():
    """{team_id, ...} for the ~124 FBS programs, via ESPN's standings
    endpoint (group=80 -> "FBS", children = conferences, each with a
    `standings.entries` list of teams) — NOT the /teams endpoint's own
    'groups'/'group' params, confirmed live to be silently ignored there
    (always returns every D1/D2/D3 team, ~800+, regardless of the filter).
    Used to restrict cfb_news.py's roster-building to FBS only — see
    news_ingest.build_indexes()'s `team_ids` docstring for why that matters
    at this team count. [] on failure (caller then builds with no
    restriction, i.e. every division — slow but not wrong)."""
    data = _get('https://site.api.espn.com/apis/v2/sports/football/college-football/standings',
                 {'group': 80}, 'fbs_ids')
    if not data:
        return set()
    ids = set()
    for conf in data.get('children', []):
        for entry in (conf.get('standings') or {}).get('entries', []):
            tid = (entry.get('team') or {}).get('id')
            if tid:
                ids.add(tid)
    return ids
