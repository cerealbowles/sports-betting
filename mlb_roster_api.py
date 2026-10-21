"""
mlb_roster_api.py — Team roster data layer for MLB, sourced from ESPN's
free `teams/{id}/roster` endpoint (same provider every other module in
this app uses) — team_catalog.py's own MLB name lookup goes through
statsapi.mlb.com instead, but that doesn't expose ESPN athlete ids, which
is what mlb_news.py's player-name matching (see news_ingest.build_indexes())
actually needs. No position/availability filtering like nfl_roster_api.py's,
since nothing here needs it yet.

Same grouped-by-position roster shape as nfl_roster_api.py's endpoint
(athletes: [{position, items: [...]}]), confirmed live, not the flat
athletes: [...] shape nba_roster_api.py/wnba_roster_api.py use.
"""
import time
import requests

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb'

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
