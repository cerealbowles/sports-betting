import time
import requests

_ESPN = 'https://site.api.espn.com/apis/site/v2/sports'
_SOURCES = {
    'NFL': (f'{_ESPN}/football/nfl/teams', {'limit': 100}),
    'CFB': (f'{_ESPN}/football/college-football/teams', {'groups': 80, 'limit': 300}),
    'NBA': (f'{_ESPN}/basketball/nba/teams', {'limit': 100}),
    'WNBA': (f'{_ESPN}/basketball/wnba/teams', {'limit': 100}),
    'NHL': (f'{_ESPN}/hockey/nhl/teams', {'limit': 100}),
}
_TTL = 24 * 3600
_cache = {}


def get_team_names(sport):
    """Sorted full team names for a sport, cached for a day; [] if the fetch fails."""
    hit = _cache.get(sport)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    names = []
    try:
        if sport == 'MLB':
            r = requests.get('https://statsapi.mlb.com/api/v1/teams',
                             params={'sportId': 1}, timeout=10)
            r.raise_for_status()
            names = [t.get('name', '') for t in r.json().get('teams', [])]
        elif sport in _SOURCES:
            url, params = _SOURCES[sport]
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            leagues = (r.json().get('sports') or [{}])[0].get('leagues') or [{}]
            names = [t.get('team', {}).get('displayName', '') for t in leagues[0].get('teams', [])]
    except Exception:
        return []
    names = sorted({n for n in names if n})
    if names:
        _cache[sport] = (time.time(), names)
    return names
