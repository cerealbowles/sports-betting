"""
Injury status from ESPN's unofficial scoreboard API.
The scoreboard responses include an `injuries` array on each competitor
for NFL. MLB and NHL are attempted via the same pattern but may return
empty lists — all failures are silent.
"""
import time
import requests

ESPN = "https://site.api.espn.com/apis/site/v2/sports"

_PATHS = {
    'mlb': 'baseball/mlb',
    'nhl': 'hockey/nhl',
    'nfl': 'football/nfl',
}

_cache = {}
_TTL   = 3600  # 1 hour


def _get(url):
    if url in _cache:
        data, ts = _cache[url]
        if time.time() - ts < _TTL:
            return data
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return _cache.get(url, (None, 0))[0]
    _cache[url] = (data, time.time())
    return data


def get_injury_map(sport):
    """
    Returns {player_name_lower: status_string} for all injured players today.
    E.g. {'gerrit cole': 'Out', 'shohei ohtani': 'Day-To-Day'}
    Returns {} on any error or if the sport doesn't expose injury data.
    """
    path = _PATHS.get(sport)
    if not path:
        return {}

    data = _get(f"{ESPN}/{path}/scoreboard")
    if not data:
        return {}

    injury_map = {}
    for event in data.get('events', []):
        for comp in event.get('competitions', []):
            for competitor in comp.get('competitors', []):
                for inj in competitor.get('injuries', []):
                    athlete = inj.get('athlete') or inj.get('player') or {}
                    name    = athlete.get('displayName') or athlete.get('fullName', '')
                    status  = (
                        inj.get('status')
                        or inj.get('type', {}).get('description', '')
                    )
                    if name and status:
                        injury_map[name.lower()] = status

    return injury_map
