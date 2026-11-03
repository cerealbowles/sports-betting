"""
cbb_roster_api.py — Team rosters for men's college basketball from ESPN's
`teams/{id}/roster`. Minimal for now: basketball_game_detail.py only needs
get_roster() to put names on injury rows. (No player model for CBB yet —
NBA/WNBA's get_active_roster/gamelog machinery is phase 2.)
"""
import time

import requests

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball'
_TTL = 6 * 3600
_cache = {}


def get_roster(team_id):
    """[{id, name, position, jersey}] or [] on failure."""
    if not team_id:
        return []
    now = time.time()
    hit = _cache.get(team_id)
    if hit and now - hit[1] < _TTL:
        return hit[0]
    try:
        r = requests.get(f'{ESPN_SITE}/teams/{team_id}/roster', timeout=15)
        r.raise_for_status()
        athletes = r.json().get('athletes') or []
    except Exception:
        return hit[0] if hit else []
    roster = [{'id': a.get('id'), 'name': a.get('displayName'),
               'position': (a.get('position') or {}).get('abbreviation', ''),
               'jersey': a.get('jersey')} for a in athletes if a.get('id')]
    _cache[team_id] = (roster, now)
    return roster
