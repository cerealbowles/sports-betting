"""
cbb_boxscore_api.py — Per-game data from ESPN's `summary?event={id}` for men's
college basketball: injury statuses (shape-compatible with
wnba_boxscore_api.get_injury_statuses, which basketball_game_detail.py calls)
and ESPN's own BPI matchup predictor.

Only the game-detail page calls this — never the schedule build, which runs
50-150 games a night (see cbb_api.py's docstring). As of 2026-11 ESPN's CBB
summaries carry no `injuries` block at all, so get_injury_statuses() returns
{} in practice; it parses the NBA/WNBA shape anyway so it starts working if
ESPN ever adds one.
"""
import time

import requests

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball'
_TTL = 600
_FAIL_BACKOFF = 300
_cache, _fail_ts = {}, {}


def _get(event_id):
    now = time.time()
    hit = _cache.get(event_id)
    if hit and now - hit[1] < _TTL:
        return hit[0]
    if now - _fail_ts.get(event_id, 0) < _FAIL_BACKOFF:
        return hit[0] if hit else None
    try:
        r = requests.get(f'{ESPN_SITE}/summary', params={'event': event_id}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        _fail_ts[event_id] = now
        return hit[0] if hit else None
    _cache[event_id] = (data, now)
    return data


def get_injury_statuses(event_id):
    """{player_id: status_text} for both teams, or {}."""
    data = _get(event_id) or {}
    out = {}
    for team in data.get('injuries') or []:
        for inj in team.get('injuries', []):
            pid, status = (inj.get('athlete') or {}).get('id'), inj.get('status')
            if pid and status:
                out[pid] = status
    return out


def get_bpi_predictor(event_id):
    """ESPN BPI's pregame win probability: {'home': pct, 'away': pct} (0-100)
    or None. Display/tracking only."""
    pred = (_get(event_id) or {}).get('predictor') or {}
    try:
        h = float((pred.get('homeTeam') or {}).get('gameProjection'))
        a = float((pred.get('awayTeam') or {}).get('gameProjection'))
    except (TypeError, ValueError):
        return None
    return {'home': h, 'away': a}
