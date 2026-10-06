"""
nfl_boxscore_api.py — Per-player box score parsing for NFL, sourced from
ESPN's free `summary?event={id}` endpoint (same endpoint nfl_api.py
already fetches per game for weather/injuries via _get_game_summary()).

Unlike basketball's box score (nba_boxscore_api.py — one stat line per
athlete), football's box score is split into up to 10 independent
categories (passing, rushing, receiving, fumbles, defensive,
interceptions, kickReturns, puntReturns, kicking, punting), each with its
own `keys`/`labels` and its own `athletes` list — a single player (e.g. a
scrambling QB, or a WR who also returns kicks) can appear in several
categories within the same game. parse_player_boxscore() below merges all
of a player's category appearances into one row per player, keyed by
athlete id, with a nested `categories` dict rather than flattening to a
fixed column set the way NBA's stat line does — football's stat shape
varies too much by position (a kicker and a cornerback have nothing in
common) to make a single flat row sensible. This is also why
PlayerGameStat (app.py) stores football's stats as JSON (`stats_json`)
instead of basketball's one-column-per-stat approach.

parse_player_boxscore() is a pure function (dict in, dict out) so
nfl_api._build_game() can reuse the `summary` it already fetches for
weather/injuries — ingestion costs zero extra API calls, same resource-
reuse pattern as nba_boxscore_api.get_live_boxscore() feeding both the
lineup-vs-actual UI and the stats warehouse off one fetch.
get_live_boxscore() below is a thin fetch+parse wrapper for callers that
don't already have a summary in hand (nfl_stats_backfill.py).
"""
import time
import requests

ESPN_NFL_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

_cache = {}
_TTL = 600  # matches nfl_api.py's own 'summary' TTL
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
        r = requests.get(ESPN_NFL_SUMMARY, params={'event': event_id}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        _fail_ts[key] = now
        return _cache.get(key, (None, 0))[0]
    _cache[key] = (data, now)
    return data


def parse_player_boxscore(summary):
    """summary: a full `summary?event=` response (e.g. from nfl_api.py's
    own _get_game_summary(), or get_live_boxscore() below).

    Returns {team_id: [{id, name, position, categories: {cat_name:
    {stat_key: value, ...}, ...}}]} — {} if summary has no boxscore yet
    (pre-game; football's boxscore, like basketball's, is empty until the
    game actually starts).

    `position` is usually None — confirmed live that this endpoint's
    athlete object (unlike NBA's, which carries `position.abbreviation`)
    has no position field at all, just id/name/jersey/headshot. Left in
    rather than dropped since ESPN could start including it, and because
    which `categories` a player appears in is itself a rough position
    signal (passing-only is a QB, rushing+receiving a RB, etc.) a future
    reader could derive without this function guessing at it."""
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
        players_by_id = {}
        for cat in team_block.get('statistics') or []:
            cat_name = cat.get('name')
            keys = cat.get('keys', [])
            for ath_entry in cat.get('athletes', []):
                athlete = ath_entry.get('athlete', {})
                pid = athlete.get('id')
                name = athlete.get('displayName')
                if not pid or not name:
                    continue
                row = players_by_id.setdefault(pid, {
                    'id': pid, 'name': name,
                    'position': (athlete.get('position') or {}).get('abbreviation'),
                    'categories': {},
                })
                stat_values = ath_entry.get('stats', [])
                row['categories'][cat_name] = dict(zip(keys, stat_values))
        out[team_id] = list(players_by_id.values())
    return out


def get_live_boxscore(event_id):
    """Fetch+parse convenience for callers with no summary already in
    hand (e.g. nfl_stats_backfill.py). nfl_api.py itself should prefer
    parse_player_boxscore(summary) directly — it already has the summary
    fetched via _get_game_summary(), so calling this instead would mean a
    second, redundant HTTP round trip for the same data."""
    return parse_player_boxscore(_get(event_id))
