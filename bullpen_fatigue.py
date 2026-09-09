"""
bullpen_fatigue.py — Point-in-time bullpen workload from MLB box scores.

Reconstructs, for every completed game, how many innings each team's
BULLPEN (non-starter pitchers) threw, via /api/v1/game/{gamePk}/boxscore.
That endpoint's `pitchers` list is starter-first, and each player's own
stats.pitching.gamesStarted flag cleanly distinguishes the starter (1) from
every reliever who appeared (0) — verified against a real box score. One
call covers both teams in a game, so this is one request per game, not per
team.

This is a genuinely different, more solid attempt than the earlier
mlb_model.py "Closer Yday" factor that got dropped on n=7: that relied on a
live-only injury/availability feed with no historical source. Bullpen usage
isn't an availability report — it's just "who pitched and for how long,"
which is fully reconstructable from box scores across the whole season, so
it can actually be backtested properly instead of eyeballed on a handful of
live games.

Point-in-time by construction: get_bullpen_fatigue_asof() only ever sums
innings from games strictly BEFORE as_of_date, using per-game cached data —
same "exact regardless of call order" property as xwoba_rolling.py, for the
same reason (each game's contribution is stored once, keyed by date, and
summed fresh per query rather than folded into a running total).

Cached on disk per season: {gamePk: {'date', 'home_id', 'away_id',
'home_bp_ip', 'away_bp_ip'}}. Incremental — ensure_games_cached() only
fetches gamePks not already cached, so a rerun with 1 new day of games only
costs that day's handful of boxscore calls, not the whole season. A cold
start for a season already in progress is slow (one request per game, ~0.1s
sleep between each) — same one-time-cost tradeoff as xwoba_rolling.py.
"""
import json
import os
import time
from datetime import date, timedelta

import requests

_BOXSCORE_URL = 'https://statsapi.mlb.com/api/v1/game/{}/boxscore'
_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; mlb-bootstrap/1.0)'}
_HERE = os.path.dirname(os.path.abspath(__file__))


def _cache_path(season):
    return os.path.join(_HERE, f'bullpen_cache_{season}.json')


def _load_cache(season):
    path = _cache_path(season)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(season, cache):
    path = _cache_path(season)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except Exception as e:
        print(f'  ! bullpen_fatigue: failed to save cache for {season}: {e}', flush=True)


def _ip_to_float(ip_str):
    """MLB innings-pitched strings use .1/.2 for thirds of an inning (1 or 2
    outs), not decimal tenths — '6.1' means 6 and 1/3 innings, not 6.1."""
    if not ip_str:
        return 0.0
    try:
        whole, _, frac = str(ip_str).partition('.')
        whole = int(whole or 0)
        frac = int(frac or 0)  # 0, 1, or 2 (thirds of an inning)
        return whole + frac / 3.0
    except (TypeError, ValueError):
        return 0.0


def _fetch_boxscore_bullpen_ip(game_pk, retries=3):
    """Returns {'home': ip, 'away': ip} (bullpen-only, starter excluded), or
    None on failure."""
    url = _BOXSCORE_URL.format(game_pk)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=20)
            r.raise_for_status()
            d = r.json()
            out = {}
            for side in ('home', 'away'):
                team = d.get('teams', {}).get(side, {})
                players = team.get('players', {})
                pitcher_ids = team.get('pitchers', [])
                bp_ip = 0.0
                for pid in pitcher_ids:
                    p = players.get(f'ID{pid}', {})
                    stat = (p.get('stats') or {}).get('pitching') or {}
                    if stat.get('gamesStarted'):
                        continue  # the starter — excluded from the bullpen total
                    bp_ip += _ip_to_float(stat.get('inningsPitched'))
                out[side] = round(bp_ip, 3)
            return out
        except Exception as e:
            if attempt == retries - 1:
                print(f'  ! bullpen_fatigue: boxscore {game_pk} failed: {e}', flush=True)
                return None
            time.sleep(1 + attempt)
    return None


def ensure_games_cached(season, games, sleep_s=0.1):
    """
    games: list of dicts with 'game_pk', 'game_date', 'home_id', 'away_id'.

    Fetches any gamePk not already cached, saving progress every 20 games —
    a full-season cold start is 1000+ boxscore calls, so this bounds how
    much work an interruption throws away (matches xwoba_rolling.py's
    rationale for periodic saves).

    Returns the (possibly-updated) cache dict.
    """
    cache = _load_cache(season)
    fetched_since_save = 0
    for g in games:
        gpk = str(g['game_pk'])
        if gpk in cache:
            continue
        result = _fetch_boxscore_bullpen_ip(g['game_pk'])
        if result is None:
            continue  # leave uncached; retried on next call/run
        cache[gpk] = {
            'date':        g['game_date'],
            'home_id':     g['home_id'],
            'away_id':     g['away_id'],
            'home_bp_ip':  result.get('home', 0.0),
            'away_bp_ip':  result.get('away', 0.0),
        }
        fetched_since_save += 1
        if fetched_since_save >= 20:
            _save_cache(season, cache)
            fetched_since_save = 0
        time.sleep(sleep_s)
    if fetched_since_save:
        _save_cache(season, cache)
    return cache


def load_cache(season):
    """Thin wrapper so callers that already know the cache is populated
    (e.g. a repeated lookup loop) don't need to import _load_cache directly."""
    return _load_cache(season)


def get_bullpen_fatigue_asof(cache, team_id, as_of_date, window_days=3):
    """
    Sum of bullpen innings pitched by team_id in games strictly before
    as_of_date, within the last `window_days` calendar days.

    `cache` is whatever ensure_games_cached()/load_cache() returned — pass
    the same cache into repeated calls rather than reloading from disk each
    time.

    Returns None if the team has no cached games in that window — callers
    should treat that as "unknown," the same convention as every other
    optional factor in build_team_dict (missing, not zero).
    """
    def _parse(s):
        y, m, d = (int(x) for x in s.split('-'))
        return date(y, m, d)

    target = _parse(as_of_date)
    window_start = target - timedelta(days=window_days)

    total = 0.0
    found = False
    for g in cache.values():
        gdate = _parse(g['date'])
        if not (window_start <= gdate < target):
            continue
        if g['home_id'] == team_id:
            total += g['home_bp_ip']
            found = True
        elif g['away_id'] == team_id:
            total += g['away_bp_ip']
            found = True

    return round(total, 2) if found else None
