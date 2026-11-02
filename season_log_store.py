"""
season_log_store.py — disk copies of completed seasons' game logs.

Each sport's _get_season_game_log() reads the local stats warehouse first
and only falls back to a live ESPN walk (18 NFL weeks, 15 CFB weeks, ~270
NBA days) when the DB has nothing for that season — which is always the
case for the PRIOR season the early-season blend needs, unless it was
backfilled. A completed season never changes, so after one successful walk
it's kept here, next to bets.db (same convention as cfb_rating.py and
market_edge_calibration.py), and every later cold worker reads the file
instead of re-walking ESPN.
"""
import json
import os

_DATA_DIR = os.path.dirname(os.environ.get('DB_PATH', '')) or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')


def _path(sport, season):
    return os.path.join(_DATA_DIR, f'{sport}_season_log_{season}.json')


def load(sport, season):
    """The stored game log for a completed season, or None if there isn't one."""
    try:
        with open(_path(sport, season)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save(sport, season, games):
    """Store a completed season's game log. Callers only pass a log built
    from a walk where every request succeeded — a failed one would leave
    games missing from the file for good."""
    path = _path(sport, season)
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(games, f)
        os.replace(tmp, path)
    except OSError:
        pass
