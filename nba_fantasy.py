"""
nba_fantasy.py — FanDuel standard NBA fantasy scoring, the basketball
counterpart to nfl_fantasy.py's compute_fantasy_points()/compute_
dst_fantasy_points(). Deliberately minimal: just the scoring formula,
read by nba_game_simulator.py to convert a simulated box-score line into
fantasy points. No DFS pool-building, salary optimizer, or projection
machinery here — those are separate concerns nfl_fantasy.py bundles
together for historical reasons; nothing downstream needs an NBA
equivalent of them yet.

No DST-equivalent exists in basketball (no concept of a defensive unit
separate from the 5 players on the floor), so unlike nfl_fantasy.py this
module has no points-allowed/sack/INT scoring at all.
"""

# ── FanDuel standard NBA scoring (publicly published ruleset) ─────────────
PTS = 1.0
REB = 1.2
AST = 1.5
BLK = 3.0
STL = 3.0
TOV = -1.0


def _num(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def compute_fantasy_points(categories):
    """One player's FanDuel fantasy points for a single game, from a flat
    `categories` dict: {points, rebounds, assists, steals, blocks,
    turnovers} — the same field names PlayerGameStat's NBA columns use
    (see nba_stats_db.py). Missing categories simply contribute 0."""
    cats = categories or {}
    pts = 0.0
    pts += _num(cats.get('points')) * PTS
    pts += _num(cats.get('rebounds')) * REB
    pts += _num(cats.get('assists')) * AST
    pts += _num(cats.get('steals')) * STL
    pts += _num(cats.get('blocks')) * BLK
    pts += _num(cats.get('turnovers')) * TOV
    return round(pts, 2)
