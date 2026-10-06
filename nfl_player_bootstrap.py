#!/usr/bin/env python3
"""
nfl_player_bootstrap.py — Offline backtest + fit for nfl_player_model.py.

Usage:
    python3 nfl_player_bootstrap.py --max-games 50     # quick smoke test
    python3 nfl_player_bootstrap.py                    # full SEASONS run

For each historical game, reconstructs each team's offense/defense grade
using ONLY games strictly before that date (no-leakage — same rule as
nba_player_bootstrap.py), fits the logistic P(home_win) ~
sigmoid(a + b*matchup_diff) via the same no-sklearn gradient descent this
app already uses (nba_ensemble_bootstrap._fit_logistic_2d, 1D version
here), and reports accuracy against the actual team-aggregate model
(nfl_model.py) so the two can be compared honestly before any blend.

Schedule/boxscore source: reads nfl_stats_db.py's local warehouse (fetch_
season_schedule/fetch_boxscore below), not a live ESPN fetch — now that
nfl_stats_backfill.py has loaded full 2024+2025 seasons there (see that
script's module docstring), re-fetching independently would just be
redundant work against the same data. This used to hit ESPN directly with
its own disk cache; a full backtest run is now a DB-only operation.
"""
import math
import os
import sys
from collections import defaultdict

SEASONS = [2024, 2025]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfl_player_model as pm
import nfl_stats_db


def fetch_season_schedule(season):
    """Completed NFL regular-season games for `season`: [{event_id,
    game_date, home_id, away_id, home_score, away_score}], chronological.
    Thin wrapper over nfl_stats_db.get_schedule_db() — kept as its own
    function so nfl_ensemble_bootstrap.py's existing
    `player_bt.fetch_season_schedule(season)` calls keep working unchanged."""
    return nfl_stats_db.get_schedule_db(season, game_type='regular')


def fetch_boxscore(event_id):
    """{team_id: [{id, name, categories}]} for this game. Thin wrapper over
    nfl_stats_db.get_game_boxscore_db() — see fetch_season_schedule()'s
    docstring for why this stays a function rather than inlining."""
    return nfl_stats_db.get_game_boxscore_db(event_id) or None


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _fit_logistic_1d(x, y, iters=20000, lr=0.3, lam=1e-4):
    """Tiny no-sklearn logistic regression: y ~ sigmoid(a + b*x). Same
    gradient-descent approach as nba_ensemble_bootstrap._fit_logistic_2d,
    1 feature instead of 2."""
    a = b = 0.0
    n = len(y)
    for _ in range(iters):
        ga = gb = 0.0
        for xi, yi in zip(x, y):
            p = _sigmoid(a + b * xi)
            err = p - yi
            ga += err
            gb += err * xi
        a -= lr * (ga / n + lam * a)
        b -= lr * (gb / n + lam * b)
    return a, b


def run(max_games=None):
    all_games = []
    for season in SEASONS:
        all_games.extend(fetch_season_schedule(season))
    all_games.sort(key=lambda g: g['game_date'])
    if max_games:
        all_games = all_games[:max_games]

    # Point-in-time per-team history: offense/defense aggregate dicts,
    # newest-first, appended AFTER each game is scored (no leakage).
    off_history = defaultdict(list)
    def_history = defaultdict(list)

    rows = []
    for i, g in enumerate(all_games):
        box = fetch_boxscore(g['event_id'])
        if not box:
            continue
        home_rows = box.get(g['home_id'], [])
        away_rows = box.get(g['away_id'], [])
        if not home_rows or not away_rows:
            continue

        home_off_grade = pm.compute_offense_grade(off_history[g['home_id']])
        home_def_grade = pm.compute_defense_grade(def_history[g['home_id']])
        away_off_grade = pm.compute_offense_grade(off_history[g['away_id']])
        away_def_grade = pm.compute_defense_grade(def_history[g['away_id']])

        pred = pm.predict(home_off_grade, home_def_grade, away_off_grade, away_def_grade)

        home_won = g['home_score'] > g['away_score']
        if pred is not None:
            rows.append({
                'date': g['game_date'],
                'matchup_diff': pred['matchup_diff'],
                'home_won': home_won,
            })

        # ── Update running state AFTER scoring (no leakage) ──────────────
        home_off, home_def = pm.parse_team_game_aggregate(home_rows)
        away_off, away_def = pm.parse_team_game_aggregate(away_rows)
        off_history[g['home_id']].insert(0, home_off)
        off_history[g['away_id']].insert(0, away_off)
        def_history[g['home_id']].insert(0, home_def)
        def_history[g['away_id']].insert(0, away_def)

        if (i + 1) % 100 == 0:
            print(f'  processed {i+1}/{len(all_games)}, {len(rows)} usable', flush=True)

    return rows


def evaluate(rows, a, b):
    correct = 0
    brier_sum = 0.0
    for r in rows:
        p = _sigmoid(a + b * r['matchup_diff'])
        pred_home = p > 0.5
        correct += int(pred_home == r['home_won'])
        brier_sum += (p - (1.0 if r['home_won'] else 0.0)) ** 2
    n = len(rows)
    return (correct / n if n else 0.0), (brier_sum / n if n else 0.0)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max-games', type=int, default=None)
    args = p.parse_args()

    print(f'Running NFL player-model backtest over {SEASONS}...', flush=True)
    rows = run(max_games=args.max_games)
    print(f'\n{len(rows)} usable games.', flush=True)

    if len(rows) < 20:
        print('Too few games to fit — aborting.')
        sys.exit(1)

    x = [r['matchup_diff'] for r in rows]
    y = [1.0 if r['home_won'] else 0.0 for r in rows]
    a, b = _fit_logistic_1d(x, y)
    acc, brier = evaluate(rows, a, b)

    home_win_rate = sum(y) / len(y)
    print(f'\nfit: INTERCEPT = {a:.4f}, SLOPE = {b:.4f}')
    print(f'player-grade model: {acc*100:.1f}% win accuracy, {brier:.4f} Brier  (n={len(rows)})')
    print(f'home-field-always baseline: {home_win_rate*100:.1f}% win rate')
