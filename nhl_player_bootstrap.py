#!/usr/bin/env python3
"""
nhl_player_bootstrap.py — Offline backtest + fit for nhl_player_model.py.

Usage:
    python3 nhl_player_bootstrap.py --max-games 50     # quick smoke test
    python3 nhl_player_bootstrap.py                    # full SEASONS run

For each historical game, reconstructs each team's skater grade and
goalie grade using ONLY games strictly before that date (no-leakage —
same rule as nfl_player_bootstrap.py), fits the logistic P(home_win) ~
sigmoid(a + b*matchup_diff), and reports accuracy.

No injury exclusion in this backtest (unavailable_ids is always empty
below) — InjuryStatus didn't exist before this feature, so there's no
historical "who was actually out" to reconstruct for past seasons; this
backtest necessarily grades the PLAIN (un-excluded) skater/goalie signal,
same documented limitation nba_player_bootstrap.py already has for its
own "who actually played" proxy.

Schedule/boxscore source: reads nhl_stats_db.py's local warehouse — same
reasoning as nfl_player_bootstrap.py (one source of truth with the live
app, no redundant live-ESPN-fetch-with-disk-cache).
"""
import math
import os
import sys
from collections import defaultdict

SEASONS = [2024, 2025]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nhl_player_model as pm
import nhl_stats_db


def fetch_season_schedule(season):
    """Thin wrapper over nhl_stats_db.get_schedule_db() — kept as its own
    function so nhl_ensemble_bootstrap.py's `player_bt.fetch_season_schedule()`
    calls keep working unchanged."""
    return nhl_stats_db.get_schedule_db(season, game_type='regular')


def fetch_boxscore(event_id):
    """{team_id: [{id, name, position, is_goalie, stats}]} for this game."""
    return nhl_stats_db.get_game_boxscore_db(event_id) or None


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _fit_logistic_1d(x, y, iters=20000, lr=0.3, lam=1e-4):
    """Same tiny no-sklearn logistic regression as nfl_player_bootstrap.py."""
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

    # Point-in-time per-team history: list of GAMES, each a list of that
    # game's player rows (raw, not collapsed) — newest-first, appended
    # AFTER each game is scored (no leakage). Unlike nfl_player_bootstrap.py's
    # pre-aggregated offense/defense dicts, this stays raw so
    # compute_skater_grade()/identify_starting_goalie()/compute_goalie_
    # grade_excluding() can all work from the same history the live app uses.
    team_history = defaultdict(list)

    rows = []
    for i, g in enumerate(all_games):
        box = fetch_boxscore(g['event_id'])
        if not box:
            continue
        home_rows = box.get(g['home_id'], [])
        away_rows = box.get(g['away_id'], [])
        if not home_rows or not away_rows:
            continue

        home_skater = pm.compute_skater_grade(team_history[g['home_id']])
        away_skater = pm.compute_skater_grade(team_history[g['away_id']])
        # No injury exclusion possible in this backtest (see module
        # docstring) — empty unavailable_ids grades the plain signal.
        home_goalie = pm.compute_goalie_grade_excluding(team_history[g['home_id']], set())
        away_goalie = pm.compute_goalie_grade_excluding(team_history[g['away_id']], set())

        pred = pm.predict(home_skater, home_goalie, away_skater, away_goalie)

        home_won = g['home_score'] > g['away_score']
        if pred is not None:
            rows.append({
                'date': g['game_date'],
                'matchup_diff': pred['matchup_diff'],
                'home_won': home_won,
            })

        # ── Update running state AFTER scoring (no leakage) ──────────────
        team_history[g['home_id']].insert(0, home_rows)
        team_history[g['away_id']].insert(0, away_rows)

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

    print(f'Running NHL player-model backtest over {SEASONS}...', flush=True)
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
    print(f'home-ice-always baseline: {home_win_rate*100:.1f}% win rate')
