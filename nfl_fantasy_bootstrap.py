#!/usr/bin/env python3
"""
nfl_fantasy_bootstrap.py — Offline fit for nfl_fantasy.py's PROJECTION_FIT.

Usage:
    python3 nfl_fantasy_bootstrap.py --max-games 50     # quick smoke test
    python3 nfl_fantasy_bootstrap.py                    # full SEASONS run

For each stored game since 2024, reconstructs — strictly from games
BEFORE that date, same no-leakage rule as nfl_player_bootstrap.py — each
player's own rolling-average fantasy output and their opponent's unit
grade (defense grade for QB/RB/REC rows, offense grade for DST rows), then
fits actual fantasy points scored ~ a + b*recent_avg + c*opponent_grade by
ordinary least squares (closed-form normal equations — no numpy/sklearn in
this codebase, same "small custom fit, no new ML dependency" convention as
nfl_player_bootstrap.py's hand-rolled logistic regression).

Reports each bucket's fit against a naive baseline (projecting the recent
average alone, with no opponent adjustment) so it's clear the opponent term
is actually earning its keep before it ships — same spirit as
nfl_player_bootstrap.py comparing its fit to a home-field-always baseline.

DOCUMENTED LIMITATION (same shape as nba_player_bootstrap.py's): this is
fit AND evaluated on the same rows (in-sample), not a held-out test set —
consistent with how the rest of this app's model fits are reported, but it
means these MAE numbers are a best case, not a guarantee of live accuracy.

Position bucketing here is role-based (nfl_fantasy.classify_bucket), not
roster-based — see that function's docstring for why a historical backtest
can't lean on nfl_roster_api's CURRENT-roster position lookup.
"""
import os
import sys
from collections import defaultdict

SEASONS = [2024, 2025]
MIN_ROWS_TO_FIT = 30

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfl_fantasy as nf
import nfl_player_model as pm
import nfl_stats_db


def fetch_season_schedule(season):
    return nfl_stats_db.get_schedule_db(season, game_type='regular')


def fetch_boxscore(event_id):
    return nfl_stats_db.get_game_boxscore_db(event_id) or None


def _solve_3x3(A, b):
    """Gaussian elimination with partial pivoting for a 3x3 system. A:
    3 rows of 3 coefficients, b: 3 constants. Returns [x0, x1, x2], or None
    if the system is singular (degenerate/collinear training data)."""
    n = 3
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            return None
        M[col], M[pivot_row] = M[pivot_row], M[col]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_ols(rows):
    """rows: [(x1, x2, y), ...]. Fits y ~ a + b1*x1 + b2*x2 by OLS normal
    equations. Returns {'intercept', 'recent_avg', 'opp_grade'} or None if
    too few rows / a degenerate fit."""
    n = len(rows)
    if n < MIN_ROWS_TO_FIT:
        return None
    s1 = sx1 = sx2 = sx1x1 = sx2x2 = sx1x2 = sy = sx1y = sx2y = 0.0
    for x1, x2, y in rows:
        s1 += 1.0
        sx1 += x1; sx2 += x2
        sx1x1 += x1 * x1; sx2x2 += x2 * x2; sx1x2 += x1 * x2
        sy += y; sx1y += x1 * y; sx2y += x2 * y

    A = [[s1, sx1, sx2], [sx1, sx1x1, sx1x2], [sx2, sx1x2, sx2x2]]
    b = [sy, sx1y, sx2y]
    sol = _solve_3x3(A, b)
    if sol is None:
        return None
    return {'intercept': sol[0], 'recent_avg': sol[1], 'opp_grade': sol[2]}


def _mae(rows, predict_fn):
    if not rows:
        return None
    return sum(abs(predict_fn(x1, x2) - y) for x1, x2, y in rows) / len(rows)


def run(max_games=None):
    all_games = []
    for season in SEASONS:
        all_games.extend(fetch_season_schedule(season))
    all_games.sort(key=lambda g: g['game_date'])
    if max_games:
        all_games = all_games[:max_games]

    player_hist = defaultdict(list)   # player_id -> fantasy pts, newest-first
    dst_hist    = defaultdict(list)   # team_id -> DST fantasy pts, newest-first
    off_hist    = defaultdict(list)   # team_id -> offense aggregate dicts, newest-first
    def_hist    = defaultdict(list)   # team_id -> defense aggregate dicts, newest-first

    rows = defaultdict(list)  # bucket -> [(recent_avg, opp_grade, actual_pts)]
    n_games_used = 0

    for i, g in enumerate(all_games):
        box = fetch_boxscore(g['event_id'])
        if not box:
            continue
        home_rows = box.get(g['home_id'], [])
        away_rows = box.get(g['away_id'], [])
        if not home_rows or not away_rows:
            continue
        n_games_used += 1

        home_def_grade = pm.compute_defense_grade(def_hist[g['home_id']])
        away_def_grade = pm.compute_defense_grade(def_hist[g['away_id']])
        home_off_grade = pm.compute_offense_grade(off_hist[g['home_id']])
        away_off_grade = pm.compute_offense_grade(off_hist[g['away_id']])

        for side_rows, opp_def_grade in ((home_rows, away_def_grade), (away_rows, home_def_grade)):
            if opp_def_grade is None:
                continue
            for r in side_rows:
                pid = r.get('id')
                if not pid:
                    continue
                cats = r.get('categories', {})
                bucket = nf.classify_bucket(cats)
                if bucket is None:
                    continue
                pts = nf.compute_fantasy_points(cats)
                recent_avg = nf.rolling_average(player_hist[pid])
                if recent_avg is not None:
                    rows[bucket].append((recent_avg, opp_def_grade, pts))
                player_hist[pid].insert(0, pts)

        home_pa, away_pa = g['away_score'], g['home_score']
        for team_id, side_rows, opp_off_grade, points_allowed in (
            (g['home_id'], home_rows, away_off_grade, home_pa),
            (g['away_id'], away_rows, home_off_grade, away_pa),
        ):
            if opp_off_grade is None:
                continue
            dst_pts = nf.compute_dst_fantasy_points(points_allowed, nf.aggregate_dst_game(side_rows))
            recent_avg = nf.rolling_average(dst_hist[team_id])
            if recent_avg is not None:
                rows['DST'].append((recent_avg, opp_off_grade, dst_pts))
            dst_hist[team_id].insert(0, dst_pts)

        home_off, home_def = pm.parse_team_game_aggregate(home_rows)
        away_off, away_def = pm.parse_team_game_aggregate(away_rows)
        off_hist[g['home_id']].insert(0, home_off)
        off_hist[g['away_id']].insert(0, away_off)
        def_hist[g['home_id']].insert(0, home_def)
        def_hist[g['away_id']].insert(0, away_def)

        if (i + 1) % 100 == 0:
            print(f'  processed {i+1}/{len(all_games)}, games with box scores: {n_games_used}', flush=True)

    return rows


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max-games', type=int, default=None)
    args = p.parse_args()

    print(f'Running NFL fantasy-projection backtest over {SEASONS}...', flush=True)
    rows_by_bucket = run(max_games=args.max_games)

    print(f'\nPROJECTION_FIT = {{', flush=True)
    for bucket in ('QB', 'RB', 'REC', 'DST'):
        rows = rows_by_bucket.get(bucket, [])
        fit = fit_ols(rows)
        if fit is None:
            print(f"  # {bucket}: only {len(rows)} usable rows (need >= {MIN_ROWS_TO_FIT}) — kept default")
            continue

        def _fit_predict(x1, x2, f=fit):
            return max(0.0, f['intercept'] + f['recent_avg'] * x1 + f['opp_grade'] * x2)

        def _baseline_predict(x1, x2):
            return x1

        mae_fit = _mae(rows, _fit_predict)
        mae_baseline = _mae(rows, _baseline_predict)
        print(f"  '{bucket}': {{'intercept': {fit['intercept']:.4f}, "
              f"'recent_avg': {fit['recent_avg']:.4f}, 'opp_grade': {fit['opp_grade']:.4f}}},"
              f"  # n={len(rows)}, MAE {mae_fit:.2f} vs baseline (recent-avg-only) MAE {mae_baseline:.2f}")
    print('}', flush=True)
