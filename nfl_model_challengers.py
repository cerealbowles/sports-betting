#!/usr/bin/env python3
"""
nfl_model_challengers.py — Phase 4: test challengers to project_points()
(NFL_Fantasy_Analytics_Projection_Plan.md, section 6), using the Phase 2
dataset and Phase 3's enrich() features.

Usage:
    python3 nfl_model_challengers.py nfl_playergame_dataset.jsonl

MODEL 0 (the existing baseline) is NOT refit here — every row in the
Phase 2 dataset already carries `projection`, the real point-in-time
output of the live project_points() (same coefficients currently in
nfl_fantasy.PROJECTION_FIT). Evaluating it is just scoring that column
against `actual` on the holdout — this guarantees Model 0's reported
number is what the live app would actually have produced, not a
re-derived approximation of it.

CHALLENGERS (Model 1 family — "opportunity-aware", per the plan's own
section 6): each adds exactly ONE feature on top of Model 0's two inputs
(recent_avg, opponent_defense_grade) and is fit via ordinary least squares
(closed-form normal equations, pure Python — no numpy/sklearn in this
project's dependencies, same "no-sklearn fit" convention nfl_player_
bootstrap.py already uses for its own logistic fit) on the TRAIN split
only (seasons 2024-2025):
  - volume: recent touches/game appropriate to the bucket (pass attempts
    for QB, rush+target touches for RB, targets for REC) — the plan's own
    "does volume add information beyond the recent average?" question.
  - td_trend: recent-vs-season TD-rate trend — the ONE signal Phase 3's
    miss analysis actually validated on a holdout (nfl_miss_analysis.py).
  - combined: both features together.

EVALUATION: every model is scored on the IDENTICAL holdout row set per
bucket (season 2026, intersected across whichever features each model
needs, so a bucket's MAE comparison is always apples-to-apples) — MAE,
RMSE, and bias (mean actual-minus-predicted, to catch systematic over/
under-projection the plan explicitly asks for, not just magnitude).
Promotion verdict follows the plan's own section 6 criteria: a challenger
must beat Model 0's MAE, AND the gain must hold on both halves of the
holdout split chronologically (a single favorable split isn't enough).

This script decides nothing on its own — it's reporting, not deploying.
Promoting a challenger means editing nfl_fantasy.PROJECTION_FIT, a
separate, deliberate change to the live app, not a side effect of running
this file.
"""
import sys

import nfl_miss_analysis as ma
import nfl_playergame_dataset as ds

BUCKETS = ('QB', 'RB', 'REC')


def _solve_linear_system(A, b):
    """Gaussian elimination with partial pivoting. A: n x n, b: length n.
    Raises ValueError on a singular/near-singular system (collinear
    features) rather than returning a garbage solution."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot_row] = M[pivot_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-9:
            raise ValueError('singular matrix — features are collinear or too few rows')
        for r in range(col + 1, n):
            factor = M[r][col] / pivot
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    beta = [0.0] * n
    for row in range(n - 1, -1, -1):
        s = M[row][n] - sum(M[row][c] * beta[c] for c in range(row + 1, n))
        beta[row] = s / M[row][row]
    return beta


def ols_fit(feature_rows, targets):
    """feature_rows: list of feature vectors, each already including the
    leading 1.0 intercept term. targets: list of floats, same length.
    Returns the fitted coefficient vector."""
    k = len(feature_rows[0])
    XtX = [[0.0] * k for _ in range(k)]
    Xty = [0.0] * k
    for row, y in zip(feature_rows, targets):
        for i in range(k):
            Xty[i] += row[i] * y
            for j in range(k):
                XtX[i][j] += row[i] * row[j]
    return _solve_linear_system(XtX, Xty)


def predict(coefs, feature_row):
    return sum(c * x for c, x in zip(coefs, feature_row))


FEATURE_SPECS = {
    'volume': lambda r: r['_recent_volume_pg'],
    'td_trend': lambda r: r['_td_rate_trend'],
}


def _base_features(row):
    return [1.0, row['recent_fantasy_pts_avg'], row['opponent_defense_grade']]


def _usable(row, extra_keys):
    if row['projection'] is None or row['actual'] is None:
        return False
    if row['recent_fantasy_pts_avg'] is None or row['opponent_defense_grade'] is None:
        return False
    for key in extra_keys:
        if FEATURE_SPECS[key](row) is None:
            return False
    return True


def mae(errors):
    return sum(abs(e) for e in errors) / len(errors) if errors else None


def rmse(errors):
    return (sum(e * e for e in errors) / len(errors)) ** 0.5 if errors else None


def bias(errors):
    return (sum(errors) / len(errors)) if errors else None


def evaluate_model(rows, predict_fn):
    """predict_fn(row) -> predicted points. Returns {'mae', 'rmse', 'bias', 'n'}
    using (actual - predicted) as the error, so a positive bias means the
    model UNDER-projects on average, negative means it OVER-projects —
    same sign convention as projection_error elsewhere in this app."""
    errors = [r['actual'] - predict_fn(r) for r in rows]
    return {'mae': mae(errors), 'rmse': rmse(errors), 'bias': bias(errors), 'n': len(rows)}


def run(path):
    rows = ds.load_jsonl(path)
    for r in rows:
        ma.enrich(r)
    train_rows, holdout_rows = ds.season_split(rows, ma.HOLDOUT_SEASON)
    print(f'{len(rows)} rows — train {len(train_rows)}, holdout {len(holdout_rows)}\n')

    challengers = {
        'volume': ['volume'],
        'td_trend': ['td_trend'],
        'combined': ['volume', 'td_trend'],
    }

    overall_verdicts = {}

    for bucket in BUCKETS:
        print(f'=== {bucket} ===')
        bucket_train = [r for r in train_rows if r['bucket'] == bucket]
        bucket_holdout = [r for r in holdout_rows if r['bucket'] == bucket]

        # Model 0: no fitting — score the dataset's own `projection` column.
        model0_rows = [r for r in bucket_holdout if _usable(r, [])]
        model0_eval = evaluate_model(model0_rows, lambda r: r['projection'])
        print(f"Model 0 (existing baseline): MAE={model0_eval['mae']:.3f} "
              f"RMSE={model0_eval['rmse']:.3f} bias={model0_eval['bias']:+.3f} (n={model0_eval['n']})")

        for name, extra_keys in challengers.items():
            train_usable = [r for r in bucket_train if _usable(r, extra_keys)]
            holdout_usable = [r for r in bucket_holdout if _usable(r, extra_keys)]
            if len(train_usable) < 30 or len(holdout_usable) < 15:
                print(f'  {name}: skipped (too few usable rows — train={len(train_usable)}, holdout={len(holdout_usable)})')
                continue

            def feat_row(r, extra_keys=extra_keys):
                return _base_features(r) + [FEATURE_SPECS[k](r) for k in extra_keys]

            X_train = [feat_row(r) for r in train_usable]
            y_train = [r['actual'] for r in train_usable]
            coefs = ols_fit(X_train, y_train)

            # Score Model 0 AND the challenger on the IDENTICAL holdout
            # subset this challenger actually needed (apples-to-apples;
            # model0_rows above used a broader subset that doesn't require
            # the challenger's extra feature).
            model0_on_same_rows = evaluate_model(holdout_usable, lambda r: r['projection'])
            challenger_eval = evaluate_model(holdout_usable, lambda r, c=coefs: predict(c, feat_row(r)))

            mae_delta = model0_on_same_rows['mae'] - challenger_eval['mae']
            improved = mae_delta > 0

            # Consistency check: split the holdout chronologically in half
            # — the plan's own "consistent gains across time periods, not
            # just one favorable split" criterion.
            sorted_rows = sorted(holdout_usable, key=lambda r: r['game_date'])
            mid = len(sorted_rows) // 2
            half1, half2 = sorted_rows[:mid], sorted_rows[mid:]
            consistent = None
            if len(half1) >= 10 and len(half2) >= 10:
                h1_m0 = evaluate_model(half1, lambda r: r['projection'])['mae']
                h1_ch = evaluate_model(half1, lambda r, c=coefs: predict(c, feat_row(r)))['mae']
                h2_m0 = evaluate_model(half2, lambda r: r['projection'])['mae']
                h2_ch = evaluate_model(half2, lambda r, c=coefs: predict(c, feat_row(r)))['mae']
                consistent = (h1_ch < h1_m0) and (h2_ch < h2_m0)

            coef_str = ', '.join(f'{k}={c:.3f}' for k, c in zip(['intercept', 'recent_avg', 'opp_grade'] + extra_keys, coefs))
            print(f'  {name}: fit [{coef_str}]')
            print(f"    holdout MAE={challenger_eval['mae']:.3f} (Model 0 on same rows: {model0_on_same_rows['mae']:.3f}, "
                  f"delta={mae_delta:+.3f}) bias={challenger_eval['bias']:+.3f} n={challenger_eval['n']}")
            verdict = 'PROMOTE' if (improved and consistent) else ('IMPROVED BUT INCONSISTENT' if improved else 'REJECT')
            print(f'    both-halves-improved={consistent}  VERDICT: {verdict}')
            overall_verdicts[(bucket, name)] = verdict
        print()

    print('=== SUMMARY ===')
    for (bucket, name), verdict in overall_verdicts.items():
        print(f'  {bucket} / {name}: {verdict}')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'nfl_playergame_dataset.jsonl'
    run(path)
