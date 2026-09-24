#!/usr/bin/env python3
"""
spread_model_fit.py — One-time (well, "refit periodically") fit of a real
margin/spread model per sport, built from scratch — NOT derived from the
win-probability model's logit like the old spread_proxy.py.

Reuses each sport's existing win-prob factor decomposition (factors_json,
stored on every game_predictions row at prediction time — [(label, logit
contribution), ...]) as the regression's input columns, but fits an
INDEPENDENT set of weights per factor against actual final-score margin
(home_score - away_score), the same way mlb_total_model.py etc. fit their
own coefficients against actual totals rather than reusing win-prob's.

Label canonicalization: MLB's factor labels carry a leading team
abbreviation (e.g. "BOS SP Barrel%", "TB OPS" — always labeled with
whichever team's stat drove the differential, see mlb_model.py's `_add`
calls). Left as-is, that's ~300 near-one-hot columns on ~2,300 games —
severe overfitting (first pass: OOS R² was NEGATIVE). Detected and
stripped here data-drivenly: any label whose "suffix after the first
space" appears under 2+ different first tokens across the dataset is
treated as team-prefixed and canonicalized to just the suffix (e.g. "BOS
SP Barrel%" and "TB SP Barrel%" both become "SP Barrel%") — collapsing to
~13 shared categories, the same shape as every other sport's factor list.
Labels that only ever appear one way (e.g. "Home Field", "H2H Record")
are left untouched.

Methodology (mirrors spread_proxy.py's documented validation approach):
  1. Chronological 70/30 split per sport (train on the older 70%, hold out
     the newest 30% — never a random split).
  2. Ridge regression (L2-regularized) on the train split: margin ~ sum(w_i
     * factor_i) + intercept. Alpha picked per sport via 5-fold CV on the
     train split (grid search), not a fixed guess — the whole point of
     regularizing is to let held-out performance pick how much to trust
     this factor count against this sample size, not eyeball it.
  3. Evaluate on the held-out test split: correlation, R², residual sigma
     (for the cover-probability normal approximation), and — where a real
     market spread line exists (spread_close/open, ESPN-backfilled,
     recent-only) — ATS win rate against that line, directly comparable to
     spread_proxy's own reported cover accuracy and to break-even (52.4%
     at standard -110 vig).
  4. Refit on the FULL dataset (same alpha) for the production
     coefficients; the held-out numbers from step 3 are what's reported
     for honesty, same as the total models' docstrings.

Usage: python3 spread_model_fit.py
Prints per-sport validation numbers, then a COEFFS dict ready to paste
into spread_model.py.
"""
import json
import os

os.environ['DISABLE_STARTUP_TASKS'] = '1'

import numpy as np

SPORTS = ['MLB', 'NFL', 'CFB', 'NBA', 'NHL', 'WNBA']
MIN_GAMES_FOR_FIT = 100
ALPHA_GRID = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]


def _load_rows(GamePrediction, sport):
    rows = GamePrediction.query.filter(
        GamePrediction.sport == sport,
        GamePrediction.home_won.isnot(None),
        GamePrediction.home_score.isnot(None),
        GamePrediction.away_score.isnot(None),
    ).order_by(GamePrediction.game_date.asc(), GamePrediction.id.asc()).all()
    out = []
    for r in rows:
        try:
            factors = json.loads(r.factors_json or '[]')
        except Exception:
            continue
        if not factors:
            continue
        out.append({
            'game_date': r.game_date,
            'margin': r.home_score - r.away_score,
            'factors': factors,
            'spread_line': r.spread_close if r.spread_close is not None else r.spread_open,
            'home_prob': r.home_prob,
        })
    return out


# mlb_model.py falls back to these when a team dict has no 'abbr' key
# (ha = home.get('abbr', 'Hm'), aa = away.get('abbr', 'Aw')) — not
# themselves valid team codes (lowercase 2nd letter fails the isupper()
# check below) but mean exactly the same thing a real team abbreviation
# would in this position, so they canonicalize the same way.
_FALLBACK_ABBR_TOKENS = {'Hm', 'Aw'}


def _looks_like_team_code(token):
    """Team abbreviations (BOS, TB, SD, ATH...) are short and ALL-CAPS —
    unlike genuine multi-word label prefixes ("Bye week", "Short week",
    "Home Field"), which are regular Capitalized words. A frequency-based
    "appears under 2+ different first tokens" heuristic (first pass of this
    script) wrongly merged "Bye week (home)"/"Short week (home)" into one
    bogus "week (home)" column, and missed genuine team prefixes that only
    ever appeared under a single team in the data (e.g. "DET SP+BP ERA" —
    the one team whose starter fell back to plain ERA instead of SIERA/xFIP).
    This pattern check fixes both: it doesn't need repetition to fire, and
    it can't mistake a real two-word label for a team prefix."""
    if token in _FALLBACK_ABBR_TOKENS:
        return True
    return token.isalpha() and token.isupper() and 2 <= len(token) <= 4


def _canonicalize_labels(rows):
    """Strip team-abbreviation prefixes — see module docstring and
    _looks_like_team_code. Mutates each row's factors list of (label,
    contrib) into canonical form and returns the sorted set of canonical
    labels actually used."""
    for row in rows:
        canon = []
        for label, contrib in row['factors']:
            if ' ' in label:
                first, suffix = label.split(' ', 1)
                if _looks_like_team_code(first):
                    canon.append((suffix, contrib))
                    continue
            canon.append((label, contrib))
        row['factors'] = canon

    return sorted({label for row in rows for label, _ in row['factors']})


def _design_matrix(rows, factor_names):
    n, k = len(rows), len(factor_names)
    X = np.zeros((n, k))
    y = np.zeros(n)
    for i, row in enumerate(rows):
        fdict = dict(row['factors'])
        for j, name in enumerate(factor_names):
            X[i, j] = fdict.get(name, 0.0)
        y[i] = row['margin']
    return X, y


def _fit_ridge(X, y, alpha):
    """Ridge with an unpenalized intercept: center X/y, fit slopes via
    (X^T X + alpha*I)^-1 X^T y on centered data, recover intercept after."""
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - x_mean
    yc = y - y_mean
    k = X.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(k)
    b = Xc.T @ yc
    w = np.linalg.solve(A, b)
    intercept = y_mean - x_mean @ w
    return intercept, w


def _predict(X, intercept, w):
    return intercept + X @ w


def _pick_alpha(X, y, n_folds=5):
    """5-fold CV grid search over ALPHA_GRID, folds in chronological blocks
    (not shuffled — consistent with the whole script's chronological-split
    philosophy: never validate on a fold that's earlier than what it's
    compared against in spirit, though within-train CV order matters less
    than the outer train/test split)."""
    n = len(y)
    fold_bounds = np.linspace(0, n, n_folds + 1).astype(int)
    best_alpha, best_mse = ALPHA_GRID[0], float('inf')
    for alpha in ALPHA_GRID:
        mses = []
        for f in range(n_folds):
            lo, hi = fold_bounds[f], fold_bounds[f + 1]
            if hi <= lo:
                continue
            val_idx = np.arange(lo, hi)
            train_idx = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
            if len(train_idx) < 20:
                continue
            intercept, w = _fit_ridge(X[train_idx], y[train_idx], alpha)
            pred = _predict(X[val_idx], intercept, w)
            mses.append(float(np.mean((pred - y[val_idx]) ** 2)))
        if mses:
            avg_mse = sum(mses) / len(mses)
            if avg_mse < best_mse:
                best_mse, best_alpha = avg_mse, alpha
    return best_alpha


def _ats_accuracy(rows, pred_margins):
    """ATS win rate: does sign(pred_margin - (-line)) match sign(actual_margin - (-line))?
    Only over rows with a real market spread_line. Excludes pushes."""
    correct, total = 0, 0
    for row, pm in zip(rows, pred_margins):
        line = row['spread_line']
        if line is None:
            continue
        cover_target = -line  # home covers when actual_margin > -line
        actual = row['margin']
        if actual == cover_target:
            continue  # push
        pred_pick_home = pm > cover_target
        actual_home_covered = actual > cover_target
        total += 1
        if pred_pick_home == actual_home_covered:
            correct += 1
    return (correct / total, total) if total else (None, 0)


def _spread_proxy_ats_accuracy(sport, rows):
    """Same ATS grading, but picking sides via the OLD spread_proxy.py
    (implied_margin derived from win-prob logit) instead of this new
    from-scratch model — the actual baseline being evaluated against."""
    import spread_proxy
    correct, total = 0, 0
    for row in rows:
        line = row['spread_line']
        if line is None or row['home_prob'] is None:
            continue
        margin = spread_proxy.implied_margin(sport, row['home_prob'])
        if margin is None:
            continue
        cover_target = -line
        actual = row['margin']
        if actual == cover_target:
            continue
        pred_pick_home = margin > cover_target
        actual_home_covered = actual > cover_target
        total += 1
        if pred_pick_home == actual_home_covered:
            correct += 1
    return (correct / total, total) if total else (None, 0)


def fit_sport(GamePrediction, sport):
    rows = _load_rows(GamePrediction, sport)
    if len(rows) < MIN_GAMES_FOR_FIT:
        print(f'{sport}: only {len(rows)} usable games — skipping (need >= {MIN_GAMES_FOR_FIT})')
        return None

    factor_names = _canonicalize_labels(rows)

    split = int(len(rows) * 0.7)
    train, test = rows[:split], rows[split:]

    X_train, y_train = _design_matrix(train, factor_names)
    alpha = _pick_alpha(X_train, y_train)
    intercept_train, w_train = _fit_ridge(X_train, y_train, alpha)

    X_test, y_test = _design_matrix(test, factor_names)
    pred_test = _predict(X_test, intercept_train, w_train)
    resid = y_test - pred_test
    corr = float(np.corrcoef(pred_test, y_test)[0, 1]) if len(test) > 1 else None
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else None
    sigma = float(np.std(resid))
    ats_acc, ats_n = _ats_accuracy(test, pred_test)
    proxy_acc, proxy_n = _spread_proxy_ats_accuracy(sport, test)

    # Production fit: full dataset, same factor set/alpha.
    X_full, y_full = _design_matrix(rows, factor_names)
    intercept_full, w_full = _fit_ridge(X_full, y_full, alpha)
    pred_full = _predict(X_full, intercept_full, w_full)
    sigma_full = float(np.std(y_full - pred_full))

    print(f'\n=== {sport} ===')
    print(f'  n={len(rows)}  train={len(train)}  test={len(test)}  factors={len(factor_names)}  alpha={alpha}')
    print(f'  OOS test: corr={corr:.3f}  R²={r2:.3f}  sigma={sigma:.2f}' if corr is not None else '  OOS test: n/a')
    if ats_n:
        breakeven = 0.524
        verdict = 'BEATS break-even' if (ats_acc or 0) > breakeven else 'below break-even'
        print(f'  OOS ATS accuracy vs real market line: {ats_acc:.3f} ({ats_n} graded, non-push) — {verdict} ({breakeven})')
    else:
        print('  OOS ATS accuracy: no market spread line in test window')
    if proxy_n:
        print(f'  old spread_proxy ATS on same window: {proxy_acc:.3f} ({proxy_n} graded)')

    weights = {name: round(float(c), 4) for name, c in zip(factor_names, w_full)}
    result = {
        'intercept': round(float(intercept_full), 4),
        'weights': weights,
        'sigma': round(sigma_full, 3),
        'alpha': alpha,
        'validation': {
            'n': len(rows), 'test_n': len(test),
            'corr': round(corr, 3) if corr is not None else None,
            'r2': round(r2, 3) if r2 is not None else None,
            'ats_accuracy': round(ats_acc, 3) if ats_acc is not None else None,
            'ats_n': ats_n,
        },
    }
    return result


def main():
    from app import app, GamePrediction
    all_results = {}
    with app.app_context():
        for sport in SPORTS:
            res = fit_sport(GamePrediction, sport)
            if res:
                all_results[sport] = res

    print('\n\n' + '=' * 70)
    print('COEFFS = ' + json.dumps(all_results, indent=4))


if __name__ == '__main__':
    main()
