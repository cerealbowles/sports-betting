#!/usr/bin/env python3
"""
cbb_bootstrap.py — Calibrates cbb_model.py against past seasons, walk-forward
(every prediction uses only games before its date — no leakage).

Two questions:
  1. Margin → win probability. cbb_model turns the rating's predicted home
     margin into a probability with sigmoid(margin / MARGIN_SCALE); this fits
     MARGIN_SCALE by maximum likelihood on season 2025 and grades it on 2026.
  2. Do the NBA/WNBA model's factors add anything on top of the rating?
     Win%, home/road split record, raw PPG / PPG allowed, last-5 form and
     back-to-back are rebuilt point-in-time and added to the logit one at a
     time (fit on 2025, graded on 2026 by log loss and Brier). Only a factor
     that improves the held-out season is worth a live weight.

Usage:
    python3 cbb_bootstrap.py            # uses instance/cbb_rating_games_*.json (fetches if missing)

Writes cbb_refit_results.json (gitignored) with every number printed.
"""
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

import cbb_rating as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, 'cbb_refit_results.json')
FACTORS = ('win_pct', 'split_pct', 'ppg', 'ppg_allowed', 'form_l5', 'b2b')


def walk(season, all_games):
    """[(game, rating_prediction, feature_dict)] for every D1-vs-D1 game in
    `season`, ratings refit daily on prior games only."""
    pool = [g for s in (season - 2, season - 1, season) for g in all_games.get(s, [])]
    target = all_games[season]
    # Point-in-time NBA-style team state, accumulated as the season plays out.
    rec = defaultdict(lambda: {'w': 0, 'l': 0, 'hw': 0, 'hl': 0, 'rw': 0, 'rl': 0,
                               'pf': 0, 'pa': 0, 'res': [], 'last': None})
    out = []
    by_date = defaultdict(list)
    for g in target:
        by_date[g['date']].append(g)
    for d in sorted(by_date):
        f = R.fit([g for g in pool if g['date'] < d], season)
        for g in by_date[d]:
            if R.OTHER_CONF in (g['home_conf'], g['away_conf']):
                continue
            p = R.predict(f, g['home'], g['away'], g['home_conf'], g['away_conf'], g['neutral'])
            out.append((g, p, _features(rec[g['home']], rec[g['away']], d, g['neutral'])))
        for g in by_date[d]:   # update state only after the whole day is predicted
            hw = g['home_score'] > g['away_score']
            for team, won, pf, pa, home in ((g['home'], hw, g['home_score'], g['away_score'], True),
                                            (g['away'], not hw, g['away_score'], g['home_score'], False)):
                r = rec[team]
                r['w' if won else 'l'] += 1
                if not g['neutral']:
                    key = ('h' if home else 'r') + ('w' if won else 'l')
                    r[key] += 1
                r['pf'] += pf
                r['pa'] += pa
                r['res'].append(won)
                r['last'] = d
    return out


def _pct(w, l):
    return w / (w + l) if w + l else 0.5


def _features(h, a, d, neutral):
    def ppg(r):
        n = r['w'] + r['l']
        return (r['pf'] / n, r['pa'] / n) if n else (None, None)

    def rest(r):
        if not r['last']:
            return None
        return (datetime.strptime(d, '%Y-%m-%d') - datetime.strptime(r['last'], '%Y-%m-%d')).days

    def form(r):
        last = r['res'][-5:]
        return sum(last) / len(last) if last else 0.5

    hp, ha = ppg(h)
    ap, aa = ppg(a)
    hr, ar = rest(h), rest(a)
    b2b = 0.0
    if hr is not None and ar is not None:
        b2b = (1.0 if ar <= 1 else 0.0) - (1.0 if hr <= 1 else 0.0)
    return {
        'win_pct': _pct(h['w'], h['l']) - _pct(a['w'], a['l']),
        # neutral site: nobody's home/road record applies
        'split_pct': 0.0 if neutral else _pct(h['hw'], h['hl']) - _pct(a['rw'], a['rl']),
        'ppg': (hp - ap) if hp is not None and ap is not None else 0.0,
        'ppg_allowed': (aa - ha) if ha is not None and aa is not None else 0.0,
        'form_l5': form(h) - form(a),
        'b2b': b2b,
    }


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_logit(X, y, l2=1e-3, iters=50):
    """Newton-Raphson logistic regression, no intercept (home edge already
    lives in the rating margin)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (p - y) + l2 * beta
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _design(rows, factors):
    m = np.array([p['margin'] for _, p, _ in rows])
    cols = [m] + [np.array([f[k] for _, _, f in rows]) for k in factors]
    y = np.array([1.0 if g['home_score'] > g['away_score'] else 0.0 for g, _, _ in rows])
    return np.column_stack(cols), y


def evaluate(train, test):
    results = {}
    X, y = _design(train, ())
    b = _fit_logit(X, y)
    scale = 1 / b[0]
    Xt, yt = _design(test, ())
    pt = 1 / (1 + np.exp(-(Xt @ b)))
    base = {'margin_scale': round(scale, 3), 'logloss': round(_logloss(pt, yt), 5),
            'brier': round(float(np.mean((pt - yt) ** 2)), 5),
            'accuracy': round(float(np.mean((pt > 0.5) == (yt == 1))), 4)}
    results['rating_only'] = base
    print(f"rating only: scale={scale:.2f}  test logloss={base['logloss']}  brier={base['brier']}  acc={base['accuracy']}")
    for k in FACTORS:
        X, y = _design(train, (k,))
        b = _fit_logit(X, y)
        Xt, yt = _design(test, (k,))
        pt = 1 / (1 + np.exp(-(Xt @ b)))
        ll = _logloss(pt, yt)
        results[k] = {'coef': round(float(b[1]), 4), 'margin_scale': round(1 / b[0], 3),
                      'logloss': round(ll, 5), 'delta_vs_rating': round(ll - base['logloss'], 5),
                      'brier': round(float(np.mean((pt - yt) ** 2)), 5)}
        print(f"  + {k:12s} coef={b[1]:+.4f}  test logloss={ll:.5f}  Δ={ll - base['logloss']:+.5f}")
    X, y = _design(train, FACTORS)
    b = _fit_logit(X, y)
    Xt, yt = _design(test, FACTORS)
    pt = 1 / (1 + np.exp(-(Xt @ b)))
    ll = _logloss(pt, yt)
    results['all_factors'] = {'coefs': dict(zip(FACTORS, [round(float(v), 4) for v in b[1:]])),
                              'logloss': round(ll, 5), 'delta_vs_rating': round(ll - base['logloss'], 5)}
    print(f"  + all six     test logloss={ll:.5f}  Δ={ll - base['logloss']:+.5f}")
    return results


def calibration_table(test, scale):
    bins = defaultdict(lambda: [0, 0.0, 0])
    for g, p, _ in test:
        ph = 1 / (1 + math.exp(-p['margin'] / scale))
        fav_p = max(ph, 1 - ph)
        fav_won = (g['home_score'] > g['away_score']) == (ph >= 0.5)
        b = min(int(fav_p * 20) / 20, 0.95)
        bins[b][0] += 1
        bins[b][1] += fav_p
        bins[b][2] += fav_won
    rows = []
    for b in sorted(bins):
        n, sp, w = bins[b]
        rows.append({'bin': f'{b:.2f}', 'n': n, 'predicted': round(sp / n, 3), 'actual': round(w / n, 3)})
        print(f"  fav {b:.2f}-{b + 0.05:.2f}  n={n:5d}  predicted {sp / n:.3f}  actual {w / n:.3f}")
    return rows


def main():
    all_games = {s: R.load_season(s) for s in (2023, 2024, 2025, 2026)}
    print('games per season:', {s: len(v) for s, v in all_games.items()})
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    train = walk(2025, all_games)
    test = walk(2026, all_games)
    print(f'walk-forward: {len(train)} train games (2025), {len(test)} test games (2026)')
    results = {'n_train': len(train), 'n_test': len(test), 'eval': evaluate(train, test)}
    print('calibration on 2026 (favorite win% by predicted bucket):')
    results['calibration_2026'] = calibration_table(test, results['eval']['rating_only']['margin_scale'])
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', OUT_PATH)


if __name__ == '__main__':
    sys.exit(main())
