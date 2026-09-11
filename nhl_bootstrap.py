#!/usr/bin/env python3
"""
nhl_bootstrap.py — Calibrate the NHL model against recent completed seasons.

Usage:
    python3 nhl_bootstrap.py               # fetch all seasons, run refit
    python3 nhl_bootstrap.py --refit-only  # skip fetch, use cached data

Reuses nhl_backfill.py's fetch/point-in-time-stat builders (same season
data, same no-leaked-outcome rule), scores each game with
nhl_model.predict(), and runs logistic regression to suggest coefficient
updates. Same advisory-only pattern as nfl_bootstrap.py/cfb_bootstrap.py —
writes nhl_refit_results.json but never auto-applies it.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from nhl_backfill import (
    DEFAULT_SEASONS, fetch_season_games, build_pointintime_stats,
    build_recent_form, build_team_dict,
)
import nhl_model

SEASONS = DEFAULT_SEASONS


def build_season_rows(season, games):
    pit_snap = build_pointintime_stats(games)
    rows = []
    for g in games:
        h, a, gd = g['home_name'], g['away_name'], g['game_date']
        home = build_team_dict(pit_snap.get((h, gd), {}), build_recent_form(games, h, gd))
        away = build_team_dict(pit_snap.get((a, gd), {}), build_recent_form(games, a, gd))
        result = nhl_model.predict(home, away)
        rows.append({
            'season':       season,
            'game_date':    gd,
            'home_team':    h,
            'away_team':    a,
            'home_won':     1 if g['home_won'] else 0,
            'home_prob':    result['home_prob'],
            'factors_json': json.dumps(result['factors']),
        })
    return rows


# ── Logistic regression refit ──────────────────────────────────────────────────

def refit(training_rows):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    seasons_covered = sorted({r['season'] for r in training_rows})

    factor_names = sorted({label
                            for row in training_rows
                            for label, _ in json.loads(row['factors_json'])})

    X, y = [], []
    for row in training_rows:
        fd = {f: 0.0 for f in factor_names}
        for label, contrib in json.loads(row['factors_json']):
            fd[label] = float(contrib)
        X.append([fd[f] for f in factor_names])
        y.append(float(row['home_won']))
    X = np.array(X)
    y = np.array(y)

    home_win_rate  = y.mean()
    baseline_brier = brier_score_loss(y, [home_win_rate] * len(y))
    current_probs  = [r['home_prob'] for r in training_rows]
    current_brier  = brier_score_loss(y, current_probs)
    current_acc    = sum(1 for p, a in zip(current_probs, y)
                         if (p >= 0.5) == (a >= 0.5)) / len(y)

    lr = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000, solver='lbfgs')
    lr.fit(X, y)

    refit_probs = lr.predict_proba(X)[:, 1]
    refit_brier = brier_score_loss(y, refit_probs)
    refit_acc   = sum(1 for p, a in zip(refit_probs, y)
                      if (p >= 0.5) == (a >= 0.5)) / len(y)

    season_stats = {}
    for season in seasons_covered:
        mask = [r['season'] == season for r in training_rows]
        y_s  = np.array([a for a, m in zip(y,  mask) if m])
        p_s  = [p for p, m in zip(current_probs, mask) if m]
        season_stats[season] = {
            'n':    len(y_s),
            'hwr':  round(float(y_s.mean()), 3),
            'brier': round(float(brier_score_loss(y_s, p_s)), 4),
            'acc':  round(sum(1 for p, a in zip(p_s, y_s)
                              if (p >= 0.5) == (a >= 0.5)) / len(y_s), 3),
        }

    W = 72
    seasons_str = ' + '.join(str(s) for s in seasons_covered)
    print('\n' + '═' * W)
    print(f'  NHL Model Refit  —  {seasons_str}  —  {len(training_rows):,} total games')
    print('═' * W)
    for s, st in season_stats.items():
        print(f'  {s}: {st["n"]:,} games  |  home win rate {st["hwr"]*100:.1f}%  '
              f'|  current model {st["brier"]:.4f} Brier ({st["acc"]*100:.1f}% acc)')
    print(f'\n  Combined home win rate: {home_win_rate*100:.1f}%')
    print()
    print(f'  Brier score  (↓ better, 0.25 = pure random):')
    print(f'    Baseline (always {home_win_rate*100:.0f}%):    {baseline_brier:.4f}')
    print(f'    Current model:             {current_brier:.4f}  ({current_acc*100:.1f}% pick accuracy)')
    print(f'    Refit (in-sample):         {refit_brier:.4f}  ({refit_acc*100:.1f}% pick accuracy)')
    print(f'    Improvement:               {(current_brier - refit_brier)*1000:+.1f} mBrier')
    print()

    # Back-to-back is a rarer event than every-game factors — bootstrap coeff
    # is valid but noisier; treat as advisory rather than definitive.
    _ADVISORY = {'Back-to-back (home)', 'Back-to-back (away)'}

    print('─' * W)
    print(f'  {"Factor":<36}  {"Coeff":>7}  Suggested action')
    print('─' * W)
    coefficients = dict(zip(factor_names, lr.coef_[0]))
    for name, coef in sorted(coefficients.items(), key=lambda x: abs(x[1] - 1.0), reverse=True):
        if name in _ADVISORY:
            action = '~  rare event — advisory only, validate on Model page'
        else:
            delta = coef - 1.0
            if abs(delta) > 0.35:
                action = '★  INCREASE weight' if delta > 0 else '★  REDUCE weight'
            elif abs(delta) > 0.15:
                action = '↑  increase weight' if delta > 0 else '↓  reduce weight'
            elif abs(delta) > 0.05:
                action = '~  slight adjust'
            else:
                action = '✓  well-calibrated'
        print(f'  {name:<36}  {coef:>+7.3f}  {action}')
    print('─' * W)
    print()
    print('  NOTE: in-sample refit is optimistic. Validate on the Model Performance page.')
    print('═' * W)

    out = {
        'seasons':          seasons_covered,
        'n_games':          len(training_rows),
        'home_win_rate':    round(float(home_win_rate), 4),
        'brier_baseline':   round(float(baseline_brier), 4),
        'brier_current':    round(float(current_brier), 4),
        'brier_refit':      round(float(refit_brier), 4),
        'accuracy_current': round(float(current_acc), 4),
        'accuracy_refit':   round(float(refit_acc), 4),
        'per_season':       season_stats,
        'coefficients':     {k: round(float(v), 4) for k, v in coefficients.items()},
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nhl_refit_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Full results → nhl_refit_results.json')
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    seasons_str = ' + '.join(str(s) for s in SEASONS)
    print('═' * 62)
    print(f'  NHL Model Bootstrap — {seasons_str}')
    print('═' * 62)
    print()

    all_rows = []
    for season in SEASONS:
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f'nhl_training_{season}.json')
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            games = cached['games']
            print(f'\n── Season {season} ──────────────────────────────────────')
            print(f'  Loaded from cache: {len(games):,} games  '
                  f'(delete nhl_training_{season}.json to re-fetch)')
        else:
            print(f'\n── Season {season} ──────────────────────────────────────')
            games = fetch_season_games(season)
            if not games:
                print(f'  No games found for {season}, skipping.', flush=True)
                continue
            with open(cache_path, 'w') as f:
                json.dump({'season': season, 'games': games}, f)
            print(f'  Cached → nhl_training_{season}.json')

        rows = build_season_rows(season, games)
        print(f'  → {len(rows)} training rows', flush=True)
        all_rows.extend(rows)

    if not all_rows:
        print('No training data found.', flush=True)
        return

    combined_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'nhl_training_combined.json')
    with open(combined_path, 'w') as f:
        json.dump(all_rows, f)
    print(f'\n  Combined: {len(all_rows):,} total rows → nhl_training_combined.json')

    refit(all_rows)


if __name__ == '__main__':
    combined_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'nhl_training_combined.json')
    if '--refit-only' in sys.argv and os.path.exists(combined_path):
        print('Loading cached combined training data...')
        with open(combined_path) as f:
            rows = json.load(f)
        print(f'  {len(rows):,} rows from seasons: '
              f'{sorted({r["season"] for r in rows})}')
        refit(rows)
    else:
        main()
