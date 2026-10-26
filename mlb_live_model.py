#!/usr/bin/env python3
"""
mlb_live_model.py — Fit and backtest an in-game ("live") win-probability
model from mlb_pbp_backfill.py's play-by-play data, and check how it
compares to the win probability MLB's own Stats API already computes for
each play (the `home_win_prob` column `/winProbability` gave us directly —
see that script's docstring).

Scope, and what this deliberately does NOT do yet:
  This fits P(home wins) from in-game STATE alone — inning, outs, score
  diff, baserunners. It does not (yet) blend in mlb_model.py's pregame
  team-strength prediction, because that requires a historical, point-in-
  time pregame prediction for each backfilled game, and this sqlite dev DB's
  game_predictions table only has today's live rows (confirmed: 1 MLB row,
  not a backfilled history — that history lives on prod). So the real "does
  this complement our pregame model" question isn't answerable from local
  data yet; what IS answerable now, and what this backtests, is a narrower
  but still useful question: does a from-scratch state-based model add
  anything over just reusing MLB's own precomputed win probability? If it
  doesn't, there's no reason to maintain our own version of this layer —
  the complement work would skip straight to blending OUR pregame number
  with MLB's existing win-probability column, not training a parallel
  state-only model.

Usage:
    python3 mlb_live_model.py                  # fit + backtest on whatever's in mlb_pbp.db
    python3 mlb_live_model.py --min-games 500   # refuse to run below a sample-size floor
"""
import argparse
import os
import sqlite3

import numpy as np

_DB_DIR = os.path.dirname(os.environ.get('DB_PATH', '')) or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
DB_PATH = os.path.join(_DB_DIR, 'mlb_pbp.db')


def load_plays():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    games = {row['game_pk']: dict(row) for row in conn.execute(
        'SELECT game_pk, home_score, away_score FROM games')}
    plays = [dict(row) for row in conn.execute(
        'SELECT game_pk, at_bat_index, inning, half, outs_after, balls, strikes, '
        'home_score, away_score, on_1b, on_2b, on_3b, home_win_prob '
        'FROM plays ORDER BY game_pk, at_bat_index')]
    conn.close()
    return games, plays


def _progress(inning, half, outs):
    """0..1 fraction of a standard 9-inning game completed, measured in
    half-innings + outs (so a late-inning, 2-out state scores higher than
    the same inning at 0 outs). Extras (inning > 9) clamp at 1.0 — by then
    "how much game is left" isn't well-defined by raw inning count anyway
    (effectively sudden death), and the model has score_diff/outs/runners to
    work with regardless."""
    completed_half_innings = (inning - 1) * 2 + (1 if half == 'bottom' else 0)
    raw = completed_half_innings + (outs or 0) / 3.0
    return min(raw / 18.0, 1.0)


def build_features(games, plays):
    """Returns (X, y, groups, mlb_wp) — groups is game_pk per row, for a
    group-aware (not row-aware) train/test split; mlb_wp is MLB's own
    home_win_prob for the same row, as the comparison baseline."""
    X, y, groups, mlb_wp = [], [], [], []
    for p in plays:
        g = games.get(p['game_pk'])
        if g is None or g['home_score'] is None or g['away_score'] is None:
            continue
        if p['home_win_prob'] is None:
            continue  # winProbability fetch failed/missing for this play; skip rather than impute
        frac = _progress(p['inning'], p['half'], p['outs_after'])
        score_diff = (p['home_score'] or 0) - (p['away_score'] or 0)
        row = [
            frac,
            score_diff,
            score_diff * frac,            # a 1-run lead means more late than early
            p['outs_after'] or 0,
            p['on_1b'] or 0,
            p['on_2b'] or 0,
            p['on_3b'] or 0,
            1 if p['half'] == 'bottom' else 0,
        ]
        X.append(row)
        y.append(1 if g['home_score'] > g['away_score'] else 0)
        groups.append(p['game_pk'])
        mlb_wp.append(p['home_win_prob'] / 100.0)
    return np.array(X), np.array(y), np.array(groups), np.array(mlb_wp)


FEATURE_NAMES = ['game_progress', 'score_diff', 'score_diff_x_progress',
                 'outs', 'on_1b', 'on_2b', 'on_3b', 'is_bottom_half']


def _bucket_report(label, y_true, p_ours, p_mlb, frac):
    from sklearn.metrics import brier_score_loss, log_loss
    print(f'\n  {label} (n={len(y_true)}):')
    print(f'    ours   — brier {brier_score_loss(y_true, p_ours):.4f}  '
          f'logloss {log_loss(y_true, p_ours, labels=[0, 1]):.4f}')
    print(f'    MLB wp — brier {brier_score_loss(y_true, p_mlb):.4f}  '
          f'logloss {log_loss(y_true, p_mlb, labels=[0, 1]):.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-games', type=int, default=200)
    ap.add_argument('--test-frac', type=float, default=0.2)
    args = ap.parse_args()

    games, plays = load_plays()
    print(f'[mlb_live_model] {DB_PATH}: {len(games)} games, {len(plays)} plays loaded', flush=True)
    if len(games) < args.min_games:
        print(f'! only {len(games)} games backfilled, below --min-games {args.min_games} — '
              f'backfill is likely still running; re-run once it has more. Stopping.', flush=True)
        return

    X, y, groups, mlb_wp = build_features(games, plays)
    print(f'{len(X)} usable play-rows across {len(set(groups))} games '
          f'(dropped rows missing winProbability data)', flush=True)

    # Group-aware split: hold out whole GAMES, never split one game's plays
    # across train/test (consecutive plays in the same game are highly
    # correlated — a row-level split would leak and overstate accuracy).
    rng = np.random.default_rng(42)
    game_ids = np.array(sorted(set(groups)))
    rng.shuffle(game_ids)
    n_test_games = max(1, int(len(game_ids) * args.test_frac))
    test_games = set(game_ids[:n_test_games])
    test_mask = np.array([g in test_games for g in groups])
    train_mask = ~test_mask

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000, solver='lbfgs')
    lr.fit(X[train_mask], y[train_mask])

    p_ours_test = lr.predict_proba(X[test_mask])[:, 1]
    p_mlb_test = mlb_wp[test_mask]
    y_test = y[test_mask]
    frac_test = X[test_mask][:, 0]

    print(f'\nTrain: {train_mask.sum()} rows / {len(game_ids) - n_test_games} games.  '
          f'Test (held-out games): {test_mask.sum()} rows / {n_test_games} games.')

    print('\nCoefficients (standardized would be clearer, but these are raw — '
          'sign/relative magnitude still tell the story):')
    for name, coef in zip(FEATURE_NAMES, lr.coef_[0]):
        print(f'    {name:>24s}: {coef:+.4f}')

    _bucket_report('ALL plays', y_test, p_ours_test, p_mlb_test, frac_test)

    early = frac_test < 1 / 3
    mid = (frac_test >= 1 / 3) & (frac_test < 2 / 3)
    late = frac_test >= 2 / 3
    for name, mask in (('Early game (innings 1-3)', early),
                       ('Mid game (innings 4-6)', mid),
                       ('Late game (innings 7+)', late)):
        if mask.sum() > 0:
            _bucket_report(name, y_test[mask], p_ours_test[mask], p_mlb_test[mask], frac_test[mask])

    print('\nConclusion: this is OUR model vs MLB\'s OWN win-probability number, '
          'both state-only — not yet vs mlb_model.py\'s pregame prediction (see '
          'module docstring for why). If "ours" and "MLB wp" come out within '
          'noise of each other (expected, since both are built from the same '
          'kind of base-out-score history), there\'s no case for maintaining a '
          'parallel state-only model — the actual next step is blending '
          'mlb_model.py\'s pregame home_prob with MLB\'s existing win-probability '
          'column for in-progress games, which needs a historical pregame-'
          'prediction dataset this dev DB doesn\'t have (prod does).')


if __name__ == '__main__':
    main()
