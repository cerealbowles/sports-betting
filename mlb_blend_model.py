#!/usr/bin/env python3
"""
mlb_blend_model.py — The actual test mlb_live_model.py's docstring flagged
as missing: does blending mlb_model.py's PREGAME, team-aware win probability
with MLB's own in-game win-probability number beat either one alone?

Data:
  - instance/mlb_pbp.db `plays`/`games` — from mlb_pbp_backfill.py.
  - instance/mlb_pbp.db `pregame_predictions` — pulled read-only from prod's
    game_predictions table (2026 MLB season only; that's the only season
    prod has pregame history for) via:
        ssh spooky@192.168.1.23 sqlite3 -header -csv \
            /mnt/docker-data/docker-config/sports-betting/data/bets.db \
            "SELECT game_date, home_team, away_team, home_prob, home_won, \
              home_score, away_score FROM game_predictions WHERE sport='MLB' \
              AND home_prob IS NOT NULL AND home_won IS NOT NULL"
    then loaded locally — see conversation for the exact load step. Re-run
    that pull (prod keeps accumulating new settled games) and reload before
    re-running this for a bigger sample.

Because pregame_predictions only has 2026 rows, this only evaluates 2026
games, joined on (game_date, home_team, away_team) — exact-string team-name
match, confirmed to work since both sides use MLB Stats API's own team names.

Three models compared per play, bucketed by game progress:
  1. mlb_wp      — MLB's own win-probability number (state-only, league history)
  2. pregame     — just mlb_model.py's pregame home_prob, held flat/unchanged
                   through the whole game (what you'd see if you ignored the
                   in-game state entirely)
  3. blend       — logistic regression on [pregame_prob, pregame*(1-progress),
                   game_progress, score_diff, score_diff*progress, outs,
                   on_1b, on_2b, on_3b, is_bottom_half] — lets the fit decide
                   how fast the pregame prior's weight should decay as the
                   game state outweighs it.

Usage: python3 mlb_blend_model.py [--min-games 200] [--test-frac 0.2]
"""
import argparse
import os
import sqlite3

import numpy as np

_DB_DIR = os.path.dirname(os.environ.get('DB_PATH', '')) or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
DB_PATH = os.path.join(_DB_DIR, 'mlb_pbp.db')


def _progress(inning, half, outs):
    completed_half_innings = (inning - 1) * 2 + (1 if half == 'bottom' else 0)
    raw = completed_half_innings + (outs or 0) / 3.0
    return min(raw / 18.0, 1.0)


def load_joined():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('''
        SELECT p.game_pk, p.at_bat_index, p.inning, p.half, p.outs_after,
               p.home_score AS play_home_score, p.away_score AS play_away_score,
               p.on_1b, p.on_2b, p.on_3b, p.home_win_prob,
               g.home_score AS final_home_score, g.away_score AS final_away_score,
               pr.home_prob AS pregame_home_prob
        FROM plays p
        JOIN games g ON g.game_pk = p.game_pk
        JOIN pregame_predictions pr
          ON pr.game_date = g.game_date
         AND pr.home_team = g.home_team
         AND pr.away_team = g.away_team
        ORDER BY p.game_pk, p.at_bat_index
    ''').fetchall()
    conn.close()
    return [dict(r) for r in rows]


FEATURE_NAMES = ['pregame_prob', 'pregame_x_remaining', 'game_progress',
                  'score_diff', 'score_diff_x_progress', 'outs',
                  'on_1b', 'on_2b', 'on_3b', 'is_bottom_half']


def build_features(rows):
    X, y, groups, mlb_wp, pregame_flat = [], [], [], [], []
    for r in rows:
        if r['home_win_prob'] is None or r['pregame_home_prob'] is None:
            continue
        frac = _progress(r['inning'], r['half'], r['outs_after'])
        score_diff = (r['play_home_score'] or 0) - (r['play_away_score'] or 0)
        pregame = r['pregame_home_prob']
        X.append([
            pregame, pregame * (1 - frac), frac,
            score_diff, score_diff * frac, r['outs_after'] or 0,
            r['on_1b'] or 0, r['on_2b'] or 0, r['on_3b'] or 0,
            1 if r['half'] == 'bottom' else 0,
        ])
        y.append(1 if r['final_home_score'] > r['final_away_score'] else 0)
        groups.append(r['game_pk'])
        mlb_wp.append(r['home_win_prob'] / 100.0)
        pregame_flat.append(pregame)
    return (np.array(X), np.array(y), np.array(groups),
             np.array(mlb_wp), np.array(pregame_flat))


def _report(label, y_true, preds_by_name):
    from sklearn.metrics import brier_score_loss, log_loss
    print(f'\n  {label} (n={len(y_true)}):')
    for name, p in preds_by_name.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        print(f'    {name:>10s} — brier {brier_score_loss(y_true, p):.4f}  '
              f'logloss {log_loss(y_true, p, labels=[0, 1]):.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-games', type=int, default=200)
    ap.add_argument('--test-frac', type=float, default=0.2)
    args = ap.parse_args()

    rows = load_joined()
    n_games = len(set(r['game_pk'] for r in rows))
    print(f'[mlb_blend_model] {DB_PATH}: {len(rows)} joined play-rows across '
          f'{n_games} games (2026 only — pregame_predictions has no other season)', flush=True)
    if n_games < args.min_games:
        print(f'! only {n_games} joined games, below --min-games {args.min_games} — '
              f'2026 backfill is likely still running. Re-run once it has more. Stopping.', flush=True)
        return

    X, y, groups, mlb_wp, pregame_flat = build_features(rows)

    rng = np.random.default_rng(42)
    game_ids = np.array(sorted(set(groups)))
    rng.shuffle(game_ids)
    n_test = max(1, int(len(game_ids) * args.test_frac))
    test_games = set(game_ids[:n_test])
    test_mask = np.array([g in test_games for g in groups])
    train_mask = ~test_mask

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=3000, solver='lbfgs')
    lr.fit(X[train_mask], y[train_mask])

    print(f'\nTrain: {train_mask.sum()} rows / {len(game_ids) - n_test} games.  '
          f'Test (held-out games): {test_mask.sum()} rows / {n_test} games.')
    print('\nBlend coefficients:')
    for name, coef in zip(FEATURE_NAMES, lr.coef_[0]):
        print(f'    {name:>24s}: {coef:+.4f}')

    p_blend = lr.predict_proba(X[test_mask])[:, 1]
    preds = {
        'mlb_wp': mlb_wp[test_mask],
        'pregame': pregame_flat[test_mask],
        'blend': p_blend,
    }
    y_test = y[test_mask]
    frac_test = X[test_mask][:, 2]

    _report('ALL plays', y_test, preds)
    for name, mask in (
        ('Early game (innings 1-3)', frac_test < 1 / 3),
        ('Mid game (innings 4-6)', (frac_test >= 1 / 3) & (frac_test < 2 / 3)),
        ('Late game (innings 7+)', frac_test >= 2 / 3),
    ):
        if mask.sum() > 0:
            _report(name, y_test[mask], {k: v[mask] for k, v in preds.items()})

    print('\nRead this as: does "blend" beat "mlb_wp" (does knowing the pregame '
          'team-strength prior help beyond MLB\'s in-game state table), and '
          'does "blend" beat flat "pregame" (does updating on game state beat '
          'just leaving the pregame number unchanged all game)? If blend wins '
          'both, especially early, that is the case for actually wiring this '
          'into a live-game page. If blend only matches mlb_wp, the pregame '
          'prior was already priced into mlb_wp through correlated team '
          'quality signals and isn\'t adding independent information.')


if __name__ == '__main__':
    main()
