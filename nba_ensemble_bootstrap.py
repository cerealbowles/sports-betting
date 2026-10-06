#!/usr/bin/env python3
"""
nba_ensemble_bootstrap.py — Combines nba_model.py (team-aggregate, 73.0%
backtested win accuracy) and nba_player_model.py (player-level, 64.2%) into
one blended signal, and checks whether the blend beats either alone.

Reuses nba_player_bootstrap.py's schedule/boxscore functions (now reading
nba_stats_db.py's local warehouse — see that module's docstring; same event
set, same no-leakage per-player history) and, in the SAME chronological
pass, reconstructs each team's season-to-date aggregate state (wins/losses,
home/road split, PPG/PPG-allowed, L5 form, rest days) to feed nba_model.py —
mirroring nba_bootstrap.py's build_season_rows, but inline here so the two
models are scored on exactly the same games in exactly the same order
(avoids the alignment headache of merging two independently-fetched
datasets by date/team-name).

Why a blend might beat either alone even though the player model trails
badly on its own: the two signals are built from different information
(season win/loss record vs. recent individual-player production) and their
errors don't have to be correlated. Stacking (fitting blend weights via
logistic regression on each model's own probability, rather than just
averaging) lets the data decide how much to trust each signal rather than
assuming a 50/50 split.

Usage:
    python3 nba_ensemble_bootstrap.py              # full backtest + fit
"""
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nba_model
import nba_player_model as pm
import nba_player_bootstrap as player_bt
import spread_proxy

SEASONS = player_bt.SEASONS


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _recent_form(results, n=5):
    return results[:n]  # newest-first 'W'/'L' list; nba_model just counts W's


def run(max_games=None):
    all_games = []
    for season in SEASONS:
        games = player_bt.fetch_season_schedule(season)
        all_games.extend(games)
    all_games.sort(key=lambda g: g['game_date'])
    if max_games:
        all_games = all_games[:max_games]

    player_history = defaultdict(list)     # same as nba_player_bootstrap
    team_def_history = defaultdict(list)   # points allowed, newest first

    # Season-aggregate state per team, mirroring nba_bootstrap.py's inputs.
    team_state = defaultdict(lambda: {
        'wins': 0, 'losses': 0,
        'home_w': 0, 'home_l': 0, 'road_w': 0, 'road_l': 0,
        'pts_for_sum': 0.0, 'pts_against_sum': 0.0, 'g': 0,
        'form': [],            # newest-first 'W'/'L'
        'last_date': None,
    })

    def _rest_days(team_id, date_str):
        st = team_state[team_id]
        if not st['last_date']:
            return None
        import datetime as _dt
        d0 = _dt.date.fromisoformat(st['last_date'])
        d1 = _dt.date.fromisoformat(date_str)
        return (d1 - d0).days

    def _recent_ppg_allowed(team_id, window=10):
        hist = team_def_history.get(team_id, [])
        sample = hist[:window]
        return sum(sample) / len(sample) if sample else None

    def _team_dict(team_id, date_str, home):
        st = team_state[team_id]
        ppg = st['pts_for_sum'] / st['g'] if st['g'] else None
        ppg_allowed = st['pts_against_sum'] / st['g'] if st['g'] else None
        split_w, split_l = (st['home_w'], st['home_l']) if home else (st['road_w'], st['road_l'])
        return {
            'wins': st['wins'], 'losses': st['losses'],
            'split_w': split_w, 'split_l': split_l,
            'ppg': ppg, 'ppg_allowed': ppg_allowed,
            'form': _recent_form(st['form']),
            'rest_days': _rest_days(team_id, date_str),
        }

    rows = []
    for i, g in enumerate(all_games):
        box = player_bt.fetch_boxscore(g['event_id'])
        if not box:
            continue
        home_rows = box.get(g['home_id'], [])
        away_rows = box.get(g['away_id'], [])
        if not home_rows or not away_rows:
            continue

        # ── Team-aggregate model (nba_model.py) ──────────────────────────
        home_dict = _team_dict(g['home_id'], g['game_date'], home=True)
        away_dict = _team_dict(g['away_id'], g['game_date'], home=False)
        team_pred = nba_model.predict(home_dict, away_dict)
        team_prob = team_pred['home_prob']
        team_margin = spread_proxy.implied_margin('NBA', team_prob)

        # ── Player-level model (nba_player_model.py) ─────────────────────
        def _project_side(side_rows):
            out = []
            for row in side_rows:
                hist = player_history[row['player_id']]
                proj = pm.project_player(hist) if hist else None
                if proj:
                    out.append(proj)
            return out

        home_proj = _project_side(home_rows)
        away_proj = _project_side(away_rows)
        home_opp_factor = pm.opponent_factor(_recent_ppg_allowed(g['away_id']))
        away_opp_factor = pm.opponent_factor(_recent_ppg_allowed(g['home_id']))
        player_pred = (pm.predict(home_proj, away_proj,
                                   home_opp_factor=home_opp_factor,
                                   away_opp_factor=away_opp_factor)
                       if home_proj and away_proj else None)

        actual_margin = g['home_score'] - g['away_score']
        actual_total  = g['home_score'] + g['away_score']

        if team_prob is not None and player_pred and player_pred['home_prob'] is not None \
                and team_margin is not None:
            rows.append({
                'date':           g['game_date'],
                'team_prob':      team_prob,
                'player_prob':    player_pred['home_prob'],
                'team_margin':    team_margin,
                'player_margin':  player_pred['margin'],
                'player_total':   player_pred['total'],
                'actual_margin':  actual_margin,
                'actual_total':   actual_total,
                'home_won':       actual_margin > 0,
            })

        # ── Update running state AFTER scoring (no leakage) ──────────────
        for row in home_rows + away_rows:
            player_history[row['player_id']].append({
                'date': g['game_date'], 'points': row['points'], 'minutes': row['minutes'],
                'rebounds': row['rebounds'], 'assists': row['assists'],
                'fga': row['fga'], 'fta': row['fta'], 'tov': row['tov'],
            })
        for pid in list(player_history.keys()):
            player_history[pid].sort(key=lambda r: r['date'], reverse=True)

        team_def_history[g['home_id']].insert(0, g['away_score'])
        team_def_history[g['away_id']].insert(0, g['home_score'])

        home_won = actual_margin > 0
        hs, as_ = team_state[g['home_id']], team_state[g['away_id']]
        hs['wins'] += 1 if home_won else 0
        hs['losses'] += 0 if home_won else 1
        as_['wins'] += 0 if home_won else 1
        as_['losses'] += 1 if home_won else 0
        hs['home_w'] += 1 if home_won else 0
        hs['home_l'] += 0 if home_won else 1
        as_['road_w'] += 0 if home_won else 1
        as_['road_l'] += 1 if home_won else 0
        hs['pts_for_sum'] += g['home_score']; hs['pts_against_sum'] += g['away_score']; hs['g'] += 1
        as_['pts_for_sum'] += g['away_score']; as_['pts_against_sum'] += g['home_score']; as_['g'] += 1
        hs['form'].insert(0, 'W' if home_won else 'L')
        as_['form'].insert(0, 'L' if home_won else 'W')
        hs['last_date'] = g['game_date']
        as_['last_date'] = g['game_date']

        if (i + 1) % 200 == 0:
            print(f'  processed {i+1}/{len(all_games)}, {len(rows)} usable', flush=True)

    return rows


def _fit_logistic_2d(x1, x2, y, iters=20000, lr=0.1, lam=1e-4):
    """Tiny no-sklearn logistic regression: y ~ sigmoid(a + b*x1 + c*x2).
    Same gradient-descent approach used earlier in this session when
    sklearn's numpy ABI was broken in this environment."""
    a = b = c = 0.0
    n = len(y)
    for _ in range(iters):
        ga = gb = gc = 0.0
        for xi1, xi2, yi in zip(x1, x2, y):
            p = _sigmoid(a + b * xi1 + c * xi2)
            err = p - yi
            ga += err
            gb += err * xi1
            gc += err * xi2
        a -= lr * (ga / n + lam * a)
        b -= lr * (gb / n + lam * b)
        c -= lr * (gc / n + lam * c)
    return a, b, c


def _fit_linear_2d(x1, x2, y):
    """OLS closed-form fit of y ~ a + b*x1 + c*x2 via normal equations
    (no numpy/sklearn dependency — same spirit as _fit_logistic_2d)."""
    n = len(y)
    sx1 = sum(x1); sx2 = sum(x2); sy = sum(y)
    sx1x1 = sum(v * v for v in x1); sx2x2 = sum(v * v for v in x2)
    sx1x2 = sum(v1 * v2 for v1, v2 in zip(x1, x2))
    sx1y = sum(v * yi for v, yi in zip(x1, y))
    sx2y = sum(v * yi for v, yi in zip(x2, y))
    # Solve the 3x3 normal-equations system [n,sx1,sx2; sx1,sx1x1,sx1x2; sx2,sx1x2,sx2x2] . [a,b,c] = [sy,sx1y,sx2y]
    A = [[n, sx1, sx2], [sx1, sx1x1, sx1x2], [sx2, sx1x2, sx2x2]]
    bvec = [sy, sx1y, sx2y]
    # Gaussian elimination
    for col in range(3):
        pivot = A[col][col]
        if abs(pivot) < 1e-12:
            pivot = 1e-12
        for row in range(3):
            if row == col:
                continue
            factor = A[row][col] / pivot
            for k in range(3):
                A[row][k] -= factor * A[col][k]
            bvec[row] -= factor * bvec[col]
    a, b, c = (bvec[i] / A[i][i] for i in range(3))
    return a, b, c


def _brier(probs, actuals):
    return sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(probs)


def _acc(probs, actuals):
    return sum(1 for p, a in zip(probs, actuals) if (p >= 0.5) == (a >= 0.5)) / len(probs)


def _rmse(preds, actuals):
    return (sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(preds)) ** 0.5


def report(rows):
    if not rows:
        print('No usable rows.')
        return
    n = len(rows)
    y = [1.0 if r['home_won'] else 0.0 for r in rows]
    team_p = [r['team_prob'] for r in rows]
    player_p = [r['player_prob'] for r in rows]

    print(f'\n{n:,} games with both signals available\n')
    print(f"  Team-aggregate alone:  acc {_acc(team_p, y)*100:.1f}%  Brier {_brier(team_p, y):.4f}")
    print(f"  Player-level alone:    acc {_acc(player_p, y)*100:.1f}%  Brier {_brier(player_p, y):.4f}")

    # Stacked blend: fit on logits of each model's own probability.
    x1 = [_logit(p) for p in team_p]
    x2 = [_logit(p) for p in player_p]
    a, b, c = _fit_logistic_2d(x1, x2, y)
    blend_p = [_sigmoid(a + b * xi1 + c * xi2) for xi1, xi2 in zip(x1, x2)]
    print(f"  Blended (stacked):     acc {_acc(blend_p, y)*100:.1f}%  Brier {_brier(blend_p, y):.4f}")
    print(f"    fit: intercept={a:.3f}  team_weight={b:.3f}  player_weight={c:.3f}")

    # Margin blend (feeds spread) via OLS.
    team_m = [r['team_margin'] for r in rows]
    player_m = [r['player_margin'] for r in rows]
    actual_m = [r['actual_margin'] for r in rows]
    print(f"\n  Team-margin-only RMSE:   {_rmse(team_m, actual_m):.2f}")
    print(f"  Player-margin-only RMSE: {_rmse(player_m, actual_m):.2f}")
    ma, mb, mc = _fit_linear_2d(team_m, player_m, actual_m)
    blend_m = [ma + mb * tm + mc * pm_ for tm, pm_ in zip(team_m, player_m)]
    print(f"  Blended margin RMSE:     {_rmse(blend_m, actual_m):.2f}")
    print(f"    fit: intercept={ma:.3f}  team_weight={mb:.3f}  player_weight={mc:.3f}")

    out = {
        'n_games': n,
        'team_only':   {'acc': round(_acc(team_p, y), 4), 'brier': round(_brier(team_p, y), 4)},
        'player_only': {'acc': round(_acc(player_p, y), 4), 'brier': round(_brier(player_p, y), 4)},
        'blended':     {'acc': round(_acc(blend_p, y), 4), 'brier': round(_brier(blend_p, y), 4),
                         'intercept': round(a, 4), 'team_weight': round(b, 4), 'player_weight': round(c, 4)},
        'margin_blend': {'intercept': round(ma, 4), 'team_weight': round(mb, 4), 'player_weight': round(mc, 4),
                          'rmse_team_only': round(_rmse(team_m, actual_m), 3),
                          'rmse_player_only': round(_rmse(player_m, actual_m), 3),
                          'rmse_blended': round(_rmse(blend_m, actual_m), 3)},
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nba_ensemble_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nFull results -> {out_path}')


if __name__ == '__main__':
    max_games = None
    if '--max-games' in sys.argv:
        max_games = int(sys.argv[sys.argv.index('--max-games') + 1])
    rows = run(max_games=max_games)
    report(rows)
