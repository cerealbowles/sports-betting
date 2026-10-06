#!/usr/bin/env python3
"""
nfl_ensemble_bootstrap.py — Combines nfl_model.py (team-aggregate) and
nfl_player_model.py (offense/defense unit grades) into one blended signal,
and checks whether the blend beats either alone. Mirrors
nba_ensemble_bootstrap.py — same stacking-via-logistic-regression idea,
NFL side.

Reuses nfl_player_bootstrap.py's cached schedule/boxscore data (same event
set) and, in the SAME chronological pass, reconstructs each team's
season-to-date aggregate state (wins/losses, home/road split, PPG/PPG-
allowed, L3 form, rest days) to feed nfl_model.py — same reasoning as
nba_ensemble_bootstrap.py's inline team_state tracking: keeps both models
scored on exactly the same games in exactly the same order.

Note: nfl_model.py's season-aggregate state resets every season, same as
nba_model/nfl_bootstrap's own convention — team_state below is keyed by
(season, team_id), not just team_id, so a team's 2024 record doesn't bleed
into its 2025 week-1 projection.

Usage:
    python3 nfl_ensemble_bootstrap.py              # full backtest + fit
"""
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfl_model
import nfl_player_model as pm
import nfl_player_bootstrap as player_bt

SEASONS = player_bt.SEASONS


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def run(max_games=None):
    all_games = []
    for season in SEASONS:
        games = player_bt.fetch_season_schedule(season)
        for g in games:
            g = dict(g)
            g['season'] = season
            all_games.append(g)
    all_games.sort(key=lambda g: g['game_date'])
    if max_games:
        all_games = all_games[:max_games]

    off_history = defaultdict(list)
    def_history = defaultdict(list)

    team_state = defaultdict(lambda: {
        'wins': 0, 'losses': 0,
        'home_w': 0, 'home_l': 0, 'road_w': 0, 'road_l': 0,
        'pts_for_sum': 0.0, 'pts_against_sum': 0.0, 'g': 0,
        'form': [],
        'last_date': None,
    })

    def _rest_days(key, date_str):
        st = team_state[key]
        if not st['last_date']:
            return None
        import datetime as _dt
        d0 = _dt.date.fromisoformat(st['last_date'])
        d1 = _dt.date.fromisoformat(date_str)
        return (d1 - d0).days

    def _team_dict(key, date_str, home):
        st = team_state[key]
        ppg = st['pts_for_sum'] / st['g'] if st['g'] else None
        ppg_allowed = st['pts_against_sum'] / st['g'] if st['g'] else None
        split_w, split_l = (st['home_w'], st['home_l']) if home else (st['road_w'], st['road_l'])
        return {
            'wins': st['wins'], 'losses': st['losses'],
            'split_w': split_w, 'split_l': split_l,
            'ppg': ppg, 'ppg_allowed': ppg_allowed,
            'form': st['form'][:3],
            'rest_days': _rest_days(key, date_str),
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

        season = g['season']
        home_key, away_key = (season, g['home_id']), (season, g['away_id'])

        # ── Team-aggregate model (nfl_model.py) ──────────────────────────
        home_dict = _team_dict(home_key, g['game_date'], home=True)
        away_dict = _team_dict(away_key, g['game_date'], home=False)
        team_pred = nfl_model.predict(home_dict, away_dict)
        team_prob = team_pred['home_prob']

        # ── Player-grade model (nfl_player_model.py) ─────────────────────
        home_off_grade = pm.compute_offense_grade(off_history[home_key])
        home_def_grade = pm.compute_defense_grade(def_history[home_key])
        away_off_grade = pm.compute_offense_grade(off_history[away_key])
        away_def_grade = pm.compute_defense_grade(def_history[away_key])
        player_pred = pm.predict(home_off_grade, home_def_grade, away_off_grade, away_def_grade)

        home_won = g['home_score'] > g['away_score']
        if team_prob is not None and player_pred is not None:
            rows.append({
                'date':         g['game_date'],
                'team_prob':    team_prob,
                'player_prob':  player_pred['home_prob'],
                'home_won':     home_won,
            })

        # ── Update running state AFTER scoring (no leakage) ──────────────
        home_off, home_def = pm.parse_team_game_aggregate(home_rows)
        away_off, away_def = pm.parse_team_game_aggregate(away_rows)
        off_history[home_key].insert(0, home_off)
        off_history[away_key].insert(0, away_off)
        def_history[home_key].insert(0, home_def)
        def_history[away_key].insert(0, away_def)

        hs, as_ = team_state[home_key], team_state[away_key]
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

        if (i + 1) % 100 == 0:
            print(f'  processed {i+1}/{len(all_games)}, {len(rows)} usable', flush=True)
            player_bt._save_boxscore_cache()

    player_bt._save_boxscore_cache()
    return rows


def _fit_logistic_2d(x1, x2, y, iters=20000, lr=0.1, lam=1e-4):
    """Same tiny no-sklearn logistic regression as nba_ensemble_bootstrap.py."""
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


def evaluate_prob(rows, prob_key):
    correct = 0
    brier_sum = 0.0
    for r in rows:
        p = r[prob_key]
        pred_home = p > 0.5
        correct += int(pred_home == r['home_won'])
        brier_sum += (p - (1.0 if r['home_won'] else 0.0)) ** 2
    n = len(rows)
    return (correct / n if n else 0.0), (brier_sum / n if n else 0.0)


def evaluate_blend(rows, fit):
    correct = 0
    brier_sum = 0.0
    for r in rows:
        x = fit['intercept'] + fit['team_weight'] * _logit(r['team_prob']) + fit['player_weight'] * _logit(r['player_prob'])
        p = _sigmoid(x)
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

    print(f'Running NFL ensemble backtest over {SEASONS}...', flush=True)
    rows = run(max_games=args.max_games)
    print(f'\n{len(rows)} usable games.', flush=True)

    if len(rows) < 20:
        print('Too few games to fit — aborting.')
        sys.exit(1)

    team_acc, team_brier = evaluate_prob(rows, 'team_prob')
    player_acc, player_brier = evaluate_prob(rows, 'player_prob')

    x1 = [_logit(r['team_prob']) for r in rows]
    x2 = [_logit(r['player_prob']) for r in rows]
    y  = [1.0 if r['home_won'] else 0.0 for r in rows]
    a, b, c = _fit_logistic_2d(x1, x2, y)
    fit = {'intercept': round(a, 4), 'team_weight': round(b, 4), 'player_weight': round(c, 4)}
    blend_acc, blend_brier = evaluate_blend(rows, fit)

    print(f'\nfit: {fit}')
    print(f'\nteam-aggregate alone:  {team_acc*100:.1f}% win accuracy, {team_brier:.4f} Brier')
    print(f'player-grade alone:    {player_acc*100:.1f}% win accuracy, {player_brier:.4f} Brier')
    print(f'blended:               {blend_acc*100:.1f}% win accuracy, {blend_brier:.4f} Brier')
    print(f'(n={len(rows)})')
