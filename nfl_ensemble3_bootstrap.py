#!/usr/bin/env python3
"""
nfl_ensemble3_bootstrap.py — Extends nfl_ensemble_bootstrap.py's 2-signal
blend (nfl_model.py team-aggregate + nfl_player_model.py offense/defense
grades) with nfl_game_simulator.py's drive-by-drive simulated win
probability as a THIRD signal, fit the same way (logistic-regression
stacking on each signal's own logit — see nfl_ensemble_bootstrap.py's
own docstring for why this beats a raw-probability average).

Motivated by a real, measured finding (this app's own session history,
not a hunch): on 431 of this same two-season game set, the simulator's
win probability correlated only 0.511 with the existing 2-signal
ensemble's — genuinely different information, not a redundant copy — and
a NAIVE unweighted 50/50 average of the two already beat both individual
Briers (0.2199 blended vs. 0.2292 ensemble-alone, 0.2414 sim-alone). This
script replaces that naive average with a real fitted 3-way stack.

Why a separate script rather than editing nfl_ensemble_bootstrap.py in
place: that script's own rows (team_prob, player_prob, home_won) don't
carry team ids, which the simulator needs (home/away offense/defense
grades, point-in-time, to compute its own team-conditioning tilt — see
nfl_game_simulator.TEAM_CONDITIONING_K) — cheaper and safer to redo the
walk once with all three signals computed in lockstep than to change the
existing script's return shape.

Reuses nfl_player_bootstrap.py's DB-backed schedule/boxscore functions
(same event set nfl_ensemble_bootstrap.py already uses — no live ESPN
calls) for the team-aggregate/player-grade signals, and nfl_game_
simulator.build_sim_pools()/simulate_game() for the third, run at
`draws` per game (default 150, matching nfl_game_simulator.DEFAULT_
SIM_DRAWS) with that module's own fitted TEAM_CONDITIONING_K. This is
the slow part — budget several minutes per full 2-season run.

A game only gets a row if ALL THREE signals are available (same "only
score what every signal can weigh in on" requirement nfl_ensemble_
bootstrap.py already applies to its two) — a team with no usable offense/
defense grade (new team context, early-season thin history) drops that
game from the fit entirely rather than guessing a third of a signal.

Usage:
    python3 nfl_ensemble3_bootstrap.py
    python3 nfl_ensemble3_bootstrap.py --draws 50   # faster, noisier
    python3 nfl_ensemble3_bootstrap.py --max-games 100
"""
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfl_model
import nfl_player_model as pm
import nfl_player_bootstrap as player_bt
import nfl_game_simulator as nfl_sim

SEASONS = player_bt.SEASONS
DEFAULT_DRAWS = 150


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def run(draws=DEFAULT_DRAWS, max_games=None, rng=None):
    rng = rng or random.Random(7)
    from app import app as flask_app

    with flask_app.app_context():
        pools = nfl_sim.build_sim_pools()

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

            # ── Team-aggregate model (nfl_model.py) ──────────────────────
            home_dict = _team_dict(home_key, g['game_date'], home=True)
            away_dict = _team_dict(away_key, g['game_date'], home=False)
            team_pred = nfl_model.predict(home_dict, away_dict)
            team_prob = team_pred['home_prob'] if team_pred else None

            # ── Player-grade model (nfl_player_model.py) ─────────────────
            home_off_grade = pm.compute_offense_grade(off_history[home_key])
            home_def_grade = pm.compute_defense_grade(def_history[home_key])
            away_off_grade = pm.compute_offense_grade(off_history[away_key])
            away_def_grade = pm.compute_defense_grade(def_history[away_key])
            player_pred = pm.predict(home_off_grade, home_def_grade, away_off_grade, away_def_grade)
            player_prob = player_pred['home_prob'] if player_pred else None

            # ── Game simulator (nfl_game_simulator.py) ───────────────────
            sim_prob = None
            if None not in (home_off_grade, home_def_grade, away_off_grade, away_def_grade):
                home_tilt = home_off_grade - away_def_grade
                away_tilt = away_off_grade - home_def_grade
                home_wins = 0.0
                for _ in range(draws):
                    game = nfl_sim.simulate_game(pools, rng, drives_per_team=10,
                                                  home_tilt=home_tilt, away_tilt=away_tilt)
                    hs = sum(d['points'] for d in game['home']['drives'])
                    as_ = sum(d['points'] for d in game['away']['drives'])
                    if hs > as_:
                        home_wins += 1.0
                    elif hs == as_:
                        home_wins += 0.5
                sim_prob = home_wins / draws

            home_won = g['home_score'] > g['away_score']
            if team_prob is not None and player_prob is not None and sim_prob is not None:
                rows.append({
                    'date': g['game_date'],
                    'team_prob': team_prob,
                    'player_prob': player_prob,
                    'sim_prob': sim_prob,
                    'home_won': home_won,
                })

            # ── Update running state AFTER scoring (no leakage) ──────────
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

            if (i + 1) % 50 == 0:
                print(f'  processed {i+1}/{len(all_games)}, {len(rows)} usable', flush=True)

    return rows


def _fit_logistic_3d(x1, x2, x3, y, iters=20000, lr=0.1, lam=1e-4):
    """Same tiny no-sklearn logistic regression as nfl_ensemble_bootstrap.py,
    extended to a third input."""
    a = b = c = d = 0.0
    n = len(y)
    for _ in range(iters):
        ga = gb = gc = gd = 0.0
        for xi1, xi2, xi3, yi in zip(x1, x2, x3, y):
            p = _sigmoid(a + b * xi1 + c * xi2 + d * xi3)
            err = p - yi
            ga += err
            gb += err * xi1
            gc += err * xi2
            gd += err * xi3
        a -= lr * (ga / n + lam * a)
        b -= lr * (gb / n + lam * b)
        c -= lr * (gc / n + lam * c)
        d -= lr * (gd / n + lam * d)
    return a, b, c, d


def evaluate_prob(rows, prob_key):
    correct = 0
    brier_sum = 0.0
    for r in rows:
        p = r[prob_key]
        correct += int((p > 0.5) == r['home_won'])
        brier_sum += (p - (1.0 if r['home_won'] else 0.0)) ** 2
    n = len(rows)
    return (correct / n if n else 0.0), (brier_sum / n if n else 0.0)


def evaluate_blend3(rows, fit):
    correct = 0
    brier_sum = 0.0
    for r in rows:
        x = (fit['intercept'] + fit['team_weight'] * _logit(r['team_prob'])
             + fit['player_weight'] * _logit(r['player_prob'])
             + fit['sim_weight'] * _logit(r['sim_prob']))
        p = _sigmoid(x)
        correct += int((p > 0.5) == r['home_won'])
        brier_sum += (p - (1.0 if r['home_won'] else 0.0)) ** 2
    n = len(rows)
    return (correct / n if n else 0.0), (brier_sum / n if n else 0.0)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max-games', type=int, default=None)
    p.add_argument('--draws', type=int, default=DEFAULT_DRAWS)
    args = p.parse_args()

    print(f'Running NFL 3-way ensemble backtest over {SEASONS}, draws={args.draws}...', flush=True)
    rows = run(draws=args.draws, max_games=args.max_games)
    print(f'\n{len(rows)} usable games (all three signals present).', flush=True)

    if len(rows) < 20:
        print('Too few games to fit — aborting.')
        sys.exit(1)

    team_acc, team_brier = evaluate_prob(rows, 'team_prob')
    player_acc, player_brier = evaluate_prob(rows, 'player_prob')
    sim_acc, sim_brier = evaluate_prob(rows, 'sim_prob')

    x1 = [_logit(r['team_prob']) for r in rows]
    x2 = [_logit(r['player_prob']) for r in rows]
    x3 = [_logit(r['sim_prob']) for r in rows]
    y = [1.0 if r['home_won'] else 0.0 for r in rows]
    a, b, c, d = _fit_logistic_3d(x1, x2, x3, y)
    fit = {'intercept': round(a, 4), 'team_weight': round(b, 4),
           'player_weight': round(c, 4), 'sim_weight': round(d, 4)}
    blend_acc, blend_brier = evaluate_blend3(rows, fit)

    print(f'\nfit: {fit}')
    print(f'\nteam-aggregate alone:  {team_acc*100:.1f}% win accuracy, {team_brier:.4f} Brier')
    print(f'player-grade alone:    {player_acc*100:.1f}% win accuracy, {player_brier:.4f} Brier')
    print(f'simulator alone:       {sim_acc*100:.1f}% win accuracy, {sim_brier:.4f} Brier')
    print(f'3-way blended:         {blend_acc*100:.1f}% win accuracy, {blend_brier:.4f} Brier')
    print(f'(n={len(rows)})')
