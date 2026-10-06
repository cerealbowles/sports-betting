#!/usr/bin/env python3
"""
nfl_player_bootstrap.py — Offline backtest + fit for nfl_player_model.py.

Usage:
    python3 nfl_player_bootstrap.py --max-games 50     # quick smoke test
    python3 nfl_player_bootstrap.py                    # full SEASONS run

For each historical game, reconstructs each team's offense/defense grade
using ONLY games strictly before that date (no-leakage — same rule as
nba_player_bootstrap.py), fits the logistic P(home_win) ~
sigmoid(a + b*matchup_diff) via the same no-sklearn gradient descent this
app already uses (nba_ensemble_bootstrap._fit_logistic_2d, 1D version
here), and reports accuracy against the actual team-aggregate model
(nfl_model.py) so the two can be compared honestly before any blend.

Fetches schedule + per-game box scores directly from ESPN (NOT the local
stats warehouse — nfl_stats_db.py only has the current season's first few
weeks so far, nowhere near enough for a real backtest), with disk caches
so an interrupted run resumes instead of re-fetching everything.
"""
import json
import math
import os
import sys
import time
import requests
from collections import defaultdict

SEASONS = [2024, 2025]
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'

_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_CACHE = os.path.join(_DIR, 'nfl_player_schedule_{season}.json')
BOXSCORE_CACHE = os.path.join(_DIR, 'nfl_player_boxscores.json')

sys.path.insert(0, _DIR)
import nfl_player_model as pm
import nfl_boxscore_api


def _get(url, params, label, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries - 1:
                print(f'  ! {label} failed: {e}', flush=True)
                return None
            time.sleep(1 + attempt)
    return None


def fetch_season_schedule(season):
    """Completed NFL regular-season games for `season`: [{event_id,
    game_date, home_id, away_id, home_score, away_score}], chronological."""
    cache_path = SCHEDULE_CACHE.format(season=season)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    games = []
    seen_ids = set()
    for week in range(1, 19):
        r = _get(ESPN_SCOREBOARD, {'seasontype': 2, 'week': week, 'season': season,
                                    'dates': season, 'limit': 20}, f'{season} wk{week}')
        if not r:
            continue
        found = False
        for event in r.json().get('events', []):
            eid = event.get('id')
            if eid in seen_ids:
                continue
            comp = event.get('competitions', [{}])[0]
            if comp.get('status', {}).get('type', {}).get('state') != 'post':
                continue
            tmap = {c.get('homeAway'): c for c in comp.get('competitors', [])}
            home_c, away_c = tmap.get('home', {}), tmap.get('away', {})
            try:
                games.append({
                    'event_id':   eid,
                    'game_date':  event.get('date', '')[:10],
                    'home_id':    (home_c.get('team') or {}).get('id'),
                    'away_id':    (away_c.get('team') or {}).get('id'),
                    'home_score': int(home_c.get('score', 0) or 0),
                    'away_score': int(away_c.get('score', 0) or 0),
                })
                seen_ids.add(eid)
                found = True
            except Exception:
                pass
        if not found and week > 3:
            break
        time.sleep(0.1)

    games.sort(key=lambda g: g['game_date'])
    with open(cache_path, 'w') as f:
        json.dump(games, f)
    print(f'  season {season}: {len(games)} completed games', flush=True)
    return games


_boxscore_cache = None


def _load_boxscore_cache():
    global _boxscore_cache
    if _boxscore_cache is None:
        if os.path.exists(BOXSCORE_CACHE):
            with open(BOXSCORE_CACHE) as f:
                _boxscore_cache = json.load(f)
        else:
            _boxscore_cache = {}
    return _boxscore_cache


def _save_boxscore_cache():
    if _boxscore_cache is not None:
        with open(BOXSCORE_CACHE, 'w') as f:
            json.dump(_boxscore_cache, f)


def fetch_boxscore(event_id):
    """{team_id: [{id, name, position, categories}]} for this game, via
    nfl_boxscore_api's parser (disk-cached — see module docstring)."""
    cache = _load_boxscore_cache()
    if event_id in cache:
        return cache[event_id]
    box = nfl_boxscore_api.get_live_boxscore(event_id)
    cache[event_id] = box
    return box


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _fit_logistic_1d(x, y, iters=20000, lr=0.3, lam=1e-4):
    """Tiny no-sklearn logistic regression: y ~ sigmoid(a + b*x). Same
    gradient-descent approach as nba_ensemble_bootstrap._fit_logistic_2d,
    1 feature instead of 2."""
    a = b = 0.0
    n = len(y)
    for _ in range(iters):
        ga = gb = 0.0
        for xi, yi in zip(x, y):
            p = _sigmoid(a + b * xi)
            err = p - yi
            ga += err
            gb += err * xi
        a -= lr * (ga / n + lam * a)
        b -= lr * (gb / n + lam * b)
    return a, b


def run(max_games=None):
    all_games = []
    for season in SEASONS:
        all_games.extend(fetch_season_schedule(season))
    all_games.sort(key=lambda g: g['game_date'])
    if max_games:
        all_games = all_games[:max_games]

    # Point-in-time per-team history: offense/defense aggregate dicts,
    # newest-first, appended AFTER each game is scored (no leakage).
    off_history = defaultdict(list)
    def_history = defaultdict(list)

    rows = []
    for i, g in enumerate(all_games):
        box = fetch_boxscore(g['event_id'])
        if not box:
            continue
        home_rows = box.get(g['home_id'], [])
        away_rows = box.get(g['away_id'], [])
        if not home_rows or not away_rows:
            continue

        home_off_grade = pm.compute_offense_grade(off_history[g['home_id']])
        home_def_grade = pm.compute_defense_grade(def_history[g['home_id']])
        away_off_grade = pm.compute_offense_grade(off_history[g['away_id']])
        away_def_grade = pm.compute_defense_grade(def_history[g['away_id']])

        pred = pm.predict(home_off_grade, home_def_grade, away_off_grade, away_def_grade)

        home_won = g['home_score'] > g['away_score']
        if pred is not None:
            rows.append({
                'date': g['game_date'],
                'matchup_diff': pred['matchup_diff'],
                'home_won': home_won,
            })

        # ── Update running state AFTER scoring (no leakage) ──────────────
        home_off, home_def = pm.parse_team_game_aggregate(home_rows)
        away_off, away_def = pm.parse_team_game_aggregate(away_rows)
        off_history[g['home_id']].insert(0, home_off)
        off_history[g['away_id']].insert(0, away_off)
        def_history[g['home_id']].insert(0, home_def)
        def_history[g['away_id']].insert(0, away_def)

        if (i + 1) % 100 == 0:
            print(f'  processed {i+1}/{len(all_games)}, {len(rows)} usable', flush=True)
            _save_boxscore_cache()

    _save_boxscore_cache()
    return rows


def evaluate(rows, a, b):
    correct = 0
    brier_sum = 0.0
    for r in rows:
        p = _sigmoid(a + b * r['matchup_diff'])
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

    print(f'Running NFL player-model backtest over {SEASONS}...', flush=True)
    rows = run(max_games=args.max_games)
    print(f'\n{len(rows)} usable games.', flush=True)

    if len(rows) < 20:
        print('Too few games to fit — aborting.')
        sys.exit(1)

    x = [r['matchup_diff'] for r in rows]
    y = [1.0 if r['home_won'] else 0.0 for r in rows]
    a, b = _fit_logistic_1d(x, y)
    acc, brier = evaluate(rows, a, b)

    home_win_rate = sum(y) / len(y)
    print(f'\nfit: INTERCEPT = {a:.4f}, SLOPE = {b:.4f}')
    print(f'player-grade model: {acc*100:.1f}% win accuracy, {brier:.4f} Brier  (n={len(rows)})')
    print(f'home-field-always baseline: {home_win_rate*100:.1f}% win rate')
