#!/usr/bin/env python3
"""
wnba_bootstrap.py — Calibrate the WNBA model against past regular seasons.

Usage:
    python3 wnba_bootstrap.py               # fetch all seasons, run refit
    python3 wnba_bootstrap.py --refit-only  # skip fetch, use cached data

Fetches completed WNBA regular-season results from ESPN, rebuilds each team's
point-in-time record / home-road split / PPG / rest / form (using ONLY games
played earlier in that season), scores every game with wnba_model.predict(),
and runs logistic regression to suggest coefficient updates.

Point-in-time matters: the win/loss/"Home"/"Road" records ESPN attaches to a
completed game reflect the state AFTER that game, which would leak the outcome
into the win-percentage factors. This module never reads those fields.

NOTE: a WNBA season is only ~200-330 games (36-44 games x 12-15 teams / 2), so
      coefficients are noisy — treat the refit as advisory and validate on the
      Model Performance page before changing live weights.
"""
import datetime as _dt
import json
import os
import sys
import time
import requests
from collections import defaultdict
from zoneinfo import ZoneInfo

_ET = ZoneInfo('America/New_York')

SEASONS = [2022, 2023, 2024, 2025, 2026]
ESPN_WNBA = 'https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard'
_HERE = os.path.dirname(os.path.abspath(__file__))


def _et_date(iso_utc):
    """ESPN's UTC timestamp → the game's ET calendar date. The app keys
    predictions by ET date (late tip-offs land past midnight UTC), so the
    backfill must match or west-coast games double-insert."""
    dt = _dt.datetime.fromisoformat(iso_utc.replace('Z', '+00:00'))
    return dt.astimezone(_ET).strftime('%Y-%m-%d')


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


# ── Season data fetcher ───────────────────────────────────────────────────────

def fetch_season_games(season):
    """Completed WNBA regular-season games for `season`, paging ESPN's
    scoreboard day-by-day across May 1 - Oct 31 (regular season ends in
    September; postseason/All-Star/preseason are filtered by season type)."""
    games = []
    print(f'  Fetching {season} season...', flush=True)
    day = _dt.date(season, 5, 1)
    end = min(_dt.date(season, 10, 31), _dt.date.today())
    seen_ids = set()
    while day <= end:
        date_str = day.strftime('%Y%m%d')
        r = _get(ESPN_WNBA, {'dates': date_str, 'limit': 100}, date_str)
        if r:
            for event in r.json().get('events', []):
                if event.get('id') in seen_ids:
                    continue
                comp  = event.get('competitions', [{}])[0]
                state = comp.get('status', {}).get('type', {}).get('state', 'pre')
                if state != 'post':
                    continue
                # Regular season only (seasonType 2).
                if event.get('season', {}).get('type') != 2:
                    continue
                tmap   = {c['homeAway']: c for c in comp.get('competitors', [])}
                home_c = tmap.get('home', {})
                away_c = tmap.get('away', {})
                try:
                    h_score = int(home_c.get('score', 0) or 0)
                    a_score = int(away_c.get('score', 0) or 0)
                    h_name  = home_c.get('team', {}).get('displayName', '')
                    a_name  = away_c.get('team', {}).get('displayName', '')
                    if not h_name or not a_name:
                        continue
                    games.append({
                        'season':     season,
                        'game_date':  _et_date(event.get('date', '')),
                        'game_time_utc': event.get('date', ''),
                        'home_name':  h_name,
                        'away_name':  a_name,
                        'home_score': h_score,
                        'away_score': a_score,
                        'home_won':   h_score > a_score,
                    })
                    seen_ids.add(event.get('id'))
                except Exception:
                    continue
        day += _dt.timedelta(days=1)
        time.sleep(0.05)
    print(f'  → {len(games)} total games', flush=True)
    return games


# ── Point-in-time stat builders ───────────────────────────────────────────────

def build_recent_form(games, team_name, before_date, n=5):
    """Last n W/L results for team_name before before_date (newest first)."""
    tg = []
    for g in games:
        if g['game_date'] >= before_date:
            continue
        if g['home_name'] == team_name:
            tg.append((g['game_date'], 'W' if g['home_won'] else 'L'))
        elif g['away_name'] == team_name:
            tg.append((g['game_date'], 'W' if not g['home_won'] else 'L'))
    tg.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in tg[:n]]


def build_rest_days_map(games):
    """{team_name: {game_date: days since previous game, or None for the opener}}."""
    sorted_games = sorted(games, key=lambda g: g['game_date'])
    last_game = {}
    result = defaultdict(dict)
    for g in sorted_games:
        gdate = g['game_date']
        for team in (g['home_name'], g['away_name']):
            if team in last_game:
                last = _dt.datetime.strptime(last_game[team], '%Y-%m-%d').date()
                curr = _dt.datetime.strptime(gdate, '%Y-%m-%d').date()
                result[team][gdate] = (curr - last).days
            else:
                result[team][gdate] = None
            last_game[team] = gdate
    return result


def build_pointintime_stats(season_games):
    """
    Walk one season chronologically and return, for every (team, game_date),
    the win/loss record, home/road split and PPG / PPG-allowed computed ONLY
    from that team's earlier games that season. Accumulators update AFTER each
    game's snapshot is taken, so a game never sees its own result.
    Assumes at most one game per team per date (true for the WNBA).
    """
    sorted_games = sorted(season_games, key=lambda g: g['game_date'])
    wl    = defaultdict(lambda: [0, 0])
    split = defaultdict(lambda: {'home': [0, 0], 'away': [0, 0]})
    pts   = defaultdict(lambda: {'for': 0, 'against': 0, 'g': 0})
    snap  = {}

    for g in sorted_games:
        h, a = g['home_name'], g['away_name']
        for team, side in ((h, 'home'), (a, 'away')):
            w, l = wl[team]
            sp = split[team][side]
            p  = pts[team]
            snap[(team, g['game_date'])] = {
                'wins':        w,
                'losses':      l,
                'split_w':     sp[0],
                'split_l':     sp[1],
                'ppg':         round(p['for'] / p['g'], 2) if p['g'] else None,
                'ppg_allowed': round(p['against'] / p['g'], 2) if p['g'] else None,
            }
        h_won = g['home_won']
        wl[h][0 if h_won else 1] += 1
        wl[a][1 if h_won else 0] += 1
        split[h]['home'][0 if h_won else 1] += 1
        split[a]['away'][1 if h_won else 0] += 1
        pts[h]['for'] += g['home_score']; pts[h]['against'] += g['away_score']; pts[h]['g'] += 1
        pts[a]['for'] += g['away_score']; pts[a]['against'] += g['home_score']; pts[a]['g'] += 1
    return snap


def build_team_dict(snapshot, form, rest_days):
    """snapshot is one (team, game_date) entry from build_pointintime_stats, or
    {} for a team's first game of the season."""
    return {
        'wins':        snapshot.get('wins', 0),
        'losses':      snapshot.get('losses', 0),
        'split_w':     snapshot.get('split_w', 0),
        'split_l':     snapshot.get('split_l', 0),
        'ppg':         snapshot.get('ppg'),
        'ppg_allowed': snapshot.get('ppg_allowed'),
        'form':        form,
        'rest_days':   rest_days,
    }


def score_season(games, import_model):
    """Score every game in one season with point-in-time stats. Returns rows
    carrying both the model output and the actual result."""
    pit = build_pointintime_stats(games)
    rest_map = build_rest_days_map(games)
    rows = []
    for g in games:
        h, a, gd = g['home_name'], g['away_name'], g['game_date']
        home = build_team_dict(pit.get((h, gd), {}), build_recent_form(games, h, gd), rest_map[h].get(gd))
        away = build_team_dict(pit.get((a, gd), {}), build_recent_form(games, a, gd), rest_map[a].get(gd))
        result = import_model.predict(home, away)
        rows.append({
            'season':       g['season'],
            'game_date':    gd,
            'game_time_utc': g.get('game_time_utc', ''),
            'home_team':    h,
            'away_team':    a,
            'home_score':   g['home_score'],
            'away_score':   g['away_score'],
            'home_won':     1 if g['home_won'] else 0,
            'home_prob':    result['home_prob'],
            'away_prob':    result['away_prob'],
            'factors':      result['factors'],
            'factors_json': json.dumps(result['factors']),
        })
    return rows


# ── Logistic regression refit ─────────────────────────────────────────────────

def refit(training_rows):
    import numpy as np

    def brier_score_loss(y_true, y_prob):
        return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))

    def fit_logistic(X, y, C=1.0, iters=50):
        # L2-regularised logistic regression, no intercept, via Newton/IRLS.
        # Pure NumPy so the refit doesn't depend on scikit-learn/SciPy.
        w = np.zeros(X.shape[1])
        reg = np.eye(X.shape[1]) / C
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
            grad = X.T @ (p - y) + reg @ w
            hess = (X.T * (p * (1 - p))) @ X + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        return w

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
    current_acc    = sum(1 for p, a in zip(current_probs, y) if (p >= 0.5) == (a >= 0.5)) / len(y)

    coef = fit_logistic(X, y)
    refit_probs = 1.0 / (1.0 + np.exp(-np.clip(X @ coef, -30, 30)))
    refit_brier = brier_score_loss(y, refit_probs)
    refit_acc   = sum(1 for p, a in zip(refit_probs, y) if (p >= 0.5) == (a >= 0.5)) / len(y)

    season_stats = {}
    for season in seasons_covered:
        mask = [r['season'] == season for r in training_rows]
        y_s  = np.array([a for a, m in zip(y, mask) if m])
        p_s  = [p for p, m in zip(current_probs, mask) if m]
        season_stats[season] = {
            'n':     len(y_s),
            'hwr':   round(float(y_s.mean()), 3),
            'brier': round(float(brier_score_loss(y_s, p_s)), 4),
            'acc':   round(sum(1 for p, a in zip(p_s, y_s) if (p >= 0.5) == (a >= 0.5)) / len(y_s), 3),
        }

    W = 72
    print('\n' + '═' * W)
    print(f'  WNBA Model Refit  —  {" + ".join(str(s) for s in seasons_covered)}  —  {len(training_rows):,} total games')
    print('═' * W)
    for s, st in season_stats.items():
        print(f'  {s}: {st["n"]:,} games  |  home win rate {st["hwr"]*100:.1f}%  '
              f'|  current model {st["brier"]:.4f} Brier ({st["acc"]*100:.1f}% acc)')
    print(f'\n  Combined home win rate: {home_win_rate*100:.1f}%')
    print(f'\n  Brier score  (↓ better, 0.25 = pure random):')
    print(f'    Baseline (always {home_win_rate*100:.0f}%):    {baseline_brier:.4f}')
    print(f'    Current model:             {current_brier:.4f}  ({current_acc*100:.1f}% pick accuracy)')
    print(f'    Refit (in-sample):         {refit_brier:.4f}  ({refit_acc*100:.1f}% pick accuracy)')
    print(f'    Improvement:               {(current_brier - refit_brier)*1000:+.1f} mBrier\n')

    print('─' * W)
    print(f'  {"Factor":<36}  {"Coeff":>7}  Suggested action')
    print('─' * W)
    coefficients = dict(zip(factor_names, coef))
    for name, coef in sorted(coefficients.items(), key=lambda x: abs(x[1] - 1.0), reverse=True):
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
    print('  Validate on the Model Performance page before adjusting live weights.')
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
    with open(os.path.join(_HERE, 'wnba_refit_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('  Full results → wnba_refit_results.json\n')


# ── Entry point ───────────────────────────────────────────────────────────────

def load_or_fetch_season(season):
    cache_path = os.path.join(_HERE, f'wnba_training_{season}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            games = json.load(f)['games']
        print(f'  Loaded {season} from cache: {len(games):,} games '
              f'(delete wnba_training_{season}.json to re-fetch)')
        return games
    games = fetch_season_games(season)
    if games:
        with open(cache_path, 'w') as f:
            json.dump({'season': season, 'games': games}, f)
        print(f'  Cached → wnba_training_{season}.json')
    return games


def main():
    print('═' * 62)
    print(f'  WNBA Model Bootstrap — {" + ".join(str(s) for s in SEASONS)}')
    print('═' * 62)
    sys.path.insert(0, _HERE)
    import wnba_model

    all_rows = []
    for season in SEASONS:
        print(f'\n── Season {season} ──────────────────────────────────────')
        games = load_or_fetch_season(season)
        if not games:
            print(f'  No games found for {season}, skipping.', flush=True)
            continue
        rows = score_season(games, wnba_model)
        print(f'  → {len(rows)} training rows', flush=True)
        all_rows.extend(rows)

    if not all_rows:
        print('No training data found.', flush=True)
        return
    with open(os.path.join(_HERE, 'wnba_training_combined.json'), 'w') as f:
        json.dump(all_rows, f)
    print(f'\n  Combined: {len(all_rows):,} total rows → wnba_training_combined.json')
    refit(all_rows)


if __name__ == '__main__':
    combined_path = os.path.join(_HERE, 'wnba_training_combined.json')
    if '--refit-only' in sys.argv and os.path.exists(combined_path):
        with open(combined_path) as f:
            rows = json.load(f)
        print(f'  {len(rows):,} rows from seasons: {sorted({r["season"] for r in rows})}')
        refit(rows)
    else:
        main()
