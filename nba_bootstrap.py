#!/usr/bin/env python3
"""
nba_bootstrap.py — Calibrate the NBA model against recent regular-season data.

Usage:
    python3 nba_bootstrap.py               # fetch all seasons, run refit
    python3 nba_bootstrap.py --refit-only  # skip fetch, use cached data

Fetches completed NBA regular-season game results from ESPN, computes
season-level team stats and per-game rest/form from scores, scores each
game with nba_model.predict(), and runs logistic regression to suggest
coefficient updates.

NOTE: NBA has ~1,230 games/season (82 games x 30 teams / 2) — much less
      noisy than NFL's ~272, closer to NHL's ~1,300, but still validate
      live predictions on the Model Performance page before adjusting.
"""
import datetime as _dt
import json
import os
import sys
import time
import requests
from collections import defaultdict

SEASONS = [2024, 2025]
ESPN_NBA = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _parse_record(records, name):
    for rec in records:
        if rec.get('name', '').lower() == name.lower():
            return rec.get('summary', '0-0')
    return '0-0'


def _wl(summary):
    try:
        parts = summary.split('-')
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


# ── Season data fetchers ───────────────────────────────────────────────────────

def fetch_season_games(season):
    """Fetch all completed NBA regular-season games for `season` by paging
    ESPN's scoreboard day-by-day across the regular-season window (mid-Oct
    through mid-April of the following year — matches ESPN's own season
    labeling, where `season=2024` means the 2024-25 season)."""
    games = []
    print(f'  Fetching {season}-{str(season + 1)[2:]} season...', flush=True)
    start = _dt.date(season, 10, 1)
    end   = _dt.date(season + 1, 4, 20)
    day = start
    seen_ids = set()
    while day <= end:
        date_str = day.strftime('%Y%m%d')
        r = _get(ESPN_NBA, {'dates': date_str, 'limit': 100}, f'{date_str}')
        if r:
            for event in r.json().get('events', []):
                if event.get('id') in seen_ids:
                    continue
                comp  = event.get('competitions', [{}])[0]
                state = comp.get('status', {}).get('type', {}).get('state', 'pre')
                if state != 'post':
                    continue
                # Regular season only (seasonType 2); skip preseason/playoffs.
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
                    h_recs   = home_c.get('records', [])
                    a_recs   = away_c.get('records', [])
                    h_w, h_l = _wl(_parse_record(h_recs, 'overall'))
                    a_w, a_l = _wl(_parse_record(a_recs, 'overall'))
                    h_sw, h_sl = _wl(_parse_record(h_recs, 'Home'))
                    a_sw, a_sl = _wl(_parse_record(a_recs, 'Road'))
                    games.append({
                        'season':      season,
                        'game_date':   event.get('date', '')[:10],
                        'home_name':   h_name,
                        'away_name':   a_name,
                        'home_score':  h_score,
                        'away_score':  a_score,
                        'home_won':    h_score > a_score,
                        'home_wins':   h_w,   'home_losses': h_l,
                        'away_wins':   a_w,   'away_losses': a_l,
                        'home_split_w': h_sw, 'home_split_l': h_sl,
                        'away_split_w': a_sw, 'away_split_l': a_sl,
                    })
                    seen_ids.add(event.get('id'))
                except Exception:
                    continue
        day += _dt.timedelta(days=1)
        time.sleep(0.05)
    print(f'  → {len(games)} total games', flush=True)
    return games


# ── Stat builders ──────────────────────────────────────────────────────────────

def build_team_season_stats(games):
    """Compute season-aggregate PPG and PPG-allowed for each team."""
    pts = defaultdict(lambda: {'for': 0, 'against': 0, 'g': 0})
    for g in games:
        pts[g['home_name']]['for']     += g['home_score']
        pts[g['home_name']]['against'] += g['away_score']
        pts[g['home_name']]['g']       += 1
        pts[g['away_name']]['for']     += g['away_score']
        pts[g['away_name']]['against'] += g['home_score']
        pts[g['away_name']]['g']       += 1
    return {t: {'ppg':         round(v['for']     / v['g'], 2),
                'ppg_allowed': round(v['against']  / v['g'], 2)}
            for t, v in pts.items() if v['g'] > 0}


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
    """
    For each team and game, compute days since their previous game.
    Returns {team_name: {game_date: rest_days_int_or_None}}.
    """
    sorted_games = sorted(games, key=lambda g: g['game_date'])
    last_game    = {}  # team -> last game date str
    result       = defaultdict(dict)
    for g in sorted_games:
        gdate = g['game_date']
        for team in (g['home_name'], g['away_name']):
            if team in last_game:
                last = _dt.datetime.strptime(last_game[team], '%Y-%m-%d').date()
                curr = _dt.datetime.strptime(gdate, '%Y-%m-%d').date()
                result[team][gdate] = (curr - last).days
            else:
                result[team][gdate] = None   # first game of season
            last_game[team] = gdate
    return result


# ── Training row builder ───────────────────────────────────────────────────────

def build_season_rows(season, all_games, import_model):
    """Score each game in the season using season-aggregate stats + per-game rest/form."""
    season_games = [g for g in all_games if g['season'] == season]
    team_stats   = build_team_season_stats(season_games)
    rest_map     = build_rest_days_map(season_games)

    rows = []
    for g in season_games:
        h = g['home_name']
        a = g['away_name']
        h_ts = team_stats.get(h, {})
        a_ts = team_stats.get(a, {})

        home = {
            'wins':        g['home_wins'],
            'losses':      g['home_losses'],
            'split_w':     g['home_split_w'],
            'split_l':     g['home_split_l'],
            'ppg':         h_ts.get('ppg'),
            'ppg_allowed': h_ts.get('ppg_allowed'),
            'form':        build_recent_form(season_games, h, g['game_date']),
            'rest_days':   rest_map[h].get(g['game_date']),
        }
        away = {
            'wins':        g['away_wins'],
            'losses':      g['away_losses'],
            'split_w':     g['away_split_w'],
            'split_l':     g['away_split_l'],
            'ppg':         a_ts.get('ppg'),
            'ppg_allowed': a_ts.get('ppg_allowed'),
            'form':        build_recent_form(season_games, a, g['game_date']),
            'rest_days':   rest_map[a].get(g['game_date']),
        }

        result = import_model.predict(home, away)
        rows.append({
            'season':       season,
            'game_date':    g['game_date'],
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
    print(f'  NBA Model Refit  —  {seasons_str}  —  {len(training_rows):,} total games')
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

    # Back-to-back is a rarer, noisier signal — bootstrap coeff is valid but
    # should be treated as advisory rather than definitive.
    _ADVISORY = {'Back-to-back (home)', 'Back-to-back (away)'}

    print('─' * W)
    print(f'  {"Factor":<36}  {"Coeff":>7}  Suggested action')
    print('─' * W)
    coefficients = dict(zip(factor_names, lr.coef_[0]))
    for name, coef in sorted(coefficients.items(), key=lambda x: abs(x[1] - 1.0), reverse=True):
        if name in _ADVISORY:
            action = '~  rarer event — advisory only, validate on Model page'
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
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nba_refit_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Full results → nba_refit_results.json')
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    seasons_str = ' + '.join(str(s) for s in SEASONS)
    print('═' * 62)
    print(f'  NBA Model Bootstrap — {seasons_str}')
    print('═' * 62)
    print()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nba_model

    all_rows = []
    for season in SEASONS:
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f'nba_training_{season}.json')
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            games = cached['games']
            print(f'\n── Season {season}-{str(season + 1)[2:]} ──────────────────────────────────────')
            print(f'  Loaded from cache: {len(games):,} games  '
                  f'(delete nba_training_{season}.json to re-fetch)')
        else:
            print(f'\n── Season {season}-{str(season + 1)[2:]} ──────────────────────────────────────')
            games = fetch_season_games(season)
            if not games:
                print(f'  No games found for {season}, skipping.', flush=True)
                continue
            with open(cache_path, 'w') as f:
                json.dump({'season': season, 'games': games}, f)
            print(f'  Cached → nba_training_{season}.json')

        rows = build_season_rows(season, games, nba_model)
        print(f'  → {len(rows)} training rows', flush=True)
        all_rows.extend(rows)

    if not all_rows:
        print('No training data found.', flush=True)
        return

    combined_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'nba_training_combined.json')
    with open(combined_path, 'w') as f:
        json.dump(all_rows, f)
    print(f'\n  Combined: {len(all_rows):,} total rows → nba_training_combined.json')

    refit(all_rows)


if __name__ == '__main__':
    combined_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'nba_training_combined.json')
    if '--refit-only' in sys.argv and os.path.exists(combined_path):
        print('Loading cached combined training data...')
        with open(combined_path) as f:
            rows = json.load(f)
        print(f'  {len(rows):,} rows from seasons: '
              f'{sorted({r["season"] for r in rows})}')
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        refit(rows)
    else:
        main()
