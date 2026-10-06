#!/usr/bin/env python3
"""
nba_player_bootstrap.py — Offline backtest for nba_player_model.py.

Usage:
    python3 nba_player_bootstrap.py --max-games 50     # quick smoke test
    python3 nba_player_bootstrap.py                    # full 2024+2025 run

For each historical game, reconstructs what nba_player_model.predict() would
have projected using ONLY games strictly before that date for each player's
rolling average (no future leakage) — see project_player()'s docstring for
why this ordering matters.

IMPORTANT, DOCUMENTED LIMITATION: "active roster" here is each team's ACTUAL
boxscore participants for that historical game (players with minutes > 0) —
i.e. this uses the real outcome's participant list, not a pre-game guess.
That's a legitimate backtest methodology (it answers "if the player-level
approach had perfect information about tonight's active roster, how good
would the projection be"), but it means live performance will likely be
somewhat worse than this backtest shows, because a real live run only has
nba_roster_api.get_active_roster()'s best-effort recent-participation guess,
not hindsight. This script's results are an upper bound on live performance,
not a prediction of it.

This also means a new / first-game-of-season player can't be projected (no
prior games exist to average) — those players are silently excluded from
their team's projected score for that game, which will systematically
undercount scoring early in the season for teams relying on new additions.
Documented, not fixed, for this first pass.

No pace adjustment in this backtest (pace_factor left at the nba_player_model
default of 1.0) — adding bball_total_model's per-team pace fetch would
roughly double the number of HTTP calls this script already makes; worth
adding in a follow-up once the no-pace/no-opponent-adjustment baseline's
numbers justify the cost.

Opponent-defense adjustment: each team's own points-allowed history is
tracked chronologically (team_history below) alongside player_history, using
only games strictly before the one being projected — same no-leakage rule.
Added after the first backtest (2,452 games, no opponent adjustment: 59.1%
win accuracy, Brier 0.2597 — worse than the 0.2474 constant-baseline Brier)
pointed at raw player-production sums missing the opponent-quality signal
team win% already captures.
"""
import datetime as _dt
import json
import math
import os
import sys
import time
import requests
from collections import defaultdict

SEASONS = [2024, 2025]
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
ESPN_SUMMARY    = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary'

_DIR = os.path.dirname(os.path.abspath(__file__))
# Disk caches so a killed/interrupted run (e.g. ESPN rate-limiting a long
# sequential fetch) resumes instead of re-fetching ~2,500 games from
# scratch. Same spirit as nba_bootstrap.py's nba_training_{season}.json
# cache, just also caching the much more numerous per-game boxscore calls.
SCHEDULE_CACHE  = os.path.join(_DIR, 'nba_player_schedule_{season}.json')
# v2 adds fga/fta/tov (for usage-weighted projection) that v1 didn't parse —
# new cache file rather than a migration, since the cached rows don't carry
# enough raw data to backfill the new fields without re-fetching anyway.
BOXSCORE_CACHE  = os.path.join(_DIR, 'nba_player_boxscores_v2.json')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nba_player_model as pm


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


# ── Schedule fetch (which games happened, and their ESPN event ids) ───────────

def fetch_season_schedule(season):
    """Completed NBA regular-season games for `season`, with event ids and
    team ids (not just names — needed to key per-player gamelogs)."""
    cache_path = SCHEDULE_CACHE.format(season=season)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    games = []
    start = _dt.date(season, 10, 1)
    end   = _dt.date(season + 1, 4, 20)
    day = start
    seen_ids = set()
    while day <= end:
        date_str = day.strftime('%Y%m%d')
        r = _get(ESPN_SCOREBOARD, {'dates': date_str, 'limit': 100}, date_str)
        if r:
            for event in r.json().get('events', []):
                eid = event.get('id')
                if eid in seen_ids:
                    continue
                comp = event.get('competitions', [{}])[0]
                if comp.get('status', {}).get('type', {}).get('state') != 'post':
                    continue
                if event.get('season', {}).get('type') != 2:
                    continue
                tmap = {c['homeAway']: c for c in comp.get('competitors', [])}
                home_c, away_c = tmap.get('home', {}), tmap.get('away', {})
                try:
                    games.append({
                        'event_id':   eid,
                        'game_date':  event.get('date', '')[:10],
                        'home_id':    home_c['team']['id'],
                        'away_id':    away_c['team']['id'],
                        'home_score': int(home_c.get('score', 0) or 0),
                        'away_score': int(away_c.get('score', 0) or 0),
                    })
                    seen_ids.add(eid)
                except (KeyError, TypeError):
                    continue
        day += _dt.timedelta(days=1)
        time.sleep(0.05)
    games = sorted(games, key=lambda g: g['game_date'])
    with open(cache_path, 'w') as f:
        json.dump(games, f)
    return games


# ── Per-game boxscore fetch (actual participants + their per-game line) ───────

_boxscore_cache = None  # loaded lazily so a plain `import` doesn't touch disk


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


def _attempts(made_attempt_str):
    """'7-14' -> 14.0 (the attempts half of ESPN's made-attempt composite
    fields FG/3PT/FT). Returns 0.0 on anything unparseable."""
    try:
        return float(str(made_attempt_str).split('-')[1])
    except (IndexError, ValueError, TypeError):
        return 0.0


def fetch_boxscore(event_id):
    """Returns {team_id: [{player_id, minutes, points, rebounds, assists,
    fga, fta, tov}]} for both teams in this game, or None on failure.
    fga/fta/tov feed nba_player_model's usage-weighted projection (FGA +
    0.44*FTA + TOV is the standard scoring-possessions-used estimator).
    Cached to disk (BOXSCORE_CACHE) — a completed game's boxscore never
    changes, so this is safe to reuse across runs indefinitely."""
    cache = _load_boxscore_cache()
    if event_id in cache:
        return cache[event_id]

    r = _get(ESPN_SUMMARY, {'event': event_id}, event_id)
    if not r:
        return None
    box = r.json().get('boxscore', {})
    out = {}
    for team_block in box.get('players', []):
        team_id = team_block.get('team', {}).get('id')
        stats_block = team_block.get('statistics', [{}])[0]
        names = stats_block.get('names', [])
        idx = {name: names.index(name) for name in
               ('MIN', 'PTS', 'REB', 'AST', 'FG', 'FT', 'TO') if name in names}
        rows = []
        for ath in stats_block.get('athletes', []):
            pid = (ath.get('athlete') or {}).get('id')
            vals = ath.get('stats', [])
            if not pid or not vals:
                continue
            try:
                minutes = float(vals[idx['MIN']]) if 'MIN' in idx else 0.0
            except (ValueError, TypeError):
                minutes = 0.0
            if minutes <= 0:
                continue
            def _num(key):
                try:
                    return float(vals[idx[key]]) if key in idx else 0.0
                except (ValueError, TypeError):
                    return 0.0
            rows.append({
                'player_id': pid,
                'minutes':   minutes,
                'points':    _num('PTS'),
                'rebounds':  _num('REB'),
                'assists':   _num('AST'),
                'fga':       _attempts(vals[idx['FG']]) if 'FG' in idx else 0.0,
                'fta':       _attempts(vals[idx['FT']]) if 'FT' in idx else 0.0,
                'tov':       _num('TO'),
            })
        if team_id:
            out[team_id] = rows
    result = out or None
    cache[event_id] = result  # cache the None too — don't re-request a dud every run
    return result


# ── Backtest loop ──────────────────────────────────────────────────────────────

def run_backtest(max_games=None):
    all_games = []
    for season in SEASONS:
        print(f'Fetching {season}-{str(season+1)[2:]} schedule...', flush=True)
        games = fetch_season_schedule(season)
        print(f'  -> {len(games)} completed games', flush=True)
        all_games.extend(games)
    all_games.sort(key=lambda g: g['game_date'])
    if max_games:
        all_games = all_games[:max_games]

    # player_id -> chronological list of {date, points, minutes} seen so far
    player_history = defaultdict(list)
    # team_id -> list of points allowed in each game, newest first (mirrors
    # player_history's ordering/no-leakage rule)
    team_def_history = defaultdict(list)

    def _recent_ppg_allowed(team_id, window=10):
        hist = team_def_history.get(team_id, [])
        sample = hist[:window]
        return sum(sample) / len(sample) if sample else None

    results = []
    for i, g in enumerate(all_games):
        box = fetch_boxscore(g['event_id'])
        if not box:
            continue
        home_rows = box.get(g['home_id'], [])
        away_rows = box.get(g['away_id'], [])
        if not home_rows or not away_rows:
            continue

        def _project_side(rows):
            projs = []
            for row in rows:
                hist = player_history[row['player_id']]
                proj = pm.project_player(hist) if hist else None
                if proj:
                    projs.append(proj)
            return projs

        home_proj = _project_side(home_rows)
        away_proj = _project_side(away_rows)
        # home_opp_factor scales the HOME team's score by the AWAY team's
        # recent defense (and vice versa) — see nba_player_model.predict().
        home_opp_factor = pm.opponent_factor(_recent_ppg_allowed(g['away_id']))
        away_opp_factor = pm.opponent_factor(_recent_ppg_allowed(g['home_id']))
        pred = (pm.predict(home_proj, away_proj,
                            home_opp_factor=home_opp_factor,
                            away_opp_factor=away_opp_factor)
                if home_proj and away_proj else None)

        actual_margin = g['home_score'] - g['away_score']
        actual_total  = g['home_score'] + g['away_score']
        if pred:
            results.append({
                'date':          g['game_date'],
                'pred_margin':   pred['margin'],
                'actual_margin': actual_margin,
                'pred_total':    pred['total'],
                'actual_total':  actual_total,
                'pred_home_prob': pred['home_prob'],
                'home_won':      actual_margin > 0,
            })

        # Append this game's lines to history AFTER projecting (no leakage).
        for row in home_rows + away_rows:
            player_history[row['player_id']].append({
                'date':     g['game_date'],
                'points':   row['points'],
                'minutes':  row['minutes'],
                'rebounds': row['rebounds'],
                'assists':  row['assists'],
                'fga':      row['fga'],
                'fta':      row['fta'],
                'tov':      row['tov'],
            })
        # project_player() reads newest-first; keep history in that order.
        for pid in list(player_history.keys()):
            player_history[pid].sort(key=lambda r: r['date'], reverse=True)

        # Points ALLOWED by each team this game = the other side's score.
        team_def_history[g['home_id']].insert(0, g['away_score'])
        team_def_history[g['away_id']].insert(0, g['home_score'])

        if (i + 1) % 100 == 0:
            print(f'  processed {i+1}/{len(all_games)} games, {len(results)} usable', flush=True)
            _save_boxscore_cache()  # so a kill/timeout mid-run doesn't lose fetched work

    _save_boxscore_cache()
    return results


def report(results):
    if not results:
        print('No usable backtest rows — nothing to report.')
        return

    n = len(results)
    margin_errs = [r['pred_margin'] - r['actual_margin'] for r in results]
    total_errs  = [r['pred_total'] - r['actual_total'] for r in results]
    rmse_margin = math.sqrt(sum(e * e for e in margin_errs) / n)
    rmse_total  = math.sqrt(sum(e * e for e in total_errs) / n)
    sigma_fit   = (sum(e * e for e in margin_errs) / n) ** 0.5

    probs = [r['pred_home_prob'] for r in results if r['pred_home_prob'] is not None]
    actuals = [1.0 if r['home_won'] else 0.0 for r in results if r['pred_home_prob'] is not None]
    brier = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(probs) if probs else None
    acc = sum(1 for p, a in zip(probs, actuals) if (p >= 0.5) == (a >= 0.5)) / len(probs) if probs else None
    home_win_rate = sum(actuals) / len(actuals) if actuals else None

    print('\n' + '=' * 64)
    print(f'  Player Model Backtest — {n:,} games')
    print('=' * 64)
    print(f'  Margin RMSE:  {rmse_margin:.2f} pts')
    print(f'  Total RMSE:   {rmse_total:.2f} pts')
    print(f'  Fitted sigma (residual std of margin): {sigma_fit:.2f}')
    if brier is not None:
        print(f'  Brier score:  {brier:.4f}  (baseline {home_win_rate*(1-home_win_rate):.4f} if always picking majority)')
        print(f'  Win accuracy: {acc*100:.1f}%')
    print('=' * 64)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'nba_player_backtest_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'n_games': n,
            'margin_rmse': round(rmse_margin, 3),
            'total_rmse': round(rmse_total, 3),
            'sigma_fit': round(sigma_fit, 3),
            'brier': round(brier, 4) if brier is not None else None,
            'accuracy': round(acc, 4) if acc is not None else None,
        }, f, indent=2)
    print(f'  Full results -> {out_path}')


if __name__ == '__main__':
    max_games = None
    if '--max-games' in sys.argv:
        max_games = int(sys.argv[sys.argv.index('--max-games') + 1])
    results = run_backtest(max_games=max_games)
    report(results)
