#!/usr/bin/env python3
"""
wnba_player_bootstrap.py — Offline backtest for wnba_player_model.py.

Usage:
    python3 wnba_player_bootstrap.py --max-games 50     # quick smoke test
    python3 wnba_player_bootstrap.py                    # full 2024+2025 run

For each historical game, reconstructs what wnba_player_model.predict() would
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
wnba_roster_api.get_active_roster()'s best-effort recent-participation guess,
not hindsight. This script's results are an upper bound on live performance,
not a prediction of it.

This also means a new / first-game-of-season player can't be projected (no
prior games exist to average) — those players are silently excluded from
their team's projected score for that game, which will systematically
undercount scoring early in the season for teams relying on new additions.
Documented, not fixed, for this first pass.

No pace adjustment in this backtest (pace_factor left at the wnba_player_model
default of 1.0) — adding bball_total_model's per-team pace fetch would mean
yet another data source to reconcile; worth adding in a follow-up once the
no-pace/no-opponent-adjustment baseline's numbers justify the cost.

Opponent-defense adjustment: each team's own points-allowed history is
tracked chronologically (team_history below) alongside player_history, using
only games strictly before the one being projected — same no-leakage rule.
Included from the start here, mirroring nba_player_bootstrap.py's own
history — that script's first backtest WITHOUT this adjustment scored
worse than its constant-baseline Brier (raw player-production sums alone
don't carry the opponent-quality signal team win% already captures); this
run doesn't repeat that ablation, it just starts from the lesson learned.

Schedule/boxscore source: reads wnba_stats_db.py's local warehouse (fetch_
season_schedule/fetch_boxscore below), not a live ESPN fetch — SEASONS below
is exactly what wnba_stats_backfill.py has already loaded there (see that
script's module docstring). This used to be its own live fetch with a disk
cache (hitting ESPN directly, ~2,500 HTTP round trips for a 2-season run);
now that the warehouse covers the same seasons for the live app's own
predictions anyway, re-fetching independently was redundant — one source of
truth, and a full backtest run is now a DB-only operation (seconds, not the
original run's long sequential-fetch wall time).
"""
import json
import math
import os
import sys
from collections import defaultdict

SEASONS = [2024, 2025]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wnba_player_model as pm
import wnba_stats_db


def fetch_season_schedule(season):
    """Completed WNBA regular-season games for `season`, with event ids and
    team ids (not just names — needed to key per-player gamelogs). Thin
    wrapper over wnba_stats_db.get_schedule_db() — kept as its own function
    (rather than inlining that call at each use site) so wnba_ensemble_
    bootstrap.py's existing `player_bt.fetch_season_schedule(season)` calls
    keep working unchanged."""
    return wnba_stats_db.get_schedule_db(season, game_type='regular')


def fetch_boxscore(event_id):
    """Returns {team_id: [{player_id, minutes, points, rebounds, assists,
    fga, fta, tov}]} for both teams in this game, or None if nothing's
    stored for it. Thin wrapper over wnba_stats_db.get_game_boxscore_db() —
    see fetch_season_schedule()'s docstring for why this stays a function
    rather than inlining."""
    return wnba_stats_db.get_game_boxscore_db(event_id) or None


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
        # recent defense (and vice versa) — see wnba_player_model.predict().
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
                             'wnba_player_backtest_results.json')
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
