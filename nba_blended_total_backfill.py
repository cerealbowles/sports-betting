#!/usr/bin/env python3
"""
nba_blended_total_backfill.py — Fills blended_total_proj/blended_total_
over_prob/blended_total_went_over/blended_total_pick_roi on resolved NBA
GamePrediction rows that already have total_proj (from nba_total_backfill.
py) but no blended value — the player-level signal this blends in
(nba_ensemble_model.TOTAL_FIT) was never backtested at scale until now;
see that module's TOTAL_FIT comment for the point-in-time numbers that
justified swapping it into the live app (R² 0.068 team-only pace-adjusted
-> 0.095 blended, on this exact sample).

Reconstructs each team's active roster for a PAST game the same way
nba_ensemble_bootstrap.py's backtest does: whoever's PlayerGameStat row
for that exact game_id has did_not_play=False IS the active roster — the
real box score stands in for a historical injury/lineup feed that doesn't
exist. Each active player's own point-in-time gamelog (last 10 games,
strictly before this one) feeds nba_player_model.project_player(), same
as the live path.

Usage:
    python3 nba_blended_total_backfill.py --dry-run
    python3 nba_blended_total_backfill.py --force   # recompute already-filled rows
"""
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ASSUMED_PRICE = -110


def _app_season(game_date):
    y, m = int(game_date[:4]), int(game_date[5:7])
    return y if m >= 10 else y - 1


def run(dry_run=False, force=False):
    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, GamePrediction, TeamGameStat, PlayerGameStat, _total_over_prob
    import nba_player_model as pm
    import nba_ensemble_model as em

    print('─' * 58)
    print(f'  NBA Blended Total Backfill{"  (dry run)" if dry_run else ""}')
    print('─' * 58)

    totals = {'filled': 0, 'no_team_id': 0, 'no_game_id': 0, 'no_player_proj': 0, 'no_team_total': 0}

    with flask_app.app_context():
        name_to_id = {}
        for name, tid in db.session.query(TeamGameStat.team_name, TeamGameStat.team_id).filter(
                TeamGameStat.sport == 'NBA').distinct().all():
            if name and tid:
                name_to_id[name] = tid

        team_rows_cache = {}
        def team_rows(team_id, season):
            key = (team_id, season)
            if key not in team_rows_cache:
                team_rows_cache[key] = TeamGameStat.query.filter_by(
                    sport='NBA', team_id=team_id, season=season, game_type='regular'
                ).order_by(TeamGameStat.game_date).all()
            return team_rows_cache[key]

        player_hist_cache = {}
        def player_history_before(player_id, before_date):
            if player_id not in player_hist_cache:
                player_hist_cache[player_id] = PlayerGameStat.query.filter_by(
                    sport='NBA', player_id=player_id, did_not_play=False
                ).order_by(PlayerGameStat.game_date.desc()).all()
            return [
                {'date': r.game_date, 'points': r.points, 'minutes': r.minutes,
                 'fga': r.fga, 'fta': r.fta, 'tov': r.turnovers}
                for r in player_hist_cache[player_id] if r.game_date < before_date
            ]

        def recent_ppg_allowed(team_id, season, before_date, window=10):
            rows = [r for r in team_rows(team_id, season) if r.game_date < before_date]
            sample = rows[-window:]
            vals = [r.points_allowed for r in sample if r.points_allowed is not None]
            return (sum(vals) / len(vals)) if vals else None

        def find_game_id(home_team, away_team, game_date):
            r = TeamGameStat.query.filter_by(
                sport='NBA', team_name=home_team, opponent_name=away_team,
                game_date=game_date, is_home=True).first()
            return r.game_id if r else None

        def active_projections(game_id, team_id, before_date):
            players = PlayerGameStat.query.filter_by(
                sport='NBA', game_id=game_id, team_id=team_id, did_not_play=False).all()
            out = []
            for p in players:
                hist = player_history_before(p.player_id, before_date)
                proj = pm.project_player(hist) if hist else None
                if proj:
                    out.append(proj)
            return out

        filters = [
            GamePrediction.sport == 'NBA',
            GamePrediction.total_proj.isnot(None),
        ]
        if not force:
            filters.append(GamePrediction.blended_total_proj.is_(None))
        rows = GamePrediction.query.filter(*filters).all()
        print(f'\n{len(rows)} row(s) with total_proj, missing blended_total_proj.')

        for pred in rows:
            season = _app_season(pred.game_date)
            home_id, away_id = name_to_id.get(pred.home_team), name_to_id.get(pred.away_team)
            if not home_id or not away_id:
                totals['no_team_id'] += 1
                continue

            game_id = find_game_id(pred.home_team, pred.away_team, pred.game_date)
            if not game_id:
                totals['no_game_id'] += 1
                continue

            home_active = active_projections(game_id, home_id, pred.game_date)
            away_active = active_projections(game_id, away_id, pred.game_date)
            if not home_active or not away_active:
                totals['no_player_proj'] += 1
                continue

            home_opp_factor = pm.opponent_factor(recent_ppg_allowed(away_id, season, pred.game_date))
            away_opp_factor = pm.opponent_factor(recent_ppg_allowed(home_id, season, pred.game_date))
            player_pred = pm.predict(home_active, away_active,
                                      home_opp_factor=home_opp_factor, away_opp_factor=away_opp_factor)
            if not player_pred:
                totals['no_player_proj'] += 1
                continue

            blended = em.blend_total(pred.total_proj, player_pred['total'])
            if blended is None:
                totals['no_team_total'] += 1
                continue

            over_prob = _total_over_prob('NBA', blended, pred.total_pick_line)

            if dry_run:
                print(f'  ✓ {pred.away_team} @ {pred.home_team} ({pred.game_date}): '
                      f'blended={round(blended,1)}  (team-only={pred.total_proj})')
                totals['filled'] += 1
                continue

            pred.blended_total_proj = round(blended, 1)
            pred.blended_total_over_prob = over_prob
            pred.blended_total_went_over = None
            pred.blended_total_pick_roi = None

            if pred.total_actual is not None and pred.total_pick_line is not None:
                if pred.total_actual == pred.total_pick_line:
                    pred.blended_total_pick_roi = 0.0
                else:
                    pred.blended_total_went_over = pred.total_actual > pred.total_pick_line
                    pick_over = (over_prob or 0.5) >= 0.5
                    pick_correct = (pick_over == pred.blended_total_went_over)
                    profit = ASSUMED_PRICE / 100.0 if ASSUMED_PRICE > 0 else 100.0 / (-ASSUMED_PRICE)
                    pred.blended_total_pick_roi = round(profit if pick_correct else -1.0, 4)

            totals['filled'] += 1

        if not dry_run:
            db.session.commit()

    print('\n' + '─' * 58)
    print(f'  Filled: {totals["filled"]}')
    print(f'  Skipped, no team id: {totals["no_team_id"]}')
    print(f'  Skipped, no game id match: {totals["no_game_id"]}')
    print(f'  Skipped, no player projection (box score gap): {totals["no_player_proj"]}')
    print(f'  Skipped, blend returned None: {totals["no_team_total"]}')
    if dry_run:
        print('\n  Dry run — no writes made.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    run(dry_run=args.dry_run, force=args.force)
