#!/usr/bin/env python3
"""
nba_total_backfill.py — Retroactively fills total_proj/total_pick_line/
total_over_prob/total_actual/total_went_over/total_pick_roi on resolved
NBA GamePrediction rows that have a market total_close (from the Odds API
historical backfill) but no model projection — total_proj was never
persisted for NBA before 2026-09-24's totals-tracking rollout, and NBA's
2026-27 season hadn't tipped off yet by then, so there's zero historical
sample to grade from the live path alone.

Unlike backfill_spread_accuracy.py (which only needed home_prob, already
stored on every old row), totals needed real new data collection first:
bball_total_model.py's pace inputs (FGA/OREB/TOV/FTA) were never captured
historically, and ESPN's live team-statistics endpoint has no "as of this
past date" mode — seeing nba_boxscore_pace_backfill.py's docstring for how
those pace components got reconstructed per game from real box scores.

This script replays, as faithfully as possible, what nba_api.py's live
path would have computed on each game's own date:
  - ppg/ppg_allowed: this team's regular-season TeamGameStat rows strictly
    BEFORE the game, blended with the prior season's final numbers below
    EARLY_SEASON_GAMES (mirrors nba_api._apply_prior_season_blend exactly).
  - pace (fga/oreb/tov/fta): averaged over this team's current-season rows
    strictly before the game, UN-blended — the live code never blends pace,
    only ppg, so a team's first few games of a season still produce no
    total_proj here, same real limitation the live site would have had.
  - fed into bball_total_model.predict_total() with the SAME CALIBRATION
    coefficients the live model uses — no separate formula maintained here.

KNOWN APPROXIMATION: the "pick line" graded against is total_close (the
market's CLOSING total), not an actual pre-game line — the odds backfill
only pulled total_close, not total_open, to stay inside the API credit
budget. Same spirit as backfill_spread_accuracy.py's ASSUMED_PRICE: a
real limitation of backfilling from data that wasn't captured live, stated
plainly rather than papered over. Price is likewise assumed -110 each side
(no historical over/under PRICE was ever captured for these rows).

Usage:
    python3 nba_total_backfill.py --dry-run
    python3 nba_total_backfill.py --force   # recompute rows already filled
"""
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ASSUMED_PRICE = -110
EARLY_SEASON_GAMES = 4   # mirrors nba_api.EARLY_SEASON_GAMES


def _app_season(game_date):
    y, m = int(game_date[:4]), int(game_date[5:7])
    return y if m >= 10 else y - 1


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def run(dry_run=False, force=False):
    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, GamePrediction, TeamGameStat, _total_over_prob
    import bball_total_model

    print('─' * 58)
    print(f'  NBA Total Backfill{"  (dry run — no writes)" if dry_run else ""}')
    print('─' * 58)

    totals = {'filled': 0, 'no_team_id': 0, 'no_pace': 0, 'no_ppg': 0, 'model_none': 0}

    with flask_app.app_context():
        filters = [
            GamePrediction.sport == 'NBA',
            GamePrediction.home_won.isnot(None),
            GamePrediction.total_close.isnot(None),
        ]
        if not force:
            filters.append(GamePrediction.total_proj.is_(None))
        rows = GamePrediction.query.filter(*filters).all()
        print(f'\n{len(rows)} resolved row(s) with total_close, missing total_proj.')
        if not rows:
            return

        # team_name -> team_id, built once from whatever TeamGameStat already
        # knows (NBA team ids are stable franchise ids, no mid-dataset
        # renames to worry about).
        name_to_id = {}
        for r in db.session.query(TeamGameStat.team_name, TeamGameStat.team_id).filter(
                TeamGameStat.sport == 'NBA').distinct().all():
            if r[0] and r[1]:
                name_to_id[r[0]] = r[1]

        # Cache of (team_id, season) -> list of (game_date, points, points_allowed,
        # fga, oreb, tov, fta) regular-season rows, sorted by date — avoids
        # re-querying per game for a team that appears in many rows.
        season_rows_cache = {}

        def _season_rows(team_id, season):
            key = (team_id, season)
            if key not in season_rows_cache:
                q = TeamGameStat.query.filter_by(
                    sport='NBA', team_id=team_id, season=season, game_type='regular'
                ).order_by(TeamGameStat.game_date).all()
                season_rows_cache[key] = q
            return season_rows_cache[key]

        def _team_inputs(team_name, season, before_date):
            team_id = name_to_id.get(team_name)
            if not team_id:
                return None, None, None

            cur = [r for r in _season_rows(team_id, season) if r.game_date < before_date]
            cur_games = len(cur)
            cur_ppg = _avg([r.points for r in cur])
            cur_ppga = _avg([r.points_allowed for r in cur])

            eff_ppg, eff_ppga = cur_ppg, cur_ppga
            if cur_games < EARLY_SEASON_GAMES:
                prior = _season_rows(team_id, season - 1)
                prior_ppg = _avg([r.points for r in prior])
                prior_ppga = _avg([r.points_allowed for r in prior])
                w = cur_games / EARLY_SEASON_GAMES
                if prior_ppg is not None:
                    eff_ppg = w * cur_ppg + (1 - w) * prior_ppg if cur_ppg is not None else prior_ppg
                if prior_ppga is not None:
                    eff_ppga = w * cur_ppga + (1 - w) * prior_ppga if cur_ppga is not None else prior_ppga

            pace = {
                'fga':  _avg([r.fga for r in cur]),
                'oreb': _avg([r.oreb for r in cur]),
                'tov':  _avg([r.tov for r in cur]),
                'fta':  _avg([r.fta for r in cur]),
            }
            return team_id, (eff_ppg, eff_ppga), pace

        for pred in rows:
            season = _app_season(pred.game_date)
            home_id, home_ppg, home_pace = _team_inputs(pred.home_team, season, pred.game_date)
            away_id, away_ppg, away_pace = _team_inputs(pred.away_team, season, pred.game_date)

            if not home_id or not away_id:
                totals['no_team_id'] += 1
                continue
            if home_ppg[0] is None or home_ppg[1] is None or away_ppg[0] is None or away_ppg[1] is None:
                totals['no_ppg'] += 1
                continue
            if None in home_pace.values() or None in away_pace.values():
                totals['no_pace'] += 1
                continue

            result = bball_total_model.predict_total(
                'NBA', home_id, home_ppg[0], home_ppg[1], away_id, away_ppg[0], away_ppg[1],
                home_pace_components=home_pace, away_pace_components=away_pace,
            )
            if result is None:
                totals['model_none'] += 1
                continue

            proj = result['total_projection']
            pick_line = pred.total_close
            over_prob = _total_over_prob('NBA', proj, pick_line)

            if dry_run:
                print(f'  ✓ {pred.away_team} @ {pred.home_team} ({pred.game_date}): '
                      f'proj={proj}  close={pick_line}  over_prob={round(over_prob, 3) if over_prob else None}')
                totals['filled'] += 1
                continue

            pred.total_proj = proj
            pred.total_pick_line = pick_line
            pred.total_over_prob = over_prob
            pred.total_over_price = ASSUMED_PRICE
            pred.total_under_price = ASSUMED_PRICE
            pred.total_actual = pred.home_score + pred.away_score
            pred.total_went_over = None
            pred.total_pick_roi = None

            if pred.total_actual == pick_line:
                pred.total_pick_roi = 0.0   # push
            else:
                pred.total_went_over = pred.total_actual > pick_line
                pick_over = (over_prob or 0.5) >= 0.5
                pick_correct = (pick_over == pred.total_went_over)
                profit = ASSUMED_PRICE / 100.0 if ASSUMED_PRICE > 0 else 100.0 / (-ASSUMED_PRICE)
                pred.total_pick_roi = round(profit if pick_correct else -1.0, 4)

            totals['filled'] += 1

        if not dry_run:
            db.session.commit()

    print('\n' + '─' * 58)
    print(f'  Filled: {totals["filled"]}')
    print(f'  Skipped, no team id match: {totals["no_team_id"]}')
    print(f'  Skipped, no ppg data (team has 0 games AND no prior season): {totals["no_ppg"]}')
    print(f'  Skipped, no pace data (team has 0 games this season yet): {totals["no_pace"]}')
    print(f'  Skipped, model returned None: {totals["model_none"]}')
    if dry_run:
        print('\n  Dry run — no writes made.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    run(dry_run=args.dry_run, force=args.force)
