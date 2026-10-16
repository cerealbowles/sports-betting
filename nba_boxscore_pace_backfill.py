#!/usr/bin/env python3
"""
nba_boxscore_pace_backfill.py — Fills two things for already-ingested NBA
regular-season games that predate nba_api.py's live capture of them (see
that module's Final-game ingest block):

  1. TeamGameStat.fga/oreb/tov/fta — needed before nba_total_backfill.py
     can reconstruct a team's point-in-time pace entering any past game
     (bball_total_model.py's possession estimator has nothing to average
     over otherwise).
  2. NbaPlay rows — the full play-by-play, raw (see that class's docstring
     for why raw events are kept rather than only aggregated).

Both come off the SAME ESPN summary?event= response, so this walks each
game ONCE (one HTTP fetch, shared via nba_boxscore_api's own cache) and
fills whichever of the two is still missing — a game already fully
backfilled on a prior run is skipped entirely, a game missing only plays
(e.g. from before this script's play-by-play step existed) only pays for
the pace-recheck, not a second full pass.

Free, uncapped ESPN endpoint — no Odds API credits involved.

Usage:
    python3 nba_boxscore_pace_backfill.py --dry-run
    python3 nba_boxscore_pace_backfill.py --season 2024 2025
"""
import os
import sys
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

SLEEP = 0.15
# The live app writes this same sqlite file continuously (odds polling, bet
# settlement) — a long batch run like this one WILL collide with it. Retry
# with backoff instead of dying on the first collision; failed games are
# simply left missing and picked up by the next run (idempotent either way).
LOCK_RETRY_DELAYS = (2, 5, 10, 20)


def _with_lock_retry(fn, label):
    """Runs fn(), retrying on sqlite 'database is locked' with backoff.
    Returns True on success, False if every retry was also locked (fn's
    own exception propagates for any OTHER error)."""
    import sqlalchemy.exc
    for attempt, delay in enumerate((0,) + LOCK_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            fn()
            return True
        except sqlalchemy.exc.OperationalError as e:
            if 'database is locked' not in str(e).lower():
                raise
            from app import db
            db.session.rollback()
            print(f'    {label}: database locked, retry {attempt + 1}/{len(LOCK_RETRY_DELAYS)}'
                  f'{" — giving up" if attempt == len(LOCK_RETRY_DELAYS) else ""}')
    return False


def run(seasons, dry_run, limit=None):
    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, TeamGameStat, NbaPlay
    import nba_boxscore_api

    print('─' * 58)
    print(f'  NBA Box Score + Play-by-Play Backfill — seasons {seasons}'
          f'{"  (dry run)" if dry_run else ""}')
    print('─' * 58)

    totals = {'games': 0, 'pace_filled': 0, 'plays_filled': 0, 'no_data': 0, 'lock_failed': 0}

    with flask_app.app_context():
        for season in seasons:
            all_game_ids = {row[0] for row in db.session.query(TeamGameStat.game_id).filter(
                TeamGameStat.sport == 'NBA',
                TeamGameStat.season == season,
                TeamGameStat.game_type == 'regular',
            ).distinct().all()}
            has_pace = {row[0] for row in db.session.query(TeamGameStat.game_id).filter(
                TeamGameStat.sport == 'NBA',
                TeamGameStat.season == season,
                TeamGameStat.game_type == 'regular',
                TeamGameStat.fga.isnot(None),
            ).distinct().all()}
            has_plays = {row[0] for row in db.session.query(NbaPlay.game_id).filter(
                NbaPlay.season == season,
            ).distinct().all()}

            game_ids = sorted(g for g in all_game_ids if g not in has_pace or g not in has_plays)
            if not game_ids:
                print(f'\nSeason {season}: nothing missing.')
                continue
            if limit:
                game_ids = game_ids[:limit]
            print(f'\nSeason {season}: {len(game_ids)} game(s) missing pace and/or plays'
                  f'{f" (limited to {limit})" if limit else ""}.')

            for i, game_id in enumerate(game_ids):
                totals['games'] += 1
                needs_pace = game_id not in has_pace
                needs_plays = game_id not in has_plays
                try:
                    summary = nba_boxscore_api.get_live_summary(game_id)
                except Exception as e:
                    print(f'  {game_id}: fetch failed ({e})')
                    continue
                time.sleep(SLEEP)

                if not summary:
                    totals['no_data'] += 1
                    continue

                rows = TeamGameStat.query.filter_by(sport='NBA', game_id=game_id).all()
                game_date_et = rows[0].game_date if rows else None
                game_type = rows[0].game_type if rows else 'regular'

                if needs_pace:
                    pace = nba_boxscore_api.get_team_pace_stats(game_id)
                    for r in rows:
                        p = pace.get(r.team_id)
                        if not p or p.get('fga') is None:
                            continue
                        if dry_run:
                            print(f'  ✓ pace {game_id} {r.team_name}: fga={p["fga"]} '
                                  f'oreb={p["oreb"]} tov={p["tov"]} fta={p["fta"]}')
                            continue
                        r.fga, r.oreb, r.tov, r.fta = p['fga'], p['oreb'], p['tov'], p['fta']
                        totals['pace_filled'] += 1

                if needs_plays and game_date_et:
                    if dry_run:
                        import nba_playbyplay_api
                        plays = nba_playbyplay_api.parse_plays(summary, game_id)
                        print(f'  ✓ plays {game_id}: {len(plays)} play(s) parsed')
                        totals['plays_filled'] += len(plays)
                    else:
                        import nba_stats_db
                        ok = _with_lock_retry(
                            lambda: nba_stats_db.ingest_play_by_play(
                                game_id, game_date_et, season, game_type, summary),
                            f'{game_id} plays')
                        if ok:
                            totals['plays_filled'] += 1
                        else:
                            totals['lock_failed'] += 1

                if not dry_run and needs_pace:
                    ok = _with_lock_retry(db.session.commit, f'{game_id} pace')
                    if not ok:
                        db.session.rollback()
                        totals['lock_failed'] += 1

                if not dry_run and (i + 1) % 50 == 0:
                    print(f'  ...{i + 1}/{len(game_ids)} games processed')

    print('\n' + '─' * 58)
    print(f'  Games processed: {totals["games"]}')
    print(f'  TeamGameStat pace rows filled: {totals["pace_filled"]}')
    print(f'  Games with play-by-play ingested: {totals["plays_filled"]}')
    print(f'  Games with no summary data: {totals["no_data"]}')
    print(f'  Games left for next run (lock contention): {totals["lock_failed"]}')
    if dry_run:
        print('\n  Dry run — no writes made.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--season', nargs='+', type=int, default=[2024, 2025])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()
    run(args.season, args.dry_run, args.limit)
