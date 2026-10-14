#!/usr/bin/env python3
"""
nfl_playbyplay_backfill.py — Targeted re-ingest for games already in the
warehouse: re-fetches each stored game's ESPN summary and re-runs
nfl_stats_db.ingest_play_by_play() against it, so games ingested before
this parser fix (see nfl_playbyplay_api.py's _SPECIAL_TEAMS_AND_TURNOVER_
TYPES / NflPlay's updated class docstring in app.py) pick up the punt/
kickoff/field-goal/interception/fumble-recovery rows that were previously
silently dropped, and get a correct is_turnover on the rows that need it.

Deliberately NOT nfl_stats_backfill.py's full scoreboard walk — this
doesn't need to discover new games, only re-parse ones this warehouse
already has a NflPlay/TeamGameStat row for, so it reads its own game list
straight from the DB instead of re-walking every week of every season.

Idempotent, same as nfl_stats_db.ingest_play_by_play() itself: re-running
this (e.g. after a second parser fix later) only adds/updates rows, never
duplicates — safe to re-run.

Usage (run inside the container, same as every other *_backfill.py script
in this app):
    docker compose exec sports-betting python3 nfl_playbyplay_backfill.py
    docker compose exec sports-betting python3 nfl_playbyplay_backfill.py --dry-run
    docker compose exec sports-betting python3 nfl_playbyplay_backfill.py --limit 5
"""
import argparse
import time

import nfl_boxscore_api
import nfl_stats_db


def _stored_games():
    """[(game_id, game_date, season, game_type), ...] for every distinct
    game this warehouse already has a TeamGameStat row for — the full
    set of games worth re-parsing, not just ones that already have
    NflPlay rows (a game whose play-by-play fetch failed entirely the
    first time round has zero NflPlay rows and would otherwise never be
    retried by this script)."""
    from app import app as flask_app, db, TeamGameStat

    with flask_app.app_context():
        rows = (db.session.query(TeamGameStat.game_id, TeamGameStat.game_date,
                                  TeamGameStat.season, TeamGameStat.game_type)
                .filter(TeamGameStat.sport == 'NFL', TeamGameStat.is_home.is_(True))
                .order_by(TeamGameStat.game_date.asc())
                .all())
        return [tuple(r) for r in rows]


def backfill(dry_run=False, limit=None):
    games = _stored_games()
    if limit:
        games = games[:limit]
    print(f'{len(games)} stored games to re-parse.')

    added_by_type = {}
    failed = 0
    for i, (game_id, game_date, season, game_type) in enumerate(games, 1):
        try:
            summary = nfl_boxscore_api.get_live_summary(game_id)
        except Exception as e:
            print(f'  [{i}/{len(games)}] {game_id} ({game_date}) — fetch failed: {e}')
            failed += 1
            continue

        if not dry_run:
            from app import app as flask_app, NflPlay

            with flask_app.app_context():
                before_ids = {
                    r.play_id for r in NflPlay.query.filter_by(game_id=game_id)
                    .with_entities(NflPlay.play_id).all()
                }
            nfl_stats_db.ingest_play_by_play(game_id, game_date, season, game_type, summary)
            with flask_app.app_context():
                new_rows = (NflPlay.query.filter_by(game_id=game_id)
                            .filter(~NflPlay.play_id.in_(before_ids) if before_ids else True)
                            .all())
                for r in new_rows:
                    added_by_type[r.play_type] = added_by_type.get(r.play_type, 0) + 1
        if i % 25 == 0 or i == len(games):
            print(f'  [{i}/{len(games)}] through {game_date}...')
        time.sleep(0.2)  # be polite to ESPN's free, unofficial API

    print(f'\nDone — {len(games) - failed}/{len(games)} games re-parsed ({failed} fetch failures).')
    if added_by_type:
        print('New rows by play_type (previously-dropped types this run added):')
        for t, c in sorted(added_by_type.items(), key=lambda kv: -kv[1]):
            print(f'  {t}: {c}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help="fetch and report counts, don't write")
    parser.add_argument('--limit', type=int, default=None, help='only re-parse the first N stored games (oldest first)')
    args = parser.parse_args()
    backfill(dry_run=args.dry_run, limit=args.limit)
