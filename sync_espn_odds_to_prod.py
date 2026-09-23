#!/usr/bin/env python3
"""
sync_espn_odds_to_prod.py — One-time sync of the spread_open/spread_close/
total_open/total_close columns from instance/bets.db (where the ESPN odds
backfill ran) into the live prod DB (data/bets.db), without touching
anything else in prod (bets, settings, any picks/outcomes written since the
snapshot was taken).

Targeted UPDATE by (sport, game_date, home_team, away_team) — not a file
overwrite — so it's safe to run even if prod has moved on since
instance/bets.db was copied out (new bets placed, new picks generated,
outcomes resolved). Only the 4 new odds columns are touched, and only for
rows where the backfill actually found data.

Needs root: data/bets.db is owned by root (Docker-created). Run with sudo.

Usage:
    sudo python3 sync_espn_odds_to_prod.py [--dry-run]
"""
import sqlite3
import sys

SRC = 'instance/bets.db'
DST = 'data/bets.db'


def ensure_columns(conn):
    cols = {row[1] for row in conn.execute('PRAGMA table_info(game_predictions)')}
    for col in ('spread_open', 'spread_close', 'total_open', 'total_close'):
        if col not in cols:
            conn.execute(f'ALTER TABLE game_predictions ADD COLUMN {col} FLOAT DEFAULT NULL')


def run(dry_run=False):
    src = sqlite3.connect(SRC)
    dst = sqlite3.connect(DST)

    ensure_columns(dst)

    rows = src.execute("""
        SELECT sport, game_date, home_team, away_team,
               spread_open, spread_close, total_open, total_close
        FROM game_predictions
        WHERE spread_close IS NOT NULL OR total_close IS NOT NULL
    """).fetchall()

    print(f'{len(rows)} row(s) in {SRC} have odds data to sync.')

    updated = missing = unchanged = 0
    for sport, game_date, home_team, away_team, so, sc, to, tc in rows:
        cur = dst.execute("""
            SELECT id, spread_open, spread_close, total_open, total_close
            FROM game_predictions
            WHERE sport=? AND game_date=? AND home_team=? AND away_team=?
        """, (sport, game_date, home_team, away_team)).fetchone()

        if cur is None:
            missing += 1
            continue

        pid, cur_so, cur_sc, cur_to, cur_tc = cur
        if (cur_so, cur_sc, cur_to, cur_tc) == (so, sc, to, tc):
            unchanged += 1
            continue

        if not dry_run:
            dst.execute("""
                UPDATE game_predictions
                SET spread_open=?, spread_close=?, total_open=?, total_close=?
                WHERE id=?
            """, (so, sc, to, tc, pid))
        updated += 1

    if not dry_run:
        dst.commit()

    print(f'  Updated:          {updated}')
    print(f'  Already in sync:  {unchanged}')
    print(f'  No matching row in prod (game not present there): {missing}')
    if dry_run:
        print('\nDry run — no writes made.')

    src.close()
    dst.close()


if __name__ == '__main__':
    run(dry_run='--dry-run' in sys.argv)
