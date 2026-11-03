#!/usr/bin/env python3
"""
cbb_stats_backfill.py — Seeds / catches up the CBB rows of the local stats
warehouse (TeamGameStat, sport='CBB') from cbb_rating's season caches.

    python3 cbb_stats_backfill.py                  # seasons 2024-2026 + current, from cache
    python3 cbb_stats_backfill.py --seasons 2026
    python3 cbb_stats_backfill.py --recent 3       # refresh + ingest the last 3 days

The season caches (instance/cbb_rating_games_<season>.json) are filled by
cbb_rating.load_season(), which fetches any day it hasn't successfully
fetched yet — so a first run on a fresh deploy pulls each season's ~160
D1 scoreboards once (~10-15 s per season) and every later run is local.
Idempotent: ingest_games() upserts on (sport, game_id, team_id).
"""
import argparse
import sys
from datetime import datetime, timedelta

import cbb_rating
import cbb_stats_db


def backfill_seasons(seasons):
    total = 0
    for season in seasons:
        games = cbb_rating.load_season(season)
        n = cbb_stats_db.ingest_games(games)
        print(f'[cbb-backfill] season {season}: {len(games)} games -> {n} TeamGameStat rows', flush=True)
        total += n
    return total


def backfill_recent(days=3):
    """Re-fetch the last `days` ET dates into the season cache (catches games
    that finalized while the app was down) and ingest those games."""
    today = datetime.now(cbb_rating._ET).date()
    dates = [(today - timedelta(days=d)).strftime('%Y-%m-%d') for d in range(1, days + 1)]
    by_season = {}
    for d in dates:
        by_season.setdefault(cbb_rating.season_for_date(d), []).append(d)
    n = 0
    for season, ds in by_season.items():
        first, last = cbb_rating.season_window(season)
        ds = [d for d in ds if first.strftime('%Y-%m-%d') <= d <= last.strftime('%Y-%m-%d')]
        if not ds:
            continue
        games = cbb_rating.load_season(season, force_days=ds)
        n += cbb_stats_db.ingest_games([g for g in games if g['date'] in ds])
    print(f'[cbb-backfill] recent {days}d: {n} TeamGameStat rows', flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', type=int, nargs='*')
    ap.add_argument('--recent', type=int)
    args = ap.parse_args()
    if args.recent:
        backfill_recent(args.recent)
        return 0
    current = cbb_rating.season_for_date(datetime.now(cbb_rating._ET).strftime('%Y-%m-%d'))
    backfill_seasons(args.seasons or sorted({2024, 2025, 2026, current}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
