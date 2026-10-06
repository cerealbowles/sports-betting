#!/usr/bin/env python3
"""
nhl_stats_backfill.py — One-time (and safely re-runnable) catch-up: walks
ESPN's scoreboard day-by-day over a date range and ingests every completed
game's full box score into the local stats warehouse (app.py's
PlayerGameStat/TeamGameStat tables, written via nhl_stats_db.ingest_team_game()/ingest_player_stats()).

See nhl_stats_db.py's module docstring for why this table exists —
nhl_player_model.py's skater/goalie grades read from it. Going forward,
nhl_api.py ingests each newly-Final game automatically — this script
exists for the one-time catch-up of games that finished BEFORE that
wiring went in (i.e. everything already played this season), and as a
manual repair tool if a gap is ever found later.

Preseason is included (game_type is stored per row either way, so anything
reading this table back can filter it out if it wants to) — same
"capture everything, let readers filter" approach as nba_stats_backfill.py,
even though nhl_api.py's own existing display logic (team records,
standings) already pulls from the official NHL API rather than this
warehouse and has no preseason-filtering concern of its own.

Idempotent: nhl_stats_db.ingest_team_game()/ingest_player_stats() upsert on (sport, game_id,
player_id/team_id), so re-running this script only adds genuinely new
games — already-ingested ones just get overwritten with identical data.

Usage (run inside the container so `from app import ...` resolves against
the real app/DB, same as every other *_backfill.py script in this app):
    docker compose exec sports-betting python3 nhl_stats_backfill.py
    docker compose exec sports-betting python3 nhl_stats_backfill.py --dry-run
    docker compose exec sports-betting python3 nhl_stats_backfill.py --start 2026-10-01 --end 2026-10-06
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

import nhl_boxscore_api
import nhl_stats_db

ESPN_NHL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
_ET = ZoneInfo('America/New_York')


def _get_nhl_season():
    """NHL season year (ESPN convention: `season=2026` means the 2026-27
    season) — mirrors nhl_api._get_nhl_season(), duplicated here rather than
    imported so this script has no dependency on nhl_api.py (which itself
    now depends on this module for ingestion — see nhl_stats_db.py's
    docstring on keeping this a one-directional, non-circular chain)."""
    today = datetime.now(_ET).date()
    return today.year if today.month >= 8 else today.year - 1


def _fetch_day(date_str):
    try:
        r = requests.get(ESPN_NHL, params={'dates': date_str, 'limit': 100}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f'  ! scoreboard fetch failed for {date_str}: {e}')
        return None


def backfill(start_date, end_date, dry_run=False):
    day = start_date
    total_games = 0
    while day <= end_date:
        date_str = day.strftime('%Y%m%d')
        data = _fetch_day(date_str)
        day += timedelta(days=1)
        if not data:
            continue

        finalized = [
            event for event in data.get('events', [])
            if (event.get('competitions', [{}])[0]
                    .get('status', {}).get('type', {}).get('state', 'pre')) == 'post'
        ]
        if not finalized:
            continue

        # Fetch this day's box scores in parallel (several games on a given
        # day, each needing its own summary fetch) — same ThreadPoolExecutor
        # fix nba_stats_backfill.py needed after its own sequential per-game
        # fetches turned out to be the actual bottleneck for a full
        # historical season, applied here from the start instead of
        # relearning it the slow way.
        boxscores = {}
        with ThreadPoolExecutor(max_workers=min(len(finalized), 8)) as ex:
            futures = {ex.submit(nhl_boxscore_api.get_live_boxscore, event.get('id')): event.get('id')
                       for event in finalized}
            for f in as_completed(futures):
                eid = futures[f]
                try:
                    boxscores[eid] = f.result()
                except Exception:
                    boxscores[eid] = {}

        for event in finalized:
            comp = event.get('competitions', [{}])[0]

            # Per-event, not a single outer variable derived from "today" —
            # a backfill date range can (and for a historical season, does)
            # span a different season than the one in progress right now.
            # Derived from each event's OWN date (same month>=8 rule as
            # _get_nhl_season(), just applied to the event instead of
            # "today") — NOT from ESPN's own event.season.year field, which
            # uses the season's ENDING year (e.g. 2025 for the 2024-25
            # season) rather than this app's STARTING-year convention (2024
            # for that same season, matching _get_nhl_season()'s own
            # convention everywhere else in this app). Using ESPN's field
            # directly was tried and confirmed wrong: it would store a
            # season number one higher than this app's own, including
            # colliding with the real current season's number for a
            # just-finished season (e.g. 2025-26 games landing under the
            # same `season` value nhl_api.py uses for 2026-27).
            event_date_str = (event.get('date') or '')[:10]  # 'YYYY-MM-DD'
            if event_date_str:
                ev_year, ev_month = int(event_date_str[:4]), int(event_date_str[5:7])
                season = ev_year if ev_month >= 8 else ev_year - 1
            else:
                season = _get_nhl_season()

            event_id = event.get('id')
            tmap = {c.get('homeAway'): c for c in comp.get('competitors', [])}
            home_c, away_c = tmap.get('home', {}), tmap.get('away', {})
            home_id = (home_c.get('team') or {}).get('id')
            away_id = (away_c.get('team') or {}).get('id')
            if not home_id or not away_id:
                continue
            try:
                home_score = float(home_c.get('score'))
                away_score = float(away_c.get('score'))
            except (TypeError, ValueError):
                continue

            game_date_et = event.get('date', '')[:10]
            game_type = nhl_stats_db.GAME_TYPE_BY_ESPN_SEASON_TYPE.get(
                event.get('season', {}).get('type'), 'regular')
            home_name = (home_c.get('team') or {}).get('displayName', '')
            away_name = (away_c.get('team') or {}).get('displayName', '')
            h_ab = (home_c.get('team') or {}).get('abbreviation', '?')
            a_ab = (away_c.get('team') or {}).get('abbreviation', '?')

            boxscore = boxscores.get(event_id)
            if not boxscore:
                print(f'  ! no boxscore for {event_id} ({a_ab} @ {h_ab}, {game_date_et})')
                continue

            print(f'  {game_date_et}  {a_ab} @ {h_ab}  ({game_type})')
            if not dry_run:
                nhl_stats_db.ingest_team_game(event_id, game_date_et, season, game_type,
                                               home_id, away_id, home_name, away_name,
                                               home_score, away_score)
                nhl_stats_db.ingest_player_stats(event_id, game_date_et, season, game_type,
                                                  home_id, away_id, boxscore)
            total_games += 1
        time.sleep(0.2)  # be polite to ESPN's free, unofficial API

    print(f'\nDone — {total_games} games {"would be " if dry_run else ""}ingested.')
    return total_games


def backfill_recent(days=3):
    """Self-heal hook, called from app.py's daily cron (see _start_cache_warmer):
    re-walks just the last `days` days and re-ingests anything Final.
    Idempotent, so this is cheap insurance rather than a real backfill —
    covers a game that went Final while the container was down/mid-deploy,
    or any 20-min cron tick that errored before reaching the ingest call in
    nhl_api.py's _build_game(), without needing anyone to notice and run
    the full script by hand. Silent on failure (best-effort, same
    philosophy as every other cron job in app.py) — a miss here just means
    nhl_stats_db stays one cycle further behind, not a broken page."""
    end = datetime.now(_ET).date()
    start = end - timedelta(days=days - 1)
    return backfill(start, end, dry_run=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', help='YYYY-MM-DD, default: season start (Sept 1)')
    p.add_argument('--end', help='YYYY-MM-DD, default: today ET')
    p.add_argument('--season', type=int, default=None, help='e.g. 2024 for the 2024-25 season, default: current season')
    p.add_argument('--dry-run', action='store_true', help="Print what would be ingested without writing to the DB")
    args = p.parse_args()

    season = args.season or _get_nhl_season()
    start = datetime.strptime(args.start, '%Y-%m-%d').date() if args.start else datetime(season, 9, 1).date()
    # Default end: today, capped to June 30 of the following year (NHL
    # seasons span two calendar years, same as NBA's) — lets a historical
    # --season run without also needing an explicit --end.
    end = (datetime.strptime(args.end, '%Y-%m-%d').date() if args.end
           else min(datetime.now(_ET).date(), datetime(season + 1, 6, 30).date()))

    print(f'Backfilling NHL stats {start} -> {end} (season {season})...')
    backfill(start, end, dry_run=args.dry_run)
