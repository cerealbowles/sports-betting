#!/usr/bin/env python3
"""
nba_stats_backfill.py — One-time (and safely re-runnable) catch-up: walks
ESPN's scoreboard day-by-day over a date range and ingests every completed
game's full box score into the local stats warehouse (app.py's
PlayerGameStat/TeamGameStat tables, written via nba_stats_db.ingest_game).

See nba_stats_db.py's module docstring for why this table exists:
nba_roster_api.get_player_gamelog() reads from it before ever falling back
to a live per-player ESPN fetch. Going forward, nba_api.py's _build_game()
ingests each newly-Final game automatically — this script exists for the
one-time catch-up of games that finished BEFORE that wiring went in (i.e.
everything already played this season), and as a manual repair tool if a
gap is ever found later.

Preseason is included deliberately (unlike nba_api._get_season_game_log,
which only keeps regular season for the team-record/standings page) — with
the 2026-27 regular season not yet underway, preseason games are the only
real game data available right now, and nba_player_model's recent-
production projections are more useful with *some* current-roster signal
than none. game_type is stored per row either way, so anything reading
this table back can filter preseason out if it specifically wants to.

Idempotent: nba_stats_db.ingest_game() upserts on (sport, game_id,
player_id/team_id), so re-running this script only adds genuinely new
games — already-ingested ones just get overwritten with identical data.

Usage (run inside the container so `from app import ...` resolves against
the real app/DB, same as every other *_backfill.py script in this app):
    docker compose exec sports-betting python3 nba_stats_backfill.py
    docker compose exec sports-betting python3 nba_stats_backfill.py --dry-run
    docker compose exec sports-betting python3 nba_stats_backfill.py --start 2026-10-01 --end 2026-10-06
"""
import argparse
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

import nba_boxscore_api
import nba_stats_db

ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
_ET = ZoneInfo('America/New_York')


def _get_nba_season():
    """NBA season year (ESPN convention: `season=2026` means the 2026-27
    season) — mirrors nba_api._get_nba_season(), duplicated here rather than
    imported so this script has no dependency on nba_api.py (which itself
    now depends on this module for ingestion — see nba_stats_db.py's
    docstring on keeping this a one-directional, non-circular chain)."""
    today = datetime.now(_ET).date()
    return today.year if today.month >= 10 else today.year - 1


def _fetch_day(date_str):
    try:
        r = requests.get(ESPN_NBA, params={'dates': date_str, 'limit': 100}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f'  ! scoreboard fetch failed for {date_str}: {e}')
        return None


def backfill(start_date, end_date, dry_run=False):
    season = _get_nba_season()
    day = start_date
    total_games = 0
    while day <= end_date:
        date_str = day.strftime('%Y%m%d')
        data = _fetch_day(date_str)
        day += timedelta(days=1)
        if not data:
            continue

        for event in data.get('events', []):
            comp = event.get('competitions', [{}])[0]
            state = comp.get('status', {}).get('type', {}).get('state', 'pre')
            if state != 'post':
                continue  # only finalized games have a real box score

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
            game_type = nba_stats_db.GAME_TYPE_BY_ESPN_SEASON_TYPE.get(
                event.get('season', {}).get('type'), 'regular')
            h_ab = (home_c.get('team') or {}).get('abbreviation', '?')
            a_ab = (away_c.get('team') or {}).get('abbreviation', '?')

            boxscore = nba_boxscore_api.get_live_boxscore(event_id)
            if not boxscore:
                print(f'  ! no boxscore for {event_id} ({a_ab} @ {h_ab}, {game_date_et})')
                continue

            print(f'  {game_date_et}  {a_ab} @ {h_ab}  ({game_type})')
            if not dry_run:
                nba_stats_db.ingest_game(event_id, game_date_et, season, game_type,
                                          home_id, away_id, home_score, away_score, boxscore)
            total_games += 1
        time.sleep(0.2)  # be polite to ESPN's free, unofficial API

    print(f'\nDone — {total_games} games {"would be " if dry_run else ""}ingested.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', help='YYYY-MM-DD, default: season start (Oct 1)')
    p.add_argument('--end', help='YYYY-MM-DD, default: today ET')
    p.add_argument('--dry-run', action='store_true', help="Print what would be ingested without writing to the DB")
    args = p.parse_args()

    season = _get_nba_season()
    start = datetime.strptime(args.start, '%Y-%m-%d').date() if args.start else datetime(season, 10, 1).date()
    end = datetime.strptime(args.end, '%Y-%m-%d').date() if args.end else datetime.now(_ET).date()

    print(f'Backfilling NBA stats {start} -> {end} (season {season})...')
    backfill(start, end, dry_run=args.dry_run)
