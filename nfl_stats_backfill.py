#!/usr/bin/env python3
"""
nfl_stats_backfill.py — One-time (and safely re-runnable) catch-up: walks
ESPN's scoreboard week-by-week (same seasontype/week paging
nfl_api._get_season_game_log's live fallback already uses) and ingests
every completed game's final score into the local stats warehouse
(app.py's TeamGameStat table, written via nfl_stats_db.ingest_team_game).

See nfl_stats_db.py's module docstring for why this exists: nfl_api.py's
_build_game() ingests a game the moment it's built, but only for a week
something actually renders (the page route, or a cron tick) — this script
is the one-time catch-up for games that finished before that wiring
existed, and a manual repair tool if a gap is ever found later.

Regular season only by default (seasontype=2) — matches what
nfl_api._get_season_game_log() has always scoped itself to (team PPG/form/
rest-days inputs deliberately exclude preseason, unlike the NBA stats
warehouse, which includes preseason since it's the only data available
before an NBA regular season starts — NFL's regular season is already
well underway by the time this was written). Pass --seasontype 1 or 3 to
backfill preseason/postseason instead, if ever useful.

Idempotent: nfl_stats_db.ingest_team_game() upserts on (sport, game_id,
team_id), so re-running this script only adds genuinely new games.

Usage (run inside the container, same as every other *_backfill.py script
in this app, so `from app import ...` resolves against the real app/DB):
    docker compose exec sports-betting python3 nfl_stats_backfill.py
    docker compose exec sports-betting python3 nfl_stats_backfill.py --dry-run
    docker compose exec sports-betting python3 nfl_stats_backfill.py --weeks 1 5
"""
import argparse
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

import nfl_stats_db

ESPN_NFL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
_ET = ZoneInfo('America/New_York')


def _get_nfl_season():
    """Mirrors nfl_api._get_nfl_season() — duplicated rather than imported
    so this script has no dependency on nfl_api.py (which now depends on
    this module for ingestion; see nba_stats_backfill.py's docstring for
    why that chain stays one-directional)."""
    today = datetime.now(_ET).date()
    return today.year if today.month >= 9 else today.year - 1


def _fetch_week(season, week, seasontype):
    try:
        r = requests.get(ESPN_NFL, params={
            'seasontype': seasontype, 'week': week, 'season': season,
            'dates': season, 'limit': 20,
        }, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f'  ! scoreboard fetch failed for week {week}: {e}')
        return None


def backfill(season, week_start, week_end, seasontype=2, dry_run=False):
    game_type = nfl_stats_db.GAME_TYPE_BY_ESPN_SEASON_TYPE.get(seasontype, 'regular')
    total_games = 0
    for week in range(week_start, week_end + 1):
        data = _fetch_week(season, week, seasontype)
        if not data:
            continue
        found_any = False
        for event in data.get('events', []):
            comp = event.get('competitions', [{}])[0]
            state = comp.get('status', {}).get('type', {}).get('state', 'pre')
            if state != 'post':
                continue
            found_any = True

            tmap = {c.get('homeAway'): c for c in comp.get('competitors', [])}
            home_c, away_c = tmap.get('home', {}), tmap.get('away', {})
            home_id = (home_c.get('team') or {}).get('id')
            away_id = (away_c.get('team') or {}).get('id')
            home_name = (home_c.get('team') or {}).get('displayName', '')
            away_name = (away_c.get('team') or {}).get('displayName', '')
            if not home_id or not away_id:
                continue
            try:
                home_score = float(home_c.get('score'))
                away_score = float(away_c.get('score'))
            except (TypeError, ValueError):
                continue

            game_date_et = event.get('date', '')[:10]
            h_ab = (home_c.get('team') or {}).get('abbreviation', '?')
            a_ab = (away_c.get('team') or {}).get('abbreviation', '?')
            print(f'  wk{week}  {game_date_et}  {a_ab} @ {h_ab}')
            if not dry_run:
                nfl_stats_db.ingest_team_game(
                    event.get('id'), game_date_et, season, game_type,
                    home_id, away_id, home_name, away_name, home_score, away_score)
            total_games += 1

        if not found_any and week > 3:
            # Same early-stop heuristic as nfl_api._get_season_game_log's
            # live fallback — season not started yet, or past its end.
            break
        time.sleep(0.2)  # be polite to ESPN's free, unofficial API

    print(f'\nDone — {total_games} games {"would be " if dry_run else ""}ingested.')
    return total_games


def backfill_recent(weeks=2):
    """Self-heal hook, called from app.py's daily cron — re-walks just the
    last couple of weeks (cheap, idempotent) to catch anything missed
    during downtime/deploys. See nba_stats_backfill.backfill_recent()'s
    docstring for the full reasoning (identical, NFL side)."""
    season = _get_nfl_season()
    current = _current_week_guess()
    return backfill(season, max(1, current - weeks + 1), current, seasontype=2, dry_run=False)


def _current_week_guess():
    """Rough current NFL week from today's date — only used to bound the
    self-heal re-walk window, not for anything that needs to be exact."""
    season = _get_nfl_season()
    season_start = datetime(season, 9, 1, tzinfo=_ET).date()
    days_in = (datetime.now(_ET).date() - season_start).days
    return max(1, min(18, days_in // 7 + 1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weeks', nargs=2, type=int, metavar=('START', 'END'),
                    help='Week range, default: 1 18')
    p.add_argument('--seasontype', type=int, default=2, help='1=preseason, 2=regular (default), 3=postseason')
    p.add_argument('--dry-run', action='store_true', help="Print what would be ingested without writing to the DB")
    args = p.parse_args()

    season = _get_nfl_season()
    week_start, week_end = args.weeks if args.weeks else (1, 18)

    print(f'Backfilling NFL stats season {season}, weeks {week_start}-{week_end} (seasontype {args.seasontype})...')
    backfill(season, week_start, week_end, seasontype=args.seasontype, dry_run=args.dry_run)
