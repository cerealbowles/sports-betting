#!/usr/bin/env python3
"""
espn_odds_backfill.py — Fill in spread_open/close and total_open/close on
existing game_predictions rows, pulled from ESPN's pickcenter odds block.
Prep work for a future spread/O-U prediction model — moneyline is already
tracked separately via odds_api.py, this fills the gap nothing else covers.

Usage:
    python3 espn_odds_backfill.py                       # all sports, all rows missing data
    python3 espn_odds_backfill.py --sport NFL MLB
    python3 espn_odds_backfill.py --dry-run

Walks each sport's distinct game_date values in game_predictions, pulls
that date's ESPN scoreboard once, matches events to rows by exact
(home_team, away_team) displayName, then fetches pickcenter per matched
event. Safe to re-run: skips rows that already have spread_close and
total_close set.

Only considers COMPLETED games (home_won is set) from the last
RETENTION_FLOOR_DAYS — ESPN's pickcenter odds block empirically disappears
for events older than ~300 days (confirmed by probing week-by-week: still
present for a game 296 days back, gone for one 302 days back), so querying
older rows is a guaranteed-empty round trip per game. Games newer than the
floor but still upcoming are skipped too since they have no closing line
yet — they'll be picked up on a later re-run once they're Final.
"""
import os
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone

def _shift(date_str, days):
    return (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=days)).strftime('%Y-%m-%d')

def _norm(name):
    """Strip diacritics for matching — our NHL data source spells it
    'Montréal Canadiens', ESPN spells it 'Montreal Canadiens'."""
    return ''.join(c for c in unicodedata.normalize('NFKD', name) if not unicodedata.combining(c))

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import espn_pickcenter as espn

ALL_SPORTS = ['NFL', 'CFB', 'MLB', 'NBA', 'NHL', 'WNBA']
RETENTION_FLOOR_DAYS = 290  # conservative margin under ESPN's observed ~300-day cutoff


def run(sports=None, dry_run=False, since=None):
    sports = sports or ALL_SPORTS
    floor_date = since or (datetime.now(timezone.utc) - timedelta(days=RETENTION_FLOOR_DAYS)).strftime('%Y-%m-%d')
    print('─' * 58)
    print(f'  ESPN Odds Backfill — {sports}{"  (dry run — no writes)" if dry_run else ""}')
    print(f'  Only completed games on/after {floor_date} (ESPN odds retention floor)')
    print('─' * 58)

    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, GamePrediction

    totals = {'matched': 0, 'unmatched': 0, 'no_odds': 0}

    with flask_app.app_context():
        for sport in sports:
            all_missing = GamePrediction.query.filter(
                GamePrediction.sport == sport,
                db.or_(GamePrediction.spread_close.is_(None), GamePrediction.total_close.is_(None)),
            ).count()
            rows = GamePrediction.query.filter(
                GamePrediction.sport == sport,
                db.or_(GamePrediction.spread_close.is_(None), GamePrediction.total_close.is_(None)),
                GamePrediction.home_won.isnot(None),
                GamePrediction.game_date >= floor_date,
            ).all()
            if not rows:
                print(f'\n{sport}: nothing in range ({all_missing} row(s) missing odds, but none completed and within the retention floor).')
                continue
            dates = sorted({r.game_date for r in rows})
            note = f' (of {all_missing} missing odds total — the rest are outside ESPN\'s retention window or not yet Final)' if all_missing > len(rows) else ''
            print(f'\n{sport}: {len(rows)} row(s) in range{note}, across {len(dates)} date(s).')

            # game_date is stored ET, but ESPN's scoreboard `dates` param
            # buckets by UTC day — a Thu/Sun/Mon night game starting after
            # ~7pm ET rolls into the next UTC calendar day, so it's missing
            # from the ET-dated bucket. Fetch +/-1 day lazily, only for rows
            # that don't match on the first pass, rather than tripling every
            # sport's scoreboard calls up front.
            events_cache = {}
            def _events_for(date_str):
                if date_str not in events_cache:
                    events_cache[date_str] = espn.fetch_scoreboard_events(sport, date_str.replace('-', ''))
                return events_cache[date_str]

            for game_date in dates:
                _events_for(game_date)

            for r in rows:
                ev = None
                key = (_norm(r.home_team), _norm(r.away_team))
                for offset in (0, 1, -1):
                    probe_date = r.game_date if offset == 0 else _shift(r.game_date, offset)
                    by_teams = {(_norm(e['home_name']), _norm(e['away_name'])): e for e in _events_for(probe_date)}
                    ev = by_teams.get(key)
                    if ev:
                        break
                if not ev:
                    totals['unmatched'] += 1
                    print(f'  ✗ no ESPN match: {r.away_team} @ {r.home_team} ({r.game_date})')
                    continue

                pc = espn.fetch_pickcenter(sport, ev['event_id'])
                time.sleep(0.2)
                if not pc:
                    totals['no_odds'] += 1
                    print(f'  – no odds data: {r.away_team} @ {r.home_team} ({r.game_date})')
                    continue

                if dry_run:
                    totals['matched'] += 1
                    print(f'  ✓ {r.away_team} @ {r.home_team}: '
                          f'spread {pc["spread_open"]}→{pc["spread_close"]}  '
                          f'total {pc["total_open"]}→{pc["total_close"]}')
                    continue

                r.spread_open  = pc['spread_open']
                r.spread_close = pc['spread_close']
                r.total_open   = pc['total_open']
                r.total_close  = pc['total_close']
                totals['matched'] += 1

            if not dry_run:
                db.session.commit()

    print('\n' + '─' * 58)
    print(f'  Matched + filled: {totals["matched"]}')
    print(f'  No ESPN odds (outside retention or never posted): {totals["no_odds"]}')
    print(f'  No ESPN event match at all: {totals["unmatched"]}')
    if dry_run:
        print('\n  Dry run — no writes made. Run without --dry-run to persist.')


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    sports = ALL_SPORTS
    since = None
    if '--sport' in sys.argv:
        idx = sys.argv.index('--sport')
        sports = [s.upper() for s in sys.argv[idx + 1:] if not s.startswith('--')]
    if '--since' in sys.argv:
        since = sys.argv[sys.argv.index('--since') + 1]
    run(sports=sports, dry_run=dry_run, since=since)
