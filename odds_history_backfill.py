#!/usr/bin/env python3
"""
odds_history_backfill.py — Fill total_close on game_predictions rows from
The Odds API's historical odds endpoint, for games outside ESPN pickcenter's
~290-day retention window that espn_odds_backfill.py already covers.

Two calls per date instead of one, because our old rows have no
game_time_utc (can't target a single kickoff) and a single snapshot for a
whole day's slate would only be a true "closing" number for whichever game
is about to start at that exact moment:

  1. Discovery call — one early-day snapshot (09:00 ET) to find each
     missing row's real commence_time from the Odds API's own event data.
  2. Closing call(s) — one snapshot per distinct commence_time found that
     date, taken 5 minutes before kickoff, to read the actual closing total.

Matches events to rows by exact normalized team names (home+away, with
swap fallback), the same approach odds_api.py's lookup_game_odds already
uses in production — no alias table needed.

Costs 10 credits per call (regions=us, markets=totals — confirmed from
x-requests-remaining). Aborts if remaining quota drops below CREDIT_FLOOR
so a bug can't burn the whole key.

Usage:
    python3 odds_history_backfill.py --dry-run --since 2025-01-01 --until 2026-01-01
    python3 odds_history_backfill.py --sport NFL NBA --since 2025-01-01 --until 2026-01-01
"""
import os
import sys
import time
import sqlite3
import argparse
import unicodedata
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import json

import requests

_ET = ZoneInfo('America/New_York')

SPORT_KEYS = {
    'NFL':  'americanfootball_nfl',
    'CFB':  'americanfootball_ncaaf',
    'NBA':  'basketball_nba',
    'NHL':  'icehockey_nhl',
    'WNBA': 'basketball_wnba',
}
ALL_SPORTS = list(SPORT_KEYS.keys())

BASE = "https://api.the-odds-api.com/v4/historical/sports"
DB_PATH = os.environ.get('DB_PATH', os.path.join(os.path.dirname(__file__), 'data', 'bets.db'))
API_KEY = os.environ.get('ODDS_API_KEY', '')
CREDIT_FLOOR = int(os.environ.get('ODDS_CREDIT_FLOOR', '2000'))   # abort rather than risk running the key dry
SLEEP = 0.3

# Discovery calls are the expensive, re-runnable-for-free part — a historical
# date's event list never changes, so once fetched it's cached to disk
# forever. This means a dry run, an aborted run, and the eventual real run
# all share the same discovery spend instead of re-paying for it each time.
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(DB_PATH)) or '.',
                          'odds_backfill_discovery_cache.json')


def _load_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    try:
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass


def _normalize(name):
    name = unicodedata.normalize('NFD', name or '')
    name = ''.join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip()


class CreditFloorHit(Exception):
    pass


def _historical_call(sport_key, iso_ts, stats):
    r = requests.get(
        f"{BASE}/{sport_key}/odds/",
        params={
            'apiKey':     API_KEY,
            'regions':    'us',
            'markets':    'totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso',
            'date':       iso_ts,
        },
        timeout=15,
    )
    remaining = r.headers.get('x-requests-remaining')
    stats['calls'] += 1
    if remaining is not None:
        remaining = int(remaining)
        stats['remaining'] = remaining
        if remaining < CREDIT_FLOOR:
            raise CreditFloorHit(f'{remaining} credits remaining — below floor of {CREDIT_FLOOR}')
    r.raise_for_status()
    return r.json()


def _totals_point(event):
    """Prefer FanDuel's totals point (matches how the rest of the app sources
    lines); fall back to the average across whatever books are present."""
    pts = []
    fd = None
    for bm in event.get('bookmakers', []):
        for m in bm.get('markets', []):
            if m.get('key') != 'totals':
                continue
            for o in m.get('outcomes', []):
                if o.get('point') is None:
                    continue
                pts.append(o['point'])
                if bm.get('key') == 'fanduel':
                    fd = o['point']
    if fd is not None:
        return fd
    return round(sum(pts) / len(pts), 1) if pts else None


def _event_matches(event, home_name, away_name):
    """True if event's teams match (home,away) in either orientation —
    totals don't care which side is home, we just need the right game."""
    eh, ea = _normalize(event.get('home_team', '')), _normalize(event.get('away_team', ''))
    h, a = _normalize(home_name), _normalize(away_name)
    return (eh == h and ea == a) or (eh == a and ea == h)


def _et_time_to_utc_iso(date_str, hour, minute=0):
    dt = datetime.strptime(date_str, '%Y-%m-%d').replace(
        hour=hour, minute=minute, tzinfo=_ET
    )
    return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def fetch_missing_rows(conn, sport, since, until):
    cur = conn.execute(
        'SELECT id, game_date, home_team, away_team FROM game_predictions '
        'WHERE sport=? AND total_close IS NULL AND game_date >= ? AND game_date < ? '
        'ORDER BY game_date',
        (sport, since, until),
    )
    return cur.fetchall()


def run(sports, since, until, dry_run):
    if not API_KEY:
        print('ODDS_API_KEY not set.', file=sys.stderr)
        sys.exit(1)

    # Long busy timeout + retry on commit — the live app (odds polling, bet
    # settlement) writes to this same DB concurrently, so a lock contention
    # is routine, not exceptional, and shouldn't crash a long-running backfill.
    conn = sqlite3.connect(DB_PATH, timeout=30)
    stats = {'calls': 0, 'remaining': None}
    totals = {'filled': 0, 'no_discovery_match': 0, 'no_closing_match': 0, 'dates': 0, 'clusters': 0}
    cache = _load_cache()
    cache_hits = 0

    print('─' * 62)
    print(f'  Odds History Backfill — {sports}{"  (dry run)" if dry_run else ""}')
    print(f'  Range: {since} .. {until}')
    print('─' * 62)

    try:
        for sport in sports:
            sport_key = SPORT_KEYS[sport]
            rows = fetch_missing_rows(conn, sport, since, until)
            if not rows:
                print(f'\n{sport}: nothing missing in range.')
                continue
            by_date = {}
            for rid, gdate, home, away in rows:
                by_date.setdefault(gdate, []).append((rid, home, away))
            print(f'\n{sport}: {len(rows)} row(s) across {len(by_date)} date(s).')

            for gdate, game_rows in by_date.items():
                totals['dates'] += 1
                clusters = {}  # commence_time -> [(rid, home, away)]
                remaining_rows = list(game_rows)

                # A late-night ET game can land on the next UTC calendar day,
                # which sometimes shows up as the next game_date too (same
                # boundary issue espn_odds_backfill.py works around with its
                # own +/-1 day retry). Try this date's morning snapshot first,
                # then fall back to the previous day's for whatever's left —
                # by morning-of-day-X, a game that kicked off late on day X-1
                # has already left the live board.
                for day_offset in (0, -1):
                    if not remaining_rows:
                        break
                    probe_date = gdate if day_offset == 0 else (
                        datetime.strptime(gdate, '%Y-%m-%d') + timedelta(days=day_offset)
                    ).strftime('%Y-%m-%d')
                    cache_key = f'{sport_key}|{probe_date}'
                    if cache_key in cache:
                        events = cache[cache_key]
                        cache_hits += 1
                    else:
                        discovery_ts = _et_time_to_utc_iso(probe_date, 9)
                        try:
                            data = _historical_call(sport_key, discovery_ts, stats)
                        except CreditFloorHit as e:
                            print(f'\n  ABORT: {e}')
                            raise
                        except Exception as e:
                            print(f'  {gdate}: discovery call failed ({e}) for offset {day_offset}')
                            continue
                        time.sleep(SLEEP)
                        events = data.get('data', [])
                        cache[cache_key] = events
                        _save_cache(cache)

                    still_unmatched = []
                    for rid, home, away in remaining_rows:
                        ev = next((e for e in events if _event_matches(e, home, away)), None)
                        if not ev or not ev.get('commence_time'):
                            still_unmatched.append((rid, home, away))
                            continue
                        clusters.setdefault(ev['commence_time'], []).append((rid, home, away))
                    remaining_rows = still_unmatched

                for rid, home, away in remaining_rows:
                    totals['no_discovery_match'] += 1

                if not clusters:
                    print(f'  {gdate}: no discovery matches for any of {len(game_rows)} row(s)')
                    continue

                totals['clusters'] += len(clusters)
                print(f'  {gdate}: {len(clusters)} kickoff cluster(s) for {len(game_rows)} row(s)'
                      + ('  [dry-run: not fetching closing lines]' if dry_run else ''))

                if dry_run:
                    continue

                for commence_time, cluster_rows in clusters.items():
                    try:
                        commence_dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                    except Exception:
                        continue
                    closing_ts = (commence_dt - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
                    try:
                        close_data = _historical_call(sport_key, closing_ts, stats)
                    except CreditFloorHit as e:
                        print(f'\n  ABORT: {e}')
                        raise
                    except Exception as e:
                        print(f'    {commence_time}: closing call failed ({e})')
                        continue
                    time.sleep(SLEEP)

                    close_events = close_data.get('data', [])
                    for rid, home, away in cluster_rows:
                        ev = next((e for e in close_events if _event_matches(e, home, away)), None)
                        point = _totals_point(ev) if ev else None
                        if point is None:
                            totals['no_closing_match'] += 1
                            print(f'    ✗ no closing total: {away} @ {home}')
                            continue
                        conn.execute(
                            'UPDATE game_predictions SET total_close=? WHERE id=?',
                            (point, rid),
                        )
                        totals['filled'] += 1
                        print(f'    ✓ {away} @ {home}: total_close={point}')

                conn.commit()
    except CreditFloorHit:
        pass
    finally:
        conn.close()

    print('\n' + '─' * 62)
    print(f'  API calls made: {stats["calls"]}  (credits remaining: {stats["remaining"]})  '
          f'discovery cache hits: {cache_hits}')
    print(f'  Dates processed: {totals["dates"]}  Kickoff clusters: {totals["clusters"]}')
    if dry_run:
        est_closing_calls = totals['clusters']
        print(f'  Estimated additional closing-call cost if run for real: '
              f'{est_closing_calls} calls x 10 credits = {est_closing_calls * 10} credits')
        print('  Dry run — no writes made.')
    else:
        print(f'  Filled: {totals["filled"]}  '
              f'No discovery match: {totals["no_discovery_match"]}  '
              f'No closing match: {totals["no_closing_match"]}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sport', nargs='+', default=ALL_SPORTS, choices=ALL_SPORTS)
    p.add_argument('--since', default='2025-01-01')
    p.add_argument('--until', default='2026-01-01')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    run([s.upper() for s in args.sport], args.since, args.until, args.dry_run)
