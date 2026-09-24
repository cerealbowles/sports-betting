#!/usr/bin/env python3
"""
backfill_spread_accuracy.py — Retroactively fills spread_pick_line/
spread_cover_prob/spread_covered/spread_pick_roi on existing game_predictions
rows, using data that's already in the database.

The model's win probability (home_prob) is stored on every resolved row, and
espn_odds_backfill.py has already filled spread_open/spread_close (the
market line) for ~290 days of history. That's everything spread_proxy.
cover_prob() needs — no new data collection required, so unlike totals
(whose model projection was never persisted historically), spread accuracy
can start with a real sample instead of n=0.

Caveat: ESPN's pickcenter odds block has no per-side price, only the line.
Real spread_home_price/spread_away_price are only captured going forward
(via odds_api, in app.py's _upsert_predictions). For this one-time backfill,
both sides are assumed to be standard -110 juice — the default vig for
point-spread markets — so ROI here is an approximation. Once genuinely
live-tracked rows accumulate, the page's ROI numbers will reflect this
assumption fading out.

Usage:
    python3 backfill_spread_accuracy.py                  # all sports
    python3 backfill_spread_accuracy.py --sport NFL MLB
    python3 backfill_spread_accuracy.py --dry-run
    python3 backfill_spread_accuracy.py --force           # recompute rows already filled
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ASSUMED_PRICE = -110
ALL_SPORTS = ['NFL', 'CFB', 'MLB', 'NBA', 'NHL', 'WNBA']


def run(sports=None, dry_run=False, force=False):
    sports = sports or ALL_SPORTS
    print('─' * 58)
    print(f'  Spread Accuracy Backfill — {sports}{"  (dry run — no writes)" if dry_run else ""}')
    print('─' * 58)

    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, GamePrediction
    import spread_proxy

    totals = {'filled': 0, 'no_spread_open': 0, 'no_home_prob': 0}

    with flask_app.app_context():
        for sport in sports:
            filters = [
                GamePrediction.sport == sport,
                GamePrediction.home_won.isnot(None),
            ]
            if not force:
                filters.append(GamePrediction.spread_pick_line.is_(None))
            rows = GamePrediction.query.filter(*filters).all()
            if not rows:
                print(f'\n{sport}: nothing to backfill.')
                continue
            print(f'\n{sport}: {len(rows)} resolved row(s) without spread_pick_line.')

            filled = 0
            for r in rows:
                if r.spread_open is None:
                    totals['no_spread_open'] += 1
                    continue
                if r.home_prob is None:
                    totals['no_home_prob'] += 1
                    continue

                cover_prob = spread_proxy.cover_prob(sport, r.home_prob, r.spread_open)
                if cover_prob is None:
                    continue

                settle_line = r.spread_close if r.spread_close is not None else r.spread_open
                margin       = r.home_score - r.away_score
                cover_target = -settle_line

                if dry_run:
                    print(f'  ✓ {r.away_team} @ {r.home_team} ({r.game_date}): '
                          f'line {r.spread_open}  cover_prob {round(cover_prob, 3)}  '
                          f'margin {margin:+d}  target {cover_target:+g}')
                    filled += 1
                    continue

                r.spread_pick_line  = r.spread_open
                r.spread_cover_prob = cover_prob
                r.spread_home_price = ASSUMED_PRICE
                r.spread_away_price = ASSUMED_PRICE
                r.spread_covered    = None
                r.spread_pick_roi   = None

                if margin == cover_target:
                    r.spread_pick_roi = 0.0   # push; spread_covered stays None
                else:
                    r.spread_covered = margin > cover_target   # did HOME cover
                    pick_home    = cover_prob >= 0.5
                    pick_correct = (pick_home == r.spread_covered)
                    profit = ASSUMED_PRICE / 100.0 if ASSUMED_PRICE > 0 else 100.0 / (-ASSUMED_PRICE)
                    r.spread_pick_roi = round(profit if pick_correct else -1.0, 4)
                filled += 1

            totals['filled'] += filled
            print(f'  filled {filled} / {len(rows)}')

            if not dry_run:
                db.session.commit()

    print('\n' + '─' * 58)
    print(f'  Filled: {totals["filled"]}')
    print(f'  Skipped, no spread_open (outside ESPN retention): {totals["no_spread_open"]}')
    print(f'  Skipped, no home_prob: {totals["no_home_prob"]}')
    if dry_run:
        print('\n  Dry run — no writes made. Run without --dry-run to persist.')


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    force = '--force' in sys.argv
    sports = ALL_SPORTS
    if '--sport' in sys.argv:
        idx = sys.argv.index('--sport')
        sports = [s.upper() for s in sys.argv[idx + 1:] if not s.startswith('--')]
    run(sports=sports, dry_run=dry_run, force=force)
