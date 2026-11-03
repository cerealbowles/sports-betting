#!/usr/bin/env python3
"""
cbb_backfill.py — Populate game_predictions with completed men's college
basketball games so the Model Performance page, the confidence-blend /
market-shrink calibration loops and the spread-accuracy history have real
CBB data from day one — same role as wnba_backfill.py / nba_backfill.py.

Usage:
    python3 cbb_backfill.py                     # seasons 2025 and 2026 (default)
    python3 cbb_backfill.py --dry-run
    python3 cbb_backfill.py --seasons 2026

Walk-forward, no leakage: each date's games are scored by cbb_model.predict()
on a cbb_rating fit of games strictly before that date, and the tracking
rows' record/form come from that season's earlier games only. D1-vs-D1 games
only (no market prices non-D1 matchups anyway).

Market data: where instance/cbb_market_lines.json (research/cbb/
fetch_market.py — ESPN pickcenter, ~300-day retention) has the game, the
DraftKings closing moneylines and spread open/close are written to the same
columns espn_odds_backfill.py fills. The model probability is NOT shrunk
toward those prices (the live model only shrinks toward the price it sees
pregame, and a closing line wasn't knowable at prediction time).

A settled game's prob/factors_json is written once and then immutable;
reruns only refresh outcome/odds fields. Run backfill_spread_accuracy.py
--sport CBB afterwards to grade spread picks on the backfilled rows.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import cbb_model  # noqa: E402
import cbb_rating as R  # noqa: E402

MARKET_PATH = os.path.join(R._DATA_DIR, 'cbb_market_lines.json')


def score_season(season):
    pool = [g for s in (season - 2, season - 1, season) for g in R.load_season(s)]
    target = R.load_season(season)
    rec = defaultdict(lambda: {'wins': 0, 'losses': 0, 'res': []})
    by_date = defaultdict(list)
    for g in target:
        by_date[g['date']].append(g)
    rows = []
    for d in sorted(by_date):
        try:
            fit = R.fit([g for g in pool if g['date'] < d], season)
        except Exception:
            fit = None
        for g in by_date[d]:
            if R.OTHER_CONF in (g['home_conf'], g['away_conf']):
                continue
            teams = {}
            for side in ('home', 'away'):
                r = rec[g[side]]
                teams[side] = {'name': g[side], 'wins': r['wins'], 'losses': r['losses'],
                               'form': ['W' if w else 'L' for w in reversed(r['res'][-5:])]}
            rating = (R.predict(fit, g['home'], g['away'], g['home_conf'], g['away_conf'], g['neutral'])
                      if fit else None)
            m = cbb_model.predict(teams['home'], teams['away'], rating=rating, neutral=g['neutral'])
            m['factors'].sort(key=lambda f: abs(f[1]), reverse=True)
            rows.append({'event_id': g['id'], 'game_date': g['date'], 'game_time_utc': g['ts'],
                         'home_team': g['home'], 'away_team': g['away'],
                         'home_prob': m['home_prob'], 'away_prob': m['away_prob'],
                         'factors': m['factors'], 'home_score': g['home_score'],
                         'away_score': g['away_score'],
                         'home_won': g['home_score'] > g['away_score']})
        for g in by_date[d]:
            hw = g['home_score'] > g['away_score']
            for team, won in ((g['home'], hw), (g['away'], not hw)):
                rec[team]['wins' if won else 'losses'] += 1
                rec[team]['res'].append(won)
    return rows


def run(dry_run=False, seasons=(2025, 2026)):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    rows = []
    for s in seasons:
        season_rows = score_season(s)
        print(f'season {s}: scored {len(season_rows)} D1-vs-D1 games', flush=True)
        rows.extend(season_rows)
    market = json.load(open(MARKET_PATH)) if os.path.exists(MARKET_PATH) else {}
    acc = sum((r['home_prob'] >= 0.5) == r['home_won'] for r in rows) / max(len(rows), 1)
    print(f'straight-up accuracy: {acc:.1%}; market lines on hand for '
          f'{sum(1 for r in rows if market.get(r["event_id"]))} games')
    if dry_run:
        return

    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, GamePrediction

    now = datetime.now(timezone.utc)
    inserted = updated = 0
    with flask_app.app_context():
        existing = {(p.game_date, p.home_team, p.away_team): p
                    for p in GamePrediction.query.filter_by(sport='CBB').all()}
        for r in rows:
            key = (r['game_date'], r['home_team'], r['away_team'])
            pred = existing.get(key)
            if pred is None:
                pred = GamePrediction(sport='CBB', game_date=r['game_date'],
                                      home_team=r['home_team'], away_team=r['away_team'])
                db.session.add(pred)
                existing[key] = pred
                inserted += 1
                pred.game_time_utc = r['game_time_utc']
                pred.home_prob = r['home_prob']
                pred.away_prob = r['away_prob']
                pred.factors_json = json.dumps(r['factors'])
            else:
                updated += 1
            pred.home_score = r['home_score']
            pred.away_score = r['away_score']
            pred.home_won = r['home_won']
            if pred.outcome_set_at is None:
                pred.outcome_set_at = now
            m = market.get(r['event_id'])
            if m:
                pred.spread_open = m.get('spread_open')
                pred.spread_close = m.get('spread_close')
                try:
                    if m.get('home_moneyline') is not None and pred.home_odds is None:
                        pred.home_odds = int(m['home_moneyline'])
                    if m.get('away_moneyline') is not None and pred.away_odds is None:
                        pred.away_odds = int(m['away_moneyline'])
                except (TypeError, ValueError):
                    pass
        db.session.commit()
    print(f'inserted {inserted}, refreshed {updated}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--seasons', type=int, nargs='*', default=[2025, 2026])
    a = ap.parse_args()
    sys.exit(run(dry_run=a.dry_run, seasons=tuple(a.seasons)))
