"""
Prototype: fit one unified ranking score from the same underlying components
that Trust/Sharp/Experimental already use (c=consensus, e=edge, k=calibration,
m=market signal), instead of three hand-set weight splits.

Grid-searches weights (step 0.05, sum to 1) to maximize realistic flat-stake
ROI from betting the top-3 ranked games per day, fit on a date-ordered train
split and validated on a held-out test split. Reports the same #1/#2-3/#4-6/#7+
bucket table used in the Model Performance rank-comparison view.

Usage: python unified_score_fit.py [SPORT]
"""
import sys
from itertools import groupby, product

from app import (app, db, GamePrediction, _trust_score, _line_move_pct)

SPORT = sys.argv[1].upper() if len(sys.argv) > 1 else 'MLB'

RANK_BUCKETS = [('#1', 1, 1), ('#2-3', 2, 3), ('#4-6', 4, 6), ('#7+', 7, 999)]


def load_games():
    with app.app_context():
        preds = GamePrediction.query.filter_by(sport=SPORT).all()
        resolved = [p for p in preds if p.home_won is not None]
        games = [p for p in resolved if p.pick_roi is not None and p.home_odds and p.away_odds]
        rows = []
        for g in games:
            td = _trust_score(g.home_prob, g.home_odds, g.away_odds, g.factors_json, detail=True)
            if not td:
                continue
            lm = _line_move_pct(g.home_prob, g.closing_home_odds or g.home_odds,
                                 g.closing_away_odds or g.away_odds,
                                 g.opening_home_odds, g.opening_away_odds)
            signal_pct = lm if lm is not None else g.pick_clv
            m = max(0.0, min(1.0, 0.5 + signal_pct / 10.0)) if signal_pct is not None else 0.5
            fav_home = (g.home_prob or 0.5) >= 0.5
            rows.append({
                'id': g.id, 'date': g.game_date, 'won': fav_home == bool(g.home_won),
                'roi': g.pick_roi, 'c': td.get('c') or 0, 'e': td.get('e') or 0,
                'k': td.get('k') or 0, 'f': td.get('f') or 0, 'm': m,
            })
        return rows


def score(row, w):
    wc, we, wk, wm = w
    return wc * row['c'] + we * row['e'] + wk * row['k'] + wm * row['m']


def top3_roi(rows, w):
    rows_sorted = sorted(rows, key=lambda r: r['date'])
    profit, n = 0.0, 0
    for _, grp in groupby(rows_sorted, key=lambda r: r['date']):
        day = sorted(grp, key=lambda r: -score(r, w))[:3]
        for r in day:
            profit += r['roi']
            n += 1
    return (profit / n) if n else None, n


def grid_search(rows):
    best_w, best_roi = None, -999
    step = 0.05
    grid = [round(i * step, 2) for i in range(0, int(1 / step) + 1)]
    for wc, we, wk in product(grid, grid, grid):
        wm = round(1 - wc - we - wk, 2)
        if wm < 0 or wm > 1:
            continue
        roi, n = top3_roi(rows, (wc, we, wk, wm))
        if roi is not None and roi > best_roi:
            best_roi, best_w = roi, (wc, we, wk, wm)
    return best_w, best_roi


def bucket_stats(rows, w):
    rows_sorted = sorted(rows, key=lambda r: r['date'])
    ranked = []
    for _, grp in groupby(rows_sorted, key=lambda r: r['date']):
        day = sorted(grp, key=lambda r: -score(r, w))
        for rank, r in enumerate(day, 1):
            ranked.append({**r, 'rank': rank})
    out = []
    for label, lo, hi in RANK_BUCKETS:
        bkt = [r for r in ranked if lo <= r['rank'] <= hi]
        if not bkt:
            out.append({'label': label, 'n': 0, 'wr': None, 'roi': None})
            continue
        wr = round(100 * sum(r['won'] for r in bkt) / len(bkt), 1)
        roi = round(100 * sum(r['roi'] for r in bkt) / len(bkt), 1)
        out.append({'label': label, 'n': len(bkt), 'wr': wr, 'roi': roi})
    return out


def main():
    rows = load_games()
    print(f'{SPORT}: {len(rows)} resolved games with odds + pick_roi\n')

    dates = sorted(set(r['date'] for r in rows))
    split_idx = int(len(dates) * 0.7)
    train_dates = set(dates[:split_idx])
    train = [r for r in rows if r['date'] in train_dates]
    test = [r for r in rows if r['date'] not in train_dates]
    print(f'Train: {len(train)} games ({len(train_dates)} days) | '
          f'Test: {len(test)} games ({len(dates) - split_idx} days)\n')

    best_w, train_roi = grid_search(train)
    print(f'Fitted weights (c, e, k, m) = {best_w}  [train top-3 ROI {train_roi*100:+.1f}%]\n')

    print('-- Held-out test set, bucketed by unified score rank --')
    for row in bucket_stats(test, best_w):
        print(f"  {row['label']:6s} n={row['n']:>4}  "
              f"WR={row['wr']}%  ROI={row['roi']}%" if row['n'] else f"  {row['label']:6s} n=0")

    print('\n-- Full dataset (train+test), bucketed by unified score rank --')
    for row in bucket_stats(rows, best_w):
        print(f"  {row['label']:6s} n={row['n']:>4}  "
              f"WR={row['wr']}%  ROI={row['roi']}%" if row['n'] else f"  {row['label']:6s} n=0")


if __name__ == '__main__':
    main()
