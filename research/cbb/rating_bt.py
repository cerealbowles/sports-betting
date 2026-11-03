"""Walk-forward backtest / tuner for cbb_rating.py.

    python3 research/cbb/rating_bt.py tune     # grid on season 2025 (all D1-vs-D1 games)
    python3 research/cbb/rating_bt.py test     # current constants on 2026, incl. vs market

Each game is predicted from a fit on games strictly before its date
(refit every REFIT_EVERY game-days for speed while tuning; 1 for the
final test). Graded: margin MAE on D1-vs-D1 games, and on games with an
ESPN closing line: model MAE vs market MAE and ATS%.
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import cbb_rating as R

MARKET_PATH = os.path.join(R._DATA_DIR, 'cbb_market_lines.json')
ALL = {s: R.load_season(s, allow_fetch=False) for s in (2024, 2025, 2026)}


def walk(season, params=None, refit_every=3):
    pool = [g for s in (season - 2, season - 1, season) for g in ALL.get(s, [])]
    target = [g for g in ALL[season]]
    dates = sorted({g['date'] for g in target})
    out, f = [], None
    for i, d in enumerate(dates):
        if f is None or i % refit_every == 0:
            hist = [g for g in pool if g['date'] < d]
            f = R.fit(hist, season, params)
        for g in target:
            if g['date'] != d or OTHER in (g['home_conf'], g['away_conf']):
                continue
            p = R.predict(f, g['home'], g['away'], g['home_conf'], g['away_conf'], g['neutral'])
            out.append((g, p))
    return out


OTHER = R.OTHER_CONF


def grade(rows, market=None, since=None):
    rows = [(g, p) for g, p in rows if not since or g['date'] >= since]
    err = np.array([p['margin'] - (g['home_score'] - g['away_score']) for g, p in rows])
    res = {'n': len(rows), 'mae': round(float(np.mean(np.abs(err))), 3),
           'bias': round(float(np.mean(err)), 2)}
    if market:
        m_err, mod_err, ats_w, ats_n = [], [], 0, 0
        for g, p in rows:
            line = (market.get(g['id']) or {}).get('spread_close')
            if line is None:
                continue
            actual = g['home_score'] - g['away_score']
            mkt_margin = -line
            m_err.append(abs(mkt_margin - actual))
            mod_err.append(abs(p['margin'] - actual))
            if p['margin'] != mkt_margin and actual != mkt_margin:
                ats_n += 1
                ats_w += (p['margin'] > mkt_margin) == (actual > mkt_margin)
        res.update({'n_mkt': len(m_err), 'model_mae_mkt': round(float(np.mean(mod_err)), 3),
                    'market_mae': round(float(np.mean(m_err)), 3),
                    'ats': f'{ats_w}-{ats_n - ats_w} ({100 * ats_w / max(ats_n, 1):.1f}%)'})
    return res


def _run(params):
    rows = walk(2025, params)
    return params, grade(rows), grade(rows, since='2024-12-01')


def tune():
    grid = [dict(zip(('mov_cap', 'prior_w', 'team_pen', 'half_life'), v)) for v in itertools.product(
        (45, 70, 200), (0.05, 0.1, 0.15), (1.0, 2.0), (None, 60, 120))]
    with Pool(8) as pool:
        results = pool.map(_run, grid)
    results.sort(key=lambda r: r[1]['mae'])
    for p, all_, late in results:
        print(p, 'all', all_, 'Dec+', late)


def test():
    market = json.load(open(MARKET_PATH)) if os.path.exists(MARKET_PATH) else {}
    rows = walk(2026, refit_every=1)
    print('2026 all   ', grade(rows, market))
    print('2026 Nov   ', grade([r for r in rows if r[0]['date'] < '2025-12-01']))
    print('2026 Jan12+', grade(rows, market, since='2026-01-12'))
    json.dump([{'id': g['id'], 'date': g['date'], 'margin': p['margin'], 'total': p['total']}
               for g, p in rows], open(os.path.join(R._DATA_DIR, 'cbb_rating_bt_2026.json'), 'w'))


if __name__ == '__main__':
    {'tune': tune, 'test': test}[sys.argv[1]]()
