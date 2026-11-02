#!/usr/bin/env python3
"""Combined player projection: which inputs actually improve on the live
project_points() baseline, held out the same way Phase 4 was
(nfl_model_challengers.py): a feature is only worth promoting if it wins on
two independent train -> test splits.

Usage (repo root on PYTHONPATH):
    python3 research/nfl/combined_projection.py DATASET.jsonl NFLVERSE_GAMES.csv [BETS_DB]

DATASET   nfl_playergame_dataset.py output (point-in-time projection + actual)
NFLVERSE  nflverse/nfldata games.csv: closing spread_line (positive = home
          favored) and total_line for every game, keyed by ESPN game id.
          The app's own odds history can't reach back (ESPN drops pickcenter
          after ~300 days), so this is the only full-coverage market source.
BETS_DB   optional: a bets.db copy, for the 2026 sim blend test
          (NflSimResult weeks whose sims ran before kickoff).

Candidates, fit per bucket (QB / RB / REC) by OLS on the train split:
  base     the live projection, unrefit (what the app shows today)
  refit    a + b*projection (separates "just a slope fix" from new signal)
  +total   + market-implied team points
  +scale   + projection * (implied / team recent ppg - 1): the classic
           "this offense should score more than usual" scaling
  +script  + implied points + market team margin (favored teams run more)
"""
import csv
import json
import sqlite3
import sys
from collections import defaultdict

import nfl_model_challengers as ch

BUCKETS = ('QB', 'RB', 'REC')
SPLITS = (('2024-25 -> 2026', {2024, 2025}, {2026}),
          ('2024 -> 2025', {2024}, {2025}),
          ('2025 -> 2024', {2025}, {2024}))


def load(path_ds, path_games):
    market = {}
    for r in csv.DictReader(open(path_games)):
        if r['espn'] and r['spread_line'] and r['total_line']:
            market[r['espn']] = (float(r['spread_line']), float(r['total_line']))
    rows = []
    for line in open(path_ds):
        r = json.loads(line)
        if r.get('projection') is None or r.get('actual') is None or r['bucket'] not in BUCKETS:
            continue
        m = market.get(r['game_id'])
        if not m:
            continue
        spread, total = m
        sign = 1 if r['is_home'] else -1
        r['implied'] = total / 2 + sign * spread / 2
        r['team_margin'] = sign * spread
        tr = r.get('team_recent_ppg')
        r['scale'] = r['projection'] * (r['implied'] / tr - 1) if tr else None
        rows.append(r)
    return rows


MODELS = {
    'refit': lambda r: [1.0, r['projection']],
    '+total': lambda r: [1.0, r['projection'], r['implied']],
    '+scale': lambda r: None if r['scale'] is None else [1.0, r['projection'], r['scale']],
    '+script': lambda r: [1.0, r['projection'], r['implied'], r['team_margin']],
}


def fit(rows, feats):
    xs, ys = [], []
    for r in rows:
        x = feats(r)
        if x is not None:
            xs.append(x)
            ys.append(r['actual'])
    return ch.ols_fit(xs, ys) if len(xs) > 20 else None


def evaluate(rows):
    print('\n== Point estimate: MAE on the test season (lower is better) ==')
    verdict = defaultdict(lambda: defaultdict(list))
    coefs_out = {}
    for label, train_s, test_s in SPLITS:
        print(f'\n  split {label}')
        for b in BUCKETS:
            train = [r for r in rows if r['bucket'] == b and r['season'] in train_s]
            # identical test rows for every model: those with every feature present
            test = [r for r in rows if r['bucket'] == b and r['season'] in test_s
                    and all(f(r) is not None for f in MODELS.values())]
            if not test or not train:
                continue
            base = ch.mae([r['actual'] - r['projection'] for r in test])
            line = f'    {b:3s} n={len(test):5d}  base {base:5.2f}'
            for name, feats in MODELS.items():
                c = fit(train, feats)
                if c is None:
                    continue
                m = ch.mae([r['actual'] - ch.predict(c, feats(r)) for r in test])
                verdict[b][name].append(m < base)
                line += f'  {name} {m:5.2f} ({(m - base) / base * 100:+4.1f}%)'
                if label == SPLITS[0][0]:
                    coefs_out[(b, name)] = c
            print(line)
    print('\n  Wins vs base across splits (promote only if it wins on the first two):')
    for b in BUCKETS:
        print(f'    {b:3s} ' + '  '.join(f'{n}: {"".join("W" if w else "L" for w in ws)}' for n, ws in verdict[b].items()))
    return coefs_out


def ranges(rows, coefs, model='+total'):
    """Residual-based P10/P90: fit residual quantiles by bucket x projection
    tercile on train, check coverage on test."""
    print(f'\n== Ranges from historical misses ({model}), target 80% inside ==')
    feats = MODELS[model]
    for label, train_s, test_s in SPLITS[:2]:
        out = []
        for b in BUCKETS:
            c = fit([r for r in rows if r['bucket'] == b and r['season'] in train_s], feats)
            if c is None:
                continue
            train = [r for r in rows if r['bucket'] == b and r['season'] in train_s]
            test = [r for r in rows if r['bucket'] == b and r['season'] in test_s]
            if not test:
                continue
            preds = sorted(ch.predict(c, feats(r)) for r in train)
            cuts = (preds[len(preds) // 3], preds[2 * len(preds) // 3])
            tier = lambda p: 0 if p < cuts[0] else 1 if p < cuts[1] else 2
            res = defaultdict(list)
            for r in train:
                p = ch.predict(c, feats(r))
                res[tier(p)].append(r['actual'] - p)
            q = {t: (sorted(v)[int(0.1 * len(v))], sorted(v)[int(0.9 * len(v))]) for t, v in res.items()}
            inside = 0
            for r in test:
                p = ch.predict(c, feats(r))
                lo, hi = q[tier(p)]
                inside += p + lo <= r['actual'] <= p + hi
            out.append(f'{b} {inside / len(test) * 100:.0f}%')
        print(f'  {label}: ' + '  '.join(out))


def sim_blend(rows, coefs, db_path):
    """2026 only: does the sim median add anything on top of the combined
    projection? One season, so this can inform but not promote."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    sims = {}
    for gid, pid, med, wk in con.execute('select game_id, player_id, sim_median, week from nfl_sim_results where season=2026'):
        sims[(gid, pid)] = (med, wk)
    weeks_clean = {6, 7, 8}  # see ouija_grading: weeks 1-5 were simmed after the games
    r26 = []
    for r in rows:
        s = sims.get((r['game_id'], r['player_id']))
        if r['season'] == 2026 and s and s[1] in weeks_clean:
            r26.append(dict(r, sim=s[0], week=s[1]))
    print(f'\n== Sim blend, 2026 weeks 6-8 ({len(r26)} player-games, leave-one-week-out) ==')
    for b in BUCKETS:
        c = coefs.get((b, '+total'))
        rb = [r for r in r26 if r['bucket'] == b]
        if not c or len(rb) < 30:
            continue
        res = defaultdict(list)
        for hold in sorted(weeks_clean):
            tr = [r for r in rb if r['week'] != hold]
            te = [r for r in rb if r['week'] == hold]
            # best blend weight on the other two weeks
            best_w = min((w / 10 for w in range(11)), key=lambda w: ch.mae(
                [r['actual'] - (w * r['sim'] + (1 - w) * ch.predict(c, MODELS['+total'](r))) for r in tr]))
            for r in te:
                comb = ch.predict(c, MODELS['+total'](r))
                res['base'].append(r['actual'] - r['projection'])
                res['sim'].append(r['actual'] - r['sim'])
                res['combined'].append(r['actual'] - comb)
                res[f'blend'].append(r['actual'] - (best_w * r['sim'] + (1 - best_w) * comb))
                res['w'].append(best_w)
        print(f'  {b:3s} n={len(rb)}  base {ch.mae(res["base"]):.2f}  sim {ch.mae(res["sim"]):.2f}  '
              f'combined {ch.mae(res["combined"]):.2f}  blend {ch.mae(res["blend"]):.2f} '
              f'(sim weight ~{sum(res["w"]) / len(res["w"]):.1f})')


def main():
    rows = load(sys.argv[1], sys.argv[2])
    print(f'{len(rows)} player-games with a projection, result and market line')
    print('  by season:', dict(sorted(defaultdict(int, {s: sum(1 for r in rows if r["season"] == s) for s in (2024, 2025, 2026)}).items())))
    coefs = evaluate(rows)
    print('\n  Coefficients on 2024-25 (intercept, projection, ...):')
    for (b, name), c in sorted(coefs.items()):
        print(f'    {b:3s} {name:8s} ' + ', '.join(f'{x:+.3f}' for x in c))
    ranges(rows, coefs)
    if len(sys.argv) > 3:
        sim_blend(rows, coefs, sys.argv[3])
    production_fit(rows)


def production_fit(rows):
    """Coefficients + residual bands to paste into nfl_fantasy, fit on every
    season (same convention as PROJECTION_FIT['RB']'s Phase 4 refit)."""
    print('\n== Production fit (all seasons) ==')
    qb = [r for r in rows if r['bucket'] == 'QB']
    c = fit(qb, MODELS['+total'])
    print(f"  QB +total: intercept {c[0]:+.3f}, projection {c[1]:+.3f}, implied {c[2]:+.3f}  (n={len(qb)})")
    pred = {'QB': lambda r: ch.predict(c, MODELS['+total'](r)), 'RB': lambda r: r['projection'], 'REC': lambda r: r['projection']}
    print('  Residual bands (P10, P90 of actual - combined) by projection tier:')
    for b in BUCKETS:
        rb = [r for r in rows if r['bucket'] == b]
        ps = sorted(pred[b](r) for r in rb)
        cuts = (round(ps[len(ps) // 3], 1), round(ps[2 * len(ps) // 3], 1))
        res = defaultdict(list)
        for r in rb:
            p = pred[b](r)
            res[0 if p < cuts[0] else 1 if p < cuts[1] else 2].append(r['actual'] - p)
        bands = [(round(sorted(v)[int(0.1 * len(v))], 1), round(sorted(v)[int(0.9 * len(v))], 1)) for _, v in sorted(res.items())]
        print(f"    {b}: cuts {cuts}  bands {bands}  (n={len(rb)})")


if __name__ == '__main__':
    main()
