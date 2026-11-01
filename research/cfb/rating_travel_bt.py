"""Backtest: two-level CFB rating (conference level + within-conference team
offset) with optional travel features layered on top.

margin = HFA*(not neutral) + (C[conf_h] + t[h]) - (C[conf_a] + t[a]) + travel terms

Travel terms are home-minus-away differences, so a positive coefficient means
"the side that travels more does worse":
  dist   - miles from each team's home stadium to the venue, per 1000 mi
  tz     - time zones crossed (absolute hours)
  east2  - crossed 2+ zones heading east
  early  - kickoff at or before 10am on the team's body clock (home tz)

Walk-forward: each game day is predicted from a fit on earlier games only.
Usage: python3 research/cfb/rating_travel_bt.py
"""
import datetime as dt
import json
import math
import os
import random
import sqlite3
import statistics as st
import sys
from collections import Counter
from zoneinfo import ZoneInfo

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spread_proxy  # noqa: E402

G = sorted(json.load(open(os.path.join(HERE, 'data', 'games.json'))), key=lambda g: g['ts'])
GEO = json.load(open(os.path.join(HERE, 'data', 'geo.json')))
BASE = dict(cap=60, decay=0.25, lt=1.0, lc=1.0, lf=5.0)   # rating params tuned on 2025 earlier
FEATS = ['dist', 'tz', 'east2', 'early']


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a['lat'], a['lon'], b['lat'], b['lon']))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


# Each team's home = the city it most often hosts non-neutral games in.
hosts = {}
for g in G:
    if not g['neutral'] and g.get('city'):
        hosts.setdefault(g['home'], Counter())[f"{g['city']}|{g['state']}"] += 1
HOME = {t: GEO.get(c.most_common(1)[0][0]) for t, c in hosts.items()}


def travel(team, venue, kick_utc):
    """(miles, signed tz shift in hours [+ = went east], body-clock kickoff hour) or None."""
    home = HOME.get(team)
    if not home or not venue:
        return None
    off = lambda tz: kick_utc.astimezone(ZoneInfo(tz)).utcoffset().total_seconds() / 3600
    return (haversine(home, venue), off(venue['tz']) - off(home['tz']),
            kick_utc.astimezone(ZoneInfo(home['tz'])).hour)


for g in G:
    g['m'] = g['hs'] - g['as_']
    venue = GEO.get(f"{g.get('city')}|{g.get('state')}")
    k = dt.datetime.fromisoformat(g['ts'].replace('Z', '+00:00'))
    th, ta = travel(g['home'], venue, k), travel(g['away'], venue, k)
    g['travel_known'] = th is not None and ta is not None
    f = {}
    if g['travel_known']:
        val = lambda t: {'dist': t[0] / 1000, 'tz': abs(t[1]), 'east2': float(t[1] >= 2), 'early': float(t[2] <= 10)}
        vh, va = val(th), val(ta)
        f = {n: va[n] - vh[n] for n in FEATS}   # away minus home: + when away team travels more
    g['f'] = f

teams = sorted({g['home'] for g in G} | {g['away'] for g in G}); TI = {t: i for i, t in enumerate(teams)}
confs = sorted({g['home_conf'] for g in G} | {g['away_conf'] for g in G}); CI = {c: i for i, c in enumerate(confs)}
nT, nC, nF = len(teams), len(confs), len(FEATS)
o_c, o_t, o_f = 1, 1 + nC, 1 + nC + nT
X = np.zeros((len(G), o_f + nF))
for r, g in enumerate(G):
    X[r, 0] = 0 if g['neutral'] else 1
    X[r, o_c + CI[g['home_conf']]] += 1; X[r, o_c + CI[g['away_conf']]] -= 1
    X[r, o_t + TI[g['home']]] += 1; X[r, o_t + TI[g['away']]] -= 1
    for j, n in enumerate(FEATS):
        X[r, o_f + j] = g['f'].get(n, 0.0)
Y = np.array([g['m'] for g in G], float)
SEASON = np.array([g['season'] for g in G]); DATE = np.array([g['date'] for g in G])


def fit(mask, season, feats=(), p=BASE):
    use = np.r_[np.arange(o_f), [o_f + FEATS.index(n) for n in feats]].astype(int)
    Xm = X[mask][:, use]
    y = np.clip(Y[mask], -p['cap'], p['cap']); w = p['decay'] ** (season - SEASON[mask])
    pen = np.r_[0.0, [p['lc']] * nC, [p['lt']] * nT, [p['lf']] * len(feats)]
    beta = np.linalg.solve(Xm.T @ (Xm * w[:, None]) + np.diag(pen), Xm.T @ (w * y))
    full = np.zeros(X.shape[1]); full[use] = beta
    return full


def walk(season, feats=()):
    preds = {}
    for d in sorted(set(DATE[SEASON == season])):
        beta = fit((DATE < d) & (SEASON >= season - 2), season, feats)
        idx = np.where(DATE == d)[0]
        for i, v in zip(idx, X[idx] @ beta):
            preds[i] = v
    return preds


def mae(pr, idx=None):
    keys = pr.keys() if idx is None else [i for i in idx if i in pr]
    return st.mean(abs(pr[i] - Y[i]) for i in keys)


def market_rows():
    """2026 graded games with a live market line, joined to this game list."""
    db = sqlite3.connect(os.path.join(HERE, 'data', 'prod.db'))
    rows = db.execute("""select game_date, home_team, away_team, home_prob, -spread_pick_line
        from game_predictions where sport='CFB' and game_date >= '2026' and home_score is not null
        and spread_pick_line is not null and home_prob > 0 and home_prob < 1""").fetchall()
    byk = {}
    for i, g in enumerate(G):
        if g['season'] == 2026:
            d = dt.date.fromisoformat(g['date'])
            for dd in (d, d - dt.timedelta(days=1)):   # ET date vs UTC date for late kickoffs
                byk[(dd.isoformat(), g['home'], g['away'])] = i
    out = []
    for d, h, a, hp, mk in rows:
        i = byk.get((d, h, a))
        if i is not None:
            out.append(dict(i=i, d=d, mk=mk, mo=spread_proxy.implied_margin('CFB', hp)))
    return out


if __name__ == '__main__':
    known = sum(g['travel_known'] for g in G)
    print(f'{len(G)} games; travel known for {known} ({100 * known / len(G):.0f}%); '
          f'teams with a home location: {sum(v is not None for v in HOME.values())}')

    # 1. Coefficients on all data (equal season weight) with bootstrap 90% CIs.
    allm = np.ones(len(G), bool)
    beta = fit(allm, 2026, FEATS, dict(BASE, decay=1.0))
    random.seed(0); boots = []
    for _ in range(200):
        s = np.zeros(len(G), bool); s[np.random.randint(0, len(G), len(G))] = True
        boots.append(fit(s, 2026, FEATS, dict(BASE, decay=1.0))[o_f:])
    boots = np.sort(np.array(boots), axis=0)
    print('\ntravel coefficients (points to the side that travels LESS, per unit):')
    for j, n in enumerate(FEATS):
        units = {'dist': 'per 1000 mi', 'tz': 'per zone', 'east2': 'if 2+ zones east', 'early': 'if <=10am body clock'}[n]
        print(f'  {n:6s} {beta[o_f + j]:+.2f} {units:22s} 90% CI [{boots[10, j]:+.2f}, {boots[189, j]:+.2f}]')
    print(f'  (home field with travel terms in: {beta[0]:.2f})')

    # 2. Walk-forward accuracy, rating alone vs rating + travel.
    sets = {'rating only': (), '+ dist': ('dist',), '+ dist,tz': ('dist', 'tz'),
            '+ east2,early': ('east2', 'early'), '+ all travel': tuple(FEATS)}
    preds = {}
    for s in (2025, 2026):
        print(f'\n{s} walk-forward average miss (pts):')
        for name, fs in sets.items():
            preds[(s, name)] = walk(s, fs)
            far = [i for i in preds[(s, name)] if G[i]['travel_known'] and abs(G[i]['f']['tz']) >= 2]
            print(f"  {name:14s} all={mae(preds[(s, name)]):.3f}   games w/ 2+ zone gap (n={len(far)})={mae(preds[(s, name)], far):.3f}")

    # 3. 2026 vs the market, and whether the market already prices travel.
    J = market_rows()
    print(f'\n2026 vs market (n={len(J)}):')
    for name in ('rating only', '+ all travel'):
        pr = preds[(2026, name)]
        w = l = 0
        for x in J:
            dd = pr[x['i']] - x['mk']; rr = Y[x['i']] - x['mk']
            if dd and rr:
                w += (rr > 0) == (dd > 0); l += (rr > 0) != (dd > 0)
        print(f"  {name:14s} miss={st.mean(abs(pr[x['i']] - Y[x['i']]) for x in J):.2f}  ATS {w}-{l} ({100 * w / (w + l):.1f}%)")
    print(f"  current model  miss={st.mean(abs(x['mo'] - Y[x['i']]) for x in J):.2f}")
    print(f"  market         miss={st.mean(abs(x['mk'] - Y[x['i']]) for x in J):.2f}")
    K = [x for x in J if G[x['i']]['travel_known']]
    F = np.array([[1] + [G[x['i']]['f'][n] for n in FEATS] for x in K]); R = np.array([Y[x['i']] - x['mk'] for x in K])
    c = np.linalg.lstsq(F, R, rcond=None)[0]
    print(f'  does the market miss travel? (actual - line) on travel feats, n={len(K)}: '
          + ', '.join(f'{n} {c[j + 1]:+.2f}' for j, n in enumerate(FEATS)))
