"""
nfl_value_backtest.py — does our model find real value against FanDuel's
prices? Grades every finished FanDuel slate (uploaded on the Lab page, or a
players-list CSV passed on the command line) against what actually happened.

    python3 nfl_value_backtest.py                       # every stored slate whose games are final
    python3 nfl_value_backtest.py --csv FILE --week 6   # one CSV, not stored (season defaults to current)

For each slate it rebuilds our PRE-GAME pool the way /nfl/recap does (only
data from before the slate's first kickoff), prices it with our synthetic
salaries, matches every player to FanDuel's real price, then reports:

  1. Who ranks players better: rank correlation of actual points with
     FanDuel's salary vs our projection vs our sim median, per position.
     If FanDuel's salary wins, a "value" gap is mostly us being wrong.
  2. Value picks vs the rest: actual FanDuel points per real $1k for players
     the Lab tags Value (FD price >= $500 under ours, sim pts/$1k >= pool
     median), and by price-gap tercile.
  3. Lineups at FanDuel's real prices: built from our projection, our sim
     median, and FanDuel's own FPPG as a naive baseline, scored by actual
     points.

Caveat shared with /nfl/recap: the pool is gated by each team's CURRENT
roster (no historical rosters available), so players traded or cut since
the slate drop out. One slate is a first read, not proof.
"""
import argparse
import os
import sys
from collections import defaultdict

os.environ.setdefault('DISABLE_STARTUP_TASKS', '1')

import nfl_fd_salaries as fds


def _load_slates(args, app_mod):
    if args.csv:
        with open(args.csv, encoding='utf-8', errors='replace') as f:
            sid, rows = fds.parse_players_list(f.read())
        return [(args.season, args.week, sid, rows)]
    out = defaultdict(list)
    for r in app_mod.NflFdSalary.query.all():
        out[(r.season, r.week, r.slate_id)].append(r.as_row())
    return [(s, w, sid, rows) for (s, w, sid), rows in sorted(out.items())]


def _grade_slate(season, week, slate_id, fd_rows):
    import nfl_api
    import nfl_fantasy as nf
    import nfl_recap

    ctx = nfl_api.build_week_schedule_context(week)
    if season and ctx.get('season') != season:
        return f'slate {slate_id}: week {week} context is season {ctx.get("season")}, not {season}; skipped'
    slate_games = {r['game'] for r in fd_rows}
    games, dates = [], []
    for d in ctx['days']:
        for g in d['games']:
            if fds.game_key(g) in slate_games:
                games.append(g)
                dates.append(d['date'])
    if not games:
        return f'slate {slate_id}: none of its games found in week {week}; skipped'
    if any(g['status'] != 'Final' for g in games):
        return f'slate {slate_id}: not every game is final yet; skipped'

    single = len(games) == 1
    pool = nf.build_player_pool(games, before_date=min(dates))
    if single:
        pool = [p for p in pool if p['position'] in nf.SINGLE_GAME_POSITIONS]
        nf.assign_single_game_salaries(pool)
    else:
        nf.assign_salaries(pool)
    nfl_recap._attach_actual_points(pool, games)
    nf.attach_sim_data(pool, ctx['season'], week)

    matched = fds.match_pool(pool, fd_rows)
    rows = []
    for p in pool:
        r = matched.get(p['id'])
        if not r:
            continue
        rows.append({**p, 'model_salary': p['salary'], 'salary': r['salary'], 'fd_fppg': r['fppg'] or 0.0,
                     'price_gap': p['salary'] - r['salary']})
    if len(rows) < 10:
        return f'slate {slate_id}: only {len(rows)} players matched FanDuel names; skipped'

    lines = [f"\n=== Season {ctx['season']} week {week} · slate {slate_id} · {len(games)} game(s) · "
             f"{len(rows)}/{len(pool)} pool players matched to FanDuel ==="]

    # 1. Rank correlation with actual points, per position.
    lines.append('\n1. Rank correlation with actual points (higher = better at ordering players)')
    lines.append(f"   {'pos':<5}{'n':>4}{'FD salary':>11}{'our proj':>10}{'sim med':>9}{'FD FPPG':>9}")
    groups = defaultdict(list)
    for p in rows:
        groups[p['position']].append(p)
    groups['ALL'] = rows
    for pos in ('QB', 'RB', 'WR', 'TE', 'DST', 'ALL'):
        g = groups.get(pos)
        if not g or len(g) < 5:
            continue
        act = [p['actual'] for p in g]
        c = [fds.spearman([p[k] for p in g], act) for k in ('salary', 'projection', 'sim_median', 'fd_fppg')]
        lines.append(f"   {pos:<5}{len(g):>4}" + ''.join(f"{(v if v is not None else float('nan')):>{w}.3f}"
                                                   for v, w in zip(c, (11, 10, 9, 9))))

    # 2. Value picks vs the rest, in actual points per real $1k.
    def ppk(ps):
        sal = sum(p['salary'] for p in ps)
        return round(sum(p['actual'] for p in ps) / (sal / 1000), 2) if sal else None

    skill = [p for p in rows if p['position'] != 'DST']
    for p in skill:
        p['vpk'] = p['sim_median'] / (p['salary'] / 1000)
    vals = sorted(p['vpk'] for p in skill)
    med = vals[len(vals) // 2] if vals else 0
    tagged = [p for p in skill if p['price_gap'] >= 500 and p['vpk'] >= med]
    rest = [p for p in skill if p not in tagged]
    lines.append('\n2. Actual FanDuel points per real $1k (skill players)')
    lines.append(f"   Value-tagged  n={len(tagged):<3} {ppk(tagged)}   " +
                 ', '.join(f"{p['name']} {p['actual']:.1f}@${p['salary']:,}" for p in
                           sorted(tagged, key=lambda p: -p['price_gap'])[:6]))
    lines.append(f"   Everyone else n={len(rest):<3} {ppk(rest)}")
    by_gap = sorted(skill, key=lambda p: p['price_gap'])
    t = len(by_gap) // 3
    if t:
        for label, part in (('FD pricier than us', by_gap[:t]), ('middle', by_gap[t:-t]), ('FD cheaper than us', by_gap[-t:])):
            lines.append(f"   gap tercile: {label:<19} n={len(part):<3} {ppk(part)}")

    # 3. Lineups at FanDuel's real prices.
    lines.append("\n3. Best lineup at FanDuel's real prices, scored by actual points")
    for label, obj in (('our projection', 'projection'), ('our sim median', 'sim_median'),
                       ('FanDuel FPPG (baseline)', 'fd_fppg'), ('hindsight best', 'actual')):
        res = (nf.optimize_single_game(rows, objective=obj) if single
               else nf.optimize_lineup(rows, objective=obj))
        if res:
            lines.append(f"   {label:<24} actual {res.get('total_actual', 0):6.1f}  (${res['total_salary']:,})")
        else:
            lines.append(f"   {label:<24} infeasible with matched players")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', help='FanDuel players-list CSV to grade (not stored)')
    ap.add_argument('--week', type=int, help='NFL week the CSV belongs to (required with --csv)')
    ap.add_argument('--season', type=int, help='season (defaults to the current one)')
    args = ap.parse_args()
    if args.csv and not args.week:
        ap.error('--week is required with --csv')

    import app as app_mod
    with app_mod.app.app_context():
        slates = _load_slates(args, app_mod)
        if not slates:
            print('No FanDuel slates stored yet. Upload one on the Lab page, or pass --csv.')
            return
        for season, week, sid, rows in slates:
            print(_grade_slate(season, week, sid, rows))


if __name__ == '__main__':
    sys.exit(main())
