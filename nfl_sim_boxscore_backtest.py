#!/usr/bin/env python3
"""
nfl_sim_boxscore_backtest.py — validates nfl_game_simulator.py's predicted
box-score lines (not just fantasy-point totals) against real results.

Read-only, no persistence — same "run it standalone, print a report, touch
nothing" convention as nfl_miss_analysis.py. Deliberately does NOT call
nfl_game_simulator.simulate_week() (which upserts into NflSimResult/
NflSimGame): those two tables are the app's own daily-job ROLLING estimate
for an upcoming week (see their class docstrings in app.py), not a place
to park a backtest's output. This script re-runs simulate_matchup()
in-memory for each game and only ever prints what it finds.

For every already-Final game across the given seasons/weeks, re-runs
simulate_matchup() at that week's own no-leakage cutoff (same point-in-time
discipline as nfl_recap.py/nfl_player_bootstrap.py — before_date=the
week's own earliest game date) and compares:

  - GAME level: the simulated win pick / average score against the real
    final score (the same comparison get_sim_week_accuracy_db() already
    does for the CURRENT week off stored NflSimGame rows — this is that
    same metric, computed fresh, across however much settled history is
    requested).
  - PLAYER level: each attributed player's predicted per-category median
    (sim_stats — passingYards, rushingTouchdowns, receptions, ...)
    against their real box-score line (nfl_stats_db.get_game_boxscore_
    db()). Reports MEAN SIGNED ERROR (bias — is the simulator
    systematically over/under-predicting this category?) and MEAN
    ABSOLUTE ERROR, broken out by position and by season.

KNOWN LIMITATION: this reports bias, it doesn't fit anything. The
simulator's own constants (TEAM_CONDITIONING_K, FOURTH_DOWN_GO_FOR_IT_
RATE, drives_per_team, ...) are hand-set, not fit to data (see that
module's own docstring) — a bias found here is a lead for a human to go
adjust one of those constants, the same way two real drive-loop bugs were
found via an earlier ad-hoc version of this comparison and then fixed in
the simulator itself, not something this script corrects on its own.

A player who never appears in either team's stored box score for a game
(not ingested yet) is skipped for that game rather than counted as a real
0 — same "a real gap beats a wrong guess" convention as the rest of this
app. Position is resolved from nfl_stats_db.get_or_fetch_roster_db()'s
CURRENT-roster snapshot (same staleness caveat nfl_recap.py's own
docstring already states: a player traded/cut since won't reflect it).

Deliberately self-contained beyond nfl_game_simulator.py/nfl_api.py/
nfl_stats_db.get_game_boxscore_db()+get_or_fetch_roster_db() (all three
already stable, unmodified, read-only call paths other live pages also
use) — the one new piece of DB logic this needs (a Final game's real
final score) is implemented locally below (_get_actual_game_score)
rather than added to nfl_stats_db.py, so this script can be dropped onto
a host running an older app.py/nfl_stats_db.py without needing a deploy
first.

Usage:
    python3 nfl_sim_boxscore_backtest.py --seasons 2026 --weeks 1-6 --n 40
    python3 nfl_sim_boxscore_backtest.py --seasons 2024,2025 --n 40
    python3 nfl_sim_boxscore_backtest.py --seasons 2024,2025,2026 --n 30 --max-games 100
"""
import argparse
import random
import sys
from collections import defaultdict

import nfl_api
import nfl_game_simulator as sim
import nfl_stats_db

REG_SEASON_WEEKS = 18


def _get_actual_game_score(game_id):
    """{'home_score', 'away_score'} for one already-Final NFL game, from
    TeamGameStat's two rows (is_home True/False). None if either row
    isn't stored yet. Local copy of the same query app.py's own /nfl/
    simulations route uses (nfl_stats_db.get_actual_game_score_db) — kept
    here rather than imported so this script has no dependency on that
    function existing in whatever app.py/nfl_stats_db.py is deployed."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = TeamGameStat.query.filter_by(sport='NFL', game_id=game_id).all()
    by_home = {r.is_home: r for r in rows}
    home_row, away_row = by_home.get(True), by_home.get(False)
    if home_row is None or away_row is None or home_row.points is None or away_row.points is None:
        return None
    return {'home_score': home_row.points, 'away_score': away_row.points}

# (field, report label) — same field names nfl_game_simulator.
# attribute_plays_to_players() produces, same order nfl_fantasy.
# compute_fantasy_points() reads them in.
STAT_FIELDS = [
    ('passingYards', 'Pass Yds'), ('passingTouchdowns', 'Pass TD'), ('interceptions', 'INT'),
    ('rushingYards', 'Rush Yds'), ('rushingTouchdowns', 'Rush TD'),
    ('receivingYards', 'Rec Yds'), ('receptions', 'Rec'), ('receivingTouchdowns', 'Rec TD'),
    ('fumblesLost', 'Fum Lost'),
]


def _num(val, default=0.0):
    """Same coercion as nfl_fantasy.py's own private _num() — a real box
    score's categories dict can hold composite/string values (e.g. a
    '248/350' passer-rating-style field elsewhere in this app's data);
    anything that doesn't parse as a float falls back to `default` rather
    than crashing the whole backtest over one odd field."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _flatten_categories(cats):
    flat = {}
    for group in (cats or {}).values():
        if isinstance(group, dict):
            flat.update(group)
    return flat


def _week_cutoff(games):
    dates = [g['game_time_utc'][:10] for g in games if g.get('game_time_utc')]
    return min(dates) if dates else None


def _mean(values):
    return sum(values) / len(values) if values else None


def collect(seasons, weeks, n, max_games=None, verbose=False, drives_per_team=10):
    """Returns (game_rows, stat_rows) — see module docstring for shape.
    `max_games` caps the total number of games simulated (across every
    season/week), for a quick smoke run; None (default) processes
    everything in range. `drives_per_team` passes straight through to
    simulate_matchup() — simulate_game()'s own default (10) is this app's
    stored-data average (20.3 distinct drive_ids/game, ~half per side),
    but it's a fixed assumption, not fit to any particular week/season;
    varying it here is how to check whether the box-score bias this
    script finds (see module docstring) tracks drive COUNT rather than
    something else (per-drive yardage, attribution, ...)."""
    pools_cache = {}
    roster_cache = {}
    game_rows = []
    stat_rows = []
    processed = 0

    for season in seasons:
        pools = pools_cache.get(season)
        for week in weeks:
            if max_games is not None and processed >= max_games:
                return game_rows, stat_rows
            week_ctx = nfl_api.build_week_games_light(week, season=season)
            games = [g for g in week_ctx['games'] if g['status'] == 'Final']
            if not games:
                continue
            cutoff = _week_cutoff(week_ctx['games'])
            if pools is None:
                pools = sim.build_sim_pools()
                pools_cache[season] = pools

            for g in games:
                if max_games is not None and processed >= max_games:
                    return game_rows, stat_rows
                home_id, away_id = g['home'].get('id'), g['away'].get('id')
                if not home_id or not away_id:
                    continue
                score = _get_actual_game_score(g['game_id'])
                boxscore = nfl_stats_db.get_game_boxscore_db(g['game_id'])
                if not score or not boxscore:
                    continue  # not ingested — skip rather than guess

                rng = random.Random(f'{season}-{week}-{g["game_id"]}')
                result = sim.simulate_matchup(home_id, away_id, n=n, before_date=cutoff,
                                               rng=rng, pools=pools, drives_per_team=drives_per_team)
                processed += 1
                if verbose:
                    print(f'  {season} wk{week} {g["away"]["abbrev"]}@{g["home"]["abbrev"]} '
                          f'({g["game_id"]}) ... simulated', file=sys.stderr)

                gs = result['game']
                picked_home = gs['home_win_pct'] >= gs['away_win_pct']
                real_home_won = score['home_score'] > score['away_score']
                game_rows.append({
                    'season': season, 'week': week, 'game_id': g['game_id'],
                    'picked_home': picked_home, 'real_home_won': real_home_won,
                    'home_win_pct': gs['home_win_pct'],
                    'home_pred': gs['home_avg_score'], 'home_actual': score['home_score'],
                    'away_pred': gs['away_avg_score'], 'away_actual': score['away_score'],
                })

                for side, team_id in (('home', home_id), ('away', away_id)):
                    if team_id not in roster_cache:
                        roster_cache[team_id] = {
                            p['id']: p.get('position')
                            for p in nfl_stats_db.get_or_fetch_roster_db(team_id)
                        }
                    roster = roster_cache[team_id]
                    box_rows = boxscore.get(team_id, [])
                    actual_by_id = {r['id']: _flatten_categories(r.get('categories')) for r in box_rows}
                    for player_id, outcome in result['players'][side].items():
                        if player_id.startswith('DST_'):
                            continue
                        sim_stats = outcome.get('sim_stats')
                        if not sim_stats:
                            continue
                        actual = actual_by_id.get(player_id)
                        if actual is None:
                            continue  # not in this team's stored box score — unknown, not a real 0
                        position = roster.get(player_id, '?')
                        for field, _ in STAT_FIELDS:
                            if field not in sim_stats:
                                continue
                            pred, actual_val = _num(sim_stats[field]), _num(actual.get(field))
                            if pred == 0.0 and actual_val == 0.0:
                                # attribute_plays_to_players()'s defaultdict factory gives EVERY
                                # attributed player all 8 categories (zeroed) regardless of role —
                                # a RB's untouched passingYards=0 trivially "matches" their real 0
                                # and would otherwise flood this field's n with meaningless
                                # zero-vs-zero agreement, hiding the real signal (see this
                                # script's own session notes). Skip the trivial case; keep it
                                # whenever EITHER side is non-zero, including a real miss (predicted
                                # something, got nothing, or vice versa).
                                continue
                            stat_rows.append({
                                'season': season, 'field': field, 'position': position,
                                'pred': pred, 'actual': actual_val,
                            })
    return game_rows, stat_rows


def report_game_level(game_rows):
    print('=== GAME-LEVEL ACCURACY ===')
    if not game_rows:
        print('  no games with both a stored box score and a stored final score in range')
        return
    by_season = defaultdict(list)
    for r in game_rows:
        by_season[r['season']].append(r)
    by_season['ALL'] = game_rows

    for season in sorted(by_season, key=lambda s: (s != 'ALL', s)):
        rows = by_season[season]
        n = len(rows)
        correct = sum(1 for r in rows if r['picked_home'] == r['real_home_won'])
        brier = _mean([(r['home_win_pct'] / 100 - (1.0 if r['real_home_won'] else 0.0)) ** 2 for r in rows])
        score_err = _mean([abs(r['home_pred'] - r['home_actual']) for r in rows]
                           + [abs(r['away_pred'] - r['away_actual']) for r in rows])
        print(f'  {season}: n={n}  win-pick accuracy={100*correct/n:.1f}%  '
              f'brier={brier:.4f}  mean |score error|={score_err:.1f} pts/team')


def report_stat_level(stat_rows):
    print('\n=== PLAYER BOX-SCORE ACCURACY (predicted median vs. real) ===')
    if not stat_rows:
        print('  no attributed player-stat comparisons in range')
        return
    label_by_field = dict(STAT_FIELDS)

    for field, label in STAT_FIELDS:
        rows = [r for r in stat_rows if r['field'] == field]
        if not rows:
            continue
        bias = _mean([r['actual'] - r['pred'] for r in rows])
        mae = _mean([abs(r['actual'] - r['pred']) for r in rows])
        print(f'\n{label} (n={len(rows)}): bias(actual-pred)={bias:+.2f}  MAE={mae:.2f}')
        by_pos = defaultdict(list)
        for r in rows:
            by_pos[r['position']].append(r)
        for pos in sorted(by_pos, key=lambda p: -len(by_pos[p])):
            pos_rows = by_pos[pos]
            if len(pos_rows) < 5:
                continue
            pos_bias = _mean([r['actual'] - r['pred'] for r in pos_rows])
            pos_mae = _mean([abs(r['actual'] - r['pred']) for r in pos_rows])
            print(f'    {pos:>4} (n={len(pos_rows):>4}): bias={pos_bias:+.2f}  MAE={pos_mae:.2f}')


def _parse_weeks(spec):
    if '-' in spec:
        lo, hi = spec.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(w) for w in spec.split(',')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seasons', default='2026', help='comma-separated seasons, e.g. 2024,2025,2026')
    ap.add_argument('--weeks', default=f'1-{REG_SEASON_WEEKS}', help='week range "1-18" or list "1,2,3"')
    ap.add_argument('--n', type=int, default=40, help='simulated draws per matchup (lower = faster, noisier)')
    ap.add_argument('--max-games', type=int, default=None, help='cap total games simulated (smoke-test runs)')
    ap.add_argument('--drives-per-team', type=int, default=10,
                     help='drives per side per simulated game (simulate_game()\'s own default is 10)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    seasons = [int(s) for s in args.seasons.split(',')]
    weeks = _parse_weeks(args.weeks)

    print(f'Backtesting nfl_game_simulator against seasons={seasons} weeks={weeks} '
          f'n={args.n} drives_per_team={args.drives_per_team}'
          + (f' (max {args.max_games} games)' if args.max_games else ''))
    game_rows, stat_rows = collect(seasons, weeks, args.n, max_games=args.max_games, verbose=args.verbose,
                                    drives_per_team=args.drives_per_team)
    report_game_level(game_rows)
    report_stat_level(stat_rows)


if __name__ == '__main__':
    main()
