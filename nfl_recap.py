"""
nfl_recap.py — Season-recap compute/persist layer for the NFL lineup
builder (nfl_fantasy.py): for every already-completed week, reconstructs
what the tool would have recommended BEFORE kickoff (no knowledge of the
future — same point-in-time discipline nfl_player_bootstrap.py/
nfl_fantasy_bootstrap.py already use for their own backtests) alongside
the true best-possible lineup in hindsight, both scored by real results.

Computed ONCE per (season, week) and persisted to app.py's NflRecapWeek —
a past week's outcome never changes once every game is Final, and
building the pool costs a live ESPN roster fetch per team
(nfl_fantasy.build_player_pool()), so recomputing it on every page view
would be wasteful in the same way re-fetching a Final game's score on
every page load already is elsewhere in this app (see nfl_stats_db.py's
module docstring).

Also reconstructs two SIMULATOR-based candidates the same way — Sim
Median and Sim Ceiling, the other two objectives /nfl/lineup offers (see
nfl_fantasy.attach_sim_data()/optimize_lineup()) — so a week's real
result can be compared against all three prediction strategies at once,
not just the regression projection. Three fidelity gaps this module
can't fix with this app's data sources, documented rather than
engineered around (see templates/nfl_recap.html's caveat text, which
states these plainly):
  - Roster/position membership comes from nfl_roster_api's CURRENT-roster
    snapshot regardless of which past week is being recapped — there's no
    historical roster lookback on ESPN's free API. A player since traded
    or cut can't appear in an old week's recap even though they actually
    played; a player since signed can appear even though they weren't
    on the team yet.
  - InjuryStatus only started being recorded recently, so old weeks will
    rarely exclude anyone as confirmed-Out — self-correcting for SCORING
    (a benched/inactive player's real result is the 0 they actually
    scored, via compute_actual_week_points), but not for how realistic
    the 'recommended' pool looks for weeks before that table existed.
  - Sim Median/Sim Ceiling reuse whatever NflSimResult the daily
    background job (nfl_game_simulator._simulate_nfl_week()) last
    computed before that week rolled over to the next one — NOT a
    dedicated same-cutoff backtest the way 'recommended'/'best_possible'
    are (build_player_pool(before_date=cutoff), strict). In practice
    that's very close to pre-game (the job runs daily right up until
    rollover), but it's a real, slightly looser point-in-time guarantee
    than the regression-based candidates get, not engineered around. A
    week the background job never got to simulate (a cron/deploy gap)
    isn't a new failure mode either — attach_sim_data() already falls
    every uncovered player back to its own projection, so Sim Median/
    Sim Ceiling just come out identical to 'recommended' for that week.
"""
import json
import random

import nfl_api
import nfl_fantasy


def _week_cutoff_date(week_ctx):
    """The no-leakage boundary for this week's pool: its own earliest game
    date. Intentionally a single cutoff for the WHOLE week (not per-game)
    — a Thursday game's result must not leak into that same week's Sunday
    projections, matching nfl_player_bootstrap.py's discipline. week_ctx
    ['days'] is already ascending-sorted by nfl_api.build_week_schedule_
    context()."""
    return week_ctx['days'][0]['date']


def _is_week_complete(week_ctx):
    games = [g for day in week_ctx['days'] for g in day['games']]
    return bool(games) and all(g['status'] == 'Final' for g in games)


def _attach_actual_points(pool, week_games):
    """Mutates pool in place, adding an 'actual' key to every entry — the
    real FanDuel points that player/DST scored this week (0.0 if they
    simply don't appear in either dict: a real, not missing, result — see
    compute_actual_week_points()'s docstring)."""
    player_actual, dst_actual = nfl_fantasy.compute_actual_week_points(week_games)
    for p in pool:
        if p['position'] == 'DST':
            team_id = p['id'][len('DST_'):]
            p['actual'] = dst_actual.get(team_id, 0.0)
        else:
            p['actual'] = player_actual.get(p['id'], 0.0)


TOP_PLAYERS_LIMIT = 20  # how many of the week's real top actual-point scorers recap keeps PER POSITION, see compute_week_recap()
TOP_PLAYERS_POSITIONS = ('QB', 'RB', 'WR', 'TE', 'DST')

COMPARISON_LINEUP_COUNT = 150  # matches nfl_game_simulator.DEFAULT_SIM_DRAWS — same sample-size convention


def _generate_comparison_lineups(pool, n=COMPARISON_LINEUP_COUNT, rng=None):
    """Generates `n` alternative PRE-GAME lineups to rank our actual picks
    against — not random/garbage lineups (those would almost always score
    worse than anything built from real projections, making a percentile
    comparison meaningless), and deliberately NOT built from 'actual'
    (that would be look-ahead bias — of course a lineup partly informed by
    the real outcome would look good against it). Instead, each trial
    draws 3 random non-negative weights summing to 1 and blends this
    week's three already-computed PRE-GAME signals per player —
    `projection`, `sim_median`, `sim_ceiling` (same three objectives
    /nfl/lineup itself offers) — into a temporary '_trial_blend' field,
    then runs the exact same nfl_fantasy.optimize_lineup() ILP every other
    recap candidate uses. Each resulting lineup is therefore individually
    defensible (literally optimal under SOME reasonable pre-game weighting
    of the real available evidence), not noise.

    Returns a SORTED list of `n` real total_actual scores (not the full
    lineups — the question this answers is "where in the distribution of
    outcomes do we fall," not "what were the other lineups," so there's no
    reason to persist 150x the JSON a single lineup costs). A pool entry
    missing sim_median/sim_ceiling (shouldn't happen post attach_sim_data,
    which always backfills both) falls back to its own projection, same
    convention attach_sim_data() itself already uses.

    `rng`: an optional random.Random for reproducible trials (e.g. tests);
    defaults to the stdlib `random` module's own functions."""
    rand = rng.random if rng is not None else random.random
    scores = []
    for _ in range(n):
        w1, w2, w3 = rand(), rand(), rand()
        total = w1 + w2 + w3
        w1, w2, w3 = w1 / total, w2 / total, w3 / total
        for p in pool:
            p['_trial_blend'] = (w1 * p['projection']
                                  + w2 * p.get('sim_median', p['projection'])
                                  + w3 * p.get('sim_ceiling', p['projection']))
        lineup = nfl_fantasy.optimize_lineup(pool, objective='_trial_blend')
        if lineup is not None:
            scores.append(lineup.get('total_actual', 0.0))
    scores.sort()
    return scores


def compute_week_recap(week_ctx):
    """week_ctx: nfl_api.build_week_schedule_context(week)'s return value
    for an ALREADY-COMPLETE week (every game Final) — uses every game in
    the week, no game-subset picker (that's a live-page-only concept).

    Returns {'is_feasible': False} if the pool can't fill a full lineup
    (expected for a season's earliest tracked week(s), before enough
    rolling history exists). Otherwise returns {'is_feasible': True,
    'cutoff_date', 'recommended': {...optimize_lineup() result, plus
    'actual'...}, 'sim_median': {...}, 'sim_ceiling': {...},
    'best_possible': {...}, 'top_players_by_position': {...}} — four
    lineup candidates, same pool/salaries/actual scores, each optimized
    for a different objective (see this module's docstring for the
    sim-based candidates' own fidelity caveat), plus the TOP_PLAYERS_
    LIMIT highest real scorers from the WHOLE pool that week (not just
    the 9 players in any one lineup) PER POSITION — a flat cross-position
    ranking is dominated by QBs (who score highest on average, but a
    lineup only ever rosters one), so this groups by position instead:
    {'QB': [...], 'RB': [...], ...}, each list independently ranked and
    capped at TOP_PLAYERS_LIMIT. Lets a caller show where a picked player
    actually ranked against their real positional peers, not just whether
    they made a lineup. Also 'comparison_lineup_scores': COMPARISON_
    LINEUP_COUNT real scores from alternative pre-game lineups (see
    _generate_comparison_lineups()) — lets a caller show what PERCENTILE
    our actual picks landed at among a population of other defensible
    pre-game lineups, not just how they compare to the single hindsight-
    optimal one."""
    week_games = [g for day in week_ctx['days'] for g in day['games']]
    cutoff = _week_cutoff_date(week_ctx)

    pool = nfl_fantasy.build_player_pool(week_games, before_date=cutoff)
    if not pool:
        return {'is_feasible': False}
    nfl_fantasy.assign_salaries(pool)
    _attach_actual_points(pool, week_games)
    nfl_fantasy.attach_sim_data(pool, week_ctx['season'], week_ctx['week'])

    recommended = nfl_fantasy.optimize_lineup(pool, objective='projection')
    sim_median = nfl_fantasy.optimize_lineup(pool, objective='sim_median')
    sim_ceiling = nfl_fantasy.optimize_lineup(pool, objective='sim_ceiling')
    best_possible = nfl_fantasy.optimize_lineup(pool, objective='actual')
    if recommended is None or sim_median is None or sim_ceiling is None or best_possible is None:
        return {'is_feasible': False}

    top_players_by_position = {}
    for position in TOP_PLAYERS_POSITIONS:
        ranked = sorted((p for p in pool if p['position'] == position),
                         key=lambda p: p['actual'], reverse=True)[:TOP_PLAYERS_LIMIT]
        top_players_by_position[position] = [
            {'id': p['id'], 'name': p['name'], 'team': p['team'],
             'opponent': p['opponent'], 'actual': p['actual']}
            for p in ranked
        ]

    comparison_lineup_scores = _generate_comparison_lineups(pool)

    return {
        'is_feasible': True,
        'cutoff_date': cutoff,
        'recommended': recommended,
        'sim_median': sim_median,
        'sim_ceiling': sim_ceiling,
        'best_possible': best_possible,
        'top_players_by_position': top_players_by_position,
        'comparison_lineup_scores': comparison_lineup_scores,
    }


def _persist(season, week, recap):
    from app import app as flask_app, db, NflRecapWeek

    with flask_app.app_context():
        existing = NflRecapWeek.query.filter_by(season=season, week=week).first()
        row = existing or NflRecapWeek(season=season, week=week)

        if not recap.get('is_feasible'):
            row.is_feasible = False
        else:
            rec, sim_med, sim_ceil, best = (recap['recommended'], recap['sim_median'],
                                             recap['sim_ceiling'], recap['best_possible'])
            row.is_feasible              = True
            row.cutoff_date              = recap['cutoff_date']
            row.recommended_salary       = rec['total_salary']
            row.recommended_projection   = rec['total_projection']
            row.recommended_actual       = rec.get('total_actual')
            row.recommended_players_json = json.dumps(rec['lineup'])
            row.sim_median_salary        = sim_med['total_salary']
            row.sim_median_actual        = sim_med.get('total_actual')
            row.sim_median_players_json  = json.dumps(sim_med['lineup'])
            row.sim_ceiling_salary       = sim_ceil['total_salary']
            row.sim_ceiling_actual       = sim_ceil.get('total_actual')
            row.sim_ceiling_players_json = json.dumps(sim_ceil['lineup'])
            row.best_salary              = best['total_salary']
            row.best_actual              = best.get('total_actual')
            row.best_players_json        = json.dumps(best['lineup'])
            row.top_players_by_position_json = json.dumps(recap['top_players_by_position'])
            row.comparison_lineup_scores_json = json.dumps(recap['comparison_lineup_scores'])

        if not existing:
            db.session.add(row)
        db.session.commit()


def _row_to_dict(row):
    out = {
        'season': row.season, 'week': row.week,
        'is_feasible': row.is_feasible, 'cutoff_date': row.cutoff_date,
    }
    if row.is_feasible:
        out['recommended'] = {
            'total_salary': row.recommended_salary,
            'total_projection': row.recommended_projection,
            'total_actual': row.recommended_actual,
            'lineup': json.loads(row.recommended_players_json or '[]'),
        }
        out['sim_median'] = {
            'total_salary': row.sim_median_salary,
            'total_actual': row.sim_median_actual,
            'lineup': json.loads(row.sim_median_players_json or '[]'),
        }
        out['sim_ceiling'] = {
            'total_salary': row.sim_ceiling_salary,
            'total_actual': row.sim_ceiling_actual,
            'lineup': json.loads(row.sim_ceiling_players_json or '[]'),
        }
        out['best_possible'] = {
            'total_salary': row.best_salary,
            'total_actual': row.best_actual,
            'lineup': json.loads(row.best_players_json or '[]'),
        }
        out['top_players_by_position'] = json.loads(row.top_players_by_position_json or '{}')
        out['comparison_lineup_scores'] = json.loads(row.comparison_lineup_scores_json or '[]')
    return out


def get_or_compute_week_recap(season, week):
    """DB-first: returns the persisted recap dict for (season, week) if
    already computed. Otherwise builds that week's schedule context (this
    also ingests the week's box scores as a side effect — see nfl_api.
    _build_game — which get_game_boxscore_db then reads later in this
    same call), verifies it's actually complete, computes, persists, and
    returns it. Returns None (not persisted) if the week isn't complete
    yet — never writes a permanent row for a week that could still
    change.

    A row that's feasible but has sim_median_actual, top_players_by_
    position_json, or comparison_lineup_scores_json still NULL predates
    one of those fields (each added after this app already had
    completed-week recaps persisted) — treated as stale and recomputed+
    overwritten here rather than trusted as-is, the one exception to "a
    past week's outcome never changes once settled" this module's
    docstring otherwise promises. Every other already-settled field
    (recommended/best_possible) comes out identical on recompute; this
    just backfills whichever ones were missing."""
    from app import app as flask_app, NflRecapWeek

    with flask_app.app_context():
        existing = NflRecapWeek.query.filter_by(season=season, week=week).first()
    stale = existing is not None and existing.is_feasible and (
        existing.sim_median_actual is None or existing.top_players_by_position_json is None
        or existing.comparison_lineup_scores_json is None)
    if existing is not None and not stale:
        return _row_to_dict(existing)

    week_ctx = nfl_api.build_week_schedule_context(week)
    if not _is_week_complete(week_ctx):
        return None

    recap = compute_week_recap(week_ctx)
    _persist(season, week, recap)
    recap['season'], recap['week'] = season, week
    return recap
