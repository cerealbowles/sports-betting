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

Two fidelity gaps this module can't fix with this app's data sources,
documented rather than engineered around (see templates/nfl_recap.html's
caveat text, which states these plainly):
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
"""
import json

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


def compute_week_recap(week_ctx):
    """week_ctx: nfl_api.build_week_schedule_context(week)'s return value
    for an ALREADY-COMPLETE week (every game Final) — uses every game in
    the week, no game-subset picker (that's a live-page-only concept).

    Returns {'is_feasible': False} if the pool can't fill a full lineup
    (expected for a season's earliest tracked week(s), before enough
    rolling history exists). Otherwise returns {'is_feasible': True,
    'cutoff_date', 'recommended': {...optimize_lineup() result, plus
    'actual'...}, 'best_possible': {...}}."""
    week_games = [g for day in week_ctx['days'] for g in day['games']]
    cutoff = _week_cutoff_date(week_ctx)

    pool = nfl_fantasy.build_player_pool(week_games, before_date=cutoff)
    if not pool:
        return {'is_feasible': False}
    nfl_fantasy.assign_salaries(pool)
    _attach_actual_points(pool, week_games)

    recommended = nfl_fantasy.optimize_lineup(pool, objective='projection')
    best_possible = nfl_fantasy.optimize_lineup(pool, objective='actual')
    if recommended is None or best_possible is None:
        return {'is_feasible': False}

    return {
        'is_feasible': True,
        'cutoff_date': cutoff,
        'recommended': recommended,
        'best_possible': best_possible,
    }


def _persist(season, week, recap):
    from app import app as flask_app, db, NflRecapWeek

    with flask_app.app_context():
        existing = NflRecapWeek.query.filter_by(season=season, week=week).first()
        row = existing or NflRecapWeek(season=season, week=week)

        if not recap.get('is_feasible'):
            row.is_feasible = False
        else:
            rec, best = recap['recommended'], recap['best_possible']
            row.is_feasible              = True
            row.cutoff_date              = recap['cutoff_date']
            row.recommended_salary       = rec['total_salary']
            row.recommended_projection   = rec['total_projection']
            row.recommended_actual       = rec.get('total_actual')
            row.recommended_players_json = json.dumps(rec['lineup'])
            row.best_salary              = best['total_salary']
            row.best_actual              = best.get('total_actual')
            row.best_players_json        = json.dumps(best['lineup'])

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
        out['best_possible'] = {
            'total_salary': row.best_salary,
            'total_actual': row.best_actual,
            'lineup': json.loads(row.best_players_json or '[]'),
        }
    return out


def get_or_compute_week_recap(season, week):
    """DB-first: returns the persisted recap dict for (season, week) if
    already computed. Otherwise builds that week's schedule context (this
    also ingests the week's box scores as a side effect — see nfl_api.
    _build_game — which get_game_boxscore_db then reads later in this
    same call), verifies it's actually complete, computes, persists, and
    returns it. Returns None (not persisted) if the week isn't complete
    yet — never writes a permanent row for a week that could still
    change."""
    from app import app as flask_app, NflRecapWeek

    with flask_app.app_context():
        existing = NflRecapWeek.query.filter_by(season=season, week=week).first()
    if existing is not None:
        return _row_to_dict(existing)

    week_ctx = nfl_api.build_week_schedule_context(week)
    if not _is_week_complete(week_ctx):
        return None

    recap = compute_week_recap(week_ctx)
    _persist(season, week, recap)
    recap['season'], recap['week'] = season, week
    return recap
