"""
nfl_fantasy.py — Salary-cap lineup builder engine for NFL: FanDuel-standard
fantasy scoring, player/DST point projections fit from this app's own
stats warehouse, a synthetic (self-consistent, NOT real FanDuel) salary
assignment, and an exact lineup optimizer.

Why synthetic salaries: ESPN's free box score/roster endpoints (the only
data source this app has) carry no DFS pricing at all, and no real
historical FanDuel salary feed exists to backfill from. Rather than fake
realism with scraped/guessed numbers, salaries here are openly derived from
this app's own projections — see assign_salaries() below.

Why a real fit instead of hand-picked weights: this mirrors the house
convention nfl_player_model.py already uses for its win-probability model —
nfl_fantasy_bootstrap.py walks every stored game since 2024 point-in-time
(no leakage) and fits PROJECTION_FIT below by linear regression, same as
nfl_player_model.INTERCEPT/SLOPE is fit by nfl_player_bootstrap.py. Re-run
that script and copy its printed constants in here whenever the warehouse
gains a meaningful amount of new data.

Three projection "buckets", not four positions: QB, RB, and REC (WR+TE
combined) — see classify_bucket()'s docstring for why WR and TE share one
regression (FanDuel scores their receiving production identically) even
though they occupy different roster slots. DST is a fourth, separate
bucket, scored at the team level.

Known, documented scoring gaps (ESPN's free box score doesn't carry these):
  - 2-point conversions (no stat key for it at all)
  - Defensive fumble recoveries specifically (the 'fumbles' category's
    fumblesRecovered field doesn't distinguish own-team vs opponent
    recoveries) and safeties/blocked kicks (no stat key for either)
  - Offensive players' fumble-recovery / blocked-kick-return TDs if they
    occur off a defensive takeaway rather than their own fumble
Sacks, interceptions, defensive/return touchdowns, and points allowed ARE
captured — confirmed live against a real box score (see this module's
sibling PR description for the exact category/key names).
"""
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import nfl_player_model as pm
import nfl_roster_api
import nfl_stats_db

_ET = ZoneInfo('America/New_York')

# ── FanDuel standard scoring (publicly published ruleset) ──────────────────
PASS_YD = 0.04      # 1 pt per 25 yards
PASS_TD = 4.0
PASS_INT = -1.0
RUSH_YD = 0.1       # 1 pt per 10 yards
RUSH_TD = 6.0
RECV_YD = 0.1       # 1 pt per 10 yards
RECV_TD = 6.0
RECEPTION = 0.5     # half-PPR
FUMBLE_LOST = -2.0
RETURN_TD = 6.0     # kick/punt return touchdown

DST_SACK = 1.0
DST_INT = 2.0
DST_TD = 6.0        # any defensive touchdown (INT/fumble return)

# FanDuel's points-allowed tiers (sorted ascending ceiling, inclusive).
DST_POINTS_ALLOWED_TIERS = [
    (0, 10.0), (6, 7.0), (13, 4.0), (20, 1.0), (27, 0.0), (34, -1.0), (999, -4.0),
]

ROLLING_WINDOW = 5  # recent games averaged for a player/DST's own baseline — matches nfl_player_model.QB_WINDOW/DEF_WINDOW


def _num(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _split_first(composite, sep='/', default=0.0):
    try:
        return float(str(composite).split(sep)[0])
    except (ValueError, IndexError):
        return default


def compute_fantasy_points(categories):
    """One offensive skill player's FanDuel fantasy points for a single
    game, from the raw `categories` dict PlayerGameStat.stats_json stores
    (nfl_boxscore_api.parse_player_boxscore's per-player shape). Missing
    categories simply contribute 0 — a receiver with no 'rushing' entry
    isn't penalized for not having rushed."""
    pts = 0.0
    cats = categories or {}

    if 'passing' in cats:
        c = cats['passing']
        pts += _num(c.get('passingYards')) * PASS_YD
        pts += _num(c.get('passingTouchdowns')) * PASS_TD
        pts += _num(c.get('interceptions')) * PASS_INT

    if 'rushing' in cats:
        c = cats['rushing']
        pts += _num(c.get('rushingYards')) * RUSH_YD
        pts += _num(c.get('rushingTouchdowns')) * RUSH_TD

    if 'receiving' in cats:
        c = cats['receiving']
        pts += _num(c.get('receivingYards')) * RECV_YD
        pts += _num(c.get('receivingTouchdowns')) * RECV_TD
        pts += _num(c.get('receptions')) * RECEPTION

    if 'fumbles' in cats:
        pts += _num(cats['fumbles'].get('fumblesLost')) * FUMBLE_LOST

    for ret_cat in ('kickReturns', 'puntReturns'):
        if ret_cat in cats:
            key = 'kickReturnTouchdowns' if ret_cat == 'kickReturns' else 'puntReturnTouchdowns'
            pts += _num(cats[ret_cat].get(key)) * RETURN_TD

    return round(pts, 2)


def _points_allowed_bonus(points_allowed):
    if points_allowed is None:
        return 0.0
    for ceiling, bonus in DST_POINTS_ALLOWED_TIERS:
        if points_allowed <= ceiling:
            return bonus
    return DST_POINTS_ALLOWED_TIERS[-1][1]


def aggregate_dst_game(player_rows):
    """Collapses one team's per-player box score rows (same `categories`
    shape as compute_fantasy_points) for ONE game into {sacks, ints,
    def_td} — the defensive counterpart to nfl_player_model.
    parse_team_game_aggregate, extended with defensiveTouchdowns (that
    module's version doesn't need it; DST scoring does)."""
    sacks = ints = def_td = 0.0
    for row in player_rows:
        cats = row.get('categories', {})
        if 'defensive' in cats:
            c = cats['defensive']
            sacks += _num(c.get('sacks'))
            def_td += _num(c.get('defensiveTouchdowns'))
        if 'interceptions' in cats:
            ints += _num(cats['interceptions'].get('interceptions'))
    return {'sacks': sacks, 'ints': ints, 'def_td': def_td}


def compute_dst_fantasy_points(points_allowed, dst_agg):
    """One team's DST FanDuel fantasy points for a single game. `dst_agg`:
    aggregate_dst_game()'s output. `points_allowed`: that team's defense's
    points allowed in this game (TeamGameStat.points_allowed)."""
    pts = dst_agg['sacks'] * DST_SACK + dst_agg['ints'] * DST_INT + dst_agg['def_td'] * DST_TD
    pts += _points_allowed_bonus(points_allowed)
    return round(pts, 2)


def compute_actual_week_points(week_games):
    """week_games: ALL of one (completed) week's games, nfl_api.
    build_week_schedule_context()'s shape — each needs 'game_id' and
    'home'/'away', each of those needing 'id'/'score'. Returns
    (player_actual, dst_actual): {player_id: pts} / {team_id: pts} — the
    REAL FanDuel points everyone who recorded a box score in any of these
    games actually scored, via nfl_stats_db.get_game_boxscore_db(). Used
    by nfl_recap.py to score a past week's lineup against what really
    happened, as opposed to project_points()'s forward-looking estimate.

    A player/team absent from these dicts simply didn't appear in that
    game's stored box score (inactive, didn't suit up that week) — the
    caller should treat that as a real 0, not a missing-data gap, since
    "didn't play" IS the real DFS outcome for a bench/inactive roster
    pick."""
    player_actual, dst_actual = {}, {}
    for game in week_games:
        game_id = game.get('game_id')
        if not game_id:
            continue
        boxscore = nfl_stats_db.get_game_boxscore_db(game_id)
        home, away = game.get('home', {}), game.get('away', {})
        for team, opp in ((home, away), (away, home)):
            team_id = team.get('id')
            if not team_id:
                continue
            rows = boxscore.get(team_id, [])
            for row in rows:
                pid = row.get('id')
                if pid:
                    player_actual[pid] = compute_fantasy_points(row.get('categories', {}))
            try:
                points_allowed = float(opp.get('score'))
            except (TypeError, ValueError):
                points_allowed = None
            dst_actual[team_id] = compute_dst_fantasy_points(points_allowed, aggregate_dst_game(rows))
    return player_actual, dst_actual


# ── Position bucketing from box-score role (NOT roster position) ──────────
#
# Used only by nfl_fantasy_bootstrap.py's historical backtest, where a
# player's CURRENT roster position (nfl_roster_api) may not even resolve
# (traded/cut/retired since 2024) or may not have been their position back
# then. Serving-time position (what roster slot a player fills this week)
# always comes from nfl_roster_api.get_skill_players() instead — this is
# purely "which regression should this game's fantasy output train" bucket.
#
# WR and TE share one 'REC' bucket rather than getting split: FanDuel scores
# their receiving production with the identical formula above (no separate
# TE premium), and nothing in the box score distinguishes a slot receiver
# from an in-line tight end anyway (see nfl_boxscore_api's docstring — no
# position field here at all). Splitting them would just halve both
# buckets' sample size for a distinction FanDuel itself doesn't score.
MIN_PASS_ATT_FOR_QB = 5
MIN_RUSH_ATT_FOR_RB = 3


def classify_bucket(categories):
    """Returns 'QB' | 'RB' | 'REC' | None (no meaningful offensive stat
    line this game — a lineman, a defender, a kicker/punter) for one
    player's single-game categories dict."""
    cats = categories or {}
    pass_att = _split_first(cats.get('passing', {}).get('completions/passingAttempts', '0/0'), '/') \
        if 'passing' in cats else 0.0
    rush_att = _num(cats.get('rushing', {}).get('rushingAttempts')) if 'rushing' in cats else 0.0
    targets = _num(cats.get('receiving', {}).get('receivingTargets')) if 'receiving' in cats else 0.0
    receptions = _num(cats.get('receiving', {}).get('receptions')) if 'receiving' in cats else 0.0

    if pass_att >= MIN_PASS_ATT_FOR_QB:
        return 'QB'
    if rush_att >= max(MIN_RUSH_ATT_FOR_RB, targets, receptions):
        return 'RB'
    if targets > 0 or receptions > 0:
        return 'REC'
    if rush_att > 0:
        return 'RB'
    return None


# ── Projection ───────────────────────────────────────────────────────────
#
# Fit by nfl_fantasy_bootstrap.py, 2024+2025 regular seasons (544 games).
# Each bucket beats the naive "just use the recent average" baseline on
# in-sample MAE (fantasy points/game): QB 6.20 vs 6.75, RB 5.00 vs 5.08,
# REC 4.00 vs 4.07, DST 3.79 vs 4.20 — a real, if modest, lift from the
# opponent-grade term in every bucket (consistent with nfl_player_model.py's
# own experience that opponent quality matters some, not enormously, at
# this app's current sample size).
#
# Fit provenance: this specific run used a fresh ad-hoc ESPN pull into a
# scratch DB (not this app's own production warehouse — see this feature's
# PR description for why), so it's representative of 2024-2025 but NOT
# re-derived from whatever this app's real deployed instance has actually
# accumulated. Re-run nfl_fantasy_bootstrap.py against the real DB_PATH
# periodically and paste its printed PROJECTION_FIT block here.
PROJECTION_FIT = {
    'QB':  {'intercept': 10.6096, 'recent_avg': 0.3958, 'opp_grade': -0.3430},
    'RB':  {'intercept': 3.3537, 'recent_avg': 0.6905, 'opp_grade': -1.1591},
    'REC': {'intercept': 2.6139, 'recent_avg': 0.6573, 'opp_grade': -0.6837},
    'DST': {'intercept': 9.7173, 'recent_avg': 0.2378, 'opp_grade': -5.6024},
}

# Same fit run as PROJECTION_FIT above — n = usable training rows, mae =
# this fit's in-sample mean absolute error (fantasy points/game), baseline_
# mae = the naive "just use the recent average, no opponent term" MAE.
# Kept as structured data (not just a code comment) so templates/
# _nfl_methodology.html can render real numbers instead of hand-typed
# prose that could drift from the code — see projection_fit_summary().
PROJECTION_FIT_QUALITY = {
    'QB':  {'n': 1035, 'mae': 6.20, 'baseline_mae': 6.75},
    'RB':  {'n': 2270, 'mae': 5.00, 'baseline_mae': 5.08},
    'REC': {'n': 6232, 'mae': 4.00, 'baseline_mae': 4.07},
    'DST': {'n': 1022, 'mae': 3.79, 'baseline_mae': 4.20},
}


def projection_fit_summary():
    """PROJECTION_FIT merged with PROJECTION_FIT_QUALITY per bucket — the
    single source templates/_nfl_methodology.html reads, so the
    coefficients and their backtested accuracy can't drift apart from
    each other or from what nfl_fantasy_bootstrap.py actually printed."""
    return {b: {**fit, **PROJECTION_FIT_QUALITY.get(b, {})} for b, fit in PROJECTION_FIT.items()}


ROSTER_BUCKET = {'QB': 'QB', 'RB': 'RB', 'WR': 'REC', 'TE': 'REC'}


def rolling_average(points_history, window=ROLLING_WINDOW):
    """points_history: this player/DST's own past fantasy-point totals,
    newest first. None if empty (no sample to project from — see this
    module's docstring / nfl_fantasy_bootstrap.py's known-limitation note:
    a player with zero stored games, e.g. a true rookie, can't be
    projected and is excluded from the pool entirely rather than guessed)."""
    sample = points_history[:window]
    if not sample:
        return None
    return sum(sample) / len(sample)


def project_points(bucket, points_history, opponent_grade):
    """Projects next game's fantasy points for one player (bucket: 'QB' |
    'RB' | 'REC') or a team DST (bucket: 'DST'). `points_history`: that
    player/DST's own past fantasy-point totals, newest first.
    `opponent_grade`: the matchup's defense grade (for QB/RB/REC — the
    opposing DEFENSE's nfl_player_model.compute_defense_grade) or offense
    grade (for DST — the opposing OFFENSE's compute_offense_grade). None if
    there's no usable history for either side."""
    recent_avg = rolling_average(points_history)
    if recent_avg is None or opponent_grade is None:
        return None
    fit = PROJECTION_FIT[bucket]
    proj = fit['intercept'] + fit['recent_avg'] * recent_avg + fit['opp_grade'] * opponent_grade
    return max(0.0, round(proj, 2))


# ── Synthetic salary ────────────────────────────────────────────────────
#
# Real FanDuel pricing is relative to the week's own slate (a great matchup
# in an otherwise weak RB week prices higher than the same projection would
# in a stacked one) — so rather than an absolute points-to-dollars curve,
# salary here is each player's PERCENTILE RANK within their own position
# group *this week's pool*, mapped onto a realistic-looking band. This is
# still entirely self-consistent/synthetic (no real FanDuel data involved),
# just a better-shaped synthetic than a fixed formula would produce.
SALARY_BANDS = {
    'QB':  (6000, 8800),
    'RB':  (4500, 9200),
    'WR':  (4000, 9200),
    'TE':  (4000, 7200),
    'DST': (2500, 4800),
}
SALARY_STEP = 100


def assign_salaries(pool):
    """pool: [{..., 'position', 'projection'}, ...]. Mutates each entry in
    place, adding 'salary'. Players at the same position with the same
    projection get the same salary (tied percentile) — not an attempt to
    break ties, since there's no real signal to break them with."""
    by_position = {}
    for p in pool:
        by_position.setdefault(p['position'], []).append(p)

    for position, players in by_position.items():
        lo, hi = SALARY_BANDS[position]
        n = len(players)
        if n == 1:
            players[0]['salary'] = int(round((lo + hi) / 2 / SALARY_STEP)) * SALARY_STEP
            continue
        ranked = sorted(players, key=lambda p: p['projection'])
        for i, p in enumerate(ranked):
            pct = i / (n - 1)  # 0 = lowest projection, 1 = highest
            salary = lo + pct * (hi - lo)
            p['salary'] = int(round(salary / SALARY_STEP)) * SALARY_STEP
    return pool


def methodology_context(salary_cap=None, roster_requirements=None):
    """Everything templates/_nfl_methodology.html needs to render its
    'How is this calculated?' panel straight from this module's actual
    constants — the template never hand-types a number, so it can't drift
    out of sync with the code the way a prose explanation would. Shared by
    /nfl/lineup and /nfl/recap (both pass this into their template).

    `salary_cap`/`roster_requirements` default to this module's standard
    constants (what /nfl/recap always uses); /nfl/lineup passes its active
    user-adjusted values so the panel reflects whatever actually produced
    the lineup on screen."""
    return {
        'scoring': {
            'pass_yd': PASS_YD, 'pass_td': PASS_TD, 'pass_int': PASS_INT,
            'rush_yd': RUSH_YD, 'rush_td': RUSH_TD,
            'recv_yd': RECV_YD, 'recv_td': RECV_TD, 'reception': RECEPTION,
            'fumble_lost': FUMBLE_LOST, 'return_td': RETURN_TD,
            'dst_sack': DST_SACK, 'dst_int': DST_INT, 'dst_td': DST_TD,
            'dst_pa_tiers': DST_POINTS_ALLOWED_TIERS,
        },
        'projection_fit': projection_fit_summary(),
        'salary_bands': SALARY_BANDS,
        'salary_cap': salary_cap if salary_cap is not None else SALARY_CAP,
        'roster_requirements': roster_requirements or ROSTER_REQUIREMENTS,
    }


# ── Lineup optimizer ────────────────────────────────────────────────────
#
# FanDuel NFL "Classic" roster: QB, RB, RB, WR, WR, WR, TE, FLEX
# (RB/WR/TE), DST — 9 players, no kicker. Solved as an exact ILP via PuLP
# rather than a hand-rolled knapsack: the FLEX slot's "any of three
# positions" constraint makes this a genuine integer program (not a plain
# knapsack), and the player pools involved here (a handful of games' worth
# of rosters) are tiny for an ILP solver — exact, instant, no custom DP to
# get subtly wrong.
SALARY_CAP = 60000
# 'FLEX' is an explicit count here (not folded into the RB/WR/TE counts)
# so the lineup builder's roster-settings UI can adjust it independently —
# optimize_lineup()/_label_slots() both special-case the 'FLEX' key rather
# than treating it as a real pool position.
ROSTER_REQUIREMENTS = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1}
FLEX_POSITIONS = {'RB', 'WR', 'TE'}

# ── Slate grouping for the lineup builder's game picker ─────────────────
# Buckets a game's ESPN UTC kickoff into the slate groupings DFS players
# think in terms of. Thursday/Friday/Saturday/Monday are whole-day buckets;
# Sunday is split by ET kickoff hour. Hour cutoffs (11/15/18 ET) aren't
# exact to the minute — just enough to split the slate the way real NFL
# kickoff windows fall (9:30am international, 1:00/1:05pm, 4:05/4:25pm,
# ~8:20pm Sunday night, ~8:15pm Monday).
SLOT_ORDER = [
    ('thu',   'Thursday Night'),
    ('fri',   'Friday'),
    ('sat',   'Saturday'),
    ('intl',  'International (Early)'),
    ('early', 'Sunday 1:00 PM'),
    ('late',  'Sunday 4:00 PM'),
    ('snf',   'Sunday Night'),
    ('mnf',   'Monday Night'),
    ('other', 'Other'),
]


def classify_slate_slot(game_time_utc):
    """Buckets a game's ESPN UTC kickoff timestamp (ISO-8601 string) into
    one of SLOT_ORDER's (key, label) pairs. Malformed/missing timestamps
    fall into 'other'."""
    try:
        dt = datetime.fromisoformat(game_time_utc.replace('Z', '+00:00')).astimezone(_ET)
    except (ValueError, AttributeError, TypeError):
        return 'other', dict(SLOT_ORDER)['other']

    weekday = dt.weekday()  # Mon=0 ... Sun=6
    hour = dt.hour + dt.minute / 60
    if weekday == 3:
        key = 'thu'
    elif weekday == 4:
        key = 'fri'
    elif weekday == 5:
        key = 'sat'
    elif weekday == 6:
        if hour < 11:
            key = 'intl'
        elif hour < 15:
            key = 'early'
        elif hour < 18:
            key = 'late'
        else:
            key = 'snf'
    elif weekday == 0:
        key = 'mnf'
    else:
        key = 'other'
    return key, dict(SLOT_ORDER)[key]


def _today_et():
    return datetime.now(_ET).strftime('%Y-%m-%d')


def build_player_pool(games, before_date=None):
    """games: the week's SELECTED games (the ones checked in the UI), each
    the dict shape nfl_api._build_game() returns — needs 'home'/'away',
    each with 'id'/'name'/'abbrev'. Returns one row per eligible
    player/DST: [{'id', 'name', 'team', 'opponent', 'position',
    'projection'}, ...] — 'salary' isn't set yet, see assign_salaries()
    (needs the whole pool built first to rank within each position).

    `before_date` (optional, YYYY-MM-DD): restricts both the injury-status
    snapshot and every player/team's game history to strictly before this
    date — the no-leakage cutoff nfl_recap.py uses to reconstruct what a
    PAST week's pool would genuinely have looked like pre-kickoff. None
    (the default) means "as of today," what the live /nfl/lineup page
    wants. Roster/position still always comes from nfl_roster_api's
    CURRENT-roster snapshot regardless of `before_date` — there's no
    historical roster lookback available from this app's data sources, a
    real fidelity gap for old weeks that's documented, not fixed, by
    nfl_recap.py's caller.

    A player/DST is simply absent from the pool (not included with a
    placeholder) when there's no usable projection for them — no stored
    game history (true rookie), or no stored history yet for their
    opponent's unit grade. See this module's docstring for why that's
    left as a gap rather than papered over with a guess.

    Two network/DB batching concerns this function exists to avoid, both
    confirmed live: (1) a naive per-team/per-player loop issuing its own
    `with flask_app.app_context()` for every single call can run into the
    hundreds across a full week's rosters, which exhausts this app's small
    SQLAlchemy connection pool when the background cache-warmer thread is
    also active — see nfl_stats_db.get_fantasy_pool_data_db()'s docstring,
    the one DB round-trip this function makes. (2) each team's roster is
    its own live ESPN fetch (nfl_roster_api) — fetched in parallel across
    teams, same ThreadPoolExecutor pattern as nba_roster_api.get_active_
    roster, rather than sequentially (confirmed live: ~1-2s per team,
    30+ teams sequentially is an unacceptable page-load wait)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    as_of = before_date or _today_et()
    team_opp = {}  # team_id -> (opp_id, team_dict, opp_dict)
    for game in games:
        home, away = game.get('home', {}), game.get('away', {})
        for team, opp in ((home, away), (away, home)):
            team_id, opp_id = team.get('id'), opp.get('id')
            if team_id and opp_id and team_id not in team_opp:
                team_opp[team_id] = (opp_id, team, opp)

    if not team_opp:
        return []

    roster_by_team = {}
    with ThreadPoolExecutor(max_workers=min(len(team_opp), 12)) as ex:
        futures = {ex.submit(nfl_roster_api.get_skill_players, tid): tid for tid in team_opp}
        for f in as_completed(futures):
            tid = futures[f]
            try:
                roster_by_team[tid] = f.result()
            except Exception:
                roster_by_team[tid] = []

    all_team_ids = set(team_opp) | {opp_id for opp_id, _, _ in team_opp.values()}
    all_player_ids = [p['id'] for players in roster_by_team.values() for p in players]
    data = nfl_stats_db.get_fantasy_pool_data_db(all_team_ids, all_player_ids, as_of, before_date=before_date)
    unavailable, player_games, team_games = data['unavailable'], data['player_games'], data['team_games']

    def _unit_grades(team_id):
        off_list, def_list = [], []
        for g in team_games.get(team_id, []):
            off, defn = pm.parse_team_game_aggregate(g['player_rows'])
            off_list.append(off)
            def_list.append(defn)
        return pm.compute_offense_grade(off_list), pm.compute_defense_grade(def_list)

    pool = []
    for team_id, (opp_id, team, opp) in team_opp.items():
        team_unavailable = unavailable.get(team_id, set())
        opp_off_grade, opp_def_grade = _unit_grades(opp_id)
        team_abbrev = team.get('abbrev') or team.get('abbr') or team.get('name')
        opp_abbrev = opp.get('abbrev') or opp.get('abbr') or opp.get('name')

        for p in roster_by_team.get(team_id, []):
            if p['id'] in team_unavailable:
                continue
            points_hist = [compute_fantasy_points(g['categories']) for g in player_games.get(p['id'], [])]
            proj = project_points(ROSTER_BUCKET[p['position']], points_hist, opp_def_grade)
            if proj is None:
                continue
            pool.append({
                'id': p['id'], 'name': p['name'], 'team': team_abbrev, 'opponent': opp_abbrev,
                'position': p['position'], 'projection': proj,
            })

        dst_points_hist = [compute_dst_fantasy_points(g['points_allowed'], aggregate_dst_game(g['player_rows']))
                            for g in team_games.get(team_id, [])]
        dst_proj = project_points('DST', dst_points_hist, opp_off_grade)
        if dst_proj is not None:
            pool.append({
                'id': f"DST_{team_id}", 'name': f"{team.get('name', '')} D/ST",
                'team': team_abbrev, 'opponent': opp_abbrev,
                'position': 'DST', 'projection': dst_proj,
            })
    return pool


def optimize_lineup(pool, cap=SALARY_CAP, roster_requirements=None, objective='projection'):
    """pool: [{'id', 'name', 'team', 'position', 'salary', 'projection'}, ...]
    (one row per eligible player/DST, already projected+salaried). Returns
    {'lineup': [...], 'total_salary': int, 'total_projection': float} for
    the lineup that maximizes `objective` (a key present on every pool
    entry — 'projection' for the live page's normal "best projected
    lineup," or 'actual' for nfl_recap.py's hindsight "best possible
    lineup" over the SAME pool/salaries) under `cap`, or None if no
    feasible lineup exists (e.g. too few players at some position in the
    selected games).

    `roster_requirements` defaults to the module's standard ROSTER_
    REQUIREMENTS (what nfl_recap.py's two call sites always get, by not
    passing this — recap evaluates every week against the same standard
    roster for consistency). /nfl/lineup passes the user's adjusted counts.

    `total_projection` is always summed from 'projection' regardless of
    `objective` — this is what lets a caller compare "what this lineup was
    projected to score" against what it actually scored even when it was
    chosen to maximize something else. `total_actual` is included only
    when every pool entry carries an 'actual' key (nfl_recap.py's pools
    do; the live page's don't, so this is a no-op there)."""
    import pulp

    if not pool:
        return None

    roster_requirements = roster_requirements or ROSTER_REQUIREMENTS

    prob = pulp.LpProblem('nfl_dfs_lineup', pulp.LpMaximize)
    x = {p['id']: pulp.LpVariable(f"x_{p['id']}", cat='Binary') for p in pool}
    by_id = {p['id']: p for p in pool}

    prob += pulp.lpSum(p[objective] * x[p['id']] for p in pool)
    prob += pulp.lpSum(p['salary'] * x[p['id']] for p in pool) <= cap

    for position, count in roster_requirements.items():
        if position == 'FLEX':  # not a real pool position — handled below
            continue
        players = [p for p in pool if p['position'] == position]
        # QB/DST have no flex-style "extra" slot, so they're pinned exactly
        # at their roster count — RB/WR/TE instead get a >= floor here, with
        # the flex_total constraint below pinning their COMBINED count
        # (including the FLEX slot(s)) exactly. Without this QB/DST
        # distinction, the optimizer is free to roster two QBs or two DSTs
        # in place of a FLEX pick whenever that happens to score better per
        # dollar — confirmed live against a real pool before adding this.
        op = pulp.lpSum(x[p['id']] for p in players)
        if position in FLEX_POSITIONS:
            prob += op >= count
        else:
            prob += op == count

    flex_total = sum(roster_requirements.get(pos, 0) for pos in FLEX_POSITIONS) + roster_requirements.get('FLEX', 0)
    flex_players = [p for p in pool if p['position'] in FLEX_POSITIONS]
    prob += pulp.lpSum(x[p['id']] for p in flex_players) == flex_total

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != 'Optimal':
        return None

    lineup = [by_id[pid] for pid, var in x.items() if var.value() == 1]
    result = {
        'lineup': _label_slots(lineup, roster_requirements),
        'total_salary': sum(p['salary'] for p in lineup),
        'total_projection': round(sum(p['projection'] for p in lineup), 2),
        'cap': cap,
    }
    if all('actual' in p for p in pool):
        result['total_actual'] = round(sum(p.get('actual', 0.0) for p in lineup), 2)
    return result


def _label_slots(lineup, roster_requirements=None):
    """Tags each player in an optimizer-chosen lineup with which roster
    SLOT they fill — QB/RB/WR/TE/DST are unambiguous from roster_
    requirements alone, but the "extra" RB/WR/TE beyond those minimums
    are FLEX, and the optimizer itself has no notion of which ones those
    are (it only knows the aggregate FLEX_POSITIONS count constraint).
    Returns the lineup sorted into QB, RB..., WR..., TE..., FLEX..., DST...
    order for display."""
    roster_requirements = roster_requirements or ROSTER_REQUIREMENTS
    by_position = defaultdict(list)
    for p in lineup:
        by_position[p['position']].append(p)

    ordered = []
    for position in ('QB', 'RB', 'WR', 'TE'):
        for _ in range(roster_requirements.get(position, 0)):
            ordered.append({**by_position[position].pop(0), 'slot': position})
    for position in ('RB', 'WR', 'TE'):  # whatever's left here fills the FLEX slot(s)
        for p in by_position[position]:
            ordered.append({**p, 'slot': 'FLEX'})
    for p in by_position['DST']:
        ordered.append({**p, 'slot': 'DST'})
    return ordered
