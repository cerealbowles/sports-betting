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

Known, documented scoring gaps (ESPN's free box score doesn't carry these),
cross-checked 2026-10-14 against FanDuel's own in-app rules panel for a
real contest ($250K Sun NFL Kickoff):
  - 2-point conversions, both thrown (2PC/P) and scored (2PC/S) (no stat
    key for either at all)
  - Defensive fumble recoveries specifically (the 'fumbles' category's
    fumblesRecovered field doesn't distinguish own-team vs opponent
    recoveries), safeties, and blocked punts/kicks (no stat key for any)
  - Offensive players' fumble-recovery / blocked-kick-return TDs if they
    occur off a defensive takeaway rather than their own fumble
Sacks, interceptions, defensive/return touchdowns, and points allowed ARE
captured — confirmed live against a real box score (see this module's
sibling PR description for the exact category/key names). The 100+
rushing/receiving and 300+ passing single-game yardage bonuses (3 pts
each) are also captured — see compute_fantasy_points() — added once the
same rules panel cross-check showed they'd been omitted even though the
underlying yardage was already on hand.
"""
import random
from collections import defaultdict
from datetime import datetime, timedelta
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

# Game-level yardage bonuses (confirmed against FanDuel's own rules panel,
# a real contest screenshot) — these stack on top of the per-yard rate
# above once a player clears the threshold IN A SINGLE GAME. Unlike the
# 2-point-conversion/safety/blocked-kick gaps this module's docstring
# documents (no ESPN stat key exists for those at all), the raw yardage
# these read off of (passingYards/rushingYards/receivingYards) is already
# captured per game — this was a pure scoring omission, not a data gap.
PASS_YD_300_BONUS = 3.0
RUSH_YD_100_BONUS = 3.0
RECV_YD_100_BONUS = 3.0

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
        pass_yards = _num(c.get('passingYards'))
        pts += pass_yards * PASS_YD
        pts += _num(c.get('passingTouchdowns')) * PASS_TD
        pts += _num(c.get('interceptions')) * PASS_INT
        if pass_yards >= 300:
            pts += PASS_YD_300_BONUS

    if 'rushing' in cats:
        c = cats['rushing']
        rush_yards = _num(c.get('rushingYards'))
        pts += rush_yards * RUSH_YD
        pts += _num(c.get('rushingTouchdowns')) * RUSH_TD
        if rush_yards >= 100:
            pts += RUSH_YD_100_BONUS

    if 'receiving' in cats:
        c = cats['receiving']
        recv_yards = _num(c.get('receivingYards'))
        pts += recv_yards * RECV_YD
        pts += _num(c.get('receivingTouchdowns')) * RECV_TD
        pts += _num(c.get('receptions')) * RECEPTION
        if recv_yards >= 100:
            pts += RECV_YD_100_BONUS

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
MIN_TARGETS_FOR_USAGE = 2

# Pool-eligibility gate — separate from classify_bucket() above, which only
# decides which regression a game's output trains. A player can clear
# classify_bucket (e.g. one garbage-time catch) while never having a real
# role: PROJECTION_FIT's flat intercept term (REC ~2.6, RB ~3.3) means ANY
# player with ANY stored game history gets a nontrivial projected floor
# regardless of whether they're actually expected to touch the ball, which
# in turn makes them "cheap" (assign_salaries() prices purely off that
# inflated projection) — so the optimizer happily parks cap space on them
# as zero-value filler. Confirmed live: Derius Davis/Malachi Corley (return
# specialists/WR4s averaging 0-1 targets/game) still produced ~2 pt
# projections and $4K salaries purely from the intercept. Fixing this in
# build_player_pool (excluding true zero-role players from the pool
# entirely) rather than bolting a floor onto optimize_lineup, so the
# optimizer keeps doing one honest thing — maximize projected points over
# whoever is genuinely eligible — instead of masking a bad pool with a
# second, competing objective.
MIN_USAGE_GAMES = 2


def _had_meaningful_usage(categories, bucket):
    """One game's categories dict + this player's projection bucket →
    whether they had a real on-field role that game (not just an
    incidental stat line). QB: enough pass attempts to have actually
    started/played; RB: real carries, or enough targets to be a legit
    receiving back; REC: enough targets to be a real target earner, or
    real carries (some WRs get jet-sweep/gadget carries)."""
    cats = categories or {}
    if bucket == 'QB':
        pass_att = _split_first(cats.get('passing', {}).get('completions/passingAttempts', '0/0'), '/') \
            if 'passing' in cats else 0.0
        return pass_att >= MIN_PASS_ATT_FOR_QB
    rush_att = _num(cats.get('rushing', {}).get('rushingAttempts')) if 'rushing' in cats else 0.0
    targets = _num(cats.get('receiving', {}).get('receivingTargets')) if 'receiving' in cats else 0.0
    if bucket == 'RB':
        return rush_att >= MIN_RUSH_ATT_FOR_RB or targets >= MIN_TARGETS_FOR_USAGE
    if bucket == 'REC':
        return targets >= MIN_TARGETS_FOR_USAGE or rush_att >= MIN_RUSH_ATT_FOR_RB
    return True


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
    # RB carries a 3rd term, 'volume' (recent rush attempts + targets per
    # game — see _recent_touches_per_game()) — the ONE challenger feature
    # nfl_model_challengers.py (Phase 4, NFL_Fantasy_Analytics_Projection_
    # Plan.md) found that actually improved out-of-sample RB accuracy, and
    # the only one of three buckets x three challengers tested where the
    # gain held up on a SECOND, independent holdout split (train 2024 ->
    # holdout 2025) after looking promising on the first (train 2024-25 ->
    # holdout 2026) — QB and REC's own volume challengers looked good on
    # the first split and flipped to a loss on the second, so they were
    # NOT promoted and keep the original 2-input formula. Fit on this
    # app's own real stored warehouse (all RB player-games, 2024-2026,
    # n=2633) via nfl_model_challengers.ols_fit() — unlike the other three
    # buckets' coefficients below, which (per this dict's own long-
    # standing caveat) came from an ad-hoc scratch-DB pull, never
    # re-derived from the real deployed data.
    'RB':  {'intercept': 2.3624, 'recent_avg': 0.2706, 'opp_grade': -1.2486, 'volume': 0.4293},
    'REC': {'intercept': 2.6139, 'recent_avg': 0.6573, 'opp_grade': -0.6837},
    'DST': {'intercept': 9.7173, 'recent_avg': 0.2378, 'opp_grade': -5.6024},
}

# Same fit run as PROJECTION_FIT above — n = usable training rows, mae =
# this fit's in-sample mean absolute error (fantasy points/game), baseline_
# mae = the naive "just use the recent average, no opponent term" MAE.
# Kept as structured data (not just a code comment) so templates/
# _nfl_methodology.html can render real numbers instead of hand-typed
# prose that could drift from the code — see projection_fit_summary().
# RB's mae/n reflect the volume-augmented fit above (in-sample, full
# 2024-2026 warehouse); baseline_mae is unchanged — it's still the
# simpler "recent average only, no opponent term" comparison, unrelated
# to the volume addition.
PROJECTION_FIT_QUALITY = {
    'QB':  {'n': 1035, 'mae': 6.20, 'baseline_mae': 6.75},
    'RB':  {'n': 2633, 'mae': 4.83, 'baseline_mae': 5.08},
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


def rolling_stddev(points_history, window=ROLLING_WINDOW):
    """Sample standard deviation (ddof=1 — the honest small-sample
    estimate, not the population formula) of this player/DST's own last
    `window` fantasy-point totals, same window project_points() averages
    over so "consistent" and "the average it's consistent around" are
    talking about the same games. None if fewer than 2 games are stored —
    a single game has no spread to measure, which is a real gap (same
    "no sample" convention as rolling_average()), not zero volatility.

    This is a WEEK-TO-WEEK consistency measure from the player's own
    history — a low number means their recent fantasy output clusters
    tightly, a high number means boom/bust. It's a different thing from
    the per-game simulated spread attach_sim_data() exposes (sim_min/
    sim_median/sim_ceiling): that's "how uncertain is THIS week's single
    outcome, given this matchup," shared across every player in a
    position bucket via that bucket's backtested MAE; this is "has this
    SPECIFIC player's real output actually been steady or spiky," which
    the simulator has no equivalent of."""
    sample = points_history[:window]
    if len(sample) < 2:
        return None
    mean = sum(sample) / len(sample)
    variance = sum((x - mean) ** 2 for x in sample) / (len(sample) - 1)
    return variance ** 0.5


# Subtracted from a player's projection, scaled by their own rolling_stddev,
# to produce 'risk_adjusted' — a consistency-favoring alternative to raw
# projection for optimize_lineup(objective='risk_adjusted'). Originally
# shipped as a single flat 0.35 guess; backtested 2026-10-15 the same way
# PROJECTION_FIT/SIM_MEDIAN_PERCENTILE_OVERRIDE were (grid-search k on
# TRAIN = 2024-2025 player-game rows, confirm the gain holds on the 2026
# HOLDOUT the fit never saw — nfl_model_challengers.py's own methodology).
# Result: a flat 0.35 was wrong for 2 of 3 buckets.
#   QB:  corr(stddev, |error|) ~= 0.006 (no signal). Train best-fit k=0.00.
#        Holdout MAE: k=0 -> 6.335, k=0.35 -> 6.761 (the old flat value
#        made QB picks WORSE, not safer). -> 0.0, no penalty.
#   RB:  same story, worse. Train best-fit k=0.00. Holdout MAE: k=0 ->
#        6.050, k=0.35 -> 7.240. -> 0.0, no penalty.
#   REC (WR/TE): the one bucket with a real, holdout-confirmed effect.
#        Train best-fit k=0.25. Holdout MAE: k=0 -> 4.336, k=0.25 -> 4.213
#        (gain holds), k=0.35 -> 4.223 (close behind, slightly over-tuned).
#        -> 0.25.
#   DST: NOT covered by this backtest (nfl_playergame_dataset.py's dataset
#        is player-game grain only, DST is a team-level projection with no
#        parallel dataset — see that module's own docstring). Given the
#        guessed 0.35 turned out wrong for 2 of the 3 buckets that WERE
#        tested, defaulting an untested bucket to the same guess would be
#        repeating the mistake rather than learning from it — 0.0 (no
#        penalty) is the honest default until DST gets its own backtest.
# Revisit any of these the same way (train/holdout) if future data suggests
# a different split is worth re-checking — these aren't final, just the
# best-supported numbers as of the one backtest run so far.
RISK_PENALTY_K = {'QB': 0.0, 'RB': 0.0, 'REC': 0.25, 'DST': 0.0}


RECENT_VOLUME_WINDOW = 3  # matches nfl_model_challengers.py / nfl_playergame_dataset.py's
                          # own "recent" window — the fit above was validated at this window,
                          # not ROLLING_WINDOW, so changing this would silently invalidate it.


def _recent_touches_per_game(games, window=RECENT_VOLUME_WINDOW):
    """RB-only pregame feature: average (rush attempts + targets) over
    this player's last `window` games (fewer if not available yet) —
    see PROJECTION_FIT['RB']'s comment for where this coefficient came
    from. `games`: newest-first, each {'categories': {...}} (the same
    raw box-score shape build_player_pool() already has in hand — no
    extra DB read). None if there's no game to compute it from."""
    sample = games[:window]
    if not sample:
        return None
    total = 0.0
    for g in sample:
        cats = g.get('categories', {})
        total += _num(cats.get('rushing', {}).get('rushingAttempts'))
        total += _num(cats.get('receiving', {}).get('receivingTargets'))
    return total / len(sample)


def project_points(bucket, points_history, opponent_grade, recent_volume_pg=None):
    """Projects next game's fantasy points for one player (bucket: 'QB' |
    'RB' | 'REC') or a team DST (bucket: 'DST'). `points_history`: that
    player/DST's own past fantasy-point totals, newest first.
    `opponent_grade`: the matchup's defense grade (for QB/RB/REC — the
    opposing DEFENSE's nfl_player_model.compute_defense_grade) or offense
    grade (for DST — the opposing OFFENSE's compute_offense_grade). None if
    there's no usable history for either side.

    `recent_volume_pg` (RB only — see _recent_touches_per_game()): adds
    PROJECTION_FIT['RB']['volume'] * recent_volume_pg to the projection
    when both this value and that coefficient are present. None (the
    default) just skips the term rather than blocking the projection —
    a player missing this one feature still gets the base 2-input
    projection, the same "a real gap beats a wrong guess" convention as
    the rest of this app, not a reason to exclude them from the pool."""
    recent_avg = rolling_average(points_history)
    if recent_avg is None or opponent_grade is None:
        return None
    fit = PROJECTION_FIT[bucket]
    proj = fit['intercept'] + fit['recent_avg'] * recent_avg + fit['opp_grade'] * opponent_grade
    if recent_volume_pg is not None and 'volume' in fit:
        proj += fit['volume'] * recent_volume_pg
    return max(0.0, round(proj, 2))


# ── Monte Carlo outcome simulation (independent per player) ────────────
#
# project_points() above gives one number — the expected value. This
# section answers a different question: what does the realistic SPREAD
# around that number look like for a single game? Each draw samples a
# Gaussian centered on the player's own projection, with spread taken
# from that bucket's own backtested MAE (PROJECTION_FIT_QUALITY — the
# same number the methodology panel already shows as "how far off the
# fitted model typically lands") converted to a standard deviation via
# the standard mean-absolute-deviation/sigma relationship for a normal
# distribution: MAD = sigma * sqrt(2/pi). Deliberately NOT a second,
# separately-tuned volatility knob — reusing the one number this app has
# already validated keeps simulated spread honest instead of invented.
#
# Independent per player: this does NOT model game-script correlation
# (a QB's big passing day pulling his WR1's outcome up with him, or a
# defensive-TD game depressing the opposing offense). Two teammates'
# simulated draws are statistically unrelated here even though their
# real outcomes obviously aren't — treat these as "how volatile is THIS
# player's own game," not as a way to evaluate whether a stacked lineup's
# combined outcomes are correlated. Fixing that needs simulating the
# game's shared script first and deriving player outcomes from it, which
# is a separate, bigger piece of work than this.
_MAE_TO_SIGMA = (3.14159265358979 / 2) ** 0.5  # sqrt(pi/2) ≈ 1.2533, MAD -> Gaussian sigma


def simulate_player_points(bucket, projection, n=10, rng=None):
    """Draws `n` independent Monte Carlo outcomes for one player/DST's next
    game, centered on `projection` (project_points()'s own point estimate)
    with spread from `bucket`'s backtested MAE. Points can't go negative in
    FanDuel's own scoring (see compute_fantasy_points' max(0.0, ...) floor
    on the real formula), so draws are clamped the same way. `rng`: an
    optional random.Random for reproducible draws (e.g. tests); defaults to
    the stdlib `random` module's own functions."""
    sigma = PROJECTION_FIT_QUALITY[bucket]['mae'] * _MAE_TO_SIGMA
    gauss = rng.gauss if rng is not None else random.gauss
    return [max(0.0, round(gauss(projection, sigma), 2)) for _ in range(n)]


def simulate_pool(pool, n=10, rng=None):
    """Mutates every entry of an already-built, already-projected `pool`
    (see build_player_pool()) in place, adding 'sim' (the n raw draws),
    and 'sim_min'/'sim_median'/'sim_max' summarizing them. Named min/
    median/max rather than floor/ceiling percentiles on purpose — with
    n=10 draws, claiming a specific percentile (p10/p90) would overstate
    the precision 10 samples actually support."""
    for p in pool:
        bucket = 'DST' if p['position'] == 'DST' else ROSTER_BUCKET[p['position']]
        draws = simulate_player_points(bucket, p['projection'], n=n, rng=rng)
        ranked = sorted(draws)
        p['sim'] = draws
        p['sim_min'] = ranked[0]
        p['sim_median'] = ranked[len(ranked) // 2]
        p['sim_max'] = ranked[-1]
    return pool


# ── Game-simulator-informed objectives ──────────────────────────────────
#
# nfl_game_simulator.py (a separate module) runs a validated, calibrated
# Monte Carlo simulation (150 drive-by-drive draws/game, resampling real
# historical plays rather than drawing from a distribution — see its own
# module docstring) daily for the current week, persisted to the
# NflSimResult table. Unlike simulate_pool()/simulate_player_points()
# above (independent per-player Gaussian, zero callers anywhere in this
# codebase as of this writing), the game simulator's draws ARE correlated
# across teammates in the same simulated game — draw index i for a QB and
# his WR1 come from the same simulated drive sequence, so a big passing
# game and a big receiving game show up together the way they really do.
# attach_sim_data()/evaluate_lineup_distribution() below are what make use
# of that correlation for the lineup builder; nothing here duplicates or
# replaces the independent-Gaussian functions above.
SIM_CEILING_PERCENTILE = 0.80

# Sim Median was found (2026-10-15 backtest, docs/nfl_game_simulator_
# validation.md) to run systematically low vs. real outcomes at every
# position even after fixing the two drive-loop bugs that caused it. Rather
# than trust the literal 50th percentile of each player's 150 draws, fit
# the percentile that actually minimizes MAE against real results, same
# train/holdout discipline as nfl_model_challengers.py (fit on 2024, verify
# on a 2025 holdout before trusting it — a bucket whose "improvement"
# doesn't replicate gets left alone, not shipped). Only RB showed a real,
# holdout-confirmed effect (p50 -> p39 MAE 5.45 -> 5.35 on the SAME 2025
# games the fit never saw): RB fantasy-point distributions are skewed
# enough that a lower percentile of the simulator's own draws tracks the
# real typical outcome better than the literal median, even though it
# makes the raw signed bias worse (MAE and bias can move in opposite
# directions on a skewed distribution — minimizing one doesn't minimize
# the other). QB's own best-fit percentile (0.53) didn't hold up on
# holdout (correctly rejected); WR/TE's (0.49) was statistically identical
# to 0.50 (no real effect to apply). Buckets not listed here use the plain
# 50th percentile — this dict only overrides where a real, validated
# effect was found.
SIM_MEDIAN_PERCENTILE_OVERRIDE = {'RB': 0.39}


def _sim_rows_by_player(season, week):
    """{player_id: row} flattened from nfl_stats_db.get_sim_results_db()'s
    {game_id: {team_id: [rows]}} shape, with 'game_id' added onto each row
    (the one piece of context that shape only carries at the outer-dict
    level). Shared by attach_sim_data()/evaluate_lineup_distribution()."""
    by_player = {}
    for game_id, teams in nfl_stats_db.get_sim_results_db(season, week).items():
        for players in teams.values():
            for row in players:
                by_player[row['player_id']] = dict(row, game_id=game_id)
    return by_player


# (boom at/above, bust below) fantasy points per position — the Lab page's
# boom/bust columns and the stack card's "both boom" figure.
BOOM_BUST_THRESHOLDS = {'QB': (20, 5), 'RB': (20, 5), 'WR': (20, 5), 'TE': (15, 5), 'DST': (10, 2)}


def attach_sim_data(pool, season, week):
    """Mutates every entry of an already-built `pool` (see
    build_player_pool()) in place, adding 'sim_median', 'sim_mean',
    'sim_min', 'sim_max', 'sim_ceiling' (the SIM_CEILING_PERCENTILE-th
    percentile of that player's own 150 stored draws — well-supported at
    n=150, unlike simulate_pool()'s old n=10 Gaussian which deliberately
    avoided naming a specific percentile) and 'game_id', sourced from
    nfl_game_simulator.py's daily run via NflSimResult (the same table
    /nfl/simulations already reads — no new DB code here).

    A pool entry the simulator has no row for — a skill player or DST the
    daily 7:30am ET job hasn't caught yet (a just-added game, a restart
    before today's run) — gets its sim_median/sim_mean/sim_ceiling set
    equal to its own 'projection' instead, and no 'game_id'. This fallback
    is required, not cosmetic: optimize_lineup() sums a bare
    `p[objective]` per pool entry, and DST is pinned to an exact roster
    count, so it must stay selectable under every objective or the ILP
    goes infeasible. Callers that display these values should treat a
    missing 'game_id' as "this is a fallback, not a real simulated figure"
    (see templates/nfl_lineup.html).

    'sim_median' is the SIM_MEDIAN_PERCENTILE_OVERRIDE-th percentile of
    that player's draws for positions in that dict (currently just RB —
    see its own comment for why), the literal 50th for everyone else —
    not necessarily the actual median despite the name, which stays
    because every caller (optimize_lineup's objective name, the UI label)
    already uses it and a rename would be pure churn for no behavior
    change."""
    by_player = _sim_rows_by_player(season, week)

    for p in pool:
        row = by_player.get(p['id'])
        if row is None or not row.get('sim'):
            p['sim_median'] = p['sim_mean'] = p['sim_ceiling'] = p['sim_floor'] = p['projection']
            continue
        draws = sorted(row['sim'])
        boom_at, bust_at = BOOM_BUST_THRESHOLDS.get(p['position'], (20, 5))
        p['sim_floor'] = round(draws[int(round(0.10 * (len(draws) - 1)))], 2)
        p['boom_pct'] = round(sum(1 for d in draws if d >= boom_at) / len(draws) * 100)
        p['bust_pct'] = round(sum(1 for d in draws if d < bust_at) / len(draws) * 100)
        ceil_idx = min(len(draws) - 1, int(round(SIM_CEILING_PERCENTILE * (len(draws) - 1))))
        median_pct = SIM_MEDIAN_PERCENTILE_OVERRIDE.get(p['position'], 0.5)
        median_idx = min(len(draws) - 1, int(round(median_pct * (len(draws) - 1))))
        p['sim_median'] = round(draws[median_idx], 2) if median_pct != 0.5 else row['sim_median']
        p['sim_mean'] = row['sim_mean']
        p['sim_min'] = row['sim_min']
        p['sim_max'] = row['sim_max']
        p['sim_ceiling'] = round(draws[ceil_idx], 2)
        p['game_id'] = row['game_id']
    return pool


def evaluate_lineup_distribution(lineup, season, week):
    """lineup: optimize_lineup()'s own 'lineup' list (each entry needs 'id'
    and 'projection' — meant to be called on the SAME pool entries
    attach_sim_data() already touched, not a fresh lookup). Returns the
    TRUE joint distribution of this specific lineup's combined fantasy
    score, honoring the game simulator's within-game correlation (see
    attach_sim_data()'s docstring) rather than summing each player's own
    marginal sim_median/sim_ceiling independently, which would understate
    a stacked QB+WR1 lineup's real combined ceiling.

    Mechanism: every stored player carries 150 raw draws; draw index i for
    every player in the SAME game_id is the same simulated game instance
    (paired/correlated). Lineup entries are grouped by game_id, summed
    elementwise within each game, then the resulting per-game totals are
    summed elementwise across games — different games' outcomes are
    treated as independent (no cross-game correlation exists in this
    data). A lineup entry with no stored draws (DST, or a player the
    simulator hasn't run for) contributes a constant equal to its own
    'projection' to every draw, same convention as attach_sim_data().

    Returns {'floor': p10, 'median', 'mean', 'ceiling': p90} of the
    resulting lineup-total distribution, or None if nothing in the
    lineup has real simulated draws (nothing to combine)."""
    by_player = _sim_rows_by_player(season, week)

    by_game = defaultdict(list)
    constant = 0.0
    for p in lineup:
        row = by_player.get(p['id'])
        if row is None or not row.get('sim'):
            constant += p['projection']
            continue
        by_game[row['game_id']].append(row['sim'])

    if not by_game:
        return None

    per_game_totals = []
    for draw_lists in by_game.values():
        n = min(len(d) for d in draw_lists)
        per_game_totals.append([sum(d[i] for d in draw_lists) for i in range(n)])

    n = min(len(t) for t in per_game_totals)
    combined = sorted(constant + sum(t[i] for t in per_game_totals) for i in range(n))

    def _pct(pct):
        idx = min(len(combined) - 1, int(round(pct * (len(combined) - 1))))
        return round(combined[idx], 2)

    return {
        'draws': combined,
        'floor': _pct(0.10),
        'median': _pct(0.50),
        'mean': round(sum(combined) / len(combined), 2),
        'ceiling': _pct(0.90),
    }


# ── Synthetic salary ────────────────────────────────────────────────────
#
# Real FanDuel pricing is relative to the week's own slate (a great matchup
# in an otherwise weak RB week prices higher than the same projection would
# in a stacked one) — so rather than an absolute points-to-dollars curve,
# salary here is each player's PERCENTILE RANK within their own position
# group *this week's pool*, mapped onto a band via a per-position power
# curve: salary = lo + (hi - lo) * pct**SALARY_CURVE_EXPONENT[position].
#
# Both the bands and the exponents are fit, not guessed — calibrated once
# against a real FanDuel NFL CSV (2026-10-18 slate, 10 games, 512 rows)
# by sorting each position's real salaries ascending, treating rank
# i/(n-1) as "percentile," and grid-searching the exponent that minimizes
# RMSE against the real dollar values at that band's observed lo/hi. The
# fit was done on rows with Played >= 2 only (FanDuel's "Played" column),
# matching MIN_USAGE_GAMES below — our own pool never includes a player
# who hasn't cleared that bar, so calibrating against FanDuel's full
# field (including inactive/practice-squad filler sitting at the price
# floor) would fit a curve shaped by players our pool structurally
# excludes. That population match mattered: fitting against the full
# field first (not done here) produced a curve that, re-applied to our
# own (pre-filtered) pool, systematically underpriced it by ~$700/player
# on average against real salaries; restricting the fit to Played >= 2
# cut that to ~$430/player (RMSE $798, vs $943 unfiltered, vs ~$1250 for
# a plain linear curve, exponent 1 — real FanDuel pay is heavily
# floor-weighted even among active players, e.g. real active-RB salaries:
# median sits near the $4000 floor of a $4000-9300 band). DST came back
# exponent 1.0 (truly linear in real data, RMSE $0) — left alone. This is
# still a one-week calibration sample (the only real salary data this app
# has ever had access to, per this module's docstring) — not re-derived
# from a warehouse like PROJECTION_FIT is, so there's nothing to
# "re-run." Redo this fit (sort real Played>=2 salaries by position,
# grid-search exponent against a fresh CSV) if/when another real FanDuel
# slate export becomes available, rather than trusting one week's shape
# to hold all season.
SALARY_BANDS = {
    'QB':  (6000, 9500),
    'RB':  (4000, 9300),
    'WR':  (4000, 9000),
    'TE':  (4000, 8300),
    'DST': (3000, 4900),
}
SALARY_CURVE_EXPONENT = {
    'QB':  2.8,
    'RB':  3.2,
    'WR':  4.2,
    'TE':  9.8,
    'DST': 1.0,
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
        exponent = SALARY_CURVE_EXPONENT[position]
        n = len(players)
        if n == 1:
            players[0]['salary'] = int(round((lo + hi) / 2 / SALARY_STEP)) * SALARY_STEP
            continue
        ranked = sorted(players, key=lambda p: p['projection'])
        for i, p in enumerate(ranked):
            pct = i / (n - 1)  # 0 = lowest projection, 1 = highest
            salary = lo + (hi - lo) * (pct ** exponent)
            p['salary'] = int(round(salary / SALARY_STEP)) * SALARY_STEP
    return pool


def methodology_context(salary_cap=None, roster_requirements=None, stack_min=0, max_low_tier=None,
                         sim_enabled=False, objective='projection'):
    """Everything templates/_nfl_methodology.html needs to render its
    'How is this calculated?' panel straight from this module's actual
    constants — the template never hand-types a number, so it can't drift
    out of sync with the code the way a prose explanation would. Shared by
    /nfl/lineup and /nfl/recap (both pass this into their template).

    `salary_cap`/`roster_requirements` default to this module's standard
    constants (what /nfl/recap always uses); /nfl/lineup passes its active
    user-adjusted values so the panel reflects whatever actually produced
    the lineup on screen. `stack_min`/`max_low_tier` both default to off
    (0 / None — what /nfl/recap always gets): stack_min because it's an
    optional tournament tactic, max_low_tier because a real backtest found
    it hurts actual outcomes at every setting that does anything — see
    MAX_LOW_TIER_DEFAULT's comment above optimize_lineup().

    `sim_enabled` (default False — what /nfl/recap always gets, since it
    optimizes hindsight 'actual' outcomes, not a live simulator-informed
    objective): True only from /nfl/lineup, to surface the game-simulator
    paragraph and the active `objective` choice in the panel."""
    return {
        'scoring': {
            'pass_yd': PASS_YD, 'pass_td': PASS_TD, 'pass_int': PASS_INT,
            'rush_yd': RUSH_YD, 'rush_td': RUSH_TD,
            'recv_yd': RECV_YD, 'recv_td': RECV_TD, 'reception': RECEPTION,
            'fumble_lost': FUMBLE_LOST, 'return_td': RETURN_TD,
            'pass_300_bonus': PASS_YD_300_BONUS, 'rush_100_bonus': RUSH_YD_100_BONUS,
            'recv_100_bonus': RECV_YD_100_BONUS,
            'dst_sack': DST_SACK, 'dst_int': DST_INT, 'dst_td': DST_TD,
            'dst_pa_tiers': DST_POINTS_ALLOWED_TIERS,
        },
        'projection_fit': projection_fit_summary(),
        'salary_bands': SALARY_BANDS,
        'salary_curve_exponent': SALARY_CURVE_EXPONENT,
        'salary_cap': salary_cap if salary_cap is not None else SALARY_CAP,
        'roster_requirements': roster_requirements or ROSTER_REQUIREMENTS,
        'stack_min': stack_min,
        'max_low_tier': max_low_tier if max_low_tier is not None else MAX_LOW_TIER_DEFAULT,
        'balance_low_percentile': BALANCE_LOW_PERCENTILE,
        'sim_enabled': sim_enabled,
        'sim_ceiling_percentile': SIM_CEILING_PERCENTILE,
        'risk_penalty_k': RISK_PENALTY_K,
        'objective': objective,
    }


# ── News-sentiment signal (queryable only — see nfl_news.py/news_sentiment.py) ──
#
# Backstory: a separate backtest (not in this codebase) found winning DFS
# lineups structurally need ~3 of 9 slots filled by players below the
# 80th percentile of that week's pre-game projection — "surprise" plays
# that are genuinely hard to predict from projection alone. This doesn't
# feed projections, salaries, or optimize_lineup() — it's a read-side join
# over data that already exists, meant to be queried manually (or by a
# future player-detail view) while deciding whether a sub-80th-percentile
# player might have a reason behind an eventual breakout. Whether this is
# ever worth automating into project_points() depends on what
# nfl_news_backtest.py's sub-80th-percentile correlation slice shows.
SURPRISE_PERCENTILE_THRESHOLD = 0.80


def surprise_play_signal(pool, player_id, lookback_days=10):
    """Joins one player's position in an already-built+salaried `pool`
    (see build_player_pool()/assign_salaries()) with their recent
    news-sentiment history. `pool` must be the same week's full pool —
    percentile is only meaningful relative to that week's own slate, the
    same reasoning assign_salaries() uses for salary synthesis.

    Returns None if player_id isn't in pool, else:
    {player_id, name, position, team, opponent, projection, percentile,
     is_surprise_candidate, sentiment_score, article_count, recent_articles}
    sentiment_score is the mean of get_daily_sentiment()'s daily means over
    `lookback_days`, or None if nothing matched that player recently."""
    player = next((p for p in pool if p['id'] == player_id), None)
    if player is None:
        return None

    same_position = [p['projection'] for p in pool if p['position'] == player['position']]
    n = len(same_position)
    percentile = (sum(1 for v in same_position if v <= player['projection']) - 1) / (n - 1) if n > 1 else 1.0

    end = datetime.now(_ET).strftime('%Y-%m-%d')
    start = (datetime.now(_ET) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    daily = nfl_stats_db.get_daily_sentiment(player_id=player_id, start=start, end=end)
    mean_score = sum(d['mean_score'] for d in daily) / len(daily) if daily else None

    return {
        'player_id': player_id, 'name': player['name'], 'position': player['position'],
        'team': player['team'], 'opponent': player['opponent'], 'projection': player['projection'],
        'percentile': round(percentile, 3),
        'is_surprise_candidate': percentile < SURPRISE_PERCENTILE_THRESHOLD,
        'sentiment_score': mean_score,
        'article_count': sum(d['article_count'] for d in daily),
        'recent_articles': nfl_stats_db.get_player_articles(player_id=player_id, limit=5),
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
    'projection', 'stddev', 'risk_adjusted'}, ...] — 'salary' isn't set
    yet, see assign_salaries() (needs the whole pool built first to rank
    within each position). 'stddev' is rolling_stddev() over the same
    history project_points() averaged (None if fewer than 2 games stored);
    'risk_adjusted' is 'projection' minus RISK_PENALTY_K[bucket]*'stddev'
    (or a plain copy of 'projection' when 'stddev' is None) — a
    consistency-favoring alternative objective, see optimize_lineup(objective=
    'risk_adjusted').

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
    left as a gap rather than papered over with a guess. Also absent:
    skill players with stored history but no genuine recent on-field role
    (return specialists, deep WR4/5s who show up in the box score with an
    occasional incidental target) — see _had_meaningful_usage()/
    MIN_USAGE_GAMES. Without this gate, PROJECTION_FIT's flat intercept
    term gives every rostered player SOME nonzero floor regardless of
    real usage, which then prices them cheap and lets the optimizer park
    free cap space on them as pure filler instead of a genuine value play.

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
            games = player_games.get(p['id'], [])
            bucket = ROSTER_BUCKET[p['position']]
            recent_games = games[:ROLLING_WINDOW]
            usage_games = sum(1 for g in recent_games if _had_meaningful_usage(g['categories'], bucket))
            if usage_games < MIN_USAGE_GAMES:
                continue
            points_hist = [compute_fantasy_points(g['categories']) for g in games]
            recent_volume_pg = _recent_touches_per_game(games) if bucket == 'RB' else None
            proj = project_points(bucket, points_hist, opp_def_grade, recent_volume_pg)
            if proj is None:
                continue
            stddev = rolling_stddev(points_hist)
            pool.append({
                'id': p['id'], 'name': p['name'], 'team': team_abbrev, 'opponent': opp_abbrev,
                'position': p['position'], 'projection': proj, 'stddev': stddev,
                'risk_adjusted': max(0.0, round(proj - RISK_PENALTY_K[bucket] * stddev, 2)) if stddev is not None else proj,
            })

        dst_points_hist = [compute_dst_fantasy_points(g['points_allowed'], aggregate_dst_game(g['player_rows']))
                            for g in team_games.get(team_id, [])]
        dst_proj = project_points('DST', dst_points_hist, opp_off_grade)
        if dst_proj is not None:
            dst_stddev = rolling_stddev(dst_points_hist)
            pool.append({
                'id': f"DST_{team_id}", 'name': f"{team.get('name', '')} D/ST",
                'team': team_abbrev, 'opponent': opp_abbrev,
                'position': 'DST', 'projection': dst_proj, 'stddev': dst_stddev,
                'risk_adjusted': max(0.0, round(dst_proj - RISK_PENALTY_K['DST'] * dst_stddev, 2)) if dst_stddev is not None else dst_proj,
            })
    return pool


# ── Balance constraint (anti stars-and-scrubs) — OFF by default, see why ──
#
# Backstory: a separate backtest of the top 50 real hindsight lineups per
# week (250 lineups across 5 sampled weeks) found ZERO pure-studs builds —
# every single one needed a real mix, averaging ~3 players at 80th+
# percentile (within their position's own pool) and ~3 below the 50th
# percentile. The tempting conclusion is that the optimizer should be
# forced to include more low-percentile players too. IT WAS TRIED AND
# BACKTESTED — a real no-leakage backtest across 27 completed weeks
# (2024-2026) scoring against REAL outcomes — and it made lineups WORSE at
# every setting that actually changed anything (mean actual pts/week
# dropped 4.7-12.1 below baseline; settings loose enough not to hurt also
# did nothing). The reasoning gap: the hindsight-best lineups "needed" low
# performers below their PRE-GAME percentile because some of them
# overperformed that week — that's only visible after the fact. Forcing
# the optimizer to pick MORE low-projection players pre-game, using the
# SAME projection signal that already ranked them low, just trades a
# good pick for an arbitrary worse one with no counterbalancing
# information about which one might break out. The actual answer to
# "which low-percentile player is about to break out" is what
# surprise_play_signal() below exists to investigate (via news/sentiment),
# not a blind percentile quota. Left in as an optional, honestly-labeled
# knob (off by default) rather than removed outright, in case a future,
# better-targeted version (e.g. gated by surprise_play_signal rather than
# raw percentile) is worth revisiting — see this feature's PR description
# for the full backtest numbers before changing MAX_LOW_TIER_DEFAULT.
BALANCE_LOW_PERCENTILE = 0.35
MAX_LOW_TIER_DEFAULT = None


def _position_percentiles(pool):
    """{player_id: percentile} — each player's percentile rank of
    'projection' within players sharing their 'position' in this pool
    (0 = lowest at that position, 1 = highest; ties share a value). Same
    formula as surprise_play_signal()'s single-player version, computed
    once for the whole pool here since optimize_lineup() needs every
    player's value, not just one."""
    by_position = defaultdict(list)
    for p in pool:
        by_position[p['position']].append(p['projection'])
    out = {}
    for p in pool:
        same = by_position[p['position']]
        n = len(same)
        out[p['id']] = (sum(1 for v in same if v <= p['projection']) - 1) / (n - 1) if n > 1 else 1.0
    return out


def optimize_lineup(pool, cap=SALARY_CAP, roster_requirements=None, objective='projection', stack_min=0,
                     max_low_tier=MAX_LOW_TIER_DEFAULT, locks=None):
    """pool: [{'id', 'name', 'team', 'position', 'salary', 'projection'}, ...]
    (one row per eligible player/DST, already projected+salaried). Returns
    {'lineup': [...], 'total_salary': int, 'total_projection': float} for
    the lineup that maximizes `objective` (a key present on every pool
    entry — 'projection' for the live page's normal "best projected
    lineup," or 'actual' for nfl_recap.py's hindsight "best possible
    lineup" over the SAME pool/salaries) under `cap`, or None if no
    feasible lineup exists (e.g. too few players at some position in the
    selected games).

    `max_low_tier` (default MAX_LOW_TIER_DEFAULT = None = off; None or a
    number >= the roster size disables it): caps how many non-DST players
    can be below BALANCE_LOW_PERCENTILE within their own position — see
    the module comment above this function for why this defaults OFF
    despite sounding like a good idea (backtested to hurt real outcomes).
    A team/position with nothing but low-percentile options just makes
    those picks relatively less available rather than failing the solve,
    same graceful-degradation behavior as `stack_min`.

    `roster_requirements` defaults to the module's standard ROSTER_
    REQUIREMENTS (what nfl_recap.py's two call sites always get, by not
    passing this — recap evaluates every week against the same standard
    roster for consistency). /nfl/lineup passes the user's adjusted counts.

    `stack_min` (default 0, off): correlated-upside "stack" tactic for
    large-field, top-heavy tournaments (pointless in a cash game, where
    floor matters more than ceiling). Backtested two directions against 39
    weeks of this app's own stored history before shipping either:
    requiring the QB's team to ALSO include a pass-catcher underperformed
    the no-stack baseline (mean actual pts/week 81.0 vs. 86.9, fewer
    ceiling weeks too) — it rigidly forces a specific low-flexibility
    WR/TE onto whichever QB got picked, regardless of that week's real
    value. The REVERSE direction backtested better and is what this
    implements: per team, if its TOP-projected WR/TE (or top `stack_min`
    of them, for stack_min>1 — only stack_min=1 was directly backtested;
    higher values are an untested but directional extension) already earns
    a roster spot on pure value, that team's QB is required too — mean
    91.5 vs. 86.9 baseline, MORE ceiling weeks (7/39 vs. 6/39 scoring
    >=110). It only ever activates when the catcher pick was already good,
    instead of rigidly forcing one. A team with no QB in the pool just
    makes its own top catcher(s) unpickable rather than failing the whole
    solve, so this degrades gracefully instead of blowing up a thin
    selected-games pool.

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
    # `locks`: player ids the user pinned on /nfl/lineup — forced in (ids
    # not in this pool, e.g. from a deselected game, are ignored).
    for pid in (locks or ()):
        if pid in x:
            prob += x[pid] == 1

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

    if stack_min > 0:
        teams = {p['team'] for p in pool}
        for team in teams:
            qbs = [p for p in pool if p['position'] == 'QB' and p['team'] == team]
            catchers = [p for p in pool if p['position'] in ('WR', 'TE') and p['team'] == team]
            if not catchers:
                continue
            top_catchers = sorted(catchers, key=lambda p: p['projection'], reverse=True)[:stack_min]
            qb_sum = pulp.lpSum(x[p['id']] for p in qbs)
            for catcher in top_catchers:
                prob += qb_sum >= x[catcher['id']]

    roster_size = sum(c for pos, c in roster_requirements.items() if pos != 'DST')
    if max_low_tier is not None and max_low_tier < roster_size:
        percentiles = _position_percentiles(pool)
        low_tier = [p for p in pool if p['position'] != 'DST' and percentiles[p['id']] < BALANCE_LOW_PERCENTILE]
        prob += pulp.lpSum(x[p['id']] for p in low_tier) <= max_low_tier

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
