"""
nfl_game_simulator.py — Drive-by-drive Monte Carlo game simulation for
NFL, built on the real play-by-play this app has stored (NflPlay, 94,630+
rows / 623 games across 2024-2026 as of this module's last update — see
nfl_stats_db.get_simulator_play_pool_db()/get_fourth_down_pool_db()/
get_kickoff_pool_db()).

Why this exists, and how it differs from nfl_fantasy.simulate_pool(): that
function draws each player's fantasy outcome independently from a Gaussian
centered on their own projection — fast, but a QB's big game and his WR1's
big game are statistically unrelated there, which is wrong (real game
scripts correlate them). This module instead resamples REAL historical
plays situationally (same down/distance/field-position state a real
offense faced) to build one simulated DRIVE, chains drives into a full
game with continuous real field position, and only THEN attributes each
play's yardage to a specific player on the two teams actually playing —
so if a simulated drive includes a long completion, that same simulated
game is what credits both the QB's passing line and the receiver's
receiving line. Correlation falls out of the mechanism for free instead
of being modeled explicitly.

SCOPE AND KNOWN GAPS (real, not papered over — same convention as
nfl_fantasy.py's own documented scoring gaps):
  - 4th-down "go for it" decisions use a hand-set threshold/rate, not one
    fit from data — see FOURTH_DOWN_GO_FOR_IT_RATE's comment for why a
    real rate can't be measured from what's stored. Punts and field goal
    attempts/results themselves ARE resampled from real plays (see
    build_fourth_down_pool()) once the sim decides not to go for it.
  - Turnovers are real (NflPlay.is_turnover, ESPN-reported) and hand the
    next drive real field position (the turnover play's own
    end_yards_to_endzone) — but WHO recovers an own fumble/interception
    isn't modeled; fumble-RECOVERY by the offense itself (Fumble Recovery
    (Own)) correctly isn't a turnover, but no specific player is credited
    with forcing or recovering anything defensively (this app doesn't
    roster defensive skill stats at the player level for fantasy anyway).
  - Player attribution is probability-weighted by each team's own recent
    (default: last 5 games) share of rush attempts / receptions from
    NflPlay — not their real play-calling logic, injury status, or
    matchup-specific usage. A committee backfield's shares will look
    noisier than a real coach's actual rotation. Punts/kickoffs/field
    goals are never attributed to a player at all (this app doesn't
    roster punters/kickers for fantasy).
  - Play sampling is league-wide (a play is drawn from ALL teams'
    historical plays in a matching situation — two seasons across 32
    teams isn't deep enough to split by team AND situation without most
    buckets going empty), but the SELECTION within that league-wide pool
    is now team-conditioned: see TEAM_CONDITIONING_K / _sample_play()'s
    `tilt` parameter. Each offense's nfl_player_model.compute_offense_
    grade() vs. the opposing defense's compute_defense_grade() (the same
    signal this app's real win-probability model partially blends in —
    see nfl_player_model.predict()) shifts which REAL historical plays
    get drawn more/less often within a bucket — a play near the top of
    its bucket's own yardage distribution gets upweighted for a favorable
    matchup, one near the bottom for an unfavorable one. Still resampling
    real plays, never inventing yardage. Applies to downs 1-3, the
    short-yardage 4th-down go-for-it sampling, AND the punt/field-goal
    draw when the sim doesn't go for it (_sample_fourth_down() — a
    favorable tilt skews toward made field goals and better punts, via
    the same real-play-reweighting mechanism and the same offense/
    defense grade, since this app has no dedicated special-teams grade
    to condition on instead). The go-for-it RATE itself is also nudged by
    tilt (_effective_go_for_it_rate() — a confident matchup goes for it
    a bit more often). Kickoffs (_resolve_kickoff_start()) are the one
    remaining unconditioned piece — a kickoff's outcome depends on the
    return man/coverage unit, not offense/defense grades, so there's no
    honest signal to condition it on with what this app tracks. The
    conditioning strength for scrimmage/4th-down PLAY SELECTION
    (TEAM_CONDITIONING_K) IS now fit — see that constant's own comment
    for the real grid search behind it (544 leakage-free games, full
    2024+2025 seasons). FOURTH_DOWN_GO_FOR_IT_TILT_SCALE (how much tilt
    nudges the go-for-it RATE, a different knob from K) and
    FOURTH_DOWN_GO_FOR_IT_RATE itself remain hand-set, not backtested —
    there's no clean way to fit "how much should a mismatch change the
    decision to go for it" from what's stored the way win/loss outcomes
    let K get fit.
  - No game clock. Drives run until they end on their own merits (score/
    turnover/turnover-on-downs/stalled-4th-down); `drives_per_team`
    stands in for a real half/game duration rather than a clock that can
    force a team to hurry or run it out. Team score (simulate_game()'s
    'drives' list, and simulate_matchup()'s 'game' summary — win counts,
    avg final score, quarter breakdown) is a direct consequence of this:
    every touchdown is scored as 7 (extra point assumed automatic — this
    app has no 2-point-conversion data, same gap nfl_fantasy.py's scoring
    already documents), and "quarter" is an APPROXIMATION by drive order
    (a side's drives split into four roughly-equal groups), not real
    elapsed clock time — see _approx_quarter(). Good enough to show "how
    did scoring tend to build up over the game," not precise enough to
    trust for anything that needs a real clock (2-minute-drill behavior,
    which team has the ball at the 2:00 warning, etc.).
"""
import itertools
import math
import random
from collections import defaultdict, namedtuple

import nfl_fantasy
import nfl_player_model as pm
import nfl_stats_db

MIN_BUCKET_SAMPLES = 25   # below this, back off to a coarser bucket — see _sample_play()
MAX_PLAYS_PER_DRIVE = 25  # safety valve against a pathological chain of first downs; see module docstring
KICKOFF_YARDS_TO_ENDZONE = 75  # fallback-only now — see _resolve_kickoff_start(); real kickoffs are resampled

# 4th-down handling — NOT fit from data, hand-set, see simulate_drive()'s
# docstring for why: every 4th-down-attempted row in NflPlay is already a
# real team's choice to go for it, so there's no real "declined to go for
# it" sample to measure a true go-rate from, and stored 4th-down attempts
# skew toward short-yardage, late-game, analytically-aggressive situations
# — not an average 4th down. These two constants are a deliberately
# conservative stand-in for "most 4th downs are punted/kicked away, not
# attempted," re-tunable by feel, not backtested. What happens when the
# sim DOESN'T go for it (punt vs. field goal, and the result) IS resampled
# from real plays — see build_fourth_down_pool().
FOURTH_DOWN_SHORT_YARDAGE = 2     # only distances at or under this are ever "gone for"
FOURTH_DOWN_GO_FOR_IT_RATE = 0.3  # ...and even then, only this fraction of the time

# How much a drive's own matchup tilt (see TEAM_CONDITIONING_K below)
# nudges the go-for-it rate above — a confident, favorable-matchup
# offense goes for it a bit more than FOURTH_DOWN_GO_FOR_IT_RATE's own
# baseline, an unfavorable one a bit less. Hand-set, additive, clamped to
# a sane range — not fit from data, same reasoning as every other
# constant in this 4th-down block (nothing to fit it against)."""
FOURTH_DOWN_GO_FOR_IT_TILT_SCALE = 0.2
FOURTH_DOWN_GO_FOR_IT_RATE_MIN = 0.05
FOURTH_DOWN_GO_FOR_IT_RATE_MAX = 0.6


def _effective_go_for_it_rate(tilt):
    """FOURTH_DOWN_GO_FOR_IT_RATE, nudged by this drive's own tilt (see
    TEAM_CONDITIONING_K) and clamped to [FOURTH_DOWN_GO_FOR_IT_RATE_MIN,
    FOURTH_DOWN_GO_FOR_IT_RATE_MAX] — tilt=0 (no conditioning, e.g. a
    team with no usable grade) returns the original flat rate unchanged."""
    rate = FOURTH_DOWN_GO_FOR_IT_RATE + FOURTH_DOWN_GO_FOR_IT_TILT_SCALE * tilt
    return max(FOURTH_DOWN_GO_FOR_IT_RATE_MIN, min(FOURTH_DOWN_GO_FOR_IT_RATE_MAX, rate))


# Team-conditioning strength for _sample_play()'s weighted draw — see
# module docstring's "Play sampling is league-wide" bullet for the full
# mechanism. `tilt` passed into _sample_play() is (offense_grade -
# defense_grade), roughly -0.5..+0.5 for a real but not extreme mismatch
# (both grades center on 1.0 — see nfl_player_model.py). Each candidate
# play's weight is exp(K * percentile * tilt), where percentile is that
# play's own rank (0=worst, 1=best) by yards_gained within its bucket —
# so at tilt=+0.3 (a real favorable matchup) the best play in a bucket
# ends up roughly exp(K*0.3) times more likely to be drawn than the
# worst, tilt=0 (neutral/unknown matchup) reduces to the original uniform
# draw.
#
# FIT, not hand-set — a real grid search against 544 leakage-free
# simulated games spanning the full 2024+2025 seasons (before_date-gated
# to each game's own week, no leakage), scored against real final
# results: K in {0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, picking
# whichever minimized Brier score (same lower-is-better/0.25-is-random
# convention this app's real betting-model page uses). Result was a real
# U-shaped optimum, not a monotonic slide — K=0.0 (no conditioning)
# scored only 53.1% win-pick accuracy (barely above a coin flip) and
# Brier 0.2504; accuracy and Brier both kept improving up through K=0.5
# (58.6% accuracy, Brier 0.2445 — the best of every value tested on BOTH
# metrics), then Brier degraded monotonically again for every K above
# that, reaching 0.2983 at K=3.0. The PREVIOUS value here, K=2.5, was a
# real, measured overconfidence problem: its own calibration table showed
# "90%+ confidence" predictions only panning out 64.5% of the time. K=0.5
# fixes that — see this module's git history / the session that ran this
# backtest for the full per-K results and calibration tables.
TEAM_CONDITIONING_K = 0.5

SCORING_PLAY_TYPES = {'Rushing Touchdown', 'Passing Touchdown'}
RUSH_PLAY_TYPES = {'Rush', 'Rushing Touchdown'}
RECEPTION_PLAY_TYPES = {'Pass Reception', 'Passing Touchdown'}

# Points credited per drive outcome — see this module's docstring for the
# "extra point assumed automatic, no 2-point data" caveat.
DRIVE_POINTS = {'touchdown': 7, 'field_goal_good': 3}


def _approx_quarter(drive_index, drives_per_team):
    """drive_index: 0-based index of this side's drive (their 1st drive
    of the game is 0, etc.). Splits one side's own `drives_per_team`
    drives into four roughly-equal groups and returns which one this
    drive falls in (1-4) — an APPROXIMATION of which quarter a drive
    would fall in on average, not a real clock computation (this
    simulator has none — see module docstring). Proportional rather than
    a fixed drives-per-quarter constant so it still makes sense if
    `drives_per_team` is ever changed from its default."""
    return min(4, int(drive_index * 4 / drives_per_team) + 1)

# Bundles every resampling pool build_sim_pools() loads once per
# simulation run, threaded through every drive/play-sampling function
# below instead of a growing list of positional (buckets, all_plays, ...)
# parameters.
SimPools = namedtuple('SimPools', [
    'scrimmage_buckets', 'scrimmage_plays',      # down/distance/field-position -> [play, ...], see build_play_buckets()
    'fourth_down_buckets', 'fourth_down_plays',  # field-position -> [punt/FG play, ...], see build_fourth_down_pool()
    'kickoff_plays',                             # flat list, see nfl_stats_db.get_kickoff_pool_db()
    'tilt_weight_cache',                         # mutable {} — see _tilted_weights(), the fix for a real
                                                  # confirmed-live perf bug: tilt is FIXED for an entire
                                                  # simulate_matchup() call (one value per side, reused across
                                                  # all n draws and every play of every drive), but the exp()
                                                  # weight list was being rebuilt from scratch on every single
                                                  # play draw before this cache existed — confirmed as the
                                                  # dominant cost of a simulate_week() run (~0.51s/game, ~1150s
                                                  # for one 16-game week) via direct profiling on production.
])


def _distance_bucket(distance):
    if distance <= 3:
        return 'short'
    if distance <= 7:
        return 'medium'
    return 'long'


def _field_bucket(yards_to_endzone):
    if yards_to_endzone <= 10:
        return 'goalline'
    if yards_to_endzone <= 20:
        return 'redzone'
    if yards_to_endzone <= 50:
        return 'opp_territory'
    return 'own_territory'


def build_play_buckets(plays):
    """plays: nfl_stats_db.get_simulator_play_pool_db()'s output. Returns
    {(down, dist_bucket, field_bucket): [play, ...]} — the finest-grained
    index _sample_play() tries first before backing off. Built once per
    simulation run (not per draw) — reused across every drive of every
    simulated game; see build_sim_pools().

    Also mutates each play dict in place, adding '_percentile' — its own
    rank by yards_gained within its bucket (0.0 = worst play in the
    bucket, 1.0 = best), computed once here rather than per-draw so
    _sample_play()'s team-conditioned weighted sampling (see
    TEAM_CONDITIONING_K) stays cheap. Safe to mutate: build_sim_pools()
    builds this from a freshly-loaded play list each run, and each play
    belongs to exactly one bucket (a strict partition), so this never
    overwrites a percentile some other bucket already assigned. Only
    computed at THIS finest-bucket level — _sample_play()'s backoff
    levels (coarser candidate sets) sample uniformly, untilted; see that
    function's docstring."""
    buckets = defaultdict(list)
    for p in plays:
        key = (p['down'], _distance_bucket(p['distance']), _field_bucket(p['yards_to_endzone']))
        buckets[key].append(p)
    for candidates in buckets.values():
        ranked = sorted(candidates, key=lambda p: p['yards_gained'])
        n = len(ranked)
        for i, p in enumerate(ranked):
            p['_percentile'] = i / (n - 1) if n > 1 else 0.5
    return dict(buckets)


def _fourth_down_value(p):
    """A single scalar ranking how good a real Punt/FG play turned out
    FOR THE TEAM THAT KICKED — needed because these three play types
    don't share one natural continuous outcome the way scrimmage yardage
    does (a punt's own end_yards_to_endzone means "how pinned back is the
    opponent," not comparable on the same scale to a binary FG make/
    miss). Hand-set, not fit — same category of gap as FOURTH_DOWN_GO_
    FOR_IT_RATE (unlike TEAM_CONDITIONING_K, which IS now fit — see that
    constant's own comment — there's no real win/loss outcome to fit this
    particular ranking against, just a plausible ordering):
      - Field Goal Good: 100 — unambiguously the best 4th-down outcome
        (3 points), always ranks at/near the top of its bucket.
      - Punt: the resulting opponent field position (end_yards_to_endzone
        — higher means the opponent starts further from their own end
        zone, i.e. a better punt). Typically lands well below 100, so
        a made FG still outranks even a great punt, matching reality
        (3 points beats great field position).
      - Field Goal Missed: 10 — worse than a real punt almost always
        (no points, and the opponent typically takes over in
        reasonable field position near the kick spot), but kept above
        0 rather than tied for last with the worst punts."""
    if p['play_type'] == 'Field Goal Good':
        return 100.0
    if p['play_type'] == 'Field Goal Missed':
        return 10.0
    return p['end_yards_to_endzone'] if p['end_yards_to_endzone'] is not None else 50.0


def build_fourth_down_pool(fourth_down_plays):
    """fourth_down_plays: nfl_stats_db.get_fourth_down_pool_db()'s output
    (real Punt/Field Goal Good/Field Goal Missed plays). Returns
    {field_bucket: [play, ...]} — field position only, no down/distance
    dimension (these plays' own down is always 4; what matters here is
    just where on the field the decision happened). See
    _sample_fourth_down() for how a real team's own punt-vs-kick choice
    at a given field position becomes the simulator's choice too, instead
    of a hand-set distance cutoff.

    Also mutates each play dict in place, adding '_percentile' — its own
    rank by _fourth_down_value() within its bucket (0.0 = worst outcome
    for the kicking team, 1.0 = best) — same precompute-once-not-per-draw
    reasoning as build_play_buckets(), feeding _sample_fourth_down()'s own
    team-conditioned weighted sampling."""
    buckets = defaultdict(list)
    for p in fourth_down_plays:
        buckets[_field_bucket(p['yards_to_endzone'])].append(p)
    for candidates in buckets.values():
        ranked = sorted(candidates, key=_fourth_down_value)
        n = len(ranked)
        for i, p in enumerate(ranked):
            p['_percentile'] = i / (n - 1) if n > 1 else 0.5
    return dict(buckets)


def build_sim_pools(seasons=None):
    """Loads and buckets every resampling pool this simulator needs, ONCE
    — reused across every drive of every simulated game in a
    simulate_matchup() call. Three DB reads total (one per pool), same
    "batch, don't re-query per draw" convention as the rest of this app's
    stats layer."""
    scrimmage_plays = nfl_stats_db.get_simulator_play_pool_db(seasons=seasons)
    fourth_down_plays = nfl_stats_db.get_fourth_down_pool_db(seasons=seasons)
    kickoff_plays = nfl_stats_db.get_kickoff_pool_db(seasons=seasons)
    return SimPools(
        scrimmage_buckets=build_play_buckets(scrimmage_plays),
        scrimmage_plays=scrimmage_plays,
        fourth_down_buckets=build_fourth_down_pool(fourth_down_plays),
        fourth_down_plays=fourth_down_plays,
        kickoff_plays=kickoff_plays,
        tilt_weight_cache={},
    )


def _tilted_choice(bucket_key, candidates, tilt, pools, rng):
    """rng.choices(candidates, ...) weighted by exp(K * percentile * tilt)
    (see TEAM_CONDITIONING_K), but computing and cumulative-summing that
    weight list is itself O(bucket size) — up to ~1500+ for a well-
    populated bucket — and tilt is FIXED for an entire simulate_matchup()
    call (one value per side, reused across all n draws and every play of
    every drive that side runs). Rebuilding the weight list from scratch
    on every single play draw was confirmed live as THE dominant cost of
    a simulate_week() run (~0.51s/simulated game — profiled directly on
    production, nothing to do with DB queries or roster lookups, both of
    which turned out cheap by comparison). This memoizes the (candidates,
    cumulative-weights) pair per (bucket_key, tilt) in pools.
    tilt_weight_cache — computed once, reused for every remaining draw at
    that same tilt. random.choices()'s own `cum_weights=` form is used on
    the cache hit path specifically because it skips the O(n) summation
    rng.choices(..., weights=...) would otherwise redo every call."""
    cache_key = (bucket_key, tilt)
    cached = pools.tilt_weight_cache.get(cache_key)
    if cached is None:
        weights = [math.exp(TEAM_CONDITIONING_K * p['_percentile'] * tilt) for p in candidates]
        cum_weights = list(itertools.accumulate(weights))
        cached = (candidates, cum_weights)
        pools.tilt_weight_cache[cache_key] = cached
    cached_candidates, cum_weights = cached
    return rng.choices(cached_candidates, cum_weights=cum_weights, k=1)[0]


def _sample_play(down, distance, yards_to_endzone, pools, rng, tilt=0.0):
    """Backs off from the finest (down, distance-bucket, field-bucket) key
    to coarser ones when there aren't enough real plays to draw from —
    4th-and-long from your own 10 is a real but rare situation; forcing a
    draw from just those few stored plays would make one or two outlier
    games dictate the whole sample. Backoff order: drop field position,
    then drop distance, then (last resort) draw from every stored play at
    that down.

    `tilt`: (offense_grade - defense_grade) for this specific drive's
    matchup direction, 0.0 for no conditioning (the default — also what
    every backoff level below the finest bucket uses, see
    TEAM_CONDITIONING_K's comment for why). At the finest bucket, a
    nonzero tilt switches from a uniform draw to one weighted by each
    candidate's precomputed '_percentile' (see build_play_buckets()) —
    still only ever returns a REAL historical play, just a different one
    more or less often depending on how favorable this matchup is."""
    dist_b, field_b = _distance_bucket(distance), _field_bucket(yards_to_endzone)
    all_plays = pools.scrimmage_plays

    bucket_key = (down, dist_b, field_b)
    candidates = pools.scrimmage_buckets.get(bucket_key, [])
    if len(candidates) >= MIN_BUCKET_SAMPLES:
        if tilt == 0.0:
            return rng.choice(candidates)
        return _tilted_choice(bucket_key, candidates, tilt, pools, rng)

    by_down_dist = [p for p in all_plays if p['down'] == down and _distance_bucket(p['distance']) == dist_b]
    if len(by_down_dist) >= MIN_BUCKET_SAMPLES:
        return rng.choice(by_down_dist)

    by_down = [p for p in all_plays if p['down'] == down]
    if len(by_down) >= MIN_BUCKET_SAMPLES:
        return rng.choice(by_down)

    return rng.choice(all_plays)


def _sample_fourth_down(yards_to_endzone, pools, rng, tilt=0.0):
    """Draws a real Punt/Field Goal Good/Field Goal Missed play from
    roughly this field position — the decision (punt vs. attempt) AND
    its outcome (good/missed) both come from the same real play, so
    there's no separate "will they make it" model to get wrong; it's
    whatever a real team in a real, similar spot actually got. Backs off
    to the full fourth-down pool (ignoring field position) if the local
    bucket is thin — same philosophy as _sample_play(), simpler backoff
    since there's only one dimension here (and untilted at that backoff
    level, also same as _sample_play()). Returns None only if this
    sim run has zero stored punt/FG plays at all (caller falls back to a
    fixed spot — see _resolve_kickoff_start()).

    `tilt`: the SAME offense-vs-defense signal _sample_play() uses for
    this drive (see TEAM_CONDITIONING_K) — reused here rather than a
    dedicated special-teams grade (this app doesn't have one), on the
    reasoning that a team good enough to grade well on offense/defense
    generally also executes better on 4th down (a sharper kicker, a
    punter who directs the ball better) than a below-average team. At
    the finest bucket, weights by each candidate's precomputed
    '_percentile' (see build_fourth_down_pool()/_fourth_down_value()) —
    a favorable tilt skews toward made field goals and better punts, an
    unfavorable one toward misses and short punts. Still only ever
    returns a real historical play."""
    if not pools.fourth_down_plays:
        return None
    field_b = _field_bucket(yards_to_endzone)
    candidates = pools.fourth_down_buckets.get(field_b, [])
    if len(candidates) >= MIN_BUCKET_SAMPLES:
        if tilt == 0.0:
            return rng.choice(candidates)
        return _tilted_choice(('4th', field_b), candidates, tilt, pools, rng)
    return rng.choice(pools.fourth_down_plays)


def _resolve_kickoff_start(pools, rng):
    """Returns the real resampled field position (yards_to_endzone, in
    the RECEIVING team's own frame) the next drive should start from
    after a score — real kickoffs vary (touchback vs. a long return), so
    this draws one instead of assuming a fixed spot. The rare real
    kickoff-return-touchdown case (is_scoring_play — ~0.3% of kickoffs on
    this app's warehouse) comes back as 0.0: simulate_drive() recognizes
    a non-positive starting position as an immediate score with zero
    plays run, rather than needing special-cased bookkeeping here.
    Falls back to a fixed post-touchback spot if this sim run has no
    stored kickoff plays at all."""
    if not pools.kickoff_plays:
        return KICKOFF_YARDS_TO_ENDZONE
    kickoff = rng.choice(pools.kickoff_plays)
    if kickoff['is_scoring_play']:
        return 0.0
    return kickoff['end_yards_to_endzone'] if kickoff['end_yards_to_endzone'] is not None else KICKOFF_YARDS_TO_ENDZONE


def simulate_drive(pools, rng, start_yards_to_endzone=None, tilt=0.0):
    """Simulates one team's possession by chaining resampled real plays
    together, down by down, with continuous real field position (not a
    fixed restart spot — see _resolve_kickoff_start()/this module's
    docstring). Returns (plays, outcome, next_start) — `plays` is the raw
    list of play dicts that occurred (for attribute_plays_to_players() to
    assign to specific players; a 4th-down Punt/Field Goal play is
    included here too, though it's never attributed to a fantasy player),
    `outcome` is one of 'touchdown', 'turnover', 'field_goal_good',
    'field_goal_missed', 'punt', 'turnover_on_downs' (only when this sim
    run has zero stored fourth-down plays to resample at all — see
    _sample_fourth_down()), 'max_plays' (the safety-valve case, should be
    rare — see MAX_PLAYS_PER_DRIVE). `next_start` is the real field
    position (that team's own frame) whichever side gets the ball NEXT
    should start from — feed it straight into the next simulate_drive()
    call, same way simulate_game() does.

    `start_yards_to_endzone=None` (the default) draws an opening kickoff;
    pass a number directly to start from a specific spot (mainly useful
    for testing one situation in isolation). `tilt` (default 0.0, no
    conditioning): this drive's offense-vs-defense matchup signal, passed
    straight through to _sample_play() — see TEAM_CONDITIONING_K's
    comment and simulate_game()'s docstring for where it comes from.

    4th down does NOT sample from the scrimmage play buckets like downs
    1-3 do — see FOURTH_DOWN_GO_FOR_IT_RATE's comment for why a real
    go-rate can't be fit from what's stored. Most 4th downs instead draw
    a real punt/field-goal play via _sample_fourth_down() (NOT tilted —
    see this module's docstring)."""
    if start_yards_to_endzone is None:
        start_yards_to_endzone = _resolve_kickoff_start(pools, rng)

    down, distance, yards_to_endzone = 1, 10, start_yards_to_endzone
    drive_plays = []

    # A kickoff-return touchdown resolves to yards_to_endzone <= 0 before
    # any scrimmage play is run — see _resolve_kickoff_start()'s docstring.
    if yards_to_endzone <= 0:
        return drive_plays, 'touchdown', _resolve_kickoff_start(pools, rng)

    for _ in range(MAX_PLAYS_PER_DRIVE):
        if down == 4:
            go_for_it = (distance <= FOURTH_DOWN_SHORT_YARDAGE
                         and rng.random() <= _effective_go_for_it_rate(tilt))
            if not go_for_it:
                fd_play = _sample_fourth_down(yards_to_endzone, pools, rng, tilt=tilt)
                if fd_play is None:
                    return drive_plays, 'turnover_on_downs', _resolve_kickoff_start(pools, rng)
                drive_plays.append(fd_play)
                if fd_play['play_type'] == 'Field Goal Good':
                    return drive_plays, 'field_goal_good', _resolve_kickoff_start(pools, rng)
                next_start = (fd_play['end_yards_to_endzone'] if fd_play['end_yards_to_endzone'] is not None
                              else KICKOFF_YARDS_TO_ENDZONE)
                outcome = 'field_goal_missed' if fd_play['play_type'] == 'Field Goal Missed' else 'punt'
                return drive_plays, outcome, next_start
            # else: short-yardage go-for-it — fall through to sample a normal play below.

        play = _sample_play(down, distance, yards_to_endzone, pools, rng, tilt=tilt)
        drive_plays.append(play)

        if play['is_turnover']:
            next_start = (play['end_yards_to_endzone'] if play['end_yards_to_endzone'] is not None
                          else max(0.0, 100.0 - yards_to_endzone))
            return drive_plays, 'turnover', next_start

        if play['play_type'] in SCORING_PLAY_TYPES:
            return drive_plays, 'touchdown', _resolve_kickoff_start(pools, rng)

        yards_to_endzone = max(0.0, yards_to_endzone - play['yards_gained'])
        if yards_to_endzone <= 0:
            return drive_plays, 'touchdown', _resolve_kickoff_start(pools, rng)

        if play['yards_gained'] >= distance:
            down, distance = 1, min(10, yards_to_endzone)
        else:
            down += 1
            distance = max(1, distance - play['yards_gained'])

    return drive_plays, 'max_plays', _resolve_kickoff_start(pools, rng)


def simulate_game(pools, rng, drives_per_team=10, home_tilt=0.0, away_tilt=0.0):
    """Alternates possession between 'home' and 'away' for `drives_per_team`
    drives each (10 is this app's own stored-data average — 20.3 distinct
    drive_ids/game across the warehouse, roughly half per side). Field
    position now carries over continuously from one drive to the next real
    kickoff/punt/turnover result (see simulate_drive()) instead of
    resetting to a fixed spot every time. Returns {'home': {'plays': [...],
    'drives': [...]}, 'away': {...}} per side — 'plays' is every play from
    all of that side's drives, flattened (including any 4th-down Punt/
    Field Goal plays), for attribute_plays_to_players(); 'drives' is one
    {'outcome', 'points', 'quarter'} per drive (see DRIVE_POINTS/
    _approx_quarter()) for simulate_matchup()'s game-level summary (final
    score, win/loss, quarter breakdown).

    `home_tilt`/`away_tilt` (default 0.0, no conditioning): each side's
    own offense-vs-opposing-defense matchup signal — home's drives use
    `home_tilt` (home offense vs. away defense), away's use `away_tilt`
    (away offense vs. home defense). See simulate_matchup() for where
    these come from (nfl_player_model's offense/defense grades) and
    TEAM_CONDITIONING_K for the mechanism."""
    out = {'home': {'plays': [], 'drives': []}, 'away': {'plays': [], 'drives': []}}
    side_drive_idx = {'home': 0, 'away': 0}
    tilt_by_side = {'home': home_tilt, 'away': away_tilt}
    next_start = _resolve_kickoff_start(pools, rng)  # opening kickoff of the game
    for i in range(drives_per_team * 2):
        side = 'home' if i % 2 == 0 else 'away'
        plays, outcome, next_start = simulate_drive(pools, rng, start_yards_to_endzone=next_start,
                                                      tilt=tilt_by_side[side])
        out[side]['plays'].extend(plays)
        out[side]['drives'].append({
            'outcome': outcome,
            'points': DRIVE_POINTS.get(outcome, 0),
            'quarter': _approx_quarter(side_drive_idx[side], drives_per_team),
        })
        side_drive_idx[side] += 1
    return out


def attribute_plays_to_players(plays, usage_shares, rng):
    """plays: one side's flattened play list from simulate_game().
    usage_shares: nfl_stats_db.get_team_usage_shares_db()'s output for
    that same side's offense. Returns {player_id: categories} in the same
    shape nfl_fantasy.compute_fantasy_points() expects, so that function
    can be reused as-is rather than re-implementing scoring here.

    A play whose relevant share dict is empty (team has no stored recent
    rushers/receivers — e.g. a brand-new team context) silently drops that
    play's attribution rather than guessing a player — same "real gap
    beats a wrong guess" convention as the rest of this app. Sacks,
    incomplete passes, and any special-teams play (Punt/Kickoff/Field
    Goal) aren't attributed to anyone: FanDuel's scoring doesn't dock a
    QB for a sack/incompletion, and this app doesn't roster punters/
    kickers for fantasy at all, so there's nothing for them to contribute
    to a fantasy total."""
    cats = defaultdict(lambda: {
        'passing': {'passingYards': 0.0, 'passingTouchdowns': 0, 'interceptions': 0},
        'rushing': {'rushingYards': 0.0, 'rushingTouchdowns': 0},
        'receiving': {'receivingYards': 0.0, 'receivingTouchdowns': 0, 'receptions': 0},
    })

    qb_id = usage_shares.get('qb_id')
    rush_shares = usage_shares.get('rush_shares') or {}
    rec_shares = usage_shares.get('rec_shares') or {}

    def _weighted_pick(shares):
        if not shares:
            return None
        players = list(shares.keys())
        weights = list(shares.values())
        return rng.choices(players, weights=weights, k=1)[0]

    for p in plays:
        pt = p['play_type']
        if pt in RUSH_PLAY_TYPES:
            rusher = _weighted_pick(rush_shares)
            if rusher is None:
                continue
            cats[rusher]['rushing']['rushingYards'] += p['yards_gained']
            if pt == 'Rushing Touchdown':
                cats[rusher]['rushing']['rushingTouchdowns'] += 1
        elif pt in RECEPTION_PLAY_TYPES:
            receiver = _weighted_pick(rec_shares)
            if qb_id is None or receiver is None:
                continue
            cats[qb_id]['passing']['passingYards'] += p['yards_gained']
            cats[receiver]['receiving']['receivingYards'] += p['yards_gained']
            cats[receiver]['receiving']['receptions'] += 1
            if pt == 'Passing Touchdown':
                cats[qb_id]['passing']['passingTouchdowns'] += 1
                cats[receiver]['receiving']['receivingTouchdowns'] += 1
        # Pass Incompletion / Sack / Punt / Kickoff / Field Goal *: no FanDuel-scored
        # stat to attribute to a rostered fantasy player — see docstring.

    return dict(cats)


# Default Monte Carlo draw count for simulate_matchup()/simulate_week().
# 10 (this module's original default) made min/max essentially just "the
# worst and best of 10 random games" — too few samples for a real floor/
# ceiling. At ~0.11s/game (pure Python, no numpy, confirmed on this app's
# production warehouse), 150 costs ~16-18s per matchup — far too slow for
# a live page request, but exactly the kind of cost a DAILY BACKGROUND JOB
# (see simulate_week()/app.py's cron registration) should absorb instead
# of a visitor's page load.
DEFAULT_SIM_DRAWS = 150


GRADE_WINDOW = 5  # matches nfl_player_model.QB_WINDOW/DEF_WINDOW — the window those grades were designed around


def _team_grades(team_id, before_date=None):
    """(offense_grade, defense_grade) for one team — reuses nfl_stats_db.
    get_team_game_aggregates_db() (the same batched read nfl_api.py's live
    win-probability predictions and nfl_fantasy.build_player_pool() both
    already use) rather than re-deriving team-game aggregates here. Either
    value is None if this team has no usable stored history yet (a bye-
    heavy early season, a brand-new team context) — simulate_matchup()
    treats a None grade as "no conditioning for this side," not an error."""
    off_games, def_games = nfl_stats_db.get_team_game_aggregates_db(team_id, before_date=before_date,
                                                                      limit=GRADE_WINDOW)
    return pm.compute_offense_grade(off_games), pm.compute_defense_grade(def_games)


def simulate_matchup(home_team_id, away_team_id, n=DEFAULT_SIM_DRAWS, before_date=None, window_games=5,
                      seasons=None, drives_per_team=10, rng=None, pools=None):
    """Top-level entry point: runs `n` full, independent game simulations
    for one real matchup and returns each involved player's simulated
    fantasy-point outcomes, WITH game-script correlation preserved within
    each of the n games (see module docstring for how that differs from
    nfl_fantasy.simulate_pool()).

    `pools` (optional): a pre-built SimPools (see build_sim_pools()) — pass
    this when simulating multiple matchups in one run (see simulate_week())
    so the league-wide play pools, which don't vary by matchup, are only
    built once instead of once per game. None (the default) builds them
    here, for simulating a single matchup standalone.

    Returns {'players': {'home': {player_id: {'sim': [n values],
    'sim_min', 'sim_median', 'sim_max', 'sim_mean'}}, 'away': {...}},
    'game': {'n', 'home_wins', 'away_wins', 'ties', 'home_win_pct',
    'away_win_pct', 'home_avg_score', 'away_avg_score',
    'home_quarter_avg', 'away_quarter_avg'}} — 'game' is the team-score-
    level summary (see DRIVE_POINTS/_approx_quarter()'s "no real clock"
    caveat, module docstring). player_id only, no name/position
    resolution here (that needs a roster lookup, a separate concern —
    see nfl_roster_api.get_skill_players() for a live mapping, or resolve
    the ids the way build_player_pool() does).

    Each side's play sampling is conditioned on real team strength — see
    _team_grades()/TEAM_CONDITIONING_K: home's tilt is home's own offense
    grade minus away's defense grade (and symmetrically for away), both
    from nfl_player_model.py's existing offense/defense grade signal. A
    team with no usable stored history (grade is None on either side)
    falls back to tilt=0.0 for that side — the original untilted,
    league-wide-uniform behavior — rather than guessing a grade."""
    rng = rng or random.Random()
    pools = pools or build_sim_pools(seasons=seasons)

    home_shares = nfl_stats_db.get_team_usage_shares_db(home_team_id, before_date=before_date,
                                                         window_games=window_games)
    away_shares = nfl_stats_db.get_team_usage_shares_db(away_team_id, before_date=before_date,
                                                          window_games=window_games)

    home_off, home_def = _team_grades(home_team_id, before_date=before_date)
    away_off, away_def = _team_grades(away_team_id, before_date=before_date)
    home_tilt = (home_off - away_def) if (home_off is not None and away_def is not None) else 0.0
    away_tilt = (away_off - home_def) if (away_off is not None and home_def is not None) else 0.0

    draws = {'home': defaultdict(list), 'away': defaultdict(list)}
    home_wins = away_wins = ties = 0
    scores = {'home': [], 'away': []}
    quarter_points = {'home': [0, 0, 0, 0], 'away': [0, 0, 0, 0]}
    for _ in range(n):
        game = simulate_game(pools, rng, drives_per_team=drives_per_team,
                              home_tilt=home_tilt, away_tilt=away_tilt)
        side_score = {}
        for side, shares in (('home', home_shares), ('away', away_shares)):
            side_score[side] = sum(d['points'] for d in game[side]['drives'])
            scores[side].append(side_score[side])
            for d in game[side]['drives']:
                quarter_points[side][d['quarter'] - 1] += d['points']
            player_cats = attribute_plays_to_players(game[side]['plays'], shares, rng)
            for player_id, categories in player_cats.items():
                draws[side][player_id].append(nfl_fantasy.compute_fantasy_points(categories))
        if side_score['home'] > side_score['away']:
            home_wins += 1
        elif side_score['away'] > side_score['home']:
            away_wins += 1
        else:
            ties += 1

    players = {}
    for side in ('home', 'away'):
        side_out = {}
        for player_id, pts in draws[side].items():
            ranked = sorted(pts)
            side_out[player_id] = {
                'sim_mean': round(sum(pts) / len(pts), 2),
                'sim': pts,
                'sim_min': ranked[0],
                'sim_median': ranked[len(ranked) // 2],
                'sim_max': ranked[-1],
            }
        players[side] = side_out

    game_summary = {
        'n': n,
        'home_wins': home_wins, 'away_wins': away_wins, 'ties': ties,
        'home_win_pct': round(100 * home_wins / n, 1),
        'away_win_pct': round(100 * away_wins / n, 1),
        'home_avg_score': round(sum(scores['home']) / n, 1),
        'away_avg_score': round(sum(scores['away']) / n, 1),
        'home_quarter_avg': [round(q / n, 1) for q in quarter_points['home']],
        'away_quarter_avg': [round(q / n, 1) for q in quarter_points['away']],
        'home_tilt': round(home_tilt, 3), 'away_tilt': round(away_tilt, 3),
    }
    return {'players': players, 'game': game_summary}


def simulate_week(season=None, week=None, n=DEFAULT_SIM_DRAWS, before_date=None, window_games=5,
                   seasons=None, drives_per_team=10, rng=None):
    """Runs simulate_matchup() for every game in one NFL week and PERSISTS
    the results (nfl_stats_db.persist_sim_results_db()) instead of
    returning them — this is the function app.py's daily background job
    (_simulate_nfl_week()) calls; /nfl/simulations then reads what this
    wrote via nfl_stats_db.get_sim_results_db() rather than ever calling
    this directly on a request.

    `week`/`season` (optional): default to nfl_api.get_current_week()'s
    answer, same as every other "current week" caller in this app. Pass
    an explicit `season` to simulate a PAST season's week instead (e.g.
    validating the simulator against 2024/2025, which this app's
    warehouse has full seasons of — see nfl_api.build_week_games_light()'s
    own `season` override, which this passes straight through; ESPN's
    scoreboard serves an old season's games the same way it serves the
    current one). Combine with `before_date` for a genuine no-leakage
    backtest, same discipline as nfl_recap.py. Builds
    every resampling pool ONCE for the whole week (league-wide, not
    game-specific — see build_sim_pools()) and reuses it across every
    matchup, same batching rationale as simulate_matchup()'s own `pools`
    parameter exists for.

    Player names/positions are resolved via nfl_stats_db.get_or_fetch_
    roster_db() — a DB-backed cache in front of nfl_roster_api's own live
    fetch (see that function's docstring for why: nfl_roster_api's
    in-process cache doesn't survive across separate script runs, which
    made a multi-week backfill like "simulate weeks 1-4" needlessly
    re-fetch every team's roster live once per week instead of once
    total, confirmed live as the dominant cost of a standalone run — ~18
    minutes for one week's 32 lookups, almost entirely ESPN round-trips).
    Still the same CURRENT-roster-gated source build_player_pool() uses
    elsewhere in this app, with that function's own documented staleness
    caveat (a player traded/cut since the last snapshot won't reflect it
    until the next refresh, max_age_hours after). A player whose name
    doesn't resolve is still persisted (as player_name=None —
    nfl_stats_db.get_sim_results_db() falls back to showing their id)
    rather than dropped, matching this app's "a real gap beats a wrong
    guess" convention."""
    import nfl_api

    rng = rng or random.Random()
    week_ctx = nfl_api.build_week_games_light(week, season=season)
    actual_season, actual_week = week_ctx['season'], week_ctx['week']
    games = week_ctx['games']
    if not games:
        return

    pools = build_sim_pools(seasons=seasons)

    for g in games:
        home_id, away_id = g['home'].get('id'), g['away'].get('id')
        if not home_id or not away_id:
            continue
        sim = simulate_matchup(home_id, away_id, n=n, before_date=before_date,
                                window_games=window_games, drives_per_team=drives_per_team,
                                rng=rng, pools=pools)
        for side, team_id, opponent_id in (('home', home_id, away_id), ('away', away_id, home_id)):
            roster = {p['id']: p for p in nfl_stats_db.get_or_fetch_roster_db(team_id)}
            results = []
            for player_id, outcome in sim['players'][side].items():
                info = roster.get(player_id, {})
                results.append({
                    'player_id': player_id,
                    'player_name': info.get('name'),
                    'position': info.get('position'),
                    'n_draws': len(outcome['sim']),
                    **outcome,
                })
            nfl_stats_db.persist_sim_results_db(actual_season, actual_week, g['game_id'],
                                                 team_id, opponent_id, results)
        nfl_stats_db.persist_sim_game_summary_db(actual_season, actual_week, g['game_id'],
                                                  home_id, away_id, sim['game'])
