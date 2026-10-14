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
  - League-wide play sampling, not team- or opponent-conditioned: a play
    is drawn from ALL teams' historical plays in a matching situation,
    not this specific offense's or defense's own tendencies. Two seasons
    across 32 teams isn't deep enough to split by team AND situation
    without most buckets going empty. A real per-team pass-rate/
    explosiveness signal (nfl_player_model.py's offense/defense grades)
    isn't factored in here at all yet.
  - No game clock. Drives run until they end on their own merits (score/
    turnover/turnover-on-downs/stalled-4th-down); `drives_per_team`
    stands in for a real half/game duration rather than a clock that can
    force a team to hurry or run it out.
"""
import random
from collections import defaultdict, namedtuple

import nfl_fantasy
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

SCORING_PLAY_TYPES = {'Rushing Touchdown', 'Passing Touchdown'}
RUSH_PLAY_TYPES = {'Rush', 'Rushing Touchdown'}
RECEPTION_PLAY_TYPES = {'Pass Reception', 'Passing Touchdown'}

# Bundles every resampling pool build_sim_pools() loads once per
# simulation run, threaded through every drive/play-sampling function
# below instead of a growing list of positional (buckets, all_plays, ...)
# parameters.
SimPools = namedtuple('SimPools', [
    'scrimmage_buckets', 'scrimmage_plays',      # down/distance/field-position -> [play, ...], see build_play_buckets()
    'fourth_down_buckets', 'fourth_down_plays',  # field-position -> [punt/FG play, ...], see build_fourth_down_pool()
    'kickoff_plays',                             # flat list, see nfl_stats_db.get_kickoff_pool_db()
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
    simulated game; see build_sim_pools()."""
    buckets = defaultdict(list)
    for p in plays:
        key = (p['down'], _distance_bucket(p['distance']), _field_bucket(p['yards_to_endzone']))
        buckets[key].append(p)
    return dict(buckets)


def build_fourth_down_pool(fourth_down_plays):
    """fourth_down_plays: nfl_stats_db.get_fourth_down_pool_db()'s output
    (real Punt/Field Goal Good/Field Goal Missed plays). Returns
    {field_bucket: [play, ...]} — field position only, no down/distance
    dimension (these plays' own down is always 4; what matters here is
    just where on the field the decision happened). See
    _sample_fourth_down() for how a real team's own punt-vs-kick choice
    at a given field position becomes the simulator's choice too, instead
    of a hand-set distance cutoff."""
    buckets = defaultdict(list)
    for p in fourth_down_plays:
        buckets[_field_bucket(p['yards_to_endzone'])].append(p)
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
    )


def _sample_play(down, distance, yards_to_endzone, pools, rng):
    """Backs off from the finest (down, distance-bucket, field-bucket) key
    to coarser ones when there aren't enough real plays to draw from —
    4th-and-long from your own 10 is a real but rare situation; forcing a
    draw from just those few stored plays would make one or two outlier
    games dictate the whole sample. Backoff order: drop field position,
    then drop distance, then (last resort) draw from every stored play at
    that down."""
    dist_b, field_b = _distance_bucket(distance), _field_bucket(yards_to_endzone)
    all_plays = pools.scrimmage_plays

    candidates = pools.scrimmage_buckets.get((down, dist_b, field_b), [])
    if len(candidates) >= MIN_BUCKET_SAMPLES:
        return rng.choice(candidates)

    by_down_dist = [p for p in all_plays if p['down'] == down and _distance_bucket(p['distance']) == dist_b]
    if len(by_down_dist) >= MIN_BUCKET_SAMPLES:
        return rng.choice(by_down_dist)

    by_down = [p for p in all_plays if p['down'] == down]
    if len(by_down) >= MIN_BUCKET_SAMPLES:
        return rng.choice(by_down)

    return rng.choice(all_plays)


def _sample_fourth_down(yards_to_endzone, pools, rng):
    """Draws a real Punt/Field Goal Good/Field Goal Missed play from
    roughly this field position — the decision (punt vs. attempt) AND
    its outcome (good/missed) both come from the same real play, so
    there's no separate "will they make it" model to get wrong; it's
    whatever a real team in a real, similar spot actually got. Backs off
    to the full fourth-down pool (ignoring field position) if the local
    bucket is thin — same philosophy as _sample_play(), simpler backoff
    since there's only one dimension here. Returns None only if this
    sim run has zero stored punt/FG plays at all (caller falls back to a
    fixed spot — see _resolve_kickoff_start())."""
    if not pools.fourth_down_plays:
        return None
    field_b = _field_bucket(yards_to_endzone)
    candidates = pools.fourth_down_buckets.get(field_b, [])
    if len(candidates) >= MIN_BUCKET_SAMPLES:
        return rng.choice(candidates)
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


def simulate_drive(pools, rng, start_yards_to_endzone=None):
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
    for testing one situation in isolation).

    4th down does NOT sample from the scrimmage play buckets like downs
    1-3 do — see FOURTH_DOWN_GO_FOR_IT_RATE's comment for why a real
    go-rate can't be fit from what's stored. Most 4th downs instead draw
    a real punt/field-goal play via _sample_fourth_down()."""
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
            go_for_it = distance <= FOURTH_DOWN_SHORT_YARDAGE and rng.random() <= FOURTH_DOWN_GO_FOR_IT_RATE
            if not go_for_it:
                fd_play = _sample_fourth_down(yards_to_endzone, pools, rng)
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

        play = _sample_play(down, distance, yards_to_endzone, pools, rng)
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


def simulate_game(pools, rng, drives_per_team=10):
    """Alternates possession between 'home' and 'away' for `drives_per_team`
    drives each (10 is this app's own stored-data average — 20.3 distinct
    drive_ids/game across the warehouse, roughly half per side). Field
    position now carries over continuously from one drive to the next real
    kickoff/punt/turnover result (see simulate_drive()) instead of
    resetting to a fixed spot every time. Returns {'home': [play, ...],
    'away': [play, ...]} — all plays from all of that side's drives,
    flattened, including any 4th-down Punt/Field Goal plays; drive
    boundaries aren't needed past this point since attribution is
    per-play."""
    out = {'home': [], 'away': []}
    next_start = _resolve_kickoff_start(pools, rng)  # opening kickoff of the game
    for i in range(drives_per_team * 2):
        side = 'home' if i % 2 == 0 else 'away'
        plays, _outcome, next_start = simulate_drive(pools, rng, start_yards_to_endzone=next_start)
        out[side].extend(plays)
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


def simulate_matchup(home_team_id, away_team_id, n=10, before_date=None, window_games=5,
                      seasons=None, drives_per_team=10, rng=None):
    """Top-level entry point: runs `n` full, independent game simulations
    for one real matchup and returns each involved player's simulated
    fantasy-point outcomes, WITH game-script correlation preserved within
    each of the n games (see module docstring for how that differs from
    nfl_fantasy.simulate_pool()).

    Loads and buckets every resampling pool ONCE (see build_sim_pools()),
    then reuses them across all n simulated games, same spirit as
    build_player_pool()'s own batching rationale elsewhere in this app.

    Returns {'home': {player_id: {'sim': [n values], 'sim_min',
    'sim_median', 'sim_max'}}, 'away': {...}} — player_id only, no name/
    position resolution here (that needs a roster lookup, a separate
    concern — see nfl_roster_api.get_skill_players() for a live mapping,
    or resolve the ids the way build_player_pool() does)."""
    rng = rng or random.Random()
    pools = build_sim_pools(seasons=seasons)

    home_shares = nfl_stats_db.get_team_usage_shares_db(home_team_id, before_date=before_date,
                                                         window_games=window_games)
    away_shares = nfl_stats_db.get_team_usage_shares_db(away_team_id, before_date=before_date,
                                                          window_games=window_games)

    draws = {'home': defaultdict(list), 'away': defaultdict(list)}
    for _ in range(n):
        game = simulate_game(pools, rng, drives_per_team=drives_per_team)
        for side, shares in (('home', home_shares), ('away', away_shares)):
            player_cats = attribute_plays_to_players(game[side], shares, rng)
            for player_id, categories in player_cats.items():
                draws[side][player_id].append(nfl_fantasy.compute_fantasy_points(categories))

    result = {}
    for side in ('home', 'away'):
        side_out = {}
        for player_id, pts in draws[side].items():
            ranked = sorted(pts)
            side_out[player_id] = {
                'sim': pts,
                'sim_min': ranked[0],
                'sim_median': ranked[len(ranked) // 2],
                'sim_max': ranked[-1],
            }
        result[side] = side_out
    return result
