"""
nfl_game_simulator.py — Drive-by-drive Monte Carlo game simulation for
NFL, built on the real play-by-play this app has stored (NflPlay, 79,894
rows / 623 games across 2024-2026 as of this module's introduction — see
nfl_stats_db.get_simulator_play_pool_db()).

Why this exists, and how it differs from nfl_fantasy.simulate_pool(): that
function draws each player's fantasy outcome independently from a Gaussian
centered on their own projection — fast, but a QB's big game and his WR1's
big game are statistically unrelated there, which is wrong (real game
scripts correlate them). This module instead resamples REAL historical
plays situationally (same down/distance/field-position state a real
offense faced) to build one simulated DRIVE, chains drives into a full
game, and only THEN attributes each play's yardage to a specific player on
the two teams actually playing — so if a simulated drive includes a long
completion, that same simulated game is what credits both the QB's
passing line and the receiver's receiving line. Correlation falls out of
the mechanism for free instead of being modeled explicitly.

SCOPE AND KNOWN GAPS (real, not papered over — same convention as
nfl_fantasy.py's own documented scoring gaps):
  - Special teams aren't simulated at all. NflPlay only stores offensive
    scrimmage plays (rush/pass/sack/penalty — see that model's class
    docstring); punts, kickoffs, and field goals were never ingested.
    Every drive starts from a fixed, approximate field position (own 25,
    a typical post-touchback spot) rather than a real punt/return/kickoff
    result, and a drive that stalls on 4th down or in field-goal range
    just ends and flips possession — no punt distance, no FG make/miss.
    A real game's field position and ~3 points per made FG are therefore
    not modeled; this undercounts DST/kicker-adjacent scoring entirely
    (this app doesn't roster kickers anyway) and slightly overstates how
    often a stalled drive "costs nothing."
  - Turnovers are detected by text-searching each play for INTERCEPTED/
    FUMBLES (see get_simulator_play_pool_db()'s docstring for why — the
    stored is_turnover column is never actually set) rather than from a
    clean structured flag, and recovery spot isn't modeled (a turnover
    just ends the drive at its current spot).
  - Player attribution is probability-weighted by each team's own recent
    (default: last 5 games) share of rush attempts / receptions from
    NflPlay — not their real play-calling logic, injury status, or
    matchup-specific usage. A committee backfield's shares will look
    noisier than a real coach's actual rotation.
  - League-wide play sampling, not team- or opponent-conditioned: a play
    is drawn from ALL teams' historical plays in a matching down/distance/
    field-position situation, not this specific offense's or defense's own
    tendencies. Two seasons across 32 teams isn't deep enough to split by
    team AND situation without most buckets going empty — see this
    module's own docstring note where this gets introduced in
    build_play_buckets(). A real per-team pass-rate/explosiveness signal
    (nfl_player_model.py's offense/defense grades) isn't factored in here
    at all yet.
"""
import random
from collections import defaultdict

import nfl_fantasy
import nfl_stats_db

MIN_BUCKET_SAMPLES = 25   # below this, back off to a coarser bucket — see _sample_play()
MAX_PLAYS_PER_DRIVE = 25  # safety valve against a pathological chain of first downs; see module docstring
KICKOFF_YARDS_TO_ENDZONE = 75  # fixed starting field position every drive — see module docstring

# 4th-down handling — NOT fit from data, hand-set, see simulate_drive()'s
# docstring for why: every 4th-down row in NflPlay is already a real
# team's choice to go for it (punts/FGs were never ingested as plays at
# all), so there's no real "declined to go for it" sample to measure a
# true go-rate from, and stored 4th downs skew toward short-yardage,
# late-game, analytically-aggressive situations — not an average 4th
# down. These two constants are a deliberately conservative stand-in for
# "most 4th downs are punted/kicked away, not attempted," re-tunable by
# feel, not backtested.
FOURTH_DOWN_SHORT_YARDAGE = 2     # only distances at or under this are ever "gone for"
FOURTH_DOWN_GO_FOR_IT_RATE = 0.3  # ...and even then, only this fraction of the time

SCORING_PLAY_TYPES = {'Rushing Touchdown', 'Passing Touchdown'}
RUSH_PLAY_TYPES = {'Rush', 'Rushing Touchdown'}
RECEPTION_PLAY_TYPES = {'Pass Reception', 'Passing Touchdown'}


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
    simulated game; see simulate_matchup()."""
    buckets = defaultdict(list)
    for p in plays:
        key = (p['down'], _distance_bucket(p['distance']), _field_bucket(p['yards_to_endzone']))
        buckets[key].append(p)
    return dict(buckets)


def _sample_play(down, distance, yards_to_endzone, buckets, all_plays, rng):
    """Backs off from the finest (down, distance-bucket, field-bucket) key
    to coarser ones when there aren't enough real plays to draw from —
    4th-and-long from your own 10 is a real but rare situation; forcing a
    draw from just those few stored plays would make one or two outlier
    games dictate the whole sample. Backoff order: drop field position,
    then drop distance, then (last resort) draw from every stored play at
    that down. `all_plays`: the full unbucketed list, for that final
    fallback."""
    dist_b, field_b = _distance_bucket(distance), _field_bucket(yards_to_endzone)

    candidates = buckets.get((down, dist_b, field_b), [])
    if len(candidates) >= MIN_BUCKET_SAMPLES:
        return rng.choice(candidates)

    by_down_dist = [p for p in all_plays if p['down'] == down and _distance_bucket(p['distance']) == dist_b]
    if len(by_down_dist) >= MIN_BUCKET_SAMPLES:
        return rng.choice(by_down_dist)

    by_down = [p for p in all_plays if p['down'] == down]
    if len(by_down) >= MIN_BUCKET_SAMPLES:
        return rng.choice(by_down)

    return rng.choice(all_plays)


def simulate_drive(buckets, all_plays, rng, start_yards_to_endzone=KICKOFF_YARDS_TO_ENDZONE):
    """Simulates one team's possession by chaining resampled real plays
    together, down by down. Returns (plays, outcome) — `plays` is the raw
    list of play dicts that occurred (for attribute_plays_to_players()
    to assign to specific players), `outcome` is one of 'touchdown',
    'turnover', 'turnover_on_downs' (also covers "punted/kicked away,
    not simulated" — see FOURTH_DOWN_GO_FOR_IT_RATE), 'max_plays' (the
    safety-valve case, should be rare — see MAX_PLAYS_PER_DRIVE).

    4th down does NOT sample from the play buckets like downs 1-3 do —
    see FOURTH_DOWN_GO_FOR_IT_RATE's comment for why that would be wrong
    (there's no real "chose to punt" sample to draw from). Most 4th downs
    just end the drive here, standing in for an unsimulated punt/FG."""
    down, distance, yards_to_endzone = 1, 10, start_yards_to_endzone
    drive_plays = []

    for _ in range(MAX_PLAYS_PER_DRIVE):
        if down == 4 and (distance > FOURTH_DOWN_SHORT_YARDAGE or rng.random() > FOURTH_DOWN_GO_FOR_IT_RATE):
            return drive_plays, 'turnover_on_downs'

        play = _sample_play(down, distance, yards_to_endzone, buckets, all_plays, rng)
        drive_plays.append(play)

        if play['is_turnover']:
            return drive_plays, 'turnover'

        if play['play_type'] in SCORING_PLAY_TYPES:
            return drive_plays, 'touchdown'

        yards_to_endzone = max(0.0, yards_to_endzone - play['yards_gained'])
        if yards_to_endzone <= 0:
            return drive_plays, 'touchdown'

        if play['yards_gained'] >= distance:
            down, distance = 1, min(10, yards_to_endzone)
        else:
            down += 1
            distance = max(1, distance - play['yards_gained'])
            if down > 4:
                return drive_plays, 'turnover_on_downs'

    return drive_plays, 'max_plays'


def simulate_game(buckets, all_plays, rng, drives_per_team=10):
    """Alternates possession between 'home' and 'away' for `drives_per_team`
    drives each (10 is this app's own stored-data average — see
    nfl_stats_db.get_simulator_play_pool_db()'s sibling PR description for
    the distinct-drive-id count this was calibrated against). Every drive
    starts from the same fixed field position (see KICKOFF_YARDS_TO_ENDZONE
    and this module's docstring) — possession changing via a scoring drive
    doesn't hand the other team worse field position the way a real kickoff
    return might. Returns {'home': [play, ...], 'away': [play, ...]} — all
    plays from all of that side's drives, flattened; drive boundaries
    aren't needed past this point since attribution is per-play."""
    out = {'home': [], 'away': []}
    for side in ('home', 'away') * drives_per_team:
        plays, _outcome = simulate_drive(buckets, all_plays, rng)
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
    beats a wrong guess" convention as the rest of this app. Sacks and
    incomplete passes aren't attributed to anyone: FanDuel's scoring
    doesn't dock a QB for either (see nfl_fantasy.py's scoring constants),
    so there's nothing for them to contribute to a fantasy total."""
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
        # Pass Incompletion / Sack: no FanDuel-scored stat to attribute — see docstring.

    return dict(cats)


def simulate_matchup(home_team_id, away_team_id, n=10, before_date=None, window_games=5,
                      seasons=None, drives_per_team=10, rng=None):
    """Top-level entry point: runs `n` full, independent game simulations
    for one real matchup and returns each involved player's simulated
    fantasy-point outcomes, WITH game-script correlation preserved within
    each of the n games (see module docstring for how that differs from
    nfl_fantasy.simulate_pool()).

    Loads the play pool and builds buckets ONCE (expensive-ish — one DB
    query plus bucketing every stored play), then reuses them across all
    n simulated games, same spirit as build_player_pool()'s own batching
    rationale elsewhere in this app.

    Returns {'home': {player_id: {'sim': [n values], 'sim_min',
    'sim_median', 'sim_max'}}, 'away': {...}} — player_id only, no name/
    position resolution here (that needs a roster lookup, a separate
    concern — see nfl_roster_api.get_skill_players() for a live mapping,
    or resolve the ids the way build_player_pool() does)."""
    rng = rng or random.Random()
    all_plays = nfl_stats_db.get_simulator_play_pool_db(seasons=seasons)
    buckets = build_play_buckets(all_plays)

    home_shares = nfl_stats_db.get_team_usage_shares_db(home_team_id, before_date=before_date,
                                                         window_games=window_games)
    away_shares = nfl_stats_db.get_team_usage_shares_db(away_team_id, before_date=before_date,
                                                          window_games=window_games)

    draws = {'home': defaultdict(list), 'away': defaultdict(list)}
    for _ in range(n):
        game = simulate_game(buckets, all_plays, rng, drives_per_team=drives_per_team)
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
