"""
nba_game_simulator.py — Period-by-period Monte Carlo game simulation for
NBA, the basketball counterpart to nfl_game_simulator.py. Resamples real
historical scoring outcomes (NbaPlay, via nba_stats_db.
get_simulator_quarter_pool_db()) to build correlated team/player
outcomes, the same "resample real data instead of drawing each player's
outcome independently" idea NFL's simulator uses — see that module's own
docstring for why independent per-player draws (nba_player_model.py's
single-number projections) understate real game-script correlation.

HOW THIS DIFFERS FROM NFL'S SIMULATOR, AND WHY (read before changing
anything here):

  - Unit of resampling is one TEAM's one PERIOD of real scoring, not one
    play. NFL resamples individual plays bucketed by (down, distance,
    field position) because NflPlay carries a clean down/distance/field-
    position state machine to bucket by. Basketball has no analog —
    there's no "down" and no fixed-yardage progress metric — and
    NbaPlay's `participant_ids` is an unordered list that doesn't label
    shooter vs. assister vs. rebounder (see that model's own docstring,
    app.py), so a play-by-play role-attribution pass the way NFL's
    attribute_plays_to_players() does isn't reliably reconstructable
    without guessing an unverified ESPN play-type text taxonomy. Instead,
    the real signal this module leans on is the one thing every NbaPlay
    row reliably carries regardless of play-type wording: a real `period`
    and the running home_score/away_score. See nba_stats_db.
    get_simulator_quarter_pool_db()'s own docstring for the full
    mechanism (bucketed by (period, score_margin_bucket), the NBA
    analog of NFL's (down, distance_bucket, field_bucket)).

  - Player attribution is usage-share-based, not play-level. Because
    individual plays aren't resampled, there's no real play for a
    specific player to be credited with. Instead, each simulated team
    total is split across that team's active roster by recent SCORING
    efficiency share (nba_stats_db.get_team_usage_shares_db(), itself
    built on nba_player_model.project_player()'s usage-rate x
    per-possession-efficiency split — the same mechanism project_team_
    score() already uses to redistribute minutes), with per-draw
    multiplicative noise so a player's shot CAN run hot/cold without
    literally always taking the same fixed fraction of the team's score.
    Rebounds/assists/steals/blocks/turnovers are drawn per player from
    their own recent per-game rate with Gaussian noise — a real
    simplification: NFL's player-level correlation falls out of
    resampling actual plays, this module's player-level numbers are each
    player's OWN recent average plus noise, not resampled from real
    game events. The team-level score IS real resampled history (so two
    simulated teammates' point shares still move together when the team
    has a big or small simulated quarter) — the correlation is at the
    team-total level, not the play level, same real tradeoff documented
    rather than hidden.

  - No DST-equivalent exists in basketball (no separate "defensive unit"
    distinct from the 5 players on the floor) — this module has nothing
    corresponding to NFL's synthesized DST_{team_id} entries.

  - Team conditioning ("tilt") uses nba_player_model.opponent_factor()
    (a single team-wide points-allowed-vs-league-average multiplier) in
    place of NFL's offense_grade/defense_grade pair — NBA has no
    equivalent grade signal yet. TEAM_CONDITIONING_K below is a hand-set
    placeholder, NOT fit from a backtest the way NFL's own K=0.5 was
    (see that module's comment for the grid-search this hasn't had yet).
"""
import itertools
import math
import random
from collections import defaultdict, namedtuple

import nba_fantasy
import nba_player_model
import nba_stats_db

DEFAULT_SIM_DRAWS = 150
PERIODS_REGULATION = 4
MAX_OT_PERIODS = 4          # safety cap on simulated overtimes
MIN_BUCKET_SAMPLES = 20     # below this, back off to a coarser bucket — see _sample_quarter()

# Same (period, score_margin_bucket) coarsening nba_stats_db.
# get_simulator_quarter_pool_db() uses to build the pool — duplicated here
# (not imported) because this module also needs to bucket a LIVE simulated
# margin mid-game, not just pool rows at load time. Keep in sync with
# nba_stats_db._MARGIN_BUCKET_EDGES if that ever changes.
_MARGIN_BUCKET_EDGES = [-100, -15, -8, -3, 0, 3, 8, 15, 100]


def _margin_bucket(margin):
    for i in range(len(_MARGIN_BUCKET_EDGES) - 1):
        lo, hi = _MARGIN_BUCKET_EDGES[i], _MARGIN_BUCKET_EDGES[i + 1]
        if lo <= margin < hi or (hi == 100 and margin == 100):
            return i
    return len(_MARGIN_BUCKET_EDGES) - 2


# Team-conditioning strength for _sample_quarter()'s weighted draw — NFL's
# analog (TEAM_CONDITIONING_K in nfl_game_simulator.py) was fit via a real
# grid search against 544 leakage-free backtested games. No equivalent
# NBA backtest exists yet (see nba_player_bootstrap.py's own "UNVALIDATED"
# caveat on nba_player_model.py, which this tilt is built from) — this is
# a hand-set starting value, not a fit one, kept deliberately modest until
# a real backtest can replace it.
TEAM_CONDITIONING_K = 0.5

# Per-draw noise applied on top of each player's own recent-average share/
# rate — hand-set, not fit (same "nothing to fit it against yet" reasoning
# as TEAM_CONDITIONING_K above). Without this, a player would take the
# exact same fraction of their team's score and the exact same board/
# assist/steal/block/turnover count in literally every one of the n draws,
# which understates real game-to-game variance.
SCORING_SHARE_NOISE_SD = 0.35
RATE_STAT_NOISE_FRAC = 0.40

# sim_stats fields (simulate_matchup()) summarized by MEAN rather than
# MEDIAN — same reasoning as NFL's COUNT_STAT_FIELDS: these are discrete,
# often-small counts where the median of n draws can sit at a value the
# per-draw distribution rarely actually lands on. 'points' is the one
# continuous-ish field, summarized by median instead (see the comment
# where this is read, inside simulate_matchup()).
COUNT_STAT_FIELDS = {'rebounds', 'assists', 'steals', 'blocks', 'turnovers'}

SimPools = namedtuple('SimPools', ['quarter_pool', 'quarter_buckets', 'tilt_weight_cache'])


def build_quarter_buckets(pool):
    """pool: nba_stats_db.get_simulator_quarter_pool_db()'s output. Returns
    {(period, score_margin_bucket): [row, ...]}, each row annotated with
    '_percentile' (rank by points_scored within its own bucket, 0=worst/
    1=best) for _tilted_choice()'s weighted draw — same role as NFL's
    build_play_buckets()."""
    buckets = defaultdict(list)
    for row in pool:
        buckets[(row['period'], row['score_margin_bucket'])].append(row)
    for rows in buckets.values():
        ranked = sorted(rows, key=lambda r: r['points_scored'])
        n = len(ranked)
        for i, r in enumerate(ranked):
            r['_percentile'] = i / (n - 1) if n > 1 else 0.5
    return buckets


def build_sim_pools(seasons=None):
    """Loads and buckets the quarter-scoring pool ONCE — reused across
    every simulated game in a simulate_matchup() call. Same "batch, don't
    re-query per draw" convention as NFL's build_sim_pools()."""
    pool = nba_stats_db.get_simulator_quarter_pool_db(seasons=seasons)
    return SimPools(quarter_pool=pool, quarter_buckets=build_quarter_buckets(pool), tilt_weight_cache={})


def _tilted_choice(bucket_key, candidates, tilt, pools, rng):
    """Same memoized weighted-draw mechanism as NFL's _tilted_choice() —
    see that function's docstring for why the (candidates, cumulative-
    weights) pair is cached per (bucket_key, tilt) rather than rebuilt on
    every draw."""
    cache_key = (bucket_key, tilt)
    cached = pools.tilt_weight_cache.get(cache_key)
    if cached is None:
        weights = [math.exp(TEAM_CONDITIONING_K * p['_percentile'] * tilt) for p in candidates]
        cum_weights = list(itertools.accumulate(weights))
        cached = (candidates, cum_weights)
        pools.tilt_weight_cache[cache_key] = cached
    cached_candidates, cum_weights = cached
    return rng.choices(cached_candidates, cum_weights=cum_weights, k=1)[0]


def _sample_quarter(period, margin, pools, rng, tilt=0.0):
    """Backs off from the finest (period, margin-bucket) key to a
    period-only, then fully unconditioned, pool when there aren't enough
    real rows to draw from — same backoff spirit as NFL's _sample_play().
    Returns a real historical points-scored value (int)."""
    capped_period = min(period, 5)
    bucket_key = (capped_period, _margin_bucket(margin))
    candidates = pools.quarter_buckets.get(bucket_key, [])
    if len(candidates) >= MIN_BUCKET_SAMPLES:
        if tilt == 0.0:
            return rng.choice(candidates)['points_scored']
        return _tilted_choice(bucket_key, candidates, tilt, pools, rng)['points_scored']

    by_period = [r for r in pools.quarter_pool if r['period'] == capped_period]
    if len(by_period) >= MIN_BUCKET_SAMPLES:
        return rng.choice(by_period)['points_scored']

    return rng.choice(pools.quarter_pool)['points_scored']


def simulate_game(pools, rng, home_tilt=0.0, away_tilt=0.0, max_ot_periods=MAX_OT_PERIODS):
    """Chains periods into one full game: 4 regulation periods, then real
    overtime periods (bucketed the same way, period capped at 5 for
    lookup) for as long as the score stays tied, up to `max_ot_periods`
    as a safety cap (leaves a tied score in the vanishingly rare case
    every OT period also ties — real NBA games don't end tied, but
    nothing here forces a resolution the way a real game's extra
    overtime periods would). Each side's quarter is drawn independently
    conditioned on the SAME entering score margin (one team's lead is the
    other's deficit), not drawn jointly — a real simplification (no
    single resampled possession sequence ties both teams' same-period
    output together the way one resampled NFL drive ties a single
    team's own plays together), but each side's own full-game total is
    still built from real history, period by period, rather than one
    flat per-game Gaussian.

    Returns {'home': {'period_points': [...], 'total'}, 'away': {...}}."""
    home_total = away_total = 0
    home_periods, away_periods = [], []
    period = 1
    while True:
        margin_home = home_total - away_total
        home_pts = _sample_quarter(period, margin_home, pools, rng, tilt=home_tilt)
        away_pts = _sample_quarter(period, -margin_home, pools, rng, tilt=away_tilt)
        home_total += home_pts
        away_total += away_pts
        home_periods.append(home_pts)
        away_periods.append(away_pts)
        if period >= PERIODS_REGULATION:
            if home_total != away_total:
                break
            if period - PERIODS_REGULATION >= max_ot_periods:
                break
        period += 1
    return {
        'home': {'period_points': home_periods, 'total': home_total},
        'away': {'period_points': away_periods, 'total': away_total},
    }


def attribute_game_to_players(team_points, usage_shares, rng):
    """Splits one side's simulated game total across that team's recent
    active roster — see module docstring's "Player attribution is
    usage-share-based" section for why this works at the game level
    (not play/quarter level) and what it does and doesn't model.
    `usage_shares`: nba_stats_db.get_team_usage_shares_db()'s output.
    Returns {player_id: {'points', 'rebounds', 'assists', 'steals',
    'blocks', 'turnovers'}}. {} if usage_shares is empty (team has no
    usable recent history — caller's problem, same "real gap beats a
    guess" convention as the rest of this app)."""
    if not usage_shares:
        return {}

    noisy_share = {
        pid: max(0.01, rng.gauss(1.0, SCORING_SHARE_NOISE_SD)) * s['pts_share']
        for pid, s in usage_shares.items()
    }
    total_noisy = sum(noisy_share.values()) or 1.0

    out = {}
    for pid, s in usage_shares.items():
        out[pid] = {
            'points': round(team_points * (noisy_share[pid] / total_noisy), 2),
            'rebounds': round(max(0.0, rng.gauss(s['reb_pg'], s['reb_pg'] * RATE_STAT_NOISE_FRAC)), 2),
            'assists': round(max(0.0, rng.gauss(s['ast_pg'], s['ast_pg'] * RATE_STAT_NOISE_FRAC)), 2),
            'steals': round(max(0.0, rng.gauss(s['stl_pg'], s['stl_pg'] * RATE_STAT_NOISE_FRAC)), 2),
            'blocks': round(max(0.0, rng.gauss(s['blk_pg'], s['blk_pg'] * RATE_STAT_NOISE_FRAC)), 2),
            'turnovers': round(max(0.0, rng.gauss(s['tov_pg'], s['tov_pg'] * RATE_STAT_NOISE_FRAC)), 2),
        }
    return out


def simulate_matchup(home_team_id, away_team_id, n=DEFAULT_SIM_DRAWS, before_date=None,
                      window_games=10, seasons=None, rng=None, pools=None):
    """Top-level entry point: runs `n` full, independent game simulations
    for one real matchup. Returns {'players': {'home': {player_id:
    {'sim', 'sim_min', 'sim_median', 'sim_max', 'sim_mean', 'sim_stats',
    'player_name'}}, 'away': {...}}, 'game': {'n', 'home_wins',
    'away_wins', 'ties', 'home_win_pct', 'away_win_pct', 'home_avg_score',
    'away_avg_score', 'home_period_avg', 'away_period_avg', 'home_tilt',
    'away_tilt'}} — same overall shape as NFL's simulate_matchup(), see
    that function's own docstring, minus any DST_* entries (no basketball
    equivalent).

    `pools` (optional): a pre-built SimPools — pass this when simulating
    multiple matchups in one run (see simulate_slate()) so the pool,
    which doesn't vary by matchup, is only built once."""
    rng = rng or random.Random()
    pools = pools or build_sim_pools(seasons=seasons)

    home_shares = nba_stats_db.get_team_usage_shares_db(home_team_id, before_date=before_date,
                                                         window_games=window_games)
    away_shares = nba_stats_db.get_team_usage_shares_db(away_team_id, before_date=before_date,
                                                         window_games=window_games)

    home_log = nba_stats_db.get_team_game_log_db(home_team_id, before_date=before_date, limit=window_games)
    away_log = nba_stats_db.get_team_game_log_db(away_team_id, before_date=before_date, limit=window_games)
    home_allowed = (sum(g['points_allowed'] for g in home_log) / len(home_log)) if home_log else None
    away_allowed = (sum(g['points_allowed'] for g in away_log) / len(away_log)) if away_log else None
    # home's tilt reflects how leaky AWAY's defense has been (what home's
    # offense faces tonight), and symmetrically for away — opponent_factor()
    # returns ~1.0 for a league-average defense, >1.0 for a leaky one.
    home_tilt = (nba_player_model.opponent_factor(away_allowed) - 1.0) if away_allowed else 0.0
    away_tilt = (nba_player_model.opponent_factor(home_allowed) - 1.0) if home_allowed else 0.0

    draws = {'home': defaultdict(list), 'away': defaultdict(list)}
    stat_draws = {'home': defaultdict(lambda: defaultdict(list)), 'away': defaultdict(lambda: defaultdict(list))}
    home_wins = away_wins = ties = 0
    scores = {'home': [], 'away': []}
    period_points = {'home': [0] * PERIODS_REGULATION, 'away': [0] * PERIODS_REGULATION}

    for _ in range(n):
        game = simulate_game(pools, rng, home_tilt=home_tilt, away_tilt=away_tilt)
        for side, shares in (('home', home_shares), ('away', away_shares)):
            total = game[side]['total']
            scores[side].append(total)
            for i, pts in enumerate(game[side]['period_points']):
                period_points[side][min(i, PERIODS_REGULATION - 1)] += pts
            player_cats = attribute_game_to_players(total, shares, rng)
            for player_id, categories in player_cats.items():
                draws[side][player_id].append(nba_fantasy.compute_fantasy_points(categories))
                for field, value in categories.items():
                    stat_draws[side][player_id][field].append(value)
        if game['home']['total'] > game['away']['total']:
            home_wins += 1
        elif game['away']['total'] > game['home']['total']:
            away_wins += 1
        else:
            ties += 1

    players = {}
    for side, shares in (('home', home_shares), ('away', away_shares)):
        side_out = {}
        for player_id, pts in draws[side].items():
            ranked = sorted(pts)
            sim_stats = {}
            for field, values in stat_draws[side].get(player_id, {}).items():
                if field in COUNT_STAT_FIELDS:
                    sim_stats[field] = round(sum(values) / len(values), 2)
                else:
                    field_ranked = sorted(values)
                    sim_stats[field] = field_ranked[len(field_ranked) // 2]
            side_out[player_id] = {
                'player_name': shares.get(player_id, {}).get('name'),
                'sim': pts,
                'sim_min': ranked[0],
                'sim_median': ranked[len(ranked) // 2],
                'sim_max': ranked[-1],
                'sim_mean': round(sum(pts) / len(pts), 2),
                'sim_stats': sim_stats,
            }
        players[side] = side_out

    game_summary = {
        'n': n,
        'home_wins': home_wins, 'away_wins': away_wins, 'ties': ties,
        'home_win_pct': round(100 * home_wins / n, 1),
        'away_win_pct': round(100 * away_wins / n, 1),
        'home_avg_score': round(sum(scores['home']) / n, 1),
        'away_avg_score': round(sum(scores['away']) / n, 1),
        'home_period_avg': [round(q / n, 1) for q in period_points['home']],
        'away_period_avg': [round(q / n, 1) for q in period_points['away']],
        'home_tilt': round(home_tilt, 3), 'away_tilt': round(away_tilt, 3),
    }
    return {'players': players, 'game': game_summary}


def simulate_slate(date=None, n=DEFAULT_SIM_DRAWS, before_date=None, window_games=10,
                    seasons=None, rng=None):
    """Runs simulate_matchup() for every game on one NBA slate date and
    PERSISTS the results (nba_stats_db.persist_sim_results_db()/
    persist_sim_game_summary_db()) instead of returning them — this is
    the function app.py's daily background job (_simulate_nba_slate())
    calls; /nba/simulations then reads what this wrote rather than ever
    calling this directly on a request. NBA plays nightly rather than
    weekly, so this simulates one DATE's slate (default: today ET, via
    nba_api.build_schedule_context()) rather than a week the way NFL's
    simulate_week() does.

    `date` (YYYY-MM-DD, optional): defaults to today ET. Pass an explicit
    past date to backtest against an already-played slate (combine with
    `before_date` for genuine no-leakage, same discipline nfl_game_
    simulator.simulate_week() documents).

    No separate roster-name resolution step is needed here the way NFL's
    simulate_week() needs one (nfl_stats_db.get_or_fetch_roster_db()) —
    nba_stats_db.get_team_usage_shares_db() already carries each player's
    name straight off their own stored box-score rows (PlayerGameStat.
    player_name), so simulate_matchup() attaches it directly."""
    import nba_api

    rng = rng or random.Random()
    schedule = nba_api.build_schedule_context(target_date=date)
    if not schedule:
        return
    day = schedule[0]
    resolved_date = day['date']
    games = day['games']
    if not games:
        return

    pools = build_sim_pools(seasons=seasons)

    for g in games:
        home_id, away_id = g.get('home', {}).get('id'), g.get('away', {}).get('id')
        if not home_id or not away_id:
            continue
        sim = simulate_matchup(home_id, away_id, n=n, before_date=before_date,
                                window_games=window_games, rng=rng, pools=pools)
        for side, team_id, opponent_id in (('home', home_id, away_id), ('away', away_id, home_id)):
            results = [
                {'player_id': player_id, 'player_name': outcome.get('player_name'),
                 'n_draws': len(outcome['sim']), **outcome}
                for player_id, outcome in sim['players'][side].items()
            ]
            nba_stats_db.persist_sim_results_db(resolved_date, g['game_id'], team_id, opponent_id, results)
        nba_stats_db.persist_sim_game_summary_db(resolved_date, g['game_id'], home_id, away_id, sim['game'])
