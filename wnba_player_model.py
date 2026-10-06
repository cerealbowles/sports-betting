"""
wnba_player_model.py — Player-level WNBA score/margin/total projection.

A second, independent signal alongside wnba_model.py's team-aggregate model
(see the NBA player-model plan, mirrored here). Projects each active player's expected
production from their own recent games, sums to a team score, and derives
moneyline/spread/total from that — rather than team-season aggregates that
can't see "this team is missing its best player tonight."

UNVALIDATED — this has not been backtested yet (see wnba_player_bootstrap.py).
Do not wire into the live app until a backtest shows it beats
spread_proxy.py's documented 46.4% WNBA out-of-sample ATS figure.

Known simplifications (documented, not hidden):
  - No real pre-game confirmed-starters feed exists on ESPN's free API.
    "Active roster" is whatever the caller passes in (e.g. roster minus
    injury-flagged players) — a best-effort guess, not a confirmed lineup.
  - Projected minutes are redistributed proportionally across the active
    roster so they sum to a full team-game's worth, rather than modeling
    real rotation/usage changes when a specific player is out.
  - Opponent defensive adjustment (opponent_factor() below) scales a
    player-sum up/down by whether tonight's opponent allows more/fewer
    points than league average recently — a single team-wide multiplier,
    not a real per-player matchup (e.g. it can't see "this stingy defense
    is specifically bad against guards"). Included here from the start,
    mirroring nba_player_model.py's own history — that module's first
    backtest WITHOUT this adjustment scored worse than its constant-
    baseline Brier (raw recent-production sums alone don't carry the
    opponent-quality signal team win% already captures), which is why this
    file includes it from the start rather than repeating that finding
    here. See wnba_player_bootstrap.py for this sport's own numbers.
"""
import math

TEAM_GAME_MINUTES = 200.0  # 5 players x 40 min — WNBA quarters are 10 min
                            # (not the NBA's 12), so a full team-game is 200
                            # player-minutes, not 240. Ignores OT on purpose
                            # (same reasoning as nba_player_model.py: OT is a
                            # post-hoc game-length change, not a pre-game
                            # projection input).

# Placeholder until wnba_player_bootstrap.py fits a real residual std from
# actual backtested margins — same normal-approximation pattern as
# nba_player_model.py, just carrying its NBA-derived sigma over as a
# starting prior rather than a WNBA-specific one (no WNBA backtest has been
# run yet). Refresh this once wnba_player_bootstrap.py reports its own.
DEFAULT_SIGMA = 14.734

# Real-world WNBA home-court point advantage (distinct from wnba_model.py's
# own logit-space home-court term — this one's in raw points). Same
# placeholder caveat as DEFAULT_SIGMA above — carried over from
# nba_player_model.py's value, not yet WNBA-specific.
HOME_COURT_POINTS = 2.7

# Reference scoring environment used only to normalize the opponent-defense
# factor below (not fit from this app's own data). WNBA teams score
# meaningfully less per game than NBA teams (shorter games, slower pace) —
# recent WNBA seasons average roughly 83 PPG per team, vs. the NBA's ~114
# nba_player_model.py uses, so this does NOT just inherit that constant.
LEAGUE_AVG_PPG = 83.0


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def project_player(games, recent_window=10):
    """games: this player's own gamelog, newest first (as returned by
    wnba_roster_api.get_player_gamelog), already filtered to games strictly
    before the game being projected (caller's responsibility — see
    wnba_player_bootstrap.py for the no-leakage backtest version).

    Returns ppg/mpg (as before) plus two usage-weighted fields, averaged
    over the last `recent_window` games (or fewer if the player doesn't
    have that many yet this season — this naturally degrades to a
    season-to-date average early in the season, same idea as wnba_api.py's
    L5 'form' factor):

      usage_rate — scoring possessions used per minute on court
                   (FGA + 0.44*FTA + TOV, the standard possessions-used
                   estimator bball_total_model.py already uses team-wide,
                   here per player per minute).
      per_poss   — points scored per possession used: a volume-independent
                   efficiency rating. Distinguishes an efficient low-volume
                   player from a high-PPG "empty calorie" volume scorer who
                   needs a lot of possessions to get there — raw ppg/mpg
                   alone can't tell them apart, which matters once minutes
                   get redistributed (see project_team_score()).

    None if no games at all, or if the player never logged a scoring
    possession (usage_rate == 0 — can't define per_poss; vanishingly rare,
    e.g. a garbage-time DNP-adjacent stub game).
    """
    sample = games[:recent_window]
    if not sample:
        return None
    n = len(sample)
    mpg = sum(g['minutes'] for g in sample) / n
    ppg = sum(g['points'] for g in sample) / n
    poss_pg = sum(g['fga'] + 0.44 * g['fta'] + g['tov'] for g in sample) / n
    if mpg <= 0 or poss_pg <= 0:
        return None
    return {
        'ppg':        ppg,
        'mpg':        mpg,
        'usage_rate': poss_pg / mpg,
        'per_poss':   ppg / poss_pg,
        'n':          n,
    }


def project_team_score(active_projections, opp_def_factor=1.0):
    """active_projections: list of project_player() results (already
    None-filtered) for one team's active roster tonight.

    Minutes are redistributed to sum to a full team-game (200 — see
    TEAM_GAME_MINUTES) — this is what reallocates an inactive/injured
    player's usual minutes across the rest of the active roster, instead of
    just silently dropping their production. The redistribution is USAGE-
    WEIGHTED rather than uniform: each player keeps their own recent-average
    minutes as a baseline, and the gap between that total and 200 is
    handed out in proportion to
    usage_rate, not split evenly — in real rotations, a missing high-usage
    player's shots mostly go to the team's other high-usage players, not
    uniformly across the whole bench. Each player's EXTRA minutes are
    valued at their own (usage_rate x per_poss), same as their baseline
    minutes — this doesn't change their known per-minute scoring rate, only
    how much of the vacated playing time they're assumed to absorb.

    The result is then scaled by `opp_def_factor` (see opponent_factor()
    below). Returns None if no players are available (e.g. injury data
    wiped out the whole bench).
    """
    if not active_projections:
        return None
    total_mpg = sum(p['mpg'] for p in active_projections)
    total_usage = sum(p['usage_rate'] for p in active_projections)
    if total_mpg <= 0 or total_usage <= 0:
        return None

    extra_minutes = TEAM_GAME_MINUTES - total_mpg  # can be negative (deep bench glut)
    raw = 0.0
    for p in active_projections:
        usage_share = p['usage_rate'] / total_usage
        projected_minutes = p['mpg'] + extra_minutes * usage_share
        projected_minutes = max(0.0, projected_minutes)
        raw += projected_minutes * p['usage_rate'] * p['per_poss']
    return raw * opp_def_factor


def opponent_factor(recent_ppg_allowed, league_avg_ppg=LEAGUE_AVG_PPG):
    """How many points a team's recent opponents have been able to score on
    them, relative to league average — >1.0 means this defense allows more
    than average (projected scorers should get a bump facing them), <1.0
    means a stingier-than-average defense (projected scorers get pulled
    down). `recent_ppg_allowed` should be computed from games strictly
    before the game being projected (see wnba_player_bootstrap.py's
    team_history tracking for the no-leakage version) — using season-end
    hindsight here would leak future information into the backtest.
    Returns 1.0 (no adjustment) if recent_ppg_allowed is falsy/None, e.g.
    a team with no games played yet this season.
    """
    if not recent_ppg_allowed or league_avg_ppg <= 0:
        return 1.0
    return recent_ppg_allowed / league_avg_ppg


def predict(home_active, away_active, home_pace=None, away_pace=None,
            league_avg_pace=None, home_opp_factor=1.0, away_opp_factor=1.0,
            sigma=DEFAULT_SIGMA):
    """
    home_active/away_active: list of project_player() results for each
        team's active roster tonight (None entries already filtered out).
    home_pace/away_pace/league_avg_pace: team pace (possessions/game) from
        bball_total_model's pace fetch, if available — scales the raw
        player-sum by how fast/slow tonight's expected game pace runs
        relative to a league-average pace. Omit (None) to skip pace
        adjustment entirely (raw player-sum only).
    home_opp_factor: opponent_factor() for the AWAY team's defense — scales
        the HOME team's projected score (since the away team's defense is
        who the home team scores against tonight). away_opp_factor is the
        mirror: the HOME team's defense scaling the AWAY team's score.

    Returns None if either team has no usable active-player projections.
    """
    home_raw = project_team_score(home_active, opp_def_factor=home_opp_factor)
    away_raw = project_team_score(away_active, opp_def_factor=away_opp_factor)
    if home_raw is None or away_raw is None:
        return None

    if home_pace and away_pace and league_avg_pace:
        game_pace = (home_pace + away_pace) / 2
        pace_factor = game_pace / league_avg_pace
    else:
        pace_factor = 1.0

    home_score = home_raw * pace_factor
    away_score = away_raw * pace_factor

    margin = (home_score - away_score) + HOME_COURT_POINTS
    total = home_score + away_score
    home_prob = _phi(margin / sigma) if sigma > 0 else None

    return {
        'home_score_proj': round(home_score, 1),
        'away_score_proj': round(away_score, 1),
        'margin':          round(margin, 1),
        'total':           round(total, 1),
        'home_prob':       round(home_prob, 4) if home_prob is not None else None,
        'pace_factor':     round(pace_factor, 3),
    }
