"""
nfl_player_model.py — Offense/defense unit-grade signal for NFL, built from
the per-player box score data in the local stats warehouse (nfl_stats_db.py)
/ nfl_boxscore_api.py's parsing. A second, independent signal alongside
nfl_model.py's team-aggregate model (see nba_player_model.py for the NBA
equivalent this mirrors — same two-signal-blend idea, different internals).

UNVALIDATED until nfl_player_bootstrap.py's backtest numbers are copied into
nfl_ensemble_model.py's FIT constants (see that module once it exists).

Why this ISN'T a port of nba_player_model.py's "sum player projections to a
team score": basketball score is close to additive (minutes x per-minute
rate, summed across 5 players on the floor). Football isn't — a team's
score comes from drives, not a sum of individual box-score lines, and
skill-position production (receiving yards, rushing yards) has no clean
conversion to points the way a basketball player's own points do. Instead
of projecting a score, this computes OFFENSE and DEFENSE unit grades from
each side's recent box-score rates, each expressed as a ratio to a rough
league-average baseline (1.0 = league average, same ratio-to-baseline idea
as nba_player_model.opponent_factor, just applied to both sides and more
stats), and lets nfl_player_bootstrap.py fit a logistic directly on the
matchup differential rather than trying to hand-derive a points scale.

Known simplifications (documented, not hidden):
  - League-average baselines below (LEAGUE_AVG) are rough constants, not
    fit from this app's own data — same role as nba_player_model.LEAGUE_AVG_PPG.
  - No opponent strength-of-schedule adjustment on the grades themselves
    (e.g. a defense grade built against a slate of bad offenses looks
    better than it is) — the matchup differential in predict() partially
    compensates by comparing offense-vs-defense rather than offense alone,
    but isn't a full SOS adjustment.
  - Grade components are combined as an unweighted average of ratios, not
    a fitted weighting — a reasonable starting prior (mirrors
    nba_player_model's own "reasonable prior, refine later" philosophy),
    not a claim that QB efficiency/rushing/receiving/turnovers all matter
    equally.
  - Turnover-worthy-play rate is the noisiest component of all of these at
    small sample sizes (a single tipped-ball INT swings a team's giveaway
    rate hard over 3-5 games) — expect this to matter less once more
    seasons of data let per-team variance average out.
"""
import math

QB_WINDOW   = 5   # recent games for QB/offense grade
DEF_WINDOW  = 5   # recent games for defense grade

# Rough league-average per-game baselines (NOT fit from this app's data —
# ballpark figures from recent NFL seasons, same "reasonable prior" role as
# nba_player_model.LEAGUE_AVG_PPG). Refitting these from the warehouse once
# it holds full seasons is a reasonable follow-up.
LEAGUE_AVG = {
    'qb_ay_a':     6.3,   # adjusted yards/attempt: (passYds + 20*TD - 45*INT) / attempts
    'rush_ypc':    4.2,   # rushing yards per carry
    'recv_ypt':    7.0,   # receiving yards per target
    'give_pg':     1.3,   # giveaways (INT thrown + fumbles lost) per game — LOWER is better
    'sacks_pg':    2.5,   # defense: sacks per game
    'tfl_pg':      5.5,   # defense: tackles for loss per game
    'takeaways_pg': 1.3,  # defense: INTs + fumble recoveries per game
    'pd_pg':       5.0,   # defense: passes defended per game
}


def _num(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _attempts_half(composite, default=0.0):
    """'22/40' -> 40.0 (completions/attempts, attempts half) or
    '5-30' -> 5.0 (sacks-sackYardsLost, sacks half) — both of football's
    box score's '/'-or-'-'-joined composite fields, first half by default
    unless `second=True` wants the other side."""
    try:
        return float(str(composite).split('/')[0].split('-')[0])
    except (ValueError, IndexError):
        return default


def _split(composite, sep, idx, default=0.0):
    try:
        return float(str(composite).split(sep)[idx])
    except (ValueError, IndexError):
        return default


def compute_offense_grade(games):
    """games: this TEAM's own recent games, newest first, each with
    'categories' — the same shape nfl_stats_db.get_player_gamelog_db()
    returns per player, pre-aggregated to team level by the caller (see
    nfl_player_bootstrap.py / nfl_api.py for how team-level aggregation
    from individual player rows happens — this function just grades
    whatever aggregate dict it's handed).

    `games` here is actually a list of per-game TEAM aggregate dicts:
    [{'pass_att', 'pass_yds', 'pass_td', 'pass_int', 'rush_att',
      'rush_yds', 'recv_tgt', 'recv_yds', 'giveaways'}, ...], newest first,
    already filtered to strictly before the game being projected by the
    caller (no-leakage requirement, same as nba_player_model.project_player).

    Returns a single float grade (1.0 = league average, higher = better
    offense) or None if there's no usable sample."""
    sample = games[:QB_WINDOW]
    if not sample:
        return None
    n = len(sample)

    pass_att = sum(g['pass_att'] for g in sample)
    pass_yds = sum(g['pass_yds'] for g in sample)
    pass_td  = sum(g['pass_td'] for g in sample)
    pass_int = sum(g['pass_int'] for g in sample)
    rush_att = sum(g['rush_att'] for g in sample)
    rush_yds = sum(g['rush_yds'] for g in sample)
    recv_tgt = sum(g['recv_tgt'] for g in sample)
    recv_yds = sum(g['recv_yds'] for g in sample)
    giveaways = sum(g['giveaways'] for g in sample)

    qb_ay_a  = (pass_yds + 20 * pass_td - 45 * pass_int) / pass_att if pass_att > 0 else LEAGUE_AVG['qb_ay_a']
    rush_ypc = rush_yds / rush_att if rush_att > 0 else LEAGUE_AVG['rush_ypc']
    recv_ypt = recv_yds / recv_tgt if recv_tgt > 0 else LEAGUE_AVG['recv_ypt']
    give_pg  = giveaways / n

    ratios = [
        qb_ay_a / LEAGUE_AVG['qb_ay_a'],
        rush_ypc / LEAGUE_AVG['rush_ypc'],
        recv_ypt / LEAGUE_AVG['recv_ypt'],
        2.0 - (give_pg / LEAGUE_AVG['give_pg']),  # inverted: fewer giveaways than average -> ratio > 1
    ]
    return sum(ratios) / len(ratios)


def compute_defense_grade(games):
    """Same idea as compute_offense_grade, defense side. `games`: this
    team's own recent games, newest first, pre-aggregated team-defense
    dicts: [{'sacks', 'tfl', 'takeaways', 'pass_defended'}, ...]. Returns a
    single float grade (1.0 = league average, higher = better defense) or
    None if no usable sample."""
    sample = games[:DEF_WINDOW]
    if not sample:
        return None
    n = len(sample)

    sacks_pg     = sum(g['sacks'] for g in sample) / n
    tfl_pg       = sum(g['tfl'] for g in sample) / n
    takeaways_pg = sum(g['takeaways'] for g in sample) / n
    pd_pg        = sum(g['pass_defended'] for g in sample) / n

    ratios = [
        sacks_pg / LEAGUE_AVG['sacks_pg'],
        tfl_pg / LEAGUE_AVG['tfl_pg'],
        takeaways_pg / LEAGUE_AVG['takeaways_pg'],
        pd_pg / LEAGUE_AVG['pd_pg'],
    ]
    return sum(ratios) / len(ratios)


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# Fit by nfl_player_bootstrap.py: P(home_win) ~ sigmoid(INTERCEPT + SLOPE *
# matchup_diff), where matchup_diff = (home_off - away_def) - (away_off - home_def).
# Fit provenance: nfl_player_bootstrap.py, 2024+2025 seasons, 528 usable
# games (point-in-time, no leakage): 59.5% win accuracy, 0.2394 Brier vs.
# a 53.0% home-field-always baseline. Refit periodically — rerun
# nfl_player_bootstrap.py and copy its printed `fit:` line here.
INTERCEPT = 0.1150
SLOPE     = 0.9956


def predict(home_offense, home_defense, away_offense, away_defense):
    """All four args are compute_offense_grade()/compute_defense_grade()
    outputs for each side (None if that side had no usable sample — returns
    None if either side is missing a grade on EITHER unit, since the
    matchup differential needs all four).

    Returns {'home_prob': float, 'matchup_diff': float} — no margin/total
    output (unlike nba_player_model.predict()): these are ratio-space
    grades, not point projections, so there's no honest way to turn
    matchup_diff into a point margin without fabricating a scale. A
    moneyline-only player signal is still a real, useful second opinion;
    nfl_ensemble_model.py's blend is probability-only for the same reason.
    """
    if home_offense is None or home_defense is None or away_offense is None or away_defense is None:
        return None
    matchup_diff = (home_offense - away_defense) - (away_offense - home_defense)
    home_prob = _sigmoid(INTERCEPT + SLOPE * matchup_diff)
    return {'home_prob': round(home_prob, 4), 'matchup_diff': round(matchup_diff, 4)}


def parse_team_game_aggregate(player_rows):
    """Collapses one team's per-player box score rows (categories dicts,
    as stored in PlayerGameStat.stats_json / returned by
    nfl_boxscore_api.parse_player_boxscore) for ONE game into the team-level
    aggregate dicts compute_offense_grade()/compute_defense_grade() expect.

    Returns (offense_dict, defense_dict) — see those functions' docstrings
    for the exact keys."""
    pass_att = pass_yds = pass_td = pass_int = 0.0
    rush_att = rush_yds = 0.0
    recv_tgt = recv_yds = 0.0
    fumbles_lost = 0.0
    sacks = tfl = takeaways = pass_defended = 0.0

    for row in player_rows:
        cats = row.get('categories', {})
        if 'passing' in cats:
            c = cats['passing']
            pass_att += _split(c.get('completions/passingAttempts', '0/0'), '/', 1)
            pass_yds += _num(c.get('passingYards'))
            pass_td  += _num(c.get('passingTouchdowns'))
            pass_int += _num(c.get('interceptions'))
        if 'rushing' in cats:
            c = cats['rushing']
            rush_att += _num(c.get('rushingAttempts'))
            rush_yds += _num(c.get('rushingYards'))
        if 'receiving' in cats:
            c = cats['receiving']
            recv_tgt += _num(c.get('receivingTargets'))
            recv_yds += _num(c.get('receivingYards'))
        if 'fumbles' in cats:
            fumbles_lost += _num(cats['fumbles'].get('fumblesLost'))
        if 'defensive' in cats:
            c = cats['defensive']
            sacks += _num(c.get('sacks'))
            tfl   += _num(c.get('tacklesForLoss'))
            pass_defended += _num(c.get('passesDefended'))
        if 'interceptions' in cats:
            takeaways += _num(cats['interceptions'].get('interceptions'))

    offense = {
        'pass_att': pass_att, 'pass_yds': pass_yds, 'pass_td': pass_td, 'pass_int': pass_int,
        'rush_att': rush_att, 'rush_yds': rush_yds,
        'recv_tgt': recv_tgt, 'recv_yds': recv_yds,
        'giveaways': pass_int + fumbles_lost,
    }
    defense = {
        'sacks': sacks, 'tfl': tfl, 'takeaways': takeaways, 'pass_defended': pass_defended,
    }
    return offense, defense
