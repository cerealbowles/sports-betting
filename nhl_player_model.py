"""
nhl_player_model.py — Player-level NHL signal: a SKATER offense grade
(team-wide, same ratio-to-league-average approach as nfl_player_model.py)
plus a DEDICATED GOALIE grade, graded on that specific goaltender's own
recent save rate rather than folded into one team-wide aggregate.

Goaltending is this sport's single highest-leverage individual signal —
arguably more so than a starting QB is to an NFL offense, since one
goalie plays nearly every minute of every game and save-percentage
spreads between a true #1 and a backup are large relative to a league
that averages well under 3.5 goals/game either way. Treating it as a
dedicated signal (rather than diluting it into skater-style team
aggregates) is this module's one real departure from nfl_player_model.py's
shape, and the reason get_team_games_with_players_db() preserves
per-player identity (including is_goalie) instead of collapsing straight
to a team total the way nba_stats_db.get_team_game_aggregates_db() does.

Mirrors nfl_player_model.py's other two decisions directly: offense isn't
projected as a point total (hockey's low-scoring, highly variance-driven
scoring doesn't decompose into a believable score-a-team-will-get number
any more cleanly than football's does — see that module's docstring for
the full reasoning), and a logistic is fit directly on the matchup
differential rather than hand-deriving a goals scale.

UNVALIDATED until nhl_player_bootstrap.py's backtest replaces INTERCEPT/
SLOPE below with real fitted values.
"""
import math

SKATER_WINDOW = 10
GOALIE_WINDOW = 10

# Rough NHL league-average baselines (not fit from this app's data — same
# "reasonable prior" role as nfl_player_model.LEAGUE_AVG).
LEAGUE_AVG = {
    'goals_pg':  3.0,     # team goals per game
    'shots_pg':  30.0,    # team shots on goal per game
    'save_pct':  0.900,   # goaltender save percentage
}

# Below this many shots faced (summed across the goalie window, after
# excluding anyone confirmed OUT), a save-percentage sample is too thin to
# trust — same "excluding a workhorse leaves near-nothing behind" problem
# nfl_player_model.py's MIN_VOLUME/REPLACEMENT_LEVEL_RATIO was built to
# fix, just hockey's version of it. ~50 shots is roughly 1.5-2 games'
# worth of a backup's relief appearances.
MIN_GOALIE_SHOTS = 50
REPLACEMENT_LEVEL_RATIO = 0.90  # below-average "backup goalie" prior for a too-thin sample


def _num(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _toi_to_minutes(toi_str):
    """'17:46' -> 17.77 minutes. 0.0 on anything unparseable (DNP, empty)."""
    try:
        m, s = str(toi_str).split(':')
        return float(m) + float(s) / 60.0
    except (ValueError, AttributeError):
        return 0.0


def parse_team_skater_game(player_rows):
    """player_rows: this team's players for ONE game (the shape
    nhl_stats_db.get_team_games_with_players_db() returns per game) —
    goalies are skipped here (is_goalie True), graded separately below.
    Returns one team-skater aggregate dict for that game."""
    goals = assists = shots = plus_minus = giveaways = takeaways = 0.0
    for row in player_rows:
        if row.get('is_goalie'):
            continue
        s = row.get('stats', {})
        goals     += _num(s.get('goals'))
        assists   += _num(s.get('assists'))
        shots     += _num(s.get('shotsTotal'))
        plus_minus += _num(s.get('plusMinus'))
        giveaways += _num(s.get('giveaways'))
        takeaways += _num(s.get('takeaways'))
    return {
        'goals': goals, 'assists': assists, 'shots': shots,
        'plus_minus': plus_minus, 'giveaways': giveaways, 'takeaways': takeaways,
    }


def compute_skater_grade(games_with_players, window=SKATER_WINDOW):
    """games_with_players: this team's recent games, newest first, each a
    list of per-player rows (raw, not yet collapsed — parse_team_skater_game()
    runs per game here). Returns a single float grade (1.0 = league
    average, higher = better offense) or None if no usable sample."""
    sample = [parse_team_skater_game(g) for g in games_with_players[:window]]
    if not sample:
        return None
    n = len(sample)

    goals_pg = sum(g['goals'] for g in sample) / n
    shots_pg = sum(g['shots'] for g in sample) / n
    pm_pg    = sum(g['plus_minus'] for g in sample) / n
    puck_mgmt_pg = (sum(g['takeaways'] for g in sample) - sum(g['giveaways'] for g in sample)) / n

    ratios = [
        goals_pg / LEAGUE_AVG['goals_pg'],
        shots_pg / LEAGUE_AVG['shots_pg'],
        1.0 + pm_pg / 10.0,          # +/- swings are small in raw point terms; scaled down
        1.0 + puck_mgmt_pg / 10.0,   # positive = takes the puck away more than it gives it up
    ]
    return sum(ratios) / len(ratios)


def identify_starting_goalie(games_with_players, window=GOALIE_WINDOW):
    """Returns (player_id, name) for whoever logged the most total ice
    time among this team's goalies over the last `window` games — the
    hockey equivalent of nfl_player_model.identify_key_players()'s
    usage-based QB identification. None if no goalie appears at all."""
    toi, names = {}, {}
    for game in games_with_players[:window]:
        for row in game:
            if not row.get('is_goalie'):
                continue
            pid = row.get('player_id')
            if not pid:
                continue
            names[pid] = row.get('player_name', '')
            toi[pid] = toi.get(pid, 0.0) + _toi_to_minutes(row.get('stats', {}).get('timeOnIce'))
    if not toi:
        return None
    best_id = max(toi, key=toi.get)
    return (best_id, names[best_id])


def compute_goalie_grade_excluding(games_with_players, unavailable_ids, window=GOALIE_WINDOW):
    """Team's goaltending grade (1.0 = league-average save%, higher =
    stingier), excluding any confirmed-OUT goalie's own appearances —
    same exclude-and-regrade approach as nfl_player_model.compute_offense_
    grade_excluding(), with the same replacement-level fallback for a too-
    thin post-exclusion sample (see MIN_GOALIE_SHOTS above — this is
    exactly the scenario that matters most here: a team's actual #1
    goalie being out is the single biggest injury-driven swing in this
    sport, bigger than any skater's absence)."""
    total_saves = total_shots_against = 0.0
    for game in games_with_players[:window]:
        for row in game:
            if not row.get('is_goalie') or row.get('player_id') in unavailable_ids:
                continue
            s = row.get('stats', {})
            if _toi_to_minutes(s.get('timeOnIce')) <= 0:
                continue
            total_saves += _num(s.get('saves'))
            total_shots_against += _num(s.get('shotsAgainst'))

    if total_shots_against < MIN_GOALIE_SHOTS:
        return REPLACEMENT_LEVEL_RATIO
    save_pct = total_saves / total_shots_against
    return save_pct / LEAGUE_AVG['save_pct']


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# Fit by nhl_player_bootstrap.py: P(home_win) ~ sigmoid(INTERCEPT + SLOPE *
# matchup_diff), where matchup_diff = (home_skater - away_goalie) -
# (away_skater - home_goalie) — mirrors nfl_player_model.py's matchup-diff
# approach exactly. Placeholder until a real backtest replaces these.
INTERCEPT = 0.0
SLOPE     = 1.0


def predict(home_skater, home_goalie, away_skater, away_goalie):
    """All four args are compute_skater_grade()/compute_goalie_grade_excluding()
    outputs for each side. Returns {'home_prob': float, 'matchup_diff':
    float}, or None if any of the four is missing. No margin/total output —
    same reasoning as nfl_player_model.predict()'s docstring: these are
    ratio-space grades, not point projections."""
    if None in (home_skater, home_goalie, away_skater, away_goalie):
        return None
    matchup_diff = (home_skater - away_goalie) - (away_skater - home_goalie)
    home_prob = _sigmoid(INTERCEPT + SLOPE * matchup_diff)
    return {'home_prob': round(home_prob, 4), 'matchup_diff': round(matchup_diff, 4)}
