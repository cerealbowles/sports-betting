"""
nba_ensemble_model.py — Blends nba_model.py (team-aggregate) and
nba_player_model.py (player-level) into one combined moneyline/spread
signal, using weights fit by nba_ensemble_bootstrap.py.

STILL UNWIRED INTO THE LIVE APP — this module exists so the fitted blend
has one callable home (predict() below) rather than living only inside the
bootstrap script, per the "run alongside, compare first" rollout decision.
Promoting it into app.py/the live game cards is a separate, later decision.

Fit provenance (nba_ensemble_bootstrap.py, 2,453 games, 2024+2025 seasons,
strictly point-in-time team/player state — no hindsight leakage):
    team-aggregate alone:  65.0% win accuracy, 0.2204 Brier
    player-level alone:    64.2% win accuracy, 0.2338 Brier
    blended (this module): 66.7% win accuracy, 0.2106 Brier
    margin RMSE: team-only 14.82, player-only 17.58, blended 14.55

IMPORTANT CORRECTION this fit surfaced: nba_bootstrap.py's own reported
73.0% team-model accuracy is OVERSTATED — it computes each team's season
PPG/PPG-allowed from the full season (including games after the one being
predicted), a hindsight leak. The 65.0% team-only figure above is the
honest, leak-free number and is what this blend should be compared against,
not nba_bootstrap.py's. See nba_ensemble_bootstrap.py for the point-in-time
state reconstruction that fixes this (and consider fixing nba_bootstrap.py
itself to match, in a follow-up — it's still used standalone).

Refit periodically: rerun nba_ensemble_bootstrap.py and copy its printed
`fit:` lines into FIT below as more games accumulate.
"""
import math

# {blended_prob: (intercept, team_weight, player_weight)} — applied to
# logit(team_prob)/logit(player_prob), i.e. logistic-regression stacking.
PROB_FIT = {'intercept': -0.0607, 'team_weight': 0.5176, 'player_weight': 0.2481}

# {blended_margin: (intercept, team_weight, player_weight)} — applied to
# each model's own implied point margin directly (OLS, not logit space).
MARGIN_FIT = {'intercept': -0.1934, 'team_weight': 0.7676, 'player_weight': 0.2244}


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def blend_prob(team_prob, player_prob):
    """Combined home win probability from nba_model.py's and
    nba_player_model.py's own probabilities. Returns None if either input
    is None (can't blend with a missing signal)."""
    if team_prob is None or player_prob is None:
        return None
    f = PROB_FIT
    x = f['intercept'] + f['team_weight'] * _logit(team_prob) + f['player_weight'] * _logit(player_prob)
    return _sigmoid(x)


def blend_margin(team_margin, player_margin):
    """Combined point margin (positive = home favored) from nba_model.py's
    spread_proxy-implied margin and nba_player_model.py's own margin.
    Returns None if either input is None."""
    if team_margin is None or player_margin is None:
        return None
    f = MARGIN_FIT
    return f['intercept'] + f['team_weight'] * team_margin + f['player_weight'] * player_margin


def predict(team_prob, player_prob, team_margin, player_margin):
    """Convenience wrapper bundling both blends into one result dict, or
    None if either model's signal is missing for this game."""
    prob = blend_prob(team_prob, player_prob)
    margin = blend_margin(team_margin, player_margin)
    if prob is None or margin is None:
        return None
    return {'home_prob': round(prob, 4), 'margin': round(margin, 2)}
