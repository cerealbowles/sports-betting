"""
wnba_ensemble_model.py — Blends wnba_model.py (team-aggregate) and
wnba_player_model.py (player-level) into one combined moneyline/spread
signal, using weights fit by wnba_ensemble_bootstrap.py. Mirrors
nba_ensemble_model.py — same blend mechanics, WNBA side.

Fit provenance: see wnba_ensemble_bootstrap.py's printed `fit:` output —
PROB_FIT/MARGIN_FIT below should be copied from a real run against this
sport's own backfilled seasons (wnba_stats_backfill.py), not inherited from
NBA's numbers.

Refit periodically: rerun wnba_ensemble_bootstrap.py and copy its printed
`fit:` lines into PROB_FIT/MARGIN_FIT below as more games accumulate.
"""
import math

# {blended_prob: (intercept, team_weight, player_weight)} — applied to
# logit(team_prob)/logit(player_prob), i.e. logistic-regression stacking.
PROB_FIT = {'intercept': 0.0, 'team_weight': 1.0, 'player_weight': 0.0}

# {blended_margin: (intercept, team_weight, player_weight)} — applied to
# each model's own implied point margin directly (OLS, not logit space).
MARGIN_FIT = {'intercept': 0.0, 'team_weight': 1.0, 'player_weight': 0.0}


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def blend_prob(team_prob, player_prob):
    """Combined home win probability from wnba_model.py's and
    wnba_player_model.py's own probabilities. Returns None if either input
    is None (can't blend with a missing signal)."""
    if team_prob is None or player_prob is None:
        return None
    f = PROB_FIT
    x = f['intercept'] + f['team_weight'] * _logit(team_prob) + f['player_weight'] * _logit(player_prob)
    return _sigmoid(x)


def blend_margin(team_margin, player_margin):
    """Combined point margin (positive = home favored) from wnba_model.py's
    spread_proxy-implied margin and wnba_player_model.py's own margin.
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
