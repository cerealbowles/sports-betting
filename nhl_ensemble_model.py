"""
nhl_ensemble_model.py — Blends nhl_model.py (team-aggregate) and
nhl_player_model.py (skater + goalie grades) into one combined moneyline
signal, using weights fit by nhl_ensemble_bootstrap.py. Mirrors
nfl_ensemble_model.py's mechanics exactly (logistic-regression stacking
on each model's own logit, probability-only — see nhl_player_model.py's
docstring for why there's no honest point-margin to blend here either).

Fit provenance: see nhl_ensemble_bootstrap.py's printed `fit:` output —
PROB_FIT below should be copied from a real run against this sport's own
backfilled seasons (nhl_stats_backfill.py), not assumed from NBA/NFL/
WNBA's numbers (hockey's goalie-centric signal is a different shape from
either of those).

Refit periodically: rerun nhl_ensemble_bootstrap.py and copy its printed
`fit:` line into PROB_FIT below as more games accumulate.
"""
import math

# {intercept, team_weight, player_weight} — applied to
# logit(team_prob)/logit(player_prob), i.e. logistic-regression stacking.
# Placeholder (team-only passthrough) until nhl_ensemble_bootstrap.py's
# real fit replaces this.
PROB_FIT = {'intercept': 0.0, 'team_weight': 1.0, 'player_weight': 0.0}


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def blend_prob(team_prob, player_prob):
    """Combined home win probability from nhl_model.py's and
    nhl_player_model.py's own probabilities. Returns None if either input
    is None."""
    if team_prob is None or player_prob is None:
        return None
    f = PROB_FIT
    x = f['intercept'] + f['team_weight'] * _logit(team_prob) + f['player_weight'] * _logit(player_prob)
    return _sigmoid(x)


def predict(team_prob, player_prob):
    """Convenience wrapper, or None if either model's signal is missing."""
    prob = blend_prob(team_prob, player_prob)
    if prob is None:
        return None
    return {'home_prob': round(prob, 4)}
