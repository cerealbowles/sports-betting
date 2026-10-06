"""
nfl_ensemble_model.py — Blends nfl_model.py (team-aggregate) and
nfl_player_model.py (offense/defense unit grades) into one combined
moneyline signal, using weights fit by nfl_ensemble_bootstrap.py. Mirrors
nba_ensemble_model.py — see that module for the identical blend mechanics
(logistic-regression stacking on each model's own logit).

Probability-only blend, unlike NBA's (which also blends margin) — the
player-grade model's matchup_diff is in ratio/grade space, not points, so
there's no honest point-margin to blend; see nfl_player_model.predict()'s
docstring for why.

Fit provenance (nfl_ensemble_bootstrap.py, 2024+2025 seasons, strictly
point-in-time team/player state — no hindsight leakage, n=512):
    team-aggregate alone:   64.6% win accuracy, 0.2357 Brier
    player-grade alone:     58.0% win accuracy, 0.2415 Brier
    blended (this module):  65.2% win accuracy, 0.2231 Brier

Smaller lift than NBA's blend (+0.6pp accuracy vs. NBA's +1.7pp), and the
fitted player_weight is slightly NEGATIVE (-0.0807) — the player-grade
signal alone is the weaker of this app's two sports-with-a-player-model,
consistent with football's score not being additively decomposable from
box-score stats the way basketball's is (see nfl_player_model.py's
docstring). The Brier improvement (0.2357 -> 0.2231) is real and larger
proportionally than the accuracy gain — logistic stacking can extract a
calibration benefit from a weakly-correlated second signal even when its
own standalone accuracy trails, which is what a small negative weight
after controlling for the team signal's own logit usually means: a slight
over-correction, not literally "ignore the player model and subtract it."

Refit periodically: rerun nfl_ensemble_bootstrap.py and copy its printed
`fit:` line into PROB_FIT below as more games accumulate.
"""
import math

# {intercept, team_weight, player_weight} — applied to
# logit(team_prob)/logit(player_prob), i.e. logistic-regression stacking.
PROB_FIT = {'intercept': 0.0171, 'team_weight': 0.4677, 'player_weight': -0.0807}


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def blend_prob(team_prob, player_prob):
    """Combined home win probability from nfl_model.py's and
    nfl_player_model.py's own probabilities. Returns None if either input
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
