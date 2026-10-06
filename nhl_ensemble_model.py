"""
nhl_ensemble_model.py — Blends nhl_model.py (team-aggregate) and
nhl_player_model.py (skater + goalie grades) into one combined moneyline
signal, using weights fit by nhl_ensemble_bootstrap.py. Mirrors
nfl_ensemble_model.py's mechanics exactly (logistic-regression stacking
on each model's own logit, probability-only — see nhl_player_model.py's
docstring for why there's no honest point-margin to blend here either).

Fit provenance (nhl_ensemble_bootstrap.py, 2024+2025+2026 seasons,
strictly point-in-time team/player state — no hindsight leakage, n=2,589):
    team-aggregate alone:   56.1% win accuracy, 0.2451 Brier
    player-grade alone:     53.6% win accuracy, 0.2484 Brier
    blended (this module):  56.0% win accuracy, 0.2445 Brier

HONEST READ: this blend does NOT meaningfully beat the team-only model —
accuracy is a hair LOWER (56.0% vs 56.1%, within noise) and Brier improves
only marginally (0.2451 -> 0.2445). Unlike NBA/NFL/WNBA, where the player
signal produced a real lift, hockey's skater/goalie box-score grade just
doesn't add much here, consistent with the sport's reputation as the most
parity-driven and highest-variance of the ones this app models — see
nhl_player_model.py's docstring for the same finding at the player-model-
alone level. Deployed anyway (the fitted player_weight, 0.1144, is small
but positive, and the Brier number is technically better) so the live app
reflects the real fit rather than a hand-picked team-only override, but
don't expect this to meaningfully move NHL picks the way the other three
sports' blends do.

Refit periodically: rerun nhl_ensemble_bootstrap.py and copy its printed
`fit:` line into PROB_FIT below as more games accumulate.
"""
import math

# {intercept, team_weight, player_weight} — applied to
# logit(team_prob)/logit(player_prob), i.e. logistic-regression stacking.
PROB_FIT = {'intercept': 0.0867, 'team_weight': 0.6351, 'player_weight': 0.1144}


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
