"""
wnba_ensemble_model.py — Blends wnba_model.py (team-aggregate) and
wnba_player_model.py (player-level) into one combined moneyline/spread
signal, using weights fit by wnba_ensemble_bootstrap.py. Mirrors
nba_ensemble_model.py — same blend mechanics, WNBA side.

Fit provenance (wnba_ensemble_bootstrap.py, 2024+2025+2026 seasons,
strictly point-in-time team/player state — no hindsight leakage, n=524):
    team-aggregate alone:   64.7% win accuracy, 0.2179 Brier
    player-level alone:     63.7% win accuracy, 0.2302 Brier
    blended (this module):  67.0% win accuracy, 0.2146 Brier
    margin RMSE: team-only 13.13, player-only 14.47, blended 12.93

A bigger lift than NBA's own blend (+2.3pp accuracy here vs. NBA's
+1.7pp) — WNBA's player-level signal, while still trailing the team model
alone, combines with it more productively than NBA's does, unlike NFL's
(whose fitted player_weight actually came out negative).

Wired into the live app via wnba_api.py's _build_game() (moneyline/
favorite pick, edge %, Model Performance tracking). The margin blend
(model['ensemble_margin']) and total blend (TOTAL_FIT below) are used for
parallel spread/total TRACKING only (app.py's blended_* columns on
GamePrediction) — not yet feeding the live Spread/Total buttons
themselves, which still read the team-only signal.

Refit periodically: rerun wnba_ensemble_bootstrap.py and copy its printed
`fit:` lines into PROB_FIT/MARGIN_FIT/TOTAL_FIT below as more games
accumulate.
"""
import math

# {blended_prob: (intercept, team_weight, player_weight)} — applied to
# logit(team_prob)/logit(player_prob), i.e. logistic-regression stacking.
PROB_FIT = {'intercept': -0.037, 'team_weight': 0.647, 'player_weight': 0.269}

# {blended_margin: (intercept, team_weight, player_weight)} — applied to
# each model's own implied point margin directly (OLS, not logit space).
MARGIN_FIT = {'intercept': -0.052, 'team_weight': 0.576, 'player_weight': 0.283}

# {blended_total: (intercept, team_weight, player_weight)} — applied to each
# side's own combined-score projection directly (OLS, not logit space).
# team_total here is a PPG-sum proxy (not the real pace-adjusted
# bball_total_model.py projection) — see wnba_ensemble_bootstrap.py's
# row-collection comment for why. Placeholder until that script's own
# total-blend fit replaces this.
TOTAL_FIT = {'intercept': 0.0, 'team_weight': 1.0, 'player_weight': 0.0}


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


def blend_total(team_total, player_total):
    """Combined projected combined score (home+away) from a team-only
    PPG-sum proxy and wnba_player_model.py's own 'total' projection.
    Returns None if either input is None."""
    if team_total is None or player_total is None:
        return None
    f = TOTAL_FIT
    return f['intercept'] + f['team_weight'] * team_total + f['player_weight'] * player_total


def predict(team_prob, player_prob, team_margin, player_margin, team_total=None, player_total=None):
    """Convenience wrapper bundling the prob/margin blends (and, if both
    totals are supplied, the total blend) into one result dict, or None if
    either model's core signal (prob/margin) is missing for this game."""
    prob = blend_prob(team_prob, player_prob)
    margin = blend_margin(team_margin, player_margin)
    if prob is None or margin is None:
        return None
    out = {'home_prob': round(prob, 4), 'margin': round(margin, 2)}
    total = blend_total(team_total, player_total)
    if total is not None:
        out['total'] = round(total, 2)
    return out
