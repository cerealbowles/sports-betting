"""
nba_ensemble_model.py — Blends nba_model.py (team-aggregate) and
nba_player_model.py (player-level) into one combined moneyline/spread
signal, using weights fit by nba_ensemble_bootstrap.py.

Wired into the live app via nba_api.py's _build_game() (moneyline/favorite
pick, edge %, Model Performance tracking) — see that module for exactly
what reads model['home_prob'] post-blend. The margin blend (model
['ensemble_margin']) and total blend (new, this module's TOTAL_FIT) are
used for parallel spread/total TRACKING only (app.py's blended_* columns
on GamePrediction) — not yet feeding the live Spread/Total buttons
themselves, which still read the team-only signal. See app.py's
_upsert_predictions for the tracking wiring and this file's TOTAL_FIT
comment for why this is being run in parallel rather than swapped in.

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

# {blended_total: (intercept, team_weight, player_weight)} — applied to each
# side's own combined-score projection directly (OLS, not logit space).
# team_total here is a PPG-sum proxy (not the real pace-adjusted
# bball_total_model.py projection) — see nba_ensemble_bootstrap.py's
# row-collection comment for why; this blend's RMSE isn't directly
# comparable to the real total model's own accuracy for that reason.
#
# Fit provenance (nba_ensemble_bootstrap.py, same 2024+2025 games as the
# prob/margin fits above, n=2,461): team-only (PPG-sum) RMSE 19.70,
# player-only RMSE 21.11, blended RMSE 18.84 — a real improvement over the
# PPG-sum proxy, tracked in parallel against the real total model (see
# app.py's blended_total_* GamePrediction columns) rather than assumed to
# transfer to it.
TOTAL_FIT = {'intercept': 31.454, 'team_weight': 0.507, 'player_weight': 0.362}


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


def blend_total(team_total, player_total):
    """Combined projected combined score (home+away) from a team-only
    PPG-sum proxy and nba_player_model.py's own 'total' projection. Returns
    None if either input is None. See TOTAL_FIT's comment for the
    team_total-is-a-proxy caveat — this is a rougher signal than the real
    pace-adjusted total model, tracked in parallel to see if it helps."""
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
