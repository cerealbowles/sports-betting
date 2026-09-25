"""
NBA win-probability model.

Logit-additive: each factor contributes a signed value in log-odds space.
sigmoid(sum) → home win probability.

Initial weights hand-tuned from historical NBA data (~58-60% home win rate,
the highest home-court edge of the four major US leagues). Run
nba_bootstrap.py after a season or two of real results to calibrate
coefficients via logistic regression. Back-to-back is a rare-ish event
(~15-20% of games) like NHL's — bootstrap coeff for it should be treated as
advisory, not definitive.
"""
import math

import market_edge_calibration as _mkt_calib

LEAGUE_PPG = 114.0   # rough modern-NBA per-team-per-game scoring average

# Model-vs-market shrink rate — data-driven, see market_edge_calibration.py.
# app.py's _recompute_market_edge_shrink('NBA') overwrites this in place as
# more resolved games accumulate; this is just the value at import time
# (last snapshot, or the hardcoded 25% prior on first run).
_MKT_EDGE_RATE = _mkt_calib.load_rate('NBA')


def _sigmoid(x):
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _pct(wins, losses):
    total = wins + losses
    return wins / total if total else 0.5


def _safe_float(val, default):
    try:
        v = float(val)
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


def predict(home, away, game_time_utc=None, market_home_prob=None):
    """
    Args:
        market_home_prob – vig-free market-implied home win prob (0-1), if known.
                            Used only for the large-disagreement shrink below;
                            the model runs identically without it.
    Returns:
        home_prob  – estimated home win probability (float 0-1)
        away_prob  – 1 - home_prob
        factors    – ordered list of (label, contribution) for display
    """
    logit = 0.0
    factors = []

    def _add(label, contrib):
        nonlocal logit
        logit += contrib
        factors.append((label, round(contrib, 3)))

    # ── 1. Home court advantage (~58-60% historically, largest of the 4 leagues) ──
    _add('Home court', 0.32)

    # ── 2. Overall win-percentage differential ────────────────────────────────
    # 'blend_win_pct' (set by nba_api._apply_prior_season_blend) mixes in last
    # season's final record for teams with <EARLY_SEASON_GAMES games played
    # this season — otherwise every team defaults to an identical 0-0 (50%)
    # record at the start of the season, and this factor (and #3-5 below)
    # contribute exactly 0 for every early-season game. Same fix as
    # nfl_api.py's EARLY_SEASON_GAMES blend.
    h_wp = home.get('blend_win_pct')
    if h_wp is None:
        h_wp = _pct(home.get('wins', 0), home.get('losses', 0))
    a_wp = away.get('blend_win_pct')
    if a_wp is None:
        a_wp = _pct(away.get('wins', 0), away.get('losses', 0))
    _add('Win percentage', (h_wp - a_wp) * 2.0)

    # ── 3. Home / road split record ───────────────────────────────────────────
    h_split = home.get('blend_split_pct')
    if h_split is None:
        h_split = _pct(home.get('split_w', 0), home.get('split_l', 0))
    a_split = away.get('blend_split_pct')
    if a_split is None:
        a_split = _pct(away.get('split_w', 0), away.get('split_l', 0))
    _add('Home/road record', (h_split - a_split) * 0.8)

    # ── 4. Offensive efficiency — points scored per game ─────────────────────
    h_ppg = _safe_float(home.get('blend_ppg', home.get('ppg')), LEAGUE_PPG)
    a_ppg = _safe_float(away.get('blend_ppg', away.get('ppg')), LEAGUE_PPG)
    _add('Points per game', (h_ppg - a_ppg) * 0.04)

    # ── 5. Defensive efficiency — points allowed per game ────────────────────
    h_ppga = _safe_float(home.get('blend_ppg_allowed', home.get('ppg_allowed')), LEAGUE_PPG)
    a_ppga = _safe_float(away.get('blend_ppg_allowed', away.get('ppg_allowed')), LEAGUE_PPG)
    _add('Points allowed/G', (a_ppga - h_ppga) * 0.04)

    # ── 6. Recent form — last 5 games ─────────────────────────────────────────
    h_form = home.get('form', [])
    a_form = away.get('form', [])
    if h_form and a_form:
        h_f = h_form.count('W') / len(h_form)
        a_f = a_form.count('W') / len(a_form)
        _add('Recent form (L5)', (h_f - a_f) * 0.35)

    # ── 7. Back-to-back fatigue ────────────────────────────────────────────────
    # NBA teams play ~3-4x/week; a back-to-back (rest_days <= 1) is a well-
    # documented significant fatigue factor, unlike NFL's weekly cadence.
    h_rest = home.get('rest_days')
    a_rest = away.get('rest_days')
    if h_rest is not None and a_rest is not None:
        h_b2b = h_rest <= 1
        a_b2b = a_rest <= 1
        b2b_contrib = 0.0
        if h_b2b and not a_b2b:
            b2b_contrib = -0.22
        elif a_b2b and not h_b2b:
            b2b_contrib = 0.22
        # Single label (even when 0) so the factor accumulates one consistent
        # sample for the Factor Correlation table instead of splitting across
        # two rare home/away labels that individually never clear n>=5.
        _add('Back-to-back', b2b_contrib)

    home_prob = _sigmoid(logit)

    # Shrink large model-vs-market disagreements — same guardrail added to
    # mlb_model.py, where 50 resolved games found 10+ point edges over the
    # vig-free market won only 25% of the time vs. ~58% under that threshold.
    # The 10pt threshold is fixed; the pull-back rate (_MKT_EDGE_RATE) is
    # data-driven — app.py's _recompute_market_edge_shrink('NBA') refits it
    # from resolved game_predictions as NBA accumulates its own large-edge
    # games, starting from the MLB-derived 25% prior until then.
    if market_home_prob is not None:
        _MKT_EDGE_CAP = 0.10
        edge = home_prob - market_home_prob
        excess = abs(edge) - _MKT_EDGE_CAP
        if excess > 0:
            pull = excess * _MKT_EDGE_RATE
            home_prob -= pull if edge > 0 else -pull

    return {
        'home_prob': round(home_prob, 4),
        'away_prob': round(1.0 - home_prob, 4),
        'logit':     round(logit, 3),
        'factors':   factors,
    }
