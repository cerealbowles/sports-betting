"""
NFL win-probability model.

Logit-additive: each factor contributes a signed value in log-odds space.
sigmoid(sum) → home win probability.

Initial weights hand-tuned from historical NFL data (~57% home win rate).
Run nfl_bootstrap.py after the 2024+2025 seasons to calibrate coefficients
via logistic regression. Bye-week and short-week factors cannot be fully
validated in bootstrap (rare events); keep hand-tuned, validate on Model page.
"""
import math

LEAGUE_PPG = 23.2   # NFL 2024 season average ~23.4 PPG


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


def predict(home, away, game_time_utc=None):
    """
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

    # ── 1. Home field advantage (~57% historically) ───────────────────────────
    _add('Home field', 0.24)

    # ── 2. Overall win-percentage differential ────────────────────────────────
    h_wp = _pct(home.get('wins', 0), home.get('losses', 0))
    a_wp = _pct(away.get('wins', 0), away.get('losses', 0))
    _add('Win percentage', (h_wp - a_wp) * 2.0)

    # ── 3. Home / road split record ───────────────────────────────────────────
    h_split = _pct(home.get('split_w', 0), home.get('split_l', 0))
    a_split = _pct(away.get('split_w', 0), away.get('split_l', 0))
    _add('Home/road record', (h_split - a_split) * 0.8)

    # ── 4. Offensive efficiency — points scored per game ─────────────────────
    h_ppg = _safe_float(home.get('ppg'), LEAGUE_PPG)
    a_ppg = _safe_float(away.get('ppg'), LEAGUE_PPG)
    _add('Points per game', (h_ppg - a_ppg) * 0.05)

    # ── 5. Defensive efficiency — points allowed per game ────────────────────
    h_ppga = _safe_float(home.get('ppg_allowed'), LEAGUE_PPG)
    a_ppga = _safe_float(away.get('ppg_allowed'), LEAGUE_PPG)
    _add('Points allowed/G', (a_ppga - h_ppga) * 0.05)

    # ── 6. Recent form — last 3 games ─────────────────────────────────────────
    h_form = home.get('form', [])
    a_form = away.get('form', [])
    if h_form and a_form:
        h_f = h_form.count('W') / len(h_form)
        a_f = a_form.count('W') / len(a_form)
        _add('Recent form (L3)', (h_f - a_f) * 0.45)

    # ── 7. Rest advantage ─────────────────────────────────────────────────────
    # rest_days = days between last game and this game.
    # Normal week = 7d. Bye = 14d. Thursday short week = 4-5d.
    h_rest = home.get('rest_days')
    a_rest = away.get('rest_days')
    if h_rest is not None and a_rest is not None:
        if h_rest >= 13 and a_rest < 13:
            _add('Bye week (home)', 0.18)
        elif a_rest >= 13 and h_rest < 13:
            _add('Bye week (away)', -0.18)
        if h_rest <= 5 and a_rest >= 7:
            _add('Short week (home)', -0.14)
        elif a_rest <= 5 and h_rest >= 7:
            _add('Short week (away)', 0.14)

    home_prob = _sigmoid(logit)
    return {
        'home_prob': round(home_prob, 4),
        'away_prob': round(1.0 - home_prob, 4),
        'logit':     round(logit, 3),
        'factors':   factors,
    }
