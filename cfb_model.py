"""
College Football (FBS) win-probability model.

Logit-additive: each factor contributes a signed value in log-odds space.
sigmoid(sum) → home win probability.

Initial weights hand-tuned by analogy to nfl_model.py, adjusted for CFB's
higher home-win rate (~59-60% vs NFL's ~57%), bigger scoring/blowout
variance (talent gap between a top-10 team and a cupcake opponent dwarfs
anything in the NFL), and the addition of an AP/CFP ranking factor that
captures season-long team strength which raw win/loss can't (a 3-loss
one-loss team playing a cupcake-heavy schedule looks the same as a 9-1 team
in a brutal conference until you look at the polls).

Run cfb_bootstrap.py after enough of a season is collected to calibrate
coefficients via logistic regression, same as nfl_model.py. Bye-week and
short-week factors cannot be fully validated in bootstrap (rare events);
keep hand-tuned, validate on the Model Performance page.
"""
import math

import market_edge_calibration as _mkt_calib

LEAGUE_PPG = 28.5   # FBS scoring average runs meaningfully higher than the NFL's ~23

# Model-vs-market shrink rate — data-driven, see market_edge_calibration.py.
# app.py's _recompute_market_edge_shrink('CFB') overwrites this in place as
# more resolved games accumulate; this is just the value at import time
# (last snapshot, or the hardcoded 25% prior on first run).
_MKT_EDGE_RATE = _mkt_calib.load_rate('CFB')

_UNRANKED = 99   # ESPN's curatedRank.current sentinel for an unranked team


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


def _rank_score(rank):
    """Rank 1 → 1.0, rank 25 → 0.04, unranked (>=99 or missing) → 0.0."""
    try:
        r = int(rank)
    except (TypeError, ValueError):
        return 0.0
    if r < 1 or r > 25:
        return 0.0
    return (26 - r) / 25.0


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

    # ── 1. Home field advantage (~59-60% historically — bigger than the NFL's) ─
    _add('Home field', 0.32)

    # ── 2. AP/CFP ranking differential ─────────────────────────────────────────
    h_rank = _rank_score(home.get('rank', _UNRANKED))
    a_rank = _rank_score(away.get('rank', _UNRANKED))
    _add('Ranking', (h_rank - a_rank) * 1.4)

    # ── 3. Overall win-percentage differential ────────────────────────────────
    # 'blend_win_pct' (set by cfb_api._apply_prior_season_blend) mixes in last
    # season's final record for teams with <EARLY_SEASON_GAMES games played
    # this season — otherwise every team defaults to an identical 0-0 (50%)
    # record in week 1, and this factor (and #4-6 below) contribute exactly
    # 0 for every early-season game. Same reasoning as nfl_model.py.
    h_wp = home.get('blend_win_pct')
    if h_wp is None:
        h_wp = _pct(home.get('wins', 0), home.get('losses', 0))
    a_wp = away.get('blend_win_pct')
    if a_wp is None:
        a_wp = _pct(away.get('wins', 0), away.get('losses', 0))
    _add('Win percentage', (h_wp - a_wp) * 1.6)

    # ── 4. Home / road split record ───────────────────────────────────────────
    h_split = home.get('blend_split_pct')
    if h_split is None:
        h_split = _pct(home.get('split_w', 0), home.get('split_l', 0))
    a_split = away.get('blend_split_pct')
    if a_split is None:
        a_split = _pct(away.get('split_w', 0), away.get('split_l', 0))
    _add('Home/road record', (h_split - a_split) * 0.8)

    # ── 5. Offensive efficiency — points scored per game ─────────────────────
    h_ppg = _safe_float(home.get('blend_ppg', home.get('ppg')), LEAGUE_PPG)
    a_ppg = _safe_float(away.get('blend_ppg', away.get('ppg')), LEAGUE_PPG)
    _add('Points per game', (h_ppg - a_ppg) * 0.045)

    # ── 6. Defensive efficiency — points allowed per game ────────────────────
    h_ppga = _safe_float(home.get('blend_ppg_allowed', home.get('ppg_allowed')), LEAGUE_PPG)
    a_ppga = _safe_float(away.get('blend_ppg_allowed', away.get('ppg_allowed')), LEAGUE_PPG)
    _add('Points allowed/G', (a_ppga - h_ppga) * 0.045)

    # ── 7. Recent form — last 3 games ─────────────────────────────────────────
    h_form = home.get('form', [])
    a_form = away.get('form', [])
    # Needs a full 3-game sample for BOTH teams — a 1-0 or 2-0 start is 100%
    # form off a coin-flip-sized sample and swings this factor to its max
    # contribution off noise. Skip rather than let single early games spike it.
    if len(h_form) >= 3 and len(a_form) >= 3:
        h_f = h_form.count('W') / len(h_form)
        a_f = a_form.count('W') / len(a_form)
        _add('Recent form (L3)', (h_f - a_f) * 0.45)

    # ── 8. Rest advantage ─────────────────────────────────────────────────────
    # rest_days = days between last game and this game. Normal week = 7d.
    # Bye = 14d (most FBS teams get exactly one per season). Short week
    # (Tue/Wed/Thu "MACtion"-style games) = 4-5d.
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

    # Shrink large model-vs-market disagreements — same guardrail added to
    # mlb_model.py, where 50 resolved games found 10+ point edges over the
    # vig-free market won only 25% of the time vs. ~58% under that threshold.
    # The 10pt threshold is fixed; the pull-back rate (_MKT_EDGE_RATE) is
    # data-driven — app.py's _recompute_market_edge_shrink('CFB') refits it
    # from resolved game_predictions as CFB accumulates its own large-edge
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
