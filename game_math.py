"""
Pure betting math for the game-detail page: implied probability, no-vig,
EV, Kelly units, cover-probability normal approximation, the outlier guard,
model-agreement confidence, and the PLAY/LEAN/CHECK/PASS tag rule.

No Flask/DB imports here on purpose — keep this a pure, unit-testable module
(see test_game_math.py) that both the game-detail data assembly and any
future caller can share instead of re-deriving the same formulas.
"""
import math


def implied_prob(american_odds):
    """American odds -> implied probability (with vig)."""
    o = float(american_odds)
    return 100.0 / (o + 100.0) if o > 0 else (-o) / (-o + 100.0)


def no_vig_pair(imp_a, imp_b):
    """Two-way no-vig normalization. Returns (novig_a, novig_b, hold)."""
    total = imp_a + imp_b
    hold = total - 1.0
    return imp_a / total, imp_b / total, hold


def net_payout_b(american_odds):
    """Net payout per $1 staked ('b' in Kelly notation)."""
    o = float(american_odds)
    return o / 100.0 if o > 0 else 100.0 / (-o)


def ev(p, b):
    """Expected value per $1 staked, p = model probability for that side."""
    return p * b - (1.0 - p)


def kelly_fraction(p, b):
    return (b * p - (1.0 - p)) / b


def kelly_units(p, b, cap=0.25):
    """Quarter-Kelly (default cap) expressed in units, 1u = 1% bankroll."""
    return max(0.0, kelly_fraction(p, b) * cap) * 100.0


def cover_prob_normal(model_margin, market_line, sigma):
    """P(side covers) via normal approximation.

    `model_margin` is the model's expected scoring margin for the side in
    question (own score minus opponent's, so negative for a favored
    opponent / underdog side). `market_line` is the points that side is
    getting (positive) or giving (negative), e.g. NE +3 -> market_line=3.0.
    Side covers if actual_margin + market_line > 0, so with
    actual_margin ~ N(model_margin, sigma):
        P(cover) = Phi((model_margin + market_line) / sigma)

    No key-number adjustment (3/7) here — the spec calls that optional;
    this plain normal approximation is the baseline.
    """
    z = (model_margin + market_line) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def outlier_guard(model_line, market_line, market_gap_sd, threshold=2.0):
    """True if the model/market gap is an outlier vs. its historical spread."""
    if not market_gap_sd:
        return False
    return abs(model_line - market_line) / market_gap_sd > threshold


def confidence(p_team, p_player, start=5, step_pts=5.0, floor=1):
    """Placeholder confidence rule from the spec: start at 5, subtract 1 per
    step_pts of team/player probability disagreement (in percentage points),
    floor at `floor`."""
    disagreement_pts = round(abs(p_team - p_player) * 100.0, 6)
    c = start - math.floor(disagreement_pts / step_pts)
    return max(floor, min(start, c))


def tag_for(ev_value, confidence_value, is_outlier, play_ev=0.04, lean_ev=0.02, play_conf=4):
    """PLAY / LEAN / CHECK / PASS per the spec's tag rules. CHECK (outlier
    guard) overrides everything else."""
    if is_outlier:
        return 'CHECK'
    if ev_value >= play_ev and confidence_value >= play_conf:
        return 'PLAY'
    if ev_value >= lean_ev:
        return 'LEAN'
    return 'PASS'
