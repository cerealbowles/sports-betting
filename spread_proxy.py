"""
spread_proxy.py — Derives a rough model-implied point margin (and spread
cover probability) from the existing win-probability model, per sport. This
is NOT a real margin/spread model — we don't have one. It's a linear fit of
actual final margin against logit(model_home_prob), refit from the full
game_predictions history (instance/bets.db, 2022-2026 depending on sport).

Correlation with actual margin is real but modest and, in every sport,
weaker than the market's own closing spread (see the analysis run during
the ESPN odds backfill: model logit correlation 0.13-0.47 vs market
0.13-0.82 depending on sport). Treat this as "differs from market" framing,
not a sharper number — that's also why the card UI built on top of this
stays deliberately quiet (no glow, no "BEST EDGE" tag) for spread edges.

No equivalent exists for totals (O/U) — no sport model outputs any signal
about total runs/points/goals today, so there's nothing to fit here for
that market. That's a genuinely new model, not a derivation of this one.

Refit periodically as more backfilled history accumulates — these
coefficients are a one-time fit from the September 2026 backfill.
"""
import math

# {sport: (intercept, slope, sigma)} — actual_margin ≈ intercept + slope * logit(home_prob),
# sigma = residual std dev, used for the cover-probability normal approximation.
COEFFS = {
    'MLB':  (-0.2236, 2.5950, 4.543),
    'NFL':  ( 0.9042, 3.2622, 13.424),
    'CFB':  ( 5.8699, 5.0698, 20.046),
    'NHL':  ( 0.0702, 1.0996, 2.581),
    'NBA':  (-0.5258, 5.3636, 14.734),
    'WNBA': ( 0.2765, 6.1636, 12.967),
}


def _phi(x):
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def implied_margin(sport, home_prob):
    """Model-implied home-team margin (positive = home wins by that much), or None
    if this sport has no fitted coefficients or home_prob is degenerate."""
    coeffs = COEFFS.get(sport)
    if not coeffs or home_prob is None or not (0 < home_prob < 1):
        return None
    a, b, _ = coeffs
    logit = math.log(home_prob / (1 - home_prob))
    return a + b * logit


def cover_prob(sport, home_prob, home_spread_line):
    """P(home team covers home_spread_line) per the proxy model, or None if
    unavailable. home_spread_line follows market convention (negative = home
    favored) — home covers when actual_margin > -home_spread_line."""
    coeffs = COEFFS.get(sport)
    margin = implied_margin(sport, home_prob)
    if margin is None or home_spread_line is None:
        return None
    _, _, sigma = coeffs
    if sigma <= 0:
        return None
    return _phi((margin - (-home_spread_line)) / sigma)
