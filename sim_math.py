"""
Pure math for the simulation/player-projections page: normal/Poisson prop
probabilities, EV on both sides of a prop, win probability excluding ties,
cover/over probabilities from an empirical array of per-run outcomes, and
Pearson correlation. Same "no Flask/DB imports, pure + unit-tested" rule as
game_math.py (see test_sim_math.py), so this can be shared by the sim-page
data assembly and any future caller.
"""
import math
from collections import Counter

import game_math as gm  # reuse ev()/net_payout_b() rather than redefine them


def phi(z):
    """Standard normal CDF via the math.erf identity."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def poisson_cdf(k, lam):
    """P(X <= k) for X ~ Poisson(lam), k a non-negative integer."""
    k = int(k)
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return total


def p_over_normal(line, mean, sd):
    """Yardage-style prop ('n' kind): P(actual > line) via a normal approx."""
    if not sd:
        return 1.0 if mean > line else 0.0
    return 1.0 - phi((line - mean) / sd)


def p_over_poisson(line, lam):
    """Count-style prop ('pois' kind, e.g. pass TDs/receptions): the line is
    a half-integer (X.5), so floor(line) is the largest integer not over."""
    return 1.0 - poisson_cdf(math.floor(line), lam)


def p_anytime_td(lam):
    """Anytime-TD prop ('atd' kind): lam = expected rush + rec TDs."""
    return 1.0 - math.exp(-lam)


def p_over_empirical(line, draws):
    """Empirical P(actual > line) from the sim's own per-run draws —
    preferred over the normal/Poisson approximations above whenever the
    raw per-run values are available (per the spec)."""
    if not draws:
        return None
    return sum(1 for v in draws if v > line) / len(draws)


def ev_over_under(p_over, over_price, under_price):
    """EV per $1 on each side of a prop, using FanDuel prices for both.
    Returns (ev_over, ev_under)."""
    ev_over = gm.ev(p_over, gm.net_payout_b(over_price))
    ev_under = gm.ev(1.0 - p_over, gm.net_payout_b(under_price))
    return ev_over, ev_under


def win_prob_excl_ties(wins_a, wins_b):
    """Win probability for side A, excluding ties from the denominator
    (the spec's agreement-table convention) — None if there are no
    decisive runs at all."""
    decisive = wins_a + wins_b
    if not decisive:
        return None
    return wins_a / decisive


def cover_prob_empirical(margins, line):
    """From a list of per-run (home - away) margins and a home spread line
    (e.g. -3.0 for a 3-point home favorite), returns
    {'home_covers', 'push', 'away_covers'} as fractions of len(margins)."""
    n = len(margins)
    if not n:
        return None
    home_covers = sum(1 for m in margins if m + line > 0)
    push = sum(1 for m in margins if m + line == 0)
    away_covers = n - home_covers - push
    return {'home_covers': home_covers / n, 'push': push / n, 'away_covers': away_covers / n}


def histogram(values, bin_width, lo=None, hi=None):
    """Buckets `values` into bin_width-wide bins, returning
    [{'lo', 'hi', 'count'}, ...] sorted by bin, covering [lo, hi] (inferred
    from the data when not given) even for empty bins in between."""
    if not values:
        return []
    lo = math.floor((lo if lo is not None else min(values)) / bin_width) * bin_width
    hi = math.ceil((hi if hi is not None else max(values)) / bin_width) * bin_width
    n_bins = max(1, int(round((hi - lo) / bin_width)))
    counts = Counter()
    for v in values:
        idx = min(n_bins - 1, max(0, int((v - lo) // bin_width)))
        counts[idx] += 1
    return [{'lo': lo + i * bin_width, 'hi': lo + (i + 1) * bin_width, 'count': counts.get(i, 0)}
            for i in range(n_bins)]


def pearson_r(xs, ys):
    """Pearson correlation coefficient across paired per-run values. None
    if there are fewer than 2 pairs or either series has zero variance."""
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs, ys = xs[:n], ys[:n]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def percentile(values, pct):
    """Nearest-rank percentile (0-100) of a list of values."""
    if not values:
        return None
    ranked = sorted(values)
    idx = min(len(ranked) - 1, max(0, int(round(pct / 100.0 * (len(ranked) - 1)))))
    return ranked[idx]
