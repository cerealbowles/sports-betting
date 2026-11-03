"""
CBB (men's college basketball) win-probability model.

    margin    = cbb_rating.predict(...)['margin']      (home points, + = home favored)
    home_prob = sigmoid(margin / MARGIN_SCALE)

The margin comes from cbb_rating.py's opponent-adjusted efficiency ratings
and decomposes exactly into three logit factors — the opponent-adjusted
versions of the NBA/WNBA model's 'Points per game' / 'Points allowed/G' /
'Home court' trio:

    Adj. offense   (AdjO_home - AdjO_away) * poss / 100
    Adj. defense   (AdjD_away - AdjD_home) * poss / 100     (AdjD = pts allowed / 100)
    Home court     HFA * poss / 100, zero at a neutral site

Calibration (cbb_bootstrap.py, 2026-11-02): MARGIN_SCALE fit on the
2024-25 season walk-forward and graded on 2025-26 (5,751 D1-vs-D1 games):
72.1% straight-up, log loss 0.529, Brier 0.180. The rating's margin missed
by 8.89 pts on the 2,971 games with an ESPN/DraftKings close vs 8.79 for
the market itself, 50.4% ATS — accurate, no standalone spread edge.

The NBA/WNBA factor set — win %, home/road split record, raw PPG, raw PPG
allowed, last-5 form, back-to-back — was rebuilt point-in-time and tested
as an add-on to the rating. None moved held-out log loss by even 0.001
(back-to-back made it worse; all six together made it worse), so they are
NOT in the logit. Win % and L5 form are attached as '(tracking)' rows
(display only, excluded from the sum — same convention as cfb_model's
tracking rows) so they keep accumulating graded history.

Early season: cbb_rating carries the two prior seasons at a reduced weight,
so opening-night games get a real (if roster-blind) rating. Out-of-sample
November miss was ~10.0 pts vs ~8.9 from January on.
"""
import math

import market_edge_calibration as _mkt_calib

MARGIN_SCALE = 6.10       # points per unit of logit (cbb_bootstrap.py)
FALLBACK_HFA_PTS = 2.9    # used only when no rating fit is available
LEAGUE_PPG = 72.5         # D1 per-team scoring average, 2023-26

# Model-vs-market shrink rate — data-driven, see market_edge_calibration.py.
# app.py's _recompute_market_edge_shrink('CBB') refits it as resolved games
# with market prices accumulate; starts at the shared prior.
_MKT_EDGE_RATE = _mkt_calib.load_rate('CBB')


def _sigmoid(x):
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _pct(wins, losses):
    total = wins + losses
    return wins / total if total else 0.5


def _shrink_toward_market(home_prob, market_home_prob):
    """Same guardrail as every other sport model: pull back model-vs-market
    disagreements beyond 10 percentage points by _MKT_EDGE_RATE."""
    if market_home_prob is None:
        return home_prob
    edge = home_prob - market_home_prob
    excess = abs(edge) - 0.10
    if excess > 0:
        pull = excess * _MKT_EDGE_RATE
        home_prob -= pull if edge > 0 else -pull
    return home_prob


def predict(home, away, rating=None, neutral=False, game_time_utc=None, market_home_prob=None):
    """
    Args:
        home/away – team dicts from cbb_api._build_game (wins/losses/form...).
        rating    – cbb_rating.predict(...) output for this matchup, or None
                    when no fit is available (falls back to home court only).
        neutral   – neutral-site game (no home court).
        market_home_prob – vig-free market home prob, for the shrink only.
    Returns:
        home_prob, away_prob, logit, factors [(label, contrib)], margin,
        plus 'proj_home_pts'/'proj_away_pts'/'proj_poss' when rated.
    """
    factors = []
    if rating:
        poss = rating['poss']
        h, a = rating['home'], rating['away']
        off = (h['adj_o'] - a['adj_o']) * poss / 100
        dfn = (a['adj_d'] - h['adj_d']) * poss / 100
        # Home court is whatever's left — exact by construction, and absorbs
        # the 0.1-pt rounding in the displayed ratings.
        hca = rating['margin'] - off - dfn
        margin = rating['margin']
        if not neutral:
            factors.append(('Home court', round(hca / MARGIN_SCALE, 3)))
        else:
            dfn += hca
        factors.append(('Adj. offense', round(off / MARGIN_SCALE, 3)))
        factors.append(('Adj. defense', round(dfn / MARGIN_SCALE, 3)))
    else:
        margin = 0.0 if neutral else FALLBACK_HFA_PTS
        factors.append(('Home court', round(margin / MARGIN_SCALE, 3)))

    # ── Tracking only — NOT in the logit (see module docstring) ──────────
    h_wp = home.get('blend_win_pct')
    if h_wp is None:
        h_wp = _pct(home.get('wins', 0), home.get('losses', 0))
    a_wp = away.get('blend_win_pct')
    if a_wp is None:
        a_wp = _pct(away.get('wins', 0), away.get('losses', 0))
    factors.append(('Win percentage (tracking)', round((h_wp - a_wp) * 0.38, 3)))
    h_form, a_form = home.get('form') or [], away.get('form') or []
    if h_form and a_form:
        diff = h_form.count('W') / len(h_form) - a_form.count('W') / len(a_form)
        factors.append(('Recent form L5 (tracking)', round(diff * 0.32, 3)))

    logit = margin / MARGIN_SCALE
    home_prob = _shrink_toward_market(_sigmoid(logit), market_home_prob)
    out = {
        'home_prob': round(home_prob, 4),
        'away_prob': round(1.0 - home_prob, 4),
        'logit':     round(logit, 3),
        'factors':   factors,
        'margin':    round(margin, 2),
    }
    if rating:
        out.update({'proj_home_pts': rating['home_pts'], 'proj_away_pts': rating['away_pts'],
                    'proj_poss': rating['poss']})
    return out
