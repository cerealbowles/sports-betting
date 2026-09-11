"""
NHL win-probability model.

Logit-additive: each factor contributes a signed value in log-odds space.
sigmoid(sum) → home win probability. Same structure as nfl_model.py /
cfb_model.py so the shared calibration/backfill/bootstrap pattern applies
unchanged.

Initial weights hand-tuned from general NHL priors (~54-55% home win rate,
smaller home-ice edge than NFL/CFB since travel/rest dominate more).
Goalie save % / GAA are deliberately NOT used as predict() factors — they
aren't derivable point-in-time from final scores alone, so nhl_backfill.py
can't validate them the way it validates every other factor here. They stay
as display-only enrichment in nhl_api.py, same way nfl_model.py excludes
weather.

Run nhl_bootstrap.py after a season+ of results accumulate to calibrate
coefficients via logistic regression.
"""
import math

LEAGUE_GPG = 3.05   # NHL recent-era average goals per team per game


def _sigmoid(x):
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _pct(wins, losses):
    total = wins + losses
    return wins / total if total else 0.5


def _points_pct(wins, losses, ot_losses):
    gp = wins + losses + (ot_losses or 0)
    return (wins * 2 + (ot_losses or 0)) / (gp * 2) if gp else 0.5


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

    # ── 1. Home ice advantage (~54-55% historically — smaller than NFL/CFB) ──
    _add('Home ice', 0.12)

    # ── 2. Points-percentage differential (standings-correct: OTL is a half-win) ─
    # Weight shrunk after nhl_bootstrap.py's 2023-25 refit showed this heavily
    # overweighted/collinear with split record & form (all three measure
    # "team is good" from the same underlying results) — keep it, but small.
    h_pp = _points_pct(home.get('wins', 0), home.get('losses', 0), home.get('ot_losses', 0))
    a_pp = _points_pct(away.get('wins', 0), away.get('losses', 0), away.get('ot_losses', 0))
    _add('Points percentage', (h_pp - a_pp) * 0.22)

    # ── 3. Home / road split record ───────────────────────────────────────────
    h_split = _pct(home.get('split_w', 0), home.get('split_l', 0))
    a_split = _pct(away.get('split_w', 0), away.get('split_l', 0))
    _add('Home/road record', (h_split - a_split) * 0.08)

    # ── 4. Offensive efficiency — goals scored per game ───────────────────────
    h_gf = _safe_float(home.get('gf_pg'), LEAGUE_GPG)
    a_gf = _safe_float(away.get('gf_pg'), LEAGUE_GPG)
    _add('Goals for/G', (h_gf - a_gf) * 0.3)

    # ── 5. Defensive efficiency — goals allowed per game ──────────────────────
    h_ga = _safe_float(home.get('ga_pg'), LEAGUE_GPG)
    a_ga = _safe_float(away.get('ga_pg'), LEAGUE_GPG)
    _add('Goals against/G', (a_ga - h_ga) * 0.24)

    # ── 6. Recent form — last 5 games ─────────────────────────────────────────
    h_form = (home.get('form') or [])[-5:]
    a_form = (away.get('form') or [])[-5:]
    if h_form and a_form:
        h_f = h_form.count('W') / len(h_form)
        a_f = a_form.count('W') / len(a_form)
        _add('Recent form (L5)', (h_f - a_f) * 0.06)

    # ── 7. Back-to-back fatigue ────────────────────────────────────────────────
    # rest_days = days since each team's previous game. NHL schedules are
    # dense — 1 day (back-to-back) is common and meaningfully hurts the team
    # playing it, road back-to-backs especially (goalie/legs fatigue).
    h_rest = home.get('rest_days')
    a_rest = away.get('rest_days')
    if h_rest is not None and a_rest is not None:
        if h_rest <= 1 and a_rest > 1:
            _add('Back-to-back (home)', -0.22)
        elif a_rest <= 1 and h_rest > 1:
            _add('Back-to-back (away)', 0.22)

    home_prob = _sigmoid(logit)
    return {
        'home_prob': round(home_prob, 4),
        'away_prob': round(1.0 - home_prob, 4),
        'logit':     round(logit, 3),
        'factors':   factors,
    }
