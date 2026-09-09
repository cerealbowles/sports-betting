"""
MLB win-probability model.

Logit-additive: each factor contributes a signed value in log-odds space.
sigmoid(sum) → home win probability.

Static factors (ERA, WHIP, OPS, R/G, K-BB%, K%, xwOBA) were calibrated via
logistic regression on 5,682 games (2024–2026 regular seasons). 80% of the
full regression coefficient is applied to limit in-sample overfit.

Rolling factors (Recent form L10, Home/road split record, Starter recent ERA,
Bullpen ERA/14d, closer availability) cannot be calibrated from the bootstrap
because the bootstrap scores historical games using season-aggregate stats and
passes empty arrays for all rolling data. Their coefficients show 0.000 in the
refit output — meaning the bootstrap had no variance to measure, NOT that the
factors are invalid. These weights are hand-tuned and validated against live
2026 predictions on the Model Performance page.

Bullpen fatigue (IP last 3 days, differential) is the one rolling factor with
a real historical backtest behind it (via bullpen_fatigue.py's box-score
reconstruction, not the bootstrap's empty-array limitation above) — see its
coefficient comment near the H2H/bias section for the validation numbers.

Division game, day/night, and weekend bonuses removed — no statistically
meaningful signal across either season.
"""
import json
import math
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

_ET = ZoneInfo('America/New_York')

# MLB seasonal baselines
LEAGUE_ERA    = 4.20
LEAGUE_XERA   = 4.10   # xERA runs slightly lower than ERA on average
LEAGUE_WHIP   = 1.30
LEAGUE_RPG    = 4.50
LEAGUE_OPS    = 0.720
LEAGUE_XWOBA  = 0.320
LEAGUE_K_BB   = 0.150  # (K - BB) / BF, roughly 15% for average starter
LEAGUE_K_PCT  = 0.225  # team strikeout rate (~22.5% of PA)
LEAGUE_BP_ERA = 4.10   # league average bullpen ERA
LEAGUE_AVG_IP = 5.5    # league average starter innings per start

# MLB Stats API team ID → division
_DIVISION = {
    # AL East
    110: 'ALE', 111: 'ALE', 139: 'ALE', 141: 'ALE', 147: 'ALE',
    # AL Central
    114: 'ALC', 116: 'ALC', 118: 'ALC', 142: 'ALC', 145: 'ALC',
    # AL West
    108: 'ALW', 117: 'ALW', 133: 'ALW', 136: 'ALW', 140: 'ALW',
    # NL East
    120: 'NLE', 121: 'NLE', 143: 'NLE', 144: 'NLE', 146: 'NLE',
    # NL Central
    112: 'NLC', 113: 'NLC', 134: 'NLC', 138: 'NLC', 158: 'NLC',
    # NL West
    109: 'NLW', 115: 'NLW', 119: 'NLW', 135: 'NLW', 137: 'NLW',
}


def _sigmoid(x):
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _pct(wins, losses):
    total = wins + losses
    return wins / total if total else 0.5


def _safe_float(val, default):
    try:
        v = float(val)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _safe_float_pct(val):
    """Return float for percentage/rate values (can be 0.0), or None on failure.
    Unlike _safe_float, does NOT reject 0.0 — valid for K%, BB%, barrel%, whiff%."""
    if val is None:
        return None
    try:
        v = float(val)
        # Reject clearly invalid sentinels but allow 0.0
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None



def _game_time_et(game_time_utc):
    """Parse UTC game time string → ET datetime. Returns None on failure."""
    if not game_time_utc:
        return None
    try:
        dt = datetime.fromisoformat(game_time_utc.replace('Z', '+00:00'))
        return dt.astimezone(_ET)
    except Exception:
        return None


# Platt scaling parameters — fit at runtime from resolved game_predictions.
# None until fit_platt() is called; predict() falls back to hard clamp until then.
_platt = None  # (a, b) where calibrated_prob = sigmoid(a * raw_logit + b)

# Per-team historical bias offsets (probability scale, applied in logit space).
# Derived from 2026 per-team calibration analysis; 30% shrinkage applied to
# raw over/underestimate to avoid overfit on a small sample.
# Positive = model underestimates this team (nudge their win prob up).
# Negative = model overestimates this team (nudge their win prob down).
_TEAM_BIAS = {
    # 3-season calibration (2024-2026), 30% shrinkage, n≥160/team per season.
    # Recalibrated 2026-06-08 — previous values were 2026-only (n~150 each).
    # H2H factor: leave unchanged, revisit at n≥30 (only n=6 in production).

    # Consistent underestimates (model too pessimistic about these teams)
    'Cleveland Guardians':   +0.04,   # +6.1% avg  (2024 +5.7 · 2025 +8.0 · 2026 +2.2)
    'Houston Astros':        +0.03,   # +3.5% avg  (2024 +4.5 · 2025 +3.6 · 2026 +0.6)
    'St. Louis Cardinals':   +0.02,   # +3.2% avg  (2024 +4.0 · 2025 +0.8 · 2026 +7.3)
    'Philadelphia Phillies': +0.02,   # +3.4% avg  (2024 +4.8 · 2025 +4.3 · 2026 −2.7)
    'Milwaukee Brewers':     +0.02,   # +2.8% avg  (2024 +2.1 · 2025 +3.3 · 2026 +3.6)
    'San Diego Padres':      +0.02,   # +2.8% avg  (2024 +2.1 · 2025 +3.6 · 2026 +2.7)
    'Los Angeles Dodgers':   +0.02,   # +2.7% avg  (2024 +3.9 · 2025 +0.8 · 2026 +4.2)
    'Atlanta Braves':        +0.02,   # +2.3% avg  (2024 +3.5 · 2025 −3.1 · 2026 +12.5)

    # Consistent overestimates (model too optimistic about these teams)
    'Colorado Rockies':      -0.07,   # −9.3% avg  (2024 −5.4 · 2025 −13.9 · 2026 −7.3)
    'Chicago White Sox':     -0.04,   # −7.2% avg  (2024 −11.5 · 2025 −7.1 · 2026 +3.2; conservative)
    'Minnesota Twins':       -0.03,   # −4.9% avg  (2024 −3.9 · 2025 −6.0 · 2026 −4.4)
    'Los Angeles Angels':    -0.03,   # −4.1% avg  (2024 −4.5 · 2025 −1.9 · 2026 −8.5)
}

# app.py's _recompute_team_bias() recalibrates _TEAM_BIAS live (on startup,
# after nightly outcome resolution, and after daily pick generation) and
# persists the result here. On import, prefer that snapshot over the
# hardcoded dict above — it's a real (if slightly lagged) point-in-time
# value rather than one manually locked on 2026-06-08. Offline scripts like
# backfill_2026.py deliberately don't trigger a live recompute themselves
# (see app.py's DISABLE_STARTUP_TASKS guard); they get whatever the live app
# last computed and wrote here, which is the "snapshot once per day/week"
# behavior instead of "recompute from current DB state on every run".
_TEAM_BIAS_SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'team_bias_snapshot.json')


def _load_team_bias_snapshot(sport='MLB'):
    try:
        if not os.path.exists(_TEAM_BIAS_SNAPSHOT_PATH):
            return
        with open(_TEAM_BIAS_SNAPSHOT_PATH) as f:
            payload = json.load(f)
        entry = payload.get(sport)
        if not entry or not isinstance(entry.get('bias'), dict):
            return
        global _TEAM_BIAS
        _TEAM_BIAS = entry['bias']
        print(f'[mlb_model] loaded _TEAM_BIAS snapshot from '
              f'{entry.get("computed_at", "unknown time")} '
              f'({len(_TEAM_BIAS)} teams)', flush=True)
    except Exception as e:
        print(f'[mlb_model] failed to load team-bias snapshot, '
              f'keeping hardcoded fallback: {e}', flush=True)


_load_team_bias_snapshot('MLB')


# ── Trade-deadline seller discount (temporary, date-windowed) ───────────────
# Around the MLB trade deadline (typically July 31), teams that trade away
# good players get weaker overnight, but this model's inputs (season
# aggregates, rolling stats, _TEAM_BIAS) lag real transactions by days to
# weeks. Confirmed 2026-07-30 on live data: picks on that year's deadline
# sellers went 25.5% WR / -56.4% ROI (n=51) vs 56.4% WR / -0.2% ROI on
# everything else in the same 3-week window. Also confirmed the market
# itself hadn't priced this in either (line-movement signal on seller picks
# was statistically identical to elsewhere) — so this can't be fixed by
# reweighting the Unified Score toward market/line-move signals, only by
# discounting the affected teams' win probability directly. Full writeup in
# auto-memory 'seasonal_trade_deadline_effect'.
#
# This is a MANUAL, per-year list — there's no live transactions feed wired
# into this codebase. Update TEAMS and WINDOW each year around the deadline:
#   1. Pull that year's confirmed/rumored sellers from trade-deadline news
#      coverage (a single web search covering "<year> MLB trade deadline
#      sellers" was sufficient in 2026).
#   2. Set WINDOW to roughly 10 days before the deadline through ~3 weeks
#      after — long enough for the rolling-15-game window to absorb each
#      team's new roster, short enough that this doesn't linger once real
#      data has caught up.
#   3. If no games fall in the window (checked live each year), this is a
#      no-op — safe to leave configured for the next year's dates ahead of
#      time, or to leave stale after the window passes.
_TRADE_DEADLINE_DISCOUNT = {
    'window': (date(2026, 7, 20), date(2026, 8, 21)),  # 2026 deadline: 7/31
    'amount': -0.04,  # probability-scale nudge, same units/scale as _TEAM_BIAS
    'teams': {
        'Athletics', 'Toronto Blue Jays', 'St. Louis Cardinals', 'Texas Rangers',
        'Detroit Tigers', 'Miami Marlins', 'San Francisco Giants',
        'Washington Nationals',
    },
}


def _trade_deadline_bias(team_name, game_date):
    """Temporary per-team logit-space discount for confirmed trade-deadline
    sellers (see _TRADE_DEADLINE_DISCOUNT above). Returns 0.0 outside the
    active window or for teams not on that year's seller list. Keyed off
    the game's own date (not "today"), so backfill/bootstrap reruns over
    historical dates apply it consistently rather than drifting with when
    the script happens to run."""
    if game_date is None or not team_name:
        return 0.0
    lo, hi = _TRADE_DEADLINE_DISCOUNT['window']
    if not (lo <= game_date <= hi):
        return 0.0
    if team_name not in _TRADE_DEADLINE_DISCOUNT['teams']:
        return 0.0
    return _TRADE_DEADLINE_DISCOUNT['amount']


def fit_platt(records):
    """Fit Platt scaling from resolved predictions using Newton's method.

    records: list of (home_prob: float, home_won: bool)
    Requires ≥50 samples. Pure Python — no external dependencies.
    Fits: calibrated_prob = sigmoid(a * logit(home_prob) + b)
    """
    global _platt
    if len(records) < 50:
        return
    try:
        xs = [math.log(max(1e-6, min(1 - 1e-6, p)) /
                       (1 - max(1e-6, min(1 - 1e-6, p))))
              for p, _ in records]
        ys = [1.0 if w else 0.0 for _, w in records]
        n  = len(xs)

        a, b = 1.0, 0.0
        for _ in range(100):
            ga = gb = haa = hab = hbb = 0.0
            for x, y in zip(xs, ys):
                p   = _sigmoid(a * x + b)
                err = p - y
                w   = p * (1.0 - p)
                ga  += err * x
                gb  += err
                haa += w * x * x
                hab += w * x
                hbb += w
            det = haa * hbb - hab * hab
            if abs(det) < 1e-12:
                break
            da = (hbb * ga - hab * gb) / det
            db = (haa * gb - hab * ga) / det
            a -= da
            b -= db
            if abs(da) < 1e-7 and abs(db) < 1e-7:
                break

        _platt = (a, b)
        print(f'[mlb_model] Platt scaler fit on {n} games: '
              f'a={a:.4f} b={b:.4f}', flush=True)
    except Exception as exc:
        print(f'[mlb_model] Platt fit failed: {exc}', flush=True)


_GROUP_ORDER = [
    ('pitching',    'Pitching'),
    ('bullpen',     'Bullpen'),
    ('offense',     'Offense'),
    ('situational', 'Situational'),
]


def predict(home, away, matchup_history=None, game_time_utc=None):
    """
    Returns:
        home_prob     – estimated home win probability (clamped, float 0-1)
        away_prob     – 1 - home_prob
        factors       – [(label, logit_contrib)] for backward-compat storage
        factor_groups – {group: [(label, prob_pct)]} grouped probability contributions
        group_order   – ordered list of (key, display_name) for template iteration
    """
    ha = home.get('abbr', 'Hm')   # home abbreviation
    aa = away.get('abbr', 'Aw')   # away abbreviation

    logit = 0.0
    _raw  = []   # (label, logit_contrib, group)

    def _add(label, contrib, group='situational'):
        nonlocal logit
        logit += contrib
        _raw.append((label, round(contrib, 3), group))

    # ── 1. Home field advantage ───────────────────────────────────────────────
    _add('Home Field', 0.11, 'situational')

    # Form L10 (corr +0.015) and H/R Record (corr +0.04) removed — bootstrap
    # assigned 0.0 coefficients and correlations are indistinguishable from noise.

    # ── 2. Starting pitcher ERA estimator (SIERA → xFIP → xERA → ERA) ────────
    # Priority: SIERA (best long-run predictor) > xFIP > xERA > ERA (worst).
    # All are on the ERA scale (lower = better), so coefficient is the same.
    h_p = home.get('pitcher') or {}
    a_p = away.get('pitcher') or {}

    def _era_estimator(p):
        """Return (value, label_suffix) using best available metric."""
        # Use _safe_float (rejects 0.0 — no pitcher has 0.00 ERA estimator)
        siera = _safe_float(p.get('siera'), None)
        if siera is not None:
            return siera, 'SIERA'
        xfip = _safe_float(p.get('xfip'), None)
        if xfip is not None:
            return xfip, 'xFIP'
        xera = _safe_float(p.get('x_era'), None)
        if xera is not None:
            return xera, 'xERA'
        era = _safe_float(p.get('era'), LEAGUE_XERA)
        return era, 'ERA'

    h_era_val, h_era_lbl = _era_estimator(h_p)
    a_era_val, a_era_lbl = _era_estimator(a_p)

    # Blend SP ERA with team bullpen ERA weighted by starter's avg innings per start.
    # Starter pitches avg_ip/9 of the game; the bullpen covers the rest.
    # Falls back to league average when avg_ip or BP ERA is unavailable.
    h_avg_ip = _safe_float(h_p.get('avg_ip'), LEAGUE_AVG_IP)
    a_avg_ip = _safe_float(a_p.get('avg_ip'), LEAGUE_AVG_IP)
    h_bp_era = _safe_float((home.get('bullpen') or {}).get('era_last_14'), LEAGUE_BP_ERA)
    a_bp_era = _safe_float((away.get('bullpen') or {}).get('era_last_14'), LEAGUE_BP_ERA)

    h_sp_frac = min(h_avg_ip, 9.0) / 9.0
    a_sp_frac = min(a_avg_ip, 9.0) / 9.0
    h_blended = h_sp_frac * h_era_val + (1.0 - h_sp_frac) * h_bp_era
    a_blended = a_sp_frac * a_era_val + (1.0 - a_sp_frac) * a_bp_era

    # Label makes the bullpen blend explicit — this is NOT pure starter SIERA/xFIP,
    # it's starter-weighted-by-typical-innings + bullpen ERA(14). Previously labeled
    # just "SP {metric}", which read as a pure-starter peripheral and could look like
    # it contradicted K%/BB%/Barrel%/Whiff% (which ARE pure-starter, no BP blending)
    # when really it's measuring something different — bullpen form, not just the arm.
    era_label = f'{ha} SP+BP {h_era_lbl}'
    # SIERA coeff 0.37 — bootstrap: 0.37 × 1.4766 × 0.80 = 0.437; held (live perf better)
    # ERA/xFIP fallback coeff 0.63 — bootstrap: 0.63 × 1.5512 × 0.80 = 0.782; held (live perf better)
    era_coeff = 0.37 if h_era_lbl in ('SIERA', 'xFIP') else 0.63
    _add(era_label, (a_blended - h_blended) * era_coeff, 'pitching')

    # ── 5. Starting pitcher WHIP — REMOVED (replaced by K% / BB% below) ─────
    # WHIP was factor 5; predictive metrics K% and BB% supersede it.

    # ── 6. Pitcher K% and BB% — separate predictive factors ─────────────────
    # Coefficients are initial hand-tuned values; to be calibrated via bootstrap refit.

    # K%: higher home pitcher K% is an advantage
    h_k_pct = _safe_float_pct(h_p.get('k_pct'))
    a_k_pct = _safe_float_pct(a_p.get('k_pct'))
    # K% coeff 3.23 — bootstrap: 3.23 × 1.6473 × 0.80 = 4.25; held (live perf better)
    if h_k_pct is not None and a_k_pct is not None:
        _add(f'{ha} SP K%', (h_k_pct - a_k_pct) * 3.23, 'pitching')
    else:
        def _k_pct_from_stats(p):
            k  = p.get('k',  0) or 0
            bf = p.get('bf', 0) or 0
            return k / bf if bf >= 50 else LEAGUE_K_PCT
        _add(f'{ha} SP K%', (_k_pct_from_stats(h_p) - _k_pct_from_stats(a_p)) * 3.23, 'pitching')

    # BB% coeff 0.79 — bootstrap: 0.79 × 0.9863 × 0.80 = 0.62; held (live perf better)
    h_bb_pct = _safe_float_pct(h_p.get('bb_pct'))
    a_bb_pct = _safe_float_pct(a_p.get('bb_pct'))
    if h_bb_pct is not None and a_bb_pct is not None:
        _add(f'{ha} SP BB%', (a_bb_pct - h_bb_pct) * 0.79, 'pitching')
    else:
        def _bb_pct_from_stats(p):
            bb = p.get('bb', 0) or 0
            bf = p.get('bf', 0) or 0
            return bb / bf if bf >= 50 else 0.085
        _add(f'{ha} SP BB%', (_bb_pct_from_stats(a_p) - _bb_pct_from_stats(h_p)) * 0.79, 'pitching')

    # ── 6b. Barrel rate — contact quality against pitcher ────────────────────
    # Coefficient is hand-tuned; to be calibrated via bootstrap refit.
    # barrel_pct is on 0-1 scale (e.g. 0.06 = 6%); a 6% gap → ~0.21 logit.
    h_barrel = _safe_float_pct(h_p.get('barrel_pct'))
    a_barrel = _safe_float_pct(a_p.get('barrel_pct'))
    if h_barrel is not None and a_barrel is not None:
        _add(f'{ha} SP Barrel%', (a_barrel - h_barrel) * 3.5, 'pitching')

    # ── 6c. Whiff rate — swing-and-miss ability ───────────────────────────────
    # Coefficient is hand-tuned; to be calibrated via bootstrap refit.
    h_whiff = _safe_float_pct(h_p.get('whiff_pct'))
    a_whiff = _safe_float_pct(a_p.get('whiff_pct'))
    if h_whiff is not None and a_whiff is not None:
        _add(f'{ha} SP Whiff%', (h_whiff - a_whiff) * 1.5, 'pitching')

    # SP Recent ERA (corr +0.039, bootstrap 0.0) and SP Rest (corr 0.0) removed.

    # ── 7. Team offense — handedness-split xwOBA > overall xwOBA > OPS ───────
    h_xwoba_split = _safe_float(home.get('x_woba_split') or None, None)
    a_xwoba_split = _safe_float(away.get('x_woba_split') or None, None)
    if h_xwoba_split and a_xwoba_split:
        _add(f'{ha} xwOBA Split', (h_xwoba_split - a_xwoba_split) * 2.5, 'offense')
    else:
        h_xwoba = _safe_float(home.get('x_woba') or None, None)
        a_xwoba = _safe_float(away.get('x_woba') or None, None)
        if h_xwoba and a_xwoba:
            _add(f'{ha} xwOBA', (h_xwoba - a_xwoba) * 2.5, 'offense')
        else:
            h_ops = _safe_float(home.get('ops'), LEAGUE_OPS)
            a_ops = _safe_float(away.get('ops'), LEAGUE_OPS)
            # OPS coeff 0.22 — bootstrap: 0.22 × 1.8097 × 0.80 = 0.32; held (live perf better)
            _add(f'{ha} OPS', (h_ops - a_ops) * 0.22, 'offense')

    # ── 10. Team runs per game ────────────────────────────────────────────────
    # Prefer recent R/G (last 15 games) — more responsive to current form.
    # Falls back to season average when recent window < 5 games.
    # R/G coeff 0.10 — bootstrap: 0.10 × 0.2413 × 0.80 = 0.019; empirical floor held at 0.10.
    h_rpg = _safe_float(home.get('recent_rpg') or home.get('runs_pg'), LEAGUE_RPG)
    a_rpg = _safe_float(away.get('recent_rpg') or away.get('runs_pg'), LEAGUE_RPG)
    _add(f'{ha} R/G', (h_rpg - a_rpg) * 0.10, 'offense')

    # ── 11. Offensive K% ──────────────────────────────────────────────────────
    # Off K% coeff 0.05 — bootstrap: 0.05 × 0.7002 × 0.80 = 0.03; held (live perf better)
    h_kpct = _safe_float(home.get('k_pct'), LEAGUE_K_PCT)
    a_kpct = _safe_float(away.get('k_pct'), LEAGUE_K_PCT)
    _add(f'{ha} Off K%', (a_kpct - h_kpct) * 0.05, 'offense')

    # ── 12. Head-to-head history (current season) ─────────────────────────────
    h_id = home.get('id')
    a_id = away.get('id')
    if matchup_history and h_id and a_id:
        h2h = matchup_history.get(frozenset([h_id, a_id]), [])
        if len(h2h) >= 4:
            h_wins = sum(1 for g in h2h if g['winner'] == h_id)
            _add('H2H Record', (h_wins / len(h2h) - 0.5) * 0.6, 'situational')

    # SP IL (removed): can't be backtested (backfill has no historical injury
    # source — injuries_api only ever returns *today's* report), targets an
    # almost-empty intersection even live (a team's announced probable starter
    # is, by definition, already expected to play), and the ESPN scoreboard
    # injuries feed it depended on is sparse. Zero measured contribution, no
    # way to ever measure one. Pen ERA (corr +0.016, bootstrap 0.0) and
    # Closer Yday (corr -0.091, n=7) removed for the same reason — no
    # validated predictive signal, and no historical source to ever test one.

    # ── Bullpen fatigue (IP over last 3 days) ─────────────────────────────────
    # Unlike the earlier "Pen Fatigue" attempt above (corr 0.0, tested live-only
    # on whatever games happened to be on the board that day), this is
    # reconstructed from box scores (bullpen_fatigue.py) across the full 2026
    # season, so it's a real backtest, not a handful of live snapshots.
    # Validated 2026-07-07 on 1307 resolved MLB games: whole-sample correlation
    # is weak (-0.031, same order as several already-rejected factors above),
    # but a 4-cut chronological train/test check found it's asymmetric — the
    # "away team's bullpen is more taxed" side of the effect replicated with
    # the same sign in every out-of-sample window (+1.7% to +4.9% residual
    # win rate for the less-fatigued side), while the "home team's own
    # fatigue hurts them" side flipped sign in 3 of 4 splits (not real, or not
    # yet measurable at this sample size). Kept as a single symmetric
    # differential factor (matching every other factor in this model) rather
    # than only wiring the one-sided effect, with a coefficient shrunk hard
    # to reflect the weaker whole-sample correlation — revisit the asymmetry
    # once more of the season (or 2024-2025 box scores) are backtested.
    # _safe_float_pct (not _safe_float) — 0.0 IP in the last 3 days is a
    # real, meaningful value (a fully rested bullpen), not a missing-data
    # sentinel; _safe_float treats <=0 as invalid and would wrongly drop it.
    h_ip3 = _safe_float_pct((home.get('bullpen') or {}).get('ip_last_3'))
    a_ip3 = _safe_float_pct((away.get('bullpen') or {}).get('ip_last_3'))
    if h_ip3 is not None and a_ip3 is not None:
        # coeff 0.008 logit/IP — prob-space slope -0.004 × 4.0 (logit
        # conversion, same ×4.0 convention as Hist Adj below) × 0.5 shrink
        # for the weak whole-sample correlation.
        _add(f'{ha if a_ip3 > h_ip3 else aa} Bullpen Fatigue',
             (a_ip3 - h_ip3) * 0.008, 'bullpen')

    # ── Historical bias correction ─────────────────────────────────────────────
    # Apply per-team offsets derived from 2026 calibration (30% shrinkage).
    # Converted to logit via ×4.0 (≈ 1/sigmoid′(0) at p=0.5).
    #
    # Combined into ONE net row, not two separate ones. Each team has its own
    # independent bias correction (home's own history, away's own history), and
    # when both happen to point the same direction (e.g. home is a team the
    # model historically overrates, away is one it underrates — both corrections
    # favor away), displaying them as two separately-labeled rows that silently
    # sum together reads as a duplicate/bug even though each is individually
    # correct. Net it out instead — same total probability impact either way,
    # since it's still one additive logit sum, just one legible line instead of
    # two identical-looking ones. Label follows whichever team the NET result favors.
    h_bias = _TEAM_BIAS.get(home.get('name', ''))
    a_bias = _TEAM_BIAS.get(away.get('name', ''))
    net_bias_term = 0.0
    if h_bias:
        net_bias_term += h_bias * 4.0
    if a_bias:
        net_bias_term += -a_bias * 4.0
    if net_bias_term:
        _add(f'{ha if net_bias_term > 0 else aa} Hist Adj', net_bias_term, 'situational')

    # ── Trade-deadline seller discount ─────────────────────────────────────────
    # See _TRADE_DEADLINE_DISCOUNT above. Same sign convention and net-out
    # pattern as Hist Adj just above, but kept as its own labeled row rather
    # than merged in — this is a temporary, manually-curated, removable
    # adjustment and should stay visibly auditable (and separately trackable
    # in the Factor Correlation table) rather than blending into the
    # always-on team-bias line.
    _gd = _game_time_et(game_time_utc)
    _game_date = _gd.date() if _gd else None
    h_td = _trade_deadline_bias(home.get('name', ''), _game_date)
    a_td = _trade_deadline_bias(away.get('name', ''), _game_date)
    net_td_term = 0.0
    if h_td:
        net_td_term += h_td * 4.0
    if a_td:
        net_td_term += -a_td * 4.0
    if net_td_term:
        _add(f'{ha if net_td_term > 0 else aa} Trade Deadline', net_td_term, 'situational')

    # ── Compute probability contributions (marginal delta-p per factor) ───────
    raw_prob = _sigmoid(logit)
    factor_groups = {k: [] for k, _ in _GROUP_ORDER}
    factors_compat = []

    for label, lc, group in _raw:
        dp = raw_prob - _sigmoid(logit - lc)
        prob_pct = round(dp * 100, 1)
        if prob_pct != 0.0:
            factor_groups[group].append((label, prob_pct))
        factors_compat.append((label, lc))

    # Apply Platt scaling when fitted; fall back to hard clamp pre-calibration.
    if _platt is not None:
        a, b = _platt
        home_prob = _sigmoid(a * logit + b)
        home_prob = max(0.28, min(0.82, home_prob))  # wide safety clamp
    else:
        home_prob = max(0.38, min(0.72, raw_prob))

    # Shrink home-team overconfidence above 63%.
    # Calibration on 200 resolved games shows home_prob 65-70% wins only 37.5%
    # (vs 57.9% expected). Away-favored picks in the same range are fine (71.4% WR).
    # Pull home_prob toward 0.63 at 40% of the excess to correct this bias.
    _SHRINK_CAP  = 0.63
    _SHRINK_RATE = 0.40
    if home_prob > _SHRINK_CAP:
        home_prob = _SHRINK_CAP + _SHRINK_RATE * (home_prob - _SHRINK_CAP)

    # Edge-perspective factor view: anchor labels + signs to the favored team.
    # When the away team has the edge, flip all signs and swap {ha}/{aa} in labels
    # so that positive always means "this factor favors the edge team."
    edge_is_home = home_prob >= 0.5
    edge_abbr    = ha if edge_is_home else aa
    if edge_is_home:
        factor_groups_edge = factor_groups
    else:
        factor_groups_edge = {}
        for gk, rows in factor_groups.items():
            flipped = []
            for label, pct in rows:
                new_label = label.replace(ha, '\x00').replace(aa, ha).replace('\x00', aa)
                flipped.append((new_label, round(-pct, 1)))
            factor_groups_edge[gk] = flipped

    return {
        'home_prob':          round(home_prob, 4),
        'away_prob':          round(1.0 - home_prob, 4),
        'logit':              round(logit, 3),
        'factors':            factors_compat,
        'factor_groups':      factor_groups,
        'factor_groups_edge': factor_groups_edge,
        'edge_abbr':          edge_abbr,
        'group_order':        _GROUP_ORDER,
    }
