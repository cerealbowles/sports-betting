"""
game_detail_common.py — sport-parameterized plumbing shared by every
<sport>_game_detail.py module (nfl_game_detail.py, nba_game_detail.py,
wnba_game_detail.py, nhl_game_detail.py): the normal CDF helper, the
odds_history sport-key lookup, and the two GamePrediction-backed real-data
computations (market_gap_sd for the outlier guard, ats_record_when_edge for
the blend strip) — see nfl_game_detail.py's original versions of these for
the reasoning behind each. Pulled out here once NFL's page was proven
working, instead of copy-pasting into three more modules.
"""
import json
import math
import time

import spread_model
import game_math as gm

_cache = {}


def cached_compute(key, ttl, compute_fn):
    """TTL memoization for a pure-Python/DB-backed computation (e.g. a
    league-rank table) — same idea as nfl_api._cached_compute, just not
    every sport's <sport>_api.py module has that helper (nba_api.py/
    wnba_api.py/nhl_api.py only have the HTTP-fetch-flavored _cached_get),
    so this lives here instead of being assumed present per-sport."""
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < ttl:
            return data
    data = compute_fn()
    _cache[key] = (data, now)
    return data


def recommendation_gate_note(sport, market, tag, amer_odds=None, edge_pts=None):
    """None, or a short explanation of why a PLAY/LEAN-tagged verdict here
    still wouldn't earn this site's own real recommendation (the ★ on a
    schedule card / the Tonight page's picks list — app.py's
    _best_market_pick). This page's verdict cards only run the generic
    EV/Kelly/confidence math (game_math.py) — they have no idea about the
    extra, backtest-derived gates layered on top of the real picks list
    (a specific sport's price band, an edge floor, or a model that's never
    been validated out-of-sample for this market at all), so a card can
    show PLAY here while that exact pick would never surface there. That
    mismatch is confusing on its own (a user can't tell "the site trusts
    this" from "the site computed this number but doesn't trust it yet"),
    so this surfaces it instead of leaving it silent.

    Lazy-imports the actual gate functions/constants from app.py and
    spread_model.py rather than re-deriving their thresholds here — so
    this can never silently drift out of sync with the real gates (if one
    changes, this note reflects it automatically). None if `tag` isn't
    PLAY/LEAN (nothing to explain — the card already reads as "not a real
    pick") or no known gate actually applies."""
    if tag not in ('PLAY', 'LEAN'):
        return None
    import app as _app

    if market == 'MONEYLINE':
        if sport == 'NBA' and amer_odds is not None and not _app._nba_ml_gate_passes(amer_odds):
            lo, hi = _app._NBA_ML_WEAK_PRICE_BAND
            return (f"Not one of this site's real picks: NBA moneylines priced {lo} to {hi} "
                    f"backtested a 45-51% win rate (-16% to -20% ROI) on 726 resolved games, "
                    f"even among edges this size — excluded by price alone, regardless of edge.")
        if sport == 'NHL' and amer_odds is not None and edge_pts is not None:
            if not _app._nhl_ml_gate_passes(edge_pts, amer_odds):
                if edge_pts < _app._NHL_ML_EDGE_FLOOR:
                    return (f"Not one of this site's real picks: NHL moneylines need at least "
                            f"{_app._NHL_ML_EDGE_FLOOR}pt edge to clear this site's own floor "
                            f"(sub-floor picks backtested -21.2% ROI on 789 games) — this one "
                            f"is {edge_pts:.1f}pt.")
                lo, hi = _app._NHL_ML_COINFLIP_BAND
                return (f"Not one of this site's real picks: NHL moneylines priced {lo} to {hi} "
                        f"(near coin-flip) were the single worst price band even unfiltered "
                        f"(34.8% win rate) — excluded regardless of edge.")
    elif market == 'SPREAD':
        if not spread_model.is_validated(sport):
            return (f"Not one of this site's real picks: {sport}'s spread model isn't validated "
                    f"yet (only MLB and NHL are — see spread_model.VALIDATED_SPORTS) — this "
                    f"edge never reaches the site's live picks.")
    return None


def line_movement(short_sport, home_name, away_name, game_time_utc):
    """Line-movement series (ML no-vig %, spread, total) for one game, read
    from odds_history.get_snapshot_series() under the real Odds-API sport
    key (see odds_history_sport_key()). Shared by every <sport>_game_
    detail.py — identical logic, just parameterized by sport."""
    import odds_history

    key = hist_key(home_name, away_name, game_time_utc)
    snaps = odds_history.get_snapshot_series(odds_history_sport_key(short_sport), key)
    series = {'ml': [], 'spread': [], 'total': []}
    for s in snaps:
        if s.get('home_odds') is not None and s.get('away_odds') is not None:
            try:
                imp_h = gm.implied_prob(s['home_odds'])
                imp_a = gm.implied_prob(s['away_odds'])
                novig_h, _novig_a, _hold = gm.no_vig_pair(imp_h, imp_a)
                series['ml'].append({'t': s['t'], 'value': round(novig_h * 100, 1)})
            except Exception:
                pass
        if s.get('home_spread') is not None:
            series['spread'].append({'t': s['t'], 'value': s['home_spread']})
        if s.get('total_line') is not None:
            series['total'].append({'t': s['t'], 'value': s['total_line']})
    return series


def odds_history_sport_key(short_sport):
    """odds_history rows are written under the Odds-API sport key (e.g.
    'americanfootball_nfl', 'basketball_nba'), not the short key ('nfl',
    'nba') callers use elsewhere — odds_api.get_odds_map(short_sport) looks
    it up via odds_api.SPORT_KEYS before ever calling odds_history.record()/
    save_pregame(). Reading snapshots back with the short key finds nothing."""
    import odds_api
    return odds_api.SPORT_KEYS.get(short_sport, short_sport)


def hist_key(home_name, away_name, game_time_utc):
    from odds_api import _normalize
    event_hour = (game_time_utc or '')[:13]
    return f"{_normalize(home_name)}_{_normalize(away_name)}_{event_hour}"


_RESOLVED_STATS_TTL = 900  # resolved predictions only change when a game finishes


def market_gap_sd(sport, market):
    """Real stdev of (model line - market line) across this app's own
    resolved predictions (GamePrediction) for `sport` — the outlier guard's
    'market_gap_sd'. None (guard never trips) if fewer than 5 resolved games
    have both a stored market line and factors to recompute the model's
    line from — honest under-data behavior rather than a guessed constant.

    Walks every resolved prediction for the sport and re-runs
    spread_model.predict_margin() on each one, so it's cached (same TTL
    idea as _league_rank_table in nfl_game_detail.py): results only change
    when a new game resolves, not on every game-detail page load."""
    def _compute():
        from app import app as flask_app, GamePrediction

        with flask_app.app_context():
            rows = (GamePrediction.query
                    .filter_by(sport=sport)
                    .filter(GamePrediction.home_won.isnot(None))
                    .all())
        diffs = []
        for r in rows:
            if r.spread_pick_line is None:
                continue
            try:
                factors = json.loads(r.factors_json or '[]')
            except (TypeError, ValueError):
                factors = []
            if not factors:
                continue
            margin = spread_model.predict_margin(sport, factors)
            if margin is None:
                continue
            diffs.append(margin - r.spread_pick_line)
        if len(diffs) < 5:
            return None
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        return math.sqrt(var) or None

    return cached_compute(f'market_gap_sd_{sport}_{market}', _RESOLVED_STATS_TTL, _compute)


def ats_record_when_edge(sport, threshold_pts):
    """Real season-to-date ATS record in resolved predictions for `sport`
    where the model's spread edge was at least `threshold_pts`. None if
    fewer than 5 qualifying resolved games. Cached for the same reason as
    market_gap_sd() above — same underlying query+loop, same TTL."""
    def _compute():
        from app import app as flask_app, GamePrediction

        with flask_app.app_context():
            rows = (GamePrediction.query
                    .filter_by(sport=sport)
                    .filter(GamePrediction.spread_covered.isnot(None))
                    .all())
        w = l = 0
        for r in rows:
            try:
                factors = json.loads(r.factors_json or '[]')
            except (TypeError, ValueError):
                factors = []
            if not factors or r.spread_pick_line is None:
                continue
            margin = spread_model.predict_margin(sport, factors)
            if margin is None:
                continue
            if abs(margin - r.spread_pick_line) >= threshold_pts:
                if r.spread_covered:
                    w += 1
                else:
                    l += 1
        if w + l < 5:
            return None
        return {'threshold_pts': threshold_pts, 'w': w, 'l': l, 'pct': round(100 * w / (w + l), 1)}

    return cached_compute(f'ats_record_{sport}_{threshold_pts}', _RESOLVED_STATS_TTL, _compute)
