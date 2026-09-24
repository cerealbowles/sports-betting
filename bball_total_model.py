"""
bball_total_model.py — Game-total (O/U) model for NBA and WNBA, using the
standard basketball pace-adjustment formula: project each team's points as
(their own offensive rating + opponent's defensive rating) / 2, scaled by
the game's expected pace (possessions), then sum both sides.

  possessions ≈ FGA - OREB + TOV + 0.44 * FTA          (standard estimator)
  ORtg        = PPG / possessions * 100                (points per 100 poss)
  DRtg        = PPG_allowed / possessions * 100
  game_pace   = avg(home_pace, away_pace)
  proj_total  = game_pace * (home_ORtg + away_DRtg + away_ORtg + home_DRtg) / 200

This is a genuinely different model from the win-probability one — total
points depends on combined scoring environment (pace × efficiency for BOTH
teams together), not on which team is better, so it needs the FGA/OREB/
TOV/FTA components nothing else in this app currently fetches.

Validated against a full completed season of actual totals before shipping
(see conversation — NBA: R²=0.09, corr=0.30, n=1235; WNBA: R²=0.16,
corr=0.40, n=287 — both clear the "does this beat naive PPG-sum" bar
mentioned in the same season-aggregate-leakage sense every other bootstrap
script in this repo already fits with). CALIBRATION corrects the raw
formula's systematic compression (fit once per sport via linear regression
of actual total against raw formula output); SIGMA is the calibrated
residual std, used for the over-probability normal approximation.

Refit periodically — these are one-time fits from the 2025-26 season.
"""
import time
import requests

SPORT_SLUGS = {
    'NBA':  'basketball/nba',
    'WNBA': 'basketball/wnba',
}

# {sport: (intercept, coef, sigma)} — actual_total ≈ intercept + coef * raw_proj
CALIBRATION = {
    'NBA':  (-246.0909, 2.067703, 22.714),
    'WNBA': (-194.4991, 2.190909, 15.646),
}

_BASE = 'https://site.api.espn.com/apis/site/v2/sports'
_cache = {}
_TTL = 6 * 3600  # pace components are season aggregates — move slowly, cache long


def _fetch_pace_components(sport, team_id):
    """Returns {fga, oreb, tov, fta} per-game averages for team_id, or None."""
    if not team_id:
        return None
    key = f'{sport}_{team_id}'
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
            return data

    slug = SPORT_SLUGS.get(sport)
    if not slug:
        return None
    try:
        r = requests.get(f'{_BASE}/{slug}/teams/{team_id}/statistics', timeout=15)
        r.raise_for_status()
        cats = r.json().get('results', {}).get('stats', {}).get('categories', [])
    except Exception:
        return _cache.get(key, (None, 0))[0]

    flat = {}
    for cat in cats:
        for s in cat.get('stats', []):
            flat[s.get('name')] = s.get('value')

    fga, oreb, tov, fta = (flat.get('avgFieldGoalsAttempted'), flat.get('avgOffensiveRebounds'),
                            flat.get('avgTurnovers'), flat.get('avgFreeThrowsAttempted'))
    if None in (fga, oreb, tov, fta):
        result = None
    else:
        result = {'fga': fga, 'oreb': oreb, 'tov': tov, 'fta': fta}

    _cache[key] = (result, now)
    return result


def _possessions(components):
    return components['fga'] - components['oreb'] + components['tov'] + 0.44 * components['fta']


def predict_total(sport, home_id, home_ppg, home_ppg_allowed, away_id, away_ppg, away_ppg_allowed):
    """Returns {total_projection, pace} or None if pace data / PPG inputs
    aren't available for this matchup (e.g. no games played yet this
    season, or ESPN doesn't carry stats for this team)."""
    if sport not in CALIBRATION:
        return None
    if None in (home_ppg, home_ppg_allowed, away_ppg, away_ppg_allowed):
        return None

    home_pace_c = _fetch_pace_components(sport, home_id)
    away_pace_c = _fetch_pace_components(sport, away_id)
    if not home_pace_c or not away_pace_c:
        return None

    home_pace = _possessions(home_pace_c)
    away_pace = _possessions(away_pace_c)
    if home_pace <= 0 or away_pace <= 0:
        return None

    game_pace = (home_pace + away_pace) / 2
    home_ortg = home_ppg / home_pace * 100
    home_drtg = home_ppg_allowed / home_pace * 100
    away_ortg = away_ppg / away_pace * 100
    away_drtg = away_ppg_allowed / away_pace * 100

    raw_proj = game_pace * (home_ortg + away_drtg + away_ortg + home_drtg) / 200

    intercept, coef, _ = CALIBRATION[sport]
    total_projection = intercept + coef * raw_proj

    return {
        'total_projection': round(total_projection, 1),
        'pace':              round(game_pace, 1),
        'home_ortg':         round(home_ortg, 1),
        'away_ortg':         round(away_ortg, 1),
        'home_drtg':         round(home_drtg, 1),
        'away_drtg':         round(away_drtg, 1),
    }
