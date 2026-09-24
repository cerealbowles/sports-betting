"""
football_total_model.py — Game-total (O/U) model for NFL and CFB, using the
same pace-adjustment shape as bball_total_model.py but with plays-per-game
as the pace proxy (basketball uses estimated possessions from FGA/OREB/
TOV/FTA; football doesn't expose drive counts via this endpoint, so
offensive plays run is the next-best per-team volume signal):

  pace        = totalOffensivePlays / teamGamesPlayed
  ORtg        = PPG / pace * 100                 (points per 100 plays)
  DRtg        = PPG_allowed / pace * 100
  game_pace   = avg(home_pace, away_pace)
  proj_total  = game_pace * (home_ORtg + away_DRtg + away_ORtg + home_DRtg) / 200

Validated against a full completed season of actual totals before
shipping — NFL: R²=0.23, corr=0.48, n=272 (2025 season); CFB: R²=0.27,
corr=0.52, n=299 (2025 season, FBS-vs-FBS games only). Both clear the bar
by a wider margin than the basketball version did, likely because
plays-per-game is a more stable/less noisy pace signal than the estimated-
possession formula basketball has to use. CALIBRATION corrects the raw
formula's systematic compression (fit once per sport via linear regression
of actual total against raw formula output); SIGMA is the calibrated
residual std.

Refit periodically — these are one-time fits from the 2025 season.
"""
import time
import requests

SPORT_SLUGS = {
    'NFL': 'football/nfl',
    'CFB': 'football/college-football',
}

# {sport: (intercept, coef, sigma)} — actual_total ≈ intercept + coef * raw_proj
CALIBRATION = {
    'NFL': (-52.3002, 2.135986, 12.120),
    'CFB': (-33.3024, 1.629036, 13.865),
}

_BASE = 'https://site.api.espn.com/apis/site/v2/sports'
_cache = {}
_TTL = 6 * 3600  # season-aggregate play counts move slowly — cache long


def _fetch_pace(sport, team_id):
    """Returns plays-per-game (float) for team_id, or None."""
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

    gp    = flat.get('teamGamesPlayed') or flat.get('gamesPlayed')
    plays = flat.get('totalOffensivePlays')
    result = (plays / gp) if (gp and plays) else None

    _cache[key] = (result, now)
    return result


def predict_total(sport, home_id, home_ppg, home_ppg_allowed, away_id, away_ppg, away_ppg_allowed):
    """Returns {total_projection, pace} or None if pace data / PPG inputs
    aren't available for this matchup."""
    if sport not in CALIBRATION:
        return None
    if None in (home_ppg, home_ppg_allowed, away_ppg, away_ppg_allowed):
        return None

    home_pace = _fetch_pace(sport, home_id)
    away_pace = _fetch_pace(sport, away_id)
    if not home_pace or not away_pace or home_pace <= 0 or away_pace <= 0:
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
