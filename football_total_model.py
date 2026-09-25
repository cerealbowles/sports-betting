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

# Approximate modern-era league-average pace (offensive plays/game) and
# rating (points per 100 plays) — the baseline factor_breakdown() below
# measures each game's inputs against. Reference figures, not fit from this
# app's own data (same spirit as mlb_total_model.LEAGUE_BP_ERA); refine if
# a season's actual averages drift from these. CFB runs a faster, higher-
# scoring pace than the pros.
LEAGUE_AVG = {
    'NFL': {'pace': 64.0, 'rating': 36.0},
    'CFB': {'pace': 72.0, 'rating': 40.0},
}

_BASE = 'https://site.api.espn.com/apis/site/v2/sports'
_cache = {}
_TTL = 6 * 3600  # season-aggregate play counts move slowly — cache long
_fail_ts = {}
_FAIL_BACKOFF = 300  # 5 min — see bball_total_model._FAIL_BACKOFF, same fix:
                     # a failed fetch used to never get cached at all, so a
                     # slow/erroring team retried the full 15s timeout on
                     # every single request until the endpoint recovered.


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
    if now - _fail_ts.get(key, 0) < _FAIL_BACKOFF:
        return _cache.get(key, (None, 0))[0]

    slug = SPORT_SLUGS.get(sport)
    if not slug:
        return None
    try:
        r = requests.get(f'{_BASE}/{slug}/teams/{team_id}/statistics', timeout=15)
        r.raise_for_status()
        cats = r.json().get('results', {}).get('stats', {}).get('categories', [])
    except Exception:
        _fail_ts[key] = now
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


def factor_breakdown(sport, total_model):
    """Splits predict_total()'s raw_proj = pace * ratings_sum / 200 into
    signed per-input contributions — see bball_total_model.factor_breakdown's
    docstring, identical reasoning/formula shape (this model is deliberately
    the same shape, plays-per-game standing in for possessions) AND the same
    fix: takes predict_total()'s own result dict instead of re-fetching pace
    over the network a second time per game. Exact identity: baseline +
    sum(contribs) == predict_total(...)['total_projection'].
    """
    avg = LEAGUE_AVG.get(sport)
    if not avg or sport not in CALIBRATION or not total_model:
        return None

    game_pace = total_model['pace']
    home_ortg = total_model['home_ortg']
    home_drtg = total_model['home_drtg']
    away_ortg = total_model['away_ortg']
    away_drtg = total_model['away_drtg']
    ratings_sum = home_ortg + away_drtg + away_ortg + home_drtg

    intercept, coef, _ = CALIBRATION[sport]
    pace0 = avg['pace']
    ratings0 = 4 * avg['rating']
    k = coef / 200.0

    pace_dev = game_pace - pace0
    ratings_dev = ratings_sum - ratings0
    contribs = [
        ('Pace', k * ratings0 * pace_dev),
        ('Home Off. Rating', k * pace0 * (home_ortg - avg['rating'])),
        ('Away Def. Rating', k * pace0 * (away_drtg - avg['rating'])),
        ('Away Off. Rating', k * pace0 * (away_ortg - avg['rating'])),
        ('Home Def. Rating', k * pace0 * (home_drtg - avg['rating'])),
        ('Interaction (pace × rating)', k * pace_dev * ratings_dev),
    ]
    baseline = intercept + coef * (pace0 * ratings0 / 200.0)
    return {
        'baseline': round(baseline, 2),
        'contribs': [(label, round(c, 3)) for label, c in contribs],
    }
