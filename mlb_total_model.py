"""
mlb_total_model.py — Game-total (O/U) model for MLB.

Like hockey, baseball has no possession-count "pace" stat and a fixed game
length (9 innings) — so this is an opponent-adjusted scoring-environment
model, same shape as hockey_total_model.py. The key difference from the
naive version of that shape: baseball run totals are dominated by the
SPECIFIC starting pitcher for that game, not team-season averages — a
team-only version of this model (runs-per-game vs team runs-allowed)
only reached R²=0.024; swapping in the actual probable starter's ERA
more than doubled it (R²=0.053, see below). Uses ERA specifically (not
SIERA/xFIP, which mlb_model.py prefers for the win-prob model) because
that's what was validated — could be refined to SIERA later.

    away_pitching_ERA   = away_starter/bullpen innings blend
    home_pitching_ERA   = home_starter/bullpen innings blend
    home_expected_runs  = (home_team_RPG + away_pitching_ERA) / 2
    away_expected_runs  = (away_team_RPG + home_pitching_ERA) / 2
  proj_total          = home_expected_runs + away_expected_runs

All inputs are already fetched live by mlb_api.py for the win-prob model — no
new API calls are needed. Recent RPG, when available, contributes a limited
20% blend with season RPG.

Validated against a full completed season of actual totals before
shipping (2025 season, n=2426 games with both team stats and a probable
starter on record): corr=0.230, R²=0.053 after calibration. Still weaker
than football/basketball (R² 0.09-0.27) — a starting pitcher's ERA going
into a game is itself a full-season average, not that day's stuff/health,
so there's a real ceiling here without game-day-specific pitcher data —
but meaningfully ahead of the team-only version and NHL's total model
(R²=0.032).

Refit periodically — this is a one-time fit from the 2025 season.
"""

# actual_total ≈ intercept + coef * raw_proj
CALIBRATION = {
    'MLB': (-0.7624, 1.110980, 4.469),
}

LEAGUE_BP_ERA = 4.10
DEFAULT_STARTER_IP = 5.5
BULLPEN_SAMPLE_IP = 30.0
RECENT_OFFENSE_WEIGHT = 0.20


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _value(data, key):
    return data.get(key) if isinstance(data, dict) else data


def _bullpen_era(bullpen):
    """Shrink a recent bullpen ERA toward league average by sample size."""
    era = _safe_float(_value(bullpen, 'era_last_14'))
    if era is None:
        return LEAGUE_BP_ERA
    sample_ip = _safe_float(_value(bullpen, 'ip_last_14'))
    if sample_ip is None or sample_ip <= 0:
        return LEAGUE_BP_ERA
    weight = min(sample_ip / BULLPEN_SAMPLE_IP, 1.0)
    return LEAGUE_BP_ERA + weight * (era - LEAGUE_BP_ERA)


def _pitching_era(pitcher, bullpen):
    """Blend probable-starter ERA and bullpen ERA by expected innings."""
    starter_era = _safe_float(_value(pitcher, 'era'))
    if starter_era is None:
        return None
    starter_ip = (_safe_float(pitcher.get('avg_ip'))
                  if isinstance(pitcher, dict) else DEFAULT_STARTER_IP)
    starter_ip = starter_ip or DEFAULT_STARTER_IP
    starter_ip = min(max(starter_ip, 0.0), 9.0)
    bullpen_era = _bullpen_era(bullpen)
    return (starter_ip * starter_era + (9.0 - starter_ip) * bullpen_era) / 9.0


def _offense(runs_pg, recent_rpg):
    season = _safe_float(runs_pg)
    if season is None:
        return None
    recent = _safe_float(recent_rpg)
    if recent is None:
        return season
    return (1.0 - RECENT_OFFENSE_WEIGHT) * season + RECENT_OFFENSE_WEIGHT * recent


def predict_total(home_runs_pg, home_pitcher, away_runs_pg, away_pitcher,
                  home_bullpen=None, away_bullpen=None,
                  home_recent_rpg=None, away_recent_rpg=None):
    """Return a calibrated projection and its pitching components.

    Pitcher arguments are MLB API dictionaries. Scalar ERA values remain
    accepted for compatibility with offline callers.
    """
    home_offense = _offense(home_runs_pg, home_recent_rpg)
    away_offense = _offense(away_runs_pg, away_recent_rpg)
    home_pitching = _pitching_era(home_pitcher, home_bullpen)
    away_pitching = _pitching_era(away_pitcher, away_bullpen)
    if None in (home_offense, away_offense, home_pitching, away_pitching):
        return None

    home_exp = (home_offense + away_pitching) / 2
    away_exp = (away_offense + home_pitching) / 2
    raw_proj = home_exp + away_exp

    intercept, coef, _ = CALIBRATION['MLB']
    return {
        'total_projection': round(intercept + coef * raw_proj, 2),
        'home_pitching_era': round(home_pitching, 2),
        'away_pitching_era': round(away_pitching, 2),
        'home_offense': round(home_offense, 2),
        'away_offense': round(away_offense, 2),
    }
