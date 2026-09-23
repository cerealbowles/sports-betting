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

  home_expected_runs = (home_team_RPG + away_starter_ERA) / 2
  away_expected_runs = (away_team_RPG + home_starter_ERA) / 2
  proj_total          = home_expected_runs + away_expected_runs

Both inputs (team RPG, probable starter ERA) are already fetched live by
mlb_api.py for the win-prob model — no new API calls needed.

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


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def predict_total(home_runs_pg, home_pitcher_era, away_runs_pg, away_pitcher_era):
    """Returns {total_projection} or None if RPG/ERA inputs are missing
    (e.g. probable starter not yet announced)."""
    home_rpg = _safe_float(home_runs_pg)
    away_rpg = _safe_float(away_runs_pg)
    home_era = _safe_float(home_pitcher_era)
    away_era = _safe_float(away_pitcher_era)
    if None in (home_rpg, away_rpg, home_era, away_era):
        return None

    home_exp = (home_rpg + away_era) / 2
    away_exp = (away_rpg + home_era) / 2
    raw_proj = home_exp + away_exp

    intercept, coef, _ = CALIBRATION['MLB']
    return {'total_projection': round(intercept + coef * raw_proj, 2)}
