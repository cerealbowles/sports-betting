"""
hockey_total_model.py — Game-total (O/U) model for NHL.

Unlike basketball/football, hockey has no possession-count "pace" stat
readily available via ESPN/NHL's APIs, and games are a fixed 60 minutes
regardless of team quality — so there's no pace dimension to adjust for.
This is a simple opponent-adjusted scoring-environment model instead:

  home_expected_goals = (home_GF/game + away_GA/game) / 2
  away_expected_goals = (away_GF/game + home_GA/game) / 2
  proj_total           = home_expected_goals + away_expected_goals

Uses only GF/GA-per-game, which nhl_api.py already computes for the
win-probability model — no new live data fetch needed.

Validated against a full completed season of actual totals before shipping
(2024-25 NHL season, n=1312): corr=0.178, R²=0.032 after calibration. This
is meaningfully weaker than the football/basketball total models (R²
0.09-0.27) — hockey totals are just noisier with only team-level scoring
rates to go on (goaltender matchup, which swings NHL totals a lot, isn't
in this signal at all). Shipped anyway per explicit go-ahead, but the
weak fit is the reason this reads quietly rather than as a sharp number.

Refit periodically — these are a one-time fit from the 2024-25 season.
"""

# actual_total ≈ intercept + coef * raw_proj
CALIBRATION = {
    'NHL': (-6.3531, 2.044779, 2.2849),
}


def predict_total(home_gf_pg, home_ga_pg, away_gf_pg, away_ga_pg):
    """Returns {total_projection} or None if GF/GA inputs are missing."""
    if None in (home_gf_pg, home_ga_pg, away_gf_pg, away_ga_pg):
        return None

    home_exp = (home_gf_pg + away_ga_pg) / 2
    away_exp = (away_gf_pg + home_ga_pg) / 2
    raw_proj = home_exp + away_exp

    intercept, coef, _ = CALIBRATION['NHL']
    return {'total_projection': round(intercept + coef * raw_proj, 2)}
