Self hosted Flask application to track my sports bets using the Kelly Criterion for recommended amounts. This is just for fun but wanted to be more responsible and smarter with betting.

---

## MLB Model — Recalibrating Coefficients

The MLB win probability model uses logistic regression weights hand-tuned to the values in `mlb_model.py`. `bootstrap_2024.py` fits a fresh logistic regression against 3 seasons of historical games and tells you whether each coefficient should move up or down.

### When to re-run

- After any structural model change (new factor, removed factor, changed input data)
- After significant new data accumulates — roughly once per month mid-season
- After running `backfill_2026.py` inside Docker (so 2026 predictions reflect the latest model)

### How to run

```bash
# From the project root (outside Docker — uses local venv)
python3 bootstrap_2024.py
```

The script fetches 2026 games fresh every run. 2024 and 2025 are cached in `bootstrap_training_2024.json` / `bootstrap_training_2025.json` — delete those files to force a full re-fetch (needed after a structural model change).

Output is saved to `refit_results.json` and printed to the terminal.

### Reading the output

```
Factor             Coeff  Suggested action
─────────────────────────────────────────
Hm SP K%          +2.374  ★  INCREASE weight
Hm R/G            +0.227  ↓  reduce weight
```

The `Coeff` column is the logistic regression coefficient the bootstrap fit *for the stored factor value* (which is already `raw_diff × current_coeff`). To get the suggested new coefficient in `mlb_model.py`, apply the **80% shrinkage rule**:

```
new_coeff = current_coeff × bootstrap_coeff × 0.80
```

The 0.80 factor prevents over-fitting to historical data.

### Applying the changes

Edit the relevant lines in `mlb_model.py`. Each coefficient has an inline comment showing the derivation, e.g.:

```python
# SP K% coeff 3.23 — bootstrap: 1.70 × 2.374 × 0.80 = 3.229, rounded to 3.23
_add(f'{ha} SP K%', (h_k_pct - a_k_pct) * 3.23, 'pitching')
```

Update the comment to match the new value and calculation.

### Exceptions and judgment calls

| Factor | Notes |
|---|---|
| **R/G** | Bootstrap consistently under-rates this metric (noisy, correlated with OPS). Use the formula as a floor-check, not a strict target. Current floor: `0.10`. |
| **Rolling factors** (fatigue, H2H, injuries) | Bootstrap always returns 0 — these use empty arrays in training. Do **not** zero these out; they are hand-tuned on live data. |
| **xwOBA / xwOBA Split** | Not in bootstrap training set (FanGraphs/Statcast data not backfilled). Hand-tuned only. |

### Current coefficients (as of 2026-06-01)

| Factor | Coefficient | Bootstrap basis |
|---|---|---|
| SP K% | 3.23 | `1.70 × 2.374 × 0.80` — bootstrap suggests 4.25; held, live perf better |
| SP ERA (SIERA/xFIP) | 0.37 | `0.27 × 1.706 × 0.80` — bootstrap suggests 0.44; held, live perf better |
| SP ERA (ERA fallback) | 0.63 | `0.50 × 1.571 × 0.80` — bootstrap suggests 0.78; held, live perf better |
| SP BB% | 0.79 | `1.05 × 0.937 × 0.80` — bootstrap suggests 0.62; held, live perf better |
| OPS | 0.22 | `0.17 × 1.612 × 0.80` — bootstrap suggests 0.32; held, live perf better |
| R/G (opp-adjusted) | 0.10 | empirical floor (bootstrap → 0.019) |
| Off K% | 0.05 | `0.10 × 0.670 × 0.80` — bootstrap suggests 0.03; held, live perf better |
| Home Field | 0.11 | `0.13 × 1.050 × 0.80` — bootstrap suggests 0.09; held, live perf better |

ERA inputs are blended: `SP_ERA × (avg_ip / 9) + BP_ERA × (1 − avg_ip / 9)` so the coefficient covers both starter and bullpen quality.
R/G uses the opponent-quality-adjusted rolling 15-game average (`actual_runs × (4.20 / opp_era)`, capped ×0.60–×1.60).

### After updating coefficients

1. Delete the 2026 cache so it regenerates with the new model:
   ```bash
   # bootstrap_2024.json is the combined file; 2026 is always re-fetched automatically
   # No manual deletion needed for 2026
   ```
2. Re-run `bootstrap_2024.py` to confirm in-sample Brier improved.
3. Re-run `backfill_2026.py` inside Docker to update stored predictions with the new coefficients.
4. Monitor the **Model Performance** page — live 2026 Brier and calibration table are the ground truth.
