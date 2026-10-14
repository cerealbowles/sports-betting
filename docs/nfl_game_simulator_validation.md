# NFL game simulator — validation & calibration results

Backtest findings behind `nfl_game_simulator.py`'s current constants, kept
here since the grid search itself is expensive to rerun (~1 hour) and the
scratch data it ran against isn't persisted. If the warehouse grows a full
extra season, or the model changes meaningfully, rerun the relevant script
and update this file rather than trusting these numbers to stay current
forever.

## TL;DR

- `TEAM_CONDITIONING_K` was **2.5** (hand-set, never backtested) → fit to
  **0.5** from a real two-season backtest. 2.5 was badly overconfident:
  its "90%+ confidence" predictions were only right 64.5% of the time.
- A 3-way ensemble blend (team-aggregate + player-grade + simulator) was
  built and tested. **Not deployed** — it performs identically to the
  existing 2-way blend, because the simulator's team-conditioning signal
  is derived from the same offense/defense grades the player-grade model
  already uses, so it adds no independent information once properly fit.

## Methodology

All backtests are leakage-free: each simulated week uses `before_date`
set to that week's own earliest game date, so team grades, usage shares,
and the 4th-down/kickoff resampling pools only ever see data that
genuinely existed before that week was played. Source data is this app's
own stats warehouse (`NflPlay`/`PlayerGameStat`/`TeamGameStat`), spanning
full 2024 and 2025 regular seasons — 36 weeks, 544 completed games.

A large performance fix (see git history — `_tilted_choice()` memoizing
sampling weights per (bucket, tilt) instead of rebuilding them on every
single play draw) cut a full two-season simulation run from an estimated
~5.5 hours down to **~18 minutes**, which is what made a real grid search
practical instead of theoretical.

## Week-by-week spot checks (before the full backtest)

| Week | Win-pick accuracy | Brier | Notes |
|---|---|---|---|
| 2026 Week 1 | 43.8% (7/16) | 0.3762 | No-leakage cutoff = end of *last* season's data — weak signal, as expected |
| 2026 Week 5 | 60.0% (9/15) | 0.2154 | Cutoff has 4 weeks of current-season data — comparable to the real ensemble model |

Small samples (15-16 games), included here only to show the expected
"accuracy improves as the season accumulates real data" pattern held up
before the full backtest confirmed it at scale.

## Full 2024+2025 backtest (544 games, K=2.5, pre-fix)

```
2024: 159/272 = 58.5% accuracy, Brier 0.2833
2025: 154/272 = 56.6% accuracy, Brier 0.2881
COMBINED: 313/544 = 57.5% accuracy, Brier 0.2857 (worse than 0.25 "always 50/50")
```

### Calibration table (K=2.5) — the overconfidence finding

| Predicted confidence | Actual win rate | n |
|---|---|---|
| 50-60% | 48.4% | 95 |
| 60-70% | 55.6% | 90 |
| 70-80% | 54.3% | 105 |
| 80-90% | 59.6% | 114 |
| 90-100% | 64.5% | 124 |

A clean, monotonic overconfidence gap that grows with predicted
confidence — the 50-60% bucket is well-calibrated, but by 90-100% the
model is claiming near-certainty on games it only wins two-thirds of the
time. This directly motivated the K grid search below.

## TEAM_CONDITIONING_K grid search

Re-ran the full 544-game leakage-free backtest once per candidate K,
reusing precomputed (team_id, tilt, real_outcome) tuples across runs
since those don't depend on K — only the sampling weights do.

| K | Accuracy | Brier |
|---|---|---|
| 0.0 | 53.1% | 0.2504 |
| 0.1 | 54.2% | 0.2449 |
| 0.25 | 53.7% | 0.2461 |
| **0.5** | **58.6%** | **0.2445** |
| 1.0 | 57.4% | 0.2493 |
| 1.5 | 57.4% | 0.2621 |
| 2.0 | 58.1% | 0.2757 |
| 2.5 (old) | 57.4% | 0.2885 |
| 3.0 | 57.0% | 0.2983 |

A genuine U-shaped optimum, not a monotonic slide in either direction:

- **K=0.0 (no conditioning at all) scores only 53.1% accuracy** — barely
  above a coin flip. This confirms the conditioning mechanism itself adds
  real signal; it isn't just noise dressed up as confidence.
- Accuracy and Brier both improve together up through **K=0.5**, the best
  value on *both* metrics simultaneously.
- Above 0.5, Brier gets monotonically worse every step, up to 0.2983 at
  K=3.0 — the old value (2.5) was sitting well past the optimum, on the
  overconfident side, scoring worse than every other candidate except 3.0.

**Adopted: `TEAM_CONDITIONING_K = 0.5`** (see `nfl_game_simulator.py`).

## 3-way ensemble blend investigation

Motivating observation: comparing the simulator's win probability (at the
newly-fit K=0.5) against the *existing* 2-signal ensemble's stored
predictions on 431 of the 544 games —

```
Pearson correlation (sim vs existing ensemble): 0.511
Agreement on favorite: 65.4% (282/431)
Sim Brier (matched subset): 0.2414
Ensemble Brier (matched subset): 0.2292
Naive unweighted 50/50 blend Brier: 0.2199  ← beat both individually
```

That looked like real complementary signal, so a proper 3-way logistic
regression stack was built (`nfl_ensemble3_bootstrap.py`, mirrors the
existing `nfl_ensemble_bootstrap.py`'s 2-way fit exactly) — team-aggregate
+ player-grade + simulator, each on its own logit, same as the existing
blend.

```
team-aggregate alone:  64.6% accuracy, 0.2357 Brier
player-grade alone:    58.0% accuracy, 0.2415 Brier
simulator alone:       57.0% accuracy, 0.2450 Brier
3-way blended:         65.2% accuracy, 0.2230 Brier   (n=512)

fit: {intercept: -0.0226, team_weight: 0.4668, player_weight: 0.2613, sim_weight: -0.3154}
```

**Result: no real improvement.** The existing 2-way blend (documented in
`nfl_ensemble_model.py`) already scores 65.2% accuracy / 0.2231 Brier —
statistically identical to the 3-way fit's 65.2% / 0.2230. The fitted
`sim_weight` came back **negative**.

**Why**: `nfl_game_simulator.py`'s team-conditioning tilt is computed from
`nfl_player_model.compute_offense_grade()`/`compute_defense_grade()` — the
*same functions* the existing player-grade model (`player_prob`) is built
from. `sim_prob` and `player_prob` aren't independent signals; they're two
different transformations of substantially the same underlying input.
That also explains why `player_weight` shifted from -0.0807 (in the
existing 2-way fit) to +0.2613 here — classic multicollinearity, credit
getting redistributed between two correlated predictors without the
overall fit actually improving. The earlier naive-50/50-blend result
wasn't wrong, but it was comparing the simulator against the *already-
blended* ensemble output, which hid this shared lineage.

**Decision: not deployed.** The 3-way blend doesn't measurably improve on
the existing 2-way ensemble, and the simulator is far more expensive to
compute live than the other two signals (the reason it runs as a daily
background job rather than per-request at all). `nfl_ensemble3_bootstrap.py`
is kept in the repo as a real, reusable fitting script in case a future,
genuinely independent third signal (not derived from the same
offense/defense grades) is worth blending in instead.

## Known gaps carried forward (see `nfl_game_simulator.py`'s own module docstring for the full list)

- No real game clock — "quarter" in the UI is an approximation by drive
  order, not elapsed time.
- 4th-down go-for-it rate/scale (`FOURTH_DOWN_GO_FOR_IT_RATE`,
  `FOURTH_DOWN_GO_FOR_IT_TILT_SCALE`) and the punt/FG value heuristic
  (`_fourth_down_value()`) remain hand-set, not backtested — there's no
  clean real/declined-to-go-for-it sample to fit either against.
- Kickoffs are deliberately unconditioned (no offense/defense-grade
  signal relevant to a return unit).
- League-wide play sampling, not team- or opponent-conditioned beyond the
  tilt mechanism — two seasons across 32 teams isn't deep enough to split
  by team AND situation without buckets going empty.
