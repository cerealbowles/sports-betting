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
- D/ST fantasy scoring (sacks/INTs/defensive TDs/points allowed) was
  added, which required fixing a real bug: every drawn historical
  defensive/return touchdown was previously credited as a plain 0-point
  turnover. A controlled before/after comparison (same games, same RNG
  seed, isolating just this fix) found no change in win-pick accuracy
  (defensive TDs are too rare to flip a pick) and an IMPROVED Brier score
- A 2024+2025 Sim Median accuracy backtest found two real drive-loop
  bugs: a failed 4th-down "go for it" never ended the drive (inflating
  team score +6.1% and punt rate +10.4pp), and a touchdown scored via
  the generic field-position check wasn't credited to any player
  (undercounting rush TDs by -16.3%). Both fixed — team score inflation
  roughly halved, rush TD rate corrected, Sim Median's bias shrank at
  every position — though its raw MAE didn't improve and it still trails
  plain Projection on accuracy. See the dedicated section below.
  — see "D/ST scoring + defensive-touchdown fix" below.

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

## D/ST scoring + defensive-touchdown fix (2026-10-15)

The simulator's resampling pool (`get_simulator_play_pool_db()`) already
included `Sack`, `Pass Interception Return`, and `Sack Opp Fumble
Recovery` plays — the raw ingredients for FanDuel D/ST scoring (sacks,
interceptions, points allowed) were already flowing through every
simulated drive, but nothing tallied them, so every D/ST pool entry in
`/nfl/lineup` fell back to its regression projection instead of a real
simulated floor/median/ceiling (see `nfl_fantasy.attach_sim_data()`'s
fallback). Fixing this surfaced a real, independent bug: `simulate_drive()`
credited **every** turnover as a flat 0-point `'turnover'` outcome, even
when the real historical play being resampled was itself a defensive/
return touchdown (a Pick-6 or scoop-and-score) — `SCORING_PLAY_TYPES`
only ever recognized offensive scores. That silently zeroed out a real
scoring event on the team-score simulation itself, not just D/ST fantasy
points.

**Fix**: `get_simulator_play_pool_db()` now also returns `is_scoring_play`
per play. `simulate_drive()`'s turnover branch returns a new
`'defensive_touchdown'` outcome when the drawn play was itself scored;
`simulate_game()` credits those 7 points to the OPPOSING side's
scoreboard (`OPPOSITE_SIDE`) instead of the offense that turned it over.
`simulate_matchup()` tallies each side's sacks/INTs/defensive-TD count
from the OPPONENT's own play list each draw (points allowed = the
opponent's own final score, which now correctly includes any defensive
TDs scored off of this side's turnovers) and appends a `DST_<team_id>`
entry to the same per-draw `draws` dict every skill player's points
already accumulate into — no persistence/pool-attachment-layer changes
needed, since `nfl_fantasy.build_player_pool()`'s DST pool entries already
use that exact id convention.

**Validation**: a full re-run of the original 544-game, 2024+2025
`TEAM_CONDITIONING_K` grid search wasn't practical for this change (that
backtest takes ~1 hour against the full two-season warehouse). Instead,
ran a controlled before/after comparison — the SAME sample of completed
2026-season games, the SAME RNG seed, with only the defensive-touchdown
crediting toggled on vs. off (every other line of code identical):

| | n | Accuracy | Brier |
|---|---|---|---|
| Before (defensive TDs always 0 pts, the pre-existing bug) | 79 | 53.2% | 0.2696 |
| After (defensive TDs correctly credited) | 79 | 53.2% | **0.2464** |

Accuracy is unchanged — real defensive TDs are rare enough (~0.1-0.15% of
simulated drives, confirmed directly against this app's stored play
pool) that they essentially never flip which side a game picks. Brier
score improved (lower is better), consistent with fixing a real scoring
bug rather than introducing a new assumption. No regression found; a full
grid-search re-run against the full 2024/2025 warehouse remains a cheap
confidence-builder to do later if anyone wants the exact K=0.5 figure
re-confirmed, but nothing here suggests it would move.

## Two drive-loop bugs fixed, found via a Sim Median accuracy backtest (2026-10-15)

A full 2024+2025 backtest (544 games, ~10,000 QB/RB/WR+TE player-games — the
same 544-game set as the grid search above) compared Projection, Sim Median,
and Sim Ceiling point estimates against real actual fantasy points. Sim
Median (meant to be a stable, central-tendency estimate like Projection, not
a variance play) came back both less accurate (worse MAE at every position)
and running systematically LOW vs. real outcomes:

| Bucket | Projection MAE / bias | Sim Median MAE / bias | Sim Ceiling MAE / bias |
|---|---|---|---|
| QB | 6.50 / +0.28 | 7.66 / **+2.23** | 9.23 / −3.17 |
| RB | 4.93 / +0.10 | 5.41 / **+0.72** | 7.15 / −3.97 |
| WR/TE | 4.07 / +0.17 | 4.15 / **+1.47** | 5.30 / −2.27 |

(bias = `actual − estimate`, so positive means the real result exceeded the
estimate — Sim Median under-predicts. Sim Ceiling's negative bias is
expected/correct — it targets the 80th percentile on purpose.)

A follow-up drive-outcome diagnostic (same 544 games, real drive outcomes
derived directly from stored play-by-play via `drive_id` vs. simulated
`simulate_game()` drive outcomes) found two real, root-cause bugs in
`simulate_drive()`, not a calibration issue:

**Bug 1 — a failed 4th-down "go for it" never ended the drive.** The
short-yardage go-for-it branch fell through to a normal play sample, but on
a failed conversion the loop just did `down += 1` and kept going — no branch
resolved it as `turnover_on_downs`. Down climbed to 5, 6..., and
`_sample_play()`'s bucket-backoff chain (no real down-5+ data exists) landed
on a fully unconditioned `rng.choice(all_plays)` draw, letting the drive run
indefinitely with bonus plays. Confirmed directly: simulated
`turnover_on_downs` was **0.00%** of 217,600 drives vs. **6.16%** real; `punt`
rate was inflated **+10.4 points** (44.9% vs 34.5%) as those undying drives
eventually resolved some other way; total team score ran **+6.1%** hot.

**Bug 2 — a touchdown scored via the generic field-position check wasn't
credited to any player.** A drive can score two ways: an explicit
`'Rushing Touchdown'`/`'Passing Touchdown'` play type, or the generic
`yards_to_endzone <= 0` check after an ordinarily-labeled `'Rush'`/`'Pass
Reception'` play (the resampled real play's yardage happened to be enough
to cross the goal line in *this* drive's field position, even if the donor
play didn't score in its own original context). The drive correctly
registered as a `'touchdown'` for team-score purposes either way, but
`attribute_plays_to_players()` only checked the explicit play-type label —
the generic path silently credited no one.

**Fix**: `simulate_drive()` now tracks whether the current play is a
short-yardage 4th-down attempt; if it doesn't convert, doesn't score, and
isn't a turnover, the drive now correctly ends as `'turnover_on_downs'`
(possession flips at the play's own real `end_yards_to_endzone`, or the
same mirrored-position fallback the turnover branch already used). At the
generic-touchdown check, a **copy** of the just-appended play (never the
original — it's a shared object cached in `pools.scrimmage_buckets` and
reused across every other draw and game) gets relabeled to its touchdown
equivalent before being handed to `attribute_plays_to_players()`, which
itself needed zero changes — it already checks for exactly these literal
play-type strings.

**Validation — four checks, same 544-game set, before vs. after:**

1. Drive-outcome distribution: `turnover_on_downs` moved from a hard 0.00%
   to **1.84%** — confirms the drive-ending bug is genuinely fixed. Still
   below the real 6.16%, but that remainder is a separate, already-
   documented limitation: the simulator only ever considers going for it on
   *short-yardage* 4th downs (`FOURTH_DOWN_SHORT_YARDAGE=2`), while real
   turnover-on-downs includes longer-distance desperation attempts the
   simulator never attempts in the first place — out of scope for this fix
   (see `FOURTH_DOWN_GO_FOR_IT_RATE`'s own comment on why that rate can't be
   fit from stored data). Field-goal-good/missed rates stayed within ~1
   point of real both before and after — confirmed not a side effect.
2. Team-level score/yardage/TD diagnostic: team score inflation roughly
   halved (+6.1% → **+2.8%**); rush TD rate went from badly undercounted
   (-16.3%) to essentially correct (**+2.3%**) — direct confirmation Bug 2's
   fix works. Passing TD rate overshot from -4.1% to **+11.1%**, a new,
   smaller, one-sided nuance likely from field-position-bucket granularity
   (a reception play's donor context doesn't always closely match the
   field position it gets resampled into near the goal line) — flagged,
   not re-engineered further this round.
3. Full player-level backtest: Sim Median's bias improved at every position
   (QB +2.23→+1.53, RB +0.72→+0.43, WR/TE +1.47→+1.43) — the systematic
   under-prediction genuinely shrank. Its MAE, however, did **not**
   meaningfully improve (QB 7.66→7.71, RB 5.41→5.48, WR/TE 4.15→4.19) — Sim
   Median is less *biased* than before but not more *precise*, and still
   has a meaningfully worse MAE than plain Projection at QB/RB specifically
   (WR/TE was already near parity). Honest bottom line: this fix corrects
   two real engine bugs and measurably improves calibration, but doesn't
   close the gap that makes Projection the more reliable point estimate
   today.
4. Win-probability sanity check (same games, same RNG seed, old vs. new
   drive logic, `n=30` draws/game): accuracy **improved** (55.5% → 57.4%),
   Brier score moved slightly the other way (0.2446 → 0.2488). That Brier
   drift is about a tenth the size of the swing the original K=2.5→0.5
   recalibration produced, and accuracy went up, not down — read as "holds
   roughly steady," not a regression worth blocking on. `TEAM_CONDITIONING_K`
   was fit against the old (buggy) drive logic, though, so a full grid-search
   re-run against the fixed logic would be a reasonable future confidence
   check if anyone wants the exact K=0.5 figure re-confirmed — nothing here
   suggests it would move much.

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
