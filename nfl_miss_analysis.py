#!/usr/bin/env python3
"""
nfl_miss_analysis.py — Phase 3: analyze where project_points() misses
(NFL_Fantasy_Analytics_Projection_Plan.md, section 5), using the Phase 2
dataset (nfl_playergame_dataset.py).

Usage:
    python3 nfl_miss_analysis.py nfl_playergame_dataset.jsonl

GUARDRAIL (the plan's own words, section 5): "This phase generates
hypotheses, not proof. Any apparent pattern must be tested on later games
that were not used to discover it." That's not decoration here — every
candidate signal below is measured TWICE: once on a discovery/train
split (seasons 2024+2025) and once on a holdout split (season 2026) it
was never fit or thresholded against, via season_split() (already in
nfl_playergame_dataset.py — same chronological-holdout convention this
app's other *_bootstrap.py scripts use). A finding is only reported as
real if its direction replicates on the holdout; a train-only effect is
reported as NOT REPLICATED, not quietly dropped, so a reader can see what
was tried and rejected (same "track the experiment, not just the win"
spirit as the plan's own section 9 experiment log).

MISS DEFINITION: a "big miss" is a projection_error whose z-score
(relative to that BUCKET's own mean/std, fit on the TRAIN split only —
never the holdout, never the full pooled set) exceeds +/-1.5. Ordinary
rows (|z| <= 1.5) are the explicit CONTROL group every comparison below
is measured against — the plan's own "include ordinary outcomes as
controls; do not analyze only breakouts."

KNOWN LIMITATIONS of this pass (stated, not hidden):
  - "Salary tier" isn't in the dataset (Phase 2 explicitly didn't
    reconstruct historical salary — see that script's docstring).
    Projection quintile is used as the closest available proxy; it's
    correlated with salary (assign_salaries ranks by projection
    percentile) but isn't the real thing.
  - "Explosive plays" as a miss driver can't be measured directly — the
    Phase 2 dataset's component_outcomes are game-total yards/TDs, not
    play-level detail, so there's no per-row "did THIS player have an
    explosive play" column. Yards-per-opportunity is used as an indirect
    proxy where noted, not a real substitute.
  - Everything here is pregame-feature vs. outcome correlation on stored
    history, not a causal claim — same caveat the plan's own "insights
    explain; backtests validate" decision rule implies.
"""
import json
import sys

import nfl_playergame_dataset as ds

HOLDOUT_SEASON = 2026
MISS_Z = 1.5


def _per_game(total, games):
    return (total / games) if games else None


def _volume_per_game(usage_side, bucket):
    """A single 'touches per game' number appropriate to the bucket —
    passing attempts for QB, rush+target touches for RB, targets for
    REC. None if the relevant category is missing (e.g. a QB with no
    rushing_splits that window)."""
    if usage_side is None:
        return None
    games = usage_side['games']
    if bucket == 'QB':
        p = usage_side.get('passing')
        return _per_game(p['pass_att'], games) if p else None
    if bucket == 'RB':
        r, c = usage_side.get('rushing'), usage_side.get('receiving')
        total = (r['rush_att'] if r else 0) + (c['targets'] if c else 0)
        return _per_game(total, games) if (r or c) else None
    # REC
    c = usage_side.get('receiving')
    return _per_game(c['targets'], games) if c else None


def _td_rate(usage_side, bucket):
    """TDs per opportunity for the bucket's primary touch type — the
    signal behind the plan's own candidate question 'did a player's
    projection depend on unusually high recent efficiency or TD
    conversion?'. None if there's no usable denominator."""
    if usage_side is None:
        return None
    if bucket == 'QB':
        p = usage_side.get('passing')
        return (p['pass_td'] / p['pass_att']) if p and p['pass_att'] else None
    if bucket == 'RB':
        r, c = usage_side.get('rushing'), usage_side.get('receiving')
        touches = (r['rush_att'] if r else 0) + (c['targets'] if c else 0)
        tds = (r['rush_td'] if r else 0) + (c['rec_td'] if c else 0)
        return (tds / touches) if touches else None
    c = usage_side.get('receiving')
    return (c['rec_td'] / c['targets']) if c and c['targets'] else None


def _opponent_allowed_per_game(side, bucket):
    """Opponent's own yards+TDs allowed to this bucket, per game — the
    'matchup affected a specific component' candidate, distinct from the
    single aggregate opponent_defense_grade the model actually uses."""
    if side is None:
        return None
    games = side['games']
    b = side.get(bucket, {})
    yards_key = 'pass_yds' if bucket == 'QB' else ('rush_yds' if bucket == 'RB' else 'rec_yds')
    return _per_game(b.get(yards_key, 0), games) if games else None


def enrich(row):
    """Adds the candidate-signal columns to one dataset row, in place.
    Pure derived values from columns the dataset already has — no new DB
    reads, no leakage risk beyond what Phase 2 already guarantees."""
    bucket = row['bucket']
    row['_recent_volume_pg'] = _volume_per_game(row['recent_usage'], bucket)
    row['_season_volume_pg'] = _volume_per_game(row['season_usage'], bucket)
    row['_volume_trend'] = (
        row['_recent_volume_pg'] - row['_season_volume_pg']
        if row['_recent_volume_pg'] is not None and row['_season_volume_pg'] is not None else None
    )
    row['_recent_td_rate'] = _td_rate(row['recent_usage'], bucket)
    row['_season_td_rate'] = _td_rate(row['season_usage'], bucket)
    row['_td_rate_trend'] = (
        row['_recent_td_rate'] - row['_season_td_rate']
        if row['_recent_td_rate'] is not None and row['_season_td_rate'] is not None else None
    )
    row['_opp_recent_allowed_pg'] = _opponent_allowed_per_game(row['opponent_defense_recent'], bucket)
    row['_opp_season_allowed_pg'] = _opponent_allowed_per_game(row['opponent_defense_season'], bucket)
    row['_opp_matchup_trend'] = (
        row['_opp_recent_allowed_pg'] - row['_opp_season_allowed_pg']
        if row['_opp_recent_allowed_pg'] is not None and row['_opp_season_allowed_pg'] is not None else None
    )
    row['_thin_sample'] = row['recent_games_played'] < 3
    return row


def fit_miss_thresholds(train_rows):
    """Per-bucket mean/std of projection_error, fit on TRAIN rows only —
    the dividing line between 'big miss' and 'control' is drawn from data
    the holdout comparison never sees, exactly like a real model
    threshold would be."""
    by_bucket = {}
    for bucket in ('QB', 'RB', 'REC'):
        errors = [r['projection_error'] for r in train_rows
                  if r['bucket'] == bucket and r['projection_error'] is not None]
        if len(errors) < 10:
            continue
        mean = sum(errors) / len(errors)
        var = sum((e - mean) ** 2 for e in errors) / len(errors)
        by_bucket[bucket] = {'mean': mean, 'std': var ** 0.5, 'n': len(errors)}
    return by_bucket


def classify_miss(row, thresholds):
    """'big_positive' | 'big_negative' | 'control' | None (no projection,
    or no fitted threshold for this bucket — excluded from miss analysis
    entirely, not silently folded into 'control')."""
    t = thresholds.get(row['bucket'])
    if t is None or row['projection_error'] is None or t['std'] == 0:
        return None
    z = (row['projection_error'] - t['mean']) / t['std']
    if z >= MISS_Z:
        return 'big_positive'
    if z <= -MISS_Z:
        return 'big_negative'
    return 'control'


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def compare_groups(rows, feature_key, group_key='_miss_class'):
    """Mean of `feature_key` for each miss class present in `rows` — the
    raw comparison every hypothesis below is built from."""
    out = {}
    for cls in ('big_positive', 'control', 'big_negative'):
        vals = [r[feature_key] for r in rows if r.get(group_key) == cls]
        out[cls] = {'mean': _mean(vals), 'n': sum(1 for r in rows if r.get(group_key) == cls and r[feature_key] is not None)}
    return out


def report_hypothesis(name, hypothesis, feature_key, train_rows, holdout_rows):
    """Prints one experiment-log entry (plan section 9's format) and
    returns whether the train-set direction replicated on holdout."""
    train_cmp = compare_groups(train_rows, feature_key)
    holdout_cmp = compare_groups(holdout_rows, feature_key)

    print(f'\n--- {name} ---')
    print(f'Hypothesis: {hypothesis}')
    print(f'Feature: {feature_key}')

    def _fmt(cmp):
        parts = []
        for cls in ('big_negative', 'control', 'big_positive'):
            m, n = cmp[cls]['mean'], cmp[cls]['n']
            parts.append(f"{cls}={m:.3f}(n={n})" if m is not None else f"{cls}=n/a(n={n})")
        return ' | '.join(parts)

    print(f'Train  (2024-2025): {_fmt(train_cmp)}')
    print(f'Holdout (2026):     {_fmt(holdout_cmp)}')

    train_pos, train_neg = train_cmp['big_positive']['mean'], train_cmp['big_negative']['mean']
    hold_pos, hold_neg = holdout_cmp['big_positive']['mean'], holdout_cmp['big_negative']['mean']
    control_train, control_hold = train_cmp['control']['mean'], holdout_cmp['control']['mean']

    if None in (train_pos, train_neg, hold_pos, hold_neg, control_train, control_hold):
        print('Decision: INSUFFICIENT DATA (a group had no usable rows on train or holdout)')
        return None

    train_direction = train_pos > control_train > train_neg or train_pos < control_train < train_neg
    train_pos_higher = train_pos > train_neg
    hold_pos_higher = hold_pos > hold_neg
    replicated = train_pos_higher == hold_pos_higher

    print(f'Decision: {"REPLICATED" if replicated else "NOT REPLICATED"} — '
          f'train direction (pos {"higher" if train_pos_higher else "lower"} than neg) '
          f'{"matches" if replicated else "does NOT match"} holdout '
          f'(pos {"higher" if hold_pos_higher else "lower"} than neg)')
    return replicated


def run(path):
    rows = ds.load_jsonl(path)
    print(f'{len(rows)} rows loaded from {path}')
    for r in rows:
        enrich(r)

    train_rows, holdout_rows = ds.season_split(rows, HOLDOUT_SEASON)
    print(f'train (seasons < {HOLDOUT_SEASON}): {len(train_rows)} rows, '
          f'holdout (season {HOLDOUT_SEASON}): {len(holdout_rows)} rows')

    thresholds = fit_miss_thresholds(train_rows)
    print('\nPer-bucket projection_error mean/std (fit on train only):')
    for bucket, t in thresholds.items():
        print(f"  {bucket}: mean={t['mean']:.2f} std={t['std']:.2f} (n={t['n']})")

    for r in rows:
        r['_miss_class'] = classify_miss(r, thresholds)

    for split_name, split_rows in (('train', train_rows), ('holdout', holdout_rows)):
        counts = {}
        for r in split_rows:
            cls = r.get('_miss_class')
            if cls:
                counts[cls] = counts.get(cls, 0) + 1
        print(f'\n{split_name} miss distribution: {counts}')

        print(f'{split_name} miss rate by bucket:')
        for bucket in ('QB', 'RB', 'REC'):
            bucket_rows = [r for r in split_rows if r['bucket'] == bucket and r.get('_miss_class')]
            if not bucket_rows:
                continue
            n = len(bucket_rows)
            pos = sum(1 for r in bucket_rows if r['_miss_class'] == 'big_positive')
            neg = sum(1 for r in bucket_rows if r['_miss_class'] == 'big_negative')
            print(f'  {bucket}: {pos}/{n} big-positive ({pos/n*100:.1f}%), {neg}/{n} big-negative ({neg/n*100:.1f}%)')

        print(f'{split_name} miss rate by projection quintile (proxy for salary tier — see docstring):')
        projected_rows = [r for r in split_rows if r['projection'] is not None and r.get('_miss_class')]
        projected_rows.sort(key=lambda r: r['projection'])
        n = len(projected_rows)
        if n >= 50:
            for q in range(5):
                lo, hi = n * q // 5, n * (q + 1) // 5
                bucket_rows = projected_rows[lo:hi]
                pos = sum(1 for r in bucket_rows if r['_miss_class'] == 'big_positive')
                neg = sum(1 for r in bucket_rows if r['_miss_class'] == 'big_negative')
                proj_range = f"{bucket_rows[0]['projection']:.1f}-{bucket_rows[-1]['projection']:.1f}"
                print(f'  Q{q+1} (proj {proj_range}, n={len(bucket_rows)}): '
                      f'{pos} big-positive ({pos/len(bucket_rows)*100:.1f}%), '
                      f'{neg} big-negative ({neg/len(bucket_rows)*100:.1f}%)')

    thin_rows_train = [r for r in train_rows if r.get('_miss_class')]
    thin_errors = [abs(r['projection_error']) for r in thin_rows_train if r['_thin_sample']]
    normal_errors = [abs(r['projection_error']) for r in thin_rows_train if not r['_thin_sample']]
    if thin_errors and normal_errors:
        print(f"\nThin-sample (< 3 recent games) rows: mean |error| = {_mean(thin_errors):.2f} (n={len(thin_errors)})")
        print(f"Normal-sample rows:                  mean |error| = {_mean(normal_errors):.2f} (n={len(normal_errors)})")

    results = {}
    for name, hyp, key in [
        ('Role-change / volume trend',
         "A rising recent-vs-season volume trend (more touches lately than the season average) "
         "predicts a positive miss — the model's single rolling-average-points feature doesn't "
         "separate 'more opportunity' from 'same opportunity, lucky outcome'.",
         '_volume_trend'),
        ('TD-rate regression',
         "An unusually high recent TD rate (vs. season) predicts a NEGATIVE miss next game — "
         "TDs are high-variance and regress toward the season rate, but the model's rolling "
         "fantasy-points average bakes the hot streak straight into the projection.",
         '_td_rate_trend'),
        ('Matchup component not captured by the aggregate grade',
         "A recent (vs. season) rise in what the opponent allows to this specific bucket "
         "predicts a positive miss, even though the model's opponent_defense_grade is one "
         "aggregate number across sacks/TFL/takeaways/pass-defended, not bucket-specific.",
         '_opp_matchup_trend'),
    ]:
        replicated = report_hypothesis(name, hyp, key, train_rows, holdout_rows)
        results[name] = replicated

    print('\n=== SUMMARY ===')
    for name, replicated in results.items():
        label = 'REPLICATED' if replicated else ('NOT REPLICATED' if replicated is False else 'INSUFFICIENT DATA')
        print(f'  {name}: {label}')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'nfl_playergame_dataset.jsonl'
    run(path)
