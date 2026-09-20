#!/usr/bin/env python3
"""
wnba_backfill.py — Populate game_predictions with completed WNBA regular-season
games so the Model Performance and History pages have real WNBA data, the same
way nba_backfill.py does for the NBA.

Usage:
    python3 wnba_backfill.py                    # 2022-2026 seasons (default)
    python3 wnba_backfill.py --dry-run
    python3 wnba_backfill.py --seasons 2025 2026

Every game is scored with wnba_model.predict() using a point-in-time snapshot —
each team's record, home/road split, PPG / PPG-allowed, last-5 form and rest
days computed ONLY from games that team played EARLIER in that same season
(records reset every year). ESPN's per-game "records" fields are deliberately
not used: they reflect the state AFTER the game and would leak the result.

A settled game's prob/factors_json is written ONCE, on first backfill, and is
then immutable — reruns never recompute it, only outcome fields (score,
home_won) refresh. Safe to re-run any time, including mid-season.
"""
import json
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from wnba_bootstrap import SEASONS, fetch_season_games, score_season  # noqa: E402
import wnba_model  # noqa: E402

DEFAULT_SEASONS = SEASONS


def run(dry_run=False, seasons=None, refetch_current=True):
    seasons = seasons or DEFAULT_SEASONS
    print('─' * 58)
    print(f'  WNBA Backfill — seasons {seasons}{"  (dry run — no writes)" if dry_run else ""}')
    print('─' * 58)

    current = datetime.now().year
    rows = []
    n_seasons = 0
    for season in seasons:
        print(f'\n── Season {season} ──────────────────────────────────────')
        cache_path = os.path.join(_HERE, f'wnba_training_{season}.json')
        # Past seasons are immutable, so the bootstrap cache is reused; the
        # in-progress season is always re-fetched so new results land.
        if os.path.exists(cache_path) and not (refetch_current and season >= current):
            with open(cache_path) as f:
                games = json.load(f)['games']
            print(f'  Loaded from cache: {len(games):,} games')
        else:
            games = fetch_season_games(season)
            if games:
                with open(cache_path, 'w') as f:
                    json.dump({'season': season, 'games': games}, f)
        if not games:
            print(f'  No completed games found for {season} yet.')
            continue
        n_seasons += 1
        rows.extend(score_season(games, wnba_model))

    if not rows:
        print('\nNo completed games found for any requested season. Nothing to do.')
        return
    print(f'\nScored {len(rows)} games across {n_seasons} season(s).')

    if dry_run:
        print(f'\nPreview (first 5 of {len(rows)} games):')
        for r in rows[:5]:
            act  = 'HOME' if r['home_won'] else 'AWAY'
            pred = 'HOME' if r['home_prob'] >= 0.5 else 'AWAY'
            ok   = '✓' if act == pred else '✗'
            print(f'  {r["game_date"]}  {r["away_team"][:18]:18} @ {r["home_team"][:18]:18}  '
                  f'model {r["home_prob"]*100:.0f}% home  '
                  f'actual={r["home_score"]}-{r["away_score"]}  {ok}')
        acc = sum(1 for r in rows if (r['home_prob'] >= 0.5) == bool(r['home_won'])) / len(rows) * 100
        print(f'\nOverall pick accuracy across all {len(rows)} scored games: {acc:.1f}%')
        print('Run without --dry-run to write new records to DB.')
        return

    # Same DISABLE_STARTUP_TASKS guard as the other backfills, so importing
    # app.py here doesn't start the scheduler or warm live caches.
    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, GamePrediction

    now = datetime.now(timezone.utc)
    inserted = updated = skipped_err = 0

    with flask_app.app_context():
        for r in rows:
            try:
                pred = GamePrediction.query.filter_by(
                    sport='WNBA',
                    game_date=r['game_date'],
                    home_team=r['home_team'],
                    away_team=r['away_team'],
                ).first()
                if pred is None:
                    pred = GamePrediction(
                        sport='WNBA',
                        game_date=r['game_date'],
                        home_team=r['home_team'],
                        away_team=r['away_team'],
                    )
                    db.session.add(pred)
                    inserted += 1
                    pred.game_time_utc = r.get('game_time_utc', '')
                    pred.home_prob    = r['home_prob']
                    pred.away_prob    = r['away_prob']
                    pred.factors_json = json.dumps(r['factors'])
                else:
                    updated += 1

                # Outcome fields are idempotent — safe to re-set every run.
                pred.home_won       = bool(r['home_won'])
                pred.home_score     = r['home_score']
                pred.away_score     = r['away_score']
                pred.outcome_set_at = now
            except Exception as e:
                skipped_err += 1
                print(f'  ! error on {r.get("game_date")} '
                      f'{r.get("away_team")} @ {r.get("home_team")}: {e}')

        db.session.commit()

        resolved = GamePrediction.query.filter(
            GamePrediction.sport == 'WNBA',
            GamePrediction.home_won.isnot(None),
            GamePrediction.home_prob.isnot(None),
        ).all()
        correct = sum(1 for p in resolved if (p.home_prob >= 0.5) == bool(p.home_won))
        total_resolved = len(resolved)

    acc = correct / total_resolved * 100 if total_resolved else 0
    print('\n  ✓ Done')
    print(f'    Inserted:  {inserted:>4}  new records (freshly scored, point-in-time)')
    print(f'    Touched:   {updated:>4}  existing records (outcome fields only)')
    if skipped_err:
        print(f'    Errors:    {skipped_err:>4}')
    print(f'    Pick accuracy across {total_resolved} resolved WNBA predictions: {acc:.1f}%')
    print(f'\n  Model Performance page now has {total_resolved} resolved WNBA predictions.')
    print('  Visit /model?sport=WNBA to see calibration and factor correlation.')


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    seasons = DEFAULT_SEASONS
    if '--seasons' in sys.argv:
        idx = sys.argv.index('--seasons')
        seasons = [int(s) for s in sys.argv[idx + 1:] if s.isdigit()]
    run(dry_run=dry_run, seasons=seasons)
