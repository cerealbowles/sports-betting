#!/usr/bin/env python3
"""
cfb_backfill.py — Populate game_predictions with completed FBS regular-season
games so the Model Performance and History pages have real CFB data, the
same way nfl_backfill.py does for the NFL.

Usage:
    python3 cfb_backfill.py                    # 2024, 2025, 2026 (default)
    python3 cfb_backfill.py --dry-run
    python3 cfb_backfill.py --seasons 2024 2025

Fetches completed regular-season games from ESPN (groups=80, FBS only) and,
for any game not already in game_predictions, scores it with
cfb_model.predict() using a point-in-time snapshot — each team's win/loss
record, home/road split, PPG, PPG-allowed, last-3 form, and rest days,
computed ONLY from games that team played EARLIER in that same season
(season records reset every year). AP/CFP rank is NOT accumulated this way —
ESPN attaches each competitor's rank as of that specific game already, so
it's read straight off the raw game row instead of walked chronologically.

This deliberately does NOT reuse the win/loss/split fields ESPN attaches to
each completed game's boxscore ("records") — those reflect the record AFTER
that game finishes, so using them directly would leak the very outcome
being predicted into the win-percentage and home/road-split factors. Same
class of bug nfl_backfill.py's docstring warns about, and the same one
cfb_bootstrap.py's season-aggregate stats have (fine for a rough bootstrap
gut-check, not for real per-game predictions here).

A settled game's prob/factors_json is written ONCE, on first backfill, and
is then immutable — reruns never recompute it, only outcome fields (score,
home_won) refresh. Safe to re-run any time, including mid-season: seasons
with no completed games yet just contribute zero rows. Bowl/postseason
games (seasontype != 2) are excluded — same reasoning cfb_api.py applies to
skip them from live prediction tracking (opt-outs/portal churn).
"""
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from cfb_bootstrap import fetch_season_games, build_recent_form, build_rest_days_map
import cfb_model

DEFAULT_SEASONS = [2024, 2025, 2026]


# ── Point-in-time team stats ─────────────────────────────────────────────────

def build_pointintime_stats(season_games):
    """
    Walk one season's games in chronological order, returning — for every
    (team, game_date) that team plays — win/loss record, home/road split,
    and PPG/PPG-allowed computed ONLY from that team's earlier games this
    season. The accumulators update AFTER each game's snapshot is captured,
    so a team's own game-in-progress is never included in its own stats.

    Returns {(team_name, game_date): {wins, losses, split_w, split_l, ppg,
    ppg_allowed}}. Assumes at most one game per team per date.
    """
    sorted_games = sorted(season_games, key=lambda g: g['game_date'])
    wl    = defaultdict(lambda: [0, 0])                                    # team -> [w, l]
    split = defaultdict(lambda: {'home': [0, 0], 'away': [0, 0]})          # team -> {'home'/'away': [w, l]}
    pts   = defaultdict(lambda: {'for': 0, 'against': 0, 'g': 0})          # team -> scoring totals
    snap  = {}

    for g in sorted_games:
        h, a = g['home_name'], g['away_name']
        for team, side in ((h, 'home'), (a, 'away')):
            w, l = wl[team]
            sp = split[team][side]
            p  = pts[team]
            snap[(team, g['game_date'])] = {
                'wins':         w,
                'losses':       l,
                'split_w':      sp[0],
                'split_l':      sp[1],
                'ppg':          round(p['for'] / p['g'], 2) if p['g'] else None,
                'ppg_allowed':  round(p['against'] / p['g'], 2) if p['g'] else None,
            }

        # Now fold this game's actual result into the accumulators, so the
        # NEXT game either team plays sees it — but this one never did.
        h_won = g['home_won']
        wl[h][0 if h_won else 1] += 1
        wl[a][1 if h_won else 0] += 1
        split[h]['home'][0 if h_won else 1] += 1
        split[a]['away'][1 if h_won else 0] += 1
        pts[h]['for'] += g['home_score']; pts[h]['against'] += g['away_score']; pts[h]['g'] += 1
        pts[a]['for'] += g['away_score']; pts[a]['against'] += g['home_score']; pts[a]['g'] += 1

    return snap


def build_team_dict(snapshot, form, rest_days, rank):
    """snapshot is one (team, game_date) entry from build_pointintime_stats,
    or {} if this is that team's first game of the season."""
    return {
        'wins':        snapshot.get('wins', 0),
        'losses':      snapshot.get('losses', 0),
        'split_w':     snapshot.get('split_w', 0),
        'split_l':     snapshot.get('split_l', 0),
        'ppg':         snapshot.get('ppg'),
        'ppg_allowed': snapshot.get('ppg_allowed'),
        'form':        form,
        'rest_days':   rest_days,
        'rank':        rank,
    }


# ── Run ───────────────────────────────────────────────────────────────────────

def run(dry_run=False, seasons=None):
    seasons = seasons or DEFAULT_SEASONS
    print('─' * 58)
    print(f'  CFB Backfill — seasons {seasons}{"  (dry run — no writes)" if dry_run else ""}')
    print('─' * 58)

    all_games = []
    for season in seasons:
        print(f'\n── Season {season} ──────────────────────────────────────')
        games = fetch_season_games(season)
        if not games:
            print(f'  No completed games found for {season} yet.')
            continue
        all_games.append((season, games))

    if not all_games:
        print('\nNo completed games found for any requested season. Nothing to do.')
        return

    # Build scored rows per season (records reset each season, so point-in-time
    # stats must never mix across seasons).
    rows = []
    for season, games in all_games:
        pit_snap = build_pointintime_stats(games)
        rest_map = build_rest_days_map(games)   # once per season, not per game
        for g in games:
            h, a, gd = g['home_name'], g['away_name'], g['game_date']
            home = build_team_dict(
                pit_snap.get((h, gd), {}),
                build_recent_form(games, h, gd),
                rest_map[h].get(gd),
                g['home_rank'],
            )
            away = build_team_dict(
                pit_snap.get((a, gd), {}),
                build_recent_form(games, a, gd),
                rest_map[a].get(gd),
                g['away_rank'],
            )
            result = cfb_model.predict(home, away)
            rows.append({
                'season':       season,
                'game_date':    gd,
                'home_name':    h,
                'away_name':    a,
                'home_score':   g['home_score'],
                'away_score':   g['away_score'],
                'home_won':     g['home_won'],
                'home_prob':    result['home_prob'],
                'away_prob':    result['away_prob'],
                'factors':      result['factors'],
            })

    print(f'\nScored {len(rows)} games across {len(all_games)} season(s).')

    if dry_run:
        print(f'\nPreview (first 5 of {len(rows)} games):')
        for r in rows[:5]:
            act  = 'HOME' if r['home_won'] else 'AWAY'
            pred = 'HOME' if r['home_prob'] >= 0.5 else 'AWAY'
            ok   = '✓' if act == pred else '✗'
            print(f'  {r["game_date"]}  {r["away_name"][:18]:18} @ {r["home_name"][:18]:18}  '
                  f'model {r["home_prob"]*100:.0f}% home  '
                  f'actual={r["home_score"]}-{r["away_score"]}  {ok}')
        acc = sum(1 for r in rows if (r['home_prob'] >= 0.5) == r['home_won']) / len(rows) * 100
        print(f'\nOverall pick accuracy across all {len(rows)} scored games: {acc:.1f}%')
        print('Run without --dry-run to write new records to DB.')
        return

    # Write to DB — same DISABLE_STARTUP_TASKS guard as backfill_2026.py/nfl_backfill.py,
    # so importing app.py here doesn't start the scheduler or warm live caches.
    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, GamePrediction

    now = datetime.now(timezone.utc)
    inserted = updated = skipped_err = 0

    with flask_app.app_context():
        for r in rows:
            try:
                pred = GamePrediction.query.filter_by(
                    sport='CFB',
                    game_date=r['game_date'],
                    home_team=r['home_name'],
                    away_team=r['away_name'],
                ).first()
                is_new = pred is None

                if is_new:
                    pred = GamePrediction(
                        sport='CFB',
                        game_date=r['game_date'],
                        home_team=r['home_name'],
                        away_team=r['away_name'],
                    )
                    db.session.add(pred)
                    inserted += 1
                    pred.home_prob    = r['home_prob']
                    pred.away_prob    = r['away_prob']
                    pred.factors_json = json.dumps(r['factors'])
                else:
                    updated += 1

                # Outcome fields are idempotent — safe to re-set every run.
                pred.home_won       = r['home_won']
                pred.home_score     = r['home_score']
                pred.away_score     = r['away_score']
                pred.outcome_set_at = now

            except Exception as e:
                skipped_err += 1
                print(f'  ! error on {r.get("game_date")} '
                      f'{r.get("away_name")} @ {r.get("home_name")}: {e}')

        db.session.commit()

        resolved = GamePrediction.query.filter(
            GamePrediction.sport == 'CFB',
            GamePrediction.home_won.isnot(None),
            GamePrediction.home_prob.isnot(None),
        ).all()
        correct = sum(1 for p in resolved if (p.home_prob >= 0.5) == bool(p.home_won))
        total_resolved = len(resolved)

    acc = correct / total_resolved * 100 if total_resolved else 0

    print(f'\n  ✓ Done')
    print(f'    Inserted:  {inserted:>4}  new records (freshly scored, point-in-time)')
    print(f'    Touched:   {updated:>4}  existing records (outcome fields only)')
    if skipped_err:
        print(f'    Errors:    {skipped_err:>4}')
    print(f'    Pick accuracy across {total_resolved} resolved CFB predictions: {acc:.1f}%')
    print(f'\n  Model Performance page now has {total_resolved} resolved CFB predictions.')
    print(f'  Visit /model?sport=CFB to see calibration and factor correlation.')


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    seasons = DEFAULT_SEASONS
    if '--seasons' in sys.argv:
        idx = sys.argv.index('--seasons')
        seasons = [int(s) for s in sys.argv[idx + 1:] if s.isdigit()]
    run(dry_run=dry_run, seasons=seasons)
