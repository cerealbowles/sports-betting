#!/usr/bin/env python3
"""
nhl_backfill.py — Populate game_predictions with completed NHL regular-season
games so the Model Performance and History pages have real NHL data, the
same way nfl_backfill.py/cfb_backfill.py do for NFL/CFB.

Usage:
    python3 nhl_backfill.py                       # last 3 seasons (default)
    python3 nhl_backfill.py --dry-run
    python3 nhl_backfill.py --seasons 20242025 20252026

Fetches each team's full season schedule from api-web.nhle.com
(/v1/club-schedule-season/{abbrev}/{season}), dedupes by game id, and scores
every completed regular-season (gameType 2) game with nhl_model.predict()
using a point-in-time snapshot — each team's win/loss/OT-loss record,
home/road split, goals-for/against per game, last-5 form, and rest days,
computed ONLY from games that team played EARLIER in that same season
(season records reset every year). Same "no leaked post-game records" rule
cfb_backfill.py/nfl_backfill.py document.

A settled game's prob/factors_json is written ONCE, on first backfill, and
is then immutable — reruns never recompute it, only outcome fields (score,
home_won) refresh. Safe to re-run any time.
"""
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import nhl_model

NHL_API = 'https://api-web.nhle.com/v1'
DEFAULT_SEASONS = ['20232024', '20242025', '20252026']

# Current 32 franchises, plus ARI (Arizona, relocated to Utah/UTA after the
# 2023-24 season) so the 2023-24 season's Coyotes games are still found —
# the schedule endpoint is keyed by the abbrev a team used THAT season.
TEAM_ABBREVS = [
    'ANA', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY', 'CHI', 'COL', 'DAL', 'DET',
    'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NJD', 'NSH', 'NYI', 'NYR', 'OTT',
    'PHI', 'PIT', 'SEA', 'SJS', 'STL', 'TBL', 'TOR', 'UTA', 'VAN', 'VGK',
    'WPG', 'WSH', 'ARI',
]


def _get(url, label, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries - 1:
                print(f'  ! {label} failed: {e}', flush=True)
                return None
            time.sleep(1 + attempt)
    return None


# ── Season data fetcher ──────────────────────────────────────────────────────

def fetch_season_games(season):
    """Fetch every completed regular-season game for one NHL season by
    requesting each team's full schedule and deduping by game id."""
    print(f'  Fetching {season} team schedules...', flush=True)
    by_id = {}
    for abbrev in TEAM_ABBREVS:
        r = _get(f'{NHL_API}/club-schedule-season/{abbrev}/{season}', f'{season} {abbrev}')
        if not r:
            continue
        for g in r.json().get('games', []):
            if g.get('gameType') != 2 or g.get('gameState') not in ('FINAL', 'OFF'):
                continue
            gid = g.get('id')
            if gid in by_id:
                continue
            home = g.get('homeTeam', {})
            away = g.get('awayTeam', {})
            h_name = home.get('placeName', {}).get('default', '') + ' ' + home.get('commonName', {}).get('default', '')
            a_name = away.get('placeName', {}).get('default', '') + ' ' + away.get('commonName', {}).get('default', '')
            h_score = home.get('score')
            a_score = away.get('score')
            if h_score is None or a_score is None:
                continue
            by_id[gid] = {
                'game_id':    gid,
                'game_date':  g.get('gameDate', ''),
                'home_name':  h_name.strip(),
                'away_name':  a_name.strip(),
                'home_abbrev': home.get('abbrev', ''),
                'away_abbrev': away.get('abbrev', ''),
                'home_score': h_score,
                'away_score': a_score,
                'home_won':   h_score > a_score,
                # OT losses are worth a standings point; overtime/shootout losses
                # for the losing side matter for points-pct — mark them here.
                'went_ot':    (g.get('periodDescriptor', {}) or {}).get('periodType') in ('OT', 'SO'),
            }
        time.sleep(0.1)
    games = list(by_id.values())
    print(f'  → {len(games)} unique completed games', flush=True)
    return games


# ── Point-in-time team stats ─────────────────────────────────────────────────

def build_pointintime_stats(season_games):
    """Walk one season's games chronologically, returning per-(team, date)
    win/loss/OT-loss record, home/road split, and goals-for/against-per-game,
    computed ONLY from that team's earlier games this season."""
    sorted_games = sorted(season_games, key=lambda g: (g['game_date'], g['game_id']))
    rec   = defaultdict(lambda: {'w': 0, 'l': 0, 'otl': 0})
    split = defaultdict(lambda: {'home': [0, 0], 'away': [0, 0]})
    pts   = defaultdict(lambda: {'for': 0, 'against': 0, 'g': 0})
    last_date = {}
    snap = {}

    for g in sorted_games:
        h, a, gd = g['home_name'], g['away_name'], g['game_date']
        for team, side in ((h, 'home'), (a, 'away')):
            r  = rec[team]
            sp = split[team][side]
            p  = pts[team]
            snap[(team, gd)] = {
                'wins':        r['w'],
                'losses':      r['l'],
                'ot_losses':   r['otl'],
                'split_w':     sp[0],
                'split_l':     sp[1],
                'gf_pg':       round(p['for'] / p['g'], 2) if p['g'] else None,
                'ga_pg':       round(p['against'] / p['g'], 2) if p['g'] else None,
                'rest_days':   (datetime.strptime(gd, '%Y-%m-%d').date()
                                 - datetime.strptime(last_date[team], '%Y-%m-%d').date()).days
                                if team in last_date else None,
            }

        # Fold this game's actual result into the accumulators so the NEXT
        # game either team plays sees it — this one never did.
        h_won, went_ot = g['home_won'], g['went_ot']
        if h_won:
            rec[h]['w'] += 1
            rec[a]['otl' if went_ot else 'l'] += 1
            split[h]['home'][0] += 1
            split[a]['away'][1] += 1
        else:
            rec[a]['w'] += 1
            rec[h]['otl' if went_ot else 'l'] += 1
            split[h]['home'][1] += 1
            split[a]['away'][0] += 1
        pts[h]['for'] += g['home_score']; pts[h]['against'] += g['away_score']; pts[h]['g'] += 1
        pts[a]['for'] += g['away_score']; pts[a]['against'] += g['home_score']; pts[a]['g'] += 1
        last_date[h] = gd
        last_date[a] = gd

    return snap


def build_recent_form(games, team_name, before_date, n=5):
    tg = []
    for g in games:
        if g['game_date'] >= before_date:
            continue
        if g['home_name'] == team_name:
            tg.append((g['game_date'], 'W' if g['home_won'] else 'L'))
        elif g['away_name'] == team_name:
            tg.append((g['game_date'], 'W' if not g['home_won'] else 'L'))
    tg.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in tg[:n]]


def build_team_dict(snapshot, form):
    return {
        'wins':        snapshot.get('wins', 0),
        'losses':      snapshot.get('losses', 0),
        'ot_losses':   snapshot.get('ot_losses', 0),
        'split_w':     snapshot.get('split_w', 0),
        'split_l':     snapshot.get('split_l', 0),
        'gf_pg':       snapshot.get('gf_pg'),
        'ga_pg':       snapshot.get('ga_pg'),
        'form':        form,
        'rest_days':   snapshot.get('rest_days'),
    }


# ── Run ───────────────────────────────────────────────────────────────────────

def run(dry_run=False, seasons=None):
    seasons = seasons or DEFAULT_SEASONS
    print('─' * 58)
    print(f'  NHL Backfill — seasons {seasons}{"  (dry run — no writes)" if dry_run else ""}')
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

    rows = []
    for season, games in all_games:
        pit_snap = build_pointintime_stats(games)
        for g in games:
            h, a, gd = g['home_name'], g['away_name'], g['game_date']
            home = build_team_dict(pit_snap.get((h, gd), {}), build_recent_form(games, h, gd))
            away = build_team_dict(pit_snap.get((a, gd), {}), build_recent_form(games, a, gd))
            result = nhl_model.predict(home, away)
            rows.append({
                'season':     season,
                'game_date':  gd,
                'home_name':  h,
                'away_name':  a,
                'home_score': g['home_score'],
                'away_score': g['away_score'],
                'home_won':   g['home_won'],
                'home_prob':  result['home_prob'],
                'away_prob':  result['away_prob'],
                'factors':    result['factors'],
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

    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, GamePrediction

    now = datetime.now(timezone.utc)
    inserted = updated = skipped_err = 0

    with flask_app.app_context():
        for r in rows:
            try:
                pred = GamePrediction.query.filter_by(
                    sport='NHL',
                    game_date=r['game_date'],
                    home_team=r['home_name'],
                    away_team=r['away_name'],
                ).first()
                is_new = pred is None

                if is_new:
                    pred = GamePrediction(
                        sport='NHL',
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
            GamePrediction.sport == 'NHL',
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
    print(f'    Pick accuracy across {total_resolved} resolved NHL predictions: {acc:.1f}%')
    print(f'\n  Model Performance page now has {total_resolved} resolved NHL predictions.')
    print(f'  Visit /model?sport=NHL to see calibration and factor correlation.')


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    seasons = DEFAULT_SEASONS
    if '--seasons' in sys.argv:
        idx = sys.argv.index('--seasons')
        seasons = [s for s in sys.argv[idx + 1:] if s.isdigit()]
    run(dry_run=dry_run, seasons=seasons)
