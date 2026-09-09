#!/usr/bin/env python3
"""
backfill_2026.py — Populate game_predictions with completed 2026 MLB games.

Usage:
    python3 backfill_2026.py            # insert any completed 2026 games not yet recorded
    python3 backfill_2026.py --dry-run  # print what would be saved, don't write

Fetches completed 2026 games from MLB Stats API and, for any game not already
in game_predictions, runs the model on that game's point-in-time stats
snapshot — team pitching/batting (MLB Stats API byDateRange), xFIP/SIERA/
K%/BB%/SwStr% (FanGraphs custom date range), xwOBA (Savant, aggregated
day-by-day via xwoba_rolling.py), and bullpen fatigue (last-3-days relief IP,
reconstructed from box scores via bullpen_fatigue.py) all cumulative through
the day BEFORE the game, never including data from the game itself or from
later in the season — and writes the prediction + actual outcome.

A settled game's prob/factors_json is written ONCE, the first time it's
backfilled, and is then immutable — reruns never recompute it. Only the
outcome fields (score, home_won, pick_roi) are refreshed on every run, which
is always safe since they're just facts about what already happened.

This is deliberate, not an oversight: a completed game's "correct" prediction
is whatever was knowable as of its game date. Recomputing it later — with
today's full-season stats, today's model weights, or today's team-bias
correction — would score the game with information that didn't exist yet,
and would make the stored prediction a moving target that changes depending
on when you happened to rerun this script, rather than reflecting anything
that happened by that game's date. Model-weight changes are validated
going forward on new games, not by rewriting history.

Safe to re-run any time — already-recorded games are left untouched (aside
from outcome fields), so reruns only ever do work proportional to how many
new games have completed since the last run.
"""

import json
import os
import sys
import time
from datetime import datetime, date, timezone

# ── Bootstrap fetchers ─────────────────────────────────────────────────────────
# Reuse schedule/stats fetchers from bootstrap_2024.py
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from bootstrap_2024 import (
    fetch_team_pitching, fetch_team_batting,
    fetch_team_xwoba, fetch_team_abbrevs,
    fetch_team_xfip,
    build_fatigue_map,
    SEASON_DATES, MLB_API, HEADERS, _get,
)
import bullpen_fatigue
import mlb_model

# MLB Stats API abbreviation -> FanGraphs abbreviation, for the 7 teams where
# they differ. Without this, fetch_team_xfip()'s lookup silently drops these
# teams (key never matches), losing xFIP/SIERA/K%/BB%/Whiff% for any game
# involving them — same class of bug as the Savant team-aggregation issue.
_FG_ABBREV_ALIAS = {
    'sd': 'sdp', 'sf': 'sfg', 'tb': 'tbr', 'cws': 'chw',
    'az': 'ari', 'kc': 'kcr', 'wsh': 'wsn',
}


def fetch_completed_games(season=2026):
    """Fetch all Final games for the season up to today."""
    start, _ = SEASON_DATES[season]
    end = date.today().isoformat()
    print(f'Fetching {season} completed games ({start} → {end})...', flush=True)

    from bootstrap_2024 import _month_ranges
    games = []
    for chunk_start, chunk_end in _month_ranges(start, end):
        r = _get(MLB_API + '/schedule', {
            'sportId': 1, 'gameType': 'R', 'season': season,
            'startDate': chunk_start, 'endDate': chunk_end,
            'hydrate': 'linescore,teams',
        }, f'schedule {chunk_start}')
        if not r:
            continue
        for day in r.json().get('dates', []):
            for g in day.get('games', []):
                if g.get('status', {}).get('abstractGameState', '') != 'Final':
                    continue
                h = g.get('teams', {}).get('home', {})
                a = g.get('teams', {}).get('away', {})
                h_score = h.get('score')
                a_score = a.get('score')
                if h_score is None or a_score is None:
                    continue
                try:
                    h_score = int(h_score)
                    a_score = int(a_score)
                except (TypeError, ValueError):
                    continue
                games.append({
                    'game_date':  day['date'],
                    'game_time':  g.get('gameDate', ''),
                    'game_pk':    g.get('gamePk'),
                    'home_id':    h['team']['id'],
                    'home_name':  h['team'].get('name', ''),
                    'away_id':    a['team']['id'],
                    'away_name':  a['team'].get('name', ''),
                    'home_score': h_score,
                    'away_score': a_score,
                    'home_won':   h_score > a_score,
                })
        time.sleep(0.15)

    print(f'  → {len(games)} completed games', flush=True)
    return games


def build_team_dict(team_id, team_name, team_pitch, team_bat, xwoba_by_id, fatigue=None,
                    team_xfip=None, bullpen_fatigue_ip=None):
    """
    Build the team context dict passed to mlb_model.predict().

    team_name must match the keys in mlb_model._TEAM_BIAS (full MLB team name,
    e.g. 'Cleveland Guardians') — without it, the Hist Adj factor is silently
    never applied (home.get('name', '') never matches a _TEAM_BIAS key).

    team_pitch / team_bat / xwoba_by_id / team_xfip / bullpen_fatigue_ip are
    ALL point-in-time (as-of the game's date) — see _snapshot_for_date() in
    run() below — NOT full-season aggregates. Every factor in the model now
    uses only data available through the day before this game.

    team_xfip – {team_id: {'xfip', 'siera', 'k_pct', 'bb_pct', 'swstr_pct'}}
                from fetch_team_xfip (FanGraphs, keyed by team_id after mapping
                through abbrevs). K%/BB%/SwStr% come from here rather than
                Savant's team leaderboard — that endpoint is player-level only
                (no team column), so team-level aggregation from it always
                silently returned {}. Team-level Barrel% has no reliable
                source and is intentionally omitted (None).
    bullpen_fatigue_ip – relief innings pitched by this team in its last 3
                games strictly before this game's date (bullpen_fatigue.py,
                box-score reconstructed). None if no games in that window.
                Matches the live daily-pick flow's team['bullpen']['ip_last_3']
                field name/shape (mlb_api.py._compute_bullpen_stats) so
                mlb_model.py reads one consistent key from both paths.
    Defaults to None → gracefully missing (None fields in pitcher dict).
    """
    p   = team_pitch.get(team_id, {})
    b   = team_bat.get(team_id, {})
    xfp = (team_xfip or {}).get(team_id, {})
    return {
        'id':               team_id,
        'name':             team_name,
        'form':             [],
        'split_w':          0,
        'split_l':          0,
        'pitcher': {
            'era':         p.get('era'),
            'x_era':       None,
            'xfip':        xfp.get('xfip'),
            'siera':       xfp.get('siera'),
            'whip':        p.get('whip'),
            'k':           p.get('k', 0),
            'bb':          p.get('bb', 0),
            'bf':          p.get('bf', 0),
            'k_pct':       xfp.get('k_pct'),
            'bb_pct':      xfp.get('bb_pct'),
            'barrel_pct':  None,
            'whiff_pct':   xfp.get('swstr_pct'),
            'last_starts': [],
        },
        'x_woba':           xwoba_by_id.get(team_id),
        'ops':              b.get('ops'),
        'runs_pg':          b.get('runs_pg'),
        'k_pct':            b.get('k_pct'),
        'schedule_fatigue': fatigue,
        'bullpen':          {'ip_last_3': bullpen_fatigue_ip},
    }


def run(dry_run=False, season=2026):
    print('─' * 58)
    print(f'  2026 Season Backfill{"  (dry run — no writes)" if dry_run else ""}')
    print('─' * 58)

    games        = fetch_completed_games(season)
    team_abbrevs = fetch_team_abbrevs()

    if not games:
        print('No completed games found. Check API connectivity.')
        return

    fatigue_map = build_fatigue_map(games)

    # Bullpen fatigue (box-score reconstructed, see bullpen_fatigue.py):
    # ensure every completed game's boxscore is cached before doing any
    # per-date lookups below. Incremental — only fetches gamePks not already
    # cached, so on a warm cache this is ~1355 dict-key checks and zero
    # network calls; only a genuinely new day of games costs real requests.
    bullpen_cache = bullpen_fatigue.ensure_games_cached(season, games)

    # Every factor's stats snapshot is now point-in-time (through the day
    # before the game), fetched lazily and cached once per distinct
    # game_date — so a rerun that only has 1 new day of games only pays for
    # that 1 day's worth of API calls, not a full-season refetch. Never
    # touched at all for a date whose games are already recorded, since
    # those are skipped before this cache is ever built (see `is_new` gate
    # in the write loop below).
    #
    # xwOBA (Savant) is the one exception with real per-call cost: its
    # rolling cache (xwoba_rolling.py) walks day-by-day from season start
    # the first time it's asked about a date past what's already cached, to
    # stay under Savant's ~25k-row-per-request cap. That's a one-time cost
    # per season (amortized across every date after the first), not a
    # per-game or per-run cost — see that module's docstring.
    _snapshot_cache = {}
    _degraded_dates = set()

    def _snapshot_for_date(gdate):
        if gdate not in _snapshot_cache:
            team_pitch = fetch_team_pitching(season, as_of_date=gdate)
            team_bat   = fetch_team_batting(season, as_of_date=gdate)
            team_xwoba = fetch_team_xwoba(season, as_of_date=gdate)
            xfip_abbr  = fetch_team_xfip(season, as_of_date=gdate)

            xwoba_by_id = {tid: team_xwoba[abbrev]
                           for tid, abbrev in team_abbrevs.items()
                           if abbrev in team_xwoba}

            # Map abbreviation-keyed dict → team_id-keyed. Translate through
            # _FG_ABBREV_ALIAS for the 7 teams whose FanGraphs abbrev differs
            # from the MLB Stats API one.
            xfip_by_id = {}
            for tid, abbrev in team_abbrevs.items():
                fg_abbrev = _FG_ABBREV_ALIAS.get(abbrev, abbrev)
                if fg_abbrev in xfip_abbr:
                    xfip_by_id[tid] = xfip_abbr[fg_abbrev]

            if not xfip_abbr:
                _degraded_dates.add(gdate)

            # Bullpen fatigue: last-3-days relief IP per team, as of this
            # date (games strictly before it only — point-in-time). Cheap —
            # bullpen_cache is already fully populated by ensure_games_cached
            # above, this is just a scan/sum, no network calls.
            bp_fatigue_by_id = {
                tid: bullpen_fatigue.get_bullpen_fatigue_asof(bullpen_cache, tid, gdate)
                for tid in team_abbrevs
            }

            _snapshot_cache[gdate] = (team_pitch, team_bat, xwoba_by_id, xfip_by_id,
                                       bp_fatigue_by_id)
            time.sleep(0.15)
        return _snapshot_cache[gdate]

    def _team(game, side):
        tid  = game[f'{side}_id']
        name = game[f'{side}_name']
        fat_key = (game['game_date'], game['home_id'], game['away_id'])
        fat = fatigue_map.get(fat_key, {}).get(side)
        team_pitch, team_bat, xwoba_by_id, xfip_by_id, bp_fatigue_by_id = \
            _snapshot_for_date(game['game_date'])
        return build_team_dict(tid, name, team_pitch, team_bat, xwoba_by_id, fat,
                               xfip_by_id, bullpen_fatigue_ip=bp_fatigue_by_id.get(tid))

    if dry_run:
        # Just preview the first 5 predictions, using each game's real
        # point-in-time snapshot (stats through the day before that game).
        print(f'\nPreview (first 5 of {len(games)} games, point-in-time stats):')
        for g in games[:5]:
            home = _team(g, 'home')
            away = _team(g, 'away')
            r    = mlb_model.predict(home, away, game_time_utc=g['game_time'])
            act  = 'HOME' if g['home_won'] else 'AWAY'
            pred = 'HOME' if r['home_prob'] >= 0.5 else 'AWAY'
            ok   = '✓' if act == pred else '✗'
            print(f'  {g["game_date"]}  {g["away_name"][:12]:12} @ {g["home_name"][:12]:12}  '
                  f'model {r["home_prob"]*100:.0f}% home  '
                  f'actual={g["home_score"]}-{g["away_score"]}  {ok}')
        print(f'\nRun without --dry-run to write new records to DB.')
        return

    # Write to DB. Importing app.py normally starts the scheduler, warms live
    # API caches, and synchronously recomputes team-bias/trust-weights/Platt
    # from whatever's currently in the DB — all side effects we don't want
    # from a script import. DISABLE_STARTUP_TASKS suppresses that (see
    # app.py's _start_cache_warmer() guard); mlb_model picks up its team-bias
    # correction from the last snapshot the live app wrote, not a live
    # recompute triggered by this run.
    os.environ['DISABLE_STARTUP_TASKS'] = '1'
    from app import app as flask_app, db, GamePrediction

    now = datetime.now(timezone.utc)
    inserted = updated = skipped_err = 0

    with flask_app.app_context():
        existing_keys = set(
            db.session.query(
                GamePrediction.game_date, GamePrediction.home_team, GamePrediction.away_team
            ).filter_by(sport='MLB').all()
        )

    new_count = sum(
        1 for g in games
        if (g['game_date'], g['home_name'], g['away_name']) not in existing_keys
    )
    print(f'\n{len(games)} completed games total — {new_count} new, '
          f'{len(games) - new_count} already recorded (untouched aside from outcome fields).',
          flush=True)

    with flask_app.app_context():
        for g in games:
            try:
                pred = GamePrediction.query.filter_by(
                    sport='MLB',
                    game_date=g['game_date'],
                    home_team=g['home_name'],
                    away_team=g['away_name'],
                ).first()
                is_new = pred is None

                if is_new:
                    pred = GamePrediction(
                        sport='MLB',
                        game_date=g['game_date'],
                        game_time_utc=g['game_time'],
                        home_team=g['home_name'],
                        away_team=g['away_name'],
                    )
                    db.session.add(pred)
                    inserted += 1
                else:
                    updated += 1

                # Prob/factors are computed ONCE, on insert, from that game's
                # point-in-time snapshot — and never recomputed after. See the
                # module docstring for why this replaced the old "recompute
                # every rerun" behavior.
                if is_new:
                    home   = _team(g, 'home')
                    away   = _team(g, 'away')
                    result = mlb_model.predict(home, away, game_time_utc=g['game_time'])
                    pred.home_prob    = result['home_prob']
                    pred.away_prob    = result['away_prob']
                    pred.factors_json = json.dumps(result['factors'])

                # Record outcome (idempotent — safe to re-set the same values
                # on every run, even for games recorded long ago).
                pred.home_won       = g['home_won']
                pred.home_score     = g['home_score']
                pred.away_score     = g['away_score']
                pred.outcome_set_at = now

                # Recompute pick_roi so it stays consistent with home_prob
                # (which, for existing records, was never touched above).
                fav_home  = (pred.home_prob or 0.5) >= 0.5
                pick_odds = pred.home_odds if fav_home else pred.away_odds
                if pick_odds:
                    try:
                        o = int(pick_odds)
                        profit = o / 100.0 if o > 0 else 100.0 / (-o)
                        model_won = fav_home == g['home_won']
                        pred.pick_roi = round(profit if model_won else -1.0, 4)
                    except (TypeError, ValueError):
                        pass

            except Exception as e:
                skipped_err += 1
                print(f'  ! error on {g.get("game_date")} '
                      f'{g.get("away_name")} @ {g.get("home_name")}: {e}')

        db.session.commit()

        # Accuracy from the DB's own stored (frozen) predictions — not a live
        # recompute, so it reflects exactly what's on the Model Performance
        # page and costs no extra API calls.
        resolved = GamePrediction.query.filter(
            GamePrediction.sport == 'MLB',
            GamePrediction.home_won.isnot(None),
            GamePrediction.home_prob.isnot(None),
        ).all()
        correct = sum(1 for p in resolved if (p.home_prob >= 0.5) == bool(p.home_won))
        total_resolved = len(resolved)

    acc = correct / total_resolved * 100 if total_resolved else 0

    print(f'\n  ✓ Done')
    print(f'    Inserted:  {inserted:>4}  new records (freshly scored, point-in-time)')
    print(f'    Touched:   {updated:>4}  existing records (outcome fields only — prob/factors untouched)')
    if skipped_err:
        print(f'    Errors:    {skipped_err:>4}')
    if _degraded_dates:
        print(f'    ! FanGraphs xFIP/SIERA/K%/BB%/SwStr% came back empty for '
              f'{len(_degraded_dates)} date(s) among the new games this run — '
              f'those games were scored without those factors (degraded, not wrong: '
              f'already-recorded games are never touched regardless).')
    print(f'    Pick accuracy across {total_resolved} resolved 2026 predictions: {acc:.1f}%')
    print(f'\n  Model Performance page now has {total_resolved} resolved 2026 predictions.')
    print(f'  Visit /model to see calibration and factor correlation.')


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    run(dry_run=dry_run)
