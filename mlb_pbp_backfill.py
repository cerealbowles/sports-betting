#!/usr/bin/env python3
"""
mlb_pbp_backfill.py — Backfill play-by-play + win-probability data for
completed MLB games, as raw material for a potential in-game win-probability
model (separate from mlb_model.py's pregame spread/ML model).

Source: MLB Stats API, same host mlb_api.py already calls, no auth needed —
  - /api/v1/schedule                  — which games are Final, per season
  - /api/v1.1/game/{pk}/feed/live     — every pitch/at-bat: count, outs,
                                         runners, batter/pitcher, result
  - /api/v1/game/{pk}/winProbability  — per-play home/away win probability,
                                         WPA, and leverage index, already
                                         computed by MLB (no modeling needed
                                         to get this piece)

Storage: a dedicated sqlite file (mlb_pbp.db) next to bets.db (DB_PATH's
directory in prod, via the same bind mount other instance-data files use —
see mlb_model.py's _TEAM_BIAS_DATA_DIR for the identical reasoning: the code
dir gets reset on every deploy, instance data must not live there). Kept
separate from bets.db itself since this is plays-per-game, not games —
orders of magnitude more rows, and irrelevant to the Flask app/game_predictions
table, so no reason to load it into that schema or that process.

Idempotent / resumable by design (see backfill_2026.py for why this matters —
a prior MLB backfill script in this repo had a non-idempotence bug): each
game_pk is written to `games` only after all of its plays commit successfully,
and the main loop skips any game_pk already present in `games`. Re-running
after an interruption just resumes; it never re-fetches or double-inserts.

Usage:
    python3 mlb_pbp_backfill.py                       # 2024-2026 regular season (matches mlb_model.py's training window)
    python3 mlb_pbp_backfill.py --seasons 2021 2022 2023 2024 2025 2026
    python3 mlb_pbp_backfill.py --limit 50             # smoke-test on 50 games
    python3 mlb_pbp_backfill.py --dry-run
"""
import argparse
import os
import sqlite3
import sys
import time

import requests

HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; mlb-pbp-backfill/1.0)'}
MLB_API = 'https://statsapi.mlb.com/api/v1'
MLB_API_V11 = 'https://statsapi.mlb.com/api/v1.1'

# Matches bootstrap_2024.py's SEASON_DATES — the window mlb_model.py's
# pregame model is already trained/validated on. Override with --seasons to
# go further back; MLB Stats API itself serves pitch-level play-by-play back
# to roughly 2008 and play-by-play (without pitch tracking) much further,
# but older seasons are out of scope until this proves useful at all.
SEASON_DATES = {
    2021: ('2021-04-01', '2021-10-03'),
    2022: ('2022-04-07', '2022-10-05'),
    2023: ('2023-03-30', '2023-10-01'),
    2024: ('2024-03-20', '2024-09-29'),
    2025: ('2025-03-27', '2025-09-28'),
    2026: ('2026-03-26', '2026-09-27'),
}
DEFAULT_SEASONS = [2024, 2025, 2026]

_DB_DIR = os.path.dirname(os.environ.get('DB_PATH', '')) or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
os.makedirs(_DB_DIR, exist_ok=True)
DB_PATH = os.path.join(_DB_DIR, 'mlb_pbp.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_pk     INTEGER PRIMARY KEY,
    season      INTEGER NOT NULL,
    game_date   TEXT NOT NULL,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    home_score  INTEGER,
    away_score  INTEGER,
    num_plays   INTEGER,
    fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS plays (
    game_pk              INTEGER NOT NULL,
    at_bat_index         INTEGER NOT NULL,
    season               INTEGER NOT NULL,
    inning               INTEGER NOT NULL,
    half                 TEXT NOT NULL,       -- 'top' | 'bottom'
    outs_before          INTEGER,             -- outs at START of this play
    outs_after           INTEGER,
    balls                INTEGER,
    strikes              INTEGER,
    home_score           INTEGER,             -- score AFTER this play resolves
    away_score           INTEGER,
    batter_id            INTEGER,
    pitcher_id           INTEGER,
    bat_side             TEXT,
    pitch_hand           TEXT,
    men_on_base          TEXT,                -- MLB's own split label, e.g. 'Empty'/'RISP'
    on_1b                INTEGER,             -- 1/0 runner-on-base flags AFTER this play
    on_2b                INTEGER,
    on_3b                INTEGER,
    event                TEXT,
    event_type           TEXT,
    is_scoring_play      INTEGER,
    rbi                  INTEGER,
    num_pitches          INTEGER,
    home_win_prob        REAL,                -- 0-100, AFTER this play
    away_win_prob        REAL,
    home_win_prob_added  REAL,                -- signed, this play's swing
    leverage_index       REAL,
    PRIMARY KEY (game_pk, at_bat_index)
);

CREATE INDEX IF NOT EXISTS idx_plays_season ON plays(season);
"""


def _get(url, params, label, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30, headers=HEADERS)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries - 1:
                print(f'  ! {label} failed: {e}', flush=True)
                return None
            time.sleep(1 + attempt)
    return None


def _month_ranges(start, end):
    """Split a season's date range into <=1-month chunks (schedule endpoint
    caps how much it'll return usefully in one call) — same chunking
    bootstrap_2024.py uses for the same endpoint."""
    from datetime import date, timedelta
    y0, m0, d0 = (int(x) for x in start.split('-'))
    y1, m1, d1 = (int(x) for x in end.split('-'))
    cur = date(y0, m0, d0)
    stop = date(y1, m1, d1)
    while cur <= stop:
        nxt = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1) - timedelta(days=1)
        chunk_end = min(nxt, stop)
        yield cur.isoformat(), chunk_end.isoformat()
        cur = chunk_end + timedelta(days=1)


def fetch_season_games(season):
    """[(game_pk, game_date, home_team, away_team, home_score, away_score), ...]
    for every Final regular-season game in `season`."""
    start, end = SEASON_DATES[season]
    out = []
    for chunk_start, chunk_end in _month_ranges(start, end):
        r = _get(f'{MLB_API}/schedule', {
            'sportId': 1, 'gameType': 'R', 'season': season,
            'startDate': chunk_start, 'endDate': chunk_end,
        }, f'{season} schedule {chunk_start}..{chunk_end}')
        if r is None:
            continue
        for day in r.json().get('dates', []):
            for g in day.get('games', []):
                if (g.get('status') or {}).get('detailedState') != 'Final':
                    continue
                teams = g.get('teams', {})
                home, away = teams.get('home', {}), teams.get('away', {})
                out.append((
                    g['gamePk'], day['date'],
                    (home.get('team') or {}).get('name', ''),
                    (away.get('team') or {}).get('name', ''),
                    home.get('score'), away.get('score'),
                ))
    return out


def fetch_game_plays(game_pk, season):
    """Returns a list of row-tuples for the `plays` table, joining feed/live
    (play structure, count, runners, batter/pitcher) with winProbability
    (home/away win%, WPA, leverage) by at_bat_index. Returns None on fetch
    failure (caller should skip, not record the game as done)."""
    feed_r = _get(f'{MLB_API_V11}/game/{game_pk}/feed/live', {}, f'feed/live {game_pk}')
    if feed_r is None:
        return None
    all_plays = (feed_r.json().get('liveData') or {}).get('plays', {}).get('allPlays', [])

    wp_r = _get(f'{MLB_API}/game/{game_pk}/winProbability', {}, f'winProbability {game_pk}')
    wp_by_index = {}
    if wp_r is not None:
        try:
            for p in wp_r.json():
                wp_by_index[p.get('atBatIndex')] = p
        except Exception:
            pass

    rows = []
    for p in all_plays:
        about = p.get('about', {})
        if not about.get('isComplete'):
            continue
        idx = about.get('atBatIndex')
        result = p.get('result', {})
        matchup = p.get('matchup', {})
        count = p.get('count', {})
        wp = wp_by_index.get(idx, {})

        on_1b = 1 if matchup.get('postOnFirst') else 0
        on_2b = 1 if matchup.get('postOnSecond') else 0
        on_3b = 1 if matchup.get('postOnThird') else 0

        # outs-before = outs-after minus outs recorded on this play, unless
        # this play ended the half-inning's 3rd out (then before = after-count
        # from the *previous* play, which we don't have handy here — outs
        # reset to 0 at half-inning start is the only ambiguous case, and the
        # 'inning'/'half' change already disambiguates it for any consumer
        # grouping by half-inning, so left as the raw API value, not derived).
        rows.append((
            game_pk, idx, season,
            about.get('inning'), 'bottom' if about.get('isTopInning') is False else 'top',
            None, count.get('outs'),
            count.get('balls'), count.get('strikes'),
            result.get('homeScore'), result.get('awayScore'),
            (matchup.get('batter') or {}).get('id'),
            (matchup.get('pitcher') or {}).get('id'),
            (matchup.get('batSide') or {}).get('code'),
            (matchup.get('pitchHand') or {}).get('code'),
            ((matchup.get('splits') or {}).get('menOnBase')),
            on_1b, on_2b, on_3b,
            result.get('event'), result.get('eventType'),
            1 if about.get('isScoringPlay') else 0, result.get('rbi'),
            len(p.get('pitchIndex') or []),
            wp.get('homeTeamWinProbability'), wp.get('awayTeamWinProbability'),
            wp.get('homeTeamWinProbabilityAdded'), wp.get('leverageIndex'),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', type=int, nargs='+', default=DEFAULT_SEASONS)
    ap.add_argument('--limit', type=int, default=None, help='Stop after N new games (smoke test)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    done_pks = {row[0] for row in conn.execute('SELECT game_pk FROM games')}
    print(f'[mlb_pbp_backfill] {DB_PATH} — {len(done_pks)} games already recorded', flush=True)

    fetched = 0
    for season in args.seasons:
        if season not in SEASON_DATES:
            print(f'  ! no SEASON_DATES entry for {season}, skipping', flush=True)
            continue
        print(f'Season {season}: fetching schedule...', flush=True)
        games = fetch_season_games(season)
        todo = [g for g in games if g[0] not in done_pks]
        print(f'  {len(games)} Final games, {len(todo)} not yet backfilled', flush=True)

        for game_pk, game_date, home_team, away_team, home_score, away_score in todo:
            if args.limit is not None and fetched >= args.limit:
                print(f'Hit --limit {args.limit}, stopping.', flush=True)
                conn.close()
                return

            rows = fetch_game_plays(game_pk, season)
            if rows is None:
                print(f'  ! {game_pk} ({game_date} {away_team}@{home_team}): fetch failed, will retry next run', flush=True)
                time.sleep(0.5)
                continue
            if args.dry_run:
                print(f'  [dry-run] {game_pk} ({game_date} {away_team}@{home_team}): {len(rows)} plays', flush=True)
                fetched += 1
                time.sleep(0.3)
                continue

            conn.executemany(
                'INSERT OR REPLACE INTO plays VALUES '
                '(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                rows)
            conn.execute(
                'INSERT INTO games (game_pk, season, game_date, home_team, away_team, home_score, away_score, num_plays) '
                'VALUES (?,?,?,?,?,?,?,?)',
                (game_pk, season, game_date, home_team, away_team, home_score, away_score, len(rows)))
            conn.commit()
            done_pks.add(game_pk)
            fetched += 1
            if fetched % 25 == 0:
                print(f'  ... {fetched} games backfilled so far', flush=True)
            time.sleep(0.3)  # be polite to statsapi.mlb.com — no documented rate limit, but no need to hammer it

    conn.close()
    print(f'Done. {fetched} new games fetched this run.', flush=True)


if __name__ == '__main__':
    main()
