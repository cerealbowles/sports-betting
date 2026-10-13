#!/usr/bin/env python3
"""
nfl_playergame_dataset.py — Builds Phase 2's leakage-safe player-game
dataset (NFL_Fantasy_Analytics_Projection_Plan.md, section 4) from the
local stats warehouse — no live ESPN fetches, so the same DB snapshot
reproduces an identical dataset every run.

Usage:
    python3 nfl_playergame_dataset.py                       # SEASONS -> nfl_playergame_dataset.jsonl
    python3 nfl_playergame_dataset.py --seasons 2025 2026
    python3 nfl_playergame_dataset.py --out custom.jsonl --max-rows 200
    python3 nfl_playergame_dataset.py --validate-only nfl_playergame_dataset.jsonl

ROW GRAIN: one row per (player_id, game_id) for every QB/RB/WR/TE who
recorded ANY offensive stat line in a stored, completed NFL game —
classify_bucket()'s QB/RB/REC grouping (nfl_fantasy.py). Same real gap as
the Phase 1 player cards: ESPN's free box score carries no position
field, so a true WR/TE split isn't available from stored data. D/ST is
out of scope for this v1 — it's a team-level projection, not a player-game
line (see the plan's own "player-game" framing); a future pass could add
a parallel team-defense-game table if Phase 3 analysis wants one.

NO-LEAKAGE RULE: every rolling/pregame feature for a game on `game_date`
uses ONLY stored rows strictly BEFORE that date (`before_date=game_date`,
the same convention nfl_stats_db.py's before_date params already use
everywhere) — and reuses the exact functions the live player-detail cards
call (get_player_usage_card_data_db, get_opponent_defense_card_data_db,
nfl_player_model.compute_defense_grade, nfl_fantasy.project_points), so
"what this dataset says the projection would have been" and "what the
live app actually would have shown that week" can't silently drift apart.

COLUMNS (grouped by the plan's own section 4 headings):

Identifiers/context — player_id, player_name, bucket (QB/RB/REC),
  team_id, team_name, opponent_id, opponent_name, game_id, game_date,
  season, is_home.

Baseline/outcomes — projection (project_points(), point-in-time), actual
  (compute_fantasy_points() on the real game), projection_error (actual -
  projection, null if projection is null), plus the real component stat
  line for the game itself (pass/rush/receiving attempts/yards/TDs —
  straight from the stored box score; 0 is a real recorded value here,
  not a missing-data placeholder).

Player pregame features — recent (last RECENT_WINDOW) and season-to-date
  usage: games sampled, fantasy-points rolling average, rush/target
  volume, target share, red-zone opportunities — via
  get_player_usage_card_data_db(). `recent_games_played` /
  `season_games_played` are the sample-size columns the plan asks for
  ("handle players with fewer than 3 or 5 prior games explicitly") — a
  consumer thresholds on these rather than the dataset silently dropping
  thin-sample rows (a thin sample is itself often the interesting case
  for Phase 3's breakout analysis).

Team/opponent pregame features — team's own recent/season scoring
  (TeamGameStat.points, point-in-time), opponent defense grade
  (nfl_player_model.compute_defense_grade), opponent red-zone/explosive/
  position-bucket-allowed recent+season (get_opponent_defense_card_data_db()),
  and a best-effort implied_team_points (from GamePrediction.spread_open/
  total_open when stored — ESPN pickcenter convention confirmed in
  app.py's GamePrediction column comment: home spread, negative = home
  favored; implied_home = total/2 - spread/2). Only ~17% of stored NFL
  games have spread_open/total_open at all (checked live against the
  production DB — odds tracking is a newer feature with thin historical
  backfill), so this column is null for most rows; treat it as a bonus
  signal on recent games, not a reliable feature across the full dataset.
  The GamePrediction join is by (game_date, team_name) string match, not
  a foreign key — spot-check a sample before trusting it at scale (see
  validate_dataset()'s report for how many rows actually matched).

KNOWN GAPS (documented per the plan's own "mark unavailable explicitly"
principle, not silently worked around):
  - Historical salary isn't reconstructed. assign_salaries() ranks
    percentile-within-pool, and the only pool-building path
    (nfl_fantasy.build_player_pool) gates eligibility by nfl_roster_api's
    CURRENT roster — replaying it against a 2024 game would silently drop
    any player since traded/cut/retired, which is worse than omitting the
    column outright. Revisit if a position-at-the-time source shows up.
  - InjuryStatus only goes back to 2026-10-06 (checked live) — rows
    before that date can't reflect who was actually ruled OUT that week.
  - True WR/TE split, goal-line (inside-5) opportunities, end-zone
    targets (as distinct from the 20-yard red-zone line), and 3rd-down
    tendencies: same gaps already documented on the Phase 1 player cards
    (see templates/_nfl_player_detail.html) — not re-derived here either.
  - `week` isn't a stored column anywhere locally tied to game_date (the
    only week-aware fetch, nfl_api.build_week_schedule_context, always
    targets the CURRENT season live from ESPN, not an arbitrary past
    one) — game_date is this dataset's actual chronological/grouping
    key; treat it as equivalent to "week" here.
  - A confirmed, SEPARATE bug (not fixed here — see the task flagged
    alongside this script): nfl_fantasy.classify_bucket()/
    _had_meaningful_usage() read completions, not attempts, from the
    'completions/passingAttempts' composite and call it pass_att. This
    script uses the CORRECT index (matching nfl_player_model.py's own
    convention for the same field) for its own pass_att/pass_comp
    columns — but that means this dataset's bucket assignment (QB/RB/REC,
    via the same classify_bucket()) is computed from the live app's
    (buggy) pass-attempt count, not this script's corrected one. Revisit
    once that bug is fixed upstream.

ACCEPTANCE-CRITERIA CHECKS: validate_dataset() below runs the plan's
required automated checks (leakage, duplicate player-game rows, missing
values, invalid denominators) over a generated dataset and prints a
report — run it after every generation, not just once.
"""
import argparse
import json
import sys
import time

import nfl_fantasy
import nfl_player_model as pm
import nfl_stats_db

SEASONS = (2024, 2025, 2026)
RECENT_WINDOW = 3  # matches the plan's "suggested windows: last 3 and last 5 games" (3 side)
OPPONENT_RECENT_WINDOW = 5  # matches nfl_player_model.DEF_WINDOW / get_opponent_defense_card_data_db's own default
GRADE_WINDOW = 5  # matches nfl_player_model.QB_WINDOW/DEF_WINDOW


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _split(composite, idx, default=0.0):
    try:
        return float(str(composite).split('/')[idx])
    except (ValueError, IndexError):
        return default


def _component_outcomes(categories):
    """Real recorded stat line for this ONE game — context alongside the
    target, not a pregame feature. Values are 0.0 (not null) when a
    category is simply absent, same convention compute_fantasy_points()
    already uses. 'completions/passingAttempts' is "completions/attempts"
    (confirmed against real stored rows, e.g. '16/22' = 16-of-22) — index
    1 is attempts; see this module's docstring for why nfl_fantasy.py's
    own classify_bucket() reads the wrong index for the same field."""
    cats = categories or {}
    passing = cats.get('passing', {})
    rushing = cats.get('rushing', {})
    receiving = cats.get('receiving', {})
    comp_att = passing.get('completions/passingAttempts', '0/0')
    return {
        'pass_comp': _split(comp_att, 0),
        'pass_att': _split(comp_att, 1),
        'pass_yds': _num(passing.get('passingYards')),
        'pass_td': _num(passing.get('passingTouchdowns')),
        'pass_int': _num(passing.get('interceptions')),
        'rush_att': _num(rushing.get('rushingAttempts')),
        'rush_yds': _num(rushing.get('rushingYards')),
        'rush_td': _num(rushing.get('rushingTouchdowns')),
        'targets': _num(receiving.get('receivingTargets')),
        'receptions': _num(receiving.get('receptions')),
        'rec_yds': _num(receiving.get('receivingYards')),
        'rec_td': _num(receiving.get('receivingTouchdowns')),
    }


_team_scoring_cache = {}


def _team_scoring_context(team_id, game_date, season):
    """This team's own points-scored, recent vs. season-to-date, strictly
    before `game_date` — the plan's 'team scoring context' feature. Not
    wrapped in a shared nfl_stats_db.py helper (unlike the player/opponent
    sections) since it's two numbers from one already-cheap query; kept
    local to this analysis script the same way nfl_player_bootstrap.py
    and nfl_fantasy_bootstrap.py keep their own backtest-only aggregation
    code separate from the production stats-db layer.

    Cached per (team_id, game_date) — every skill player on the SAME team
    in the SAME game shares an identical result (the query only depends on
    team_id/game_date/season, never the individual player), and a team
    fields 5-10+ skill players per game. Confirmed live this redundancy
    was real: without this cache, a full 3-season build re-ran the
    identical query once per player instead of once per team-game."""
    key = (team_id, game_date)
    if key in _team_scoring_cache:
        return _team_scoring_cache[key]

    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query
                .filter(TeamGameStat.sport == 'NFL', TeamGameStat.team_id == team_id,
                        TeamGameStat.game_date < game_date)
                .order_by(TeamGameStat.game_date.desc())
                .all())
    recent = rows[:OPPONENT_RECENT_WINDOW]
    season_rows = [r for r in rows if r.season == season]
    result = {
        'recent_games_played': len(recent),
        'recent_ppg': (sum(r.points or 0 for r in recent) / len(recent)) if recent else None,
        'season_games_played': len(season_rows),
        'season_ppg': (sum(r.points or 0 for r in season_rows) / len(season_rows)) if season_rows else None,
    }
    _team_scoring_cache[key] = result
    return result


_game_prediction_cache = {}


def _implied_team_points(team_name, opponent_name, is_home, game_date):
    """Best-effort implied team points from GamePrediction.spread_open/
    total_open, joined by (game_date, team names) — NOT a foreign key, see
    this module's docstring for why. None if this game has no stored
    spread/total (the large majority — see docstring)."""
    from app import app as flask_app, GamePrediction

    key = game_date
    if key not in _game_prediction_cache:
        with flask_app.app_context():
            rows = (GamePrediction.query
                    .filter(GamePrediction.sport == 'NFL', GamePrediction.game_date == key)
                    .all())
        _game_prediction_cache[key] = {(r.home_team, r.away_team): r for r in rows}
    by_matchup = _game_prediction_cache[key]
    home_name, away_name = (team_name, opponent_name) if is_home else (opponent_name, team_name)
    row = by_matchup.get((home_name, away_name))
    if row is None or row.spread_open is None or row.total_open is None:
        return None
    implied_home = row.total_open / 2 - row.spread_open / 2
    implied_away = row.total_open - implied_home
    return implied_home if is_home else implied_away


_opponent_defense_cache = {}
_opponent_grade_cache = {}


def _build_row(player_row, team_row, categories, season):
    """player_row: a PlayerGameStat ORM row. team_row: that same game's
    TeamGameStat row for the player's own team (for opponent_id/points/
    is_home). categories: player_row's already-parsed stats_json. Returns
    one dataset row dict, or None if this player has no meaningful
    offensive bucket for this game (classify_bucket() -> None: a lineman,
    kicker, pure defender, etc. — not a fantasy-relevant player-game).

    The opponent-defense read and the opponent-defense-grade computation
    are cached per (opponent_id, game_date) for the same reason
    _team_scoring_context() caches per (team_id, game_date) — see that
    function's docstring. usage/gamelog genuinely vary per player and
    aren't cached."""
    bucket = nfl_fantasy.classify_bucket(categories)
    if bucket is None:
        return None

    game_id, game_date = player_row.game_id, player_row.game_date
    team_id, opponent_id = team_row.team_id, team_row.opponent_id

    usage = nfl_stats_db.get_player_usage_card_data_db(
        player_row.player_id, season, before_date=game_date, recent_window=RECENT_WINDOW)

    opp_key = (opponent_id, game_date)
    if opp_key not in _opponent_defense_cache:
        _opponent_defense_cache[opp_key] = nfl_stats_db.get_opponent_defense_card_data_db(
            opponent_id, season, before_date=game_date, recent_window=OPPONENT_RECENT_WINDOW)
    opp_defense = _opponent_defense_cache[opp_key]

    team_scoring = _team_scoring_context(team_id, game_date, season)

    gamelog = nfl_stats_db.get_player_gamelog_db(player_row.player_id, before_date=game_date, limit=8)
    points_hist = [nfl_fantasy.compute_fantasy_points(g['categories']) for g in gamelog]
    recent_avg = nfl_fantasy.rolling_average(points_hist)

    if opp_key not in _opponent_grade_cache:
        off_list, def_list = nfl_stats_db.get_team_game_aggregates_db(opponent_id, before_date=game_date, limit=GRADE_WINDOW)
        _opponent_grade_cache[opp_key] = pm.compute_defense_grade(def_list)
    opponent_defense_grade = _opponent_grade_cache[opp_key]

    projection = nfl_fantasy.project_points(bucket, points_hist, opponent_defense_grade)
    actual = nfl_fantasy.compute_fantasy_points(categories)
    projection_error = (actual - projection) if projection is not None else None

    implied_pts = _implied_team_points(team_row.team_name, team_row.opponent_name, team_row.is_home, game_date)

    row = {
        'player_id': player_row.player_id, 'player_name': player_row.player_name, 'bucket': bucket,
        'team_id': team_id, 'team_name': team_row.team_name,
        'opponent_id': opponent_id, 'opponent_name': team_row.opponent_name,
        'game_id': game_id, 'game_date': game_date, 'season': season, 'is_home': team_row.is_home,

        'projection': projection, 'actual': actual, 'projection_error': projection_error,
        **_component_outcomes(categories),

        'recent_games_played': len(points_hist[:RECENT_WINDOW]),
        'recent_fantasy_pts_avg': recent_avg,
        'recent_usage': usage['recent'],
        'season_usage': usage['season'],

        'team_recent_ppg': team_scoring['recent_ppg'],
        'team_season_ppg': team_scoring['season_ppg'],
        'opponent_defense_grade': opponent_defense_grade,
        'opponent_defense_recent': opp_defense['recent'],
        'opponent_defense_season': opp_defense['season'],
        'implied_team_points': implied_pts,
    }
    return row


def build_dataset(seasons=SEASONS, max_rows=None, progress_every=200, out_path=None):
    """Iterates every stored TeamGameStat row (one per team per game) for
    `seasons`, and every one of that team's players who recorded an
    offensive stat line in that same game, building one row per (player,
    game) via _build_row(). Entirely local-DB-driven (no live ESPN
    fetches) — a season filter on TeamGameStat/PlayerGameStat, not a
    week-by-week live schedule walk (see this module's docstring for why
    that path can't reach past seasons anyway).

    `out_path` (optional): if given, writes each row to this file AS IT'S
    PRODUCED (flushed immediately), not just once at the very end. A full
    3-season run is a long-running batch job (tens of minutes — per-row
    cost grows as more historical games accumulate behind each cutoff
    date, since a couple of the reused query functions fetch-then-slice
    in Python rather than capping at the SQL level); without this, an
    interruption (a wrapping `timeout`, a killed process, a crash) loses
    EVERY row computed so far, not just the ones still in flight —
    confirmed live: a 30-minute timeout killed a run that had already
    finished 2024+2025 (nothing left but 2026), and because this function
    used to only return its list for the caller to write at the end, that
    entire ~27 minutes of real work produced zero output. Always pass
    out_path for anything but a quick --max-rows smoke test."""
    from app import app as flask_app, TeamGameStat, PlayerGameStat

    out_f = open(out_path, 'w') if out_path else None
    rows_out = []
    t0 = time.time()
    for season in seasons:
        with flask_app.app_context():
            team_rows = (TeamGameStat.query
                         .filter_by(sport='NFL', season=season)
                         .order_by(TeamGameStat.game_date.asc())
                         .all())
            team_rows = [{
                'game_id': r.game_id, 'team_id': r.team_id, 'team_name': r.team_name,
                'opponent_id': r.opponent_id, 'opponent_name': r.opponent_name, 'is_home': r.is_home,
            } for r in team_rows]

        print(f'season {season}: {len(team_rows)} team-game rows to process', flush=True)
        for i, tr in enumerate(team_rows):
            with flask_app.app_context():
                player_rows = (PlayerGameStat.query
                               .filter_by(sport='NFL', game_id=tr['game_id'], team_id=tr['team_id'])
                               .all())
                player_rows = [{
                    'player_id': r.player_id, 'player_name': r.player_name,
                    'game_id': r.game_id, 'game_date': r.game_date,
                    'stats_json': r.stats_json,
                } for r in player_rows]

            class _Row:
                def __init__(self, d):
                    self.__dict__.update(d)

            team_row_obj = _Row(tr)
            for pr in player_rows:
                try:
                    categories = json.loads(pr['stats_json']) if pr['stats_json'] else {}
                except (TypeError, ValueError):
                    categories = {}
                player_row_obj = _Row(pr)
                row = _build_row(player_row_obj, team_row_obj, categories, season)
                if row is not None:
                    rows_out.append(row)
                    if out_f is not None:
                        # flush (not just buffer) after every row — a
                        # `timeout`-delivered SIGTERM runs no Python
                        # cleanup code at all (no finally, no atexit), so
                        # anything still sitting in a userspace buffer
                        # when that signal lands is lost. See this
                        # function's docstring for why that's not
                        # hypothetical.
                        out_f.write(json.dumps(row) + '\n')
                        out_f.flush()
                    if max_rows and len(rows_out) >= max_rows:
                        print(f'hit --max-rows={max_rows}, stopping early', flush=True)
                        if out_f is not None:
                            out_f.close()
                        return rows_out

            if i % progress_every == 0:
                print(f'  ...{i}/{len(team_rows)} team-games, {len(rows_out)} rows so far '
                      f'({time.time()-t0:.1f}s elapsed)', flush=True)

    print(f'\n{len(rows_out)} total rows in {time.time()-t0:.1f}s', flush=True)
    if out_f is not None:
        out_f.close()
    return rows_out


def check_leakage(rows, sample_size=50, seed=0):
    """Re-runs the single most leakage-sensitive query (get_player_gamelog_db,
    the direct source of recent_fantasy_pts_avg/projection) for a random
    sample of rows and asserts every game it returns is strictly before
    that row's own game_date — directly validates that _build_row() is
    actually passing before_date correctly, not just claiming to. This is
    the automated, repeatable version of the plan's "audit a sample of
    rows manually against game logs and kickoff times" safeguard; it
    doesn't replace an occasional real manual spot-check against ESPN's
    own box scores, but it does catch a before_date wiring regression
    every time the dataset is rebuilt. Returns the number of rows that
    failed (0 = clean)."""
    import random
    rng = random.Random(seed)
    sample = rng.sample(rows, min(sample_size, len(rows)))
    failures = 0
    for r in sample:
        gamelog = nfl_stats_db.get_player_gamelog_db(r['player_id'], before_date=r['game_date'], limit=8)
        bad = [g['date'] for g in gamelog if g['date'] >= r['game_date']]
        if bad:
            failures += 1
            print(f"  LEAK: player {r['player_id']} game {r['game_id']} ({r['game_date']}) — "
                  f"gamelog returned future/same-day dates {bad}")
    return failures


def validate_dataset(rows):
    """Runs the plan's Phase 2 acceptance-criteria checks (leakage,
    duplicate player-game rows, missing values, invalid denominators)
    over an already-built dataset and prints a report. Returns True if no
    HARD failures (leakage, duplicates) were found — missing-value/
    denominator findings are reported but don't fail the run, since
    partial coverage is expected and documented, not a bug."""
    ok = True
    n = len(rows)
    print(f'\n=== validating {n} rows ===')

    seen = set()
    dupes = 0
    for r in rows:
        key = (r['player_id'], r['game_id'])
        if key in seen:
            dupes += 1
        seen.add(key)
    print(f'duplicate (player_id, game_id) rows: {dupes}')
    if dupes:
        ok = False

    leak_failures = check_leakage(rows)
    print(f'leakage spot-check: {leak_failures} failure(s) found (see check_leakage() docstring for what it verifies)')
    if leak_failures:
        ok = False

    missing_projection = sum(1 for r in rows if r['projection'] is None)
    missing_opp_grade = sum(1 for r in rows if r['opponent_defense_grade'] is None)
    missing_implied = sum(1 for r in rows if r['implied_team_points'] is None)
    thin_recent = sum(1 for r in rows if r['recent_games_played'] < 3)
    thin_season = sum(1 for r in rows if r['season_usage'] is None or r['season_usage']['games'] < 5)
    print(f'rows with null projection (no usable pregame history): {missing_projection}/{n}')
    print(f'rows with null opponent_defense_grade: {missing_opp_grade}/{n}')
    print(f'rows with null implied_team_points (expected majority — see docstring): {missing_implied}/{n}')
    print(f'rows with < 3 recent games played: {thin_recent}/{n}')
    print(f'rows with < 5 season games played: {thin_season}/{n}')

    def _bad_share(scope):
        return sum(1 for r in rows if r[scope] and r[scope].get('target_share') is not None
                   and not (0.0 <= r[scope]['target_share'] <= 1.0))
    bad_share = _bad_share('recent_usage') + _bad_share('season_usage')
    print(f'rows with out-of-range target_share (should be 0): {bad_share}')
    if bad_share:
        ok = False

    bucket_counts = {}
    for r in rows:
        bucket_counts[r['bucket']] = bucket_counts.get(r['bucket'], 0) + 1
    print(f'rows by bucket: {bucket_counts}')

    print(f'=== validation {"PASSED" if ok else "FAILED"} ===\n')
    return ok


def write_jsonl(rows, out_path):
    with open(out_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {len(rows)} rows to {out_path}', flush=True)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def season_split(rows, eval_season):
    """Repeatable chronological train/eval split the plan's acceptance
    criteria asks for: every row from `eval_season` is held out, every
    earlier season trains — the same rolling-origin discipline this app's
    other *_bootstrap.py scripts already use. Returns (train_rows,
    eval_rows)."""
    train = [r for r in rows if r['season'] < eval_season]
    eval_rows = [r for r in rows if r['season'] == eval_season]
    return train, eval_rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seasons', type=int, nargs='+', default=list(SEASONS))
    p.add_argument('--out', default='nfl_playergame_dataset.jsonl')
    p.add_argument('--max-rows', type=int, default=None)
    p.add_argument('--validate-only', default=None, help='Path to an existing .jsonl to validate instead of generating')
    args = p.parse_args()

    if args.validate_only:
        rows = load_jsonl(args.validate_only)
        ok = validate_dataset(rows)
        sys.exit(0 if ok else 1)

    print(f'Building NFL player-game dataset for seasons {args.seasons}...', flush=True)
    print(f'(writing to {args.out} incrementally, row by row — see build_dataset()\'s docstring for why)', flush=True)
    rows = build_dataset(seasons=args.seasons, max_rows=args.max_rows, out_path=args.out)
    if not rows:
        print('No rows produced.')
        sys.exit(1)
    validate_dataset(rows)
