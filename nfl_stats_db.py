"""
nfl_stats_db.py — Read/write layer for NFL's local stats warehouse: the
same sport-agnostic PlayerGameStat/TeamGameStat tables (app.py)
nba_stats_db.py writes to, just the NFL side of it.

Team-level (ingest_team_game/get_season_game_log_db) replaces
nfl_api._get_season_game_log()'s 18-week live ESPN walk, re-run on every
1-hour cache miss just for PPG/recent-form/rest-days inputs — a Final
game's result never changes, so there's no reason to keep re-fetching it.

Player-level (ingest_player_stats/get_player_gamelog_db) feeds
nfl_player_model.py's offense/defense unit grades — get_team_game_
aggregates_db() below is the bridge: groups a team's stored player rows
back into per-game team aggregates (nfl_player_model.parse_team_game_
aggregate's shape) for nfl_api.py's live predictions to grade off of.

Same lazy `from app import ...` pattern as nba_stats_db.py and every
*_backfill.py script — avoids a circular import with app.py, which
imports nfl_api.py (and, transitively, this module) at module level.
"""
import json
import re
from collections import defaultdict

GAME_TYPE_BY_ESPN_SEASON_TYPE = {1: 'preseason', 2: 'regular', 3: 'postseason'}


def ingest_team_game(event_id, game_date_et, season, game_type,
                      home_id, away_id, home_name, away_name, home_score, away_score):
    """Upserts one TeamGameStat row per side for a Final NFL game. No
    boxscore fetch needed (unlike NBA) — the final score off the scoreboard
    event nfl_api._build_game() already has is everything a team-level row
    needs. Idempotent: upserts on the (sport, game_id, team_id) unique
    constraint, safe to call repeatedly (recently-Final games re-ingested
    on every cron tick, backfill re-runs, etc.)."""
    if home_score is None or away_score is None:
        return
    from app import app as flask_app, db, TeamGameStat

    with flask_app.app_context():
        for team_id, opp_id, team_name, opp_name, team_score, opp_score, is_home in (
            (home_id, away_id, home_name, away_name, home_score, away_score, True),
            (away_id, home_id, away_name, home_name, away_score, home_score, False),
        ):
            if not team_id:
                continue
            existing = TeamGameStat.query.filter_by(
                sport='NFL', game_id=event_id, team_id=team_id).first()
            tt = existing or TeamGameStat(sport='NFL', game_id=event_id, team_id=team_id)
            tt.game_date      = game_date_et
            tt.season         = season
            tt.game_type      = game_type
            tt.team_name      = team_name
            tt.opponent_id    = opp_id
            tt.opponent_name  = opp_name
            tt.is_home        = is_home
            tt.points         = team_score
            tt.points_allowed = opp_score
            tt.won            = team_score > opp_score
            if not existing:
                db.session.add(tt)
        db.session.commit()


def get_season_game_log_db(season, game_type='regular'):
    """Returns every stored NFL game for `season` in the same shape
    nfl_api._get_season_game_log()'s live ESPN walk produces:
    [{game_date, home_name, away_name, home_score, away_score, home_won}].
    [] if this table has nothing for the season yet (caller falls back to
    the live walk). See nba_stats_db.get_season_game_log_db() — identical
    approach, NFL side."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query
                .filter_by(sport='NFL', season=season, game_type=game_type, is_home=True)
                .order_by(TeamGameStat.game_date.asc())
                .all())
        return [{
            'game_date':  r.game_date,
            'home_name':  r.team_name,
            'away_name':  r.opponent_name,
            'home_score': r.points,
            'away_score': r.points_allowed,
            'home_won':   r.won,
        } for r in rows if r.team_name and r.opponent_name]


def ingest_player_stats(event_id, game_date_et, season, game_type,
                         home_id, away_id, boxscore):
    """Upserts PlayerGameStat rows for every athlete in `boxscore` —
    {team_id: [{id, name, position, categories}, ...]} as returned by
    nfl_boxscore_api.parse_player_boxscore()/get_live_boxscore(). Each
    player's full category breakdown (passing/rushing/receiving/etc.,
    whichever categories they actually appeared in) is stored as JSON in
    `stats_json` — see PlayerGameStat's class docstring in app.py for why
    football doesn't get basketball's flat per-stat columns.

    Idempotent — upserts on the (sport, game_id, player_id) unique
    constraint, same as every other ingest_* function in this app's stats
    warehouse."""
    if not boxscore:
        return
    from app import app as flask_app, db, PlayerGameStat

    with flask_app.app_context():
        for team_id, opp_id, is_home in ((home_id, away_id, True), (away_id, home_id, False)):
            rows = boxscore.get(team_id)
            if rows is None:
                continue
            for r in rows:
                existing = PlayerGameStat.query.filter_by(
                    sport='NFL', game_id=event_id, player_id=r['id']).first()
                target = existing or PlayerGameStat(sport='NFL', game_id=event_id, player_id=r['id'])
                target.game_date   = game_date_et
                target.season      = season
                target.game_type   = game_type
                target.player_name = r['name']
                target.team_id     = team_id
                target.opponent_id = opp_id
                target.is_home     = is_home
                target.position    = r.get('position')
                target.stats_json  = json.dumps(r.get('categories', {}))
                if not existing:
                    db.session.add(target)
        db.session.commit()


def ingest_team_defense_vs_position(event_id, home_id, away_id, boxscore):
    """Computes and stores each team's fpts_allowed_qb/rb/rec for this one
    game — the OPPOSING offense's players' real FanDuel fantasy points,
    bucketed via nfl_fantasy.classify_bucket() (box-score ROLE that game,
    not roster position — a WR who takes a jet-sweep carry still buckets
    as RB for that specific stat line, matching how classify_bucket is
    used everywhere else in this app). `boxscore`: same {team_id: [{id,
    name, position, categories}, ...]} shape ingest_player_stats() takes —
    call this with the SAME boxscore right after that call, no extra
    fetch needed.

    Stored rather than recomputed on demand (unlike this file's other
    aggregation helpers — see get_team_game_aggregates_db()'s docstring)
    because defense-vs-position is read repeatedly for matchup analysis
    across many weeks at once, and because it was built specifically to be
    a real, growing historical record going forward, not just a page-load
    convenience.

    Expects ingest_team_game() to have already created this game's
    TeamGameStat rows (same per-game ingestion order nfl_stats_backfill.py
    and app.py's live path already use) — a missing row is skipped rather
    than created from scratch, since this function has no score/opponent-
    name data of its own to fill one in with.

    Idempotent — just an UPDATE on the existing (sport, game_id, team_id)
    row, safe to re-run (self-heal, backfill re-runs, etc.)."""
    if not boxscore:
        return
    import nfl_fantasy
    from app import app as flask_app, db, TeamGameStat

    def _fpts_by_bucket(player_rows):
        totals = {'QB': 0.0, 'RB': 0.0, 'REC': 0.0}
        for r in player_rows:
            cats = r.get('categories', {})
            bucket = nfl_fantasy.classify_bucket(cats)
            if bucket in totals:
                totals[bucket] += nfl_fantasy.compute_fantasy_points(cats)
        return totals

    with flask_app.app_context():
        for team_id, opp_id in ((home_id, away_id), (away_id, home_id)):
            opp_rows = boxscore.get(opp_id)
            if not team_id or opp_rows is None:
                continue
            allowed = _fpts_by_bucket(opp_rows)
            existing = TeamGameStat.query.filter_by(sport='NFL', game_id=event_id, team_id=team_id).first()
            if existing is None:
                continue
            existing.fpts_allowed_qb  = allowed['QB']
            existing.fpts_allowed_rb  = allowed['RB']
            existing.fpts_allowed_rec = allowed['REC']
        db.session.commit()


def get_defense_vs_position_db(team_id, bucket, before_date=None, limit=8):
    """This team's defense's stored fpts_allowed_<bucket> for its last
    `limit` games (newest first, strictly before `before_date` if given —
    same no-leakage convention as this module's other before_date params).
    bucket: 'QB' | 'RB' | 'REC'. Returns a plain list of floats (an empty
    list for a team/bucket with no stored games yet — e.g. before this
    feature existed, or `limit` just hasn't been reached). Callers average
    it themselves (same "raw rows in, pure-Python math by the caller"
    convention as get_team_game_aggregates_db())."""
    if bucket not in ('QB', 'RB', 'REC'):
        raise ValueError(f"bucket must be 'QB', 'RB', or 'REC', got {bucket!r}")
    from app import app as flask_app, TeamGameStat
    col = {'QB': TeamGameStat.fpts_allowed_qb, 'RB': TeamGameStat.fpts_allowed_rb,
           'REC': TeamGameStat.fpts_allowed_rec}[bucket]

    with flask_app.app_context():
        q = TeamGameStat.query.filter(TeamGameStat.sport == 'NFL', TeamGameStat.team_id == team_id,
                                       col.isnot(None))
        if before_date:
            q = q.filter(TeamGameStat.game_date < before_date)
        rows = q.order_by(TeamGameStat.game_date.desc()).limit(limit).all()
        return [getattr(r, f'fpts_allowed_{bucket.lower()}') for r in rows]


def ingest_play_by_play(event_id, game_date_et, season, game_type, summary):
    """Parses `summary` via nfl_playbyplay_api.parse_play_splits() and
    writes all three of its outputs:
      - raw_plays -> NflPlay rows (bulk upsert on (game_id, play_id))
      - player_splits -> merged into each player's EXISTING PlayerGameStat
        .stats_json as new category keys ('rushing_splits',
        'receiving_splits', 'passing_splits') — ADDITIVE, existing
        categories (from ingest_player_stats) are preserved, not
        overwritten. Only merges into a row that already exists (expects
        ingest_player_stats() to have run first for this game, same order
        every caller already uses) — this never creates a sparse
        PlayerGameStat row of its own.
      - team_allowed -> merged into each team's EXISTING TeamGameStat row
        (long_plays_allowed/redzone_trips_allowed/redzone_tds_allowed),
        same match-only behavior, expects ingest_team_game() first.

    Call this with the SAME `summary` ingest_player_stats() already
    parsed a boxscore from — zero extra fetches (see nfl_playbyplay_api.py
    and nfl_boxscore_api.get_live_summary() for the shared-cache path
    backfill callers use to get that same summary without a second
    request).

    Idempotent: NflPlay rows upsert on their own unique constraint; the
    stats_json merge and TeamGameStat column updates just overwrite with
    the freshly re-parsed values on every call, safe to re-run (self-heal,
    backfill re-runs, etc.)."""
    import json as _json
    import nfl_playbyplay_api
    from app import app as flask_app, db, PlayerGameStat, TeamGameStat, NflPlay

    result = nfl_playbyplay_api.parse_play_splits(summary)
    if not result['raw_plays'] and not result['player_splits'] and not result['team_allowed']:
        return

    with flask_app.app_context():
        existing_play_ids = {
            r.play_id for r in NflPlay.query.filter_by(game_id=event_id).with_entities(NflPlay.play_id).all()
        }
        for p in result['raw_plays']:
            if not p.get('play_id'):
                continue
            if p['play_id'] in existing_play_ids:
                NflPlay.query.filter_by(game_id=event_id, play_id=p['play_id']).update({
                    'drive_id': p['drive_id'], 'sequence_number': p['sequence_number'],
                    'season': season, 'game_date': game_date_et,
                    'offense_team_id': p['offense_team_id'], 'defense_team_id': p['defense_team_id'],
                    'play_type': p['play_type'], 'text': p['text'],
                    'down': p['down'], 'distance': p['distance'], 'yards_to_endzone': p['yards_to_endzone'],
                    'yards_gained': p['yards_gained'], 'is_scoring_play': p['is_scoring_play'],
                    'is_redzone': p['is_redzone'], 'is_turnover': p['is_turnover'], 'is_no_play': p['is_no_play'],
                    'period': p['period'], 'clock': p['clock'],
                    'pass_depth': p['pass_depth'], 'yards_after_catch': p['yards_after_catch'],
                    'passer_id': p['passer_id'], 'rusher_id': p['rusher_id'], 'receiver_id': p['receiver_id'],
                    'penalty_team_id': p['penalty_team_id'], 'penalty_player_id': p['penalty_player_id'],
                    'penalty_type': p['penalty_type'], 'penalty_yards': p['penalty_yards'],
                    'penalty_status': p['penalty_status'],
                    'end_down': p['end_down'], 'end_distance': p['end_distance'],
                    'end_yards_to_endzone': p['end_yards_to_endzone'], 'end_team_id': p['end_team_id'],
                })
            else:
                db.session.add(NflPlay(
                    game_id=event_id, play_id=p['play_id'], drive_id=p['drive_id'],
                    sequence_number=p['sequence_number'], season=season, game_date=game_date_et,
                    offense_team_id=p['offense_team_id'], defense_team_id=p['defense_team_id'],
                    play_type=p['play_type'], text=p['text'],
                    down=p['down'], distance=p['distance'], yards_to_endzone=p['yards_to_endzone'],
                    yards_gained=p['yards_gained'], is_scoring_play=p['is_scoring_play'],
                    is_redzone=p['is_redzone'], is_turnover=p['is_turnover'], is_no_play=p['is_no_play'],
                    period=p['period'], clock=p['clock'],
                    pass_depth=p['pass_depth'], yards_after_catch=p['yards_after_catch'],
                    passer_id=p['passer_id'], rusher_id=p['rusher_id'], receiver_id=p['receiver_id'],
                    penalty_team_id=p['penalty_team_id'], penalty_player_id=p['penalty_player_id'],
                    penalty_type=p['penalty_type'], penalty_yards=p['penalty_yards'],
                    penalty_status=p['penalty_status'],
                    end_down=p['end_down'], end_distance=p['end_distance'],
                    end_yards_to_endzone=p['end_yards_to_endzone'], end_team_id=p['end_team_id'],
                ))

        for player_id, splits in result['player_splits'].items():
            row = PlayerGameStat.query.filter_by(sport='NFL', game_id=event_id, player_id=player_id).first()
            if row is None:
                continue
            try:
                cats = _json.loads(row.stats_json) if row.stats_json else {}
            except (TypeError, ValueError):
                cats = {}
            if splits['rush_att']:
                cats['rushing_splits'] = {k: v for k, v in splits.items() if k.startswith('rush') or k.startswith('long_rush') or k.startswith('redzone_rush')}
            if splits['targets']:
                cats['receiving_splits'] = {k: v for k, v in splits.items()
                                             if k in ('targets', 'receptions', 'rec_yds', 'rec_td', 'yac',
                                                      'deep_tgt', 'deep_rec', 'deep_rec_yds',
                                                      'short_tgt', 'short_rec', 'short_rec_yds',
                                                      'redzone_tgt', 'redzone_rec_td')}
            if splits['pass_att'] or splits['sacks_taken']:
                cats['passing_splits'] = {k: v for k, v in splits.items()
                                           if k.startswith('pass_') or k.startswith('deep_pass') or
                                              k.startswith('short_pass') or k.startswith('redzone_pass') or
                                              k == 'sacks_taken'}
            row.stats_json = _json.dumps(cats)

        for team_id, allowed in result['team_allowed'].items():
            trow = TeamGameStat.query.filter_by(sport='NFL', game_id=event_id, team_id=team_id).first()
            if trow is None:
                continue
            trow.long_plays_allowed = allowed['long_plays_allowed']
            trow.redzone_trips_allowed = allowed['redzone_trips_allowed']
            trow.redzone_tds_allowed = allowed['redzone_tds_allowed']

        db.session.commit()


def ingest_weather(event_id, weather):
    """weather: nfl_api._parse_weather()'s output ({'temp', 'gust_mph',
    'precip_pct'} or None). Stamps it onto BOTH teams' EXISTING
    TeamGameStat rows for this game (match-only, same convention as
    ingest_play_by_play's TeamGameStat update — expects ingest_team_game()
    to have already created them). A no-op if `weather` is None (dome/
    retractable roof, or ESPN simply has nothing for this game right now)
    — never writes a row of zeros that would be indistinguishable from
    "not captured yet"."""
    if not weather:
        return
    from app import app as flask_app, db, TeamGameStat

    with flask_app.app_context():
        rows = TeamGameStat.query.filter_by(sport='NFL', game_id=event_id).all()
        for row in rows:
            row.weather_temp = weather.get('temp')
            row.weather_wind_mph = weather.get('gust_mph')
            row.weather_precip_pct = weather.get('precip_pct')
        db.session.commit()


def get_player_gamelog_db(player_id, before_date=None, limit=None):
    """Returns this player's stored games, newest first: [{event_id, date,
    season, team_id, opponent_id, is_home, position, categories}] where
    `categories` is the parsed stats_json dict (e.g. {'passing': {'passingYards':
    '299', ...}, ...}) — values come back as ESPN's own strings (e.g. '22/40'
    completions/attempts composites aren't split out here the way NBA's
    gamelog parsing splits made-attempt pairs; this is raw/exploratory data,
    not yet feeding a model that would need them pre-parsed). [] if nothing
    stored for this player yet."""
    from app import app as flask_app, PlayerGameStat

    with flask_app.app_context():
        q = PlayerGameStat.query.filter_by(sport='NFL', player_id=player_id)
        if before_date:
            q = q.filter(PlayerGameStat.game_date < before_date)
        q = q.order_by(PlayerGameStat.game_date.desc())
        if limit:
            q = q.limit(limit)
        rows = q.all()
        out = []
        for r in rows:
            try:
                categories = json.loads(r.stats_json) if r.stats_json else {}
            except (TypeError, ValueError):
                categories = {}
            out.append({
                'event_id':    r.game_id,
                'date':        r.game_date,
                'season':      r.season,
                'team_id':     r.team_id,
                'opponent_id': r.opponent_id,
                'is_home':     r.is_home,
                'position':    r.position,
                'categories':  categories,
            })
        return out


def get_player_usage_card_data_db(player_id, season, before_date=None,
                                   recent_window=3, history_limit=20):
    """Aggregates this player's rushing_splits/receiving_splits/
    passing_splits (nfl_playbyplay_api.py's output, merged into
    PlayerGameStat.stats_json by ingest_play_by_play()) into a recent-
    window vs. season-to-date usage read, same split get_dst_card_data_db()/
    get_opponent_defense_card_data_db() already give the other player-card
    sections — see those docstrings for why "season" means every stored
    `season` game rather than a plain limit (no cross-season leakage early
    in a new season).

    Built on top of get_player_gamelog_db() for the player's own games
    (one query, capped at `history_limit` — plenty for both a 3-game
    recent window and a 17-game regular season) plus ONE more query for
    this player's TEAMMATES' receiving_splits on those same games, needed
    only to compute target share (this player's targets / team's total
    targets that game) — two queries total, not one per metric.

    No `team_id` parameter — each game's own team is read from that same
    game's gamelog row (get_player_gamelog_db() already returns it),
    rather than one team_id applied to the whole window. A single fixed
    team_id was a confirmed, real bug for any player who changed teams
    mid-season: the "teammates" query would silently match 0 rows for
    every game played on the OLD team (since the new team's roster never
    appears in an old team's game), undercounting the target-share
    denominator while the player's own targets for those games were still
    counted in the numerator — produced target_share > 1.0 (caught live:
    a traded WR's season target_share came out as 3.1). Scoping each
    game's teammate lookup to THAT game's own team_id, derived from the
    player's own row, fixes it structurally instead of requiring every
    caller to pass the "right" team_id (there often isn't a single one
    for a traded player's whole window).

    Returns {'recent': {...} or None, 'season': {...} or None} — each side
    is {'games', 'rushing': {...} or None, 'receiving': {...} or None,
    'passing': {...} or None, 'target_share': float or None}. A rushing/
    receiving/passing sub-dict is None if the player never recorded that
    category in the window (e.g. a pure receiver has no 'passing'), not a
    dict of zeros, so a caller can't mistake "never attempted" for
    "attempted and got nothing." target_share is None without both a
    receiving line AND a nonzero team-targets denominator for the window.
    DISPLAY-ONLY context, same as the rest of this card — see
    project_points()'s docstring in nfl_fantasy.py for why this isn't
    (yet) a projection input."""
    games = get_player_gamelog_db(player_id, before_date=before_date, limit=history_limit)
    game_ids = [g['event_id'] for g in games]
    game_team = {g['event_id']: g['team_id'] for g in games}

    team_targets_by_game = defaultdict(float)
    if game_ids:
        from app import app as flask_app, PlayerGameStat
        with flask_app.app_context():
            team_rows = (PlayerGameStat.query
                         .filter(PlayerGameStat.sport == 'NFL',
                                 PlayerGameStat.game_id.in_(game_ids))
                         .all())
        for r in team_rows:
            # A game_id has rows from BOTH teams — only count the ones on
            # the SAME team the player was actually on for THIS game (see
            # docstring: that's not always one fixed team across the
            # whole window).
            if r.team_id != game_team.get(r.game_id):
                continue
            try:
                cats = json.loads(r.stats_json) if r.stats_json else {}
            except (TypeError, ValueError):
                cats = {}
            recv_splits = cats.get('receiving_splits')
            if recv_splits:
                team_targets_by_game[r.game_id] += recv_splits.get('targets', 0)

    def _summarize(window_games):
        n = len(window_games)
        if not n:
            return None
        totals = {'rushing': None, 'receiving': None, 'passing': None}
        team_targets = 0.0
        for g in window_games:
            cats = g['categories']
            for key, label in (('rushing_splits', 'rushing'), ('receiving_splits', 'receiving'),
                                ('passing_splits', 'passing')):
                splits = cats.get(key)
                if not splits:
                    continue
                if totals[label] is None:
                    totals[label] = defaultdict(float)
                for k, v in splits.items():
                    totals[label][k] += v
            if cats.get('receiving_splits'):
                team_targets += team_targets_by_game.get(g['event_id'], 0.0)
        receiving = dict(totals['receiving']) if totals['receiving'] else None
        target_share = (receiving['targets'] / team_targets) if receiving and team_targets else None
        return {
            'games': n,
            'rushing': dict(totals['rushing']) if totals['rushing'] else None,
            'receiving': receiving,
            'passing': dict(totals['passing']) if totals['passing'] else None,
            'target_share': target_share,
        }

    recent_games = games[:recent_window]
    season_games = [g for g in games if g['season'] == season] if season else []
    return {'recent': _summarize(recent_games), 'season': _summarize(season_games)}


def get_team_explosiveness_db(team_id, before_date=None, limit=8):
    """This team's defense's long_plays_allowed/redzone_trips_allowed/
    redzone_tds_allowed for its last `limit` games (newest first,
    point-in-time if `before_date` given). Returns a list of per-game
    dicts (empty list if nothing stored yet) — same "raw rows in, caller
    does the math" convention as get_defense_vs_position_db()."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        q = TeamGameStat.query.filter(TeamGameStat.sport == 'NFL', TeamGameStat.team_id == team_id,
                                       TeamGameStat.long_plays_allowed.isnot(None))
        if before_date:
            q = q.filter(TeamGameStat.game_date < before_date)
        rows = q.order_by(TeamGameStat.game_date.desc()).limit(limit).all()
        return [{
            'game_date': r.game_date,
            'long_plays_allowed': r.long_plays_allowed,
            'redzone_trips_allowed': r.redzone_trips_allowed,
            'redzone_tds_allowed': r.redzone_tds_allowed,
        } for r in rows]


def get_dst_card_data_db(team_id, season, before_date=None, recent_window=5):
    """Single-pass batched read for a D/ST player-detail card: this team's
    own game results (points allowed), explosiveness (long plays/red-zone
    trips/TDs allowed) and fantasy-points-allowed-by-position all live on
    TeamGameStat, so ONE query covers all of them; a second query covers
    the team's own box-score defensive production (sacks/TFL/takeaways/
    pass-defended), grouped into per-game aggregates the same way
    get_team_game_aggregates_db() does. Two queries total, not one per
    metric — a caller that queried each metric separately (explosiveness,
    fpts-by-bucket x3, results, defense aggregates, x2 for recent+season)
    was confirmed live to exhaust this app's small SQLAlchemy connection
    pool when the background cache-warmer thread was also busy (a
    QueuePool timeout) — same lesson get_fantasy_pool_data_db()'s
    docstring documents for the weekly pool build.

    Returns {'recent': {...per-game averages} or None, 'season': {...} or
    None} — 'recent' = this team's last `recent_window` stored games
    regardless of season (matches the rest of this card's "last N games"
    convention); 'season' = every stored game from `season` only (so an
    early-season read doesn't quietly pull in last year's games) — None
    for either side with zero games in that window, never a dict of
    zeros, so the caller can't mistake "no games" for "allowed nothing."
    Each averaged dict also reports its own denominators ('games',
    'redzone_trips_allowed', etc.) so a small sample is visible, not
    hidden behind a smooth-looking average."""
    import nfl_player_model as pm
    from app import app as flask_app, TeamGameStat, PlayerGameStat

    with flask_app.app_context():
        tq = TeamGameStat.query.filter_by(sport='NFL', team_id=team_id)
        if before_date:
            tq = tq.filter(TeamGameStat.game_date < before_date)
        team_rows = tq.order_by(TeamGameStat.game_date.desc()).all()

        pq = PlayerGameStat.query.filter_by(sport='NFL', team_id=team_id)
        if before_date:
            pq = pq.filter(PlayerGameStat.game_date < before_date)
        player_rows = pq.order_by(PlayerGameStat.game_date.desc()).all()

    by_game = {}
    game_order = []
    for r in player_rows:
        if r.game_id not in by_game:
            by_game[r.game_id] = []
            game_order.append(r.game_id)
        try:
            categories = json.loads(r.stats_json) if r.stats_json else {}
        except (TypeError, ValueError):
            categories = {}
        by_game[r.game_id].append({'categories': categories})
    defense_by_game = {}
    for game_id in game_order:
        _, defn = pm.parse_team_game_aggregate(by_game[game_id])
        defense_by_game[game_id] = defn

    def _summarize(rows):
        games = len(rows)
        if not games:
            return None
        rz_rows = [r for r in rows if r.long_plays_allowed is not None]
        n_rz = len(rz_rows)
        redzone_trips = sum(r.redzone_trips_allowed or 0 for r in rz_rows)
        redzone_tds = sum(r.redzone_tds_allowed or 0 for r in rz_rows)
        fpts_qb = [r.fpts_allowed_qb for r in rows if r.fpts_allowed_qb is not None]
        fpts_rb = [r.fpts_allowed_rb for r in rows if r.fpts_allowed_rb is not None]
        fpts_rec = [r.fpts_allowed_rec for r in rows if r.fpts_allowed_rec is not None]
        defn_games = [defense_by_game[r.game_id] for r in rows if r.game_id in defense_by_game]
        n_def = len(defn_games)
        return {
            'games': games,
            'points_allowed_pg': sum(r.points_allowed or 0 for r in rows) / games,
            'sacks_pg': sum(d['sacks'] for d in defn_games) / n_def if n_def else None,
            'tfl_pg': sum(d['tfl'] for d in defn_games) / n_def if n_def else None,
            'takeaways_pg': sum(d['takeaways'] for d in defn_games) / n_def if n_def else None,
            'pass_defended_pg': sum(d['pass_defended'] for d in defn_games) / n_def if n_def else None,
            'redzone_games': n_rz,
            'redzone_trips_allowed': redzone_trips,
            'redzone_tds_allowed': redzone_tds,
            'redzone_td_rate': redzone_tds / redzone_trips if redzone_trips else None,
            'long_plays_allowed_pg': sum(r.long_plays_allowed or 0 for r in rz_rows) / n_rz if n_rz else None,
            'fpts_allowed_qb_pg': sum(fpts_qb) / len(fpts_qb) if fpts_qb else None,
            'fpts_allowed_rb_pg': sum(fpts_rb) / len(fpts_rb) if fpts_rb else None,
            'fpts_allowed_rec_pg': sum(fpts_rec) / len(fpts_rec) if fpts_rec else None,
        }

    recent_rows = team_rows[:recent_window]
    season_rows = [r for r in team_rows if r.season == season] if season else []
    return {'recent': _summarize(recent_rows), 'season': _summarize(season_rows)}


def get_opponent_defense_card_data_db(team_id, season, before_date=None, recent_window=5):
    """Single-pass batched read for the opponent-defense section of an
    OFFENSIVE player's card: red-zone/explosive-play allowed (one
    TeamGameStat query, same columns get_dst_card_data_db() reads) plus
    rushing/receiving/passing yards and TDs allowed broken out by the
    opposing player's own position BUCKET (one NflPlay query + one batched
    PlayerGameStat lookup to classify each participant) — three queries
    total, not one per metric, same pool-exhaustion lesson documented on
    get_dst_card_data_db().

    bucket: 'QB' | 'RB' | 'REC' (WR+TE combined — see classify_bucket()'s
    docstring; ESPN's box score carries no position field, so a true WR/TE
    split isn't reliable from stored data). A participant's bucket is
    decided from THEIR OWN categories in THAT SAME game (same convention
    fpts_allowed_qb/rb/rec already uses) — a play with an unresolved
    participant (nfl_playbyplay_api.py's documented attribution gap) or an
    unclassifiable bucket (None) is simply not counted toward any bucket,
    not guessed.

    A TD/yardage figure is credited to EVERY position it's real for — a
    passing TD play adds to both the passer's QB line (passing yards/TD)
    and the receiver's RB/REC line (receiving yards/TD), mirroring how
    fpts_allowed_qb and fpts_allowed_rec both already count the same play
    from each side's own fantasy scoring.

    Returns {'recent': {...} or None, 'season': {...} or None} — each side
    is {'games', 'redzone_trips_allowed', 'redzone_tds_allowed',
    'redzone_td_rate', 'long_plays_allowed_pg', 'QB': {'pass_yds',
    'pass_td'}, 'RB': {'rush_yds', 'rush_td', 'rec_yds', 'rec_td',
    'receptions'}, 'REC': {'rec_yds', 'rec_td', 'receptions'}} — None for
    either side with zero stored games in that window."""
    import nfl_fantasy
    from app import app as flask_app, TeamGameStat, NflPlay, PlayerGameStat

    with flask_app.app_context():
        tq = TeamGameStat.query.filter(TeamGameStat.sport == 'NFL', TeamGameStat.team_id == team_id,
                                        TeamGameStat.long_plays_allowed.isnot(None))
        if before_date:
            tq = tq.filter(TeamGameStat.game_date < before_date)
        team_rows = tq.order_by(TeamGameStat.game_date.desc()).all()

        pq = NflPlay.query.filter_by(defense_team_id=team_id)
        if before_date:
            pq = pq.filter(NflPlay.game_date < before_date)
        plays = pq.order_by(NflPlay.game_date.desc()).all()

        game_ids = {p.game_id for p in plays}
        pgs_rows = (PlayerGameStat.query.filter(PlayerGameStat.sport == 'NFL',
                                                  PlayerGameStat.game_id.in_(game_ids)).all()
                    if game_ids else [])
        cats_by_player_game = {}
        for r in pgs_rows:
            try:
                cats_by_player_game[(r.player_id, r.game_id)] = json.loads(r.stats_json) if r.stats_json else {}
            except (TypeError, ValueError):
                cats_by_player_game[(r.player_id, r.game_id)] = {}

    bucket_cache = {}
    def _bucket(player_id, game_id):
        key = (player_id, game_id)
        if key not in bucket_cache:
            bucket_cache[key] = nfl_fantasy.classify_bucket(cats_by_player_game.get(key))
        return bucket_cache[key]

    def _zero_bucket_stats():
        return {'rush_yds': 0.0, 'rush_td': 0, 'rec_yds': 0.0, 'rec_td': 0,
                'receptions': 0, 'pass_yds': 0.0, 'pass_td': 0}

    def _summarize(rows, by_game):
        games = len(rows)
        if not games:
            return None
        redzone_trips = sum(r.redzone_trips_allowed or 0 for r in rows)
        redzone_tds = sum(r.redzone_tds_allowed or 0 for r in rows)
        stats = {'QB': _zero_bucket_stats(), 'RB': _zero_bucket_stats(), 'REC': _zero_bucket_stats()}
        game_ids_in_window = {r.game_id for r in rows}
        for p in by_game:
            if p.game_id not in game_ids_in_window or p.is_no_play:
                continue
            if p.rusher_id:
                b = _bucket(p.rusher_id, p.game_id)
                if b in stats:
                    stats[b]['rush_yds'] += p.yards_gained or 0
                    if p.is_scoring_play and p.play_type == 'Rushing Touchdown':
                        stats[b]['rush_td'] += 1
            if p.receiver_id and p.play_type in ('Pass Reception', 'Passing Touchdown'):
                b = _bucket(p.receiver_id, p.game_id)
                if b in stats:
                    stats[b]['rec_yds'] += p.yards_gained or 0
                    stats[b]['receptions'] += 1
                    if p.is_scoring_play and p.play_type == 'Passing Touchdown':
                        stats[b]['rec_td'] += 1
            if p.passer_id and p.play_type in ('Pass Reception', 'Pass Incompletion', 'Passing Touchdown'):
                b = _bucket(p.passer_id, p.game_id)
                if b in stats:
                    stats[b]['pass_yds'] += p.yards_gained or 0
                    if p.is_scoring_play and p.play_type == 'Passing Touchdown':
                        stats[b]['pass_td'] += 1
        return {
            'games': games,
            'redzone_trips_allowed': redzone_trips,
            'redzone_tds_allowed': redzone_tds,
            'redzone_td_rate': redzone_tds / redzone_trips if redzone_trips else None,
            'long_plays_allowed_pg': sum(r.long_plays_allowed or 0 for r in rows) / games,
            'QB': stats['QB'], 'RB': stats['RB'], 'REC': stats['REC'],
        }

    recent_rows = team_rows[:recent_window]
    season_rows = [r for r in team_rows if r.season == season] if season else []
    return {'recent': _summarize(recent_rows, plays), 'season': _summarize(season_rows, plays)}


def get_nfl_league_defense_profile_db(season, recent_window=5):
    """Every NFL team's own per-game defensive averages in one pass, for
    comparing one team's card against the league rather than just its own
    recent-vs-season trend — the league side of the "is this matchup
    favorable" question get_dst_card_data_db()/get_opponent_defense_card_
    data_db() can't answer on their own (those two only ever look at a
    single team_id). Three queries total covering every team at once
    (not one query per team x 32), same pool-exhaustion lesson as those
    two functions' docstrings.

    Returns {team_id: {...}} — one entry per team with >=1 stored game
    this season, each a dict merging: 'games' (season sample size),
    'points_allowed_pg', 'fpts_allowed_qb_pg', 'fpts_allowed_rb_pg',
    'fpts_allowed_rec_pg', 'long_plays_allowed_pg', 'redzone_td_rate'
    (from TeamGameStat — recent_window games, regardless of season, same
    "recent" convention as the single-team card functions) and
    'sacks_pg', 'tfl_pg', 'takeaways_pg', 'pass_defended_pg' (from
    PlayerGameStat, season-to-date only — box-score defensive stats
    aren't on TeamGameStat, and a 32-team recent-window version of the
    per-game PlayerGameStat regroup below would be a second full scan for
    a window the headline metrics above already use consistently).
    A team missing a given underlying metric simply omits that key
    (never a zero), same "don't fake a sample that isn't there"
    convention as the single-team functions."""
    import nfl_player_model as pm
    from app import app as flask_app, TeamGameStat, PlayerGameStat

    with flask_app.app_context():
        team_rows = (TeamGameStat.query.filter_by(sport='NFL')
                     .order_by(TeamGameStat.game_date.desc()).all())
        player_rows = (PlayerGameStat.query.filter_by(sport='NFL', season=season).all()
                       if season else [])

    by_team = defaultdict(list)
    for r in team_rows:
        by_team[r.team_id].append(r)

    profile = {}
    for team_id, rows in by_team.items():
        recent_rows = rows[:recent_window]
        n = len(recent_rows)
        if not n:
            continue
        rz_rows = [r for r in recent_rows if r.long_plays_allowed is not None]
        n_rz = len(rz_rows)
        redzone_trips = sum(r.redzone_trips_allowed or 0 for r in rz_rows)
        redzone_tds = sum(r.redzone_tds_allowed or 0 for r in rz_rows)
        fpts_qb = [r.fpts_allowed_qb for r in recent_rows if r.fpts_allowed_qb is not None]
        fpts_rb = [r.fpts_allowed_rb for r in recent_rows if r.fpts_allowed_rb is not None]
        fpts_rec = [r.fpts_allowed_rec for r in recent_rows if r.fpts_allowed_rec is not None]
        entry = {'games': n, 'points_allowed_pg': sum(r.points_allowed or 0 for r in recent_rows) / n}
        if fpts_qb:
            entry['fpts_allowed_qb_pg'] = sum(fpts_qb) / len(fpts_qb)
        if fpts_rb:
            entry['fpts_allowed_rb_pg'] = sum(fpts_rb) / len(fpts_rb)
        if fpts_rec:
            entry['fpts_allowed_rec_pg'] = sum(fpts_rec) / len(fpts_rec)
        if n_rz:
            entry['long_plays_allowed_pg'] = sum(r.long_plays_allowed or 0 for r in rz_rows) / n_rz
            if redzone_trips:
                entry['redzone_td_rate'] = redzone_tds / redzone_trips
        profile[team_id] = entry

    by_team_game = defaultdict(list)
    for r in player_rows:
        try:
            categories = json.loads(r.stats_json) if r.stats_json else {}
        except (TypeError, ValueError):
            categories = {}
        by_team_game[(r.team_id, r.game_id)].append({'categories': categories})

    pressure_rows = defaultdict(list)
    for (team_id, _game_id), game_rows in by_team_game.items():
        _, defn = pm.parse_team_game_aggregate(game_rows)
        pressure_rows[team_id].append(defn)

    for team_id, defn_games in pressure_rows.items():
        n_def = len(defn_games)
        if not n_def:
            continue
        entry = profile.setdefault(team_id, {'games': 0})
        entry['sacks_pg'] = sum(d['sacks'] for d in defn_games) / n_def
        entry['tfl_pg'] = sum(d['tfl'] for d in defn_games) / n_def
        entry['takeaways_pg'] = sum(d['takeaways'] for d in defn_games) / n_def
        entry['pass_defended_pg'] = sum(d['pass_defended'] for d in defn_games) / n_def

    return profile


def get_team_game_aggregates_db(team_id, before_date=None, limit=5):
    """Returns (offense_list, defense_list) — up to `limit` of this team's
    most recent games, newest first, each already collapsed to one team-
    level aggregate dict via nfl_player_model.parse_team_game_aggregate().
    This is what nfl_api.py's live predictions pass straight into
    nfl_player_model.compute_offense_grade()/compute_defense_grade().

    Groups the team's stored PlayerGameStat rows (one row per player per
    game) back into per-game buckets by game_id — a Final game writes many
    player rows that all share the same game_id/game_date, so this just
    re-assembles the team's own side of that game from them. ([], []) if
    nothing stored for this team yet."""
    import nfl_player_model as pm
    from app import app as flask_app, PlayerGameStat

    with flask_app.app_context():
        q = PlayerGameStat.query.filter_by(sport='NFL', team_id=team_id)
        if before_date:
            q = q.filter(PlayerGameStat.game_date < before_date)
        q = q.order_by(PlayerGameStat.game_date.desc())
        rows = q.all()

    by_game = {}
    game_order = []
    for r in rows:
        if r.game_id not in by_game:
            by_game[r.game_id] = []
            game_order.append(r.game_id)
        try:
            categories = json.loads(r.stats_json) if r.stats_json else {}
        except (TypeError, ValueError):
            categories = {}
        by_game[r.game_id].append({'categories': categories})

    offense_list, defense_list = [], []
    for game_id in game_order[:limit]:
        off, defn = pm.parse_team_game_aggregate(by_game[game_id])
        offense_list.append(off)
        defense_list.append(defn)
    return offense_list, defense_list


def get_team_games_with_players_db(team_id, before_date=None, limit=5):
    """Returns this team's most recent games, newest first, WITHOUT
    collapsing player identity the way get_team_game_aggregates_db() above
    does — each game is [{player_id, player_name, categories}, ...], so a
    specific player's own contribution to that game can still be isolated
    and subtracted back out (see nfl_player_model.identify_key_players()/
    compute_offense_grade_excluding()). ([]) if nothing stored yet."""
    from app import app as flask_app, PlayerGameStat

    with flask_app.app_context():
        q = PlayerGameStat.query.filter_by(sport='NFL', team_id=team_id)
        if before_date:
            q = q.filter(PlayerGameStat.game_date < before_date)
        q = q.order_by(PlayerGameStat.game_date.desc())
        rows = q.all()

    by_game = {}
    game_order = []
    for r in rows:
        if r.game_id not in by_game:
            by_game[r.game_id] = []
            game_order.append(r.game_id)
        try:
            categories = json.loads(r.stats_json) if r.stats_json else {}
        except (TypeError, ValueError):
            categories = {}
        by_game[r.game_id].append({
            'player_id': r.player_id, 'player_name': r.player_name, 'categories': categories,
        })

    return [by_game[gid] for gid in game_order[:limit]]


def get_fantasy_pool_data_db(team_ids, player_ids, as_of_date, before_date=None):
    """Batched read for nfl_fantasy.build_player_pool(): everything it
    needs for a whole week's worth of teams/players in exactly THREE
    queries (one InjuryStatus, one TeamGameStat, one PlayerGameStat) inside
    a single app context, instead of the dozens-to-hundreds of tiny
    single-row queries a naive per-player/per-team loop would issue (one
    `with flask_app.app_context()` per call adds up fast across a full
    slate's roster, and was confirmed live to exhaust this app's small
    SQLAlchemy connection pool when the background cache-warmer thread is
    also busy).

    `before_date` (optional, YYYY-MM-DD): when given, restricts the
    TeamGameStat/PlayerGameStat reads to games strictly before it — the
    no-leakage cutoff nfl_recap.py needs to reconstruct what a PAST week's
    pool would have looked like using only data that existed before that
    week started. None (the default, what the live /nfl/lineup page uses)
    means "everything stored so far," identical to this function's
    original behavior.

    Returns {
      'unavailable':   {team_id: {player_id, ...}},
      'player_games':  {player_id: [{categories}, ...]},  # newest first
      'team_games':    {team_id: [{points_allowed, player_rows: [{categories}, ...]}, ...]},
                        # newest first, capped to the last `window` games —
                        # one entry per game, player_rows being that team's
                        # own players' box-score rows for it. Shape doubles
                        # as both nfl_player_model.parse_team_game_aggregate's
                        # input (via player_rows) and nfl_fantasy.
                        # aggregate_dst_game's (ditto, plus points_allowed).
    }
    All grouping/aggregation beyond this (rolling averages, offense/
    defense grades, DST scoring) is pure-Python work the caller does with
    nfl_player_model/nfl_fantasy — no further DB calls needed.

    PERFORMANCE: both the per-player and per-team PlayerGameStat reads are
    bounded to the `window` games each actually needs — neither
    rolling_average() (nfl_fantasy.py) nor compute_offense_grade()/
    compute_defense_grade() (nfl_player_model.py) ever look past the first
    `window` entries of what this returns. Confirmed live this was NOT the
    case before: the per-player query had no cap at all, pulling a
    player's ENTIRE stored history (every season on record) just to use
    the first 5 rows — profiled on a 300+ player pool at ~44K PlayerGameStat
    rows fetched and JSON-decoded for ~1,500 actually used, ~2.4s of a
    ~2.6s build_player_pool() call. The per-player cap below uses a
    ROW_NUMBER() OVER (PARTITION BY player_id ...) window function (SQLite
    3.25+, confirmed available) so "last `window` games per player" is
    computed in SQL instead of fetched-then-discarded in Python. The
    per-team read is bounded the same way as before (via team_rows' own
    already-capped recent_game_ids) but now filters directly by those
    game_ids instead of first pulling a team's full roster history."""
    from sqlalchemy import func
    from sqlalchemy.orm import aliased
    from app import app as flask_app, db, InjuryStatus, TeamGameStat, PlayerGameStat

    window = 5  # matches nfl_player_model.QB_WINDOW/DEF_WINDOW and nfl_fantasy.ROLLING_WINDOW
    team_ids = list(team_ids)
    player_ids = list(player_ids)

    with flask_app.app_context():
        inj_rows = (InjuryStatus.query
                    .filter(InjuryStatus.sport == 'NFL', InjuryStatus.team_id.in_(team_ids),
                            InjuryStatus.date <= as_of_date)
                    .order_by(InjuryStatus.date.desc())
                    .all()) if team_ids else []

        team_q = TeamGameStat.query.filter(TeamGameStat.sport == 'NFL', TeamGameStat.team_id.in_(team_ids))
        if before_date:
            team_q = team_q.filter(TeamGameStat.game_date < before_date)
        team_rows = team_q.order_by(TeamGameStat.game_date.desc()).all() if team_ids else []

        # Each team's most recent `window` game_ids, in order — TeamGameStat
        # has exactly one row per (team, game), already date-desc, so this is
        # just a straight slice per team. Computed here (before the
        # per-player box-score reads below) so the team-roster read can
        # filter directly to these game_ids instead of pulling full history.
        recent_game_ids = defaultdict(list)
        points_allowed_by_game = {}
        for r in team_rows:
            if len(recent_game_ids[r.team_id]) < window:
                recent_game_ids[r.team_id].append(r.game_id)
            points_allowed_by_game[(r.team_id, r.game_id)] = r.points_allowed
        all_recent_game_ids = {gid for gids in recent_game_ids.values() for gid in gids}

        player_rows = []
        if player_ids:
            rn = func.row_number().over(
                partition_by=PlayerGameStat.player_id,
                order_by=PlayerGameStat.game_date.desc(),
            ).label('rn')
            ranked_q = db.session.query(PlayerGameStat, rn).filter(
                PlayerGameStat.sport == 'NFL', PlayerGameStat.player_id.in_(player_ids))
            if before_date:
                ranked_q = ranked_q.filter(PlayerGameStat.game_date < before_date)
            ranked_sub = ranked_q.subquery()
            PGSRanked = aliased(PlayerGameStat, ranked_sub)
            player_rows = (db.session.query(PGSRanked)
                            .filter(ranked_sub.c.rn <= window)
                            .order_by(ranked_sub.c.rn)
                            .all())

        team_player_rows = []
        if team_ids and all_recent_game_ids:
            team_player_rows = (PlayerGameStat.query
                                 .filter(PlayerGameStat.sport == 'NFL',
                                         PlayerGameStat.team_id.in_(team_ids),
                                         PlayerGameStat.game_id.in_(all_recent_game_ids))
                                 .all())

    # Latest-per-player injury status, same "first row seen wins" rule as
    # get_latest_injury_status (date-desc order), bucketed by team.
    unavailable = defaultdict(set)
    seen_player_status = {}
    for r in inj_rows:
        if r.player_id in seen_player_status:
            continue
        seen_player_status[r.player_id] = r.status
        if r.status.lower() in _OUT_STATUSES:
            unavailable[r.team_id].add(r.player_id)

    def _cats(stats_json):
        try:
            return json.loads(stats_json) if stats_json else {}
        except (TypeError, ValueError):
            return {}

    player_games = defaultdict(list)
    for r in player_rows:
        player_games[r.player_id].append({'categories': _cats(r.stats_json)})

    rows_by_team_game = defaultdict(list)
    for r in team_player_rows:
        rows_by_team_game[(r.team_id, r.game_id)].append({'categories': _cats(r.stats_json)})

    team_games = {}
    for team_id, game_ids in recent_game_ids.items():
        team_games[team_id] = [{
            'points_allowed': points_allowed_by_game.get((team_id, gid)),
            'player_rows': rows_by_team_game.get((team_id, gid), []),
        } for gid in game_ids]

    return {
        'unavailable': dict(unavailable),
        'player_games': dict(player_games),
        'team_games': team_games,
    }


def get_schedule_db(season, game_type='regular'):
    """Returns every stored game for `season` as [{event_id, game_date,
    home_id, away_id, home_score, away_score}], chronological — the DB
    equivalent of nfl_player_bootstrap.py's old live ESPN schedule fetch.
    Same approach as nba_stats_db.get_schedule_db()."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query
                .filter_by(sport='NFL', season=season, game_type=game_type, is_home=True)
                .order_by(TeamGameStat.game_date.asc())
                .all())
        return [{
            'event_id':   r.game_id,
            'game_date':  r.game_date,
            'home_id':    r.team_id,
            'away_id':    r.opponent_id,
            'home_score': r.points,
            'away_score': r.points_allowed,
        } for r in rows]


def get_game_boxscore_db(game_id):
    """Returns {team_id: [{id, name, categories}]} for one game — the DB
    equivalent of nfl_player_bootstrap.py's old live ESPN per-game summary
    fetch (nfl_boxscore_api.get_live_boxscore). {} if nothing stored."""
    from app import app as flask_app, PlayerGameStat

    with flask_app.app_context():
        rows = PlayerGameStat.query.filter_by(sport='NFL', game_id=game_id).all()

    out = {}
    for r in rows:
        try:
            categories = json.loads(r.stats_json) if r.stats_json else {}
        except (TypeError, ValueError):
            categories = {}
        out.setdefault(r.team_id, []).append({
            'id': r.player_id, 'name': r.player_name, 'categories': categories,
        })
    return out


def ingest_injury_report(team_id, injuries, date_et):
    """Persists one team's current injury list (as returned by
    nfl_api._parse_injuries()'s per-team value: [{id, name, pos, status}])
    for `date_et` (YYYY-MM-DD). Upserts on (sport, date, player_id) — a
    same-day re-ingestion (e.g. status changes from Questionable to Out
    later in the week) updates the existing row in place rather than
    duplicating it; a new date starts a new row, which is what actually
    builds up real injury HISTORY over time (see InjuryStatus's docstring
    in app.py for why this matters — it's new data this app never kept
    before). Skips entries with no id (shouldn't happen — confirmed live
    that ESPN's injury entries always carry one — but defensive either way
    since this is a write path)."""
    if not injuries:
        return
    from app import app as flask_app, db, InjuryStatus

    with flask_app.app_context():
        for inj in injuries:
            pid = inj.get('id')
            if not pid:
                continue
            existing = InjuryStatus.query.filter_by(sport='NFL', date=date_et, player_id=pid).first()
            row = existing or InjuryStatus(sport='NFL', date=date_et, player_id=pid)
            row.team_id     = team_id
            row.player_name = inj.get('name', '')
            row.position    = inj.get('pos', '')
            row.status      = inj.get('status', '')
            if not existing:
                db.session.add(row)
        db.session.commit()


def has_injury_snapshot(team_id, date_et):
    """True if a snapshot for team_id already exists for EXACTLY date_et —
    used only to decide whether nfl_api.py needs to ingest today's fetch at
    all (skip re-writing identical rows on every page render the same
    day). Not what prediction logic should call — see
    get_latest_injury_status() below for that."""
    from app import app as flask_app, InjuryStatus

    with flask_app.app_context():
        return InjuryStatus.query.filter_by(sport='NFL', team_id=team_id, date=date_et).first() is not None


def get_latest_injury_status(team_id, as_of_date=None):
    """Returns {player_id: status} — each player's MOST RECENTLY RECORDED
    status on or before `as_of_date` (default: no cutoff, i.e. the latest
    snapshot available at all). This is deliberately NOT an exact-date
    match: ESPN's injury report reflects current-state-as-of-today
    regardless of when the game being projected actually kicks off, so a
    game a few days out should still read today's (or the most recent)
    snapshot, not wait for a snapshot dated on the game's own future date
    that will never exist until that day arrives. {} if nothing stored for
    this team yet (caller falls back to a live fetch)."""
    from app import app as flask_app, InjuryStatus

    with flask_app.app_context():
        q = InjuryStatus.query.filter_by(sport='NFL', team_id=team_id)
        if as_of_date:
            q = q.filter(InjuryStatus.date <= as_of_date)
        rows = q.order_by(InjuryStatus.date.desc()).all()

    latest = {}
    for r in rows:  # date-desc, so the first row seen per player is its latest
        if r.player_id not in latest:
            latest[r.player_id] = r.status
    return latest


_OUT_STATUSES = {'out', 'injured reserve', 'suspension', 'suspended'}


def get_unavailable_player_ids(team_id, date_et):
    """Subset of get_latest_injury_status() whose status means the player
    is confirmed not playing — same OUT-statuses convention as
    nba_boxscore_api._OUT_STATUSES (Day-To-Day/Questionable/Doubtful are
    left in: project them rather than silently zero them out, same
    reasoning as nba_roster_api.get_active_roster)."""
    statuses = get_latest_injury_status(team_id, date_et)
    return {pid for pid, status in statuses.items() if status.lower() in _OUT_STATUSES}


def _normalize_headline(headline):
    """Lowercase, punctuation-stripped, whitespace-collapsed — for
    comparing headlines across domains/URLs that are really the same
    syndicated wire story (confirmed live: GDELT backfill coverage is
    ~42% exact-headline duplicates across different domains, e.g. the
    same AP-style story republished on a dozen local-radio-affiliate
    sites). Not meant to catch loosely-similar headlines, just the
    "literally the same story, different URL" case."""
    return ' '.join(re.sub(r'[^a-z0-9 ]', '', (headline or '').lower()).split())


def ingest_news_sentiment(source, url, headline, raw_snippet, published_at, date_et,
                           matches, score, matched_terms, dedupe_by_headline=False):
    """Persists one article's sentiment score once per matched player/team
    (see nfl_news.match_players() for `matches`' shape: [{'player_id',
    'team_id', 'player_name'}, ...] — a team-only mention has
    player_id=None). Skips any (source, url, player_id) combo already
    stored, so re-running nfl_news.py's ingestion over an overlapping
    lookback window never duplicates a row — same idempotence goal as
    ingest_injury_report above, just event-level instead of daily-upsert
    (see NewsSentiment's docstring in app.py for why). `matched_terms`:
    [(phrase, weight), ...] from news_sentiment.score_text(), stored as
    JSON purely so a score can be audited back to the lexicon entries that
    produced it.

    `dedupe_by_headline` (default False — the live nfl_news.py pipeline
    doesn't need it, confirmed zero duplicate headlines there since it
    pulls directly from ESPN/CBS's own feeds): also skips a match if an
    existing row for the SAME (source, player_id-or-team_id) already has
    the same normalized headline — catches syndicated reposts of one
    story across multiple URLs/domains, which a plain (source, url,
    player_id) key can't, since each repost is a genuinely distinct URL.
    nfl_news_backfill.py (GDELT, where this problem is real) passes True.

    Returns the number of rows actually inserted."""
    if not matches:
        return 0
    from app import app as flask_app, db, NewsSentiment

    inserted = 0
    with flask_app.app_context():
        norm_headline = _normalize_headline(headline) if dedupe_by_headline else None
        for m in matches:
            pid, tid = m.get('player_id'), m.get('team_id')
            existing = NewsSentiment.query.filter_by(source=source, url=url, player_id=pid).first()
            if existing:
                continue
            if dedupe_by_headline:
                dup_scope = (NewsSentiment.query.filter_by(source=source, player_id=pid) if pid is not None
                             else NewsSentiment.query.filter_by(source=source, player_id=None, team_id=tid))
                if any(_normalize_headline(r.headline) == norm_headline for r in dup_scope.all()):
                    continue
            db.session.add(NewsSentiment(
                sport='NFL', date=date_et, published_at=published_at,
                team_id=tid, player_id=pid, player_name=m.get('player_name'),
                source=source, headline=headline, url=url, raw_snippet=raw_snippet,
                sentiment_score=score, matched_terms_json=json.dumps(matched_terms),
            ))
            inserted += 1
        db.session.commit()
    return inserted


def get_daily_sentiment(player_id=None, team_id=None, start=None, end=None, sport='NFL', only_matched=True):
    """Returns [{'date', 'mean_score', 'article_count'}], one row per
    calendar date that has at least one matched article, aggregated over
    every NewsSentiment row for this player (or team, for team-level
    mentions) in [start, end]. Raw per-article rows (get_player_articles
    below) stay the source of truth — this is the "one number per day"
    shape nfl_news_backtest.py and nfl_fantasy.surprise_play_signal() want
    without re-aggregating themselves. [] if nothing matched yet.

    `only_matched` (default True): skip articles where CORE_LEXICON/the
    sport's overlay didn't actually recognize any phrase (matched_terms_
    json == '[]', score exactly 0.0 — news_sentiment.score_text()'s
    guaranteed shape for "nothing fired", not a judgment call). Confirmed
    live this matters a lot: most matched articles are routine team news
    (signings, roster moves, historical honors) the lexicon has no terms
    for at all, not genuinely neutral injury/performance coverage — averaging
    those in as literal 0.0s dilutes every player's score toward zero
    rather than leaving it unmeasured, which is the wrong failure mode.
    Pass False to see the raw average including unscored articles (e.g.
    auditing ingestion coverage, not judging sentiment)."""
    from app import app as flask_app, NewsSentiment

    with flask_app.app_context():
        q = NewsSentiment.query.filter_by(sport=sport)
        if player_id:
            q = q.filter_by(player_id=player_id)
        if team_id:
            q = q.filter_by(team_id=team_id)
        if start:
            q = q.filter(NewsSentiment.date >= start)
        if end:
            q = q.filter(NewsSentiment.date <= end)
        if only_matched:
            q = q.filter(NewsSentiment.matched_terms_json != '[]')
        rows = q.order_by(NewsSentiment.date.asc()).all()
        scores_by_date = defaultdict(list)
        for r in rows:
            scores_by_date[r.date].append(r.sentiment_score)

    return [{'date': d, 'mean_score': sum(scores) / len(scores), 'article_count': len(scores)}
            for d, scores in sorted(scores_by_date.items())]


def get_player_articles(player_id=None, team_id=None, start=None, end=None, limit=20, sport='NFL', only_matched=True):
    """Returns the most recently published matched articles, newest first:
    [{'date', 'published_at', 'headline', 'url', 'source', 'sentiment_score',
    'matched_terms'}]. This is the raw listing the NFL player-detail view
    (/nfl/player/<id>) reads directly, as opposed to get_daily_sentiment()'s
    aggregated numbers. `start`/`end` (optional, 'YYYY-MM-DD', inclusive):
    scopes to a specific week's article window instead of just "most
    recent" — the detail view needs this to make sense for a past/future
    week, not only whatever's most recent right now. `only_matched`
    (default True): see get_daily_sentiment()'s docstring — same filter,
    same reasoning (most matched-to-a-player articles are routine team
    news the lexicon has no terms for at all, not neutral coverage)."""
    from app import app as flask_app, NewsSentiment

    with flask_app.app_context():
        q = NewsSentiment.query.filter_by(sport=sport)
        if player_id:
            q = q.filter_by(player_id=player_id)
        if team_id:
            q = q.filter_by(team_id=team_id)
        if start:
            q = q.filter(NewsSentiment.date >= start)
        if end:
            q = q.filter(NewsSentiment.date <= end)
        if only_matched:
            q = q.filter(NewsSentiment.matched_terms_json != '[]')
        rows = (q.order_by(NewsSentiment.date.desc(), NewsSentiment.created_at.desc())
                .limit(limit).all())
        return [{
            'date': r.date, 'published_at': r.published_at, 'headline': r.headline, 'url': r.url,
            'source': r.source, 'sentiment_score': r.sentiment_score,
            'matched_terms': json.loads(r.matched_terms_json or '[]'),
        } for r in rows]


def get_simulator_play_pool_db(seasons=None):
    """One flat list of usable scrimmage plays for nfl_game_simulator.py's
    bootstrap play sampler: [{down, distance, yards_to_endzone,
    play_type, yards_gained, is_turnover, is_scoring_play}], oldest
    filtering decisions baked in here rather than left to the caller:

    - Drops Penalty plays and any is_no_play row (down replayed) — neither
      carries a clean down/distance/yards signal a drive simulator can
      chain on.
    - Includes the turnover-RESULT types that occur WITHIN a normal down
      1-4 scrimmage situation (Pass Interception Return, Fumble Recovery
      (Own)/(Opponent), Sack Opp Fumble Recovery) — simulate_drive()
      checks is_turnover before anything else, so these slot straight
      into the existing down/distance bucket sampling with no new
      mechanics needed. Still excludes Punt/Kickoff/Field Goal Good/
      Missed — those aren't situated on a normal down (kickoffs are
      down=0; a punt/FG's own down/distance describes the decision TO
      kick, not a resampleable "what happens on this down" outcome); see
      get_fourth_down_pool_db()/get_kickoff_pool_db() for those.
    - `end_yards_to_endzone` is the real ESPN-computed field position
      after the play, already in the new-possessor's own reference frame
      (confirmed live — see NflPlay.end_down's column comment, app.py) —
      used by simulate_game() to hand the next drive real field position
      after a turnover instead of resetting to a fixed spot.
    - `is_turnover` IS now read straight off NflPlay.is_turnover — this
      used to be a text-regex workaround (INTERCEPTED|FUMBLES over the
      play's own text) because that column was never actually set on any
      stored row; the real bug was the ingestion parser silently dropping
      every interception/fumble-recovery play before it ever reached a
      branch that would write is_turnover (see nfl_playbyplay_api.py's
      _SPECIAL_TEAMS_AND_TURNOVER_TYPES). Now fixed at the source, so this
      reads the real ESPN-reported flag. Games ingested before that fix
      still need nfl_playbyplay_backfill.py re-run against them once to
      pick up the correction retroactively — until then, their rows here
      simply won't ever be turnovers (same "no signal yet" as any other
      not-yet-backfilled gap in this app, not silently wrong).
    - `is_scoring_play` lets simulate_drive() tell a plain turnover apart
      from a real defensive/return touchdown (a resampled Pick-6 or
      scoop-and-score) that happens to BE a turnover play — needed to
      credit the right side's scoreboard and to tally nfl_fantasy.py's
      DST_TD fantasy stat; see simulate_drive()'s own turnover branch.

    `seasons` (optional, e.g. [2024, 2025, 2026]): restricts to those
    seasons; None (the default) uses everything stored. Returns plain
    dicts, not ORM rows — this is meant to be loaded once and bucketed/
    resampled many times in pure Python, not re-queried per draw."""
    from app import app as flask_app, NflPlay

    with flask_app.app_context():
        q = NflPlay.query.filter(
            NflPlay.play_type.in_(['Rush', 'Pass Reception', 'Pass Incompletion',
                                    'Sack', 'Passing Touchdown', 'Rushing Touchdown',
                                    'Pass Interception Return', 'Fumble Recovery (Own)',
                                    'Fumble Recovery (Opponent)', 'Sack Opp Fumble Recovery']),
            NflPlay.down.in_([1, 2, 3, 4]),
            NflPlay.is_no_play.isnot(True),
        )
        if seasons:
            q = q.filter(NflPlay.season.in_(seasons))
        rows = q.with_entities(NflPlay.down, NflPlay.distance, NflPlay.yards_to_endzone,
                                NflPlay.play_type, NflPlay.yards_gained, NflPlay.is_turnover,
                                NflPlay.end_yards_to_endzone, NflPlay.is_scoring_play).all()

    return [{
        'down': r.down,
        'distance': r.distance if r.distance is not None else 10,
        'yards_to_endzone': r.yards_to_endzone,
        'play_type': r.play_type,
        'yards_gained': r.yards_gained or 0.0,
        'is_turnover': bool(r.is_turnover),
        'end_yards_to_endzone': r.end_yards_to_endzone,
        'is_scoring_play': bool(r.is_scoring_play),
    } for r in rows if r.yards_to_endzone is not None]


def get_fourth_down_pool_db(seasons=None):
    """Real Punt / Field Goal Good / Field Goal Missed plays:
    [{yards_to_endzone, play_type, end_yards_to_endzone}] — feeds
    nfl_game_simulator.py's 4th-down decision. Deliberately does NOT pick
    "punt vs. attempt a kick" with a hand-set distance cutoff: a real
    team's own choice between the two, at whatever field position it
    actually happened, is baked into which play gets resampled — the
    exact same "let the real data decide" approach the rest of this
    sampler already uses, rather than inventing a threshold. `distance`/
    `down` aren't returned: these plays' own down/distance describes
    the decision TO kick (always down=4), not a resampleable in-down
    outcome the way a scrimmage play's does — only field position
    matters for bucketing here. `end_yards_to_endzone` is None for a
    missed FG past the point ESPN's `end` object bothers tracking (rare)
    — caller falls back to a fixed spot in that case.

    `seasons`: same as get_simulator_play_pool_db()."""
    from app import app as flask_app, NflPlay

    with flask_app.app_context():
        q = NflPlay.query.filter(
            NflPlay.play_type.in_(['Punt', 'Field Goal Good', 'Field Goal Missed']),
            NflPlay.is_no_play.isnot(True),
        )
        if seasons:
            q = q.filter(NflPlay.season.in_(seasons))
        rows = q.with_entities(NflPlay.yards_to_endzone, NflPlay.play_type,
                                NflPlay.end_yards_to_endzone).all()

    return [{
        'yards_to_endzone': r.yards_to_endzone,
        'play_type': r.play_type,
        'end_yards_to_endzone': r.end_yards_to_endzone,
    } for r in rows if r.yards_to_endzone is not None]


def get_kickoff_pool_db(seasons=None):
    """Real Kickoff / Kickoff Return Touchdown plays:
    [{end_yards_to_endzone, is_scoring_play}] — feeds the field position
    the simulator hands the receiving team after a touchdown or a made
    field goal. is_scoring_play flags the rare (~0.3% of kickoffs,
    confirmed on this app's 2024-2026 warehouse) return-touchdown case,
    which the caller should treat as an immediate score for the
    RECEIVING side rather than the start of a normal drive.

    `seasons`: same as get_simulator_play_pool_db()."""
    from app import app as flask_app, NflPlay

    with flask_app.app_context():
        q = NflPlay.query.filter(
            NflPlay.play_type.in_(['Kickoff', 'Kickoff Return Touchdown']),
            NflPlay.is_no_play.isnot(True),
        )
        if seasons:
            q = q.filter(NflPlay.season.in_(seasons))
        rows = q.with_entities(NflPlay.end_yards_to_endzone, NflPlay.is_scoring_play).all()

    return [{
        'end_yards_to_endzone': r.end_yards_to_endzone,
        'is_scoring_play': bool(r.is_scoring_play),
    } for r in rows if r.end_yards_to_endzone is not None]


def _redistribute_shares(shares, unavailable):
    """shares: {player_id: fraction}, already summing to ~1.0. Zeroes out
    any player in `unavailable` and redistributes their vacated share
    proportionally across the REMAINING players (each remaining player's
    share grows in proportion to their own existing share) — not a claim
    that a specific backup "inherits" the injured player's exact role,
    just scaling everyone else up so the injured player's historical
    volume doesn't sit unused on someone who isn't playing. If every
    player with a share is unavailable (no one left to redistribute to),
    returns `shares` unchanged rather than going empty — same graceful-
    degradation convention attribute_plays_to_players() already relies on
    for a team with no stored shares at all."""
    out_total = sum(v for pid, v in shares.items() if pid in unavailable)
    if out_total <= 0:
        return shares
    remaining = {pid: v for pid, v in shares.items() if pid not in unavailable}
    remaining_total = sum(remaining.values())
    if remaining_total <= 0:
        return shares
    return {pid: v + (v / remaining_total) * out_total for pid, v in remaining.items()}


def get_team_usage_shares_db(team_id, before_date=None, window_games=5):
    """Recent offensive usage for one team, derived from NflPlay (not the
    CURRENT-roster-gated nfl_roster_api path build_player_pool() uses) —
    feeds nfl_game_simulator.py's player attribution step, which needs
    "who gets the ball when the offense runs/throws," not just "who's on
    the roster." Scoped to that team's last `window_games` distinct
    game_ids (most recent first) so a depth-chart change (new starter,
    committee shift) isn't diluted by stale early-season shares.

    Returns {'qb_id': str or None, 'rush_shares': {player_id: fraction},
    'rec_shares': {player_id: fraction}} — shares sum to ~1.0 across
    players who touched the ball in that window; a team with zero stored
    plays in the window returns all-empty (caller's problem to decide a
    fallback, same "real gap beats a guess" convention as the rest of
    this module).

    INJURY-AWARE: the raw shares above are computed from recent real
    plays only, which has no idea who's playing THIS week — so before
    returning, any player confirmed unavailable as of `before_date` (None
    = today, same "no cutoff" convention get_latest_injury_status() uses
    — see get_unavailable_player_ids(), the same Out/IR/Suspended check
    build_player_pool() already uses to drop a player from the pool
    entirely) has their rush/rec share redistributed to healthy
    teammates via _redistribute_shares(), and an unavailable qb_id is
    swapped for another roster QB via get_or_fetch_roster_db() (first
    available alternate found; left as the original if none can be
    resolved — still strictly better than crediting a sidelined starter).
    This only fixes WHO gets credited with the passing line, not HOW a
    true backup's own play style would differ — the underlying play
    outcomes still come from the league-wide resampled pool, tilted only
    by team offense/defense grade (see nfl_game_simulator.py's own
    documented scope)."""
    from app import app as flask_app, NflPlay

    with flask_app.app_context():
        q = NflPlay.query.filter(NflPlay.offense_team_id == team_id, NflPlay.is_no_play.isnot(True))
        if before_date:
            q = q.filter(NflPlay.game_date < before_date)
        game_ids = [r[0] for r in (q.with_entities(NflPlay.game_id)
                                    .distinct()
                                    .order_by(NflPlay.game_date.desc())
                                    .limit(window_games).all())]
        if not game_ids:
            return {'qb_id': None, 'rush_shares': {}, 'rec_shares': {}}

        rows = (NflPlay.query
                .filter(NflPlay.offense_team_id == team_id, NflPlay.game_id.in_(game_ids),
                        NflPlay.is_no_play.isnot(True))
                .with_entities(NflPlay.play_type, NflPlay.passer_id, NflPlay.rusher_id,
                                NflPlay.receiver_id)
                .all())

    pass_counts = defaultdict(int)
    rush_counts = defaultdict(int)
    rec_counts = defaultdict(int)
    for play_type, passer_id, rusher_id, receiver_id in rows:
        if passer_id and play_type in ('Pass Reception', 'Pass Incompletion', 'Passing Touchdown', 'Sack'):
            pass_counts[passer_id] += 1
        if rusher_id and play_type in ('Rush', 'Rushing Touchdown'):
            rush_counts[rusher_id] += 1
        if receiver_id and play_type in ('Pass Reception', 'Passing Touchdown'):
            rec_counts[receiver_id] += 1

    qb_id = max(pass_counts, key=pass_counts.get) if pass_counts else None
    rush_total = sum(rush_counts.values())
    rec_total = sum(rec_counts.values())
    rush_shares = {pid: c / rush_total for pid, c in rush_counts.items()} if rush_total else {}
    rec_shares = {pid: c / rec_total for pid, c in rec_counts.items()} if rec_total else {}

    unavailable = get_unavailable_player_ids(team_id, before_date)
    if unavailable:
        rush_shares = _redistribute_shares(rush_shares, unavailable)
        rec_shares = _redistribute_shares(rec_shares, unavailable)
        if qb_id in unavailable:
            roster = get_or_fetch_roster_db(team_id)
            alt_qbs = [p['id'] for p in roster if p['position'] == 'QB'
                       and p['id'] != qb_id and p['id'] not in unavailable]
            if alt_qbs:
                qb_id = alt_qbs[0]

    return {
        'qb_id': qb_id,
        'rush_shares': rush_shares,
        'rec_shares': rec_shares,
    }


def persist_sim_results_db(season, week, game_id, team_id, opponent_id, results):
    """Upserts one NflSimResult row per player in `results` (each:
    {player_id, player_name, position, n_draws, sim_min, sim_median,
    sim_max, sim_mean, sim}), on the (season, week, player_id) unique
    constraint — overwrites yesterday's run for the same week rather than
    accumulating a history, since what matters is today's best estimate
    (see NflSimResult's class docstring, app.py). Called by
    nfl_game_simulator.simulate_week() once per (game, side) — i.e.
    multiple times per week, each with a different `team_id` — so the
    existing-row lookup below is deliberately NOT scoped to `team_id`:
    the real unique constraint is (season, week, player_id) only, and a
    player really can turn up under two different team_ids within the
    same week's run (confirmed live — a recently-traded player whose
    get_team_usage_shares_db() attribution still reflects their OLD
    team's recent plays while nfl_roster_api's CURRENT roster already has
    them on the NEW team crashed an early version of this function with
    a real UNIQUE-constraint IntegrityError, scoping the lookup to
    team_id meant the second team's call never saw the first team's
    already-committed row). Scoping to (season, week) only means that
    rare case just gets overwritten by whichever call processes them
    last, rather than crashing the whole week's run over one player's
    ambiguous attribution."""
    import json as _json
    from app import app as flask_app, db, NflSimResult

    with flask_app.app_context():
        existing = {
            r.player_id: r for r in
            NflSimResult.query.filter_by(season=season, week=week).all()
        }
        for r in results:
            row = existing.get(r['player_id'])
            if row is None:
                row = NflSimResult(season=season, week=week, player_id=r['player_id'])
                db.session.add(row)
            row.game_id = game_id
            row.team_id = team_id
            row.opponent_id = opponent_id
            row.player_name = r.get('player_name')
            row.position = r.get('position')
            row.n_draws = r['n_draws']
            row.sim_min = r['sim_min']
            row.sim_median = r['sim_median']
            row.sim_max = r['sim_max']
            row.sim_mean = r['sim_mean']
            row.sim_draws_json = _json.dumps(r['sim'])
        db.session.commit()


def get_sim_results_db(season, week):
    """{game_id: {team_id: [{...}, ...]}} — every stored NflSimResult row
    for one week, grouped by game then by team, each team's list sorted
    by sim_median descending. Deliberately keyed by team_id rather than
    'home'/'away' — this function has no opinion on which side is which
    (the caller — /nfl/simulations — already has that from nfl_api's own
    game list, no reason to duplicate it here). Position players only
    (this app doesn't roster/attribute punters or kickers — see
    nfl_game_simulator.py's docstring)."""
    import json as _json
    from app import app as flask_app, NflSimResult

    with flask_app.app_context():
        rows = (NflSimResult.query.filter_by(season=season, week=week)
                .order_by(NflSimResult.sim_median.desc()).all())
        out = defaultdict(lambda: defaultdict(list))
        for r in rows:
            out[r.game_id][r.team_id].append({
                'player_id': r.player_id, 'player_name': r.player_name or f'id:{r.player_id}',
                'position': r.position or '?', 'opponent_id': r.opponent_id,
                'n_draws': r.n_draws, 'sim_min': r.sim_min, 'sim_median': r.sim_median,
                'sim_max': r.sim_max, 'sim_mean': r.sim_mean,
                'sim': _json.loads(r.sim_draws_json) if r.sim_draws_json else [],
            })
        return {gid: dict(teams) for gid, teams in out.items()}


def persist_sim_game_summary_db(season, week, game_id, home_team_id, away_team_id, summary):
    """Upserts one NflSimGame row — the team-score-level companion to
    persist_sim_results_db()'s per-player rows: win/loss/tie counts,
    average final score, and the approximate quarter-by-quarter scoring
    breakdown (see nfl_game_simulator._approx_quarter()'s "no real game
    clock" caveat — this is NOT a precise clock-based quarter). `summary`
    is simulate_matchup()'s own 'game' dict, passed straight through.
    Upserts on (season, week, game_id), same overwrite-previous-day
    convention as NflSimResult."""
    import json as _json
    from app import app as flask_app, db, NflSimGame

    with flask_app.app_context():
        row = NflSimGame.query.filter_by(season=season, week=week, game_id=game_id).first()
        if row is None:
            row = NflSimGame(season=season, week=week, game_id=game_id)
            db.session.add(row)
        row.home_team_id = home_team_id
        row.away_team_id = away_team_id
        row.n_sims = summary['n']
        row.home_wins = summary['home_wins']
        row.away_wins = summary['away_wins']
        row.ties = summary['ties']
        row.home_avg_score = summary['home_avg_score']
        row.away_avg_score = summary['away_avg_score']
        row.quarter_scores_json = _json.dumps({
            'home': summary['home_quarter_avg'], 'away': summary['away_quarter_avg'],
        })
        db.session.commit()


def get_sim_game_summary_db(season, week):
    """{game_id: {...}} — every stored NflSimGame row for one week. Each
    value: {'n_sims', 'home_wins', 'away_wins', 'ties', 'home_win_pct',
    'away_win_pct', 'home_avg_score', 'away_avg_score',
    'home_quarter_avg', 'away_quarter_avg'} (the last two: [Q1, Q2, Q3,
    Q4] average points, approximate — see persist_sim_game_summary_db()'s
    docstring). Win percentages are computed here from the stored counts
    rather than also stored as their own columns, so they can never drift
    from n_sims/home_wins/away_wins."""
    import json as _json
    from app import app as flask_app, NflSimGame

    with flask_app.app_context():
        rows = NflSimGame.query.filter_by(season=season, week=week).all()
        out = {}
        for r in rows:
            qs = (_json.loads(r.quarter_scores_json) if r.quarter_scores_json
                  else {'home': [0, 0, 0, 0], 'away': [0, 0, 0, 0]})
            out[r.game_id] = {
                'n_sims': r.n_sims, 'home_wins': r.home_wins, 'away_wins': r.away_wins, 'ties': r.ties,
                'home_win_pct': round(100 * r.home_wins / r.n_sims, 1) if r.n_sims else 0.0,
                'away_win_pct': round(100 * r.away_wins / r.n_sims, 1) if r.n_sims else 0.0,
                'home_avg_score': r.home_avg_score, 'away_avg_score': r.away_avg_score,
                'home_quarter_avg': qs.get('home', [0, 0, 0, 0]), 'away_quarter_avg': qs.get('away', [0, 0, 0, 0]),
            }
        return out


def get_or_fetch_roster_db(team_id, max_age_hours=24):
    """Returns [{'id', 'name', 'position'}, ...] for team_id's current
    skill-position roster — same shape/content as nfl_roster_api.
    get_skill_players(team_id), but checks NflRosterSnapshot (a DB-backed
    cache ANY process can share, see that model's class docstring in
    app.py for why nfl_roster_api's own 24h cache doesn't help a one-off
    script) before making a live ESPN call, and persists what it fetches
    for the next caller.

    A stored snapshot younger than `max_age_hours` is returned as-is, no
    live call at all. An older or missing one triggers a live
    nfl_roster_api.get_skill_players() fetch, whose result both answers
    this call AND fully replaces this team's stored snapshot (delete +
    bulk insert, not a merge) — a player who's been cut/traded since the
    last snapshot must disappear here too, not linger from a stale row
    nothing happened to overwrite. If the live fetch itself comes back
    empty (ESPN hiccup, see nfl_roster_api._get()'s own fail-backoff),
    the stale snapshot already in the DB is returned instead of an empty
    list, same "a real gap beats a wrong guess" convention as the rest of
    this app — better a few-days-stale roster than silently losing every
    player from this team's pool for one bad fetch."""
    import nfl_roster_api
    from datetime import datetime, timedelta, timezone
    from app import app as flask_app, db, NflRosterSnapshot

    with flask_app.app_context():
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        rows = NflRosterSnapshot.query.filter_by(team_id=team_id).all()
        if rows and all(r.fetched_at and r.fetched_at.replace(tzinfo=timezone.utc) >= cutoff for r in rows):
            return [{'id': r.player_id, 'name': r.player_name, 'position': r.position} for r in rows]

        live = nfl_roster_api.get_skill_players(team_id)
        if not live:
            return [{'id': r.player_id, 'name': r.player_name, 'position': r.position} for r in rows]

        NflRosterSnapshot.query.filter_by(team_id=team_id).delete()
        for p in live:
            db.session.add(NflRosterSnapshot(team_id=team_id, player_id=p['id'],
                                              player_name=p['name'], position=p.get('position')))
        db.session.commit()
        return live


def get_sim_week_accuracy_db(season, week):
    """Compares this week's stored NflSimGame rows against REAL completed
    results (TeamGameStat) — only games already Final count, so an
    upcoming/in-progress week naturally returns games_final=0 (and every
    metric None) rather than erroring or comparing against nothing.
    Mirrors the ad-hoc comparison script this app's own session history
    used to validate the simulator week by week, now a reusable query so
    /nfl/simulations can show it directly instead of it only ever being a
    one-off script.

    Returns {'games_final', 'win_pick_correct', 'win_pick_accuracy',
    'brier', 'mean_abs_score_error'}. win_pick_accuracy: % of Final games
    where the side with the higher simulated win count actually won (a
    50/50 split counts as picking home, same tie-break simulate_matchup()
    itself doesn't need to resolve since real games don't tie there).
    brier: mean squared error between simulated home-win fraction and the
    real 0/1 outcome — 0.25 is what guessing 50/50 every time scores,
    lower is better (same convention as this app's real betting-model
    performance page, model_performance.html). mean_abs_score_error:
    average |simulated avg score - real final score|, per team."""
    from app import app as flask_app, NflSimGame, TeamGameStat

    with flask_app.app_context():
        sims = NflSimGame.query.filter_by(season=season, week=week).all()
        if not sims:
            return {'games_final': 0, 'win_pick_correct': 0, 'win_pick_accuracy': None,
                    'brier': None, 'mean_abs_score_error': None}
        game_ids = [s.game_id for s in sims]
        real_rows = {
            r.game_id: r for r in
            TeamGameStat.query.filter(TeamGameStat.sport == 'NFL', TeamGameStat.game_id.in_(game_ids),
                                       TeamGameStat.is_home.is_(True)).all()
        }

    correct = 0
    total = 0
    brier_sum = 0.0
    score_err_sum = 0.0
    score_err_n = 0
    for s in sims:
        r = real_rows.get(s.game_id)
        if r is None or r.won is None or not s.n_sims:
            continue
        total += 1
        home_frac = s.home_wins / s.n_sims
        real_home_won = bool(r.won)
        picked_home = s.home_wins >= s.away_wins
        if picked_home == real_home_won:
            correct += 1
        brier_sum += (home_frac - (1.0 if real_home_won else 0.0)) ** 2
        if r.points is not None and r.points_allowed is not None:
            score_err_sum += abs(s.home_avg_score - r.points) + abs(s.away_avg_score - r.points_allowed)
            score_err_n += 2

    return {
        'games_final': total,
        'win_pick_correct': correct,
        'win_pick_accuracy': round(100 * correct / total, 1) if total else None,
        'brier': round(brier_sum / total, 4) if total else None,
        'mean_abs_score_error': round(score_err_sum / score_err_n, 1) if score_err_n else None,
    }
