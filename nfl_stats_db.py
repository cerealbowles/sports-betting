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
    team_id, opponent_id, is_home, position, categories}] where
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
                'team_id':     r.team_id,
                'opponent_id': r.opponent_id,
                'is_home':     r.is_home,
                'position':    r.position,
                'categories':  categories,
            })
        return out


def get_player_splits_summary_db(player_id, before_date=None, limit=8):
    """Aggregates this player's rushing_splits/receiving_splits/
    passing_splits (nfl_playbyplay_api.py's output, merged into
    PlayerGameStat.stats_json by ingest_play_by_play()) across their last
    `limit` stored games — point-in-time if `before_date` is given, same
    no-leakage convention as this module's other before_date params.
    Built on top of get_player_gamelog_db() rather than its own query, so
    there's one source of truth for "this player's recent games."

    Returns {'games': N, 'rushing': {...} or None, 'receiving': {...} or
    None, 'passing': {...} or None} — a sub-dict is None if the player
    never recorded that category in the window (e.g. a pure receiver has
    no 'passing'), not a dict of zeros, so a caller can't mistake "never
    attempted" for "attempted and got nothing." DISPLAY-ONLY context as of
    this writing — see project_points()'s docstring in nfl_fantasy.py for
    why this isn't (yet) a projection input."""
    games = get_player_gamelog_db(player_id, before_date=before_date, limit=limit)
    totals = {'rushing': None, 'receiving': None, 'passing': None}
    for g in games:
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
    return {
        'games': len(games),
        'rushing': dict(totals['rushing']) if totals['rushing'] else None,
        'receiving': dict(totals['receiving']) if totals['receiving'] else None,
        'passing': dict(totals['passing']) if totals['passing'] else None,
    }


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
