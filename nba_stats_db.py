"""
nba_stats_db.py — Read/write layer for the local NBA stats warehouse
(app.py's PlayerGameStat / TeamGameStat tables).

Goal: stop re-fetching every active player's season gamelog from ESPN on
every cache miss (nba_roster_api.py's _TTL keeps it to 6h, but that's still
~20 live HTTP calls per team, per cache refresh, across every NBA module
that needs it). Once a game is Final, its full box score is durable —
there's no reason to ever ask ESPN for it again. This module is the
persisted alternative: nba_roster_api.get_player_gamelog() tries here
first and only falls back to the live ESPN fetch for players this table
doesn't have data for yet (new to the league, or not yet backfilled).

Writes happen in two places, both idempotent (upsert on the (sport,
game_id, player_id/team_id) unique constraint, so re-ingesting a game
already stored just updates it in place):
  - nba_api.py's _build_game(), the moment a game's status is Final —
    reuses the boxscore it already fetches for the live lineup-comparison
    feature, so ingestion costs zero extra API calls.
  - nba_stats_backfill.py — one-time walk over already-played games to
    seed the table (see that script for the historical catch-up).

Uses the same lazy `from app import ...` pattern as every other
*_backfill.py script in this app (app.py imports this module's callers at
top level, so importing app.py back at module level here would be
circular — importing inside each function, after app.py has finished
initializing, isn't).
"""
from collections import defaultdict

GAME_TYPE_BY_ESPN_SEASON_TYPE = {1: 'preseason', 2: 'regular', 3: 'postseason'}


def ingest_game(event_id, game_date_et, season, game_type, home_id, away_id,
                 home_score, away_score, boxscore, home_name=None, away_name=None,
                 pace_stats=None):
    """Upserts PlayerGameStat rows for every athlete ESPN's boxscore lists
    for this game (did-not-play entries included — useful for seeing who
    was active/inactive, and harmless since get_player_gamelog_db() below
    filters them back out) and one TeamGameStat row per side.

    boxscore: {team_id: [row, ...]} as returned by
    nba_boxscore_api.get_live_boxscore(event_id) — must be the Live/Final
    shape (has 'starter'/'did_not_play'/stat keys), not empty/pre-game.

    pace_stats: optional {team_id: {fga, oreb, tov, fta}} from
    nba_boxscore_api.get_team_pace_stats(event_id) — stored on TeamGameStat
    so a team's point-in-time pace can be reconstructed later purely from
    its own prior Final games (see bball_total_model.py / nba_total_
    backfill.py). Omitted/None leaves those columns NULL, same as before
    this parameter existed.

    Safe to call repeatedly for the same game (e.g. once per 20-min cron
    tick while recently Final, or re-run from the backfill script) — each
    row is matched on its unique constraint and updated in place rather
    than duplicated.
    """
    if not boxscore:
        return
    from app import app as flask_app, db, PlayerGameStat, TeamGameStat

    with flask_app.app_context():
        for team_id, opp_id, team_score, opp_score, is_home, team_name, opp_name in (
            (home_id, away_id, home_score, away_score, True, home_name, away_name),
            (away_id, home_id, away_score, home_score, False, away_name, home_name),
        ):
            rows = boxscore.get(team_id)
            if rows is None:
                continue

            for r in rows:
                existing = PlayerGameStat.query.filter_by(
                    sport='NBA', game_id=event_id, player_id=r['id']).first()
                target = existing or PlayerGameStat(sport='NBA', game_id=event_id, player_id=r['id'])
                target.game_date    = game_date_et
                target.season       = season
                target.game_type    = game_type
                target.player_name  = r['name']
                target.team_id      = team_id
                target.opponent_id  = opp_id
                target.is_home      = is_home
                target.starter      = r.get('starter', False)
                target.did_not_play = r.get('did_not_play', False)
                target.minutes      = r.get('minutes')
                target.points       = r.get('points')
                target.rebounds     = r.get('rebounds')
                target.assists      = r.get('assists')
                target.steals       = r.get('steals')
                target.blocks       = r.get('blocks')
                target.turnovers    = r.get('turnovers')
                target.fouls        = r.get('fouls')
                target.plus_minus   = r.get('plus_minus')
                target.fgm          = r.get('fgm')
                target.fga          = r.get('fga')
                target.three_pm     = r.get('three_pm')
                target.three_pa     = r.get('three_pa')
                target.ftm          = r.get('ftm')
                target.fta          = r.get('fta')
                if not existing:
                    db.session.add(target)

            if team_score is not None and opp_score is not None:
                existing_team = TeamGameStat.query.filter_by(
                    sport='NBA', game_id=event_id, team_id=team_id).first()
                tt = existing_team or TeamGameStat(sport='NBA', game_id=event_id, team_id=team_id)
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
                pace = (pace_stats or {}).get(team_id) or {}
                tt.fga  = pace.get('fga')
                tt.oreb = pace.get('oreb')
                tt.tov  = pace.get('tov')
                tt.fta  = pace.get('fta')
                if not existing_team:
                    db.session.add(tt)

        db.session.commit()


def ingest_play_by_play(event_id, game_date_et, season, game_type, summary):
    """Parses `summary` via nba_playbyplay_api.parse_plays() and upserts
    every play as an NbaPlay row (matched on the (game_id, play_id) unique
    constraint, so a re-run just updates existing rows in place).

    Call this with the SAME `summary` get_live_boxscore()/get_team_pace_
    stats() already fetched for this event_id — nba_boxscore_api.get_live_
    summary() shares their cache, so this costs zero extra API calls.

    No-op if the game has no plays yet (pre-game) or parsing finds nothing.
    """
    import json
    import nba_playbyplay_api
    from app import app as flask_app, db, NbaPlay

    plays = nba_playbyplay_api.parse_plays(summary, event_id)
    if not plays:
        return

    with flask_app.app_context():
        existing_ids = {
            r.play_id for r in NbaPlay.query.filter_by(game_id=event_id)
            .with_entities(NbaPlay.play_id).all()
        }
        for p in plays:
            fields = {
                'sequence_number':   p['sequence_number'],
                'season':            season,
                'game_date':         game_date_et,
                'game_type':         game_type,
                'team_id':           p['team_id'],
                'play_type_id':      p['play_type_id'],
                'play_type':         p['play_type'],
                'text':              p['text'],
                'short_description': p['short_description'],
                'period':            p['period'],
                'clock':             p['clock'],
                'home_score':        p['home_score'],
                'away_score':        p['away_score'],
                'score_value':       p['score_value'],
                'is_scoring_play':   p['is_scoring_play'],
                'is_shooting_play':  p['is_shooting_play'],
                'points_attempted':  p['points_attempted'],
                'shot_x':            p['shot_x'],
                'shot_y':            p['shot_y'],
                'participant_ids':   json.dumps(p['participant_ids']),
            }
            if p['play_id'] in existing_ids:
                NbaPlay.query.filter_by(game_id=event_id, play_id=p['play_id']).update(fields)
            else:
                db.session.add(NbaPlay(game_id=event_id, play_id=p['play_id'], **fields))

        db.session.commit()


def get_player_gamelog_db(player_id, before_date=None, limit=None):
    """Returns this player's played games (did_not_play excluded — mirrors
    nba_roster_api.get_player_gamelog, whose ESPN source only lists games
    actually played), newest first, in the same shape nba_player_model.
    project_player() already consumes: [{event_id, date, minutes, points,
    rebounds, assists, fga, fta, tov, team_id, home}]. [] if this table has
    nothing for the player yet (caller should fall back to the live fetch).

    before_date (YYYY-MM-DD, exclusive): for no-leakage backtesting, same
    requirement nba_player_bootstrap.py already has for the live-API path.
    """
    from app import app as flask_app, PlayerGameStat

    with flask_app.app_context():
        q = PlayerGameStat.query.filter_by(sport='NBA', player_id=player_id, did_not_play=False)
        if before_date:
            q = q.filter(PlayerGameStat.game_date < before_date)
        q = q.order_by(PlayerGameStat.game_date.desc())
        if limit:
            q = q.limit(limit)
        rows = q.all()
        return [{
            'event_id': r.game_id,
            'date':     r.game_date,
            'minutes':  r.minutes or 0.0,
            'points':   r.points or 0.0,
            'rebounds': r.rebounds or 0.0,
            'assists':  r.assists or 0.0,
            'steals':   r.steals or 0.0,
            'blocks':   r.blocks or 0.0,
            'fga':      r.fga or 0.0,
            'fta':      r.fta or 0.0,
            'tov':      r.turnovers or 0.0,
            'team_id':  r.team_id,
            'home':     r.is_home,
        } for r in rows]


def get_team_game_log_db(team_id, before_date=None, limit=None):
    """Returns this team's finalized games, newest first: [{date, points,
    points_allowed, won, opponent_id, is_home}]. [] if nothing stored yet."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        q = TeamGameStat.query.filter_by(sport='NBA', team_id=team_id)
        if before_date:
            q = q.filter(TeamGameStat.game_date < before_date)
        q = q.order_by(TeamGameStat.game_date.desc())
        if limit:
            q = q.limit(limit)
        rows = q.all()
        return [{
            'date':            r.game_date,
            'points':          r.points,
            'points_allowed':  r.points_allowed,
            'won':             r.won,
            'opponent_id':     r.opponent_id,
            'is_home':         r.is_home,
        } for r in rows]


def get_season_game_log_db(season, game_type='regular'):
    """Returns every stored game for `season` in the same shape
    nba_api._get_season_game_log()'s live ESPN walk already produces:
    [{game_date, home_name, away_name, home_score, away_score, home_won}] —
    the format _compute_team_season_stats()/_team_recent_form()/
    _team_rest_days() all consume (team-name-keyed, one row per game, not
    per team-side). [] if this table has nothing for the season yet.

    `game_type` may be a single value (the default, 'regular') or a
    list/tuple of values — e.g. ('regular', 'postseason') for a recency
    read (last-5-games, rest-days) that should keep going once the
    playoffs start instead of freezing at the regular-season finale.
    Season-average reads (PPG/PPG-allowed) should keep passing the default
    so playoff games against a narrower opponent pool don't skew them.

    Reads only the is_home=True TeamGameStat row per game — each finalized
    game writes one row per side, and the home side's row already carries
    both teams' names/scores (team_name/points = home, opponent_name/
    points_allowed = away), so the away side's row would just be the exact
    same game pair restated from the other row's perspective.
    """
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        query = TeamGameStat.query.filter_by(sport='NBA', season=season, is_home=True)
        if isinstance(game_type, (list, tuple, set)):
            query = query.filter(TeamGameStat.game_type.in_(game_type))
        else:
            query = query.filter_by(game_type=game_type)
        rows = query.order_by(TeamGameStat.game_date.asc()).all()
        return [{
            'game_date':  r.game_date,
            'home_name':  r.team_name,
            'away_name':  r.opponent_name,
            'home_score': r.points,
            'away_score': r.points_allowed,
            'home_won':   r.won,
        } for r in rows if r.team_name and r.opponent_name]


def get_schedule_db(season, game_type='regular'):
    """Returns every stored game for `season` as [{event_id, game_date,
    home_id, away_id, home_score, away_score}], chronological — the DB
    equivalent of nba_player_bootstrap.py's old live ESPN schedule fetch
    (fetch_season_schedule), now that the warehouse covers full historical
    seasons (see nba_stats_backfill.py). Built from the is_home=True
    TeamGameStat row per game, same approach as get_season_game_log_db()
    above, just keeping ids (needed to key per-player gamelogs/boxscores)
    instead of names."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query
                .filter_by(sport='NBA', season=season, game_type=game_type, is_home=True)
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


def get_teams_with_data_db(season, game_type='regular'):
    """Distinct (team_id, team_name) pairs with at least one stored
    TeamGameStat row this season — the "which teams can we rank" universe
    for the game-detail page's league-rank computation, same reasoning as
    nfl_stats_db.get_teams_with_data_db()."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query
                .filter_by(sport='NBA', season=season, game_type=game_type)
                .with_entities(TeamGameStat.team_id, TeamGameStat.team_name)
                .distinct()
                .all())
        return [(r[0], r[1]) for r in rows if r[0]]


def get_team_stat_totals_db(team_id, before_date=None, limit=20):
    """Season-to-date shooting/rebounding/playmaking totals for one team,
    summed from its players' PlayerGameStat rows (fgm/fga/three_pm/
    three_pa/ftm/fta/rebounds/assists/turnovers — all stored per-player,
    see ingest_game()), plus points/points_allowed from TeamGameStat.
    Powers the game-detail page's "Team comparison" ranked-stat section —
    there's no existing team-level aggregate query for this; raw columns
    exist, this is the first thing to sum them. None if nothing stored yet
    for this team."""
    from app import app as flask_app, PlayerGameStat, TeamGameStat

    with flask_app.app_context():
        tq = TeamGameStat.query.filter_by(sport='NBA', team_id=team_id)
        if before_date:
            tq = tq.filter(TeamGameStat.game_date < before_date)
        tq = tq.order_by(TeamGameStat.game_date.desc())
        if limit:
            tq = tq.limit(limit)
        team_rows = tq.all()
        game_ids = [r.game_id for r in team_rows]
        if not game_ids:
            return None

        points = sum(r.points or 0 for r in team_rows)
        points_allowed = sum(r.points_allowed or 0 for r in team_rows)
        games_played = len(team_rows)

        prows = (PlayerGameStat.query
                 .filter_by(sport='NBA', team_id=team_id)
                 .filter(PlayerGameStat.game_id.in_(game_ids))
                 .all())
        totals = {'fgm': 0, 'fga': 0, 'three_pm': 0, 'three_pa': 0, 'ftm': 0, 'fta': 0,
                   'rebounds': 0, 'assists': 0, 'turnovers': 0}
        for r in prows:
            for key in totals:
                totals[key] += getattr(r, key, None) or 0

        return {
            'games_played': games_played,
            'ppg': round(points / games_played, 1) if games_played else None,
            'ppg_allowed': round(points_allowed / games_played, 1) if games_played else None,
            'fg_pct': round(100 * totals['fgm'] / totals['fga'], 1) if totals['fga'] else None,
            'three_pct': round(100 * totals['three_pm'] / totals['three_pa'], 1) if totals['three_pa'] else None,
            'rpg': round(totals['rebounds'] / games_played, 1) if games_played else None,
            'apg': round(totals['assists'] / games_played, 1) if games_played else None,
            'topg': round(totals['turnovers'] / games_played, 1) if games_played else None,
        }


def get_game_boxscore_db(game_id):
    """Returns {team_id: [{player_id, minutes, points, rebounds, assists,
    fga, fta, tov}]} for one game — the DB equivalent of nba_player_
    bootstrap.py's old live ESPN per-game summary fetch (fetch_boxscore).
    Excludes did_not_play rows and non-participants (minutes <= 0), same
    filter the old live fetch applied. {} if nothing stored for this game."""
    from app import app as flask_app, PlayerGameStat

    with flask_app.app_context():
        rows = (PlayerGameStat.query
                .filter_by(sport='NBA', game_id=game_id, did_not_play=False)
                .all())

    out = {}
    for r in rows:
        if not r.minutes or r.minutes <= 0:
            continue
        out.setdefault(r.team_id, []).append({
            'player_id': r.player_id,
            'minutes':   r.minutes or 0.0,
            'points':    r.points or 0.0,
            'rebounds':  r.rebounds or 0.0,
            'assists':   r.assists or 0.0,
            'steals':    r.steals or 0.0,
            'blocks':    r.blocks or 0.0,
            'turnovers': r.turnovers or 0.0,
            'fga':       r.fga or 0.0,
            'fta':       r.fta or 0.0,
            'tov':       r.turnovers or 0.0,
        })
    return out


# ── nba_game_simulator.py support: possession-level pool + usage shares ───
#
# NFL's simulator resamples real PLAYS, bucketed by (down, distance, field
# position) — NbaPlay has no equivalent bucketing key (basketball has no
# down/distance/field-position concept), and its `participant_ids` is an
# unordered list (see NbaPlay's own docstring, app.py) that doesn't label
# shooter vs. assister vs. rebounder, so a play-by-play role-attribution
# pass the way nfl_game_simulator.attribute_plays_to_players() does isn't
# reliably reconstructable here without guessing an unverified ESPN
# play-type text taxonomy.
#
# Real resampling unit here is one TEAM's one PERIOD of actual scoring,
# bucketed by (period, score_margin_bucket) — the margin that team held
# entering that period. This needs no play-type parsing at all: every
# NbaPlay row already carries the running home_score/away_score and a real
# `period`, so a team's period-by-period scoring is reconstructed purely
# from those two columns. It's coarser than NFL's per-play resampling, but
# still genuinely resampling real historical outcomes (captures real
# blowout garbage-time compression / clutch tightening by score margin),
# and the "which exact plays happened" granularity NFL needs for player
# attribution isn't needed here — see get_team_usage_shares_db() below for
# how player-level attribution is done instead.
_MARGIN_BUCKET_EDGES = [-100, -15, -8, -3, 0, 3, 8, 15, 100]


def _margin_bucket(margin):
    for i in range(len(_MARGIN_BUCKET_EDGES) - 1):
        lo, hi = _MARGIN_BUCKET_EDGES[i], _MARGIN_BUCKET_EDGES[i + 1]
        if lo <= margin < hi or (hi == 100 and margin == 100):
            return i
    return len(_MARGIN_BUCKET_EDGES) - 2


def get_simulator_quarter_pool_db(seasons=None):
    """One flat list of real per-team, per-period scoring outcomes for
    nba_game_simulator.py's resampler: [{team_id, period, score_margin_
    bucket, points_scored}]. `period` is capped at 5 (every period 5+,
    i.e. any overtime, collapses into one bucket — OT frequency/length
    isn't itself modeled). `score_margin_bucket` is this team's own
    (team_score - opp_score) entering the period, bucketed by
    _margin_bucket() (same coarsening role as NFL's _distance_bucket/
    _field_bucket). Walks each game's NbaPlay rows in insertion order
    (`id` — matches ESPN's own chronological play order, since ingestion
    inserts them in the order parse_plays() returns them) rather than
    parsing `sequence_number`/`clock` strings.

    `seasons`: optional, restricts to those seasons; None uses everything
    stored. Returns plain dicts, meant to be loaded once and bucketed many
    times, same convention as NFL's get_simulator_play_pool_db()."""
    from app import app as flask_app, NbaPlay, TeamGameStat

    with flask_app.app_context():
        q = NbaPlay.query.filter(NbaPlay.period.isnot(None),
                                  NbaPlay.home_score.isnot(None),
                                  NbaPlay.away_score.isnot(None))
        if seasons:
            q = q.filter(NbaPlay.season.in_(seasons))
        rows = (q.order_by(NbaPlay.game_id, NbaPlay.id)
                .with_entities(NbaPlay.game_id, NbaPlay.period,
                               NbaPlay.home_score, NbaPlay.away_score).all())
        if not rows:
            return []

        game_ids = sorted({r.game_id for r in rows})
        home_team, away_team = {}, {}
        for t in (TeamGameStat.query
                  .filter(TeamGameStat.sport == 'NBA', TeamGameStat.game_id.in_(game_ids))
                  .with_entities(TeamGameStat.game_id, TeamGameStat.team_id, TeamGameStat.is_home).all()):
            if t.is_home:
                home_team[t.game_id] = t.team_id
            else:
                away_team[t.game_id] = t.team_id

    pool = []

    def flush(game_id, period, start_h, start_a, end_h, end_a):
        hp, ap = home_team.get(game_id), away_team.get(game_id)
        if hp is None or ap is None or period is None:
            return
        capped_period = min(period, 5)
        margin_at_start = start_h - start_a
        pool.append({'team_id': hp, 'period': capped_period,
                     'score_margin_bucket': _margin_bucket(margin_at_start),
                     'points_scored': end_h - start_h})
        pool.append({'team_id': ap, 'period': capped_period,
                     'score_margin_bucket': _margin_bucket(-margin_at_start),
                     'points_scored': end_a - start_a})

    cur_game = cur_period = None
    period_start_h = period_start_a = 0
    last_h = last_a = 0
    for r in rows:
        if r.game_id != cur_game:
            if cur_game is not None:
                flush(cur_game, cur_period, period_start_h, period_start_a, last_h, last_a)
            cur_game, cur_period = r.game_id, r.period
            period_start_h = period_start_a = 0
            last_h = last_a = 0
        elif r.period != cur_period:
            flush(cur_game, cur_period, period_start_h, period_start_a, last_h, last_a)
            period_start_h, period_start_a = last_h, last_a
            cur_period = r.period
        if r.home_score is not None:
            last_h = r.home_score
        if r.away_score is not None:
            last_a = r.away_score
    if cur_game is not None:
        flush(cur_game, cur_period, period_start_h, period_start_a, last_h, last_a)

    return pool


def get_team_usage_shares_db(team_id, before_date=None, window_games=10, unavailable_ids=None):
    """Recent per-player production shares for one team, from PlayerGameStat
    — feeds nba_game_simulator.py's player attribution step (NBA's
    counterpart to NFL's get_team_usage_shares_db(), adapted since NBA has
    no rush/rec role split: every active player shares in all 5 fantasy
    categories to some degree).

    Returns {player_id: {'name', 'mpg', 'pts_share', 'reb_pg', 'ast_pg',
    'stl_pg', 'blk_pg', 'tov_pg'}}. `pts_share` is this player's fraction
    of the TEAM's scoring (usage_rate x per_poss x minutes, via
    nba_player_model.project_player() — the same usage-weighted efficiency
    split project_team_score() already uses to redistribute minutes,
    reused here instead of reinventing it) — sums to ~1.0 across players
    with a usable projection. `reb_pg`/`ast_pg`/`stl_pg`/`blk_pg`/`tov_pg`
    are plain per-game averages over the window (no usage-weighting
    applies to these the way it does for scoring efficiency).

    `unavailable_ids` (optional, e.g. from nba_boxscore_api.
    get_unavailable_player_ids for tonight's specific game): players to
    drop before computing shares — same purpose as NFL's injury-aware
    redistribution, but simpler here: an unavailable player is just
    excluded outright (their team's shares then sum to <1.0 of the OLD
    total, i.e. the remaining players' shares scale up automatically
    relative to each other) rather than modeling who specifically absorbs
    their minutes.

    A team with no players clearing project_player()'s own minimum (no
    scoring possessions logged) returns {} — caller's problem to decide a
    fallback, same "real gap beats a guess" convention as the rest of this
    module."""
    import nba_player_model
    from app import app as flask_app, PlayerGameStat

    unavailable_ids = set(unavailable_ids or [])
    with flask_app.app_context():
        q = PlayerGameStat.query.filter_by(sport='NBA', team_id=team_id, did_not_play=False)
        if before_date:
            q = q.filter(PlayerGameStat.game_date < before_date)
        rows = q.order_by(PlayerGameStat.game_date.desc()).all()

    by_player = defaultdict(list)
    names = {}
    for r in rows:
        if r.player_id in unavailable_ids:
            continue
        names[r.player_id] = r.player_name
        by_player[r.player_id].append({
            'event_id': r.game_id, 'date': r.game_date, 'minutes': r.minutes or 0.0,
            'points': r.points or 0.0, 'rebounds': r.rebounds or 0.0, 'assists': r.assists or 0.0,
            'steals': r.steals or 0.0, 'blocks': r.blocks or 0.0,
            'fga': r.fga or 0.0, 'fta': r.fta or 0.0, 'tov': r.turnovers or 0.0,
        })

    raw_scores = {}
    extras = {}
    for pid, games in by_player.items():
        sample = games[:window_games]
        if not sample:
            continue
        proj = nba_player_model.project_player(games, recent_window=window_games)
        n = len(sample)
        extras[pid] = {
            'reb_pg': sum(g['rebounds'] for g in sample) / n,
            'ast_pg': sum(g['assists'] for g in sample) / n,
            'stl_pg': sum(g['steals'] for g in sample) / n,
            'blk_pg': sum(g['blocks'] for g in sample) / n,
            'tov_pg': sum(g['tov'] for g in sample) / n,
        }
        if proj:
            raw_scores[pid] = proj['mpg'] * proj['usage_rate'] * proj['per_poss']
            extras[pid]['mpg'] = proj['mpg']

    total_raw = sum(raw_scores.values())
    if total_raw <= 0:
        return {}

    return {
        pid: {
            'name': names.get(pid),
            'mpg': extras[pid].get('mpg', 0.0),
            'pts_share': raw_scores[pid] / total_raw,
            'reb_pg': extras[pid]['reb_pg'], 'ast_pg': extras[pid]['ast_pg'],
            'stl_pg': extras[pid]['stl_pg'], 'blk_pg': extras[pid]['blk_pg'],
            'tov_pg': extras[pid]['tov_pg'],
        }
        for pid in raw_scores
    }


def persist_sim_results_db(game_date, game_id, team_id, opponent_id, results):
    """Upserts one NbaSimResult row per player in `results` (each:
    {player_id, player_name, n_draws, sim_min, sim_median, sim_max,
    sim_mean, sim, sim_stats}), on the (game_date, player_id) unique
    constraint — overwrites the previous run for the same slate date
    rather than accumulating history, same "today's best estimate"
    convention as NFL's NflSimResult (see that model's docstring, app.py).
    Not scoped to team_id in the existing-row lookup for the same reason
    NFL's version isn't — see that function's own docstring for the
    recently-traded-player edge case this avoids."""
    import json as _json
    from app import app as flask_app, db, NbaSimResult

    with flask_app.app_context():
        existing = {r.player_id: r for r in NbaSimResult.query.filter_by(game_date=game_date).all()}
        for r in results:
            row = existing.get(r['player_id'])
            if row is None:
                row = NbaSimResult(game_date=game_date, player_id=r['player_id'])
                db.session.add(row)
            row.game_id = game_id
            row.team_id = team_id
            row.opponent_id = opponent_id
            row.player_name = r.get('player_name')
            row.n_draws = r['n_draws']
            row.sim_min = r['sim_min']
            row.sim_median = r['sim_median']
            row.sim_max = r['sim_max']
            row.sim_mean = r['sim_mean']
            row.sim_draws_json = _json.dumps(r['sim'])
            row.sim_stats_json = _json.dumps(r['sim_stats']) if r.get('sim_stats') else None
        db.session.commit()


def get_sim_results_db(game_date):
    """{game_id: {team_id: [{...}, ...]}} — every stored NbaSimResult row
    for one slate date, grouped by game then team, each team's list sorted
    by sim_median descending. Same shape/purpose as NFL's
    get_sim_results_db()."""
    import json as _json
    from app import app as flask_app, NbaSimResult

    with flask_app.app_context():
        rows = (NbaSimResult.query.filter_by(game_date=game_date)
                .order_by(NbaSimResult.sim_median.desc()).all())
        out = defaultdict(lambda: defaultdict(list))
        for r in rows:
            out[r.game_id][r.team_id].append({
                'player_id': r.player_id, 'player_name': r.player_name or f'id:{r.player_id}',
                'opponent_id': r.opponent_id, 'n_draws': r.n_draws,
                'sim_min': r.sim_min, 'sim_median': r.sim_median, 'sim_max': r.sim_max, 'sim_mean': r.sim_mean,
                'sim': _json.loads(r.sim_draws_json) if r.sim_draws_json else [],
                'sim_stats': _json.loads(r.sim_stats_json) if r.sim_stats_json else None,
            })
        return {gid: dict(teams) for gid, teams in out.items()}


def persist_sim_game_summary_db(game_date, game_id, home_team_id, away_team_id, summary):
    """Upserts one NbaSimGame row — the team-score-level companion to
    persist_sim_results_db()'s per-player rows. `summary` is
    simulate_matchup()'s own 'game' dict, passed straight through. Upserts
    on (game_date, game_id), same overwrite-previous-run convention as
    NbaSimResult."""
    import json as _json
    from app import app as flask_app, db, NbaSimGame

    with flask_app.app_context():
        row = NbaSimGame.query.filter_by(game_date=game_date, game_id=game_id).first()
        if row is None:
            row = NbaSimGame(game_date=game_date, game_id=game_id)
            db.session.add(row)
        row.home_team_id = home_team_id
        row.away_team_id = away_team_id
        row.n_sims = summary['n']
        row.home_wins = summary['home_wins']
        row.away_wins = summary['away_wins']
        row.ties = summary['ties']
        row.home_avg_score = summary['home_avg_score']
        row.away_avg_score = summary['away_avg_score']
        row.period_scores_json = _json.dumps({
            'home': summary['home_period_avg'], 'away': summary['away_period_avg'],
        })
        db.session.commit()


def get_sim_game_summary_db(game_date):
    """{game_id: {...}} — every stored NbaSimGame row for one slate date.
    Same shape/purpose as NFL's get_sim_game_summary_db(), with
    'home_period_avg'/'away_period_avg' (4 real periods, no "approximate"
    caveat needed — unlike NFL's quarter buckets, this simulator's periods
    ARE the real resampling unit, see get_simulator_quarter_pool_db())
    in place of NFL's quarter_avg keys."""
    import json as _json
    from app import app as flask_app, NbaSimGame

    with flask_app.app_context():
        rows = NbaSimGame.query.filter_by(game_date=game_date).all()
        out = {}
        for r in rows:
            ps = (_json.loads(r.period_scores_json) if r.period_scores_json
                  else {'home': [0, 0, 0, 0], 'away': [0, 0, 0, 0]})
            out[r.game_id] = {
                'n_sims': r.n_sims, 'home_wins': r.home_wins, 'away_wins': r.away_wins, 'ties': r.ties,
                'home_win_pct': round(100 * r.home_wins / r.n_sims, 1) if r.n_sims else 0.0,
                'away_win_pct': round(100 * r.away_wins / r.n_sims, 1) if r.n_sims else 0.0,
                'home_avg_score': r.home_avg_score, 'away_avg_score': r.away_avg_score,
                'home_period_avg': ps.get('home', [0, 0, 0, 0]), 'away_period_avg': ps.get('away', [0, 0, 0, 0]),
            }
        return out


def get_actual_game_score_db(game_id):
    """{'home_score', 'away_score'} for one already-Final NBA game, from
    TeamGameStat's two rows for this game_id. None if either side isn't
    stored yet. Same purpose as NFL's get_actual_game_score_db()."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = TeamGameStat.query.filter_by(sport='NBA', game_id=game_id).all()

    by_home = {r.is_home: r for r in rows}
    home_row, away_row = by_home.get(True), by_home.get(False)
    if home_row is None or away_row is None or home_row.points is None or away_row.points is None:
        return None
    return {'home_score': home_row.points, 'away_score': away_row.points}


def get_sim_slate_accuracy_db(game_date):
    """Compares `game_date`'s stored NbaSimGame rows against REAL completed
    results (TeamGameStat) — same shape/semantics as NFL's
    get_sim_week_accuracy_db(), see that function's own docstring for the
    full explanation of each returned field."""
    from app import app as flask_app, NbaSimGame, TeamGameStat

    with flask_app.app_context():
        sims = NbaSimGame.query.filter_by(game_date=game_date).all()
        if not sims:
            return {'games_final': 0, 'win_pick_correct': 0, 'win_pick_accuracy': None,
                    'brier': None, 'mean_abs_score_error': None}
        game_ids = [s.game_id for s in sims]
        real_rows = {
            r.game_id: r for r in
            TeamGameStat.query.filter(TeamGameStat.sport == 'NBA', TeamGameStat.game_id.in_(game_ids),
                                       TeamGameStat.is_home.is_(True)).all()
        }

    correct = total = 0
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
