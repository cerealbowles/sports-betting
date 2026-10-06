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
GAME_TYPE_BY_ESPN_SEASON_TYPE = {1: 'preseason', 2: 'regular', 3: 'postseason'}


def ingest_game(event_id, game_date_et, season, game_type, home_id, away_id,
                 home_score, away_score, boxscore, home_name=None, away_name=None):
    """Upserts PlayerGameStat rows for every athlete ESPN's boxscore lists
    for this game (did-not-play entries included — useful for seeing who
    was active/inactive, and harmless since get_player_gamelog_db() below
    filters them back out) and one TeamGameStat row per side.

    boxscore: {team_id: [row, ...]} as returned by
    nba_boxscore_api.get_live_boxscore(event_id) — must be the Live/Final
    shape (has 'starter'/'did_not_play'/stat keys), not empty/pre-game.

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
                if not existing_team:
                    db.session.add(tt)

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

    Reads only the is_home=True TeamGameStat row per game — each finalized
    game writes one row per side, and the home side's row already carries
    both teams' names/scores (team_name/points = home, opponent_name/
    points_allowed = away), so the away side's row would just be the exact
    same game pair restated from the other row's perspective.
    """
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query
                .filter_by(sport='NBA', season=season, game_type=game_type, is_home=True)
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
            'fga':       r.fga or 0.0,
            'fta':       r.fta or 0.0,
            'tov':       r.turnovers or 0.0,
        })
    return out
