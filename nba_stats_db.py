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
                 home_score, away_score, boxscore):
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
        for team_id, opp_id, team_score, opp_score, is_home in (
            (home_id, away_id, home_score, away_score, True),
            (away_id, home_id, away_score, home_score, False),
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
                tt.opponent_id    = opp_id
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
