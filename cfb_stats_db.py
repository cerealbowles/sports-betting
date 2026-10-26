"""
cfb_stats_db.py — Read/write layer for CFB's local stats warehouse: the
same sport-agnostic TeamGameStat table (app.py) nfl_stats_db.py/
nba_stats_db.py write to, just the CFB side of it.

cfb_api.py has no equivalent today — _get_season_game_log() re-walks up to
15 weeks of live ESPN scoreboard on every 1-hour cache miss just to get
PPG/recent-form/rest-days/strength-of-schedule inputs. A Final game's
result never changes, so there's no reason to keep re-fetching it — same
reasoning as nfl_stats_db.py's identical module docstring.

Same lazy `from app import ...` pattern as nfl_stats_db.py and every
*_backfill.py script — avoids a circular import with app.py, which
imports cfb_api.py (and, transitively, this module) at module level.
"""
GAME_TYPE_BY_ESPN_SEASON_TYPE = {1: 'preseason', 2: 'regular', 3: 'postseason'}


def ingest_team_game(event_id, game_date_et, season, game_type,
                      home_id, away_id, home_name, away_name, home_score, away_score):
    """Upserts one TeamGameStat row per side for a Final CFB game. No
    boxscore fetch needed — the final score off the scoreboard event
    cfb_api._build_game() already has is everything a team-level row
    needs. Idempotent: upserts on the (sport, game_id, team_id) unique
    constraint, safe to call repeatedly. See nfl_stats_db.ingest_team_game,
    identical approach."""
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
                sport='CFB', game_id=event_id, team_id=team_id).first()
            tt = existing or TeamGameStat(sport='CFB', game_id=event_id, team_id=team_id)
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
    """Returns every stored CFB game for `season` in the same shape
    cfb_api._get_season_game_log()'s live ESPN walk produces:
    [{game_date, home_name, away_name, home_score, away_score, home_won}].
    [] if this table has nothing for the season yet (caller falls back to
    the live walk). Does NOT include home_conf/away_conf/home_rank/
    away_rank — those drive _opponent_weight (PPG/form/SoS weighting) and
    aren't stored on TeamGameStat, so callers needing them still fall back
    to the live walk until/unless that's added. See nba_stats_db.py's
    identical-approach docstring."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query
                .filter_by(sport='CFB', season=season, game_type=game_type, is_home=True)
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
