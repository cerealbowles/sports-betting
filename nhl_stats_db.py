"""
nhl_stats_db.py — Read/write layer for NHL's local stats warehouse: the
same sport-agnostic PlayerGameStat/TeamGameStat/InjuryStatus tables
(app.py) the other sports write to, NHL side.

Keyed throughout by ESPN's own team/player ids (ingested from
nhl_boxscore_api.py's parsing of ESPN's summary endpoint) — NOT the
official NHL API's own ids/abbreviations that nhl_api.py's existing
display logic (team records, standings, goalie-of-the-night) already
uses. Bridging those two id spaces (by team abbreviation, which is
standardized across both providers) is nhl_api.py's job, at the one call
site that needs both — see that module's _get_espn_event_map().

Player stats are stored as JSON (stats_json), same reasoning as
nfl_stats_db.py: a skater's and a goalie's box score lines share almost
no columns (goals/assists/shots vs. goals-against/saves/save%), so one
flat row shape doesn't fit either role well. `position` holds ESPN's own
label (e.g. 'Center', 'Goalie') — reliably present for NHL, unlike NBA/NFL.
"""
import json

GAME_TYPE_BY_ESPN_SEASON_TYPE = {1: 'preseason', 2: 'regular', 3: 'postseason'}


def ingest_team_game(event_id, game_date_et, season, game_type,
                      home_id, away_id, home_name, away_name, home_score, away_score):
    """Upserts one TeamGameStat row per side for a Final NHL game — no
    boxscore needed, the scoreboard event's own final score is enough.
    Mirrors nfl_stats_db.ingest_team_game()."""
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
            existing = TeamGameStat.query.filter_by(sport='NHL', game_id=event_id, team_id=team_id).first()
            tt = existing or TeamGameStat(sport='NHL', game_id=event_id, team_id=team_id)
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


def ingest_player_stats(event_id, game_date_et, season, game_type, home_id, away_id, boxscore):
    """boxscore: {team_id: [{id, name, position, is_goalie, stats}]} as
    returned by nhl_boxscore_api.parse_player_boxscore()/get_live_boxscore().
    Upserts PlayerGameStat rows (sport='NHL'), storing `stats` as JSON."""
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
                    sport='NHL', game_id=event_id, player_id=r['id']).first()
                target = existing or PlayerGameStat(sport='NHL', game_id=event_id, player_id=r['id'])
                target.game_date   = game_date_et
                target.season      = season
                target.game_type   = game_type
                target.player_name = r['name']
                target.team_id     = team_id
                target.opponent_id = opp_id
                target.is_home     = is_home
                target.position    = r.get('position')
                target.stats_json  = json.dumps(r.get('stats', {}))
                if not existing:
                    db.session.add(target)
        db.session.commit()


def get_schedule_db(season, game_type='regular'):
    """Returns every stored game for `season` as [{event_id, game_date,
    home_id, away_id, home_score, away_score}], chronological — same
    approach as nba_stats_db.get_schedule_db()."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query
                .filter_by(sport='NHL', season=season, game_type=game_type, is_home=True)
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


def get_season_game_log_db(season, game_type='regular'):
    """Returns [{game_date, home_name, away_name, home_score, away_score,
    home_won}] for `season` — the DB equivalent of nhl_api.py's own
    NHL-API-sourced recent-form fetch, built instead from the ESPN-sourced
    warehouse. NOT currently wired into nhl_api.py's existing form/rest-
    days logic (that already reads cheaply from the official NHL API
    directly) — kept here for parity with the other sports' stats_db
    modules and for ad-hoc querying."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query
                .filter_by(sport='NHL', season=season, game_type=game_type, is_home=True)
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


def get_game_boxscore_db(game_id):
    """{team_id: [{id, name, position, is_goalie, stats}]} for one game —
    DB equivalent of nhl_boxscore_api.get_live_boxscore(). {} if nothing
    stored."""
    from app import app as flask_app, PlayerGameStat

    with flask_app.app_context():
        rows = PlayerGameStat.query.filter_by(sport='NHL', game_id=game_id).all()

    out = {}
    for r in rows:
        try:
            stats = json.loads(r.stats_json) if r.stats_json else {}
        except (TypeError, ValueError):
            stats = {}
        out.setdefault(r.team_id, []).append({
            'id': r.player_id, 'name': r.player_name, 'position': r.position,
            'is_goalie': r.position == 'Goalie', 'stats': stats,
        })
    return out


def get_team_games_with_players_db(team_id, before_date=None, limit=10):
    """Returns this team's most recent games, newest first, each a list of
    [{player_id, player_name, position, is_goalie, stats}, ...] — per-
    player identity preserved (not collapsed to a team total), so a
    specific skater/goalie's own contribution can be isolated. Wider
    default window than football's (10 vs 5) — hockey teams play far more
    games per season, and per-player scoring is noisier game-to-game."""
    from app import app as flask_app, PlayerGameStat

    with flask_app.app_context():
        q = PlayerGameStat.query.filter_by(sport='NHL', team_id=team_id)
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
            stats = json.loads(r.stats_json) if r.stats_json else {}
        except (TypeError, ValueError):
            stats = {}
        by_game[r.game_id].append({
            'player_id': r.player_id, 'player_name': r.player_name,
            'position': r.position, 'is_goalie': r.position == 'Goalie', 'stats': stats,
        })

    return [by_game[gid] for gid in game_order[:limit]]


def get_player_gamelog_db(player_id, before_date=None, limit=None):
    """Returns this player's stored games, newest first: [{event_id, date,
    team_id, opponent_id, is_home, position, stats}]. [] if nothing stored."""
    from app import app as flask_app, PlayerGameStat

    with flask_app.app_context():
        q = PlayerGameStat.query.filter_by(sport='NHL', player_id=player_id)
        if before_date:
            q = q.filter(PlayerGameStat.game_date < before_date)
        q = q.order_by(PlayerGameStat.game_date.desc())
        if limit:
            q = q.limit(limit)
        rows = q.all()

    out = []
    for r in rows:
        try:
            stats = json.loads(r.stats_json) if r.stats_json else {}
        except (TypeError, ValueError):
            stats = {}
        out.append({
            'event_id': r.game_id, 'date': r.game_date, 'team_id': r.team_id,
            'opponent_id': r.opponent_id, 'is_home': r.is_home,
            'position': r.position, 'stats': stats,
        })
    return out


# ── Injury status — reuses app.py's shared InjuryStatus table (sport='NHL') ──

def ingest_injury_report(team_id, injuries, date_et):
    """injuries: [{id, name, pos, status}] (nhl_boxscore_api.get_injury_statuses()
    returns {player_id: status} instead — nhl_api.py's call site adapts that
    into this shape, same as nfl_api.py's identical call). Upserts on
    (sport, date, player_id)."""
    if not injuries:
        return
    from app import app as flask_app, db, InjuryStatus

    with flask_app.app_context():
        for inj in injuries:
            pid = inj.get('id')
            if not pid:
                continue
            existing = InjuryStatus.query.filter_by(sport='NHL', date=date_et, player_id=pid).first()
            row = existing or InjuryStatus(sport='NHL', date=date_et, player_id=pid)
            row.team_id     = team_id
            row.player_name = inj.get('name', '')
            row.position    = inj.get('pos', '')
            row.status      = inj.get('status', '')
            if not existing:
                db.session.add(row)
        db.session.commit()


def has_injury_snapshot(team_id, date_et):
    from app import app as flask_app, InjuryStatus
    with flask_app.app_context():
        return InjuryStatus.query.filter_by(sport='NHL', team_id=team_id, date=date_et).first() is not None


def get_latest_injury_status(team_id, as_of_date=None):
    """Most recent status per player on or before as_of_date — same
    deliberately-not-exact-date-match reasoning as nfl_stats_db's."""
    from app import app as flask_app, InjuryStatus

    with flask_app.app_context():
        q = InjuryStatus.query.filter_by(sport='NHL', team_id=team_id)
        if as_of_date:
            q = q.filter(InjuryStatus.date <= as_of_date)
        rows = q.order_by(InjuryStatus.date.desc()).all()

    latest = {}
    for r in rows:
        if r.player_id not in latest:
            latest[r.player_id] = r.status
    return latest


_OUT_STATUSES = {'out', 'injured reserve', 'suspension', 'suspended'}


def get_unavailable_player_ids(team_id, as_of_date=None):
    statuses = get_latest_injury_status(team_id, as_of_date)
    return {pid for pid, status in statuses.items() if status.lower() in _OUT_STATUSES}
