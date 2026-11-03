"""
cbb_stats_db.py — Read/write layer for men's college basketball in the
local stats warehouse (app.py's TeamGameStat table, sport='CBB').

Same role as wnba_stats_db.py / nba_stats_db.py with one difference in
where the rows come from. NBA/WNBA ingest each Final game from the per-game
ESPN `summary` box score they already fetch for the player model. CBB
slates run 50-150 games a night, so the schedule build doesn't fetch a
summary per game; instead every TeamGameStat row is written from the
scoreboard's own team totals (cbb_rating.parse_event's flat game dict),
which cbb_api.py already has in hand — zero extra calls:
  - cbb_api.build_schedule_context(): every Final game on the page.
  - cbb_stats_backfill.py: bulk seed / catch-up from cbb_rating's season
    caches (instance/cbb_rating_games_<season>.json).

Shooting/rebounding/assist totals (get_team_stat_totals_db) are read from
those same season caches rather than PlayerGameStat, since no CBB player
box scores are stored (yet). The scoreboard carries no turnovers or
offensive rebounds, so 'topg' is always None and TeamGameStat.oreb/tov
stay null for CBB.

Lazy `from app import ...` inside each function, same reason as every
other *_stats_db.py (app.py imports this module's callers at top level).
"""
from collections import defaultdict

import cbb_rating

SPORT = 'CBB'
GAME_TYPE_BY_ESPN_SEASON_TYPE = {1: 'preseason', 2: 'regular', 3: 'postseason'}


def ingest_games(games):
    """Upserts two TeamGameStat rows (one per side) for each completed game
    dict in `games` (cbb_rating.parse_event shape). Idempotent — matched on
    the (sport, game_id, team_id) unique constraint. Returns rows written."""
    games = [g for g in games or [] if g and g.get('home_id') and g.get('away_id')]
    if not games:
        return 0
    from app import app as flask_app, db, TeamGameStat

    n = 0
    with flask_app.app_context():
        ids = [g['id'] for g in games]
        existing = {}
        for i in range(0, len(ids), 500):
            for row in TeamGameStat.query.filter(TeamGameStat.sport == SPORT,
                                                 TeamGameStat.game_id.in_(ids[i:i + 500])).all():
                existing[(row.game_id, row.team_id)] = row
        for g in games:
            game_type = GAME_TYPE_BY_ESPN_SEASON_TYPE.get(g.get('season_type'), 'regular')
            for side, opp, is_home in (('home', 'away', True), ('away', 'home', False)):
                tid = str(g[f'{side}_id'])
                row = existing.get((g['id'], tid))
                if row is None:
                    row = TeamGameStat(sport=SPORT, game_id=g['id'], team_id=tid)
                    db.session.add(row)
                row.game_date = g['date']
                row.season = g.get('season')
                row.game_type = game_type
                row.team_name = g[side]
                row.opponent_id = str(g[f'{opp}_id'])
                row.opponent_name = g[opp]
                # Neutral-site games still have an ESPN-designated "home"
                # side; is_home follows it so the one-row-per-game readers
                # below keep working. Use the season cache's 'neutral' flag
                # where home/road actually matters.
                row.is_home = is_home
                row.points = g[f'{side}_score']
                row.points_allowed = g[f'{opp}_score']
                row.won = g[f'{side}_score'] > g[f'{opp}_score']
                row.fga = g.get(f'{side}_fga')
                row.fta = g.get(f'{side}_fta')
                n += 1
        db.session.commit()
    return n


def get_team_game_log_db(team_id, before_date=None, limit=None):
    """This team's finalized games, newest first: [{date, points,
    points_allowed, won, opponent_id, is_home}]."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        q = TeamGameStat.query.filter_by(sport=SPORT, team_id=str(team_id))
        if before_date:
            q = q.filter(TeamGameStat.game_date < before_date)
        q = q.order_by(TeamGameStat.game_date.desc())
        if limit:
            q = q.limit(limit)
        return [{'date': r.game_date, 'points': r.points, 'points_allowed': r.points_allowed,
                 'won': r.won, 'opponent_id': r.opponent_id, 'is_home': r.is_home} for r in q.all()]


def get_season_game_log_db(season, game_type='regular'):
    """Every stored game for `season`, one row per game, in the shape
    cbb_api's season-log consumers use: [{game_date, home_name, away_name,
    home_score, away_score, home_won}]. See wnba_stats_db's identical reader
    for the game_type (single value or tuple) convention."""
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        q = TeamGameStat.query.filter_by(sport=SPORT, season=season, is_home=True)
        if isinstance(game_type, (list, tuple, set)):
            q = q.filter(TeamGameStat.game_type.in_(game_type))
        else:
            q = q.filter_by(game_type=game_type)
        return [{'game_date': r.game_date, 'home_name': r.team_name, 'away_name': r.opponent_name,
                 'home_score': r.points, 'away_score': r.points_allowed, 'home_won': r.won}
                for r in q.order_by(TeamGameStat.game_date.asc()).all()
                if r.team_name and r.opponent_name]


def get_teams_with_data_db(season, game_type='regular'):
    """Distinct (team_id, team_name) with at least one stored game in
    `season`, D1 programs only (non-D1 opponents are stored on their D1
    opponent's game but excluded here so ranks are 'of ~365')."""
    d1 = {tid for tid, _name, conf in _season_teams(season) if conf != cbb_rating.OTHER_CONF}
    from app import app as flask_app, TeamGameStat

    with flask_app.app_context():
        rows = (TeamGameStat.query.filter_by(sport=SPORT, season=season, game_type=game_type)
                .with_entities(TeamGameStat.team_id, TeamGameStat.team_name).distinct().all())
    return [(r[0], r[1]) for r in rows if r[0] and (not d1 or r[0] in d1)]


def _season_teams(season):
    seen = {}
    for g in cbb_rating.load_season(season, allow_fetch=False):
        for side in ('home', 'away'):
            seen[str(g[f'{side}_id'])] = (g[side], g[f'{side}_conf'])
    return [(tid, name, conf) for tid, (name, conf) in seen.items()]


_index_cache = {}   # season -> (n_games, {team_id: [(date, side, opp, game)] newest first})


def _team_index(season):
    games = cbb_rating.load_season(season, allow_fetch=False)
    cached = _index_cache.get(season)
    if cached and cached[0] == len(games):
        return cached[1]
    idx = defaultdict(list)
    for g in games:
        for side, opp in (('home', 'away'), ('away', 'home')):
            idx[str(g[f'{side}_id'])].append((g['date'], side, opp, g))
    for rows in idx.values():
        rows.sort(key=lambda r: r[0], reverse=True)
    _index_cache[season] = (len(games), dict(idx))
    return _index_cache[season][1]


def get_team_stat_totals_db(team_id, before_date=None, limit=20, season=None):
    """Per-game shooting/rebounding/playmaking for one team over its last
    `limit` games before `before_date` — same return shape as
    wnba_stats_db.get_team_stat_totals_db(), computed from the scoreboard
    team totals in cbb_rating's season cache. topg is always None (the
    scoreboard has no turnovers)."""
    team_id = str(team_id)
    if season is None:
        season = cbb_rating.season_for_date(before_date) if before_date else None
    if not season:
        return None
    rows = [r for r in _team_index(season).get(team_id, [])
            if not before_date or r[0] < before_date]
    if limit:
        rows = rows[:limit]
    if not rows:
        return None
    t = defaultdict(float)
    for _, side, opp, g in rows:
        t['pts'] += g[f'{side}_score']
        t['pa'] += g[f'{opp}_score']
        for k in ('fgm', 'fga', 'tpm', 'tpa', 'ftm', 'fta', 'reb', 'ast'):
            t[k] += g.get(f'{side}_{k}') or 0
    n = len(rows)
    return {
        'games_played': n,
        'ppg': round(t['pts'] / n, 1),
        'ppg_allowed': round(t['pa'] / n, 1),
        'fg_pct': round(100 * t['fgm'] / t['fga'], 1) if t['fga'] else None,
        'three_pct': round(100 * t['tpm'] / t['tpa'], 1) if t['tpa'] else None,
        'rpg': round(t['reb'] / n, 1),
        'apg': round(t['ast'] / n, 1),
        'topg': None,
    }
