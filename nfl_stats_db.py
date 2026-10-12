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
    nfl_player_model/nfl_fantasy — no further DB calls needed."""
    from app import app as flask_app, InjuryStatus, TeamGameStat, PlayerGameStat

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

        player_q = PlayerGameStat.query.filter(PlayerGameStat.sport == 'NFL',
                                                PlayerGameStat.player_id.in_(player_ids))
        if before_date:
            player_q = player_q.filter(PlayerGameStat.game_date < before_date)
        player_rows = player_q.order_by(PlayerGameStat.game_date.desc()).all() if player_ids else []

        team_player_q = PlayerGameStat.query.filter(PlayerGameStat.sport == 'NFL',
                                                     PlayerGameStat.team_id.in_(team_ids))
        if before_date:
            team_player_q = team_player_q.filter(PlayerGameStat.game_date < before_date)
        team_player_rows = team_player_q.order_by(PlayerGameStat.game_date.desc()).all() if team_ids else []

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

    # Each team's most recent `window` game_ids, in order — TeamGameStat
    # has exactly one row per (team, game), already date-desc, so this is
    # just a straight slice per team.
    recent_game_ids = defaultdict(list)
    points_allowed_by_game = {}
    for r in team_rows:
        if len(recent_game_ids[r.team_id]) < window:
            recent_game_ids[r.team_id].append(r.game_id)
        points_allowed_by_game[(r.team_id, r.game_id)] = r.points_allowed

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
