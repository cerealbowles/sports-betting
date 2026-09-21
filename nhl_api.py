import time
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import odds_api
import injuries_api
import nhl_model

_ET = ZoneInfo('America/New_York')

def _today_et():
    return datetime.now(_ET).strftime('%Y-%m-%d')

NHL_API = "https://api-web.nhle.com/v1"

_cache = {}
_TTL = {
    'schedule': 1800,   # 30 min
    'standings': 3600,  # 1 hour
    'recent':   3600,   # 1 hour
    'goalie':   21600,  # 6 hours
}


def _cached_get(url, key, ttl):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < ttl:
            return data
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return _cache[key][0] if key in _cache else None
    _cache[key] = (data, now)
    return data


def _period_label(pd):
    ptype = (pd or {}).get('periodType', 'REG')
    if ptype == 'SO':
        return 'Shootout'
    if ptype == 'OT':
        return 'OT'
    num = (pd or {}).get('number')
    return f"P{num}" if num else ''


def _live_clock_map():
    """{game_id: 'P2 · 13:48'} for in-progress games. The schedule endpoint
    carries the period but no clock; scoreboard/now has both, so live cards
    combine them. Short cache since the clock changes every second."""
    data = _cached_get(f"{NHL_API}/scoreboard/now", 'nhl_scoreboard_now', 15)
    out = {}
    for day in (data or {}).get('gamesByDate', []):
        for g in day.get('games', []):
            if g.get('gameState') not in ('LIVE', 'CRIT'):
                continue
            label = _period_label(g.get('periodDescriptor'))
            clock = g.get('clock') or {}
            if label == 'Shootout':
                text = label
            elif clock.get('inIntermission'):
                text = f"{label} · INT" if label else 'INT'
            elif clock.get('timeRemaining'):
                text = f"{label} · {clock['timeRemaining']}" if label else clock['timeRemaining']
            else:
                text = label
            out[g.get('id')] = text
    return out


def _get_schedule_raw():
    return _cached_get(f"{NHL_API}/schedule/now", 'nhl_sched_now', _TTL['schedule'])


def get_live_scores(date_str=None):
    """
    Returns {game_key: score_dict} for NHL games on `date_str` (YYYY-MM-DD ET,
    defaults to today) with a 2-min cache. Passing a past date lets an open bet
    from a prior day keep showing its final score until the bet is closed.
    """
    from odds_api import _normalize
    target = date_str or datetime.now(_ET).strftime('%Y-%m-%d')
    url    = f"{NHL_API}/schedule/now" if not date_str else f"{NHL_API}/schedule/{target}"
    data   = _cached_get(url, f'nhl_scores_{target}', 120)
    scores = {}
    if not data:
        return scores
    clocks = _live_clock_map()
    for day in data.get('gameWeek', []):
        if day.get('date') != target:
            continue
        for game in day.get('games', []):
            a_data = game.get('awayTeam', {})
            h_data = game.get('homeTeam', {})
            h_name = h_data.get('name', {}).get('default', h_data.get('abbrev', ''))
            a_name = a_data.get('name', {}).get('default', a_data.get('abbrev', ''))
            gk     = f"{_normalize(h_name)}_{_normalize(a_name)}"
            state  = game.get('gameState', '')
            if state in ('LIVE', 'CRIT'):
                status = 'Live'
            elif state in ('FINAL', 'OFF'):
                status = 'Final'
            else:
                status = 'Preview'
            period = None
            if status == 'Live':
                period = clocks.get(game.get('id')) or _period_label(game.get('periodDescriptor')) or None
            elif status == 'Final':
                pd    = game.get('periodDescriptor', {})
                ptype = pd.get('periodType', 'REG')
                period = 'Final/OT' if ptype == 'OT' else ('Final/SO' if ptype == 'SO' else 'Final')
            scores[gk] = {
                'status':     status,
                'away_abbr':  a_data.get('abbrev', ''),
                'home_abbr':  h_data.get('abbrev', ''),
                'away_logo':  f"https://assets.nhle.com/logos/nhl/svg/{a_data.get('abbrev', '')}_light.svg" if a_data.get('abbrev') else '',
                'home_logo':  f"https://assets.nhle.com/logos/nhl/svg/{h_data.get('abbrev', '')}_light.svg" if h_data.get('abbrev') else '',
                'away_score': a_data.get('score'),
                'home_score': h_data.get('score'),
                'period':     period,
                'away_logo':  f"https://assets.nhle.com/logos/nhl/svg/{a_data.get('abbrev', '')}_light.svg" if a_data.get('abbrev') else None,
                'home_logo':  f"https://assets.nhle.com/logos/nhl/svg/{h_data.get('abbrev', '')}_light.svg" if h_data.get('abbrev') else None,
            }
    return scores


def get_live_game_states(date_str=None):
    """
    Returns {game_key: state_dict} for NHL games on `date_str` (defaults to
    today, ET) — status, score, and a period/clock text — matching what the
    /nhl schedule page's game cards render server-side. Reuses
    get_live_scores()'s 2-min cache (same NHL API call) so polling the page
    adds no extra API traffic.
    """
    from odds_api import _normalize
    target = date_str or datetime.now(_ET).strftime('%Y-%m-%d')
    url    = f"{NHL_API}/schedule/now" if not date_str else f"{NHL_API}/schedule/{target}"
    data   = _cached_get(url, f'nhl_scores_{target}', 120)
    states = {}
    if not data:
        return states
    clocks = _live_clock_map()
    for day in data.get('gameWeek', []):
        if day.get('date') != target:
            continue
        for game in day.get('games', []):
            a_data = game.get('awayTeam', {})
            h_data = game.get('homeTeam', {})
            h_name = h_data.get('name', {}).get('default', h_data.get('abbrev', ''))
            a_name = a_data.get('name', {}).get('default', a_data.get('abbrev', ''))
            gk     = f"{_normalize(h_name)}_{_normalize(a_name)}"
            state  = game.get('gameState', '')
            if state in ('LIVE', 'CRIT'):
                status = 'Live'
            elif state in ('FINAL', 'OFF'):
                status = 'Final'
            else:
                status = 'Preview'
            clock_text = None
            if status == 'Live':
                clock_text = clocks.get(game.get('id')) or _period_label(game.get('periodDescriptor')) or None
            states[gk] = {
                'status':     status,
                'away_score': a_data.get('score'),
                'home_score': h_data.get('score'),
                'live_state': {'clock_text': clock_text} if status == 'Live' else None,
            }
    return states


def _get_standings_map():
    """Returns {abbrev: standings_row} for all NHL teams."""
    data = _cached_get(f"{NHL_API}/standings/now", 'nhl_standings', _TTL['standings'])
    result = {}
    if not data:
        return result
    for row in data.get('standings', []):
        abbrev = row.get('teamAbbrev', {}).get('default', '')
        if abbrev:
            result[abbrev] = row
    return result


def _get_recent_form_map():
    """Query past 3 weeks of completed games and build team form sequences
    plus each team's most recent game date (for back-to-back detection)."""
    today = datetime.now(timezone.utc)
    past_dates = [
        (today - timedelta(weeks=i)).strftime('%Y-%m-%d')
        for i in range(1, 4)
    ]

    def _fetch_week(date):
        return _cached_get(f"{NHL_API}/schedule/{date}", f'nhl_sched_{date}', _TTL['recent'])

    all_games = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_fetch_week, d): d for d in past_dates}
        for future in as_completed(futures):
            try:
                data = future.result()
                if data:
                    for day in data.get('gameWeek', []):
                        for game in day.get('games', []):
                            if game.get('gameState') in ('FINAL', 'OFF'):
                                all_games.append((day['date'], game))
            except Exception:
                pass

    form = {}
    last_game_date = {}
    for game_date, game in sorted(all_games, key=lambda x: x[0]):
        away = game.get('awayTeam', {})
        home = game.get('homeTeam', {})
        a_ab = away.get('abbrev', '')
        h_ab = home.get('abbrev', '')
        a_s  = away.get('score', 0) or 0
        h_s  = home.get('score', 0) or 0
        if a_ab:
            form.setdefault(a_ab, []).append('W' if a_s > h_s else 'L')
            last_game_date[a_ab] = game_date
        if h_ab:
            form.setdefault(h_ab, []).append('W' if h_s > a_s else 'L')
            last_game_date[h_ab] = game_date

    return {ab: results[-10:] for ab, results in form.items()}, last_game_date


def _rest_days(last_game_date, abbrev, today_str):
    gdate = last_game_date.get(abbrev)
    if not gdate:
        return None
    try:
        last = datetime.strptime(gdate, '%Y-%m-%d').date()
        curr = datetime.strptime(today_str, '%Y-%m-%d').date()
        return (curr - last).days
    except Exception:
        return None


def _get_team_goalie(abbrev):
    data = _cached_get(f"{NHL_API}/club-stats/{abbrev}/now", f'nhl_goalie_{abbrev}', _TTL['goalie'])
    if not data:
        return None
    goalies = data.get('goalies', [])
    if not goalies:
        return None
    goalies.sort(key=lambda g: g.get('gamesStarted', 0), reverse=True)
    g     = goalies[0]
    fname = g.get('firstName', {}).get('default', '')
    lname = g.get('lastName', {}).get('default', '')
    sv    = g.get('savePercentage', 0) or 0
    gaa   = g.get('goalsAgainstAverage', 0) or 0
    wins  = g.get('wins', 0)
    loss  = g.get('losses', 0)
    otl   = g.get('overtimeLosses', 0)
    return {
        'name':   f"{fname} {lname}".strip(),
        'sv_pct': f"{sv:.3f}",
        'gaa':    f"{gaa:.2f}",
        'record': f"{wins}-{loss}-{otl}",
    }


def _build_team_info(team_data, side, standings, form, goalie, rest_days=None, fresh_record=True):
    abbrev = team_data.get('abbrev', '')
    st     = standings.get(abbrev, {})
    gp     = st.get('gamesPlayed', 1) or 1

    if side == 'home':
        split_w = st.get('homeWins', 0)
        split_l = st.get('homeLosses', 0) + st.get('homeOtLosses', 0)
        split_label = 'Home'
    else:
        split_w = st.get('roadWins', 0)
        split_l = st.get('roadLosses', 0) + st.get('roadOtLosses', 0)
        split_label = 'Away'

    return {
        'abbrev':      abbrev,
        'logo_url':    f'https://assets.nhle.com/logos/nhl/svg/{abbrev}_light.svg',
        'name':        st.get('teamName', {}).get('default', abbrev),
        'wins':        st.get('wins', 0),
        'losses':      st.get('losses', 0),
        'ot_losses':   st.get('otLosses', 0),
        # Display record. wins/losses above stay as last season's numbers (the model
        # uses them as a prior); the card shows 0-0-0 until standings are for this season.
        'rec_w':       st.get('wins', 0) if fresh_record else 0,
        'rec_l':       st.get('losses', 0) if fresh_record else 0,
        'rec_otl':     st.get('otLosses', 0) if fresh_record else 0,
        'split_w':     split_w,
        'split_l':     split_l,
        'split_label': split_label,
        'side':        side,
        'form':        form.get(abbrev, []),
        'score':       team_data.get('score'),
        'goalie':      goalie,
        'gf_pg':       round(st.get('goalFor', 0) / gp, 2),
        'ga_pg':       round(st.get('goalAgainst', 0) / gp, 2),
        'l10':         f"{st.get('l10Wins',0)}-{st.get('l10Losses',0)}-{st.get('l10OtLosses',0)}",
        'rest_days':   rest_days,
    }


def get_today_game_count():
    """Cheap today's-game count for the sport-chip badge — reuses the same
    cached raw schedule fetch as build_schedule_context() without paying for
    any of its standings/form/odds/injury enrichment."""
    today = _today_et()
    data = _get_schedule_raw()
    if not data:
        return 0
    for day in data.get('gameWeek', []):
        if day.get('date') == today:
            return len(day.get('games', []))
    return 0


def _get_period_linescore(game_id):
    """Period-by-period goals for one game from the gamecenter boxscore
    endpoint — only meaningful once a game has started, and this is a
    per-game call (not covered by the schedule/now cache), so callers
    should only fetch it for Live/Final games. Tries the couple of key
    paths NHL's api-web has used for this and fails soft to None (linescore
    just won't render) rather than break the page if the shape drifts."""
    if not game_id:
        return None
    data = _cached_get(f"{NHL_API}/gamecenter/{game_id}/boxscore", f'nhl_box_{game_id}', 60)
    if not data:
        return None
    try:
        by_period = (data.get('boxscore') or {}).get('linescore', {}).get('byPeriod')
        if not by_period:
            by_period = data.get('linescore', {}).get('byPeriod')
        if not by_period:
            return None

        def _val(v):
            return int(v) if v is not None else None

        periods = []
        for i, p in enumerate(by_period):
            pd = p.get('periodDescriptor', {}) or {}
            ptype = pd.get('periodType', 'REG')
            num   = pd.get('number', i + 1)
            if ptype == 'OT':
                label = 'OT'
            elif ptype == 'SO':
                label = 'SO'
            else:
                label = str(num)
            periods.append({'label': label, 'away': _val(p.get('away')), 'home': _val(p.get('home'))})
        # Pad to at least 3 regulation periods so the table doesn't jump
        # around mid-game, matching the padding _parse_linescores() does
        # for the other sports.
        reg = [p for p in periods if p['label'] not in ('OT', 'SO')]
        ot  = [p for p in periods if p['label'] in ('OT', 'SO')]
        while len(reg) < 3:
            reg.append({'label': str(len(reg) + 1), 'away': None, 'home': None})
        periods = reg + ot
        return periods or None
    except Exception:
        return None


def build_schedule_context():
    """Returns today's NHL games ready for the template."""
    from odds_api import _normalize
    raw        = _get_schedule_raw()
    standings  = _get_standings_map()
    form, last_game_date = _get_recent_form_map()
    odds_map   = odds_api.get_odds_map('nhl')
    injury_map = injuries_api.get_injury_map('nhl')
    if not raw:
        return []

    today_str = _today_et()
    today_games = []
    for day in raw.get('gameWeek', []):
        if day.get('date') == today_str:
            today_games = day.get('games', [])
            break

    if not today_games:
        return []

    # Collect team abbrevs for parallel goalie fetching
    abbrevs = set()
    for game in today_games:
        abbrevs.add(game.get('awayTeam', {}).get('abbrev', ''))
        abbrevs.add(game.get('homeTeam', {}).get('abbrev', ''))
    abbrevs.discard('')

    goalies = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_get_team_goalie, ab): ab for ab in abbrevs}
        for future in as_completed(futures):
            ab = futures[future]
            try:
                goalies[ab] = future.result()
            except Exception:
                goalies[ab] = None

    live_clocks = _live_clock_map()
    games = []
    for game in today_games:
        state = game.get('gameState', '')
        if state in ('LIVE', 'CRIT'):
            status = 'Live'
        elif state in ('FINAL', 'OFF'):
            status = 'Final'
        else:
            status = 'Preview'

        a_data = game.get('awayTeam', {})
        h_data = game.get('homeTeam', {})
        a_ab   = a_data.get('abbrev', '')
        h_ab   = h_data.get('abbrev', '')

        # /standings/now still returns last season's final table during the
        # preseason (gameType 1) — don't show those as the teams' current records.
        game_season = game.get('season')
        def _fresh(ab):
            if game.get('gameType') == 1:
                return False
            sid = standings.get(ab, {}).get('seasonId')
            return not (sid and game_season and str(sid) != str(game_season))
        away = _build_team_info(a_data, 'away', standings, form, goalies.get(a_ab),
                                 _rest_days(last_game_date, a_ab, today_str), _fresh(a_ab))
        home = _build_team_info(h_data, 'home', standings, form, goalies.get(h_ab),
                                 _rest_days(last_game_date, h_ab, today_str), _fresh(h_ab))

        # Inject injury status onto goalie info
        for team in (away, home):
            if team.get('goalie') and team['goalie'].get('name'):
                team['goalie']['injury'] = injury_map.get(team['goalie']['name'].lower())

        game_odds = odds_api.lookup_game_odds(odds_map, home['name'], away['name'],
                                               game_date=game.get('startTimeUTC', '')[:13] or None)

        # Run the model
        try:
            market_home_prob = game_odds.get('home_implied') if game_odds else None
            model = nhl_model.predict(home, away, game_time_utc=game.get('startTimeUTC', ''),
                                       market_home_prob=market_home_prob)
            model['factors'].sort(key=lambda f: abs(f[1]), reverse=True)
        except Exception:
            model = None

        # Playoff series context
        series_info = None
        ss = game.get('seriesStatus')
        if ss:
            top    = ss.get('topSeedTeamAbbrev', '')
            top_w  = ss.get('topSeedWins', 0)
            bot_w  = ss.get('bottomSeedWins', 0)
            bot    = h_ab if a_ab == top else a_ab
            needed = ss.get('neededToWin', 4)
            if top_w == needed or bot_w == needed:
                winner = top if top_w == needed else bot
                series_info = f"{winner} wins series {needed}-{min(top_w, bot_w)}"
            elif top_w > bot_w:
                series_info = f"{top} leads {top_w}–{bot_w}"
            elif bot_w > top_w:
                series_info = f"{bot} leads {bot_w}–{top_w}"
            else:
                series_info = f"Series tied {top_w}–{bot_w}"

        games.append({
            'game_id':       game.get('id'),
            'game_time_utc': game.get('startTimeUTC', ''),
            'status':        status,
            'venue':         game.get('venue', {}).get('default', ''),
            'away':          away,
            'home':          home,
            'series_info':   series_info,
            # api-web gameType: 1=preseason, 2=regular season, 3=playoffs.
            'is_preseason':  game.get('gameType') == 1,
            'live_clock':    (live_clocks.get(game.get('id')) or _period_label(game.get('periodDescriptor'))) if status == 'Live' else '',
            'model':         model,
            'odds':          game_odds,
            'bet_name':      f"{a_ab} @ {h_ab}",
            'game_key':      f"{_normalize(home['name'])}_{_normalize(away['name'])}",
            'linescore':     _get_period_linescore(game.get('id')) if status in ('Live', 'Final') else None,
        })

    try:
        date_display = datetime.now(_ET).strftime('%a, %b %-d')
    except Exception:
        date_display = today_str

    return [{'date': today_str, 'date_display': date_display, 'games': games}]
