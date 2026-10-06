import time
import requests
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import odds_api
import nba_model
import bball_total_model
import nba_roster_api
import nba_player_model
import nba_ensemble_model
import nba_boxscore_api
import nba_stats_db
import spread_proxy

_ET = ZoneInfo('America/New_York')

def _today_et():
    return datetime.now(_ET).strftime('%Y-%m-%d')

def _event_date_et(date_str):
    """Convert an ESPN event's UTC ISO timestamp to its ET calendar date.
    Late tip-offs (10pm+ ET) land past midnight UTC, so comparing the raw
    UTC string against an ET date string (e.g. via startswith) misses them."""
    if not date_str:
        return ''
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.astimezone(_ET).strftime('%Y-%m-%d')
    except ValueError:
        return ''

ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

_cache = {}
# scoreboard: 120s matches nfl_api.py/get_live_scores()'s TTL — live-game
# state (clock/period/score) needs to be fresh on manual refresh.
# season_log: 1 hour — team season stats don't need live-game freshness.
_TTL = {'scoreboard': 120, 'season_log': 3600, 'prior_season_stats': 24 * 3600}


def _parse_linescores(comp, regulation_periods=4):
    """Quarter-by-quarter scoring from ESPN's per-competitor `linescores`
    array — only present once a game has started. Always includes all
    `regulation_periods` columns, padding not-yet-played quarters with None
    so the table doesn't jump around as the game progresses, plus an
    OT/OT2/... column for each overtime actually played. Values come back
    as floats from ESPN (e.g. 10.0) even though scores are always whole
    numbers — cast to int for display. Returns None pre-tipoff."""
    competitors = comp.get('competitors', [])
    home_ls = next((c.get('linescores') for c in competitors if c.get('homeAway') == 'home'), None) or []
    away_ls = next((c.get('linescores') for c in competitors if c.get('homeAway') == 'away'), None) or []
    if not home_ls and not away_ls:
        return None
    n = max(len(home_ls), len(away_ls), regulation_periods)

    def _val(arr, i):
        if i >= len(arr):
            return None
        v = arr[i].get('value')
        return int(v) if v is not None else None

    periods = []
    for i in range(n):
        if i < regulation_periods:
            label = str(i + 1)
        else:
            ot_num = i - regulation_periods + 1
            label = 'OT' if ot_num == 1 else f'OT{ot_num}'
        periods.append({'label': label, 'away': _val(away_ls, i), 'home': _val(home_ls, i)})
    return periods


def _cached_get(url, params, key, ttl):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < ttl:
            return data
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return _cache[key][0] if key in _cache else None
    _cache[key] = (data, now)
    return data


def _parse_record(records, name):
    for rec in records:
        if rec.get('name', '').lower() == name.lower():
            return rec.get('summary', '0-0')
    return '—'


def _record_to_wl(summary):
    """'6-1' → (6, 1)"""
    try:
        parts = summary.split('-')
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def _get_nba_season():
    """NBA season year (ESPN convention: `season=2024` means the 2024-25
    season) currently in progress or most recently completed. New season
    tips off in October."""
    today = datetime.now(_ET).date()
    return today.year if today.month >= 10 else today.year - 1


# Below this many current-season games played, a team's win%/split/ppg
# factors are blended with its final prior-season numbers — same fix and
# same reasoning as nfl_api.EARLY_SEASON_GAMES (see that constant's
# docstring): without it, 4 of the model's factors compute off a 0-0
# no-data default for every one of a new season's first few games. Weight
# ramps linearly to 100% current-season data by the time a team has played
# this many games.
EARLY_SEASON_GAMES = 4


# ── Season game log ────────────────────────────────────────────────────────────

def _get_season_game_log(season):
    """
    Returns list of {game_date, home_name, away_name, home_score, away_score,
    home_won} for every completed NBA regular-season game in `season`
    (ESPN's `season` year, e.g. 2024 = 2024-25) up through today.

    Reads the local stats warehouse (nba_stats_db.get_season_game_log_db,
    backed by TeamGameStat — see that module's docstring) first — every
    Final game nba_api.py itself builds gets ingested there automatically,
    and nba_stats_backfill.py catches up anything from before that wiring
    existed or missed during downtime. Only falls back to the live
    day-by-day ESPN scoreboard walk below when the DB has nothing for this
    season yet (e.g. a brand new season with no games finalized/ingested
    at all) — same "DB first, live fetch as the only-if-empty fallback"
    pattern as nba_roster_api.get_player_gamelog().
    """
    try:
        db_games = nba_stats_db.get_season_game_log_db(season)
        if db_games:
            return db_games
    except Exception:
        pass

    key = f'nba_season_log_{season}'
    now_ts = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now_ts - ts < _TTL['season_log']:
            return data

    games = []
    seen_ids = set()
    start = datetime(season, 10, 1, tzinfo=_ET).date()
    today = datetime.now(_ET).date()
    end = min(today, datetime(season + 1, 6, 30, tzinfo=_ET).date())
    day = start
    while day <= end:
        date_str = day.strftime('%Y%m%d')
        data = _cached_get(ESPN_NBA, {'dates': date_str, 'limit': 100},
                            f'nba_day_{date_str}', _TTL['season_log'])
        day += timedelta(days=1)
        if not data:
            continue
        for event in data.get('events', []):
            if event.get('id') in seen_ids:
                continue
            comp  = event.get('competitions', [{}])[0]
            state = comp.get('status', {}).get('type', {}).get('state', 'pre')
            if state != 'post':
                continue
            if event.get('season', {}).get('type') != 2:
                continue
            tmap   = {c['homeAway']: c for c in comp.get('competitors', [])}
            home_c = tmap.get('home', {})
            away_c = tmap.get('away', {})
            try:
                h_score = int(home_c.get('score', 0) or 0)
                a_score = int(away_c.get('score', 0) or 0)
                h_name  = home_c.get('team', {}).get('displayName', '')
                a_name  = away_c.get('team', {}).get('displayName', '')
                if not h_name or not a_name:
                    continue
                games.append({
                    'game_date':  event.get('date', '')[:10],
                    'home_name':  h_name,
                    'away_name':  a_name,
                    'home_score': h_score,
                    'away_score': a_score,
                    'home_won':   h_score > a_score,
                })
                seen_ids.add(event.get('id'))
            except Exception:
                pass

    _cache[key] = (games, now_ts)
    return games


def _compute_team_season_stats(games):
    """Compute {team_name: {ppg, ppg_allowed}} from season game scores."""
    pts = defaultdict(lambda: {'for': 0, 'against': 0, 'g': 0})
    for g in games:
        pts[g['home_name']]['for']     += g['home_score']
        pts[g['home_name']]['against'] += g['away_score']
        pts[g['home_name']]['g']       += 1
        pts[g['away_name']]['for']     += g['away_score']
        pts[g['away_name']]['against'] += g['home_score']
        pts[g['away_name']]['g']       += 1
    return {
        t: {
            'ppg':         round(v['for']     / v['g'], 1),
            'ppg_allowed': round(v['against'] / v['g'], 1),
        }
        for t, v in pts.items() if v['g'] > 0
    }


def _get_prior_season_team_stats(season):
    """Final wins/losses, home/road split, and ppg/ppg_allowed for every team
    from `season` (the season *before* the one currently in progress). Used
    to blend in prior-season signal for early-current-season games — see
    EARLY_SEASON_GAMES. Mirrors nfl_api._get_prior_season_team_stats."""
    key = f'nba_prior_stats_{season}'
    now_ts = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now_ts - ts < _TTL['prior_season_stats']:
            return data

    games = _get_season_game_log(season)
    wl = defaultdict(lambda: {'wins': 0, 'losses': 0, 'split_w': 0, 'split_l': 0})
    for g in games:
        h, a, h_won = g['home_name'], g['away_name'], g['home_won']
        wl[h]['wins']    += 1 if h_won else 0
        wl[h]['losses']  += 0 if h_won else 1
        wl[h]['split_w'] += 1 if h_won else 0
        wl[h]['split_l'] += 0 if h_won else 1
        wl[a]['wins']    += 0 if h_won else 1
        wl[a]['losses']  += 1 if h_won else 0
        wl[a]['split_w'] += 0 if h_won else 1
        wl[a]['split_l'] += 1 if h_won else 0

    ppg = _compute_team_season_stats(games)
    result = {
        t: {**v, 'ppg': ppg.get(t, {}).get('ppg'), 'ppg_allowed': ppg.get(t, {}).get('ppg_allowed')}
        for t, v in wl.items()
    }
    _cache[key] = (result, now_ts)
    return result


def _apply_prior_season_blend(team, prior_stats):
    """When `team` has played fewer than EARLY_SEASON_GAMES games this
    season, blend its current-season win%/split%/ppg/ppg_allowed with last
    season's final numbers, weighted toward current-season data as more of
    it accumulates. Sets 'blend_*' keys consumed by nba_model.predict();
    leaves the raw wins/losses/ppg fields untouched so the game card still
    displays the team's actual current-season record. Mirrors
    nfl_api._apply_prior_season_blend."""
    cur_games = team.get('wins', 0) + team.get('losses', 0)
    if cur_games >= EARLY_SEASON_GAMES:
        return
    prior = prior_stats.get(team.get('name', ''))
    if not prior:
        return

    w = cur_games / EARLY_SEASON_GAMES  # weight given to current-season data

    cur_win_pct = team['wins'] / cur_games if cur_games else 0.5
    prior_total = prior['wins'] + prior['losses']
    prior_win_pct = prior['wins'] / prior_total if prior_total else 0.5
    team['blend_win_pct'] = w * cur_win_pct + (1 - w) * prior_win_pct

    cur_split_total = team.get('split_w', 0) + team.get('split_l', 0)
    cur_split_pct = team['split_w'] / cur_split_total if cur_split_total else 0.5
    prior_split_total = prior['split_w'] + prior['split_l']
    prior_split_pct = prior['split_w'] / prior_split_total if prior_split_total else 0.5
    team['blend_split_pct'] = w * cur_split_pct + (1 - w) * prior_split_pct

    cur_ppg = team.get('ppg')
    if prior.get('ppg') is not None:
        team['blend_ppg'] = w * cur_ppg + (1 - w) * prior['ppg'] if cur_ppg is not None else prior['ppg']
    cur_ppga = team.get('ppg_allowed')
    if prior.get('ppg_allowed') is not None:
        team['blend_ppg_allowed'] = (
            w * cur_ppga + (1 - w) * prior['ppg_allowed'] if cur_ppga is not None else prior['ppg_allowed']
        )


def _team_recent_form(games, team_name, n=5):
    """Last n W/L results for team_name from game_log, newest first."""
    tg = []
    for g in games:
        if g['home_name'] == team_name:
            tg.append((g['game_date'], 'W' if g['home_won'] else 'L'))
        elif g['away_name'] == team_name:
            tg.append((g['game_date'], 'W' if not g['home_won'] else 'L'))
    tg.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in tg[:n]]


def _team_rest_days(games, team_name, game_date):
    """Days between team_name's last completed game and game_date."""
    past = [g['game_date'] for g in games
            if (g['home_name'] == team_name or g['away_name'] == team_name)
            and g['game_date'] < game_date]
    if not past:
        return None
    try:
        last = datetime.strptime(max(past), '%Y-%m-%d').date()
        curr = datetime.strptime(game_date, '%Y-%m-%d').date()
        return (curr - last).days
    except Exception:
        return None


# ── Live scores ────────────────────────────────────────────────────────────────

def get_live_scores(date_str=None):
    """
    Returns {game_key: score_dict} for NBA games on `date_str` (YYYY-MM-DD ET,
    defaults to today) with a 2-min cache. Passing a past date lets an open bet
    from a prior day keep showing its final score until the bet is closed.
    """
    from odds_api import _normalize
    target_str = date_str or _today_et()
    params = {'dates': target_str.replace('-', '')} if date_str else {}
    data = _cached_get(ESPN_NBA, params, f'nba_scores_{target_str}', 120)
    scores = {}
    if not data:
        return scores
    for event in data.get('events', []):
        if _event_date_et(event.get('date', '')) != target_str:
            continue
        comp      = event.get('competitions', [{}])[0]
        status_obj = comp.get('status', {}).get('type', {})
        state     = status_obj.get('state', 'pre')
        if state == 'in':
            status = 'Live'
        elif state == 'post':
            status = 'Final'
        else:
            status = 'Preview'
        teams = {}
        for competitor in comp.get('competitors', []):
            side   = competitor.get('homeAway', 'home')
            team   = competitor.get('team', {})
            teams[side] = {
                'name':  team.get('displayName', ''),
                'abbr':  team.get('abbreviation', ''),
                'score': competitor.get('score'),
            }
        home = teams.get('home', {})
        away = teams.get('away', {})
        if not home or not away:
            continue
        gk = f"{_normalize(home['name'])}_{_normalize(away['name'])}"
        period = None
        if status == 'Live':
            period = status_obj.get('detail', '')
        elif status == 'Final':
            period = 'Final'
        scores[gk] = {
            'status':     status,
            'away_abbr':  away.get('abbr', ''),
            'home_abbr':  home.get('abbr', ''),
            'away_logo':  f"https://a.espncdn.com/i/teamlogos/nba/500/{away.get('abbr', '').lower()}.png" if away.get('abbr') else '',
            'home_logo':  f"https://a.espncdn.com/i/teamlogos/nba/500/{home.get('abbr', '').lower()}.png" if home.get('abbr') else '',
            'away_score': away.get('score'),
            'home_score': home.get('score'),
            'period':     period,
            'away_logo':  f"https://a.espncdn.com/i/teamlogos/nba/500/{away.get('abbr', '').lower()}.png" if away.get('abbr') else None,
            'home_logo':  f"https://a.espncdn.com/i/teamlogos/nba/500/{home.get('abbr', '').lower()}.png" if home.get('abbr') else None,
        }
    return scores


def get_live_game_states(date_str=None):
    """
    Returns {game_key: state_dict} for NBA games on `date_str` (defaults to
    today, ET) — status, score, and a clock/period text — matching what the
    /nba schedule page's game cards render server-side. Reuses
    get_live_scores()'s 2-min cache (same ESPN call) so polling the page
    adds no extra API traffic.
    """
    from odds_api import _normalize
    target_str = date_str or _today_et()
    params = {'dates': target_str.replace('-', '')} if date_str else {}
    data = _cached_get(ESPN_NBA, params, f'nba_scores_{target_str}', 120)
    states = {}
    if not data:
        return states
    for event in data.get('events', []):
        if _event_date_et(event.get('date', '')) != target_str:
            continue
        comp       = event.get('competitions', [{}])[0]
        status_obj = comp.get('status', {}).get('type', {})
        state      = status_obj.get('state', 'pre')
        if state == 'in':
            status = 'Live'
        elif state == 'post':
            status = 'Final'
        else:
            status = 'Preview'
        teams = {}
        for competitor in comp.get('competitors', []):
            side = competitor.get('homeAway', 'home')
            team = competitor.get('team', {})
            teams[side] = {'name': team.get('displayName', ''), 'score': competitor.get('score')}
        home = teams.get('home', {})
        away = teams.get('away', {})
        if not home or not away:
            continue
        gk = f"{_normalize(home['name'])}_{_normalize(away['name'])}"
        clock_text = status_obj.get('detail') if status == 'Live' else None
        states[gk] = {
            'status':     status,
            'away_score': away.get('score'),
            'home_score': home.get('score'),
            'live_state': {'clock_text': clock_text} if status == 'Live' else None,
        }
    return states


# ── Schedule context ───────────────────────────────────────────────────────────

def _build_game(event, team_stats, game_log, nba_odds_map, prior_stats=None):
    """Builds one game's full display dict — teams, live state, model, odds.
    Shared by build_schedule_context() (the /nba page and the daily-digest
    cache warmer)."""
    from odds_api import _normalize

    comp = event.get('competitions', [{}])[0]
    venue_obj = comp.get('venue', {})
    venue = venue_obj.get('fullName', '')
    if venue_obj.get('city'):
        venue += f", {venue_obj['city']}"

    status_obj = comp.get('status', {}).get('type', {})
    state = status_obj.get('state', 'pre')
    if state == 'in':
        status = 'Live'
    elif state == 'post':
        status = 'Final'
    else:
        status = 'Preview'

    teams = {}
    for competitor in comp.get('competitors', []):
        side = competitor.get('homeAway', 'home')
        team = competitor.get('team', {})
        records = competitor.get('records', [])
        overall = _parse_record(records, 'overall')
        home_r  = _parse_record(records, 'Home')
        away_r  = _parse_record(records, 'Road')
        w, l = _record_to_wl(overall)

        if side == 'home':
            split_summary = home_r
            split_label   = 'Home'
        else:
            split_summary = away_r
            split_label   = 'Away'

        split_w, split_l = _record_to_wl(split_summary)

        abbrev = team.get('abbreviation', '')
        teams[side] = {
            'id':          team.get('id'),
            'name':        team.get('displayName', ''),
            'abbrev':      abbrev,
            'abbr':        abbrev,
            'logo_url':    f"https://a.espncdn.com/i/teamlogos/nba/500/{abbrev.lower()}.png",
            'wins':        w,
            'losses':      l,
            'split_w':     split_w,
            'split_l':     split_l,
            'split_label': split_label,
            'side':        side,
            'form':        [],
            'score':       competitor.get('score'),
            'ppg':         None,
            'ppg_allowed': None,
            'rest_days':   None,
            'injuries':    [],
        }

    away = teams.get('away', {})
    home = teams.get('home', {})

    # Enrich with season stats, recent form, and rest days entering *this*
    # game's date.
    game_date_et = _event_date_et(event.get('date', ''))
    for team in (home, away):
        name = team.get('name', '')
        ts   = team_stats.get(name, {})
        team['ppg']         = ts.get('ppg')
        team['ppg_allowed'] = ts.get('ppg_allowed')
        team['form']        = _team_recent_form(game_log, name)
        team['rest_days']   = _team_rest_days(game_log, name, game_date_et)
        if prior_stats:
            _apply_prior_season_blend(team, prior_stats)

    game_odds = odds_api.lookup_game_odds(nba_odds_map, home.get('name', ''), away.get('name', ''),
                                           game_date=event.get('date', '')[:13] or None)

    # Run the model
    try:
        market_home_prob = game_odds.get('home_implied') if game_odds else None
        model = nba_model.predict(home, away, game_time_utc=event.get('date', ''),
                                   market_home_prob=market_home_prob)
        model['factors'].sort(key=lambda f: abs(f[1]), reverse=True)
    except Exception:
        model = None

    # Blend in the player-level signal (nba_player_model.py) via the fitted
    # weights in nba_ensemble_model.py — see nba_ensemble_bootstrap.py for
    # the backtest behind this (2,453 games: 66.7% blended accuracy / 0.2106
    # Brier vs. 65.0%/0.2204 team-only and 64.2%/0.2338 player-only). Wrapped
    # in its own try/except, independent of the team model above — a roster/
    # gamelog fetch failure (new team-level data this app hasn't needed
    # before) should degrade to the team-only model, never break the page.
    # `lineups` is built alongside the blend (same active-roster fetch,
    # reused rather than hit twice) and attached to the game dict below
    # regardless of whether the blend itself succeeds — a projected lineup
    # is useful even on games the ensemble can't price.
    lineups = {'home': [], 'away': []}
    event_id = event.get('id')
    try:
        unavailable_ids = nba_boxscore_api.get_unavailable_player_ids(event_id)
    except Exception:
        unavailable_ids = set()

    if model is not None:
        try:
            team_prob = model['home_prob']
            team_margin = spread_proxy.implied_margin('NBA', team_prob)

            def _active_projections(team_id):
                """Returns (projections for nba_player_model.predict(),
                lineup rows with id/name for display) from one fetch."""
                active = nba_roster_api.get_active_roster(team_id, unavailable_ids=unavailable_ids)
                projections, lineup = [], []
                for p in active:
                    proj = nba_player_model.project_player(p['games'])
                    if proj:
                        projections.append(proj)
                        lineup.append({'id': p['id'], 'name': p['name'], **proj})
                # Highest-projected-minutes first — who the model expects
                # to play the most, not roster order.
                lineup.sort(key=lambda row: row['mpg'], reverse=True)
                return projections, lineup

            home_active, lineups['home'] = _active_projections(home.get('id'))
            away_active, lineups['away'] = _active_projections(away.get('id'))

            home_ppg_allowed = home.get('blend_ppg_allowed', home.get('ppg_allowed'))
            away_ppg_allowed = away.get('blend_ppg_allowed', away.get('ppg_allowed'))
            home_opp_factor = nba_player_model.opponent_factor(away_ppg_allowed)
            away_opp_factor = nba_player_model.opponent_factor(home_ppg_allowed)

            player_pred = (nba_player_model.predict(
                               home_active, away_active,
                               home_opp_factor=home_opp_factor,
                               away_opp_factor=away_opp_factor)
                           if home_active and away_active else None)

            ensemble = (nba_ensemble_model.predict(
                            team_prob, player_pred['home_prob'],
                            team_margin, player_pred['margin'])
                        if player_pred and team_margin is not None else None)

            model['team_only_prob'] = team_prob
            model['player_prob']    = player_pred['home_prob'] if player_pred else None
            model['blended']        = ensemble is not None
            if ensemble:
                model['home_prob'] = ensemble['home_prob']
                model['away_prob'] = round(1.0 - ensemble['home_prob'], 4)
                model['ensemble_margin'] = ensemble['margin']
        except Exception:
            # Leave `model` as the team-only prediction — same fallback
            # philosophy as the try/except around nba_model.predict() above.
            model['blended'] = False

    # Once the game has actually started, match each projected player
    # against ESPN's live/final boxscore (nba_boxscore_api.get_live_boxscore)
    # by player id and attach their real line + starter flag — the
    # "prediction vs. actual" comparison. Pre-game this is a no-op (the
    # boxscore endpoint returns {} before tip-off) and every lineup row
    # just carries its projection.
    if status in ('Live', 'Final'):
        try:
            boxscore = nba_boxscore_api.get_live_boxscore(event_id)
            for side, team in (('home', home), ('away', away)):
                actual_by_id = {row['id']: row for row in boxscore.get(team.get('id'), [])}
                for row in lineups[side]:
                    actual = actual_by_id.get(row['id'])
                    if actual:
                        row['actual'] = actual

            # Once Final, persist the box score into the local stats
            # warehouse (nba_stats_db.py) — reuses the boxscore fetch above,
            # so this costs no extra API calls. Idempotent (upsert), so it's
            # fine that this runs on every _build_game() call for a Final
            # game, not just the first time it goes Final (the 20-min
            # _check_finalized_games cron and every real page visit both
            # call this). Future reads (nba_roster_api.get_player_gamelog)
            # use this table instead of re-fetching each player's gamelog
            # from ESPN.
            if status == 'Final' and boxscore:
                home_score = float(home['score']) if home.get('score') is not None else None
                away_score = float(away['score']) if away.get('score') is not None else None
                game_type = nba_stats_db.GAME_TYPE_BY_ESPN_SEASON_TYPE.get(
                    event.get('season', {}).get('type'), 'regular')
                nba_stats_db.ingest_game(
                    event_id, game_date_et, _get_nba_season(), game_type,
                    home.get('id'), away.get('id'), home_score, away_score, boxscore,
                    home_name=home.get('name'), away_name=away.get('name'))
        except Exception:
            pass

    # Pace-adjusted total (O/U) projection — separate from the win-prob
    # model above, needs its own FGA/OREB/TOV/FTA fetch per team (see
    # bball_total_model.py for why this can't reuse the win-prob factors).
    # Uses blend_ppg/blend_ppg_allowed the same way nba_model.predict() does
    # (falls back to raw ppg once a team has played EARLY_SEASON_GAMES) —
    # without it, early-season totals were built off a handful of raw
    # games per team, which swings PPG/PPG-allowed and the resulting total
    # projection more than the market's own line does. Same fix as
    # nfl_api.py/cfb_api.py's total_inputs blocks.
    # total_inputs is stashed alongside the projection so a future
    # total-model recalibration can replay this exact game.
    home_ppg         = home.get('blend_ppg', home.get('ppg'))
    home_ppg_allowed = home.get('blend_ppg_allowed', home.get('ppg_allowed'))
    away_ppg         = away.get('blend_ppg', away.get('ppg'))
    away_ppg_allowed = away.get('blend_ppg_allowed', away.get('ppg_allowed'))
    total_inputs = {
        'home_id': home.get('id'), 'home_ppg': home_ppg, 'home_ppg_allowed': home_ppg_allowed,
        'away_id': away.get('id'), 'away_ppg': away_ppg, 'away_ppg_allowed': away_ppg_allowed,
    }
    try:
        total_model = bball_total_model.predict_total(
            'NBA', home.get('id'), home_ppg, home_ppg_allowed,
            away.get('id'), away_ppg, away_ppg_allowed)
    except Exception:
        total_model = None
    try:
        total_model_breakdown = bball_total_model.factor_breakdown('NBA', total_model)
    except Exception:
        total_model_breakdown = None

    espn_odds  = (comp.get('odds') or [{}])[0]
    odds_line  = espn_odds.get('details', '')
    over_under = espn_odds.get('overUnder')

    a_ab = away.get('abbrev', '')
    h_ab = home.get('abbrev', '')

    return {
        'game_id':       event.get('id'),
        'game_time_utc': event.get('date', ''),
        'status':        status,
        'venue':         venue,
        'away':          away,
        'home':          home,
        'odds_line':     odds_line,
        'over_under':    over_under,
        'sport':         'NBA',
        'odds':          game_odds,
        'model':         model,
        'lineups':       lineups,
        'total_model':   total_model,
        'total_model_breakdown': total_model_breakdown,
        'total_inputs':  total_inputs,
        'bet_name':      f"{a_ab} @ {h_ab}",
        'game_key':      f"{_normalize(home['name'])}_{_normalize(away['name'])}",
        'linescore':     _parse_linescores(comp) if status != 'Preview' else None,
        # ESPN season.type: 1=preseason, 2=regular, 3=postseason/play-in.
        # _upsert_predictions uses this to skip writing preseason games into
        # game_predictions — min-effort preseason results would otherwise
        # corrupt the Model Performance page's accuracy/calibration tracking.
        'is_preseason':  event.get('season', {}).get('type') == 1,
    }


def get_today_game_count():
    """Cheap today's-game count for the sport-chip badge — shares the same
    cached scoreboard fetch/cache key as build_schedule_context() without
    paying for any of its team-stats/odds/model enrichment."""
    today_str = _today_et()
    data = _cached_get(ESPN_NBA, {}, f'nba_{today_str}', _TTL['scoreboard'])
    if not data:
        return 0
    events = data.get('events', [])
    return len([e for e in events if _event_date_et(e.get('date', '')) == today_str])


def build_schedule_context(target_date=None):
    """Returns target_date's (default today, YYYY-MM-DD ET) NBA games from
    ESPN with model predictions. Empty list during offseason."""
    today_str = target_date or _today_et()
    params = {'dates': today_str.replace('-', '')} if target_date else {}
    data = _cached_get(ESPN_NBA, params, f'nba_{today_str}', _TTL['scoreboard'])
    if not data:
        return []

    events = data.get('events', [])
    today_events = [
        e for e in events
        if _event_date_et(e.get('date', '')) == today_str
    ]
    if not today_events:
        return []

    season       = _get_nba_season()
    game_log     = _get_season_game_log(season)
    team_stats   = _compute_team_season_stats(game_log)
    prior_stats  = _get_prior_season_team_stats(season - 1)
    nba_odds_map = odds_api.get_odds_map('nba')

    games = [_build_game(event, team_stats, game_log, nba_odds_map, prior_stats) for event in today_events]

    try:
        date_display = datetime.strptime(today_str, '%Y-%m-%d').strftime('%a, %b %-d')
    except Exception:
        date_display = today_str

    return [{'date': today_str, 'date_display': date_display, 'games': games}]
