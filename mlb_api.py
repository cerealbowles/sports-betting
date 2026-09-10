import time
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# All three major North American leagues schedule by US Eastern time.
# Using UTC here caused the page to flip to "tomorrow" around 8pm in Colorado.
_ET = ZoneInfo('America/New_York')

def _today_et():
    return datetime.now(_ET).strftime('%Y-%m-%d')
import odds_api
import weather_api
import mlb_model
import statcast_api
import fangraphs_api

MLB_API = "https://statsapi.mlb.com/api/v1"

# Hours behind Eastern Time for each team's home city (during summer / DST)
_TEAM_TZ_OFFSET = {
    # ET (0 h): BAL, BOS, TB, TOR, NYY, CLE, DET, WSH, NYM, PHI, ATL, MIA, CIN, PIT
    110: 0, 111: 0, 139: 0, 141: 0, 147: 0,
    114: 0, 116: 0,
    120: 0, 121: 0, 143: 0, 144: 0, 146: 0,
    113: 0, 134: 0,
    # CT (1 h): KC, MIN, CWS, HOU, TEX, CHC, STL, MIL
    118: 1, 142: 1, 145: 1, 117: 1, 140: 1, 112: 1, 138: 1, 158: 1,
    # MT (2 h): ARI, COL
    109: 2, 115: 2,
    # PT (3 h): LAA, OAK/ATH, SEA, LAD, SD, SF
    108: 3, 133: 3, 136: 3, 119: 3, 135: 3, 137: 3,
}

def _logo(team_id):
    """Official MLB static CDN — keyed by MLBAM team ID, always correct."""
    return f'https://www.mlbstatic.com/team-logos/{team_id}.svg'

_cache = {}
_TTL = {
    'schedule': 120,    # 2 min — keeps live scores reasonably fresh
    'recent':   3600,   # 1 hour
    'standings': 3600,  # 1 hour
    'stats':    21600,  # 6 hours
    'gamelog':  1800,   # 30 min
    'roster':   3600,   # 1 hour
}

LG_ERA = 4.20  # MLB league-average ERA used for opponent-quality normalization


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
    except Exception as e:
        print(f'[mlb_api] _cached_get failed for {key} ({url}): {e}', flush=True)
        return _cache[key][0] if key in _cache else None
    _cache[key] = (data, now)
    return data


def get_live_scores():
    """
    Returns {game_key: score_dict} for today's MLB games.
    Reuses the 2-min cached schedule so no extra API call is needed.
    score_dict keys: status, away_abbr, home_abbr, away_score, home_score, period
    """
    from odds_api import _normalize
    data = _get_schedule_raw()
    scores = {}
    if not data:
        return scores
    for date_obj in data.get('dates', []):
        for game in date_obj.get('games', []):
            h     = game['teams']['home']
            a     = game['teams']['away']
            h_name = h['team']['name']
            a_name = a['team']['name']
            gk    = f"{_normalize(h_name)}_{_normalize(a_name)}"
            _st_obj  = game.get('status', {})
            state    = _st_obj.get('abstractGameState', 'Preview')
            _detail  = _st_obj.get('detailedState', '')
            _PRE     = {'Warmup', 'Pre-Game', 'Delayed Start', 'Scheduled'}
            if state == 'Live' and _detail in _PRE:
                state = 'Preview'
            ls    = game.get('linescore', {})
            period = None
            bases  = None
            outs_n = None
            if state == 'Live':
                _half_raw = ls.get('inningHalf', '')
                half   = {'Top': 'Top', 'Bottom': 'Bot', 'Middle': 'Mid', 'End': 'End'}.get(_half_raw, _half_raw[:3])
                inning = ls.get('currentInningOrdinal', '').rstrip('stndrh')
                outs_n = ls.get('outs')
                if outs_n == 3:
                    period = f"End {inning}".strip()
                else:
                    period = f"{half} {inning}".strip()
                    if outs_n is not None:
                        period += f" · {outs_n}out{'s' if outs_n != 1 else ''}"
                offense = ls.get('offense') or {}
                bases = {
                    'first':  bool(offense.get('first')),
                    'second': bool(offense.get('second')),
                    'third':  bool(offense.get('third')),
                    'outs':   outs_n if outs_n is not None else 0,
                }
            elif state == 'Final':
                period = 'Final'
            scores[gk] = {
                'status':     state,
                'away_abbr':  a['team'].get('abbreviation', ''),
                'home_abbr':  h['team'].get('abbreviation', ''),
                'away_score': a.get('score'),
                'home_score': h.get('score'),
                'period':     period,
                'bases':      bases,
                'game_pk':    game.get('gamePk'),
            }
    return scores


def get_game_boxscore(game_pk):
    """
    Trimmed current-game batting/pitching lines for one game, used by the schedule
    page's per-card detail panel and mirrored to LifeOS (same shape, same
    LIFEOS_API_TOKEN gate as get_live_scores()). Pulled from the boxscore endpoint
    (not the full play-by-play live feed) since only current-game totals are needed.
    """
    if not game_pk:
        return None
    data = _cached_get(f"{MLB_API}/game/{game_pk}/boxscore", {}, f'boxscore_{game_pk}', 20)
    if not data:
        return None

    def _side(side_key):
        side = data.get('teams', {}).get(side_key, {})
        players = side.get('players', {})

        batters = []
        for pid in side.get('batters', []):
            p = players.get(f'ID{pid}')
            bat = (p or {}).get('stats', {}).get('batting', {})
            if not bat:
                continue
            batters.append({
                'name': p.get('person', {}).get('fullName', ''),
                'pos':  p.get('position', {}).get('abbreviation', ''),
                'ab':   bat.get('atBats', 0),
                'r':    bat.get('runs', 0),
                'h':    bat.get('hits', 0),
                'rbi':  bat.get('rbi', 0),
                'bb':   bat.get('baseOnBalls', 0),
                'so':   bat.get('strikeOuts', 0),
            })

        pitchers = []
        for pid in side.get('pitchers', []):
            p = players.get(f'ID{pid}')
            pit = (p or {}).get('stats', {}).get('pitching', {})
            if not pit:
                continue
            pitchers.append({
                'name':    p.get('person', {}).get('fullName', ''),
                'ip':      pit.get('inningsPitched', '0.0'),
                'h':       pit.get('hits', 0),
                'r':       pit.get('runs', 0),
                'er':      pit.get('earnedRuns', 0),
                'bb':      pit.get('baseOnBalls', 0),
                'so':      pit.get('strikeOuts', 0),
                'pitches': pit.get('numberOfPitches', 0),
            })

        return {
            'abbr':     side.get('team', {}).get('abbreviation', ''),
            'batters':  batters,
            'pitchers': pitchers,
        }

    return {'away': _side('away'), 'home': _side('home')}


def _parse_ip(ip_str):
    """MLB API stores IP as '3.2' meaning 3⅔ innings — convert to decimal."""
    try:
        parts = (ip_str or '0').split('.')
        return float(parts[0]) + (float(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)
    except Exception:
        return 0.0


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _get_schedule_raw(days=1):
    today = _today_et()
    return _cached_get(
        f"{MLB_API}/schedule",
        {'sportId': 1, 'startDate': today, 'endDate': today,
         'hydrate': 'probablePitcher,linescore,team,venue', 'gameType': 'R,F,D,L,W'},
        f'sched_{today}',
        _TTL['schedule'],
    )


def _get_recent_data(days_back=20):
    """Returns (form_map, matchup_map, sched, recent_runs_map) from a single API call.
    form_map:        {team_id: ['W','L', ...]} — last 10 games per team
    matchup_map:     {frozenset({id_a, id_b}): [{'date': str, 'winner': id}, ...]}
    sched:           {team_id: [{'date', 'is_home'}, ...]} sorted ascending
    recent_runs_map: {team_id: [runs, ...]} — last 15 games' runs scored
    """
    end   = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    start = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    data = _cached_get(
        f"{MLB_API}/schedule",
        {'sportId': 1, 'startDate': start, 'endDate': end,
         'hydrate': 'linescore', 'gameType': 'R'},
        f'recent_{start}',
        _TTL['recent'],
    )
    form     = {}
    matchups = {}
    sched    = {}  # {team_id: [{'date', 'is_home'}, ...]} sorted ascending
    runs     = {}  # {team_id: [runs_scored, ...]} chronological
    if not data:
        return form, matchups, sched, runs
    for date_obj in data.get('dates', []):
        for game in date_obj.get('games', []):
            if game.get('status', {}).get('abstractGameState') != 'Final':
                continue
            away = game['teams']['away']
            home = game['teams']['home']
            a_id, h_id = away['team']['id'], home['team']['id']
            a_s,  h_s  = away.get('score', 0) or 0, home.get('score', 0) or 0
            game_date  = date_obj['date']
            form.setdefault(a_id, []).append('W' if a_s > h_s else 'L')
            form.setdefault(h_id, []).append('W' if h_s > a_s else 'L')
            # Store (runs_scored, opponent_id) so we can normalize by opponent ERA later
            runs.setdefault(a_id, []).append((a_s, h_id))
            runs.setdefault(h_id, []).append((h_s, a_id))
            winner = a_id if a_s > h_s else h_id
            key = frozenset([a_id, h_id])
            matchups.setdefault(key, []).append({'date': game_date, 'winner': winner})
            sched.setdefault(a_id, []).append({'date': game_date, 'is_home': False})
            sched.setdefault(h_id, []).append({'date': game_date, 'is_home': True})
    for key in matchups:
        matchups[key].sort(key=lambda x: x['date'])
    for tid in sched:
        sched[tid].sort(key=lambda x: x['date'])
    recent_runs = {tid: v[-15:] for tid, v in runs.items()}
    return {tid: v[-10:] for tid, v in form.items()}, matchups, sched, recent_runs


def _get_team_era_map():
    """Returns {team_id: era_float} for current season. Cached 6h."""
    season = datetime.utcnow().year
    data = _cached_get(
        f"{MLB_API}/teams/stats",
        {'stats': 'season', 'season': season, 'group': 'pitching', 'sportId': 1},
        f'team_era_{season}',
        _TTL['stats'],
    )
    if not data:
        return {}
    result = {}
    for split in (data.get('stats') or [{}])[0].get('splits', []):
        tid = (split.get('team') or {}).get('id')
        era = split.get('stat', {}).get('era')
        if tid and era:
            try:
                result[tid] = float(era)
            except (TypeError, ValueError):
                pass
    return result


def _get_standings_splits():
    """Returns {team_id: {'home': (W, L), 'away': (W, L)}} for all MLB teams."""
    season = datetime.utcnow().year
    data = _cached_get(
        f"{MLB_API}/standings",
        {'leagueId': '103,104', 'season': season, 'standingsTypes': 'regularSeason'},
        f'standings_{season}',
        _TTL['standings'],
    )
    result = {}
    if not data:
        return result
    for division in data.get('records', []):
        for tr in division.get('teamRecords', []):
            tid = tr['team']['id']
            home_w = home_l = away_w = away_l = 0
            for s in tr.get('records', {}).get('splitRecords', []):
                stype = s.get('type', '')
                if stype == 'home':
                    home_w, home_l = s.get('wins', 0), s.get('losses', 0)
                elif stype == 'away':
                    away_w, away_l = s.get('wins', 0), s.get('losses', 0)
            result[tid] = {'home': (home_w, home_l), 'away': (away_w, away_l)}
    return result


def _get_team_bullpen_info(team_id):
    """Returns (active_rps, il_rps) each as [{id, name}] using 40-man roster.
    active_rps: RPs available to pitch today.
    il_rps: RPs currently on IL or other non-active status.
    """
    season = datetime.utcnow().year
    data = _cached_get(
        f"{MLB_API}/teams/{team_id}/roster",
        {'rosterType': '40Man', 'season': season},
        f'roster40_{team_id}_{season}',
        _TTL['roster'],
    )
    if not data:
        return [], []
    active, il = [], []
    for p in data.get('roster', []):
        # API returns code='1' and abbreviation='P' for all pitchers on 40-man;
        # filter by type instead since 'RP' code is never present.
        if p.get('position', {}).get('type') != 'Pitcher':
            continue
        entry = {'id': p['person']['id'], 'name': p['person']['fullName']}
        sc = p.get('status', {}).get('code', '')
        if sc == 'A':
            active.append(entry)
        elif sc:
            il.append(entry)
    return active, il


def _get_pitcher_season_stats(person_id):
    season = datetime.utcnow().year
    data = _cached_get(
        f"{MLB_API}/people/{person_id}/stats",
        {'stats': 'season', 'group': 'pitching', 'season': season},
        f'pitcher_{person_id}_{season}',
        _TTL['stats'],
    )
    if not data:
        return {}
    for split_group in data.get('stats', []):
        for s in split_group.get('splits', []):
            st = s.get('stat', {})
            return {
                'era':    st.get('era', '—'),
                'whip':   st.get('whip', '—'),
                'wins':   st.get('wins', 0),
                'losses': st.get('losses', 0),
                'ip':     st.get('inningsPitched', '0'),
                'gs':     st.get('gamesStarted', 0),
                'k':      st.get('strikeOuts', 0),
                'bb':     st.get('baseOnBalls', 0),
                'bf':     st.get('battersFaced', 0),
            }
    return {}


def _get_pitcher_game_log_raw(person_id):
    season = datetime.utcnow().year
    return _cached_get(
        f"{MLB_API}/people/{person_id}/stats",
        {'stats': 'gameLog', 'group': 'pitching', 'season': season},
        f'gamelog_{person_id}_{season}',
        _TTL['gamelog'],
    )


def _get_pitcher_last_starts(person_id, n=3):
    """Returns the last n starting appearances as [{ip, er, k, result}, ...]."""
    data = _get_pitcher_game_log_raw(person_id)
    if not data:
        return []
    starts = []
    for split_group in data.get('stats', []):
        for s in split_group.get('splits', []):
            st = s.get('stat', {})
            ip_str = st.get('inningsPitched', '0') or '0'
            if _parse_ip(ip_str) < 2.0:
                continue
            result = 'W' if st.get('wins', 0) else ('L' if st.get('losses', 0) else 'ND')
            starts.append({
                'date':   s.get('date', ''),
                'ip':     ip_str,
                'er':     st.get('earnedRuns', 0),
                'k':      st.get('strikeOuts', 0),
                'result': result,
            })
    return starts[-n:]


def _get_team_batting(team_id):
    season = datetime.utcnow().year
    data = _cached_get(
        f"{MLB_API}/teams/{team_id}/stats",
        {'stats': 'season', 'group': 'hitting', 'season': season},
        f'batting_{team_id}_{season}',
        _TTL['stats'],
    )
    if not data:
        return {}
    for split_group in data.get('stats', []):
        for s in split_group.get('splits', []):
            st = s.get('stat', {})
            gp   = st.get('gamesPlayed', 0) or 1
            runs = st.get('runs', 0) or 0
            pa   = st.get('plateAppearances', 0) or 1
            return {
                'runs_pg': round(runs / gp, 1),
                'ops':     st.get('ops', '—'),
                'avg':     st.get('avg', '—'),
                'k_pct':   round(st.get('strikeOuts', 0) / pa, 4),
                'bb_pct':  round(st.get('baseOnBalls', 0) / pa, 4),
            }
    return {}


def _compute_bullpen_stats(active_rps, il_rps, game_logs_by_id):
    """
    active_rps: [{id, name}] currently available relief pitchers
    il_rps:     [{id, name}] RPs on IL or other unavailable status
    game_logs_by_id: {person_id: raw_api_response}
    Returns dict with ip_last_3, era_last_14, closer_yesterday, closer_name, top_unavail.
    """
    today_et  = datetime.now(_ET).date()
    yesterday = (today_et - timedelta(days=1)).strftime('%Y-%m-%d')
    d3_ago    = (today_et - timedelta(days=3)).strftime('%Y-%m-%d')
    d14_ago   = (today_et - timedelta(days=14)).strftime('%Y-%m-%d')

    rv_stats = {}
    for rv in active_rps:
        pid = rv['id']
        raw = game_logs_by_id.get(pid)
        s_ip = s_sv = ip_3d = er_14d = ip_14d = 0
        pitched_yday = False
        if raw:
            for sg in raw.get('stats', []):
                for s in sg.get('splits', []):
                    st = s.get('stat', {})
                    gd = s.get('date', '')
                    ip = _parse_ip(st.get('inningsPitched', '0') or '0')
                    er = st.get('earnedRuns', 0) or 0
                    sv = st.get('saves', 0) or 0
                    s_ip += ip
                    s_sv += sv
                    if gd >= d3_ago:
                        ip_3d += ip
                    if gd >= d14_ago:
                        ip_14d += ip
                        er_14d += er
                    if gd == yesterday:
                        pitched_yday = True
        rv_stats[pid] = {
            'name': rv['name'], 'season_ip': s_ip, 'season_saves': s_sv,
            'ip_3d': ip_3d, 'er_14d': er_14d, 'ip_14d': ip_14d,
            'pitched_yday': pitched_yday,
        }

    total_ip_3d  = round(sum(v['ip_3d']  for v in rv_stats.values()), 1)
    total_ip_14d = sum(v['ip_14d'] for v in rv_stats.values())
    total_er_14d = sum(v['er_14d'] for v in rv_stats.values())
    era_14d = round((total_er_14d / total_ip_14d) * 9, 2) if total_ip_14d > 0 else None

    # Closer: highest saves, break ties by season IP
    ranked = sorted(rv_stats.values(), key=lambda v: (v['season_saves'], v['season_ip']), reverse=True)
    closer = ranked[0] if ranked else None

    return {
        'ip_last_3':        total_ip_3d,
        'era_last_14':      era_14d,
        'closer_yesterday': closer['pitched_yday'] if closer else False,
        'closer_name':      closer['name'].split()[-1] if closer else None,
        'top_unavail':      [r['name'].split()[-1] for r in il_rps[:3]],
    }


def _schedule_fatigue(team_id, is_home, venue_team_id, game_log):
    """
    Returns schedule-fatigue metrics for a team entering today's game.

    game_log: [{'date': 'YYYY-MM-DD', 'is_home': bool}] sorted ascending (from _get_recent_data).
    venue_team_id: the home team's ID (used to look up the venue's timezone).

    Returns dict:
      rest_days      – days since last game (0 = back-to-back, None = unknown)
      road_trip_len  – consecutive road games ending today (0 if playing at home)
      homestand_len  – consecutive home games ending today (0 if away)
      tz_shift       – abs(team_home_tz – venue_tz) in hours; 0 for home team
    """
    today = datetime.now(_ET).date()

    rest_days = None
    if game_log:
        last_date = datetime.strptime(game_log[-1]['date'], '%Y-%m-%d').date()
        delta = (today - last_date).days - 1  # 0 = played yesterday
        rest_days = max(0, delta)

    road_trip_len = homestand_len = 0
    if is_home:
        for g in reversed(game_log):
            if g['is_home']:
                homestand_len += 1
            else:
                break
        homestand_len += 1  # include today
    else:
        for g in reversed(game_log):
            if not g['is_home']:
                road_trip_len += 1
            else:
                break
        road_trip_len += 1  # include today

    team_tz   = _TEAM_TZ_OFFSET.get(team_id, 0)
    venue_tz  = _TEAM_TZ_OFFSET.get(venue_team_id, 0)
    tz_shift  = abs(team_tz - venue_tz) if not is_home else 0

    return {
        'rest_days':     rest_days,
        'road_trip_len': road_trip_len,
        'homestand_len': homestand_len,
        'tz_shift':      tz_shift,
    }


# ── Context builder ───────────────────────────────────────────────────────────

def _build_team_info(side_data, side, recent, pitcher_stats, pitcher_logs, team_batting, splits, sc_pitchers=None, sc_teams=None, team_bullpen=None, team_sched=None, venue_team_id=None, fg_pitchers=None, sc_metrics=None, recent_runs_map=None, team_era_map=None):
    team = side_data.get('team', {})
    rec  = side_data.get('leagueRecord', {})
    tid  = team.get('id', 0)

    pitcher = side_data.get('probablePitcher')
    p_info  = None
    if pitcher:
        pid   = pitcher['id']
        stats = pitcher_stats.get(pid, {})
        logs  = pitcher_logs.get(pid, [])
        sc    = (sc_pitchers or {}).get(pid, {})
        throws = pitcher.get('pitchHand', {}).get('code', '')

        # FanGraphs — keyed by MLBAM ID (xMLBAMID field); covers xFIP, SIERA,
        # K%, BB%, SwStr% which are more reliable from FanGraphs than Savant.
        fg = (fg_pitchers or {}).get(pid, {})
        # Savant statcast leaderboard — barrel rate and hard-hit % per MLBAM ID
        sm = (sc_metrics or {}).get(pid, {})

        gs     = stats.get('gs', 0)
        avg_ip = round(_parse_ip(stats.get('ip', '0')) / gs, 2) if gs > 0 else None

        p_info = {
            'name':         pitcher.get('fullName', ''),
            'throws':       throws,
            'era':          stats.get('era', '—'),
            'x_era':        sc.get('x_era'),
            'xfip':         fg.get('xfip'),
            'siera':        fg.get('siera'),
            'whip':         stats.get('whip', '—'),
            'record':       f"{stats.get('wins', 0)}-{stats.get('losses', 0)}",
            'k':            stats.get('k', 0),
            'bb':           stats.get('bb', 0),
            'bf':           stats.get('bf', 0),
            'gs':           gs,
            'avg_ip':       avg_ip,
            'last_starts':  logs,
            'k_pct':        fg.get('k_pct'),
            'bb_pct':       fg.get('bb_pct'),
            'whiff_pct':    fg.get('swstr_pct'),
            'barrel_pct':   sm.get('barrel_pct'),
            'hard_hit_pct': sm.get('hard_hit_pct'),
        }

    team_splits = splits.get(tid, {})
    split_rec   = team_splits.get(side, (0, 0))
    batting     = team_batting.get(tid, {})
    abbr        = team.get('abbreviation', '')
    abbr_lower  = abbr.lower()
    sc_team     = (sc_teams or {}).get(abbr_lower, {})
    bullpen     = (team_bullpen or {}).get(tid)

    is_home  = (side == 'home')
    game_log = (team_sched or {}).get(tid, [])
    fatigue  = _schedule_fatigue(tid, is_home, venue_team_id or tid, game_log)

    raw_runs = (recent_runs_map or {}).get(tid, [])
    if team_era_map and raw_runs and isinstance(raw_runs[0], (list, tuple)):
        adj = []
        for entry in raw_runs:
            r_val, opp_id = entry[0], entry[1]
            opp_era = (team_era_map or {}).get(opp_id, LG_ERA)
            mult = min(max(LG_ERA / max(opp_era, 2.0), 0.60), 1.60)
            adj.append(round(r_val * mult, 2))
        recent_runs = adj
    else:
        recent_runs = [e[0] if isinstance(e, (list, tuple)) else e for e in raw_runs]
    recent_rpg  = round(sum(recent_runs) / len(recent_runs), 2) if len(recent_runs) >= 5 else None

    return {
        'id':                tid,
        'name':              team.get('name', ''),
        'abbr':              abbr,
        'logo_url':          _logo(tid),
        'wins':              rec.get('wins', 0),
        'losses':            rec.get('losses', 0),
        'split_w':           split_rec[0],
        'split_l':           split_rec[1],
        'side':              side,
        'form':              recent.get(tid, []),
        'pitcher':           p_info,
        'score':             side_data.get('score'),
        'runs_pg':           batting.get('runs_pg', '—'),
        'ops':               batting.get('ops', '—'),
        'k_pct':             batting.get('k_pct', None),
        'bb_pct':            batting.get('bb_pct', None),
        'x_woba':            sc_team.get('x_woba'),
        'bullpen':           bullpen,
        'schedule_fatigue':  fatigue,
        'recent_rpg':        recent_rpg,
        'recent_rpg_n':      len(recent_runs),
    }


def get_today_game_count():
    """Cheap today's-game count for the sport-chip badge — reuses the same
    cached raw schedule fetch as build_schedule_context() without paying for
    any of its pitcher/bullpen/odds/model enrichment."""
    raw = _get_schedule_raw()
    if not raw:
        return 0
    dates = raw.get('dates', [])
    return len(dates[0].get('games', [])) if dates else 0


def build_schedule_context():
    """Returns today's games with full context ready for the template."""
    from odds_api import _normalize
    raw        = _get_schedule_raw()
    recent, matchups, team_sched, recent_runs_map = _get_recent_data()
    team_era_map = _get_team_era_map()
    splits     = _get_standings_splits()
    odds_map   = odds_api.get_odds_map('mlb')
    sc_pitchers       = statcast_api.get_pitcher_statcast()
    sc_teams          = statcast_api.get_team_statcast()
    sc_batting_splits = statcast_api.get_team_batting_splits()
    # Predictive pitcher metrics — fail silently if either source is unavailable
    try:
        fg_pitchers = fangraphs_api.get_pitcher_xfip()
    except Exception:
        fg_pitchers = {}
    try:
        sc_metrics = statcast_api.get_pitcher_metrics()
    except Exception:
        sc_metrics = {}
    if not raw:
        return []

    # Collect IDs for all probable pitchers and teams in today's games
    pitcher_ids          = set()
    team_ids             = set()
    starter_by_team      = {}   # tid -> probable starter person_id (exclude from bullpen)
    for date_obj in raw.get('dates', []):
        for game in date_obj.get('games', []):
            for side in ('away', 'home'):
                p = game['teams'][side].get('probablePitcher')
                if p:
                    pitcher_ids.add(p['id'])
                tid = game['teams'][side].get('team', {}).get('id')
                if tid:
                    team_ids.add(tid)
                    if p:
                        starter_by_team[tid] = p['id']

    # Phase 1: fetch 40-man rosters to identify active RPs and IL-placed RPs
    team_bullpen_info = {}  # tid -> (active_rps, il_rps)
    with ThreadPoolExecutor(max_workers=16) as ex:
        rf = {ex.submit(_get_team_bullpen_info, tid): tid for tid in team_ids}
        for f in as_completed(rf):
            try:
                team_bullpen_info[rf[f]] = f.result()
            except Exception:
                team_bullpen_info[rf[f]] = ([], [])

    all_rp_ids = {
        rp['id']
        for active_rps, _ in team_bullpen_info.values()
        for rp in active_rps
    }

    # Phase 2: fetch all stats in parallel (SP stats + team batting + RP game logs)
    pitcher_stats = {}
    pitcher_logs  = {}
    team_batting  = {}
    reliever_logs = {}

    def _fetch(task):
        kind, eid = task
        if kind == 'ps':
            return kind, eid, _get_pitcher_season_stats(eid)
        if kind == 'pl':
            return kind, eid, _get_pitcher_last_starts(eid)
        if kind == 'rgl':
            return kind, eid, _get_pitcher_game_log_raw(eid)
        return kind, eid, _get_team_batting(eid)

    tasks = (
        [('ps', pid) for pid in pitcher_ids]         +
        [('pl', pid) for pid in pitcher_ids]         +
        [('tb', tid) for tid in team_ids]            +
        [('rgl', rp_id) for rp_id in all_rp_ids]
    )
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                kind, eid, result = future.result()
                if kind == 'ps':
                    pitcher_stats[eid] = result
                elif kind == 'pl':
                    pitcher_logs[eid]  = result
                elif kind == 'rgl':
                    reliever_logs[eid] = result
                else:
                    team_batting[eid]  = result
            except Exception:
                pass

    # Compute bullpen stats for each team, excluding today's probable starter
    team_bullpen = {}
    for tid in team_ids:
        active_rps, il_rps = team_bullpen_info.get(tid, ([], []))
        starter_id = starter_by_team.get(tid)
        if starter_id:
            active_rps = [rp for rp in active_rps if rp['id'] != starter_id]
        team_bullpen[tid] = _compute_bullpen_stats(active_rps, il_rps, reliever_logs)

    result = []
    for date_obj in raw.get('dates', []):
        date_str = date_obj['date']
        try:
            dt           = datetime.strptime(date_str, '%Y-%m-%d')
            date_display = dt.strftime('%a, %b %-d')
        except Exception:
            date_display = date_str

        games = []
        for game in date_obj.get('games', []):
            _status_obj    = game.get('status', {})
            status         = _status_obj.get('abstractGameState', '')
            _detailed      = _status_obj.get('detailedState', '')
            # MLB API marks warmup / pre-game as "Live" — treat these as Preview
            # so they remain bettable until first pitch is actually thrown.
            _PRE_GAME_STATES = {'Warmup', 'Pre-Game', 'Delayed Start', 'Scheduled'}
            if status == 'Live' and _detailed in _PRE_GAME_STATES:
                status = 'Preview'
            game_time_utc = game.get('gameDate', '')
            venue         = game.get('venue', {}).get('name', '')

            h_tid = game['teams']['home'].get('team', {}).get('id')
            away = _build_team_info(game['teams']['away'], 'away', recent, pitcher_stats, pitcher_logs, team_batting, splits, sc_pitchers, sc_teams, team_bullpen, team_sched, h_tid, fg_pitchers, sc_metrics, recent_runs_map, team_era_map)
            home = _build_team_info(game['teams']['home'], 'home', recent, pitcher_stats, pitcher_logs, team_batting, splits, sc_pitchers, sc_teams, team_bullpen, team_sched, h_tid, fg_pitchers, sc_metrics, recent_runs_map, team_era_map)

            # Inject handedness-split xwOBA: each team bats vs the opposing pitcher
            if sc_batting_splits:
                for batting_team, opp_team in ((away, home), (home, away)):
                    opp_throws = (opp_team.get('pitcher') or {}).get('throws', '')
                    batting_team['opp_throws'] = opp_throws
                    abbr = batting_team['abbr'].lower()
                    team_splits_sc = sc_batting_splits.get(abbr, {})
                    if opp_throws == 'L':
                        batting_team['x_woba_split'] = team_splits_sc.get('xwoba_vs_lhp')
                    elif opp_throws == 'R':
                        batting_team['x_woba_split'] = team_splits_sc.get('xwoba_vs_rhp')

            # Live game inning context
            inning_info = None
            if status == 'Live':
                ls = game.get('linescore', {})
                _half_raw = ls.get('inningHalf', '')
                half   = {'Top': 'Top', 'Bottom': 'Bot', 'Middle': 'Mid', 'End': 'End'}.get(_half_raw, _half_raw[:3])
                inning = ls.get('currentInningOrdinal', '').rstrip('stndrh')
                outs   = ls.get('outs')
                if inning:
                    inning_info = f"{half} {inning}"
                    if outs is not None and outs != 3:
                        inning_info += f" · {outs} out{'s' if outs != 1 else ''}"

            # Series split (works for regular season and playoffs alike)
            series_num   = game.get('seriesGameNumber', 1)
            series_total = game.get('gamesInSeries', 1)
            series_info  = None
            if series_total > 1:
                a_id = game['teams']['away']['team']['id']
                h_id = game['teams']['home']['team']['id']
                past = matchups.get(frozenset([a_id, h_id]), [])
                # Take only the most recent (series_num - 1) meetings — these are the prior games in this series
                prev = past[-(series_num - 1):] if series_num > 1 else []
                a_wins = sum(1 for g in prev if g['winner'] == a_id)
                h_wins = sum(1 for g in prev if g['winner'] == h_id)
                gm = f"Game {series_num} of {series_total}"
                if series_num == 1:
                    series_info = gm
                elif a_wins > h_wins:
                    series_info = f"{away['abbr']} leads {a_wins}–{h_wins} · {gm}"
                elif h_wins > a_wins:
                    series_info = f"{home['abbr']} leads {h_wins}–{a_wins} · {gm}"
                else:
                    series_info = f"Tied {a_wins}–{h_wins} · {gm}"

            game_date = game_time_utc[:13] if game_time_utc else None
            game_odds = odds_api.lookup_game_odds(odds_map, home['name'], away['name'], game_date=game_date)
            try:
                model = mlb_model.predict(home, away, matchups, game_time_utc=game_time_utc)
            except Exception:
                model = None

            # Linescore only for games that have actually started
            linescore = None
            raw_ls = game.get('linescore', {}) if status in ('Live', 'Final') else {}
            innings = raw_ls.get('innings', [])
            if innings:
                linescore = {
                    'innings': [
                        {
                            'num':       inn.get('num', ''),
                            'away_runs': inn.get('away', {}).get('runs', ''),
                            'home_runs': inn.get('home', {}).get('runs', ''),
                        }
                        for inn in innings
                    ],
                    'totals': {
                        'away': {
                            'r': raw_ls.get('teams', {}).get('away', {}).get('runs', 0),
                            'h': raw_ls.get('teams', {}).get('away', {}).get('hits', 0),
                            'e': raw_ls.get('teams', {}).get('away', {}).get('errors', 0),
                        },
                        'home': {
                            'r': raw_ls.get('teams', {}).get('home', {}).get('runs', 0),
                            'h': raw_ls.get('teams', {}).get('home', {}).get('hits', 0),
                            'e': raw_ls.get('teams', {}).get('home', {}).get('errors', 0),
                        },
                    },
                }

            games.append({
                'game_pk':       game.get('gamePk'),
                'game_time_utc': game_time_utc,
                'status':        status,
                'venue':         venue,
                'away':          away,
                'home':          home,
                'inning_info':   inning_info,
                'series_info':   series_info,
                'linescore':     linescore,
                'weather':       weather_api.get_game_weather(venue, 'mlb'),
                'odds':          game_odds,
                'model':         model,
                'bet_name':      f"{away['abbr']} @ {home['abbr']}",
                'game_key':      f"{_normalize(home['name'])}_{_normalize(away['name'])}",
            })

        if games:
            result.append({
                'date':         date_str,
                'date_display': date_display,
                'games':        games,
            })

    refreshed_at = datetime.now(_ET).strftime('%-I:%M %p ET')
    for day in result:
        day['refreshed_at'] = refreshed_at
    return result
