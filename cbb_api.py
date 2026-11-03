"""
cbb_api.py — Men's college basketball (NCAA Division I) schedule, live
scores and model predictions, from ESPN's free scoreboard. Same public
surface as nba_api.py / wnba_api.py (build_schedule_context,
get_live_scores, get_live_game_states, get_today_game_count,
get_season_phase, _get_season_game_log) so app.py's day-based sport
plumbing treats it like any other.

What's different from the NBA/WNBA modules:
  * Scale. ~365 D1 teams and 50-150 games a night. The scoreboard is
    requested with groups=50 (all of D1 — without it ESPN returns only a
    handful of featured games) and nothing per-game is fetched while
    building the slate (no summary/roster/boxscore calls; NBA/WNBA do
    several per game for their player model).
  * The model. Raw PPG / win% mostly measure schedule strength across 32
    conferences, so cbb_model.py runs on cbb_rating.py's opponent-adjusted
    efficiency ratings (AdjO / AdjD / tempo). The NBA-style inputs (record,
    home/road split, PPG, form, rest) are still computed here for the card
    and tracking rows — see cbb_model.py's docstring for the backtest.
  * Context NBA doesn't have: neutral sites (no home court in the model),
    conference membership, AP Top-25 rank.
"""
import time
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

import cbb_model
import cbb_rating
import cbb_stats_db
import odds_api
from dt_fmt import fmt as _fmt

_ET = ZoneInfo('America/New_York')

ESPN_CBB = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
D1_GROUP = cbb_rating.D1_GROUP
_SB_PARAMS = {'groups': D1_GROUP, 'limit': 500}

_cache = {}
_TTL = {'scoreboard': 120, 'season_log': 3600, 'prior_season_stats': 24 * 3600}

# Set True by app.py's _cache_only() — serve whatever is cached, never fetch.
_CACHE_ONLY = False

# Final games already written to TeamGameStat by this process — a busy
# night re-renders the same ~100 finals on every page load otherwise.
_ingested_ids = set()

# Same early-season prior blend as wnba_api.EARLY_SEASON_GAMES — only feeds
# the tracking rows here (the rating already carries last season).
EARLY_SEASON_GAMES = 4


def _today_et():
    return datetime.now(_ET).strftime('%Y-%m-%d')


def _event_date_et(date_str):
    if not date_str:
        return ''
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00')).astimezone(_ET).strftime('%Y-%m-%d')
    except ValueError:
        return ''


def _cached_get(url, params, key, ttl):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if _CACHE_ONLY or now - ts < ttl:
            return data
    if _CACHE_ONLY:
        return None
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = odds_api.stamp_fetched(r.json(), now)
    except Exception:
        return _cache[key][0] if key in _cache else None
    _cache[key] = (data, now)
    return data


def _scoreboard(date_str=None):
    """D1 scoreboard for an ET date (default today). One cache entry per
    date shared by every caller (schedule, live scores, counts, phase)."""
    target = date_str or _today_et()
    params = dict(_SB_PARAMS, dates=target.replace('-', ''))
    return _cached_get(ESPN_CBB, params, f'cbb_{target}', _TTL['scoreboard'])


def _get_cbb_season(date_str=None):
    """Season label = the calendar year the season ends in (2026-27 → 2027),
    matching ESPN's `season=` convention and cbb_rating."""
    return cbb_rating.season_for_date(date_str or _today_et())


def logo_url(team_id):
    return f'https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png' if team_id else ''


def _parse_record(records, *names):
    for rec in records or []:
        if rec.get('name', '').lower() in names or rec.get('type', '').lower() in names:
            return rec.get('summary')
    return None


def _record_to_wl(summary):
    try:
        w, l = summary.split('-')[:2]
        return int(w), int(l)
    except Exception:
        return 0, 0


def _status(comp):
    state = comp.get('status', {}).get('type', {}).get('state', 'pre')
    return {'in': 'Live', 'post': 'Final'}.get(state, 'Preview')


def _parse_linescores(comp):
    """Half-by-half scoring (2 regulation halves + OT columns). None pre-tip."""
    competitors = comp.get('competitors', [])
    home_ls = next((c.get('linescores') for c in competitors if c.get('homeAway') == 'home'), None) or []
    away_ls = next((c.get('linescores') for c in competitors if c.get('homeAway') == 'away'), None) or []
    if not home_ls and not away_ls:
        return None

    def _val(arr, i):
        if i >= len(arr):
            return None
        v = arr[i].get('value')
        return int(v) if v is not None else None

    periods = []
    for i in range(max(len(home_ls), len(away_ls), 2)):
        label = f'H{i + 1}' if i < 2 else ('OT' if i == 2 else f'OT{i - 1}')
        periods.append({'label': label, 'away': _val(away_ls, i), 'home': _val(home_ls, i)})
    return periods


# ── Season game log ────────────────────────────────────────────────────────

def _get_season_game_log(season, game_type='regular'):
    """[{game_date, home_name, away_name, home_score, away_score, home_won,
    neutral}] for every completed game in `season`. Reads cbb_rating's
    season cache (already kept current for the rating fit, so no second
    ESPN walk); falls back to the TeamGameStat warehouse if that's empty."""
    types = set(game_type) if isinstance(game_type, (list, tuple, set)) else {game_type}
    games = cbb_rating.load_season(season, allow_fetch=not _CACHE_ONLY)
    if games:
        out = []
        for g in games:
            gt = cbb_stats_db.GAME_TYPE_BY_ESPN_SEASON_TYPE.get(g.get('season_type'), 'regular')
            if gt in types:
                out.append({'game_date': g['date'], 'home_name': g['home'], 'away_name': g['away'],
                            'home_score': g['home_score'], 'away_score': g['away_score'],
                            'home_won': g['home_score'] > g['away_score'], 'neutral': g['neutral']})
        return out
    try:
        return cbb_stats_db.get_season_game_log_db(season, game_type=game_type)
    except Exception:
        return []


def _get_recent_game_log(season):
    """Regular season + postseason — recency reads (form, rest) shouldn't
    freeze when conference/NCAA tournaments start."""
    return _get_season_game_log(season, game_type=('regular', 'postseason'))


def _compute_team_season_stats(games):
    pts = defaultdict(lambda: {'for': 0, 'against': 0, 'g': 0})
    for g in games:
        for team, pf, pa in ((g['home_name'], g['home_score'], g['away_score']),
                             (g['away_name'], g['away_score'], g['home_score'])):
            pts[team]['for'] += pf
            pts[team]['against'] += pa
            pts[team]['g'] += 1
    return {t: {'ppg': round(v['for'] / v['g'], 1), 'ppg_allowed': round(v['against'] / v['g'], 1),
                'games': v['g']}
            for t, v in pts.items() if v['g']}


def _get_prior_season_team_stats(season):
    key = f'cbb_prior_stats_{season}'
    now = time.time()
    if key in _cache and now - _cache[key][1] < _TTL['prior_season_stats']:
        return _cache[key][0]
    wl = defaultdict(lambda: {'wins': 0, 'losses': 0})
    games = _get_season_game_log(season, game_type=('regular', 'postseason'))
    for g in games:
        wl[g['home_name']]['wins' if g['home_won'] else 'losses'] += 1
        wl[g['away_name']]['losses' if g['home_won'] else 'wins'] += 1
    _cache[key] = (dict(wl), now)
    return _cache[key][0]


def _apply_prior_season_blend(team, prior_stats):
    """Blend win% with last season's for teams with < EARLY_SEASON_GAMES
    played (tracking row only — see wnba_api._apply_prior_season_blend)."""
    cur = team.get('wins', 0) + team.get('losses', 0)
    if cur >= EARLY_SEASON_GAMES:
        return
    prior = prior_stats.get(team.get('name', ''))
    if not prior or not prior['wins'] + prior['losses']:
        return
    w = cur / EARLY_SEASON_GAMES
    cur_pct = team['wins'] / cur if cur else 0.5
    team['blend_win_pct'] = w * cur_pct + (1 - w) * prior['wins'] / (prior['wins'] + prior['losses'])


def _team_recent_form(games, team_name, n=5):
    tg = []
    for g in games:
        if g['home_name'] == team_name:
            tg.append((g['game_date'], 'W' if g['home_won'] else 'L'))
        elif g['away_name'] == team_name:
            tg.append((g['game_date'], 'L' if g['home_won'] else 'W'))
    tg.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in tg[:n]]


def _team_rest_days(games, team_name, game_date):
    past = [g['game_date'] for g in games
            if (g['home_name'] == team_name or g['away_name'] == team_name) and g['game_date'] < game_date]
    if not past:
        return None
    try:
        return (datetime.strptime(game_date, '%Y-%m-%d') - datetime.strptime(max(past), '%Y-%m-%d')).days
    except ValueError:
        return None


# ── Live scores ──────────────────────────────────────────────────────────

def _iter_day_events(date_str=None):
    target = date_str or _today_et()
    data = _scoreboard(target)
    for event in (data or {}).get('events', []):
        if _event_date_et(event.get('date', '')) == target:
            yield event


def _sides(comp):
    teams = {}
    for c in comp.get('competitors', []):
        t = c.get('team', {})
        teams[c.get('homeAway', 'home')] = {'id': t.get('id'), 'name': t.get('displayName', ''),
                                           'abbr': t.get('abbreviation', ''), 'score': c.get('score')}
    return teams.get('home'), teams.get('away')


def get_live_scores(date_str=None):
    """{game_key: score_dict} for `date_str` (ET, default today) — same
    shape as wnba_api.get_live_scores (open-bet score chips)."""
    from odds_api import _normalize
    scores = {}
    for event in _iter_day_events(date_str):
        comp = event.get('competitions', [{}])[0]
        home, away = _sides(comp)
        if not home or not away:
            continue
        status = _status(comp)
        detail = comp.get('status', {}).get('type', {}).get('detail', '')
        scores[f"{_normalize(home['name'])}_{_normalize(away['name'])}"] = {
            'status': status,
            'away_abbr': away['abbr'], 'home_abbr': home['abbr'],
            'away_logo': logo_url(away['id']) or None, 'home_logo': logo_url(home['id']) or None,
            'away_score': away['score'], 'home_score': home['score'],
            'period': detail if status == 'Live' else ('Final' if status == 'Final' else None),
        }
    return scores


def get_live_game_states(date_str=None):
    """{game_key: {status, away_score, home_score, live_state}} — powers the
    schedule page's live poll (/api/cbb/live-scores)."""
    from odds_api import _normalize
    states = {}
    for event in _iter_day_events(date_str):
        comp = event.get('competitions', [{}])[0]
        home, away = _sides(comp)
        if not home or not away:
            continue
        status = _status(comp)
        detail = comp.get('status', {}).get('type', {}).get('detail')
        states[f"{_normalize(home['name'])}_{_normalize(away['name'])}"] = {
            'status': status, 'away_score': away['score'], 'home_score': home['score'],
            'live_state': {'clock_text': detail} if status == 'Live' else None,
        }
    return states


def get_today_game_count():
    return sum(1 for _ in _iter_day_events())


def get_season_phase():
    data = _scoreboard()
    if not data:
        return None
    try:
        t = data['leagues'][0]['season']['type']['type']
    except (KeyError, IndexError, TypeError):
        return None
    return {1: 'preseason', 2: 'regular', 3: 'postseason'}.get(t)


# ── Schedule context ─────────────────────────────────────────────────────

def _build_game(event, ctx):
    from odds_api import _normalize

    comp = event.get('competitions', [{}])[0]
    venue_obj = comp.get('venue', {}) or {}
    addr = venue_obj.get('address') or {}
    venue = venue_obj.get('fullName', '')
    if addr.get('city'):
        venue += f", {addr['city']}" + (f", {addr['state']}" if addr.get('state') else '')
    status = _status(comp)
    neutral = bool(comp.get('neutralSite'))
    game_date_et = _event_date_et(event.get('date', ''))
    fit, ranks = ctx['fit'], ctx['ranks']

    teams = {}
    for c in comp.get('competitors', []):
        side = c.get('homeAway', 'home')
        t = c.get('team', {})
        records = c.get('records', [])
        w, l = _record_to_wl(_parse_record(records, 'overall', 'total') or '0-0')
        split_summary = _parse_record(records, 'home') if side == 'home' else _parse_record(records, 'road')
        sw, sl = _record_to_wl(split_summary or '0-0')
        cw, cl = _record_to_wl(_parse_record(records, 'vs. conf.', 'vsconf') or '0-0')
        ap = (c.get('curatedRank') or {}).get('current')
        name = t.get('displayName', '')
        conf = cbb_rating.conference_name(t.get('conferenceId'))
        team = {
            'id': t.get('id'), 'name': name,
            'short_name': t.get('shortDisplayName') or t.get('location') or name,
            'abbrev': t.get('abbreviation', ''), 'abbr': t.get('abbreviation', ''),
            'logo_url': logo_url(t.get('id')),
            'wins': w, 'losses': l, 'split_w': sw, 'split_l': sl,
            'split_label': 'Home' if side == 'home' else 'Away',
            'conf_w': cw, 'conf_l': cl, 'conference': conf,
            'rank_display': str(ap) if ap and ap <= 25 else None,
            'side': side, 'score': c.get('score'),
            'form': [], 'ppg': None, 'ppg_allowed': None, 'rest_days': None, 'injuries': [],
            'ratings': None,
        }
        ts = ctx['team_stats'].get(name, {})
        team['ppg'], team['ppg_allowed'] = ts.get('ppg'), ts.get('ppg_allowed')
        team['form'] = _team_recent_form(ctx['recent_log'], name)
        team['rest_days'] = _team_rest_days(ctx['recent_log'], name, game_date_et)
        _apply_prior_season_blend(team, ctx['prior_stats'])
        if fit:
            r = cbb_rating.team_ratings(fit, name, conf)
            r['rank'] = ranks.get(name)
            r['n_teams'] = len(ranks)
            team['ratings'] = r
        teams[side] = team

    home, away = teams.get('home', {}), teams.get('away', {})
    game_odds = odds_api.lookup_game_odds(ctx['odds_map'], home.get('name', ''), away.get('name', ''),
                                          game_date=event.get('date', '')[:13] or None)
    rating = None
    if fit and home and away:
        rating = cbb_rating.predict(fit, home['name'], away['name'], home['conference'],
                                    away['conference'], neutral=neutral)
    try:
        model = cbb_model.predict(home, away, rating=rating, neutral=neutral,
                                  game_time_utc=event.get('date', ''),
                                  market_home_prob=game_odds.get('home_implied') if game_odds else None)
        model['factors'].sort(key=lambda f: abs(f[1]), reverse=True)
    except Exception:
        model = None

    if status == 'Final':
        g = cbb_rating.parse_event(event)
        if g:
            ctx['finals'].append(g)

    espn_odds = (comp.get('odds') or [{}])[0]
    season_type = event.get('season', {}).get('type')
    note = next((n.get('headline') for n in comp.get('notes', []) if n.get('headline')), None)
    a_ab, h_ab = away.get('abbrev', ''), home.get('abbrev', '')
    return {
        'game_id': event.get('id'),
        'game_time_utc': event.get('date', ''),
        'status': status,
        'venue': venue,
        'neutral': neutral,
        'conference_game': bool(comp.get('conferenceCompetition')),
        'away': away, 'home': home,
        'odds_line': espn_odds.get('details', ''),
        'over_under': espn_odds.get('overUnder'),
        'sport': 'CBB',
        'odds': game_odds,
        'model': model,
        'lineups': {'home': [], 'away': []},
        'bet_name': f'{a_ab} @ {h_ab}',
        'game_key': f"{_normalize(home.get('name', ''))}_{_normalize(away.get('name', ''))}",
        'linescore': _parse_linescores(comp) if status != 'Preview' else None,
        'live_detail': comp.get('status', {}).get('type', {}).get('shortDetail') if status == 'Live' else None,
        'is_preseason': season_type == 1,
        'is_playoffs': season_type == 3,
        'game_type_label': note,
        'broadcast': ', '.join(b for bc in comp.get('broadcasts') or [] for b in bc.get('names') or []) or None,
    }


def build_schedule_context(target_date=None):
    """[{date, date_display, games}] for target_date (ET, default today), or
    [] when there are no D1 games / ESPN is unreachable."""
    today_str = target_date or _today_et()
    events = list(_iter_day_events(today_str))
    if not events:
        return []
    season = _get_cbb_season(today_str)
    game_log = _get_season_game_log(season)
    fit = cbb_rating.get_fit(today_str, allow_fetch=not _CACHE_ONLY)
    ranks = {t: r['rank'] for t, r in cbb_rating.rank_table(fit)} if fit else {}
    ctx = {
        'fit': fit, 'ranks': ranks,
        'team_stats': _compute_team_season_stats(game_log),
        'recent_log': _get_recent_game_log(season),
        'prior_stats': _get_prior_season_team_stats(season - 1),
        'odds_map': odds_api.get_odds_map('cbb', events=events),
        'finals': [],
    }
    games = [_build_game(e, ctx) for e in events]
    new_finals = [g for g in ctx['finals'] if g['id'] not in _ingested_ids]
    if new_finals and not _CACHE_ONLY:
        try:
            cbb_stats_db.ingest_games(new_finals)
            _ingested_ids.update(g['id'] for g in new_finals)
        except Exception:
            pass
    games.sort(key=lambda g: (g['game_time_utc'], g['game_id'] or ''))
    date_display = _fmt(datetime.strptime(today_str, '%Y-%m-%d'), '%a, %b %-d')
    return [{'date': today_str, 'date_display': date_display, 'games': games}]


def get_power_ratings(as_of=None):
    """[(team, ratings)] ranked by AdjEM for the ratings table, or []."""
    fit = cbb_rating.get_fit(as_of or _today_et(), allow_fetch=not _CACHE_ONLY)
    if not fit:
        return []
    return cbb_rating.rank_table(fit)
