"""
cfb_rating.py — Two-level CFB power rating: each team's strength is its
conference's level plus its own offset within that conference.

    margin = HFA * (not neutral) + (C[conf_home] + t[home]) - (C[conf_away] + t[away])

Fit by ridge regression on every completed FBS game from the prior two
seasons through the day before the game being rated (prior seasons count
at PRIOR_SEASON_WEIGHT). The team penalty pulls a team with few games
toward its conference's level instead of toward the league average —
that's what lets the rating see a P4-vs-FCS mismatch in week 1, the
blind spot behind most of the old model's spread error (it called 21+
point favorites by ~17 on average; the market and the results said ~32).

Backtest (2026-10-31, research/cfb/, walk-forward, params tuned on 2025
and tested once on 2026): average miss on the 409 graded 2026 games with
a market line was 12.37 pts vs 14.14 for the old cfb_model-derived margin
and 11.47 for the market. A 75/25 blend with the old margin (what
cfb_model.predict_live uses) was 12.31. No ATS edge (47.9%) — this is an
accuracy fix, not a pick source. Per-team home field and kickoff-time
home field were tested and rejected (no persistence year to year).

Also computes the eastbound-travel flag (team crossed 2+ time zones
heading east). Backtest: +3.8 pts to the other side on all 2023-26 games
(90% CI 1.2-5.7) but only ~0.85 pts against the rating out-of-sample, and
19-11 ATS on 30 2026 games — logged as a TRACKING factor, not used.

Data: ESPN scoreboard (weeks 1-16 + postseason). Completed seasons are
cached to disk permanently; the current season refreshes every few hours.
Venue cities are geocoded once via Open-Meteo and cached to disk.
"""
import json
import math
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import requests

ESPN_CFB = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard'
GEOCODE_URL = 'https://geocoding-api.open-meteo.com/v1/search'
_FBS_GROUP = 80
_ET = ZoneInfo('America/New_York')

# Tuned on the 2025 walk-forward (flat optimum — anything nearby scores
# within 0.02 pts): MOV cap, prior-season game weight, team / conference
# ridge penalties.
MOV_CAP = 60
PRIOR_SEASON_WEIGHT = 0.25
TEAM_PENALTY = 1.0
CONF_PENALTY = 1.0
CURRENT_SEASON_TTL = 3 * 3600

# Same mapping as cfb_api._CFB_CONFERENCES — duplicated (like
# cfb_bootstrap.py does) so this module has no cfb_api import cycle.
_CONFERENCES = {
    '151': 'American', '1': 'ACC', '4': 'Big 12', '5': 'Big Ten', '12': 'C-USA',
    '18': 'Independent', '15': 'MAC', '17': 'Mountain West', '9': 'Pac-12',
    '8': 'SEC', '37': 'Sun Belt',
}
_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
    'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
    'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
    'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
    'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
}

# Next to bets.db, same convention as market_edge_calibration.py.
_DATA_DIR = os.path.dirname(os.environ.get('DB_PATH', '')) or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
_GAMES_PATH = os.path.join(_DATA_DIR, 'cfb_rating_games_{season}.json')
_GEO_PATH = os.path.join(_DATA_DIR, 'cfb_venue_geo.json')

_lock = threading.Lock()
_seasons = {}       # season -> (games, fetched_at)
_fits = {}          # (season, as_of_date, n_games) -> fit dict
_geo = None
_geo_failed = {}    # "City|ST" -> time of last failed lookup
_geo_lock = threading.Lock()   # serializes cfb_venue_geo.json writes


def season_for_date(date_str):
    """CFB season a YYYY-MM-DD date belongs to (Jan bowls → prior year)."""
    d = datetime.strptime(date_str[:10], '%Y-%m-%d')
    return d.year if d.month >= 8 else d.year - 1


# ── Data ─────────────────────────────────────────────────────────────────────

def _parse_event(event, season):
    comp = (event.get('competitions') or [{}])[0]
    if comp.get('status', {}).get('type', {}).get('state') != 'post':
        return None
    sides = {c.get('homeAway'): c for c in comp.get('competitors', [])}
    h, a = sides.get('home'), sides.get('away')
    if not h or not a:
        return None
    addr = (comp.get('venue') or {}).get('address') or {}
    try:
        return {
            'id': event['id'], 'season': season, 'ts': event['date'],
            'date': _event_date_et(event['date']),
            'neutral': bool(comp.get('neutralSite')),
            'city': addr.get('city'), 'state': addr.get('state'),
            'home': h['team']['displayName'], 'away': a['team']['displayName'],
            'home_conf': _CONFERENCES.get(str(h['team'].get('conferenceId')), 'FCS'),
            'away_conf': _CONFERENCES.get(str(a['team'].get('conferenceId')), 'FCS'),
            'margin': int(h.get('score') or 0) - int(a.get('score') or 0),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _event_date_et(ts):
    return datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(_ET).strftime('%Y-%m-%d')


def _fetch_week(season, stype, week):
    r = requests.get(ESPN_CFB, params={'seasontype': stype, 'week': week, 'dates': season,
                                       'groups': _FBS_GROUP, 'limit': 400}, timeout=15)
    r.raise_for_status()
    return r.json().get('events', [])


def _fetch_season(season):
    # 17 weekly requests, fetched in parallel — one at a time they added
    # ~10s to any page build that found the current season's copy stale.
    weeks = [(2, w) for w in range(1, 17)] + [(3, 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_fetch_week, season, stype, week) for stype, week in weeks]
    games = {}
    for f in futures:
        try:
            events = f.result()
        except Exception:
            # A failed week would silently drop games from the fit — fail the
            # whole season fetch instead so the caller keeps its last good copy.
            return None
        for ev in events:
            g = _parse_event(ev, season)
            if g:
                games[g['id']] = g
    return sorted(games.values(), key=lambda g: g['ts'])


def _load_season(season, current, allow_fetch):
    """Completed seasons: disk cache forever. Current season: refresh after
    CURRENT_SEASON_TTL. Returns [] when nothing is available."""
    cached = _seasons.get(season)
    if cached and (not current or time.time() - cached[1] < CURRENT_SEASON_TTL or not allow_fetch):
        return cached[0]
    path = _GAMES_PATH.format(season=season)
    if not cached and os.path.exists(path):
        try:
            with open(path) as f:
                blob = json.load(f)
            _seasons[season] = cached = (blob['games'], blob['fetched_at'])
            if not current or time.time() - blob['fetched_at'] < CURRENT_SEASON_TTL or not allow_fetch:
                return blob['games']
        except Exception:
            pass
    if not allow_fetch:
        return cached[0] if cached else []
    games = _fetch_season(season)
    if games is None:
        return cached[0] if cached else []
    now = time.time()
    _seasons[season] = (games, now)
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'games': games, 'fetched_at': now}, f)
    except OSError:
        pass
    return games


# ── Fit ──────────────────────────────────────────────────────────────────────

def _fit(games, season):
    teams = sorted({g['home'] for g in games} | {g['away'] for g in games})
    confs = sorted({g['home_conf'] for g in games} | {g['away_conf'] for g in games})
    ti = {t: i for i, t in enumerate(teams)}
    ci = {c: i for i, c in enumerate(confs)}
    n_c, n_t = len(confs), len(teams)
    X = np.zeros((len(games), 1 + n_c + n_t))
    y = np.empty(len(games))
    w = np.empty(len(games))
    for r, g in enumerate(games):
        X[r, 0] = 0.0 if g['neutral'] else 1.0
        X[r, 1 + ci[g['home_conf']]] += 1
        X[r, 1 + ci[g['away_conf']]] -= 1
        X[r, 1 + n_c + ti[g['home']]] += 1
        X[r, 1 + n_c + ti[g['away']]] -= 1
        y[r] = max(-MOV_CAP, min(MOV_CAP, g['margin']))
        w[r] = PRIOR_SEASON_WEIGHT ** (season - g['season'])
    pen = np.r_[0.0, [CONF_PENALTY] * n_c, [TEAM_PENALTY] * n_t]
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(pen), X.T @ (w * y))
    return {
        'hfa': float(beta[0]),
        'conf': {c: float(beta[1 + ci[c]]) for c in confs},
        'team': {t: float(beta[1 + n_c + ti[t]]) for t in teams},
        'n_games': len(games),
    }


def get_fit(as_of_date, allow_fetch=True):
    """Rating fit using every completed game dated before as_of_date
    (YYYY-MM-DD, ET). None if there's no data to fit."""
    season = season_for_date(as_of_date)
    current = season_for_date(datetime.now(_ET).strftime('%Y-%m-%d'))
    # A cache-only caller (app.py's _cache_only, the Today hub) must never
    # wait behind another thread's multi-season ESPN fetch.
    if not _lock.acquire(blocking=allow_fetch):
        return None
    try:
        games = []
        for s in (season - 2, season - 1, season):
            games.extend(_load_season(s, current=(s >= current), allow_fetch=allow_fetch))
        games = [g for g in games if g['date'] < as_of_date]
        if len(games) < 200:
            return None
        key = (season, as_of_date, len(games))
        fit = _fits.get(key)
        if fit is None:
            try:
                fit = _fit(games, season)
            except np.linalg.LinAlgError:
                return None
            if len(_fits) > 64:
                _fits.clear()
            _fits[key] = fit
        return fit
    finally:
        _lock.release()


def team_rating(fit, team_name, conference):
    """(conference level, within-conference offset) for one team. A team
    the fit hasn't seen gets its conference's level and a 0 offset."""
    return fit['conf'].get(conference or 'FCS', 0.0), fit['team'].get(team_name, 0.0)


def predict_margin(fit, home_name, home_conf, away_name, away_conf, neutral=False):
    """Rating-implied home margin (positive = home favored)."""
    hc, ht = team_rating(fit, home_name, home_conf)
    ac, at = team_rating(fit, away_name, away_conf)
    return (0.0 if neutral else fit['hfa']) + hc + ht - ac - at


# ── Eastbound travel (tracking only) ──────────────────────────────────────────

def _load_geo():
    global _geo
    if _geo is None:
        try:
            with open(_GEO_PATH) as f:
                _geo = json.load(f)
        except Exception:
            _geo = {}
    return _geo


def _geocode(city, state, allow_fetch):
    """{lat, lon, tz} for a US city, cached to disk. None if unknown."""
    if not city or not state:
        return None
    geo = _load_geo()
    key = f'{city}|{state}'
    if key in geo or not allow_fetch or time.time() - _geo_failed.get(key, 0) < 3600:
        return geo.get(key)
    try:
        r = requests.get(GEOCODE_URL, params={'name': city, 'count': 20, 'country_code': 'US'}, timeout=5)
        r.raise_for_status()
        want = _STATES.get(state, state)
        hit = next((x for x in r.json().get('results') or [] if x.get('admin1') == want), None)
    except Exception:
        # Don't cache a network failure as "unknown" on disk, but back off for
        # an hour so an outage doesn't cost a timeout on every game build.
        _geo_failed[key] = time.time()
        return None
    with _geo_lock:
        geo[key] = hit and {'lat': hit['latitude'], 'lon': hit['longitude'], 'tz': hit['timezone']}
        try:
            with open(_GEO_PATH, 'w') as f:
                json.dump(geo, f)
        except OSError:
            pass
    return geo[key]


def _home_city(team_name):
    """City|state a team most often hosts non-neutral games in, from the
    cached seasons. None if it never hosts (most FCS teams)."""
    counts = Counter()
    for games, _ in list(_seasons.values()):
        for g in games:
            if g['home'] == team_name and not g['neutral'] and g.get('city'):
                counts[(g['city'], g['state'])] += 1
    return counts.most_common(1)[0][0] if counts else None


def eastbound_2tz(team_name, venue_city, venue_state, kickoff_utc, allow_fetch=True):
    """True if this team crossed 2+ time zones heading east to reach the
    venue, False if not, None if either location is unknown."""
    home = _home_city(team_name)
    if not home:
        return None
    src = _geocode(home[0], home[1], allow_fetch)
    dst = _geocode(venue_city, venue_state, allow_fetch)
    if not src or not dst:
        return None
    try:
        k = datetime.fromisoformat(kickoff_utc.replace('Z', '+00:00'))
    except (AttributeError, ValueError):
        k = datetime.now(_ET)
    off = lambda tz: k.astimezone(ZoneInfo(tz)).utcoffset() / timedelta(hours=1)
    return off(dst['tz']) - off(src['tz']) >= 2


# Fitted on all 2023-26 games (see module docstring) — used only to size the
# tracking factor's display value, never added to the model's logit.
EASTBOUND_PTS = 3.8


def prefetch(events, allow_fetch=True):
    """Warm everything rating_inputs() will need for a slate of ESPN
    scoreboard events: the fit for each game date, then every unseen home
    city and venue geocoded in parallel. Without this, a week with new
    venues paid one sequential geocode round-trip per lookup inside the
    per-game build loop."""
    dates, places, teams = set(), set(), set()
    for ev in events:
        comp = (ev.get('competitions') or [{}])[0]
        if ev.get('date'):
            try:
                dates.add(_event_date_et(ev['date']))
            except ValueError:
                pass
        addr = (comp.get('venue') or {}).get('address') or {}
        places.add((addr.get('city'), addr.get('state')))
        teams.update((c.get('team') or {}).get('displayName', '') for c in comp.get('competitors', []))
    for d in sorted(dates):
        get_fit(d, allow_fetch=allow_fetch)
    if not allow_fetch:
        return
    places.update(_home_city(t) or (None, None) for t in teams if t)
    geo = _load_geo()
    todo = [(c, s) for c, s in places if c and s and f'{c}|{s}' not in geo]
    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda p: _geocode(p[0], p[1], True), todo))


def rating_inputs(home, away, neutral, venue_city, venue_state, kickoff_utc, allow_fetch=True):
    """Everything cfb_model.predict_live needs from this module, or None if
    no fit is available. home/away are cfb_api team dicts (name, conference)."""
    if not kickoff_utc:
        return None
    try:
        as_of = _event_date_et(kickoff_utc)
    except ValueError:
        return None
    fit = get_fit(as_of, allow_fetch=allow_fetch)
    if fit is None:
        return None
    h_name, a_name = home.get('name', ''), away.get('name', '')
    h_conf, a_conf = home.get('conference'), away.get('conference')
    out = {
        'margin': predict_margin(fit, h_name, h_conf, a_name, a_conf, neutral),
        'home_rating': sum(team_rating(fit, h_name, h_conf)),
        'away_rating': sum(team_rating(fit, a_name, a_conf)),
        'neutral': neutral,
        'eastbound': None,
    }
    try:
        h_east = eastbound_2tz(h_name, venue_city, venue_state, kickoff_utc, allow_fetch)
        a_east = eastbound_2tz(a_name, venue_city, venue_state, kickoff_utc, allow_fetch)
        if h_east is not None and a_east is not None:
            out['eastbound'] = int(a_east) - int(h_east)   # +1 = away team flew east 2+ zones
    except Exception:
        pass
    return out
