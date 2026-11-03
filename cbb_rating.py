"""
cbb_rating.py — Opponent-adjusted efficiency ratings for men's college
basketball (Division I), KenPom-style: every team gets an adjusted
offense, adjusted defense and adjusted tempo, each split into its
conference's level plus the team's own offset within that conference.

    pts per 100 poss (team X vs Y) = MU + HFA_E*side + (Co[conf X] + o[X]) - (Cd[conf Y] + d[Y])
    possessions (X vs Y)           = MU_T + (Ct[conf X] + t[X]) + (Ct[conf Y] + t[Y])

side = +1/2 for the home team, -1/2 for the away team, 0 at a neutral
site. Fit by weighted ridge regression on every completed D1 game from
the prior two seasons through the day before the game being rated
(prior seasons count at PRIOR_SEASON_WEIGHT ** seasons_ago). As in
cfb_rating.py, the team penalty pulls a team with few games toward its
conference's level instead of toward the national average — the thing
that lets the fit see a Big Ten-vs-SWAC mismatch in the first week.

Why not the NBA/WNBA inputs (raw PPG / PPG allowed / win%) alone: with
~360 teams and wildly different schedules, raw per-game numbers mostly
measure who a team played. The NBA-style factors are still computed and
tested as add-ons on top of this rating in cbb_bootstrap.py.

Possessions: ESPN's scoreboard carries FGA and FTA per team but not
offensive rebounds or turnovers, so a game's possessions are estimated as
FGA + 0.475*FTA + POSS_ADJ, averaged over both teams (D1 averages for
TOV and OREB are close, so they largely cancel; POSS_ADJ absorbs the
remainder). Margins are exact regardless — only the per-100 scale of
AdjO/AdjD/tempo carries that approximation.

Data: ESPN day scoreboards (groups=50 = all of D1). Completed seasons are
cached to disk permanently; the current season re-fetches only its most
recent days every few hours.
"""
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import requests

ESPN_CBB = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
D1_GROUP = 50
_ET = ZoneInfo('America/New_York')

# Tuned by cbb_bootstrap.py's walk-forward (see that script's docstring).
MOV_CAP = 70            # cap on per-game margin when fitting (barely binds; flat 45-200)
PRIOR_SEASON_WEIGHT = 0.15
TEAM_PENALTY = 2.0      # ridge strength on team offsets (in game-weight units)
CONF_PENALTY = 1.0
TEMPO_PENALTY = 4.0
POSS_ADJ = 1.5
HALF_LIFE_DAYS = None   # optional extra recency decay on every game's weight
CURRENT_SEASON_TTL = 3 * 3600
REFRESH_DAYS = 4        # current season: re-fetch this many most recent days each refresh
MIN_GAMES = 300

# ESPN conferenceId → short label. Anything else (non-D1 opponents,
# independents) lands in 'Other'.
CONFERENCES = {
    '1': 'America East', '62': 'American', '3': 'A-10', '2': 'ACC', '46': 'ASUN',
    '8': 'Big 12', '4': 'Big East', '5': 'Big Sky', '6': 'Big South', '7': 'Big Ten',
    '9': 'Big West', '10': 'CAA', '11': 'C-USA', '45': 'Horizon', '12': 'Ivy',
    '13': 'MAAC', '14': 'MAC', '16': 'MEAC', '18': 'MVC', '44': 'Mountain West',
    '19': 'NEC', '20': 'OVC', '22': 'Patriot', '23': 'SEC', '24': 'SoCon',
    '25': 'Southland', '26': 'SWAC', '49': 'Summit', '27': 'Sun Belt', '30': 'WAC/UAC',
    '29': 'WCC',
}
OTHER_CONF = 'Other'

# Next to bets.db (the /data volume in Docker), same convention as
# cfb_rating.py / market_edge_calibration.py.
_DATA_DIR = os.path.dirname(os.environ.get('DB_PATH', '')) or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
_GAMES_PATH = os.path.join(_DATA_DIR, 'cbb_rating_games_{season}.json')

_seasons = {}       # season -> (games, fetched_at)
_fits = {}          # (season, as_of_date, n_games) -> fit
_lock = threading.Lock()


def conference_name(conf_id):
    return CONFERENCES.get(str(conf_id), OTHER_CONF) if conf_id is not None else OTHER_CONF


def season_for_date(date_str):
    """CBB season label for a YYYY-MM-DD date: the calendar year the season
    ends in (ESPN's convention — 2025-26 is season 2026). Jul-Dec belongs to
    the next year's season."""
    d = datetime.strptime(date_str[:10], '%Y-%m-%d')
    return d.year + 1 if d.month >= 7 else d.year


def season_window(season):
    """(first, last) calendar dates worth scanning for a season."""
    return date(season - 1, 11, 1), date(season, 4, 10)


def _event_date_et(ts):
    return datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(_ET).strftime('%Y-%m-%d')


def _stat(competitor, name):
    for s in competitor.get('statistics') or []:
        if s.get('name') == name:
            try:
                return float(s.get('displayValue'))
            except (TypeError, ValueError):
                return None
    return None


_BOX_STATS = (('fgm', 'fieldGoalsMade'), ('fga', 'fieldGoalsAttempted'),
              ('tpm', 'threePointFieldGoalsMade'), ('tpa', 'threePointFieldGoalsAttempted'),
              ('ftm', 'freeThrowsMade'), ('fta', 'freeThrowsAttempted'),
              ('reb', 'rebounds'), ('ast', 'assists'))


def parse_event(event):
    """One completed game → flat dict, or None (not final / malformed)."""
    comp = (event.get('competitions') or [{}])[0]
    if comp.get('status', {}).get('type', {}).get('state') != 'post':
        return None
    sides = {c.get('homeAway'): c for c in comp.get('competitors', [])}
    h, a = sides.get('home'), sides.get('away')
    if not h or not a:
        return None
    try:
        hs, as_ = int(h.get('score') or 0), int(a.get('score') or 0)
    except (TypeError, ValueError):
        return None
    if hs == 0 and as_ == 0:
        return None   # postponed/cancelled games occasionally come back 'post' 0-0

    def _poss(c):
        fga, fta = _stat(c, 'fieldGoalsAttempted'), _stat(c, 'freeThrowsAttempted')
        return fga + 0.475 * fta + POSS_ADJ if fga is not None and fta is not None else None

    hp, ap = _poss(h), _poss(a)
    poss = (hp + ap) / 2 if hp and ap else None
    # Overtime periods add possessions without being "the team's tempo" —
    # scale back to a 40-minute game.
    n_periods = max(len(h.get('linescores') or []), 2)
    if poss and n_periods > 2:
        poss *= 40.0 / (40 + 5 * (n_periods - 2))

    addr = (comp.get('venue') or {}).get('address') or {}
    return {
        'id': event['id'],
        'season': (event.get('season') or {}).get('year'),
        'season_type': (event.get('season') or {}).get('type'),
        'ts': event.get('date', ''),
        'date': _event_date_et(event['date']),
        'neutral': bool(comp.get('neutralSite')),
        'conf_game': bool(comp.get('conferenceCompetition')),
        'city': addr.get('city'), 'state': addr.get('state'),
        'home': h['team']['displayName'], 'away': a['team']['displayName'],
        'home_id': h['team'].get('id'), 'away_id': a['team'].get('id'),
        'home_abbr': h['team'].get('abbreviation'), 'away_abbr': a['team'].get('abbreviation'),
        'home_conf': conference_name(h['team'].get('conferenceId')),
        'away_conf': conference_name(a['team'].get('conferenceId')),
        'home_score': hs, 'away_score': as_,
        # Scoreboard box totals (no turnovers/offensive boards on this endpoint).
        **{f'{side}_{key}': _stat(c, name) for side, c in (('home', h), ('away', a))
           for key, name in _BOX_STATS},
        'periods': n_periods,
        'poss': round(poss, 2) if poss else None,
    }


# ── Data ─────────────────────────────────────────────────────────────────────

def fetch_day(day_str):
    """All D1 events on one ET date (YYYY-MM-DD). Raises on HTTP failure."""
    r = requests.get(ESPN_CBB, params={'dates': day_str.replace('-', ''), 'groups': D1_GROUP,
                                       'limit': 500}, timeout=20)
    r.raise_for_status()
    return r.json().get('events', [])


def _fetch_days(days):
    """{day: [games]} for every day fetched OK; failed days are omitted."""
    out = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {d: pool.submit(fetch_day, d) for d in days}
    for d, f in futures.items():
        try:
            events = f.result()
        except Exception:
            continue
        games = []
        for ev in events:
            g = parse_event(ev)
            if g and g['date'] == d:
                games.append(g)
        out[d] = games
    return out


def _days_between(start, end):
    d, out = start, []
    while d <= end:
        out.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    return out


def _save(season, games, fetched_at, fetched_days):
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        tmp = _GAMES_PATH.format(season=season) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'games': games, 'fetched_at': fetched_at, 'days': sorted(fetched_days)}, f)
        os.replace(tmp, _GAMES_PATH.format(season=season))
    except OSError:
        pass


def load_season(season, allow_fetch=True, force_days=None):
    """Every completed D1 game in `season`, chronological. Completed seasons
    are read from disk forever once fully fetched; the current season
    re-fetches its last REFRESH_DAYS days after CURRENT_SEASON_TTL (plus any
    day never fetched successfully). [] when nothing is available."""
    today = datetime.now(_ET).date()
    first, last = season_window(season)
    current = first <= today <= last + timedelta(days=1)
    cached = _seasons.get(season)
    if cached is None:
        path = _GAMES_PATH.format(season=season)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    blob = json.load(f)
                cached = (blob['games'], blob['fetched_at'], set(blob.get('days') or []))
                _seasons[season] = cached
            except Exception:
                cached = None
    if cached and not force_days:
        fresh = time.time() - cached[1] < CURRENT_SEASON_TTL
        wanted = set(_days_between(first, min(last, today - timedelta(days=1))))
        missing = wanted - cached[2]
        if not allow_fetch or (not missing and (not current or fresh)):
            return cached[0]
    if not allow_fetch:
        return cached[0] if cached else []

    games_by_id = {g['id']: g for g in (cached[0] if cached else [])}
    have_days = set(cached[2]) if cached else set()
    end = min(last, today)
    all_days = _days_between(first, end)
    todo = [d for d in all_days if d not in have_days]
    if current:
        todo += [d for d in all_days[-REFRESH_DAYS:] if d not in todo]
    if force_days:
        todo += [d for d in force_days if d not in todo]
    if not todo:
        return cached[0] if cached else []
    fetched = _fetch_days(todo)
    today_str = today.strftime('%Y-%m-%d')
    for d, games in fetched.items():
        for gid in [gid for gid, g in games_by_id.items() if g['date'] == d]:
            if gid not in {g['id'] for g in games}:
                games_by_id.pop(gid)
        for g in games:
            games_by_id[g['id']] = g
        # Today's slate isn't done — don't mark it as fetched for good.
        if d < today_str:
            have_days.add(d)
    games = sorted(games_by_id.values(), key=lambda g: g['ts'])
    now = time.time()
    _seasons[season] = (games, now, have_days)
    _save(season, games, now, have_days)
    return games


# ── Fit ──────────────────────────────────────────────────────────────────────

def _ridge_sparse(cols, vals, y, w, pen):
    """Weighted ridge solve where every row has the same small number of
    nonzeros: cols/vals are (n_rows, k) arrays. Builds X'WX directly from
    the nonzeros (no scipy in the prod image, and a dense 25k x 800 X'X
    took ~1s per fit — too slow for the bootstrap's daily refits)."""
    n = len(pen)
    A = np.diag(np.asarray(pen, dtype=float))
    b = np.zeros(n)
    k = cols.shape[1]
    for i in range(k):
        wi = w * vals[:, i]
        np.add.at(b, cols[:, i], wi * y)
        for j in range(k):
            np.add.at(A, (cols[:, i], cols[:, j]), wi * vals[:, j])
    return np.linalg.solve(A, b)


def fit(games, season, params=None):
    """Fit offense/defense/tempo ratings on `games` (all dated before the
    target). `params` overrides the module constants (bootstrap tuning)."""
    p = {'mov_cap': MOV_CAP, 'prior_w': PRIOR_SEASON_WEIGHT, 'team_pen': TEAM_PENALTY,
         'conf_pen': CONF_PENALTY, 'tempo_pen': TEMPO_PENALTY, 'half_life': HALF_LIFE_DAYS}
    if params:
        p.update(params)
    games = [g for g in games if g.get('poss')]
    if len(games) < 50:
        raise ValueError('too few games')
    ref = datetime.strptime(max(g['date'] for g in games), '%Y-%m-%d') + timedelta(days=1)

    def _weight(g):
        gw = p['prior_w'] ** max(0, season - g['season'])
        if p['half_life']:
            age = (ref - datetime.strptime(g['date'], '%Y-%m-%d')).days
            gw *= 0.5 ** (age / p['half_life'])
        return gw
    teams = sorted({g['home'] for g in games} | {g['away'] for g in games})
    confs = sorted({g['home_conf'] for g in games} | {g['away_conf'] for g in games})
    ti = {t: i for i, t in enumerate(teams)}
    ci = {c: i for i, c in enumerate(confs)}
    team_conf = {}
    for g in games:   # latest conference wins (realignment)
        team_conf[g['home']] = g['home_conf']
        team_conf[g['away']] = g['away_conf']
    n_c, n_t, n_g = len(confs), len(teams), len(games)

    # ── efficiency: two rows per game (each side's offense vs other's defense)
    # columns: [mu, hfa, Co(n_c), o(n_t), Cd(n_c), d(n_t)]
    oc, ot = 2, 2 + n_c
    dc, dt = 2 + n_c + n_t, 2 + 2 * n_c + n_t
    cols = np.zeros((2 * n_g, 6), dtype=np.int64)
    vals = np.zeros((2 * n_g, 6))
    y = np.empty(2 * n_g)
    w = np.empty(2 * n_g)
    cap = p['mov_cap']
    for r, g in enumerate(games):
        poss = g['poss']
        margin = g['home_score'] - g['away_score']
        # Cap the margin (blowout garbage time), keeping the game's total.
        if abs(margin) > cap:
            total = g['home_score'] + g['away_score']
            margin = math.copysign(cap, margin)
            hs, as_ = (total + margin) / 2, (total - margin) / 2
        else:
            hs, as_ = g['home_score'], g['away_score']
        gw = _weight(g)
        side = 0.0 if g['neutral'] else 0.5
        for k, (off, dfn, pts, sd) in enumerate((
                (g['home'], g['away'], hs, side), (g['away'], g['home'], as_, -side))):
            row = 2 * r + k
            cols[row] = (0, 1, oc + ci[team_conf[off]], ot + ti[off],
                         dc + ci[team_conf[dfn]], dt + ti[dfn])
            vals[row] = (1.0, sd, 1.0, 1.0, -1.0, -1.0)
            y[row] = 100.0 * pts / poss
            w[row] = gw
    pen = np.r_[0.0, 0.0, [p['conf_pen']] * n_c, [p['team_pen']] * n_t,
                [p['conf_pen']] * n_c, [p['team_pen']] * n_t]
    beta = _ridge_sparse(cols, vals, y, w, pen)

    # ── tempo: one row per game
    cols_t = np.zeros((n_g, 5), dtype=np.int64)
    vals_t = np.ones((n_g, 5))
    yt = np.empty(n_g)
    wt = np.empty(n_g)
    for r, g in enumerate(games):
        h, a = g['home'], g['away']
        cols_t[r] = (0, 1 + ci[team_conf[h]], 1 + n_c + ti[h], 1 + ci[team_conf[a]], 1 + n_c + ti[a])
        yt[r] = g['poss']
        wt[r] = _weight(g)
    pen_t = np.r_[0.0, [p['conf_pen']] * n_c, [p['tempo_pen']] * n_t]
    bt = _ridge_sparse(cols_t, vals_t, yt, wt, pen_t)

    mu, mu_t = float(beta[0]), float(bt[0])
    return {
        'season': season,
        'mu': mu, 'hfa': float(beta[1]), 'mu_t': mu_t,
        'conf_o': {c: float(beta[oc + i]) for c, i in ci.items()},
        'conf_d': {c: float(beta[dc + i]) for c, i in ci.items()},
        'conf_t': {c: float(bt[1 + i]) for c, i in ci.items()},
        'team_o': {t: float(beta[ot + i]) for t, i in ti.items()},
        'team_d': {t: float(beta[dt + i]) for t, i in ti.items()},
        'team_t': {t: float(bt[1 + n_c + i]) for t, i in ti.items()},
        'team_conf': team_conf,
        'n_games': n_g,
    }


def team_ratings(f, team, conf=None, ndigits=1):
    """{adj_o, adj_d, adj_em, adj_t} for one team, in points per 100
    possessions (adj_d: points allowed — lower is better) and possessions
    per 40 minutes against an average opponent (adj_t). A team the fit hasn't
    seen gets its conference level."""
    conf = conf or f['team_conf'].get(team, OTHER_CONF)
    o = f['mu'] + f['conf_o'].get(conf, 0.0) + f['team_o'].get(team, 0.0)
    d = f['mu'] - f['conf_d'].get(conf, 0.0) - f['team_d'].get(team, 0.0)
    t = f['mu_t'] + f['conf_t'].get(conf, 0.0) + f['team_t'].get(team, 0.0)
    r = (lambda v: round(v, ndigits)) if ndigits is not None else (lambda v: v)
    return {'adj_o': r(o), 'adj_d': r(d), 'adj_em': r(o - d), 'adj_t': r(t)}


def predict(f, home, away, home_conf=None, away_conf=None, neutral=False):
    """Rating-implied {margin, total, poss, home_pts, away_pts} (margin
    positive = home favored)."""
    h = team_ratings(f, home, home_conf, ndigits=None)
    a = team_ratings(f, away, away_conf, ndigits=None)
    poss = h['adj_t'] + a['adj_t'] - f['mu_t']
    side = 0.0 if neutral else 0.5
    h_eff = h['adj_o'] + a['adj_d'] - f['mu'] + f['hfa'] * side
    a_eff = a['adj_o'] + h['adj_d'] - f['mu'] - f['hfa'] * side
    hp, ap = h_eff * poss / 100, a_eff * poss / 100
    return {'margin': round(hp - ap, 2), 'total': round(hp + ap, 1), 'poss': round(poss, 1),
            'home_pts': round(hp, 1), 'away_pts': round(ap, 1),
            'home': team_ratings(f, home, home_conf), 'away': team_ratings(f, away, away_conf)}


def rank_table(f):
    """[(team, ratings)] sorted by adj_em descending, with 'rank' set."""
    rows = [(t, {**team_ratings(f, t), 'conference': c}) for t, c in f['team_conf'].items()
            if c != OTHER_CONF]
    rows.sort(key=lambda r: r[1]['adj_em'], reverse=True)
    for i, (_, r) in enumerate(rows, 1):
        r['rank'] = i
    return rows


def get_fit(as_of_date, allow_fetch=True):
    """Ratings fit on every completed game dated before as_of_date
    (YYYY-MM-DD, ET), the two prior seasons included. None if unavailable."""
    season = season_for_date(as_of_date)
    if not _lock.acquire(blocking=allow_fetch):
        return None
    try:
        games = []
        for s in (season - 2, season - 1, season):
            games.extend(load_season(s, allow_fetch=allow_fetch))
        games = [g for g in games if g['date'] < as_of_date]
        if len(games) < MIN_GAMES:
            return None
        key = (season, as_of_date, len(games))
        result = _fits.get(key)
        if result is None:
            try:
                result = fit(games, season)
            except (np.linalg.LinAlgError, ValueError):
                return None
            if len(_fits) > 32:
                _fits.clear()
            _fits[key] = result
        return result
    finally:
        _lock.release()
