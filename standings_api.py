"""Full-league standings for the Stats page (The Almanac), all six sports.

Every sport comes back in one normalized row shape so the template doesn't
branch per sport:

    {abbr, name, logo, group, w, l, otl, pct, gp, diff, pf, pa,
     home, away, l10, streak, form}

  - `otl` is NHL overtime losses / NFL ties (None elsewhere).
  - `diff`/`pf`/`pa` are season totals of runs, goals or points.
  - `form` is this team's last ≤10 results, NEWEST FIRST — built from the
    season game log each sport module already loads (NFL/NBA/WNBA/CFB),
    NHL's recent-form map, or an MLB schedule window. It's the one source of
    truth for Last 5 / Last 10 / streak on the Stats page; a feed's own
    l10/streak string is used when it has one, else derived from `form`.

Sources: NHL api-web /standings/now, MLB statsapi /standings, and ESPN's
/apis/v2 standings for the rest (season pinned to each module's own
"current or most recent" season helper — ESPN flips its default to the
next season early, e.g. WNBA showing an all-0-0 2027 table in October).
"""
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

_ET = ZoneInfo('America/New_York')
_TTL = 30 * 60
_cache = {}

_ESPN = 'https://site.api.espn.com/apis/v2/sports/{path}/standings'
_ESPN_PATHS = {'NFL': 'football/nfl', 'NBA': 'basketball/nba',
               'WNBA': 'basketball/wnba', 'CFB': 'football/college-football'}
SPORTS = ('NFL', 'NBA', 'NHL', 'MLB', 'CFB', 'WNBA')


def _get_json(url, params=None):
    # One retry: the NHL and ESPN feeds occasionally drop a single request,
    # and a miss here blanks the whole Stats page (and that league's search
    # entries) until the next fetch.
    for attempt in (1, 2):
        try:
            r = requests.get(url, params=params or {}, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 2:
                return None
            time.sleep(0.5)


def _cached(key, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[1] < _TTL:
        return hit[0]
    val = fn()
    if val:
        _cache[key] = (val, time.time())
    return val if val is not None else (hit[0] if hit else None)


def _wl(s):
    """'12-5' / '12-5-1' → (12, 5); None if unparseable."""
    try:
        parts = [int(float(x)) for x in str(s).split('-')]
        return parts[0], parts[1]
    except (ValueError, IndexError, TypeError):
        return None


def _record_str(form):
    if not form:
        return None
    w = sum(1 for r in form if r == 'W')
    return f"{w}-{len(form) - w}"


def _streak_str(form):
    if not form:
        return None
    n = 1
    while n < len(form) and form[n] == form[0]:
        n += 1
    return f"{form[0]}{n}"


def _finish(rows):
    """Fill derived fields, then rank by win pct (ties → differential)."""
    for r in rows:
        decided = r['w'] + r['l'] + (r['otl'] or 0)
        r['gp'] = r.get('gp') or decided
        r['pct'] = (r['w'] / decided) if decided else None
        r['form'] = (r.get('form') or [])[:10]
        r['l10'] = r.get('l10') or _record_str(r['form'])
        r['streak'] = r.get('streak') or _streak_str(r['form'])
    rows.sort(key=lambda r: (r['pct'] if r['pct'] is not None else -1, r['diff'] or 0), reverse=True)
    for i, r in enumerate(rows):
        r['rank'] = i + 1
    return rows


# ── Form (last 10, newest first) ────────────────────────────────────────────

def _form_from_game_log(log):
    """{team display name: ['W','L',...]} newest first, from the
    {game_date, home_name, away_name, home_won} logs the ESPN-sport modules
    expose via _get_season_game_log."""
    by_team = {}
    for g in sorted(log or [], key=lambda g: g['game_date'], reverse=True):
        by_team.setdefault(g['home_name'], []).append('W' if g['home_won'] else 'L')
        by_team.setdefault(g['away_name'], []).append('L' if g['home_won'] else 'W')
    return {k: v[:10] for k, v in by_team.items()}


def _espn_form(sport):
    try:
        if sport == 'NFL':
            import nfl_api as m
            season = m._get_nfl_season()
        elif sport == 'NBA':
            import nba_api as m
            season = m._get_nba_season()
        elif sport == 'WNBA':
            import wnba_api as m
            season = m._get_wnba_season()
        else:
            import cfb_api as m
            season = m._get_cfb_season()
        return season, _form_from_game_log(m._get_season_game_log(season))
    except Exception:
        return None, {}


# ── ESPN sports ─────────────────────────────────────────────────────────────

def _espn_groups(node, out):
    """Flatten ESPN's children tree to (group name, entries) pairs — NFL/NBA/
    WNBA are conference → entries, CFB is conference → entries too (FBS)."""
    entries = (node.get('standings') or {}).get('entries')
    if entries:
        out.append((node.get('name') or '', entries))
    for c in node.get('children', []):
        _espn_groups(c, out)
    return out


def _espn_rows(sport):
    season, form = _espn_form(sport)
    params = {'seasontype': 2}
    # NBA's ESPN standings `season` is the END year (2027 = 2026-27), unlike
    # nba_api's own start-year helper — its default is already the current
    # season, so only pin the others.
    if season and sport != 'NBA':
        params['season'] = season
    data = _get_json(_ESPN.format(path=_ESPN_PATHS[sport]), params)
    if not data:
        return None
    rows = []
    for group, entries in _espn_groups(data, []):
        for e in entries:
            t = e.get('team', {})
            # CFB repeats every stat name per split (home/away/conference) —
            # first occurrence is the overall one, so never overwrite.
            st = {}
            for s in e.get('stats', []):
                for k in (s.get('name'), s.get('type')):
                    if k:
                        st.setdefault(k, s)
            num = lambda k: (st.get(k) or {}).get('value')
            disp = lambda k: (st.get(k) or {}).get('displayValue')
            name = t.get('displayName', '')
            ties = num('ties')
            gp = None
            if num('avgPointsFor') and num('pointsFor'):
                gp = round(num('pointsFor') / num('avgPointsFor'))
            # CFB's wins/losses stats are CONFERENCE-only; 'overall' is the
            # real record everywhere it's present.
            wl = _wl(disp('overall')) or (int(num('wins') or 0), int(num('losses') or 0))
            rows.append({
                'abbr': t.get('abbreviation', ''), 'name': name,
                'logo': ((t.get('logos') or [{}])[0]).get('href', ''),
                'group': group.replace(' Conference', ''),
                'w': wl[0], 'l': wl[1],
                'otl': int(ties) if ties else None, 'gp': gp,
                'diff': num('pointDifferential') if num('pointDifferential') is not None else num('differential'),
                'pf': num('pointsFor'), 'pa': num('pointsAgainst'),
                'home': disp('Home'), 'away': disp('Road') or disp('Away'),
                'l10': disp('Last Ten Games'), 'streak': disp('streak'),
                'form': form.get(name, []),
            })
    return {'season': season, 'rows': _finish(rows)}


# ── NHL ─────────────────────────────────────────────────────────────────────

def _nhl_rows():
    data = _get_json('https://api-web.nhle.com/v1/standings/now')
    if not data:
        return None
    try:
        import nhl_api
        form_map, _last = nhl_api._get_recent_form_map()  # chronological
        form_map = {k: list(reversed(v)) for k, v in form_map.items()}
        season = nhl_api._get_nhl_season_start_year()
    except Exception:
        form_map, season = {}, None
    rows = []
    for s in data.get('standings', []):
        abbr = (s.get('teamAbbrev') or {}).get('default', '')
        rows.append({
            'abbr': abbr, 'name': (s.get('teamName') or {}).get('default', abbr),
            'logo': s.get('teamLogo', ''),
            'group': s.get('divisionName', ''),
            'w': s.get('wins', 0), 'l': s.get('losses', 0), 'otl': s.get('otLosses', 0),
            'gp': s.get('gamesPlayed'), 'diff': s.get('goalDifferential'),
            'pf': s.get('goalFor'), 'pa': s.get('goalAgainst'),
            'home': f"{s.get('homeWins', 0)}-{s.get('homeLosses', 0)}-{s.get('homeOtLosses', 0)}",
            'away': f"{s.get('roadWins', 0)}-{s.get('roadLosses', 0)}-{s.get('roadOtLosses', 0)}",
            'l10': f"{s.get('l10Wins', 0)}-{s.get('l10Losses', 0)}-{s.get('l10OtLosses', 0)}",
            'streak': f"{s.get('streakCode', '')}{s.get('streakCount', '')}" or None,
            'form': form_map.get(abbr, []),
        })
    rows = _finish(rows)
    # NHL ranks by points (2 per W, 1 per OTL), not win pct.
    rows.sort(key=lambda r: (2 * r['w'] + (r['otl'] or 0), r['diff'] or 0), reverse=True)
    for i, r in enumerate(rows):
        r['rank'] = i + 1
        r['pts'] = 2 * r['w'] + (r['otl'] or 0)
    return {'season': season, 'rows': rows}


# ── MLB ─────────────────────────────────────────────────────────────────────

_MLB = 'https://statsapi.mlb.com/api/v1'


def _mlb_season():
    today = datetime.now(_ET).date()
    return today.year if today.month >= 3 else today.year - 1


def _mlb_form(season):
    """{team_id: [...]} newest first from the last 30 days of the regular
    season (or up to yesterday, mid-season) — one schedule call."""
    meta = _get_json(f'{_MLB}/seasons/{season}', {'sportId': 1}) or {}
    end = (meta.get('seasons') or [{}])[0].get('regularSeasonEndDate')
    yesterday = (datetime.now(_ET) - timedelta(days=1)).strftime('%Y-%m-%d')
    end = min(end, yesterday) if end else yesterday
    start = (datetime.strptime(end, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
    data = _get_json(f'{_MLB}/schedule', {'sportId': 1, 'startDate': start, 'endDate': end, 'gameType': 'R'})
    form = {}
    for d in reversed((data or {}).get('dates', [])):
        for g in reversed(d.get('games', [])):
            if g.get('status', {}).get('abstractGameState') != 'Final':
                continue
            a, h = g['teams']['away'], g['teams']['home']
            a_won = (a.get('score') or 0) > (h.get('score') or 0)
            form.setdefault(a['team']['id'], []).append('W' if a_won else 'L')
            form.setdefault(h['team']['id'], []).append('L' if a_won else 'W')
    return form


def _mlb_rows():
    season = _mlb_season()
    data = _get_json(f'{_MLB}/standings', {'leagueId': '103,104', 'season': season,
                                           'standingsTypes': 'regularSeason', 'hydrate': 'team,division'})
    if not data:
        return None
    form = _mlb_form(season)
    rows = []
    for rec in data.get('records', []):
        group = (rec.get('division') or {}).get('nameShort') or (rec.get('division') or {}).get('name', '')
        for t in rec.get('teamRecords', []):
            team = t.get('team', {})
            splits = {s['type']: f"{s['wins']}-{s['losses']}" for s in (t.get('records') or {}).get('splitRecords', [])}
            rows.append({
                'abbr': team.get('abbreviation', ''), 'name': team.get('name', ''),
                'logo': f"https://www.mlbstatic.com/team-logos/{team.get('id')}.svg",
                'group': group, 'w': t.get('wins', 0), 'l': t.get('losses', 0), 'otl': None,
                'gp': t.get('gamesPlayed'), 'diff': t.get('runDifferential'),
                'pf': t.get('runsScored'), 'pa': t.get('runsAllowed'),
                'home': splits.get('home'), 'away': splits.get('away'), 'l10': splits.get('lastTen'),
                'streak': (t.get('streak') or {}).get('streakCode'),
                'form': form.get(team.get('id'), []),
            })
    return {'season': season, 'rows': _finish(rows)}


def get_standings(sport):
    """{'season': int|None, 'rows': [...], 'groups': [name, ...]} or None if
    the feed is unreachable. Cached 30 min per sport."""
    sport = sport.upper()
    if sport not in SPORTS:
        return None

    def _load():
        if sport == 'NHL':
            return _nhl_rows()
        if sport == 'MLB':
            return _mlb_rows()
        return _espn_rows(sport)

    out = _cached(f'standings_{sport}', _load)
    if out:
        out['groups'] = sorted({r['group'] for r in out['rows'] if r['group']})
    return out
