"""Per-game live feed for Game Center's Gamecast / Box score / Play-by-play
tabs, the MLB "At the plate" card and the win-probability chart — one
normalized shape for all six sports, served as JSON by
/api/game/<sport>/<id>/feed (app.py) so the page can poll it while live.

    {
      'plays':    [{period, clock, text, scoring, score, team}]   newest first
      'boxscore': [{abbr, groups: [{label, cols, rows: [[name, ...]]}]}]  away, home
      'winprob':  {points: [home win prob 0-1, ...], marks: [{i, label}]} | None
      'at_bat':   MLB live only — batter/pitcher, count, pitches | None
    }

Sources (all free, no keys): ESPN site summary?event= (NBA/WNBA/NFL/CFB —
plays, boxscore and winprobability in one response), NHL api-web
gamecenter play-by-play + boxscore, MLB statsapi v1.1 feed/live (plays,
boxscore, current at-bat) + v1 winProbability.

Win probability here is display only — the live-WP angle was already
investigated for betting (no edge, see the MLB PBP backfill notes) and
nothing here feeds a model.
"""
import time

import requests

_TTL = 20            # seconds — short enough to feel live on a 30s poll
_cache = {}
_ESPN = 'https://site.api.espn.com/apis/site/v2/sports/{path}/summary'
_ESPN_PATHS = {'NBA': 'basketball/nba', 'WNBA': 'basketball/wnba',
               'NFL': 'football/nfl', 'CFB': 'football/college-football'}
_NHL = 'https://api-web.nhle.com/v1/gamecenter/{id}/{kind}'
_MLB = 'https://statsapi.mlb.com/api'
_MAX_PLAYS = 300


def _get_json(url, params=None):
    key = (url, tuple(sorted((params or {}).items())))
    hit = _cache.get(key)
    if hit and time.time() - hit[1] < _TTL:
        return hit[0]
    try:
        r = requests.get(url, params=params or {}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return hit[0] if hit else None
    _cache[key] = (data, time.time())
    return data


def _score(a, h):
    return f"{a}–{h}" if a is not None and h is not None else None


# ── ESPN (NBA / WNBA / NFL / CFB) ───────────────────────────────────────────

def _espn_period_label(sport, n):
    if not n:
        return ''
    reg = 4
    if n <= reg:
        return f"Q{n}" if sport in ('NBA', 'WNBA', 'NFL', 'CFB') else str(n)
    return 'OT' if n == reg + 1 else f"{n - reg}OT"


def _espn_plays(sport, s):
    if sport in ('NFL', 'CFB'):
        drives = (s.get('drives') or {})
        raw = [p for d in (drives.get('previous') or []) for p in d.get('plays', [])]
        cur = drives.get('current') or {}
        raw += cur.get('plays', []) if cur else []
    else:
        raw = s.get('plays') or []
    out = []
    for p in raw:
        text = (p.get('text') or '').strip()
        if not text:
            continue
        team = ((p.get('team') or {}).get('abbreviation')) if isinstance(p.get('team'), dict) else None
        out.append({'id': str(p.get('id', '')),
                    'period': _espn_period_label(sport, (p.get('period') or {}).get('number')),
                    'period_n': (p.get('period') or {}).get('number'),
                    'clock': (p.get('clock') or {}).get('displayValue', ''),
                    'text': text, 'scoring': bool(p.get('scoringPlay')),
                    'score': _score(p.get('awayScore'), p.get('homeScore')), 'team': team})
    return out


def _espn_boxscore(s):
    teams = []
    for t in ((s.get('boxscore') or {}).get('players') or []):
        groups = []
        for st in t.get('statistics', []):
            rows = []
            for a in st.get('athletes', []):
                if a.get('didNotPlay') or not a.get('stats'):
                    continue
                rows.append([(a.get('athlete') or {}).get('shortName') or (a.get('athlete') or {}).get('displayName', '')]
                            + a['stats'])
            if rows:
                groups.append({'label': (st.get('name') or st.get('text') or 'Players').title(),
                               'cols': ['Player'] + (st.get('labels') or []), 'rows': rows})
        teams.append({'abbr': (t.get('team') or {}).get('abbreviation', ''),
                      'side': t.get('homeAway') or '', 'groups': groups})
    # ESPN lists away first in practice, but say so explicitly.
    teams.sort(key=lambda t: 0 if t['side'] == 'away' else 1)
    return teams


def _espn_winprob(sport, s, plays):
    wp = s.get('winprobability') or []
    if not wp:
        return None
    period_by_id = {p['id']: p['period_n'] for p in plays}
    points, marks, last = [], [], None
    for i, w in enumerate(wp):
        points.append(round(w.get('homeWinPercentage', 0.5), 4))
        n = period_by_id.get(str(w.get('playId')))
        if n and (last is None or n > last):
            marks.append({'i': i, 'label': _espn_period_label(sport, n)})
            last = n
    return {'points': points, 'marks': marks}


def _espn_feed(sport, game_id):
    s = _get_json(_ESPN.format(path=_ESPN_PATHS[sport]), {'event': game_id})
    if not s:
        return None
    plays = _espn_plays(sport, s)
    return {'plays': list(reversed(plays))[:_MAX_PLAYS], 'boxscore': _espn_boxscore(s),
            'winprob': _espn_winprob(sport, s, plays), 'at_bat': None}


# ── NHL ─────────────────────────────────────────────────────────────────────

_NHL_SHOWN = {'goal', 'penalty', 'shot-on-goal', 'period-start', 'period-end', 'game-end'}


def _nhl_period(pd):
    n, kind = pd.get('number'), pd.get('periodType', 'REG')
    if kind == 'SO':
        return 'SO'
    if kind == 'OT':
        return 'OT' if n == 4 else f"{n - 3}OT"
    return f"P{n}"


def _nhl_feed(game_id):
    pbp = _get_json(_NHL.format(id=game_id, kind='play-by-play'))
    box = _get_json(_NHL.format(id=game_id, kind='boxscore'))
    if not pbp and not box:
        return None
    names, team_abbr = {}, {}
    for r in (pbp or {}).get('rosterSpots', []):
        names[r['playerId']] = f"{(r.get('firstName') or {}).get('default', '')[:1]}. {(r.get('lastName') or {}).get('default', '')}"
    for side in ('awayTeam', 'homeTeam'):
        t = (pbp or box or {}).get(side) or {}
        team_abbr[t.get('id')] = t.get('abbrev', '')
    plays = []
    for p in (pbp or {}).get('plays', []):
        kind = p.get('typeDescKey')
        if kind not in _NHL_SHOWN:
            continue
        d = p.get('details') or {}
        team = team_abbr.get(d.get('eventOwnerTeamId'))
        if kind == 'goal':
            assists = [names.get(d.get(k)) for k in ('assist1PlayerId', 'assist2PlayerId') if d.get(k)]
            text = f"Goal — {names.get(d.get('scoringPlayerId'), 'Unknown')}" + (
                f" ({', '.join(a for a in assists if a)})" if assists else ' (unassisted)')
            if d.get('shotType'):
                text += f", {d['shotType']} shot"
        elif kind == 'penalty':
            who = names.get(d.get('committedByPlayerId'), team or '')
            text = f"Penalty — {who}: {(d.get('descKey') or '').replace('-', ' ')} ({d.get('duration', 2)} min)"
        elif kind == 'shot-on-goal':
            text = f"Shot on goal — {names.get(d.get('shootingPlayerId'), team or '')}"
        else:
            text = kind.replace('-', ' ').capitalize()
        plays.append({'period': _nhl_period(p.get('periodDescriptor') or {}), 'clock': p.get('timeInPeriod', ''),
                      'text': text, 'scoring': kind == 'goal',
                      'score': _score(d.get('awayScore'), d.get('homeScore')) if kind == 'goal' else None,
                      'team': team})
    teams = []
    pg = (box or {}).get('playerByGameStats') or {}
    for side in ('awayTeam', 'homeTeam'):
        stats = pg.get(side) or {}
        skaters = (stats.get('forwards') or []) + (stats.get('defense') or [])
        groups = []
        if skaters:
            groups.append({'label': 'Skaters', 'cols': ['Player', 'G', 'A', 'P', '+/-', 'SOG', 'HIT', 'TOI'],
                           'rows': [[(s.get('name') or {}).get('default', ''), s.get('goals', 0), s.get('assists', 0),
                                     s.get('points', 0), s.get('plusMinus', 0), s.get('sog', 0), s.get('hits', 0),
                                     s.get('toi', '')] for s in skaters]})
        goalies = [g for g in (stats.get('goalies') or []) if g.get('toi') not in (None, '00:00')]
        if goalies:
            groups.append({'label': 'Goalies', 'cols': ['Player', 'SA', 'GA', 'SV%', 'TOI'],
                           'rows': [[(g.get('name') or {}).get('default', ''), g.get('shotsAgainst', 0),
                                     g.get('goalsAgainst', 0),
                                     f"{g['savePctg']:.3f}".lstrip('0') if g.get('savePctg') is not None else '—',
                                     g.get('toi', '')] for g in goalies]})
        teams.append({'abbr': ((box or pbp or {}).get(side) or {}).get('abbrev', ''), 'side': side[:4], 'groups': groups})
    return {'plays': list(reversed(plays))[:_MAX_PLAYS], 'boxscore': teams, 'winprob': None, 'at_bat': None}


# ── MLB ─────────────────────────────────────────────────────────────────────

def _mlb_boxscore(ld, game_teams):
    teams = []
    for side in ('away', 'home'):
        t = ((ld.get('boxscore') or {}).get('teams') or {}).get(side) or {}
        players = t.get('players') or {}
        bat, pit = [], []
        for pid in t.get('batters', []):
            p = players.get(f'ID{pid}') or {}
            b = (p.get('stats') or {}).get('batting') or {}
            if not b or not b.get('plateAppearances') and not b.get('atBats'):
                continue
            bat.append([(p.get('person') or {}).get('fullName', ''), b.get('atBats', 0), b.get('runs', 0),
                        b.get('hits', 0), b.get('rbi', 0), b.get('baseOnBalls', 0), b.get('strikeOuts', 0)])
        for pid in t.get('pitchers', []):
            p = players.get(f'ID{pid}') or {}
            s = (p.get('stats') or {}).get('pitching') or {}
            if not s:
                continue
            pit.append([(p.get('person') or {}).get('fullName', ''), s.get('inningsPitched', '0.0'), s.get('hits', 0),
                        s.get('runs', 0), s.get('earnedRuns', 0), s.get('baseOnBalls', 0), s.get('strikeOuts', 0),
                        s.get('numberOfPitches', 0)])
        groups = []
        if bat:
            groups.append({'label': 'Batting', 'cols': ['Batter', 'AB', 'R', 'H', 'RBI', 'BB', 'K'], 'rows': bat})
        if pit:
            groups.append({'label': 'Pitching', 'cols': ['Pitcher', 'IP', 'H', 'R', 'ER', 'BB', 'K', 'P'], 'rows': pit})
        teams.append({'abbr': ((game_teams.get(side) or {}).get('abbreviation') or ''), 'side': side, 'groups': groups})
    return teams


def _mlb_at_bat(feed):
    """Current plate appearance — only while the game is Live."""
    if ((feed.get('gameData') or {}).get('status') or {}).get('abstractGameState') != 'Live':
        return None
    ld = feed.get('liveData') or {}
    cp = (ld.get('plays') or {}).get('currentPlay') or {}
    m = cp.get('matchup') or {}
    if not m.get('batter'):
        return None
    half = (cp.get('about') or {}).get('halfInning')
    bat_side, pit_side = ('away', 'home') if half == 'top' else ('home', 'away')
    teams = ((ld.get('boxscore') or {}).get('teams') or {})
    game_teams = (feed.get('gameData') or {}).get('teams') or {}

    def _line(side, pid, kind):
        p = ((teams.get(side) or {}).get('players') or {}).get(f"ID{pid}") or {}
        return ((p.get('stats') or {}).get(kind) or {}).get('summary', '')

    pitches, sz_top, sz_bot = [], None, None
    for e in cp.get('playEvents', []):
        if not e.get('isPitch'):
            continue
        pd, det = e.get('pitchData') or {}, e.get('details') or {}
        c = pd.get('coordinates') or {}
        sz_top, sz_bot = pd.get('strikeZoneTop') or sz_top, pd.get('strikeZoneBottom') or sz_bot
        pitches.append({'n': e.get('pitchNumber'), 'call': (det.get('call') or {}).get('description', ''),
                        'is_strike': bool(det.get('isStrike')), 'in_play': bool(det.get('isInPlay')),
                        'type': (det.get('type') or {}).get('description', ''),
                        'mph': pd.get('startSpeed'), 'px': c.get('pX'), 'pz': c.get('pZ')})
    abbr = lambda side: ((game_teams.get(side) or {}).get('abbreviation') or '')
    return {'batter': m['batter'].get('fullName'), 'batter_team': abbr(bat_side),
            'batter_line': _line(bat_side, m['batter'].get('id'), 'batting'),
            'pitcher': (m.get('pitcher') or {}).get('fullName'), 'pitcher_team': abbr(pit_side),
            'pitcher_line': _line(pit_side, (m.get('pitcher') or {}).get('id'), 'pitching'),
            'count': cp.get('count') or {}, 'pitches': list(reversed(pitches)),
            'sz_top': sz_top or 3.4, 'sz_bot': sz_bot or 1.6}


def _mlb_feed(game_pk):
    feed = _get_json(f'{_MLB}/v1.1/game/{game_pk}/feed/live')
    if not feed:
        return None
    ld = feed.get('liveData') or {}
    plays = []
    for p in (ld.get('plays') or {}).get('allPlays', []):
        r, a = p.get('result') or {}, p.get('about') or {}
        if not r.get('description'):
            continue
        plays.append({'period': f"{'Top' if a.get('halfInning') == 'top' else 'Bot'} {a.get('inning')}",
                      'clock': '', 'text': r['description'], 'scoring': bool(a.get('isScoringPlay')),
                      'score': _score(r.get('awayScore'), r.get('homeScore')), 'team': None})
    winprob = None
    wp = _get_json(f'{_MLB}/v1/game/{game_pk}/winProbability')
    if isinstance(wp, list) and wp:
        points, marks, last = [], [], None
        for i, w in enumerate(wp):
            if w.get('homeTeamWinProbability') is None:
                continue
            points.append(round(w['homeTeamWinProbability'] / 100, 4))
            inn = (w.get('about') or {}).get('inning')
            if inn and (last is None or inn > last):
                marks.append({'i': len(points) - 1, 'label': str(inn)})
                last = inn
        winprob = {'points': points, 'marks': marks} if points else None
    game_teams = (feed.get('gameData') or {}).get('teams') or {}
    return {'plays': list(reversed(plays))[:_MAX_PLAYS], 'boxscore': _mlb_boxscore(ld, game_teams),
            'winprob': winprob, 'at_bat': _mlb_at_bat(feed)}


def get_live_winprob(sport, game_id):
    """Latest home win probability (0-1) for one game, or None — the light
    path for the Tonight hub's featured card: MLB hits only the small
    winProbability endpoint, never the ~1MB feed/live."""
    sport = (sport or '').upper()
    try:
        if sport == 'MLB':
            wp = _get_json(f'{_MLB}/v1/game/{game_id}/winProbability')
            pts = [w['homeTeamWinProbability'] for w in (wp or []) if w.get('homeTeamWinProbability') is not None]
            return round(pts[-1] / 100, 4) if pts else None
        if sport in _ESPN_PATHS:
            s = _get_json(_ESPN.format(path=_ESPN_PATHS[sport]), {'event': game_id})
            wp = (s or {}).get('winprobability') or []
            return wp[-1].get('homeWinPercentage') if wp else None
    except Exception:
        return None
    return None


def get_feed(sport, game_id):
    """Normalized feed for one game, or None if the source is unreachable.
    Never raises — a shape drift degrades to empty sections, not a 500."""
    sport = (sport or '').upper()
    try:
        if sport == 'MLB':
            return _mlb_feed(game_id)
        if sport == 'NHL':
            return _nhl_feed(game_id)
        if sport in _ESPN_PATHS:
            return _espn_feed(sport, game_id)
    except Exception:
        return {'plays': [], 'boxscore': [], 'winprob': None, 'at_bat': None}
    return None
