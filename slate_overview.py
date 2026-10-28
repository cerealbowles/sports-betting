"""
Slate overview for a schedule page: a US map with a dot at every game's
venue, plus a short "Tonight at a glance" list of facts about the day's games.

Everything here is derived from fields the schedule context already carries
(records, last-5 form, rest days, ppg/ppg allowed, game times) plus the arena
coordinates in team_venues.py — no new API calls. Every fact rule returns
nothing when nothing qualifies, so a quiet night shows a shorter list rather
than filler.

The map image is static/us-lower48.svg, drawn with d3-geo's default
geoAlbers() projection (standard parallels 29.5°/45.5°, rotate -96°, center
-0.6°/38.7°, scale 1070, translate 480/250). project() below is the same
projection in plain Python, so a venue's lat/lon lands on the same spot of
that image. If the SVG is ever regenerated with different settings, update
MAP_VIEWBOX and the constants below to match.

Venue = the home team's arena. Neutral-site games (NBA Cup in Las Vegas,
international games) will show at the home team's arena instead.
"""
import math
from datetime import datetime

import team_venues

# viewBox of static/us-lower48.svg: (min_x, min_y, width, height)
MAP_VIEWBOX = (81, -3, 797, 513)

_PHI0, _PHI1 = math.radians(29.5), math.radians(45.5)
_N = (math.sin(_PHI0) + math.sin(_PHI1)) / 2
_C = 1 + math.sin(_PHI0) * (2 * _N - math.sin(_PHI0))
_R0 = math.sqrt(_C) / _N
_SCALE, _TX, _TY, _ROTATE = 1070, 480, 250, 96


def _conic_raw(lam, phi):
    r = math.sqrt(_C - 2 * _N * math.sin(phi)) / _N
    return r * math.sin(lam * _N), _R0 - r * math.cos(lam * _N)


_CX, _CY = _conic_raw(math.radians(-0.6), math.radians(38.7))


def project(lat, lon):
    """(x, y) in the SVG's coordinate space for a lat/lon (lower 48 + Toronto)."""
    x, y = _conic_raw(math.radians(lon + _ROTATE), math.radians(lat))
    return _TX + _SCALE * (x - _CX), _TY - _SCALE * (y - _CY)


def _to_pct(x, y):
    mx, my, w, h = MAP_VIEWBOX
    return (x - mx) / w * 100, (y - my) / h * 100


# ── helpers ────────────────────────────────────────────────────────────────

def _abbr(team):
    return team.get('abbrev') or team.get('abbr') or ''


def _record(team):
    w, l = team.get('wins'), team.get('losses')
    if w is None or l is None:
        return ''
    return f"{w}-{l}"


def _games_played(team):
    return (team.get('wins') or 0) + (team.get('losses') or 0)


def _rest_days(team):
    """team['rest_days'] everywhere except MLB, which nests it under
    schedule_fatigue instead of exposing it on the team dict directly."""
    rd = team.get('rest_days')
    if rd is None:
        rd = (team.get('schedule_fatigue') or {}).get('rest_days')
    return rd


def _game_id(g):
    """g['game_id'] everywhere except MLB, which calls its primary id
    game_pk instead."""
    return g.get('game_id') or g.get('game_pk')


def _win_pct(team):
    gp = _games_played(team)
    return (team.get('wins') or 0) / gp if gp else None


def _streak(form):
    """('W'|'L', n) for the current streak from a newest-first form list."""
    if not form:
        return None, 0
    first, n = form[0], 0
    for r in form:
        if r != first:
            break
        n += 1
    return first, n


def _matchup(g):
    return f"{_abbr(g['away'])} @ {_abbr(g['home'])}"


def _parse_utc(s):
    try:
        return datetime.fromisoformat((s or '').replace('Z', '+00:00'))
    except ValueError:
        return None


# ── fact rules ─────────────────────────────────────────────────────────────
# Each returns a fact dict or None:
#   {'label': 'GAME OF THE NIGHT', 'title': 'POR @ LAL', 'detail': '...'}
# The tip-off window fact also carries 'utc_first'/'utc_last' so the page can
# render those times in the viewer's own time zone.

def _fact_game_of_night(games, used):
    best = None
    for g in games:
        a, h = g['away'], g['home']
        pa, ph = _win_pct(a), _win_pct(h)
        if pa is None or ph is None or _games_played(a) + _games_played(h) < 3:
            continue
        score = ((pa + ph) / 2, (a.get('wins') or 0) + (h.get('wins') or 0))
        if score[0] >= 0.6 and (best is None or score > best[0]):
            best = (score, g)
    if not best:
        return None
    g = best[1]
    used.add(g.get('game_id'))
    a, h = g['away'], g['home']
    w = (a.get('wins') or 0) + (h.get('wins') or 0)
    l = (a.get('losses') or 0) + (h.get('losses') or 0)
    note = 'Two unbeatens' if not a.get('losses') and not h.get('losses') else 'Best combined record'
    return {'label': 'GAME OF THE NIGHT', 'title': _matchup(g),
            'detail': f"{note} · {w}-{l} combined"}


def _fact_mismatch(games, used):
    best = None
    for g in games:
        if g.get('game_id') in used:
            continue
        a, h = g['away'], g['home']
        if _games_played(a) < 2 or _games_played(h) < 2:
            continue
        gap = abs(_win_pct(a) - _win_pct(h))
        if gap >= 0.4 and (best is None or gap > best[0]):
            best = (gap, g)
    if not best:
        return None
    g = best[1]
    used.add(g.get('game_id'))
    a, h = g['away'], g['home']
    fav, dog = (a, h) if _win_pct(a) > _win_pct(h) else (h, a)
    verb = 'visits' if fav is a else 'hosts'
    return {'label': 'BIGGEST MISMATCH ON PAPER', 'title': _matchup(g),
            'detail': f"{_abbr(fav)} ({_record(fav)}) {verb} {_abbr(dog)} ({_record(dog)})"}


def _all_teams(games):
    for g in games:
        yield g['away']
        yield g['home']


def _fact_perfect(games, listed):
    teams = [t for t in _all_teams(games) if not t.get('losses') and (t.get('wins') or 0) >= 2]
    if not teams:
        return None
    teams.sort(key=lambda t: -(t.get('wins') or 0))
    listed.update(_abbr(t) for t in teams)
    return {'label': 'STILL PERFECT',
            'title': ' · '.join(f"{_abbr(t)} {_record(t)}" for t in teams[:4]), 'detail': ''}


def _fact_winless(games, listed):
    teams = [t for t in _all_teams(games) if not t.get('wins') and (t.get('losses') or 0) >= 2]
    if not teams:
        return None
    teams.sort(key=lambda t: -(t.get('losses') or 0))
    listed.update(_abbr(t) for t in teams)
    return {'label': 'STILL WINLESS',
            'title': ' · '.join(f"{_abbr(t)} {_record(t)}" for t in teams[:4]), 'detail': ''}


def _fact_streaks(games, listed):
    rows = []
    for t in _all_teams(games):
        if _abbr(t) in listed:
            continue
        kind, n = _streak(t.get('form') or [])
        if n >= 3:
            rows.append((n, kind, t))
    if not rows:
        return None
    rows.sort(key=lambda r: -r[0])
    return {'label': 'STREAKING',
            'title': ' · '.join(f"{_abbr(t)} {kind}{n}" for n, kind, t in rows[:4]),
            'detail': 'Current streaks, last 5 games'}


def _fact_back_to_back(games):
    tired = []
    for g in games:
        for side, other in (('away', 'home'), ('home', 'away')):
            t, o = g[side], g[other]
            if _rest_days(t) == 1:
                tired.append((t, o))
    if not tired:
        return None
    if len(tired) == 1:
        t, o = tired[0]
        detail = '2nd night of a back-to-back'
        o_rest = _rest_days(o)
        if (o_rest or 0) >= 2:
            detail += f" · {_abbr(o)} is rested ({o_rest} days)"
        return {'label': 'BACK-TO-BACK', 'title': _abbr(t), 'detail': detail}
    return {'label': 'BACK-TO-BACKS',
            'title': ' · '.join(_abbr(t) for t, _ in tired[:5]),
            'detail': 'Playing the 2nd night of a back-to-back'}


def _fact_longest_trip(games, sport):
    best = None
    for g in games:
        a, h = _abbr(g['away']), _abbr(g['home'])
        miles = team_venues.travel_miles(a, h, sport)
        if miles is not None and (best is None or miles > best[0]):
            best = (miles, g)
    if not best or best[0] < 1000:
        return None
    miles, g = best
    a, h = _abbr(g['away']), _abbr(g['home'])
    home = team_venues.get(h, sport)
    detail = f"{miles:,} mi from home"
    when = _parse_utc(g.get('game_time_utc'))
    if when:
        tz = team_venues.timezone_change_hours(a, h, when, sport)
        if tz:
            detail += f" · {abs(tz)} time zone{'s' if abs(tz) != 1 else ''} {'east' if tz > 0 else 'west'}"
    return {'label': 'LONGEST TRIP', 'title': f"{a} at {home['city'].split(',')[0]}", 'detail': detail}


def _fact_high_scoring(games):
    best = None
    for g in games:
        a, h = g['away'], g['home']
        if (a.get('games_played') or 0) < 3 or (h.get('games_played') or 0) < 3:
            continue
        vals = [a.get('ppg'), a.get('ppg_allowed'), h.get('ppg'), h.get('ppg_allowed')]
        if any(v is None for v in vals):
            continue
        total = (h['ppg'] + a['ppg_allowed']) / 2 + (a['ppg'] + h['ppg_allowed']) / 2
        if best is None or total > best[0]:
            best = (total, g)
    if not best:
        return None
    total, g = best
    return {'label': 'HIGHEST-SCORING MATCHUP', 'title': _matchup(g),
            'detail': f"~{total:.0f} combined pts on season scoring averages"}


def _fact_window(games, sport):
    timed = [(g, _parse_utc(g.get('game_time_utc'))) for g in games if g.get('status') not in ('Final',)]
    timed = [(g, t) for g, t in timed if t]
    if len(timed) < 2:
        return None
    timed.sort(key=lambda gt: gt[1])
    first, last = timed[0][0], timed[-1][0]
    if timed[0][1] == timed[-1][1]:
        return None

    def city(g):
        v = team_venues.get(_abbr(g['home']), sport)
        return v['city'].split(',')[0] if v else _abbr(g['home'])

    return {'label': 'TIP-OFF WINDOW', 'title': '', 'kind': 'window',
            'utc_first': first.get('game_time_utc'), 'utc_last': last.get('game_time_utc'),
            'detail': f"{city(first)} tips first, {city(last)} last"}


MAX_FACTS = 6


def build_facts(games, sport='nba'):
    used, listed = set(), set()
    facts = [
        _fact_game_of_night(games, used),
        _fact_mismatch(games, used),
        _fact_perfect(games, listed),
        _fact_winless(games, listed),
        _fact_streaks(games, listed),
        _fact_back_to_back(games),
        _fact_longest_trip(games, sport),
        _fact_high_scoring(games),
    ]
    facts = [f for f in facts if f][:MAX_FACTS - 1]
    window = _fact_window(games, sport)
    if window:
        facts.append(window)
    return facts


# ── markers ────────────────────────────────────────────────────────────────

def _team_view(team):
    return {'abbr': _abbr(team), 'name': team.get('name', ''), 'record': _record(team),
            'logo_url': team.get('logo_url'), 'score': team.get('score')}


def build_markers(games, sport='nba'):
    markers = []
    for g in games:
        venue = team_venues.get(_abbr(g['home']), sport)
        if not venue:
            continue
        x, y = _to_pct(*project(venue['lat'], venue['lon']))
        # Two arenas a few miles apart (LAL/LAC, NY/BKN) would stack exactly;
        # nudge the later one down so both stay clickable.
        while any(abs(m['_x'] - x) < 1.2 and abs(m['_y'] - y) < 1.8 for m in markers):
            y += 2.4
        markers.append({
            '_x': x, '_y': y,
            'id': str(_game_id(g)),
            'game_key': g.get('game_key'),
            'href': f"/{sport}/game/{_game_id(g)}",
            'status': g.get('status'),
            'utc': g.get('game_time_utc'),
            'tz': venue['tz'],
            'venue': g.get('venue') or venue['city'],
            'away': _team_view(g['away']),
            'home': _team_view(g['home']),
        })
    for m in markers:
        crowded_right = any(0 < o['_x'] - m['_x'] < 8 and abs(o['_y'] - m['_y']) < 4 for o in markers)
        m['label_side'] = 'l' if m['_x'] > 82 or crowded_right else 'r'
    for m in markers:
        m['x'], m['y'] = round(m.pop('_x'), 2), round(m.pop('_y'), 2)
        m['pop_h'] = 'right' if m['x'] > 55 else 'left'
        m['pop_v'] = 'bottom' if m['y'] > 50 else 'top'
    return markers


def build(games, sport='nba'):
    """Context for templates/_slate_overview.html, or None when no game can be
    placed on the map (empty slate, or abbreviations team_venues doesn't know)."""
    games = [g for g in (games or []) if g.get('away') and g.get('home')]
    markers = build_markers(games, sport)
    if not markers:
        return None
    return {
        'markers': markers,
        'facts': build_facts(games, sport),
        'vb_w': MAP_VIEWBOX[2],
        'vb_h': MAP_VIEWBOX[3],
    }
