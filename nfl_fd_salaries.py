"""
nfl_fd_salaries.py — real FanDuel prices from FanDuel's own "players list"
CSV export (contest lobby -> Download players list), for the Lab's value
column and for building lineups at real prices instead of nfl_fantasy's
synthetic ones.

Why uploads rather than a feed: FanDuel has no public salary API, the free
historical source (RotoGuru) stops at 2021, and this app's warehouse starts
in 2024. One download per contest is the only reliable source, so each
upload is stored and becomes backtest data too (see nfl_value_backtest.py).

CSV columns used: Id, Position, Nickname, FPPG, Salary, Game, Team,
Roster Position. The slate id is the part of Id before the dash (FanDuel
uses one id per contest slate, e.g. '135326-62239' -> '135326').
"""
import csv
import io
import re

FD_POSITION = {'QB': 'QB', 'RB': 'RB', 'WR': 'WR', 'TE': 'TE', 'D': 'DST', 'DST': 'DST'}
# FanDuel abbreviations that differ from ESPN's (what the rest of the app uses).
FD_TEAM_ALIAS = {'JAC': 'JAX', 'WAS': 'WSH', 'LA': 'LAR'}


def _team(abbr):
    abbr = (abbr or '').upper()
    return FD_TEAM_ALIAS.get(abbr, abbr)


def norm_name(name):
    """Lowercase letters only, suffixes (Jr/Sr/II/III/IV) dropped, so
    'Luther Burden III' (ESPN) and 'Luther Burden' (FanDuel) match."""
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', (name or '').lower())
    return re.sub(r'[^a-z]', '', s)


def parse_players_list(text):
    """Returns (slate_id, [row, ...]) from a FanDuel players-list CSV.
    Each row: {'fd_id', 'name', 'norm', 'position', 'team', 'game',
    'salary', 'fppg', 'roster_position'}. Kickers and unknown positions are
    skipped (this app doesn't model kickers). A player listed more than once
    (single-game files can list an MVP variant) keeps the cheapest row,
    which is the base/FLEX price. Raises ValueError if it isn't a FanDuel
    players list."""
    text = (text or '').lstrip('﻿')
    reader = csv.DictReader(io.StringIO(text))
    fields = {(f or '').strip() for f in (reader.fieldnames or [])}
    if not {'Id', 'Position', 'Nickname', 'Salary', 'Team', 'Game'} <= fields:
        raise ValueError("That doesn't look like a FanDuel players list (needs Id, Position, Nickname, Salary, Team, Game).")

    by_key = {}
    slate_ids = set()
    for r in reader:
        r = {(k or '').strip(): (v or '').strip() for k, v in r.items() if k}
        pos = FD_POSITION.get(r.get('Position', '').upper())
        if not pos:
            continue
        try:
            salary = int(float(r['Salary']))
        except (KeyError, ValueError):
            continue
        fd_id = r.get('Id', '')
        slate_ids.add(fd_id.split('-')[0])
        try:
            fppg = round(float(r.get('FPPG') or 0), 2)
        except ValueError:
            fppg = None
        row = {
            'fd_id': fd_id, 'name': r.get('Nickname', ''), 'norm': norm_name(r.get('Nickname', '')),
            'position': pos, 'team': _team(r.get('Team')),
            'game': '@'.join(_team(t) for t in r.get('Game', '').split('@')),
            'salary': salary, 'fppg': fppg, 'roster_position': r.get('Roster Position', ''),
            'injury': (r.get('Injury Indicator') or '').upper(),
        }
        key = (row['norm'], row['team'], pos)
        if key not in by_key or salary < by_key[key]['salary']:
            by_key[key] = row
    if not by_key:
        raise ValueError('No QB/RB/WR/TE/DST rows found in that file.')
    slate_id = sorted(slate_ids)[0] if len(slate_ids) == 1 else '+'.join(sorted(slate_ids))
    return slate_id, list(by_key.values())


def game_key(game):
    """'AWAY@HOME' for an nfl_api game dict — FanDuel's Game column format."""
    return f"{game['away']['abbrev']}@{game['home']['abbrev']}".upper()


def pick_slate(slates, selected_games, single):
    """slates: {slate_id: [rows]}. Picks the uploaded slate that prices the
    games on screen: its games must cover every selected game, single-game
    view only considers one-game slates and the standard view only
    multi-game ones (their price scales differ). Smallest covering slate
    wins. Returns slate_id or None."""
    wanted = {game_key(g) for g in selected_games}
    if not wanted:
        return None
    best = None
    for sid, rows in slates.items():
        games = {r['game'] for r in rows}
        if (len(games) == 1) != single or not wanted <= games:
            continue
        if best is None or len(games) < best[0]:
            best = (len(games), sid)
    return best[1] if best else None


# FanDuel injury indicators that mean the player won't play.
FD_OUT = {'O', 'IR', 'NA', 'D', 'SUS', 'PUP'}


def match_pool(pool, rows):
    """Returns {pool_id: fd_row} — by normalized name + team (+ position),
    DST by team alone."""
    by_name = {(r['norm'], r['team']): r for r in rows if r['position'] != 'DST'}
    by_dst = {r['team']: r for r in rows if r['position'] == 'DST'}
    out = {}
    for p in pool:
        team = (p.get('team') or '').upper()
        r = by_dst.get(team) if p['position'] == 'DST' else by_name.get((norm_name(p['name']), team))
        if r:
            out[p['id']] = r
    return out


def spearman(xs, ys):
    """Rank correlation, average ranks for ties. None under 3 points."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(cov / (vx * vy), 3) if vx and vy else None
