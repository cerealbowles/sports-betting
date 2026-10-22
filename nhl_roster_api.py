"""
nhl_roster_api.py — Team roster data layer for NHL, sourced from ESPN's
free `teams/{id}/roster` endpoint (same provider every other module in
this app uses). Built purely for nhl_news.py's player-name matching (see
news_ingest.build_indexes()) — no position/availability filtering like
nfl_roster_api.py's, since nothing here needs it yet.

Same grouped-by-position roster shape as nfl_roster_api.py's endpoint
(athletes: [{position, items: [...]}]), confirmed live, not the flat
athletes: [...] shape nba_roster_api.py/wnba_roster_api.py use.

get_player_season_summary() below also hits ESPN's `athletes/{id}/gamelog`
endpoint (same family nba_roster_api.py uses for its own gamelog) — gives
the game_detail injuries panel a real season stat line (G/A/P for skaters,
record/SV%/GAA for goalies) for each injured player, so an unfamiliar name
like "Ryan Poehling" reads as "depth forward" vs "Troy Terry" reading as
"leading scorer" instead of both just being an opaque status pill.
"""
import time
import requests

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl'
ESPN_WEB  = 'https://site.web.api.espn.com/apis/common/v3/sports/hockey/nhl'

_cache = {}
_TTL = 24 * 3600
_fail_ts = {}
_FAIL_BACKOFF = 300


def _get(url, params, key):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
            return data
    if now - _fail_ts.get(key, 0) < _FAIL_BACKOFF:
        return _cache.get(key, (None, 0))[0]
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        _fail_ts[key] = now
        return _cache.get(key, (None, 0))[0]
    _cache[key] = (data, now)
    return data


def get_roster(team_id):
    """Returns [{id, name}] for team_id's current roster, or [] on failure."""
    data = _get(f'{ESPN_SITE}/teams/{team_id}/roster', None, f'roster_{team_id}')
    if not data:
        return []
    out = []
    for group in data.get('athletes', []):
        for item in group.get('items', []):
            pid = item.get('id')
            name = item.get('displayName') or item.get('fullName')
            if pid and name:
                out.append({'id': pid, 'name': name})
    return out


def _toi_to_seconds(mmss):
    try:
        m, s = str(mmss).split(':')
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return 0


def _seconds_to_toi(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f'{m}:{s:02d}'


def _season_totals(names, events, is_goalie):
    """Sums a season's per-game `events` (ESPN gamelog rows) into a single
    stat line, keyed off `names` (ESPN's own stat-name order for this
    endpoint — not assumed fixed, since the position-dependent key set
    differs between skaters and goalies)."""
    idx = {n: i for i, n in enumerate(names)}

    def num(ev, key):
        i = idx.get(key)
        stats = ev.get('stats') or []
        if i is None or i >= len(stats):
            return 0.0
        try:
            return float(stats[i])
        except (ValueError, TypeError):
            return 0.0

    games = len(events)
    toi_i = idx.get('timeOnIcePerGame')
    toi_total = sum(_toi_to_seconds(ev['stats'][toi_i])
                     for ev in events if toi_i is not None and toi_i < len(ev.get('stats') or []))
    toi_avg = _seconds_to_toi(toi_total / games) if games else '0:00'

    if is_goalie:
        shots_against = sum(num(ev, 'shotsAgainst') for ev in events)
        goals_against = sum(num(ev, 'goalsAgainst') for ev in events)
        saves = sum(num(ev, 'saves') for ev in events)
        return {
            'is_goalie': True, 'games': games, 'starts': int(sum(num(ev, 'gameStarted') for ev in events)),
            'wins': int(sum(num(ev, 'wins') for ev in events)),
            'losses': int(sum(num(ev, 'losses') for ev in events)),
            'otl': int(sum(num(ev, 'overtimeLosses') for ev in events)),
            'saves': int(saves), 'goals_against': int(goals_against),
            'shutouts': int(sum(num(ev, 'shutouts') for ev in events)),
            'save_pct': round(saves / shots_against, 3) if shots_against else None,
            'gaa': round(goals_against / games, 2) if games else None,
            'toi_per_game': toi_avg,
        }

    return {
        'is_goalie': False, 'games': games, 'goals': int(sum(num(ev, 'goals') for ev in events)),
        'assists': int(sum(num(ev, 'assists') for ev in events)),
        'points': int(sum(num(ev, 'points') for ev in events)),
        'plus_minus': int(sum(num(ev, 'plusMinus') for ev in events)),
        'pim': int(sum(num(ev, 'penaltyMinutes') for ev in events)),
        'shots': int(sum(num(ev, 'shotsTotal') for ev in events)),
        'toi_per_game': toi_avg,
    }


def get_player_season_summary(player_id, position):
    """Returns this player's aggregated stat line from ESPN's gamelog —
    whichever season (newest first) has at least one logged game. Usually
    the current season, but falls back to last season for a player hurt
    before this one started (an IR player with 0 games this year still
    needs a stat line to show what the team is missing). None on failure
    or a player with no gamelog at all (e.g. a rookie yet to debut).

    Skaters: {games, goals, assists, points, plus_minus, pim, shots,
    toi_per_game, label, is_current}.
    Goalies: {games, starts, wins, losses, otl, saves, goals_against,
    shutouts, save_pct, gaa, toi_per_game, label, is_current}.
    """
    data = _get(f'{ESPN_WEB}/athletes/{player_id}/gamelog', None, f'gamelog_{player_id}')
    if not data:
        return None
    names = data.get('names', [])
    regular = [st for st in data.get('seasonTypes', [])
               if 'Regular Season' in (st.get('displayName') or '')]
    is_goalie = position == 'G'

    # "Current" means this literal season, by year — not just whichever
    # entry sorts first. A player hurt before this season started (0 games
    # logged yet) still has their *last* season as the first entry with any
    # games, and that's last season, not this one, even though it's first
    # in the list.
    import nhl_api
    start_year = nhl_api._get_nhl_season_start_year()
    current_label_year = f'{start_year}-{str(start_year + 1)[-2:]}'

    for st in regular:
        events = [ev for cat in st.get('categories', []) for ev in cat.get('events', [])]
        if not events:
            continue
        totals = _season_totals(names, events, is_goalie)
        label = st.get('displayName', '')
        totals['label'] = label
        totals['is_current'] = label.startswith(current_label_year)
        return totals
    return None
