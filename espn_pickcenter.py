"""
espn_pickcenter.py — Shared helpers for pulling spread/total (O/U) lines from
ESPN's site API, across all six sports we track. Used by espn_odds_backfill.py
to fill in game_predictions.spread_open/close and total_open/close, which
nothing else currently populates (moneyline is already tracked separately via
odds_api.py/The Odds API).

ESPN only keeps the per-event `pickcenter` odds block for roughly the last
~300 days — a completed game older than that will return no odds data at
all, regardless of how it's queried. There's no way to recover it; this is
a hard ESPN-side retention limit, not a bug here.

No custom User-Agent, matching nfl_bootstrap.py's finding that ESPN's
edge/WAF 403s a self-identifying UA string but accepts requests' own default.
"""
import time
import requests

SPORT_SLUGS = {
    'NFL':  'football/nfl',
    'CFB':  'football/college-football',
    'MLB':  'baseball/mlb',
    'NBA':  'basketball/nba',
    'NHL':  'hockey/nhl',
    'WNBA': 'basketball/wnba',
}

_BASE = 'https://site.api.espn.com/apis/site/v2/sports'
_FBS_GROUP = 80  # matches cfb_api.py — without it ESPN drops most FBS games


def _get(url, params, label, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f'  ! {label} failed: {e}', flush=True)
                return None
            time.sleep(1 + attempt)
    return None


def fetch_scoreboard_events(sport, date_str):
    """date_str is YYYYMMDD. Returns [{event_id, home_name, away_name, state}, ...]
    using the same displayName strings ESPN's own live schedule fetchers use
    (nfl_api.py/cfb_api.py already build game_predictions rows from these, so
    matching is exact-string for those two sports)."""
    slug = SPORT_SLUGS[sport]
    params = {'dates': date_str, 'limit': 400}
    if sport == 'CFB':
        params['groups'] = _FBS_GROUP
    data = _get(f'{_BASE}/{slug}/scoreboard', params, f'{sport} scoreboard {date_str}')
    if not data:
        return []
    out = []
    for event in data.get('events', []):
        comp = event.get('competitions', [{}])[0]
        tmap = {c.get('homeAway'): c for c in comp.get('competitors', [])}
        home = tmap.get('home', {}).get('team', {}).get('displayName', '')
        away = tmap.get('away', {}).get('team', {}).get('displayName', '')
        if not home or not away:
            continue
        out.append({
            'event_id':  event.get('id'),
            'home_name': home,
            'away_name': away,
            'state':     comp.get('status', {}).get('type', {}).get('state', 'pre'),
        })
    return out


def _open_close(side):
    """pointSpread/total sides look like {'open': {'line': '-3'}, 'close': {'line': '-5.5'}}
    or {'open': {'line': 'o52.5'}, 'close': {'line': 'o54.5'}} for totals — strip the
    leading o/u letter totals use to mark over/under."""
    if not side:
        return None, None
    def _num(bucket):
        line = (side.get(bucket) or {}).get('line')
        if line is None:
            return None
        try:
            return float(str(line).lstrip('ou'))
        except (TypeError, ValueError):
            return None
    return _num('open'), _num('close')


def fetch_pickcenter(sport, event_id):
    """Returns {spread_open, spread_close, total_open, total_close} (home-team
    spread; negative = home favored) or None if ESPN has no odds for this
    event (outside the retention window, or never had a market)."""
    slug = SPORT_SLUGS[sport]
    data = _get(f'{_BASE}/{slug}/summary', {'event': event_id}, f'{sport} summary {event_id}')
    if not data:
        return None
    pc = data.get('pickcenter')
    if not pc:
        return None
    entry = pc[0]  # primary/consensus book ESPN surfaces first
    spread_open, spread_close = _open_close(entry.get('pointSpread', {}).get('home'))
    total_open, total_close   = _open_close(entry.get('total', {}).get('over'))
    if spread_open is None and spread_close is None and total_open is None and total_close is None:
        return None
    return {
        'spread_open':  spread_open,
        'spread_close': spread_close,
        'total_open':   total_open,
        'total_close':  total_close,
    }
