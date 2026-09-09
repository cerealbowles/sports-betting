import os
import json
import time
import unicodedata
import requests
import odds_history
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo('America/New_York')

BASE = "https://api.the-odds-api.com/v4/sports"

_cache = {}
_TTL = 7200   # 2 hours — matches cron schedule below
_FETCH_WINDOW = (8, 22)  # safety gate: never hit API outside this ET window
# Fixed ET hours the cron job fires — predictable, budget-safe
_SCHEDULE_HOURS = (8, 10, 12, 14, 16, 18, 20)

# File-based cache so restarts don't burn API quota
_CACHE_FILE = os.path.join(
    os.path.dirname(os.environ.get('DB_PATH', os.path.join(os.path.dirname(__file__), 'instance', 'bets.db'))),
    'odds_cache.json',
)

def _fc_load():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _fc_save(key, data, ts):
    try:
        fc = _fc_load()
        fc[key] = {'data': data, 'ts': ts}
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, 'w') as f:
            json.dump(fc, f)
    except Exception:
        pass

SPORT_KEYS = {
    'mlb': 'baseball_mlb',
    'nhl': 'icehockey_nhl',
    'nfl': 'americanfootball_nfl',
    'cfb': 'americanfootball_ncaaf',
}

# Abbreviate long bookmaker names for compact display
_BOOK_LABELS = {
    'draftkings':        'DK',
    'fanduel':           'FD',
    'betmgm':            'MGM',
    'caesars':           'CZR',
    'pointsbet':         'PB',
    'bovada':            'BOV',
    'williamhill_us':    'WH',
    'barstool':          'BS',
    'betrivers':         'BR',
    'unibet_us':         'UB',
    'mybookieag':        'MB',
    'betonlineag':       'BOL',
    'lowvig':            'LV',
    'pinnacle':          'PIN',
    'betway':            'BW',
}


def _normalize(name):
    """Lowercase and strip accents for fuzzy team-name matching."""
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip()


def _book_label(key):
    return _BOOK_LABELS.get(key.lower(), key[:4].upper())


def _american_to_implied(price):
    """American odds → raw implied probability (no vig removal)."""
    if price > 0:
        return 100 / (price + 100)
    return abs(price) / (abs(price) + 100)


def _fanduel_price(bookmakers, team_name):
    """Return the FanDuel American odds for team_name, or None if not listed."""
    norm = _normalize(team_name)
    for bm in bookmakers:
        if bm.get('key') != 'fanduel':
            continue
        for market in bm.get('markets', []):
            if market.get('key') != 'h2h':
                continue
            for o in market.get('outcomes', []):
                if _normalize(o.get('name', '')) == norm:
                    return o.get('price')
    return None


def _fanduel_totals(bookmakers):
    """Return (line, over_odds, under_odds) from FanDuel totals market, or (None, None, None)."""
    for bm in bookmakers:
        if bm.get('key') != 'fanduel':
            continue
        for market in bm.get('markets', []):
            if market.get('key') != 'totals':
                continue
            line = over_odds = under_odds = None
            for o in market.get('outcomes', []):
                name = (o.get('name') or '').lower()
                if name == 'over':
                    over_odds = o.get('price')
                    line = o.get('point')
                elif name == 'under':
                    under_odds = o.get('price')
                    if line is None:
                        line = o.get('point')
            if line is not None:
                return line, over_odds, under_odds
    return None, None, None


def get_last_fetch_time():
    """Return the most recent successful fetch timestamp across all sports, or None."""
    if _cache:
        ts = max(ts for _, ts in _cache.values())
        return datetime.fromtimestamp(ts, tz=_ET) if ts else None
    # Fall back to file cache
    try:
        fc = _fc_load()
        if fc:
            ts = max(e['ts'] for e in fc.values())
            return datetime.fromtimestamp(ts, tz=_ET) if ts else None
    except Exception:
        pass
    return None


def get_next_fetch_time():
    """Return the next scheduled cron slot (ET) after now."""
    from datetime import timedelta
    now = datetime.now(_ET)
    for h in _SCHEDULE_HOURS:
        candidate = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if candidate > now:
            return candidate
    # Past last slot today → first slot tomorrow
    return (now + timedelta(days=1)).replace(
        hour=_SCHEDULE_HOURS[0], minute=0, second=0, microsecond=0
    )


def _get_api_keys():
    """Return list of API keys from ODDS_API_KEY (comma-separated)."""
    raw = os.environ.get('ODDS_API_KEY', '')
    return [k.strip() for k in raw.split(',') if k.strip()]

# Track exhausted keys so we skip them without retrying until process restart
_exhausted_keys = set()


def get_odds_map(sport):
    """
    Returns {normalized_home_team: game_odds_dict} for today's games (FanDuel only).
    Returns {} silently if no API keys are set or on any error.
    Rotates through comma-separated keys in ODDS_API_KEY on 401/quota exhaustion.

    game_odds_dict = {
        'away_team':    str,
        'home_team':    str,
        'away_best':    int,    # American odds (FanDuel)
        'home_best':    int,
        'away_implied': float,  # vig-free implied prob
        'home_implied': float,
    }
    """
    keys = _get_api_keys()
    if not keys:
        return {}

    sport_key = SPORT_KEYS.get(sport, sport)
    cache_key  = f'odds_{sport_key}'
    now = time.time()

    # 1. In-memory cache (fastest, lives until restart)
    if cache_key in _cache:
        data, ts = _cache[cache_key]
        if now - ts < _TTL:
            return data

    # 2. File cache (survives restarts — restarts don't burn API quota)
    fc = _fc_load()
    if cache_key in fc:
        entry = fc[cache_key]
        if now - entry['ts'] < _TTL:
            data = entry['data']
            _cache[cache_key] = (data, entry['ts'])
            return data

    # 3. Outside fetch window — return stale cache rather than burning a token
    et_hour = datetime.now(_ET).hour
    if not (_FETCH_WINDOW[0] <= et_hour < _FETCH_WINDOW[1]):
        cached = _cache.get(cache_key, ({}, 0))[0]
        if not cached:
            cached = fc.get(cache_key, {}).get('data', {})
        return cached

    # 4. Live fetch — try each key until one succeeds
    active_keys = [k for k in keys if k not in _exhausted_keys] or keys
    r = None
    used_key = None
    for api_key in active_keys:
        try:
            r = requests.get(
                f"{BASE}/{sport_key}/odds/",
                params={
                    'apiKey':       api_key,
                    'bookmakers':   'fanduel',
                    'markets':      'h2h,totals',
                    'oddsFormat':   'american',
                    'dateFormat':   'iso',
                },
                timeout=10,
            )
            if r.status_code == 401:
                print(f'[odds] key ...{api_key[-6:]} exhausted/invalid — trying next', flush=True)
                _exhausted_keys.add(api_key)
                continue
            remaining = r.headers.get('x-requests-remaining')
            if remaining is not None:
                print(f'[odds] key ...{api_key[-6:]}: {remaining} requests remaining', flush=True)
                if int(remaining) == 0:
                    _exhausted_keys.add(api_key)
            r.raise_for_status()
            used_key = api_key
            break
        except requests.HTTPError:
            continue
        except Exception:
            break

    if r is None or used_key is None:
        return _cache.get(cache_key, ({}, 0))[0]

    try:
        events = r.json()
    except Exception:
        return _cache.get(cache_key, ({}, 0))[0]

    result = {}
    for event in events:
        home_name = event.get('home_team', '')
        away_name = event.get('away_team', '')
        if not home_name or not away_name:
            continue

        away_price = _fanduel_price(event.get('bookmakers', []), away_name)
        home_price = _fanduel_price(event.get('bookmakers', []), home_name)

        if away_price is None or home_price is None:
            print(f'[odds] no FanDuel price for {away_name} @ {home_name} '
                  f'(away={away_price}, home={home_price})', flush=True)
            continue

        bkm = event.get('bookmakers', [])
        total_line, over_odds, under_odds = _fanduel_totals(bkm)

        ou_str = f' | O/U {total_line}' if total_line is not None else ''
        print(f'[odds] {sport_key}: {away_name} {away_price:+d} @ '
              f'{home_name} {home_price:+d}{ou_str}', flush=True)

        raw_away = _american_to_implied(away_price)
        raw_home = _american_to_implied(home_price)
        total    = raw_away + raw_home
        vig_free_away = raw_away / total if total else 0.5
        vig_free_home = raw_home / total if total else 0.5

        # Use hour-precision UTC timestamp so same-day series games don't collide
        # e.g. "2026-05-16T00" vs "2026-05-16T17" are distinct keys
        event_time = (event.get('commence_time') or '')[:13]  # 'YYYY-MM-DDTHH'
        hist_key = f"{_normalize(home_name)}_{_normalize(away_name)}_{event_time}"
        # Only record pre-game (Preview) odds — once the game has started, price
        # swings reflect in-game win probability, not market/sharp movement,
        # and would corrupt line-movement analysis.
        commence_raw = event.get('commence_time') or ''
        is_preview = True
        if commence_raw:
            try:
                commence_dt = datetime.fromisoformat(commence_raw.replace('Z', '+00:00'))
                is_preview = commence_dt > datetime.now(timezone.utc)
            except Exception:
                is_preview = True
        if is_preview:
            odds_history.record(sport_key, hist_key, home_price, away_price)
        movement = odds_history.get_movement(sport_key, hist_key)

        map_key = f"{_normalize(home_name)}_{event_time}"
        result[map_key] = {
            'away_team':    away_name,
            'home_team':    home_name,
            'away_best':    away_price,
            'home_best':    home_price,
            'away_implied': round(vig_free_away, 4),
            'home_implied': round(vig_free_home, 4),
            'total_line':   total_line,
            'over_odds':    over_odds,
            'under_odds':   under_odds,
            **movement,
        }

    _cache[cache_key] = (result, now)
    _fc_save(cache_key, result, now)
    return result


def lookup_game_odds(odds_map, home_name, away_name, game_date=None):
    """
    Look up odds for a specific game by home+date key.
    Falls back to searching all dates if no date provided (NHL/NFL).
    Handles home/away reversal between Odds API and schedule APIs.
    """
    h_norm = _normalize(home_name)
    a_norm = _normalize(away_name)

    def _swapped(r):
        return {
            'home_team':    r.get('away_team'),
            'away_team':    r.get('home_team'),
            'home_best':    r.get('away_best'),
            'away_best':    r.get('home_best'),
            'home_implied': r.get('away_implied'),
            'away_implied': r.get('home_implied'),
            'opening_home': r.get('opening_away'),
            'opening_away': r.get('opening_home'),
            'prev_home':    r.get('prev_away'),
            'prev_away':    r.get('prev_home'),
        }

    if game_date:
        # Try exact key first, then progressively shorter prefixes
        # game_date may be 'YYYY-MM-DDTHH' (13) or 'YYYY-MM-DD' (10)
        for prefix_len in (13, 10):
            prefix = game_date[:prefix_len]
            r = odds_map.get(f'{h_norm}_{prefix}')
            if r:
                return r
            r = odds_map.get(f'{a_norm}_{prefix}')
            if r:
                print(f'[odds] swap: {home_name} @ {away_name} on {prefix} — reversing', flush=True)
                return _swapped(r)
        return None

    # No date provided — scan all keys for a team name match (NHL/NFL)
    for key, val in odds_map.items():
        if key.startswith(h_norm + '_'):
            return val
        if key.startswith(a_norm + '_'):
            return _swapped(val)
    return None
