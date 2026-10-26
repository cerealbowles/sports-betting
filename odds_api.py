import os
import json
import time
import unicodedata
import requests
import odds_history
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo('America/New_York')

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

_cache = {}
_TTL = 300   # 5 min — short, just to avoid re-parsing on rapid page loads

# File-based cache so a restart doesn't throw away the last known lines
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
    'nba': 'basketball_nba',
    'wnba': 'basketball_wnba',
}

# ESPN scoreboard URL slug + whether the CFB-only FBS-group filter applies
_ESPN_SLUGS = {
    'mlb':  'baseball/mlb',
    'nhl':  'hockey/nhl',
    'nfl':  'football/nfl',
    'cfb':  'football/college-football',
    'nba':  'basketball/nba',
    'wnba': 'basketball/wnba',
}
_FBS_GROUP = 80  # matches cfb_api.py — without it ESPN drops most FBS games


def _normalize(name):
    """Lowercase and strip accents for fuzzy team-name matching."""
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip()


def _american_to_implied(price):
    """American odds → raw implied probability (no vig removal)."""
    if price > 0:
        return 100 / (price + 100)
    return abs(price) / (abs(price) + 100)


def _num(line):
    """'+4.5' / '-4.5' / 'o43.5' / 'u43.5' → 4.5 / -4.5 / 43.5 / 43.5."""
    if line is None:
        return None
    try:
        return float(str(line).lstrip('ou'))
    except (TypeError, ValueError):
        return None


def _int_odds(odds):
    if odds is None:
        return None
    try:
        return int(float(odds))
    except (TypeError, ValueError):
        return None


def _espn_provider(comp):
    """Pick DraftKings if present (it always has been in testing), else
    whatever ESPN lists first — defensive fallback, not expected to trigger."""
    odds_list = comp.get('odds') or []
    if not odds_list:
        return None
    for o in odds_list:
        if (o.get('provider') or {}).get('name') == 'DraftKings':
            return o
    return odds_list[0]


def _espn_spread(entry):
    """Return (home_point, home_price, away_price) — home_point negative
    means home favored, matching odds_api's existing convention."""
    ps = entry.get('pointSpread') or {}
    home_close = (ps.get('home') or {}).get('close') or {}
    away_close = (ps.get('away') or {}).get('close') or {}
    home_point = _num(home_close.get('line'))
    home_price = _int_odds(home_close.get('odds'))
    away_price = _int_odds(away_close.get('odds'))
    return home_point, home_price, away_price


def _espn_moneyline(entry):
    """Return (home_price, away_price) American odds."""
    ml = entry.get('moneyline') or {}
    home_price = _int_odds(((ml.get('home') or {}).get('close') or {}).get('odds'))
    away_price = _int_odds(((ml.get('away') or {}).get('close') or {}).get('odds'))
    return home_price, away_price


def _espn_total(entry):
    """Return (line, over_odds, under_odds)."""
    tot = entry.get('total') or {}
    over_close  = (tot.get('over') or {}).get('close') or {}
    under_close = (tot.get('under') or {}).get('close') or {}
    line = _num(over_close.get('line'))
    if line is None:
        line = _num(under_close.get('line'))
    over_odds  = _int_odds(over_close.get('odds'))
    under_odds = _int_odds(under_close.get('odds'))
    return line, over_odds, under_odds


def get_last_fetch_time():
    """Return the most recent successful fetch timestamp across all sports, or None."""
    if _cache:
        ts = max(ts for _, ts in _cache.values())
        return datetime.fromtimestamp(ts, tz=_ET) if ts else None
    try:
        fc = _fc_load()
        if fc:
            ts = max(e['ts'] for e in fc.values())
            return datetime.fromtimestamp(ts, tz=_ET) if ts else None
    except Exception:
        pass
    return None


def get_next_fetch_time():
    """Return the next scheduled cron warm-cache slot (ET) after now — same
    schedule app.py's cron job runs on, unrelated to any quota now."""
    from datetime import timedelta
    hours = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    now = datetime.now(_ET)
    for h in hours:
        candidate = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if candidate > now:
            return candidate
    return (now + timedelta(days=1)).replace(hour=hours[0], minute=0, second=0, microsecond=0)


# Sports whose card odds stop updating at kickoff (ESPN drops the `odds` key
# entirely once a game goes live/final) — keep showing the last pre-game
# price via odds_history's pregame snapshot instead of blanking the card.
_FREEZE_AFTER_START = set(SPORT_KEYS.values())


def _fetch_espn_events(sport, date_str=None):
    """Self-contained ESPN scoreboard fetch for sports with no other ESPN
    call already in scope (currently just MLB). `date_str` is YYYYMMDD;
    omitted means ESPN's own default window for that sport."""
    slug = _ESPN_SLUGS.get(sport)
    if not slug:
        return []
    params = {'limit': 400}
    if date_str:
        params['dates'] = date_str
    if sport == 'cfb':
        params['groups'] = _FBS_GROUP
    try:
        r = requests.get(f"{ESPN_BASE}/{slug}/scoreboard", params=params, timeout=15)
        r.raise_for_status()
        return r.json().get('events', [])
    except Exception as e:
        print(f'[odds] ESPN scoreboard fetch failed for {sport}: {e}', flush=True)
        return []


def get_odds_map(sport, events=None, date_str=None):
    """
    Returns {normalized_home_team: game_odds_dict} for the given ESPN
    `events` list (competitions' .odds[], DraftKings). If `events` is None,
    fetches ESPN's scoreboard itself for `sport` — used by sports that don't
    already have an ESPN event list in scope elsewhere (currently just MLB).

    game_odds_dict = {
        'away_team':    str,
        'home_team':    str,
        'away_best':    int,    # American odds (DraftKings)
        'home_best':    int,
        'away_implied': float,  # vig-free implied prob
        'home_implied': float,
        'total_line':   float,  # game total (O/U), None if not priced
        'home_spread':  float,  # home-team spread, negative = home favored, None if not priced
    }
    """
    sport_key = SPORT_KEYS.get(sport, sport)
    cache_key = f'odds_{sport_key}_{date_str}' if date_str else f'odds_{sport_key}'
    now = time.time()

    if events is None:
        if cache_key in _cache:
            data, ts = _cache[cache_key]
            if now - ts < _TTL:
                return data
        events = _fetch_espn_events(sport, date_str=date_str)

    if not events:
        cached = _cache.get(cache_key, ({}, 0))[0]
        if not cached:
            cached = _fc_load().get(cache_key, {}).get('data', {})
        return cached

    result = {}
    for event in events:
        comp = (event.get('competitions') or [{}])[0]
        tmap = {c.get('homeAway'): c for c in comp.get('competitors', [])}
        home_name = (tmap.get('home', {}).get('team') or {}).get('displayName', '')
        away_name = (tmap.get('away', {}).get('team') or {}).get('displayName', '')
        if not home_name or not away_name:
            continue

        entry_raw = _espn_provider(comp)
        if entry_raw is None:
            continue

        home_price, away_price = _espn_moneyline(entry_raw)
        if home_price is None or away_price is None:
            continue

        home_spread, home_spread_price, away_spread_price = _espn_spread(entry_raw)
        total_line, over_odds, under_odds = _espn_total(entry_raw)

        ou_str = f' | O/U {total_line}' if total_line is not None else ''
        print(f'[odds] {sport_key}: {away_name} {away_price:+d} @ '
              f'{home_name} {home_price:+d}{ou_str}', flush=True)

        raw_away = _american_to_implied(away_price)
        raw_home = _american_to_implied(home_price)
        total    = raw_away + raw_home
        vig_free_away = raw_away / total if total else 0.5
        vig_free_home = raw_home / total if total else 0.5

        # Hour-precision UTC timestamp so same-day series games don't collide
        event_date = event.get('date') or ''
        event_time = event_date[:13]  # 'YYYY-MM-DDTHH'
        hist_key = f"{_normalize(home_name)}_{_normalize(away_name)}_{event_time}"

        # Only record pre-game odds into the CLV/line-movement history —
        # once the game has started, price swings reflect in-game win
        # probability, not market/sharp movement.
        is_preview = True
        if event_date:
            try:
                commence_dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                is_preview = commence_dt > datetime.now(timezone.utc)
            except Exception:
                is_preview = True
        if is_preview:
            odds_history.record(sport_key, hist_key, home_price, away_price,
                                 home_spread=home_spread, home_spread_price=home_spread_price,
                                 away_spread_price=away_spread_price, total_line=total_line,
                                 over_odds=over_odds, under_odds=under_odds)
        else:
            odds_history.save_live(sport_key, hist_key, home_price, away_price)
        movement = odds_history.get_movement(sport_key, hist_key)

        map_key = f"{_normalize(home_name)}_{event_time}"
        entry = {
            'away_team':    away_name,
            'home_team':    home_name,
            'away_best':    away_price,
            'home_best':    home_price,
            'away_implied': round(vig_free_away, 4),
            'home_implied': round(vig_free_home, 4),
            'total_line':   total_line,
            'over_odds':    over_odds,
            'under_odds':   under_odds,
            'home_spread':        home_spread,
            'home_spread_price':  home_spread_price,
            'away_spread_price':  away_spread_price,
        }
        if sport_key in _FREEZE_AFTER_START:
            # Keep the last pre-game prices on the card once the game goes
            # live/final — ESPN drops the `odds` key entirely at that point.
            if is_preview:
                odds_history.save_pregame(sport_key, hist_key, entry)
            else:
                entry = odds_history.get_pregame(sport_key, hist_key) or entry
        result[map_key] = {**entry, **movement}

    # Also carry forward any games ESPN no longer lists odds for at all
    # (gone live/final and dropped from this scoreboard fetch window) but
    # that still have a frozen pregame snapshot from an earlier fetch.
    _cache[cache_key] = (result, now)
    _fc_save(cache_key, result, now)
    return result


def lookup_game_odds(odds_map, home_name, away_name, game_date=None):
    """
    Look up odds for a specific game by home+date key.
    Falls back to searching all dates if no date provided (NHL/NFL).
    Handles home/away reversal between ESPN's display order and schedule APIs.
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
            # Total (over/under) isn't home/away-oriented, so it carries
            # through unchanged regardless of which side got swapped.
            'total_line':   r.get('total_line'),
            'over_odds':    r.get('over_odds'),
            'under_odds':   r.get('under_odds'),
            # Spread IS home/away-oriented — flips sign (and prices swap sides) when the side swaps.
            'home_spread':        (-r['home_spread']) if r.get('home_spread') is not None else None,
            'home_spread_price':  r.get('away_spread_price'),
            'away_spread_price':  r.get('home_spread_price'),
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
