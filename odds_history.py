"""
Persists FanDuel odds snapshots to SQLite so we can show line movement.

Each time get_odds_map() fetches fresh odds, it calls record() per game.
A new row is only written when the odds actually change, so the table stays small.

get_movement() returns opening + previous observations for display.
"""
import json
import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_HAS_HOUR_BUCKET = re.compile(r'_\d{4}-\d{2}-\d{2}T\d{2}$')

_ET = ZoneInfo('America/New_York')


def _normalize(name):
    """Lowercase and strip accents — must match odds_api._normalize exactly,
    since game_key is built from it on both the write (odds_api) and read
    (classify_movement) sides. Duplicated here to avoid a circular import."""
    name = unicodedata.normalize('NFD', name or '')
    name = ''.join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip()


def _implied(odds):
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _vig_free(home_odds, away_odds):
    rh, ra = _implied(home_odds), _implied(away_odds)
    total = rh + ra
    if not total:
        return None, None
    return rh / total, ra / total


def _et_day_start_utc(dt):
    """Return UTC isoformat of midnight ET on the same ET calendar date as dt.

    All MLB games are scheduled in Eastern Time, so this gives the correct
    same-day lower bound regardless of when UTC rolls over.
    """
    if getattr(dt, 'tzinfo', None):
        et = dt.astimezone(_ET)
    else:
        et = dt.replace(tzinfo=timezone.utc).astimezone(_ET)
    return datetime(et.year, et.month, et.day, tzinfo=_ET).astimezone(timezone.utc).isoformat()


def _today_et_start_utc():
    """UTC isoformat of midnight ET today."""
    now_et = datetime.now(_ET)
    return datetime(now_et.year, now_et.month, now_et.day, tzinfo=_ET).astimezone(timezone.utc).isoformat()

_DB_PATH = os.environ.get(
    'DB_PATH',
    os.path.join(os.path.dirname(__file__), 'instance', 'bets.db'),
)

_CREATE = '''
CREATE TABLE IF NOT EXISTS odds_snapshot (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sport       TEXT    NOT NULL,
    game_key    TEXT    NOT NULL,
    home_odds   INTEGER NOT NULL,
    away_odds   INTEGER NOT NULL,
    recorded_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_odds_key ON odds_snapshot(sport, game_key);
CREATE TABLE IF NOT EXISTS pregame_odds (
    sport       TEXT NOT NULL,
    game_key    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (sport, game_key)
);
CREATE TABLE IF NOT EXISTS live_odds (
    sport       TEXT NOT NULL,
    game_key    TEXT NOT NULL,
    home_odds   INTEGER NOT NULL,
    away_odds   INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (sport, game_key)
);
'''


# Columns added after odds_snapshot's original ML-only launch — CREATE TABLE
# IF NOT EXISTS above never alters an existing table, so an explicit
# ADD COLUMN migration is needed for a production DB that already has rows.
_SNAPSHOT_MIGRATED_COLS = {
    'home_spread':       'REAL',
    'home_spread_price': 'INTEGER',
    'away_spread_price': 'INTEGER',
    'total_line':        'REAL',
    'over_odds':         'INTEGER',
    'under_odds':        'INTEGER',
}


def _migrate(c):
    existing = {row[1] for row in c.execute('PRAGMA table_info(odds_snapshot)').fetchall()}
    for col, col_type in _SNAPSHOT_MIGRATED_COLS.items():
        if col not in existing:
            c.execute(f'ALTER TABLE odds_snapshot ADD COLUMN {col} {col_type}')


def _conn():
    c = sqlite3.connect(_DB_PATH, check_same_thread=False)
    c.executescript(_CREATE)
    _migrate(c)
    return c


def record(sport, game_key, home_odds, away_odds, home_spread=None,
           home_spread_price=None, away_spread_price=None, total_line=None,
           over_odds=None, under_odds=None):
    """Write a snapshot only if any tracked value changed since the last
    observation — moneyline, spread line/price, or total line/price."""
    try:
        with _conn() as c:
            last = c.execute(
                'SELECT home_odds, away_odds, home_spread, home_spread_price, '
                'away_spread_price, total_line, over_odds, under_odds '
                'FROM odds_snapshot WHERE sport=? AND game_key=? ORDER BY id DESC LIMIT 1',
                (sport, game_key),
            ).fetchone()
            new_vals = (home_odds, away_odds, home_spread, home_spread_price,
                        away_spread_price, total_line, over_odds, under_odds)
            if last and tuple(last) == new_vals:
                return  # unchanged — skip
            c.execute(
                'INSERT INTO odds_snapshot (sport, game_key, home_odds, away_odds, '
                'home_spread, home_spread_price, away_spread_price, total_line, '
                'over_odds, under_odds, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (sport, game_key, *new_vals, datetime.now(timezone.utc).isoformat()),
            )
    except Exception:
        pass


def save_pregame(sport, game_key, payload):
    """Overwrite the stored pre-game odds display payload for a game. Called on
    every pre-game refresh, so once a game starts (and callers stop saving) the
    row holds the last odds seen before kickoff."""
    try:
        with _conn() as c:
            c.execute(
                'INSERT OR REPLACE INTO pregame_odds (sport, game_key, payload, recorded_at) '
                'VALUES (?,?,?,?)',
                (sport, game_key, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
            )
    except Exception:
        pass


def get_pregame(sport, game_key):
    """Last pre-game odds payload saved by save_pregame(), or None."""
    try:
        with _conn() as c:
            row = c.execute(
                'SELECT payload FROM pregame_odds WHERE sport=? AND game_key=?',
                (sport, game_key),
            ).fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def save_live(sport, game_key, home_odds, away_odds):
    """Overwrite the current in-play moneyline for a live game.

    Deliberately a separate table from odds_snapshot (pre-game only, used
    for CLV/line-movement analysis) so in-game price swings never leak into
    that history — this just holds the single freshest live price per game,
    for the open-bet card's "current market odds" readout once a game goes
    Live (see api_live_scores in app.py).
    """
    try:
        with _conn() as c:
            c.execute(
                'INSERT OR REPLACE INTO live_odds (sport, game_key, home_odds, away_odds, recorded_at) '
                'VALUES (?,?,?,?,?)',
                (sport, game_key, home_odds, away_odds,
                 datetime.now(timezone.utc).isoformat()),
            )
    except Exception:
        pass


def get_live(sport, game_key):
    """Freshest in-play moneyline recorded by save_live(), or None if the
    book hasn't offered (or we haven't yet polled) a live price for this
    game."""
    try:
        with _conn() as c:
            row = c.execute(
                'SELECT home_odds, away_odds FROM live_odds WHERE sport=? AND game_key=?',
                (sport, game_key),
            ).fetchone()
        return {'home_odds': row[0], 'away_odds': row[1]} if row else None
    except Exception:
        return None


def get_movement(sport, game_key):
    """
    Returns a movement dict or {} if fewer than 2 observations exist.

    {
      'opening_home': int,   # first observed home odds
      'opening_away': int,
      'prev_home':    int,   # observation before the most recent
      'prev_away':    int,
    }
    """
    try:
        with _conn() as c:
            rows = c.execute(
                'SELECT home_odds, away_odds FROM odds_snapshot '
                'WHERE sport=? AND game_key=? ORDER BY id ASC',
                (sport, game_key),
            ).fetchall()
        if len(rows) < 2:
            return {}
        return {
            'opening_home': rows[0][0],
            'opening_away': rows[0][1],
            'prev_home':    rows[-2][0],
            'prev_away':    rows[-2][1],
        }
    except Exception:
        return {}


def get_latest(sport, game_key, game_start=None):
    """Most recent pre-game snapshot — used for live CLV/line-move on open bets.

    Modern game_keys are hour-bucketed (home_away_YYYY-MM-DDTHH), so an exact
    match already uniquely identifies this one game — two teams playing each
    other in the same UTC hour essentially never happens. No date bound is
    needed: the earliest snapshot ever recorded for that exact key is valid,
    even if it's days old and nothing has changed since (record() only writes
    a row when odds actually change, so an old "latest" is normal, not stale).

    Legacy bare keys (home_away, no hour bucket — bets placed before this
    format existed) are ambiguous across multiple meetings between the same
    teams, so those still fall back to a day-scoped LIKE match against
    game_start (or today, if unknown).
    """
    try:
        with _conn() as c:
            if _HAS_HOUR_BUCKET.search(game_key or ''):
                row = c.execute(
                    'SELECT home_odds, away_odds FROM odds_snapshot '
                    'WHERE sport=? AND game_key=? ORDER BY id DESC LIMIT 1',
                    (sport, game_key),
                ).fetchone()
            else:
                lower = _et_day_start_utc(game_start) if game_start else _today_et_start_utc()
                row = c.execute(
                    'SELECT home_odds, away_odds FROM odds_snapshot '
                    'WHERE sport=? AND (game_key=? OR game_key LIKE ?) AND recorded_at >= ? '
                    'ORDER BY id DESC LIMIT 1',
                    (sport, game_key, game_key + '_%', lower),
                ).fetchone()
        return {'home_odds': row[0], 'away_odds': row[1]} if row else None
    except Exception:
        return None


def get_closing_line(sport, game_key, before_dt):
    """Last snapshot before before_dt (game start) for this exact game.

    The upper bound (before_dt) always applies, so in-game/post-game odds
    are never mistaken for the closing line. Modern hour-bucketed game_keys
    already uniquely identify the one game on an exact match, so no lower
    bound is needed — a snapshot from days before kickoff is still valid.
    Legacy bare keys (no hour bucket) are ambiguous across multiple meetings
    between the same teams, so those still get a same-ET-day lower bound:
    a PHI game on 5/20 will never see 5/19 snapshots.
    """
    try:
        cutoff = before_dt.isoformat() if hasattr(before_dt, 'isoformat') else str(before_dt)
        with _conn() as c:
            if _HAS_HOUR_BUCKET.search(game_key or ''):
                row = c.execute(
                    'SELECT home_odds, away_odds FROM odds_snapshot '
                    'WHERE sport=? AND game_key=? AND recorded_at <= ? '
                    'ORDER BY id DESC LIMIT 1',
                    (sport, game_key, cutoff),
                ).fetchone()
            else:
                lower = _et_day_start_utc(before_dt)
                row = c.execute(
                    'SELECT home_odds, away_odds FROM odds_snapshot '
                    'WHERE sport=? AND (game_key=? OR game_key LIKE ?) '
                    'AND recorded_at >= ? AND recorded_at <= ? '
                    'ORDER BY id DESC LIMIT 1',
                    (sport, game_key, game_key + '_%', lower, cutoff),
                ).fetchone()
        return {'home_odds': row[0], 'away_odds': row[1]} if row else None
    except Exception:
        return None


def _classify_series(series, game_start):
    """Shared shape-classification core, extracted from the original ML-only
    classify_movement so Spread/Total can reuse the exact same logic against
    their own implied-probability series instead of duplicating it.

    `series` is a list of (datetime, implied_prob 0-1) tuples for the picked
    side, already filtered to pre-game snapshots. Returns (profile,
    total_move_pct) — see classify_movement's docstring for what each
    profile label means; this is the same taxonomy for every market.
    """
    def _dir(label, move):
        return f"{label}_{'for' if (move or 0) >= 0 else 'against'}"

    if len(series) < 3:
        game_over = datetime.now(timezone.utc) >= game_start
        if not game_over:
            return None, None  # still pending — more snapshots may arrive before game time
        if len(series) == 0:
            return 'no_data', None
        if len(series) == 1:
            return 'static', 0.0
        move = round((series[-1][1] - series[0][1]) * 100, 2)
        return _dir('minimal', move), move

    series.sort(key=lambda x: x[0])
    times = [s[0] for s in series]
    probs = [s[1] for s in series]
    t0, t1 = times[0], times[-1]
    if (t1 - t0).total_seconds() <= 0:
        return None, None

    p0, p_last = probs[0], probs[-1]
    total_move = (p_last - p0) * 100  # percentage points

    max_dev_signed = 0.0
    for p in probs:
        dev = (p - p0) * 100
        if abs(dev) > abs(max_dev_signed):
            max_dev_signed = dev
    retraced = abs(max_dev_signed) - abs(total_move)
    is_reversal = abs(max_dev_signed) > 1.0 and retraced > 0.5 * abs(max_dev_signed) and (
        (max_dev_signed > 0) != (total_move >= 0) or abs(total_move) < 0.5 * abs(max_dev_signed)
    )
    if is_reversal:
        return _dir('reversal', total_move), round(total_move, 2)
    if abs(total_move) < 0.5:
        return 'flat', round(total_move, 2)

    mid_t = t0 + (t1 - t0) / 2
    mid_val = probs[0]
    for t, p in zip(times, probs):
        if t <= mid_t:
            mid_val = p
        else:
            break
    move_by_mid = (mid_val - p0) * 100
    frac_by_mid = (move_by_mid / total_move) if abs(total_move) > 0.3 else None

    if frac_by_mid is not None and frac_by_mid >= 0.65:
        return _dir('early_sharp', total_move), round(total_move, 2)
    if frac_by_mid is not None and frac_by_mid <= 0.30:
        return _dir('late_sharp', total_move), round(total_move, 2)
    return _dir('sustained', total_move), round(total_move, 2)


def _movement_game_key(home_name, away_name, game_start):
    h_norm = _normalize(home_name)
    a_norm = _normalize(away_name)
    event_hour_key = game_start.astimezone(timezone.utc).strftime('%Y-%m-%dT%H')
    return f'{h_norm}_{a_norm}_{event_hour_key}'


def classify_movement(sport, home_name, away_name, fav_home, game_start=None):
    """Classify the pre-game moneyline movement profile on the model-pick side.

    Matches the EXACT hour-bucketed game_key odds_api.py wrote at record time
    (derived from game_start, the same way odds_api builds it from commence_time).
    A loose LIKE match on just team names would blend together snapshots from
    a team's other games against the same opponent on different dates — common
    in MLB, which plays 3-4 game series — so this requires game_start.

    Returns (profile, total_move_pct) where profile is one of:
      'early_sharp_for'/'early_sharp_against' — most of the move happened in
                      the first half of the observed pre-game window, then flat
      'late_sharp_for'/'late_sharp_against'   — flat early, most of the move
                      happened closer to game time
      'sustained_for'/'sustained_against'     — steady drift across the whole
                      window, no early/late skew
      'reversal_for'/'reversal_against'       — moved one way then retraced
                      more than half of that move; direction = where it net
                      ended up at game time
      'minimal_for'/'minimal_against' — game started/finished, only 2 pre-game
                      snapshots exist — a real move happened but too few points
                      to tell its shape, though direction is still known
      'flat'        — 3+ snapshots, but net move under 0.5 percentage points
                      (drifted and came back — not the same as never moving;
                      not direction-split since the net move is too small to
                      call a direction meaningful)
      'static'      — game has started/finished and only 1 pre-game snapshot
                      was ever recorded — the line genuinely never changed
                      (record() only writes a row when odds actually change)
      'no_data'     — game has started/finished and zero pre-game snapshots
                      were ever recorded for it

    '_for' means the line moved toward the model's pick (the pick's implied
    probability rose); '_against' means it moved toward the opponent. This is
    the whole point of tracking movement at all — "Late Sharp against our
    pick" and "Late Sharp for our pick" are very different signals and were
    previously being lumped into one undifferentiated 'late_sharp' bucket.

    Returns (None, None) if game_start is missing, or if the game hasn't
    started yet and fewer than 3 snapshots exist — still pending, not final.

    fav_home: True if the model's pick is the home side (selects which
    column's implied probability to track).
    game_start: datetime — required; also used as the cutoff (snapshots at/after
    this are excluded as in-game odds).
    """
    if not game_start:
        return None, None
    game_key = _movement_game_key(home_name, away_name, game_start)
    try:
        with _conn() as c:
            rows = c.execute(
                'SELECT home_odds, away_odds, recorded_at FROM odds_snapshot '
                'WHERE sport=? AND game_key=? ORDER BY id ASC',
                (sport, game_key),
            ).fetchall()
    except Exception:
        return None, None

    series = []
    for h, a, ts in rows:
        if abs(h) > 1000 or abs(a) > 1000:
            continue  # implausible odds — data artifact
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            continue
        if dt >= game_start:
            continue
        vf_h, vf_a = _vig_free(h, a)
        if vf_h is None:
            continue
        series.append((dt, vf_h if fav_home else vf_a))

    return _classify_series(series, game_start)


def classify_spread_movement(sport, home_name, away_name, pick_is_home, game_start=None):
    """Same taxonomy as classify_movement, applied to the spread market: at
    each snapshot, the picked side's vig-free implied probability of
    covering (derived from that moment's home_spread_price/away_spread_price
    — the line value itself isn't held fixed across the series, only the
    market's confidence in the pick's side at whatever number was posted
    at that time). See classify_movement's docstring for the profile
    taxonomy and return shape; identical here."""
    if not game_start:
        return None, None
    game_key = _movement_game_key(home_name, away_name, game_start)
    try:
        with _conn() as c:
            rows = c.execute(
                'SELECT home_spread_price, away_spread_price, recorded_at FROM odds_snapshot '
                'WHERE sport=? AND game_key=? AND home_spread_price IS NOT NULL '
                'AND away_spread_price IS NOT NULL ORDER BY id ASC',
                (sport, game_key),
            ).fetchall()
    except Exception:
        return None, None

    series = []
    for h, a, ts in rows:
        if abs(h) > 1000 or abs(a) > 1000:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            continue
        if dt >= game_start:
            continue
        vf_h, vf_a = _vig_free(h, a)
        if vf_h is None:
            continue
        series.append((dt, vf_h if pick_is_home else vf_a))

    return _classify_series(series, game_start)


def classify_total_movement(sport, home_name, away_name, pick_is_over, game_start=None):
    """Same taxonomy as classify_movement, applied to the total market: at
    each snapshot, the picked side's (over/under) vig-free implied
    probability from that moment's over_odds/under_odds. See
    classify_movement's docstring for the profile taxonomy and return
    shape; identical here."""
    if not game_start:
        return None, None
    game_key = _movement_game_key(home_name, away_name, game_start)
    try:
        with _conn() as c:
            rows = c.execute(
                'SELECT over_odds, under_odds, recorded_at FROM odds_snapshot '
                'WHERE sport=? AND game_key=? AND over_odds IS NOT NULL '
                'AND under_odds IS NOT NULL ORDER BY id ASC',
                (sport, game_key),
            ).fetchall()
    except Exception:
        return None, None

    series = []
    for o, u, ts in rows:
        if abs(o) > 1000 or abs(u) > 1000:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            continue
        if dt >= game_start:
            continue
        vf_o, vf_u = _vig_free(o, u)
        if vf_o is None:
            continue
        series.append((dt, vf_o if pick_is_over else vf_u))

    return _classify_series(series, game_start)


def get_storage_stats():
    """Row count, distinct games, date range, and an estimated byte size for
    odds_snapshot — shown on the Settings page since purging is disabled and
    this table now grows indefinitely until rolled up into per-game categories.
    """
    try:
        with _conn() as c:
            row = c.execute(
                'SELECT COUNT(*), COUNT(DISTINCT sport || game_key), '
                'MIN(recorded_at), MAX(recorded_at) FROM odds_snapshot'
            ).fetchone()
            # Rough per-row size: two TEXT columns (sport, game_key, recorded_at)
            # + two INTEGER columns + sqlite row overhead (~24 bytes).
            avg_len = c.execute(
                'SELECT AVG(LENGTH(sport) + LENGTH(game_key) + LENGTH(recorded_at)) '
                'FROM odds_snapshot'
            ).fetchone()[0] or 0
        count, n_games, oldest, newest = row
        est_bytes = int(count * (avg_len + 24)) if count else 0
        age_days = None
        if oldest:
            try:
                oldest_dt = datetime.fromisoformat(oldest.replace('Z', '+00:00'))
                age_days = (datetime.now(timezone.utc) - oldest_dt).days
            except Exception:
                pass
        return {
            'count': count or 0,
            'n_games': n_games or 0,
            'oldest': oldest,
            'newest': newest,
            'age_days': age_days,
            'est_bytes': est_bytes,
        }
    except Exception:
        return {'count': 0, 'n_games': 0, 'oldest': None, 'newest': None, 'age_days': None, 'est_bytes': 0}


def purge_old(days=3):
    """Remove snapshots for games that are more than `days` old."""
    try:
        cutoff = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        from datetime import timedelta
        cutoff -= timedelta(days=days)
        with _conn() as c:
            c.execute(
                "DELETE FROM odds_snapshot WHERE recorded_at < ?",
                (cutoff.isoformat(),),
            )
    except Exception:
        pass
