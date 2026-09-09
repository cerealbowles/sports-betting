"""
Persists FanDuel odds snapshots to SQLite so we can show line movement.

Each time get_odds_map() fetches fresh odds, it calls record() per game.
A new row is only written when the odds actually change, so the table stays small.

get_movement() returns opening + previous observations for display.
"""
import os
import sqlite3
import unicodedata
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
'''


def _conn():
    c = sqlite3.connect(_DB_PATH, check_same_thread=False)
    c.executescript(_CREATE)
    return c


def record(sport, game_key, home_odds, away_odds):
    """Write a snapshot only if odds changed since the last observation."""
    try:
        with _conn() as c:
            last = c.execute(
                'SELECT home_odds, away_odds FROM odds_snapshot '
                'WHERE sport=? AND game_key=? ORDER BY id DESC LIMIT 1',
                (sport, game_key),
            ).fetchone()
            if last and last[0] == home_odds and last[1] == away_odds:
                return  # unchanged — skip
            c.execute(
                'INSERT INTO odds_snapshot (sport, game_key, home_odds, away_odds, recorded_at) '
                'VALUES (?,?,?,?,?)',
                (sport, game_key, home_odds, away_odds,
                 datetime.now(timezone.utc).isoformat()),
            )
    except Exception:
        pass


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


def get_latest(sport, game_key):
    """Most recent snapshot from today (ET) — used for live CLV on open bets.

    Bounded to midnight ET today so yesterday's final in-game odds never
    surface as today's pre-game line.
    """
    try:
        lower = _today_et_start_utc()
        with _conn() as c:
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
    """Last snapshot before before_dt (game start) on the same ET calendar date.

    Lower bound = midnight ET on the game date, so only same-day odds are
    considered. A PHI game on 5/20 will never see 5/19 snapshots.
    """
    try:
        cutoff = before_dt.isoformat() if hasattr(before_dt, 'isoformat') else str(before_dt)
        lower  = _et_day_start_utc(before_dt)
        with _conn() as c:
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
    def _dir(label, move):
        return f"{label}_{'for' if (move or 0) >= 0 else 'against'}"

    if not game_start:
        return None, None
    h_norm = _normalize(home_name)
    a_norm = _normalize(away_name)
    event_hour_key = game_start.astimezone(timezone.utc).strftime('%Y-%m-%dT%H')
    game_key = f'{h_norm}_{a_norm}_{event_hour_key}'
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
        if game_start and dt >= game_start:
            continue
        vf_h, vf_a = _vig_free(h, a)
        if vf_h is None:
            continue
        series.append((dt, vf_h if fav_home else vf_a))

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
