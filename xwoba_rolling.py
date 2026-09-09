"""
xwoba_rolling.py — Incremental, point-in-time team xwOBA from Baseball Savant.

Savant's leaderboard/expected_statistics endpoint (used by
bootstrap_2024.fetch_team_xwoba() when as_of_date is omitted) only supports
a `year` param — full season-to-date-as-of-fetch-time, not point-in-time.

Savant's statcast_search/csv endpoint DOES support date bounds
(game_date_gt/game_date_lt), but silently caps results at ~25,000 rows per
request. A single day of league-wide pitches is already ~5,500 rows
(verified), so any single request spanning more than ~4-5 days risks silent
truncation — no error, just missing rows, which would quietly produce a
wrong (undercounted, and non-obviously-so) team aggregate. We chunk
day-by-day to stay safely under that cap.

To avoid re-fetching the whole season on every call, each day's PA-weighted
per-team (numerator, denominator) contribution is folded once and cached
on disk, keyed by calendar date — NOT as a running cumulative total. A
running total would give the wrong answer for any as_of_date earlier than
the furthest date already cached (e.g. game B, on an earlier date, gets
backfilled after game A on a later date already advanced the cache — a
running-total design would silently include A's date in B's "point-in-time"
snapshot). Storing per-day contributions and summing only the days <=
as_of_date on every call is what actually makes this exact for any query
order, at the cost of a trivial re-sum (a season is ~180 days).

Methodology matches Savant's own team/player xwOBA leaderboard: for every
plate-appearance-ending pitch, use estimated_woba_using_speedangle when the
ball was put in play (an "expected" value derived from exit velocity/launch
angle); otherwise (strikeout, walk, HBP — nothing to estimate, since there's
no batted ball) fall back to the actual woba_value for that outcome. Weight
by woba_denom (1 for PAs that count as a wOBA trial, 0 for the handful that
don't, e.g. sac bunts) and average.
"""
import csv
import io
import json
import os
import time
from datetime import date, timedelta

import requests

_STATCAST_SEARCH_URL = 'https://baseballsavant.mlb.com/statcast_search/csv'
_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; mlb-bootstrap/1.0)'}
_HERE = os.path.dirname(os.path.abspath(__file__))


def _cache_path(season):
    return os.path.join(_HERE, f'xwoba_rolling_cache_{season}.json')


def _parse_iso(s):
    y, m, d = (int(x) for x in s.split('-'))
    return date(y, m, d)


def _load_cache(season):
    path = _cache_path(season)
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            data.setdefault('days', {})
            return data
        except Exception:
            pass
    return {'days': {}}


def _save_cache(season, cache):
    path = _cache_path(season)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except Exception as e:
        print(f'  ! xwoba_rolling: failed to save cache for {season}: {e}', flush=True)


def _fetch_day_csv(day_iso, retries=3):
    """Fetch every pitch league-wide for one calendar day (regular season)."""
    d = _parse_iso(day_iso)
    params = {
        'all': 'true',
        'hfGT': 'R|',
        'hfSea': f'{day_iso[:4]}|',
        'player_type': 'batter',
        'game_date_gt': (d - timedelta(days=1)).isoformat(),
        'game_date_lt': (d + timedelta(days=1)).isoformat(),
        'type': 'details',
    }
    for attempt in range(retries):
        try:
            r = requests.get(_STATCAST_SEARCH_URL, params=params, headers=_HEADERS, timeout=60)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt == retries - 1:
                print(f'  ! xwoba_rolling: statcast_search {day_iso} failed: {e}', flush=True)
                return None
            time.sleep(1 + attempt)
    return None


def _compute_day_totals(day_iso):
    """Fetch one day of pitches, return {'num': {team: float}, 'den': {team: float}}
    for that day only, or None on fetch failure."""
    text = _fetch_day_csv(day_iso)
    if text is None:
        return None

    num, den = {}, {}
    for row in csv.DictReader(io.StringIO(text)):
        events = (row.get('events') or '').strip()
        if not events:
            continue  # not the last pitch of a plate appearance

        try:
            denom = float(row.get('woba_denom') or 0)
        except (TypeError, ValueError):
            denom = 0.0
        if denom <= 0:
            continue  # PA that doesn't count as a wOBA trial (e.g. sac bunt)

        est = row.get('estimated_woba_using_speedangle')
        val_raw = est if est not in (None, '', 'null') else row.get('woba_value')
        try:
            value = float(val_raw)
        except (TypeError, ValueError):
            continue

        topbot = row.get('inning_topbot')
        home = (row.get('home_team') or '').lower().strip()
        away = (row.get('away_team') or '').lower().strip()
        team = away if topbot == 'Top' else home
        if not team:
            continue

        num[team] = num.get(team, 0.0) + value * denom
        den[team] = den.get(team, 0.0) + denom

    return {'num': num, 'den': den}


def get_team_xwoba_asof(season, as_of_date, season_start):
    """
    Point-in-time PA-weighted team xwOBA through the day BEFORE as_of_date.

    season_start: 'YYYY-MM-DD' — SEASON_DATES[season][0] from bootstrap_2024.
    Returns {team_abbr_lower: xwoba_float}. Empty dict if no games have been
    played yet as of as_of_date, or if every fetch attempt in this call failed.

    Exact regardless of call order: each day's contribution is cached once
    and summed fresh on every call for just the days <= as_of_date, so
    backfilling an earlier date after a later one has already been cached
    still gets the correct (smaller) snapshot.
    """
    target_end_dt = _parse_iso(as_of_date) - timedelta(days=1)
    start_dt = _parse_iso(season_start)
    if target_end_dt < start_dt:
        return {}
    target_end = target_end_dt.isoformat()

    cache = _load_cache(season)
    days = cache['days']

    day = start_dt
    changed = False
    fetched_since_save = 0
    while day <= target_end_dt:
        iso = day.isoformat()
        if iso not in days:
            totals = _compute_day_totals(iso)
            if totals is None:
                break  # stop; missing day(s) retried on next call/run
            days[iso] = totals
            changed = True
            fetched_since_save += 1
            # Save every few days, not just at the end — a cold start can
            # walk 100+ days at several seconds each; without this, killing
            # the process partway through (or it crashing on a bad response)
            # would throw away everything fetched so far instead of resuming
            # from where it left off on the next run.
            if fetched_since_save >= 5:
                _save_cache(season, cache)
                fetched_since_save = 0
            time.sleep(0.2)
        day += timedelta(days=1)

    if changed and fetched_since_save:
        _save_cache(season, cache)

    num_tot, den_tot = {}, {}
    for iso, d in days.items():
        if iso > target_end:
            continue
        for t, v in d['num'].items():
            num_tot[t] = num_tot.get(t, 0.0) + v
        for t, v in d['den'].items():
            den_tot[t] = den_tot.get(t, 0.0) + v

    return {t: round(num_tot[t] / den_tot[t], 4)
            for t in den_tot if den_tot[t] > 0}
