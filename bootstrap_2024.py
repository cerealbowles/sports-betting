#!/usr/bin/env python3
"""
bootstrap_2024.py — Build MLB model training data from historical seasons.

Usage:
    python3 bootstrap_2024.py               # fetch all seasons, run refit
    python3 bootstrap_2024.py --refit-only  # skip fetch, use cached combined data

Covers all completed seasons defined in SEASONS below.
Each season uses its own team stats — 2024 stats score 2024 games, etc.
"""

import csv
import io
import json
import os
import sys
import time
import requests

SEASONS    = [2024, 2025, 2026]  # 2026 always re-fetched (in-progress)
MLB_API    = 'https://statsapi.mlb.com/api/v1'
LG_ERA     = 4.20  # MLB league-average ERA for opponent-quality normalization
SAVANT_URL = 'https://baseballsavant.mlb.com/leaderboard/expected_statistics'
SAVANT_STATCAST_URL = 'https://baseballsavant.mlb.com/leaderboard/statcast'
HEADERS    = {'User-Agent': 'Mozilla/5.0 (compatible; mlb-bootstrap/1.0)'}

# Approximate regular season date ranges per season
SEASON_DATES = {
    2024: ('2024-03-20', '2024-09-29'),
    2025: ('2025-03-27', '2025-09-28'),
    2026: ('2026-03-26', '2026-09-27'),
}

# Hours behind Eastern Time for each team's home city (summer / DST).
# Must stay in sync with mlb_api._TEAM_TZ_OFFSET.
TEAM_TZ_OFFSET = {
    # ET (0): BAL, BOS, TB, TOR, NYY, CLE, DET, WSH, NYM, PHI, ATL, MIA, CIN, PIT
    110: 0, 111: 0, 139: 0, 141: 0, 147: 0,
    114: 0, 116: 0,
    120: 0, 121: 0, 143: 0, 144: 0, 146: 0,
    113: 0, 134: 0,
    # CT (1): KC, MIN, CWS, HOU, TEX, CHC, STL, MIL
    118: 1, 142: 1, 145: 1, 117: 1, 140: 1, 112: 1, 138: 1, 158: 1,
    # MT (2): ARI, COL
    109: 2, 115: 2,
    # PT (3): LAA, OAK/ATH, SEA, LAD, SD, SF
    108: 3, 133: 3, 136: 3, 119: 3, 135: 3, 137: 3,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url, params, label, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30, headers=HEADERS)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries - 1:
                print(f'  ! {label} failed: {e}', flush=True)
                return None
            time.sleep(1 + attempt)
    return None


def _day_before(iso_date):
    """iso_date ('YYYY-MM-DD') → the previous calendar day, same format."""
    from datetime import date, timedelta
    y, m, d = (int(x) for x in iso_date.split('-'))
    return (date(y, m, d) - timedelta(days=1)).isoformat()


def _month_ranges(start_date, end_date):
    """Split a date range into month-sized chunks for the schedule endpoint."""
    from datetime import date, timedelta
    sy, sm = int(start_date[:4]), int(start_date[5:7])
    ey, em = int(end_date[:4]),   int(end_date[5:7])
    ranges = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        # first day of this month
        chunk_start = max(date(y, m, 1), date(*[int(x) for x in start_date.split('-')]))
        # last day of this month (first of next minus 1)
        if m == 12:
            chunk_end = min(date(y + 1, 1, 1) - timedelta(days=1),
                            date(*[int(x) for x in end_date.split('-')]))
        else:
            chunk_end = min(date(y, m + 1, 1) - timedelta(days=1),
                            date(*[int(x) for x in end_date.split('-')]))
        ranges.append((chunk_start.isoformat(), chunk_end.isoformat()))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return ranges


# ── Per-season fetchers ────────────────────────────────────────────────────────

def fetch_schedule(season):
    """All final regular season games for the given season."""
    start, end = SEASON_DATES[season]
    print(f'  Fetching {season} schedule ({start} → {end})...', flush=True)
    games = []
    for chunk_start, chunk_end in _month_ranges(start, end):
        r = _get(MLB_API + '/schedule', {
            'sportId': 1, 'gameType': 'R', 'season': season,
            'startDate': chunk_start, 'endDate': chunk_end,
            'hydrate': 'linescore,teams',
        }, f'schedule {chunk_start}')
        if not r:
            continue
        for day in r.json().get('dates', []):
            for g in day.get('games', []):
                if g.get('status', {}).get('abstractGameState', '') != 'Final':
                    continue
                h = g.get('teams', {}).get('home', {})
                a = g.get('teams', {}).get('away', {})
                h_score = h.get('score')
                a_score = a.get('score')
                if h_score is None or a_score is None:
                    continue
                games.append({
                    'season':     season,
                    'game_date':  day['date'],
                    'game_time':  g.get('gameDate', ''),
                    'home_id':    h['team']['id'],
                    'home_name':  h['team'].get('name', ''),
                    'away_id':    a['team']['id'],
                    'away_name':  a['team'].get('name', ''),
                    'home_score': int(h_score),
                    'away_score': int(a_score),
                    'home_won':   int(h_score) > int(a_score),
                })
        time.sleep(0.15)
    print(f'    → {len(games)} games', flush=True)
    return games


def fetch_team_pitching(season, as_of_date=None):
    """ERA, WHIP, K, BB, BF per team for the given season.

    as_of_date (optional, 'YYYY-MM-DD'): return point-in-time cumulative stats
    through the day BEFORE as_of_date (stats='byDateRange'), so a team's own
    game that day never leaks into its own pre-game snapshot. Verified against
    the live API — byDateRange returns a distinct, smaller sample than the
    full-season split for any date before season end.

    If omitted, returns stats='season' (full season-to-date-as-of-now) — used
    for already-completed seasons in bootstrap_2024.py's weight-fitting pass,
    where "as of now" and "as of any game date" are the same thing anyway.
    """
    if as_of_date:
        start, _ = SEASON_DATES[season]
        end = _day_before(as_of_date)
        if end < start:
            return {}  # no games played yet this season as of this date
        params = {'stats': 'byDateRange', 'season': season, 'group': 'pitching',
                  'sportId': 1, 'startDate': start, 'endDate': end}
        label = f'{season} team pitching thru {end}'
    else:
        params = {'stats': 'season', 'season': season, 'group': 'pitching', 'sportId': 1}
        label = f'{season} team pitching'
    r = _get(MLB_API + '/teams/stats', params, label)
    if not r:
        return {}
    result = {}
    for split in (r.json().get('stats') or [{}])[0].get('splits', []):
        tid = (split.get('team') or {}).get('id')
        s   = split.get('stat', {})
        if not tid:
            continue
        try:
            result[tid] = {
                'era':  float(s.get('era',  0) or 0),
                'whip': float(s.get('whip', 0) or 0),
                'k':    int(s.get('strikeOuts',   0) or 0),
                'bb':   int(s.get('baseOnBalls',  0) or 0),
                'bf':   int(s.get('battersFaced', 0) or 0),
                'hr':   int(s.get('homeRuns',     0) or 0),
                'ip':   float(s.get('inningsPitched', 0) or 0),
            }
        except (TypeError, ValueError):
            continue
    return result


def fetch_team_batting(season, as_of_date=None):
    """OPS, runs/game, K% per team for the given season.

    as_of_date (optional, 'YYYY-MM-DD'): same point-in-time semantics as
    fetch_team_pitching() — stats through the day before as_of_date.
    """
    if as_of_date:
        start, _ = SEASON_DATES[season]
        end = _day_before(as_of_date)
        if end < start:
            return {}
        params = {'stats': 'byDateRange', 'season': season, 'group': 'hitting',
                  'sportId': 1, 'startDate': start, 'endDate': end}
        label = f'{season} team batting thru {end}'
    else:
        params = {'stats': 'season', 'season': season, 'group': 'hitting', 'sportId': 1}
        label = f'{season} team batting'
    r = _get(MLB_API + '/teams/stats', params, label)
    if not r:
        return {}
    result = {}
    for split in (r.json().get('stats') or [{}])[0].get('splits', []):
        tid = (split.get('team') or {}).get('id')
        s   = split.get('stat', {})
        if not tid:
            continue
        try:
            games = max(int(s.get('gamesPlayed', 1) or 1), 1)
            pa    = max(int(s.get('plateAppearances', 1) or 1), 1)
            result[tid] = {
                'ops':     float(s.get('ops', 0) or 0),
                'runs_pg': round(int(s.get('runs', 0) or 0) / games, 2),
                'k_pct':   round(int(s.get('strikeOuts', 0) or 0) / pa, 4),
            }
        except Exception:
            continue
    return result


def fetch_team_xwoba(season, as_of_date=None):
    """PA-weighted offensive xwOBA from Savant (keyed by lowercase team abbrev).

    as_of_date (optional, 'YYYY-MM-DD'): point-in-time via xwoba_rolling's
    incremental day-by-day statcast_search cache — through the day before
    as_of_date. See xwoba_rolling.py's module docstring for why this needs
    day-by-day chunking (Savant's season leaderboard has no date param, and
    its pitch-level search silently caps ~25k rows/request).

    If omitted, falls back to Savant's season leaderboard (full season, used
    for already-completed seasons in bootstrap_2024.py's fitting pass, where
    "as of now" and "as of any game date" are the same thing anyway).
    """
    if as_of_date:
        import xwoba_rolling
        start, _ = SEASON_DATES[season]
        try:
            return xwoba_rolling.get_team_xwoba_asof(season, as_of_date, start)
        except Exception as e:
            print(f'  ! fetch_team_xwoba(as_of={as_of_date}) failed: {e}', flush=True)
            return {}

    r = _get(SAVANT_URL, {'type': 'batter', 'year': season, 'min': 1, 'csv': 'true'},
             f'{season} team xwOBA')
    if not r:
        return {}
    team_pa, team_sum = {}, {}
    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            team = (row.get('team_abbrev') or row.get('team') or '').lower().strip()
            pa   = int(row.get('pa', '0') or 0)
            # Try multiple column names Savant has used across years
            xw_raw = (row.get('est_woba') or row.get('xwoba') or
                      row.get('estimated_woba') or row.get('xwOBA') or '')
            if not team or pa < 1 or not xw_raw or xw_raw in ('', '.', 'null', 'NA'):
                continue
            xw = float(xw_raw)
            team_pa[team]  = team_pa.get(team, 0) + pa
            team_sum[team] = team_sum.get(team, 0.0) + xw * pa
        except Exception:
            continue
    return {t: round(team_sum[t] / team_pa[t], 4) for t in team_pa if team_pa[t] > 0}


def fetch_team_abbrevs():
    """team_id → lowercase abbreviation (stable across seasons)."""
    r = _get(MLB_API + '/teams', {'sportId': 1}, 'team abbrevs')
    if not r:
        return {}
    return {t['id']: t.get('abbreviation', '').lower()
            for t in r.json().get('teams', [])}


def fetch_team_xfip(season, as_of_date=None):
    """
    xFIP and SIERA per team from FanGraphs (keyed by lowercase team abbrev).
    Returns {team_abbr_lower: {'xfip': float|None, 'siera': float|None}}.
    Silent failure returns {}.

    as_of_date (optional, 'YYYY-MM-DD'): point-in-time via FanGraphs'
    month=1000 custom date-range split (startdate=season start, enddate=day
    before as_of_date) — verified against the live API to return a genuinely
    different, smaller aggregate than the full-season query, so this is a
    real point-in-time cut, not a no-op passthrough.

    Re-running bootstrap_2024.py will re-fetch and recalibrate these values.
    """
    try:
        import fangraphs_api
        if as_of_date:
            start, _ = SEASON_DATES[season]
            end = _day_before(as_of_date)
            if end < start:
                return {}
            return fangraphs_api.get_team_xfip_bootstrap(season, start_date=start, end_date=end)
        return fangraphs_api.get_team_xfip_bootstrap(season)
    except Exception as e:
        print(f'  ! fetch_team_xfip({season}) failed: {e}', flush=True)
        return {}


def fetch_team_pitcher_metrics(season):
    """
    K%, BB%, barrel rate, and whiff rate per team from Savant statcast leaderboard
    — aggregated to team level (PA-weighted when PA available, else averaged).

    Returns {team_abbr_lower: {'k_pct': float|None, 'bb_pct': float|None,
                               'barrel_pct': float|None, 'whiff_pct': float|None}}.
    Silent failure returns {}.
    """
    r = _get(
        SAVANT_STATCAST_URL,
        {'type': 'pitcher', 'year': season, 'position': '', 'min': 1, 'csv': 'true'},
        f'{season} pitcher statcast metrics',
    )
    if not r:
        return {}

    def _f(row, *cols):
        for c in cols:
            v = row.get(c, '')
            if v and v not in ('', '.', 'null', 'NA', '-'):
                try:
                    return float(v)
                except ValueError:
                    pass
        return None

    team_pa     = {}   # team_lower -> total PA
    team_k      = {}   # team_lower -> sum(k_pct * pa)
    team_bb     = {}   # team_lower -> sum(bb_pct * pa)
    team_barrel = {}   # team_lower -> sum(barrel_pct * pa)
    team_whiff  = {}   # team_lower -> sum(whiff_pct * pa)
    # Unweighted fallback accumulators
    team_count  = {}
    team_k2     = {}
    team_bb2    = {}
    team_barrel2 = {}
    team_whiff2  = {}

    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            team = (row.get('team_abbrev') or row.get('team') or '').lower().strip()
            if not team:
                continue
            # Skip multi-team rows (FanGraphs/Savant uses '- - -' or similar)
            if len(team) > 5:
                continue

            pa_raw = row.get('pa', '') or row.get('total_pa', '') or ''
            pa = int(pa_raw) if pa_raw and pa_raw not in ('', 'null') else 0

            k_pct      = _f(row, 'k_percent', 'k%', 'strikeout_percent')
            bb_pct     = _f(row, 'bb_percent', 'bb%', 'walk_percent')
            barrel_pct = _f(row, 'barrel_batted_rate', 'barrel_bip_rate', 'barrel_rate')
            whiff_pct  = _f(row, 'whiff_percent', 'whiff_pct',
                            'swinging_strike_pct', 'swstr_pct')

            if pa > 0:
                team_pa[team] = team_pa.get(team, 0) + pa
                if k_pct is not None:
                    team_k[team]      = team_k.get(team, 0.0) + k_pct * pa
                if bb_pct is not None:
                    team_bb[team]     = team_bb.get(team, 0.0) + bb_pct * pa
                if barrel_pct is not None:
                    team_barrel[team] = team_barrel.get(team, 0.0) + barrel_pct * pa
                if whiff_pct is not None:
                    team_whiff[team]  = team_whiff.get(team, 0.0) + whiff_pct * pa
            else:
                team_count[team] = team_count.get(team, 0) + 1
                if k_pct is not None:
                    team_k2[team]      = team_k2.get(team, 0.0) + k_pct
                if bb_pct is not None:
                    team_bb2[team]     = team_bb2.get(team, 0.0) + bb_pct
                if barrel_pct is not None:
                    team_barrel2[team] = team_barrel2.get(team, 0.0) + barrel_pct
                if whiff_pct is not None:
                    team_whiff2[team]  = team_whiff2.get(team, 0.0) + whiff_pct
        except Exception:
            continue

    result = {}
    all_teams = set(list(team_pa.keys()) + list(team_count.keys()))
    for team in all_teams:
        pa_total = team_pa.get(team, 0)
        if pa_total > 0:
            def _wavg(d, t=team, pa=pa_total):
                return round(d[t] / pa, 6) if t in d else None
            result[team] = {
                'k_pct':      _wavg(team_k),
                'bb_pct':     _wavg(team_bb),
                'barrel_pct': _wavg(team_barrel),
                'whiff_pct':  _wavg(team_whiff),
            }
        else:
            n = team_count.get(team, 0)
            def _avg(d, t=team, cnt=n):
                return round(d[t] / cnt, 6) if cnt and t in d else None
            result[team] = {
                'k_pct':      _avg(team_k2),
                'bb_pct':     _avg(team_bb2),
                'barrel_pct': _avg(team_barrel2),
                'whiff_pct':  _avg(team_whiff2),
            }

    return result


# ── Schedule fatigue ──────────────────────────────────────────────────────────

def build_fatigue_map(games):
    """
    Given a season's game list, compute schedule fatigue for every team entering
    every game. Processes chronologically so each game only sees games before it.

    Returns {(game_date, home_id, away_id): {'home': fatigue_dict, 'away': fatigue_dict}}

    fatigue_dict keys (match mlb_api._schedule_fatigue output):
      rest_days      – days since last game (0=B2B, None=first game of season)
      road_trip_len  – consecutive road games ending today (0 if home)
      homestand_len  – consecutive home games ending today (0 if away)
      tz_shift       – abs timezone-hour delta from team's home city (away only)
    """
    from datetime import datetime as _dt

    sorted_games = sorted(games, key=lambda g: (g['game_date'], g['home_id']))
    team_log = {}  # team_id -> [{'date': str, 'is_home': bool}, ...]
    result   = {}

    for g in sorted_games:
        h_id  = g['home_id']
        a_id  = g['away_id']
        gdate = g['game_date']
        gdate_obj = _dt.strptime(gdate, '%Y-%m-%d').date()

        key = (gdate, h_id, a_id)
        result[key] = {}

        for team_id, is_home in ((h_id, True), (a_id, False)):
            log = team_log.get(team_id, [])

            # Days of rest before this game (0 = played yesterday)
            rest_days = None
            if log:
                last = _dt.strptime(log[-1]['date'], '%Y-%m-%d').date()
                rest_days = max(0, (gdate_obj - last).days - 1)

            # Consecutive home/road streak including today
            road_trip_len = homestand_len = 0
            if is_home:
                for entry in reversed(log):
                    if entry['is_home']:
                        homestand_len += 1
                    else:
                        break
                homestand_len += 1
            else:
                for entry in reversed(log):
                    if not entry['is_home']:
                        road_trip_len += 1
                    else:
                        break
                road_trip_len += 1

            # Timezone shift from team's home city to today's venue (away only)
            tz_shift = 0
            if not is_home:
                tz_shift = abs(TEAM_TZ_OFFSET.get(team_id, 0) -
                               TEAM_TZ_OFFSET.get(h_id, 0))

            result[key]['home' if is_home else 'away'] = {
                'rest_days':     rest_days,
                'road_trip_len': road_trip_len,
                'homestand_len': homestand_len,
                'tz_shift':      tz_shift,
            }

        # Append to log AFTER computing so this game counts for the next one
        team_log.setdefault(h_id, []).append({'date': gdate, 'is_home': True})
        team_log.setdefault(a_id, []).append({'date': gdate, 'is_home': False})

    return result


def build_recent_rpg_map(games, window=15, team_era=None):
    """
    For each game, compute each team's rolling {window}-game opponent-adjusted
    runs-per-game using only games *before* this one (no lookahead bias).

    team_era: {team_id: era_float} — when provided, normalizes each game's runs
    scored by the opponent's ERA relative to league average so that 13 runs
    against Colorado (ERA ~6.5) counts less than 13 runs against an ace staff.
    Multiplier is capped to [0.60, 1.60] to avoid extreme adjustments.

    Returns {(game_date, home_id, away_id): {'home': float|None, 'away': float|None}}
    None when fewer than 5 prior games exist (matches mlb_api threshold).
    """
    sorted_games = sorted(games, key=lambda g: (g['game_date'], g['home_id']))
    team_runs = {}   # team_id -> [adj_runs_scored, ...] chronological
    result    = {}

    for g in sorted_games:
        h_id = g['home_id']
        a_id = g['away_id']
        key  = (g['game_date'], h_id, a_id)

        def _rpg(tid):
            recent = team_runs.get(tid, [])[-window:]
            return round(sum(recent) / len(recent), 2) if len(recent) >= 5 else None

        result[key] = {'home': _rpg(h_id), 'away': _rpg(a_id)}

        # Normalize runs by opponent ERA before appending
        h_runs = g.get('home_score', 0)
        a_runs = g.get('away_score', 0)
        if team_era:
            h_era  = team_era.get(h_id, LG_ERA)
            a_era  = team_era.get(a_id, LG_ERA)
            # home scored against away pitching; normalize by away team ERA
            h_mult = min(max(LG_ERA / max(a_era, 2.0), 0.60), 1.60)
            a_mult = min(max(LG_ERA / max(h_era, 2.0), 0.60), 1.60)
            h_runs = round(h_runs * h_mult, 2)
            a_runs = round(a_runs * a_mult, 2)

        team_runs.setdefault(h_id, []).append(h_runs)
        team_runs.setdefault(a_id, []).append(a_runs)

    return result


# ── Build training rows for one season ────────────────────────────────────────

def build_season_rows(season, team_abbrevs, import_model):
    print(f'\n── Season {season} ──────────────────────────────────────', flush=True)
    games      = fetch_schedule(season)
    team_pitch = fetch_team_pitching(season)
    team_bat   = fetch_team_batting(season)
    team_xwoba = fetch_team_xwoba(season)

    # Predictive pitcher metrics (xFIP/SIERA and Savant K%/BB%/barrel%/whiff%)
    # These are keyed by lowercase team abbreviation.
    # Silent failure returns {} so the model falls back to descriptive stats.
    xfip_by_abbrev    = fetch_team_xfip(season)
    metrics_by_abbrev = fetch_team_pitcher_metrics(season)

    n_pitch = len(team_pitch)
    n_xwoba = len(team_xwoba)
    n_xfip  = len(xfip_by_abbrev)
    n_met   = len(metrics_by_abbrev)
    print(f'  Team pitching: {n_pitch} teams  |  '
          f'Team batting: {len(team_bat)} teams  |  '
          f'xwOBA: {n_xwoba} teams {"(falling back to OPS)" if n_xwoba == 0 else ""}  |  '
          f'xFIP/SIERA: {n_xfip} teams  |  '
          f'Savant metrics: {n_met} teams',
          flush=True)

    xwoba_by_id = {tid: team_xwoba[abbrev]
                   for tid, abbrev in team_abbrevs.items()
                   if abbrev in team_xwoba}

    # Build tid → xfip/siera and tid → savant metrics using team_abbrevs map
    xfip_by_id    = {tid: xfip_by_abbrev[abbrev]
                     for tid, abbrev in team_abbrevs.items()
                     if abbrev in xfip_by_abbrev}
    metrics_by_id = {tid: metrics_by_abbrev[abbrev]
                     for tid, abbrev in team_abbrevs.items()
                     if abbrev in metrics_by_abbrev}

    fatigue_map  = build_fatigue_map(games)
    team_era     = {tid: info['era'] for tid, info in team_pitch.items() if info.get('era')}
    recent_rpg_map = build_recent_rpg_map(games, team_era=team_era)

    rows = []
    for g in games:
        h_id = g['home_id']
        a_id = g['away_id']
        fat  = fatigue_map.get((g['game_date'], h_id, a_id), {})
        rpg  = recent_rpg_map.get((g['game_date'], h_id, a_id), {})

        def team_pitcher(tid):
            p   = team_pitch.get(tid, {})
            xfp = xfip_by_id.get(tid, {})
            met = metrics_by_id.get(tid, {})
            return {
                'era':         p.get('era'),
                'x_era':       None,
                'xfip':        xfp.get('xfip'),
                'siera':       xfp.get('siera'),
                'whip':        p.get('whip'),
                'k':           p.get('k', 0),
                'bb':          p.get('bb', 0),
                'bf':          p.get('bf', 0),
                'k_pct':       met.get('k_pct'),
                'bb_pct':      met.get('bb_pct'),
                'barrel_pct':  met.get('barrel_pct'),
                'whiff_pct':   met.get('whiff_pct'),
                'last_starts': [],
            }

        def team_batting_dict(tid):
            b = team_bat.get(tid, {})
            return {
                'x_woba':  xwoba_by_id.get(tid),
                'ops':     b.get('ops'),
                'runs_pg': b.get('runs_pg'),
                'k_pct':   b.get('k_pct'),
            }

        home = {'id': h_id, 'form': [], 'split_w': 0, 'split_l': 0,
                'pitcher': team_pitcher(h_id), 'schedule_fatigue': fat.get('home'),
                'recent_rpg': rpg.get('home'),
                **team_batting_dict(h_id)}
        away = {'id': a_id, 'form': [], 'split_w': 0, 'split_l': 0,
                'pitcher': team_pitcher(a_id), 'schedule_fatigue': fat.get('away'),
                'recent_rpg': rpg.get('away'),
                **team_batting_dict(a_id)}

        result = import_model.predict(home, away, game_time_utc=g['game_time'])
        rows.append({
            'season':       season,
            'game_date':    g['game_date'],
            'home_team':    g['home_name'],
            'away_team':    g['away_name'],
            'home_won':     1 if g['home_won'] else 0,
            'home_prob':    result['home_prob'],
            'factors_json': json.dumps(result['factors']),
        })

    print(f'  → {len(rows)} training rows', flush=True)
    return rows


# ── Logistic regression refit ──────────────────────────────────────────────────

def refit(training_rows):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    seasons_covered = sorted({r['season'] for r in training_rows})

    factor_names = set()
    for row in training_rows:
        for label, _ in json.loads(row['factors_json']):
            factor_names.add(label)
    factor_names = sorted(factor_names)

    X, y = [], []
    for row in training_rows:
        fdict = {f: 0.0 for f in factor_names}
        for label, contrib in json.loads(row['factors_json']):
            fdict[label] = float(contrib)
        X.append([fdict[f] for f in factor_names])
        y.append(float(row['home_won']))
    X = np.array(X)
    y = np.array(y)

    home_win_rate  = y.mean()
    baseline_brier = brier_score_loss(y, [home_win_rate] * len(y))
    current_probs  = [r['home_prob'] for r in training_rows]
    current_brier  = brier_score_loss(y, current_probs)
    current_acc    = sum(1 for p, a in zip(current_probs, y)
                         if (p >= 0.5) == (a >= 0.5)) / len(y)

    lr = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000, solver='lbfgs')
    lr.fit(X, y)

    refit_probs = lr.predict_proba(X)[:, 1]
    refit_brier = brier_score_loss(y, refit_probs)
    refit_acc   = sum(1 for p, a in zip(refit_probs, y)
                      if (p >= 0.5) == (a >= 0.5)) / len(y)

    # Per-season breakdown
    season_stats = {}
    for season in seasons_covered:
        mask = [r['season'] == season for r in training_rows]
        y_s = np.array([a for a, m in zip(y, mask) if m])
        p_s = [p for p, m in zip(current_probs, mask) if m]
        season_stats[season] = {
            'n':    len(y_s),
            'hwr':  round(float(y_s.mean()), 3),
            'brier': round(float(brier_score_loss(y_s, p_s)), 4),
            'acc':  round(sum(1 for p, a in zip(p_s, y_s)
                              if (p >= 0.5) == (a >= 0.5)) / len(y_s), 3),
        }

    W = 72
    seasons_str = ' + '.join(str(s) for s in seasons_covered)
    print('\n' + '═' * W)
    print(f'  Bootstrap Refit  —  {seasons_str}  —  {len(training_rows):,} total games')
    print('═' * W)

    for s, st in season_stats.items():
        print(f'  {s}: {st["n"]:,} games  |  home win rate {st["hwr"]*100:.1f}%  '
              f'|  current model {st["brier"]:.4f} Brier ({st["acc"]*100:.1f}% acc)')

    print(f'\n  Combined home win rate: {home_win_rate*100:.1f}%')
    print()
    print(f'  Brier score  (↓ better, 0.25 = pure random):')
    print(f'    Baseline (always {home_win_rate*100:.0f}%):    {baseline_brier:.4f}')
    print(f'    Current model:             {current_brier:.4f}  ({current_acc*100:.1f}% pick accuracy)')
    print(f'    Refit (in-sample):         {refit_brier:.4f}  ({refit_acc*100:.1f}% pick accuracy)')
    print(f'    Improvement:               {(current_brier-refit_brier)*1000:+.1f} mBrier')
    print()
    print('  NOTE: In-sample refit is optimistic (~0.5–1% overfit typical).')
    print('  Validate against live 2026 predictions on the Model page.')
    print('─' * W)
    print(f'  {"Factor":<36}  {"Coeff":>7}  Suggested action')
    print('─' * W)

    # Rolling factors: bootstrap always passes empty arrays for these, so they
    # contribute 0.0 to every training row → coefficient is meaningless (not 0 = bad).
    # These labels must match what mlb_model.predict() actually emits (using team abbr).
    # The pattern below matches any label ending with the known rolling suffixes.
    # New calibratable factors (xFIP, SIERA, K%, BB%, barrel%, whiff%) are NOT listed
    # here — they have real variance in the bootstrap data and can be calibrated.
    _ROLLING_SUFFIXES = {
        'Form L10',
        'H/R Record',
        'SP Recent ERA',
        'Pen ERA',
        'Closer Yday',
        'Pen Fatigue',
        'SP Rest',
        'B2B',
    }

    def _is_rolling(name):
        for suffix in _ROLLING_SUFFIXES:
            if name.endswith(suffix):
                return True
        return False

    coefficients = dict(zip(factor_names, lr.coef_[0]))
    for name, coef in sorted(coefficients.items(),
                              key=lambda x: abs(x[1] - 1.0), reverse=True):
        if _is_rolling(name):
            action = '↻  rolling — bootstrap blind, hand-tuned weight kept'
        else:
            delta = coef - 1.0
            if abs(delta) > 0.35:
                action = '★  INCREASE weight' if delta > 0 else '★  REDUCE weight'
            elif abs(delta) > 0.15:
                action = '↑  increase weight' if delta > 0 else '↓  reduce weight'
            elif abs(delta) > 0.05:
                action = '~  slight adjust'
            else:
                action = '✓  well-calibrated'
        print(f'  {name:<36}  {coef:>+7.3f}  {action}')

    print('─' * W)
    print()
    print('  ↻  Rolling factors: bootstrap passes empty arrays → coeff is always 0,')
    print('     NOT a signal to remove. Weights are hand-tuned; validate on Model page.')
    print('  Not listed: factor absent from bootstrap training set (e.g. injury, H2H).')
    print('═' * W)

    out = {
        'seasons':          seasons_covered,
        'n_games':          len(training_rows),
        'home_win_rate':    round(float(home_win_rate), 4),
        'brier_baseline':   round(float(baseline_brier), 4),
        'brier_current':    round(float(current_brier), 4),
        'brier_refit':      round(float(refit_brier), 4),
        'accuracy_current': round(float(current_acc), 4),
        'accuracy_refit':   round(float(refit_acc), 4),
        'per_season':       season_stats,
        'coefficients':     {k: round(float(v), 4) for k, v in coefficients.items()},
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refit_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Full results → refit_results.json')
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    seasons_str = ' + '.join(str(s) for s in SEASONS)
    print('═' * 62)
    print(f'  MLB Model Bootstrap — {seasons_str}')
    print('═' * 62)
    print()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mlb_model

    team_abbrevs = fetch_team_abbrevs()
    print(f'  Team abbrev map: {len(team_abbrevs)} teams', flush=True)

    import datetime as _dt
    current_year = _dt.date.today().year

    all_rows = []
    for season in SEASONS:
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f'bootstrap_training_{season}.json')
        # Never cache in-progress seasons — always re-fetch so today's results count
        use_cache = (season < current_year) and os.path.exists(cache_path)
        if use_cache:
            with open(cache_path) as f:
                rows = json.load(f)
            print(f'\n── Season {season} ──────────────────────────────────────')
            print(f'  Loaded from cache: {len(rows):,} rows  '
                  f'(delete bootstrap_training_{season}.json to re-fetch)')
        else:
            rows = build_season_rows(season, team_abbrevs, mlb_model)
            if season < current_year:
                with open(cache_path, 'w') as f:
                    json.dump(rows, f)
                print(f'  Cached → bootstrap_training_{season}.json')
            else:
                print(f'  (2026 in-progress — not cached)')
        all_rows.extend(rows)

    combined_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'bootstrap_training.json')
    with open(combined_path, 'w') as f:
        json.dump(all_rows, f)
    print(f'\n  Combined: {len(all_rows):,} total rows → bootstrap_training.json')

    refit(all_rows)


if __name__ == '__main__':
    combined_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'bootstrap_training.json')
    if '--refit-only' in sys.argv and os.path.exists(combined_path):
        print('Loading cached combined training data...')
        with open(combined_path) as f:
            rows = json.load(f)
        print(f'  {len(rows):,} rows from seasons: '
              f'{sorted({r["season"] for r in rows})}')
        refit(rows)
    else:
        main()
