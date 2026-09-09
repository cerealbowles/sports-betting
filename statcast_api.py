"""
Baseball Savant / Statcast data — free CSV leaderboard endpoint.

Provides xERA (expected ERA based on quality of contact) per pitcher,
which is far more predictive than raw ERA over small samples.

Data is cached 6 hours; failures fall back to stale cache silently.
"""
import csv
import io
import time
import requests
from datetime import datetime

_cache = {}
_TTL = 21600  # 6 hours

_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; sports-betting-tracker/1.0)'}


def _year():
    return datetime.utcnow().year


def get_pitcher_statcast():
    """
    Returns {mlbam_id: {'x_era': float|None, 'x_woba': float|None}}
    for all pitchers with at least 1 PA in the current season.
    xERA = expected ERA based on exit velocity / launch angle (not luck).
    """
    year = _year()
    key = f'sc_pitcher_{year}'
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
            return data

    url = (
        f'https://baseballsavant.mlb.com/leaderboard/expected_statistics'
        f'?type=pitcher&year={year}&min=1&csv=true'
    )
    try:
        r = requests.get(url, timeout=20, headers=_HEADERS)
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        result = {}
        for row in reader:
            try:
                pid_raw = row.get('player_id') or row.get('pitcher') or ''
                pid = int(pid_raw)
                if not pid:
                    continue

                def _f(col, *alts):
                    for c in (col,) + alts:
                        v = row.get(c, '')
                        if v and v not in ('', '.', 'null', 'NA'):
                            try:
                                return float(v)
                            except ValueError:
                                pass
                    return None

                result[pid] = {
                    'x_era':  _f('est_era', 'xera', 'xERA'),
                    'x_woba': _f('est_woba', 'xwoba', 'xwOBA'),
                }
            except Exception:
                continue
        _cache[key] = (result, now)
        return result
    except Exception:
        return _cache.get(key, ({}, 0))[0]


def get_team_statcast():
    """
    Returns {team_abbr_lower: {'x_woba': float|None, 'barrel_pct': float|None}}
    for all teams — offensive Statcast quality.
    """
    year = _year()
    key = f'sc_team_{year}'
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
            return data

    url = (
        f'https://baseballsavant.mlb.com/leaderboard/expected_statistics'
        f'?type=batter&year={year}&min=1&csv=true'
    )
    # Aggregate individual batter xwOBA → team xwOBA (PA-weighted)
    try:
        r = requests.get(url, timeout=20, headers=_HEADERS)
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        team_pa   = {}
        team_woba_sum = {}
        for row in reader:
            try:
                team = (row.get('team_abbrev') or row.get('team') or '').lower().strip()
                if not team:
                    continue
                pa_raw = row.get('pa', '0') or '0'
                pa = int(pa_raw)
                if pa < 1:
                    continue
                woba_raw = row.get('est_woba') or row.get('xwoba') or ''
                if not woba_raw or woba_raw in ('', '.', 'null'):
                    continue
                woba = float(woba_raw)
                team_pa[team] = team_pa.get(team, 0) + pa
                team_woba_sum[team] = team_woba_sum.get(team, 0.0) + woba * pa
            except Exception:
                continue
        result = {}
        for team, pa_total in team_pa.items():
            if pa_total > 0:
                result[team] = {'x_woba': round(team_woba_sum[team] / pa_total, 4)}
        _cache[key] = (result, now)
        return result
    except Exception:
        return _cache.get(key, ({}, 0))[0]


def get_pitcher_metrics(year=None):
    """
    Returns {mlbam_id (int): {'barrel_pct': float|None, 'hard_hit_pct': float|None,
                               'avg_ev': float|None}}
    from the Savant statcast pitcher leaderboard.

    K%, BB%, SwStr% are sourced from FanGraphs (fangraphs_api) which has
    better coverage and cleaner data for those metrics.

    Note: Savant CSVs have a UTF-8 BOM that shifts column parsing — strip it
    before parsing so player_id correctly contains the MLBAM ID.
    """
    year = year or _year()
    key = f'sc_pitcher_metrics_{year}'
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
            return data

    url = (
        f'https://baseballsavant.mlb.com/leaderboard/statcast'
        f'?type=pitcher&year={year}&position=&min=1&csv=true'
    )
    try:
        r = requests.get(url, timeout=20, headers=_HEADERS)
        r.raise_for_status()
        # Strip BOM — Savant CSVs start with ﻿ which shifts column parsing
        text = r.text.lstrip('﻿')
        reader = csv.DictReader(io.StringIO(text))
        result = {}
        for row in reader:
            try:
                pid_raw = row.get('player_id', '')
                if not pid_raw:
                    continue
                pid = int(pid_raw)
                if pid <= 0:
                    continue

                def _f(*cols):
                    for c in cols:
                        v = (row.get(c) or '').strip()
                        if v and v not in ('.', 'null', 'NA', '-'):
                            try:
                                return float(v)
                            except ValueError:
                                pass
                    return None

                # Barrel rate: 'brl_percent' on 0-100 scale → convert to 0-1
                brl_raw = _f('brl_percent', 'brl_pa', 'barrel_batted_rate')
                barrel_pct = round(brl_raw / 100.0, 4) if brl_raw is not None else None

                hard_raw = _f('ev95percent', 'hard_hit_percent', 'hard_hit_pct')
                hard_pct = round(hard_raw / 100.0, 4) if hard_raw is not None else None

                result[pid] = {
                    'barrel_pct':   barrel_pct,
                    'hard_hit_pct': hard_pct,
                    'avg_ev':       _f('avg_hit_speed', 'exit_velocity_avg'),
                }
            except Exception:
                continue
        if result:
            _cache[key] = (result, now)
        return result if result else _cache.get(key, ({}, 0))[0]
    except Exception:
        return _cache.get(key, ({}, 0))[0]


def get_team_batting_splits():
    """
    Returns {team_abbr_lower: {'xwoba_vs_lhp': float|None, 'xwoba_vs_rhp': float|None}}
    — PA-weighted xwOBA split by opposing pitcher handedness.

    Baseball Savant expected_statistics endpoint supports a pitcherThrows filter.
    Falls back to stale cache silently on any failure.
    """
    year = _year()
    key = f'sc_batting_splits_{year}'
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
            return data

    result = {}
    for hand, field in (('L', 'xwoba_vs_lhp'), ('R', 'xwoba_vs_rhp')):
        url = (
            f'https://baseballsavant.mlb.com/leaderboard/expected_statistics'
            f'?type=batter&year={year}&min=1&pitcherThrows={hand}&csv=true'
        )
        try:
            r = requests.get(url, timeout=20, headers=_HEADERS)
            r.raise_for_status()
            reader = csv.DictReader(io.StringIO(r.text))
            team_pa = {}
            team_woba_sum = {}
            for row in reader:
                try:
                    team = (row.get('team_abbrev') or row.get('team') or '').lower().strip()
                    if not team:
                        continue
                    pa = int(row.get('pa', '0') or '0')
                    if pa < 1:
                        continue
                    woba_raw = row.get('est_woba') or row.get('xwoba') or ''
                    if not woba_raw or woba_raw in ('', '.', 'null'):
                        continue
                    woba = float(woba_raw)
                    team_pa[team] = team_pa.get(team, 0) + pa
                    team_woba_sum[team] = team_woba_sum.get(team, 0.0) + woba * pa
                except Exception:
                    continue
            for team, pa_total in team_pa.items():
                if pa_total > 0:
                    result.setdefault(team, {})[field] = round(team_woba_sum[team] / pa_total, 4)
        except Exception:
            pass

    _cache[key] = (result, now)
    return result
