"""
FanGraphs pitcher metrics — xFIP, SIERA, K%, BB%, SwStr%.

These estimators are far more predictive than ERA for future performance:
  SIERA  – best overall ERA estimator; K%, BB%, GB% in a non-linear model
  xFIP   – HR/FB-normalized FIP; strips park and strand-rate luck
  K%     – strikeout rate; the most stable, skill-based pitching metric
  BB%    – walk rate; command signal, negatively predictive
  SwStr% – swinging-strike rate; pitch-quality signal independent of results

Endpoint:
    GET https://www.fangraphs.com/api/leaders/major-league/data
    params: pos=all, stats=pit, lg=all, qual=0, pageitems=5000, season={year},
            month=0, type=8, team=0, ind=0

Response: JSON {"data": [...]} where each row has Name (HTML-wrapped),
Team (HTML-wrapped), xMLBAMID (MLBAM player ID), xFIP, SIERA, K%, BB%, SwStr%.

We key results by xMLBAMID (int) for direct lookup — no name matching needed.

Cache TTL: 6 hours. Silent failure returns stale cache or {}.
"""
import re
import time
import requests
from datetime import datetime

_cache = {}
_TTL = 21600  # 6 hours

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; sports-betting-tracker/1.0)',
    'Accept': 'application/json',
    'Referer': 'https://www.fangraphs.com/',
}

_FG_URL = 'https://www.fangraphs.com/api/leaders/major-league/data'

_HTML_TAG = re.compile(r'<[^>]+>')


def _strip_html(s):
    return _HTML_TAG.sub('', s or '').strip()


def _year():
    return datetime.utcnow().year


def _safe(v):
    """Float or None. Handles '-', '', None, HTML."""
    if v is None:
        return None
    s = _strip_html(str(v)).strip()
    if s in ('', '-', 'null', 'NA', 'N/A'):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _pct(v):
    """Float on 0-1 scale, or None. FanGraphs K%/BB% may come as 0.28 or 28.0."""
    f = _safe(v)
    if f is None:
        return None
    return round(f / 100.0 if f > 1.0 else f, 4)


def _fetch_raw(year, start_date=None, end_date=None):
    """
    start_date/end_date ('YYYY-MM-DD'), if both given, request FanGraphs'
    custom date-range split (month=1000 + startdate/enddate) instead of the
    full season (month=0). Verified against the live API: a bounded range
    returns a materially smaller, genuinely different aggregate than the
    full-season query (tested 41.2 IP vs 112 IP for the same pitcher over
    the same partial season) — this is a real point-in-time cut, not a
    no-op param.
    """
    params = {
        'pos': 'all', 'stats': 'pit', 'lg': 'all', 'qual': '0',
        'pageitems': '5000', 'pagenum': '1', 'season': str(year),
        'type': '8', 'team': '0', 'ind': '0',
    }
    if start_date and end_date:
        params['month'] = '1000'
        params['startdate'] = start_date
        params['enddate'] = end_date
    else:
        params['month'] = '0'
    try:
        r = requests.get(_FG_URL, params=params, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get('data', [])
    except Exception:
        return []


def get_pitcher_xfip(year=None):
    """
    Returns {mlbam_id (int): {'xfip', 'siera', 'fip', 'k_pct', 'bb_pct', 'swstr_pct'}}

    Keyed by MLBAM player ID via FanGraphs xMLBAMID field — no name matching needed.
    Used in mlb_api._build_team_info for individual starter lookups.
    """
    year = year or _year()
    key = f'fg_pitcher_{year}'
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
            return data

    rows = _fetch_raw(year)
    result = {}
    for row in rows:
        try:
            mlbam = row.get('xMLBAMID')
            if not mlbam:
                continue
            mid = int(mlbam)
            if mid <= 0:
                continue
            result[mid] = {
                'xfip':      _safe(row.get('xFIP')),
                'siera':     _safe(row.get('SIERA')),
                'fip':       _safe(row.get('FIP')),
                'k_pct':     _pct(row.get('K%')),
                'bb_pct':    _pct(row.get('BB%')),
                'swstr_pct': _pct(row.get('SwStr%')),
            }
        except Exception:
            continue

    if result:
        _cache[key] = (result, now)
    return result if result else _cache.get(key, ({}, 0))[0]


def get_team_xfip(year=None, start_date=None, end_date=None):
    """
    Returns {team_abbr_lower: {'xfip', 'siera', 'k_pct', 'bb_pct', 'swstr_pct'}}
    IP-weighted across all pitchers for that team. Used in bootstrap.

    start_date/end_date: point-in-time date-bounded aggregate (see
    _fetch_raw's docstring). Cached separately per date range so a
    point-in-time call never collides with (or gets served) the full-season
    cache entry.
    """
    year = year or _year()
    key = (f'fg_team_{year}' if not (start_date and end_date)
           else f'fg_team_{year}_{start_date}_{end_date}')
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < _TTL:
            return data

    rows = _fetch_raw(year, start_date, end_date)

    ip_sum     = {}
    xfip_sum   = {}
    siera_sum  = {}
    k_sum      = {}
    bb_sum     = {}
    swstr_sum  = {}

    for row in rows:
        try:
            team = _strip_html(row.get('Team') or '').lower()
            # Skip multi-team rows — FanGraphs uses '- - -' or similar
            if not team or len(team) > 5 or '-' in team:
                continue

            ip    = _safe(row.get('IP')) or 0.0
            xfip  = _safe(row.get('xFIP'))
            siera = _safe(row.get('SIERA'))
            k     = _pct(row.get('K%'))
            bb    = _pct(row.get('BB%'))
            swstr = _pct(row.get('SwStr%'))

            if ip <= 0:
                continue
            ip_sum[team] = ip_sum.get(team, 0.0) + ip
            if xfip  is not None: xfip_sum[team]  = xfip_sum.get(team,  0.0) + xfip  * ip
            if siera is not None: siera_sum[team] = siera_sum.get(team, 0.0) + siera * ip
            if k     is not None: k_sum[team]     = k_sum.get(team,     0.0) + k     * ip
            if bb    is not None: bb_sum[team]    = bb_sum.get(team,    0.0) + bb    * ip
            if swstr is not None: swstr_sum[team] = swstr_sum.get(team, 0.0) + swstr * ip
        except Exception:
            continue

    result = {}
    for team, ip_total in ip_sum.items():
        if ip_total <= 0:
            continue
        result[team] = {
            'xfip':      round(xfip_sum[team]  / ip_total, 3) if team in xfip_sum  else None,
            'siera':     round(siera_sum[team] / ip_total, 3) if team in siera_sum else None,
            'k_pct':     round(k_sum[team]     / ip_total, 4) if team in k_sum     else None,
            'bb_pct':    round(bb_sum[team]    / ip_total, 4) if team in bb_sum    else None,
            'swstr_pct': round(swstr_sum[team] / ip_total, 4) if team in swstr_sum else None,
        }

    if result:
        _cache[key] = (result, now)
    return result if result else _cache.get(key, ({}, 0))[0]


def get_team_xfip_bootstrap(season, start_date=None, end_date=None):
    """Thin wrapper for bootstrap_2024.py."""
    return get_team_xfip(year=season, start_date=start_date, end_date=end_date)
