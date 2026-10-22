"""
Static per-team stadium reference data for NFL — coordinates, surface, and
home timezone. Nothing in this codebase has this today (team_catalog.py
only fetches display NAMES live from ESPN, nothing about venues;
weather_api.py's NFL_OUTDOOR dict has lat/lon for ~17 outdoor stadiums only,
keyed by a venue-name substring, no surface/timezone/indoor entries).

Keyed by ESPN team abbreviation (what every team dict in this app already
carries as team['abbr']) rather than ESPN's numeric team id, since that's
the field actually available to callers without an extra lookup.

Used by the game-detail page's "situation" section: travel distance/
timezone change (haversine from the away team's home stadium to the game
venue) and surface note. Public, essentially-static facts (stadiums rarely
move) — update on a relocation/renovation, not on a schedule.
"""
import math

# (city, lat, lon, surface, dome, timezone)
#   surface: 'grass' | 'turf'
#   dome:    True if a fully enclosed or typically-closed roof (no real
#            weather exposure) — retractable roofs that are usually closed
#            for the sport's season count as True here; Seattle/Vegas-style
#            partial covers that still expose the field count as False.
TEAMS = {
    'ARI': ('Glendale, AZ',        33.5276, -112.2626, 'grass', True,  'America/Phoenix'),
    'ATL': ('Atlanta, GA',         33.7554,  -84.4008, 'turf',  True,  'America/New_York'),
    'BAL': ('Baltimore, MD',       39.2780,  -76.6227, 'grass', False, 'America/New_York'),
    'BUF': ('Orchard Park, NY',    42.7738,  -78.7870, 'turf',  False, 'America/New_York'),
    'CAR': ('Charlotte, NC',       35.2258,  -80.8528, 'grass', False, 'America/New_York'),
    'CHI': ('Chicago, IL',         41.8623,  -87.6167, 'grass', False, 'America/Chicago'),
    'CIN': ('Cincinnati, OH',      39.0955,  -84.5160, 'turf',  False, 'America/New_York'),
    'CLE': ('Cleveland, OH',       41.5061,  -81.6995, 'grass', False, 'America/New_York'),
    'DAL': ('Arlington, TX',       32.7473,  -97.0945, 'turf',  True,  'America/Chicago'),
    'DEN': ('Denver, CO',          39.7439, -105.0201, 'grass', False, 'America/Denver'),
    'DET': ('Detroit, MI',         42.3400,  -83.0456, 'turf',  True,  'America/New_York'),
    'GB':  ('Green Bay, WI',       44.5013,  -88.0622, 'grass', False, 'America/Chicago'),
    'HOU': ('Houston, TX',         29.6847,  -95.4107, 'turf',  True,  'America/Chicago'),
    'IND': ('Indianapolis, IN',    39.7601,  -86.1639, 'turf',  True,  'America/New_York'),
    'JAX': ('Jacksonville, FL',    30.3239,  -81.6373, 'grass', False, 'America/New_York'),
    'KC':  ('Kansas City, MO',     39.0489,  -94.4839, 'grass', False, 'America/Chicago'),
    'LV':  ('Las Vegas, NV',       36.0909, -115.1833, 'turf',  True,  'America/Los_Angeles'),
    'LAC': ('Inglewood, CA',       33.9535, -118.3392, 'turf',  True,  'America/Los_Angeles'),
    'LAR': ('Inglewood, CA',       33.9535, -118.3392, 'turf',  True,  'America/Los_Angeles'),
    'MIA': ('Miami Gardens, FL',   25.9580,  -80.2389, 'grass', False, 'America/New_York'),
    'MIN': ('Minneapolis, MN',     44.9735,  -93.2575, 'turf',  True,  'America/Chicago'),
    'NE':  ('Foxborough, MA',      42.0909,  -71.2643, 'turf',  False, 'America/New_York'),
    'NO':  ('New Orleans, LA',     29.9511,  -90.0812, 'turf',  True,  'America/Chicago'),
    'NYG': ('East Rutherford, NJ', 40.8128,  -74.0742, 'turf',  False, 'America/New_York'),
    'NYJ': ('East Rutherford, NJ', 40.8128,  -74.0742, 'turf',  False, 'America/New_York'),
    'PHI': ('Philadelphia, PA',    39.9008,  -75.1675, 'grass', False, 'America/New_York'),
    'PIT': ('Pittsburgh, PA',      40.4468,  -80.0158, 'grass', False, 'America/New_York'),
    'SF':  ('Santa Clara, CA',     37.4030, -121.9700, 'grass', False, 'America/Los_Angeles'),
    'SEA': ('Seattle, WA',         47.5952, -122.3316, 'turf',  False, 'America/Los_Angeles'),
    'TB':  ('Tampa, FL',           27.9759,  -82.5033, 'grass', False, 'America/New_York'),
    'TEN': ('Nashville, TN',       36.1665,  -86.7713, 'grass', False, 'America/Chicago'),
    'WSH': ('Landover, MD',        38.9076,  -76.8645, 'grass', False, 'America/New_York'),
}


def get(abbr):
    """Returns a dict for a team abbreviation, or None if not in the table
    (e.g. an unrecognized/renamed abbreviation — callers should hide the
    situation fields that need this rather than guess)."""
    row = TEAMS.get((abbr or '').upper())
    if not row:
        return None
    city, lat, lon, surface, dome, tz = row
    return {'city': city, 'lat': lat, 'lon': lon, 'surface': surface, 'dome': dome, 'tz': tz}


def _haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8  # Earth radius, miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def travel_miles(away_abbr, home_abbr):
    """One-way distance, in miles, from the away team's home stadium to the
    home team's stadium (the game venue, for all but international-site
    games, which this doesn't special-case). None if either team is
    unrecognized."""
    a, h = get(away_abbr), get(home_abbr)
    if not a or not h:
        return None
    return round(_haversine_miles(a['lat'], a['lon'], h['lat'], h['lon']))


def timezone_change_hours(away_abbr, home_abbr, at_utc):
    """Hours the away team is shifting across (positive = traveling east,
    e.g. West Coast team flying to the East Coast), evaluated at `at_utc`
    (a timezone-aware datetime) so DST is handled correctly. None if either
    team is unrecognized."""
    from zoneinfo import ZoneInfo
    a, h = get(away_abbr), get(home_abbr)
    if not a or not h:
        return None
    away_off = at_utc.astimezone(ZoneInfo(a['tz'])).utcoffset()
    home_off = at_utc.astimezone(ZoneInfo(h['tz'])).utcoffset()
    return round((home_off - away_off).total_seconds() / 3600)
