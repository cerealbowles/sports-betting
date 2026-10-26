"""
Static per-team stadium reference data — coordinates, surface, and home
timezone — for NFL and MLB. Nothing in this codebase has this today
(team_catalog.py only fetches display NAMES live from ESPN/MLB Stats API,
nothing about venues; weather_api.py's MLB_OUTDOOR/NFL_OUTDOOR dicts have
lat/lon for outdoor stadiums only, keyed by a venue-name substring, no
surface/timezone/indoor entries, and skip domed parks entirely).

NFL rows are keyed by ESPN team abbreviation (what every NFL team dict in
this app already carries as team['abbr']); MLB rows are keyed by the MLB
Stats API's own abbreviation (same convention mlb_api.py's team dicts use,
e.g. 'AZ' not 'ARI', 'CWS' not 'CHW', 'ATH' for the Athletics) — these are
each sport's existing convention, not ESPN's numeric team id, since that's
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
NFL_TEAMS = {
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

# Back-compat alias — existing NFL callers (nfl_game_detail.py) import this
# module-level name directly via team_venues.get()'s default sport.
TEAMS = NFL_TEAMS

MLB_TEAMS = {
    'ATH': ('West Sacramento, CA',  38.5805, -121.5144, 'grass', False, 'America/Los_Angeles'),
    'ATL': ('Atlanta, GA',          33.8908,  -84.4678, 'grass', False, 'America/New_York'),
    'AZ':  ('Phoenix, AZ',          33.4455, -112.0667, 'turf',  True,  'America/Phoenix'),
    'BAL': ('Baltimore, MD',        39.2838,  -76.6217, 'grass', False, 'America/New_York'),
    'BOS': ('Boston, MA',           42.3467,  -71.0972, 'grass', False, 'America/New_York'),
    'CHC': ('Chicago, IL',          41.9484,  -87.6553, 'grass', False, 'America/Chicago'),
    'CIN': ('Cincinnati, OH',       39.0979,  -84.5086, 'grass', False, 'America/New_York'),
    'CLE': ('Cleveland, OH',        41.4962,  -81.6852, 'grass', False, 'America/New_York'),
    'COL': ('Denver, CO',           39.7559, -104.9942, 'grass', False, 'America/Denver'),
    'CWS': ('Chicago, IL',          41.8299,  -87.6338, 'grass', False, 'America/Chicago'),
    'DET': ('Detroit, MI',          42.3390,  -83.0485, 'grass', False, 'America/New_York'),
    'HOU': ('Houston, TX',          29.7573,  -95.3555, 'grass', True,  'America/Chicago'),
    'KC':  ('Kansas City, MO',      39.0517,  -94.4803, 'grass', False, 'America/Chicago'),
    'LAA': ('Anaheim, CA',          33.8003, -117.8827, 'grass', False, 'America/Los_Angeles'),
    'LAD': ('Los Angeles, CA',      34.0739, -118.2400, 'grass', False, 'America/Los_Angeles'),
    'MIA': ('Miami, FL',            25.7781,  -80.2197, 'turf',  True,  'America/New_York'),
    'MIL': ('Milwaukee, WI',        43.0280,  -87.9712, 'grass', True,  'America/Chicago'),
    'MIN': ('Minneapolis, MN',      44.9817,  -93.2777, 'grass', False, 'America/Chicago'),
    'NYM': ('Queens, NY',           40.7571,  -73.8458, 'grass', False, 'America/New_York'),
    'NYY': ('Bronx, NY',            40.8296,  -73.9262, 'grass', False, 'America/New_York'),
    'PHI': ('Philadelphia, PA',     39.9061,  -75.1665, 'grass', False, 'America/New_York'),
    'PIT': ('Pittsburgh, PA',       40.4469,  -80.0057, 'grass', False, 'America/New_York'),
    'SD':  ('San Diego, CA',        32.7073, -117.1566, 'grass', False, 'America/Los_Angeles'),
    'SEA': ('Seattle, WA',          47.5914, -122.3325, 'grass', True,  'America/Los_Angeles'),
    'SF':  ('San Francisco, CA',    37.7786, -122.3893, 'grass', False, 'America/Los_Angeles'),
    'STL': ('St. Louis, MO',        38.6226,  -90.1928, 'grass', False, 'America/Chicago'),
    'TB':  ('St. Petersburg, FL',   27.7683,  -82.6534, 'turf',  True,  'America/New_York'),
    'TEX': ('Arlington, TX',        32.7473,  -97.0817, 'grass', True,  'America/Chicago'),
    'TOR': ('Toronto, ON',          43.6414,  -79.3894, 'turf',  True,  'America/Toronto'),
    'WSH': ('Washington, DC',       38.8730,  -77.0074, 'grass', False, 'America/New_York'),
}

_BY_SPORT = {'nfl': NFL_TEAMS, 'mlb': MLB_TEAMS}


def get(abbr, sport='nfl'):
    """Returns a dict for a team abbreviation, or None if not in the table
    (e.g. an unrecognized/renamed abbreviation — callers should hide the
    situation fields that need this rather than guess)."""
    row = _BY_SPORT.get(sport, NFL_TEAMS).get((abbr or '').upper())
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


def travel_miles(away_abbr, home_abbr, sport='nfl'):
    """One-way distance, in miles, from the away team's home stadium to the
    home team's stadium (the game venue, for all but international-site
    games, which this doesn't special-case). None if either team is
    unrecognized."""
    a, h = get(away_abbr, sport), get(home_abbr, sport)
    if not a or not h:
        return None
    return round(_haversine_miles(a['lat'], a['lon'], h['lat'], h['lon']))


# Full display-name → abbreviation, for callers that only have the team
# name stored (e.g. GamePrediction.home_team/away_team, saved as ESPN's
# displayName for NFL and the MLB Stats API's team name for MLB — neither
# persists the abbreviation itself). Keys match those APIs' own display
# names exactly; used by abbr_from_name() below.
NFL_NAME_TO_ABBR = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WSH',
}

MLB_NAME_TO_ABBR = {
    'Athletics': 'ATH', 'Atlanta Braves': 'ATL', 'Arizona Diamondbacks': 'AZ',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
    'Chicago White Sox': 'CWS', 'Detroit Tigers': 'DET', 'Houston Astros': 'HOU',
    'Kansas City Royals': 'KC', 'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM', 'New York Yankees': 'NYY', 'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD', 'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SF', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH',
}

_NAME_TO_ABBR_BY_SPORT = {'nfl': NFL_NAME_TO_ABBR, 'mlb': MLB_NAME_TO_ABBR}


def abbr_from_name(name, sport='nfl'):
    """Abbreviation for a team's full display name, or None if unrecognized
    (unsupported sport, or a name that doesn't match exactly — callers
    should treat that as "can't compute travel for this game" rather than
    guess)."""
    return _NAME_TO_ABBR_BY_SPORT.get(sport, {}).get(name)


def timezone_change_hours(away_abbr, home_abbr, at_utc, sport='nfl'):
    """Hours the away team is shifting across (positive = traveling east,
    e.g. West Coast team flying to the East Coast), evaluated at `at_utc`
    (a timezone-aware datetime) so DST is handled correctly. None if either
    team is unrecognized."""
    from zoneinfo import ZoneInfo
    a, h = get(away_abbr, sport), get(home_abbr, sport)
    if not a or not h:
        return None
    away_off = at_utc.astimezone(ZoneInfo(a['tz'])).utcoffset()
    home_off = at_utc.astimezone(ZoneInfo(h['tz'])).utcoffset()
    return round((home_off - away_off).total_seconds() / 3600)
