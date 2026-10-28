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

# NBA rows are keyed by ESPN's NBA abbreviation (what nba_api.py's team dicts
# carry as team['abbrev'] — note ESPN's short forms: 'GS', 'NO', 'NY', 'SA',
# 'UTAH', 'WSH'). Coordinates are the arena, not the city centre. Surface is
# always hardwood and every arena is indoors, so those two columns are
# constant here and only kept so get() returns the same shape for every sport.
NBA_TEAMS = {
    'ATL':  ('Atlanta, GA',        33.7573,  -84.3963, 'hardwood', True, 'America/New_York'),
    'BOS':  ('Boston, MA',         42.3662,  -71.0621, 'hardwood', True, 'America/New_York'),
    'BKN':  ('Brooklyn, NY',       40.6826,  -73.9754, 'hardwood', True, 'America/New_York'),
    'CHA':  ('Charlotte, NC',      35.2251,  -80.8392, 'hardwood', True, 'America/New_York'),
    'CHI':  ('Chicago, IL',        41.8807,  -87.6742, 'hardwood', True, 'America/Chicago'),
    'CLE':  ('Cleveland, OH',      41.4965,  -81.6882, 'hardwood', True, 'America/New_York'),
    'DAL':  ('Dallas, TX',         32.7905,  -96.8103, 'hardwood', True, 'America/Chicago'),
    'DEN':  ('Denver, CO',         39.7487, -105.0077, 'hardwood', True, 'America/Denver'),
    'DET':  ('Detroit, MI',        42.3411,  -83.0553, 'hardwood', True, 'America/Detroit'),
    'GS':   ('San Francisco, CA',  37.7680, -122.3877, 'hardwood', True, 'America/Los_Angeles'),
    'HOU':  ('Houston, TX',        29.7508,  -95.3621, 'hardwood', True, 'America/Chicago'),
    'IND':  ('Indianapolis, IN',   39.7640,  -86.1555, 'hardwood', True, 'America/Indiana/Indianapolis'),
    'LAC':  ('Inglewood, CA',      33.9450, -118.3430, 'hardwood', True, 'America/Los_Angeles'),
    'LAL':  ('Los Angeles, CA',    34.0430, -118.2673, 'hardwood', True, 'America/Los_Angeles'),
    'MEM':  ('Memphis, TN',        35.1382,  -90.0506, 'hardwood', True, 'America/Chicago'),
    'MIA':  ('Miami, FL',          25.7814,  -80.1870, 'hardwood', True, 'America/New_York'),
    'MIL':  ('Milwaukee, WI',      43.0451,  -87.9172, 'hardwood', True, 'America/Chicago'),
    'MIN':  ('Minneapolis, MN',    44.9795,  -93.2761, 'hardwood', True, 'America/Chicago'),
    'NO':   ('New Orleans, LA',    29.9490,  -90.0821, 'hardwood', True, 'America/Chicago'),
    'NY':   ('New York, NY',       40.7505,  -73.9934, 'hardwood', True, 'America/New_York'),
    'OKC':  ('Oklahoma City, OK',  35.4634,  -97.5151, 'hardwood', True, 'America/Chicago'),
    'ORL':  ('Orlando, FL',        28.5392,  -81.3839, 'hardwood', True, 'America/New_York'),
    'PHI':  ('Philadelphia, PA',   39.9012,  -75.1720, 'hardwood', True, 'America/New_York'),
    'PHX':  ('Phoenix, AZ',        33.4457, -112.0712, 'hardwood', True, 'America/Phoenix'),
    'POR':  ('Portland, OR',       45.5316, -122.6668, 'hardwood', True, 'America/Los_Angeles'),
    'SAC':  ('Sacramento, CA',     38.5802, -121.4997, 'hardwood', True, 'America/Los_Angeles'),
    'SA':   ('San Antonio, TX',    29.4270,  -98.4375, 'hardwood', True, 'America/Chicago'),
    'TOR':  ('Toronto, ON',        43.6435,  -79.3791, 'hardwood', True, 'America/Toronto'),
    'UTAH': ('Salt Lake City, UT', 40.7683, -111.9011, 'hardwood', True, 'America/Denver'),
    'WSH':  ('Washington, DC',     38.8981,  -77.0209, 'hardwood', True, 'America/New_York'),
}

# Other common NBA abbreviation spellings (NBA.com / Basketball-Reference /
# odds feeds) → ESPN's, so a caller holding a non-ESPN abbreviation still
# resolves instead of silently dropping off the map.
_NBA_ALIASES = {
    'BRK': 'BKN', 'CHO': 'CHA', 'GSW': 'GS', 'NOP': 'NO', 'NYK': 'NY',
    'PHO': 'PHX', 'SAS': 'SA', 'UTA': 'UTAH', 'WAS': 'WSH',
}

# NHL rows are keyed by the official NHL API's own abbreviation (what
# nhl_api.py's team dicts carry as team['abbrev'] — bridged from ESPN's
# shorter spellings via _ESPN_TO_NHL_ABBREV in nhl_api.py before it ever
# reaches here, so this table never sees 'NJ'/'LA'/'SJ'/'TB'). Arizona's
# relocation means there's no 'ARI' row — Utah's franchise plays as 'UTA'.
NHL_TEAMS = {
    'ANA': ('Anaheim, CA',         33.8078, -117.8765, 'ice', True, 'America/Los_Angeles'),
    'BOS': ('Boston, MA',          42.3662,  -71.0621, 'ice', True, 'America/New_York'),
    'BUF': ('Buffalo, NY',         42.8750,  -78.8769, 'ice', True, 'America/New_York'),
    'CGY': ('Calgary, AB',         51.0375, -114.0519, 'ice', True, 'America/Edmonton'),
    'CAR': ('Raleigh, NC',         35.8033,  -78.7219, 'ice', True, 'America/New_York'),
    'CHI': ('Chicago, IL',         41.8807,  -87.6742, 'ice', True, 'America/Chicago'),
    'COL': ('Denver, CO',          39.7487, -105.0077, 'ice', True, 'America/Denver'),
    'CBJ': ('Columbus, OH',        39.9692,  -83.0061, 'ice', True, 'America/New_York'),
    'DAL': ('Dallas, TX',          32.7905,  -96.8103, 'ice', True, 'America/Chicago'),
    'DET': ('Detroit, MI',         42.3411,  -83.0553, 'ice', True, 'America/Detroit'),
    'EDM': ('Edmonton, AB',        53.5469, -113.4973, 'ice', True, 'America/Edmonton'),
    'FLA': ('Sunrise, FL',         26.1584,  -80.3256, 'ice', True, 'America/New_York'),
    'LAK': ('Los Angeles, CA',     34.0430, -118.2673, 'ice', True, 'America/Los_Angeles'),
    'MIN': ('St. Paul, MN',        44.9447,  -93.1011, 'ice', True, 'America/Chicago'),
    'MTL': ('Montreal, QC',        45.4961,  -73.5693, 'ice', True, 'America/Toronto'),
    'NSH': ('Nashville, TN',       36.1593,  -86.7785, 'ice', True, 'America/Chicago'),
    'NJD': ('Newark, NJ',          40.7336,  -74.1711, 'ice', True, 'America/New_York'),
    'NYI': ('Elmont, NY',          40.7230,  -73.5872, 'ice', True, 'America/New_York'),
    'NYR': ('New York, NY',        40.7505,  -73.9934, 'ice', True, 'America/New_York'),
    'OTT': ('Ottawa, ON',          45.2969,  -75.9272, 'ice', True, 'America/Toronto'),
    'PHI': ('Philadelphia, PA',    39.9012,  -75.1720, 'ice', True, 'America/New_York'),
    'PIT': ('Pittsburgh, PA',      40.4395,  -79.9892, 'ice', True, 'America/New_York'),
    'SJS': ('San Jose, CA',        37.3327, -121.9012, 'ice', True, 'America/Los_Angeles'),
    'SEA': ('Seattle, WA',         47.6221, -122.3540, 'ice', True, 'America/Los_Angeles'),
    'STL': ('St. Louis, MO',       38.6266,  -90.2026, 'ice', True, 'America/Chicago'),
    'TBL': ('Tampa, FL',           27.9427,  -82.4518, 'ice', True, 'America/New_York'),
    'TOR': ('Toronto, ON',         43.6435,  -79.3791, 'ice', True, 'America/Toronto'),
    'UTA': ('Salt Lake City, UT',  40.7683, -111.9011, 'ice', True, 'America/Denver'),
    'VAN': ('Vancouver, BC',       49.2778, -123.1089, 'ice', True, 'America/Vancouver'),
    'VGK': ('Las Vegas, NV',       36.1028, -115.1784, 'ice', True, 'America/Los_Angeles'),
    'WSH': ('Washington, DC',      38.8981,  -77.0209, 'ice', True, 'America/New_York'),
    'WPG': ('Winnipeg, MB',        49.8926,  -97.1435, 'ice', True, 'America/Winnipeg'),
}

# WNBA rows are keyed by ESPN's WNBA abbreviation (what wnba_api.py's team
# dicts carry as team['abbrev']/team['abbr'] — same ESPN basketball
# scoreboard convention as NBA, but a different team set, so this is NOT
# the same table as NBA_TEAMS: no CHA/DET/MEM/etc., and includes the two
# 2026 expansion franchises (Portland, Toronto).
WNBA_TEAMS = {
    'ATL': ('College Park, GA',    33.6358,  -84.4461, 'hardwood', True, 'America/New_York'),
    'CHI': ('Chicago, IL',         41.8534,  -87.6192, 'hardwood', True, 'America/Chicago'),
    'CON': ('Uncasville, CT',      41.4932,  -72.0979, 'hardwood', True, 'America/New_York'),
    'DAL': ('Arlington, TX',       32.7297,  -97.1081, 'hardwood', True, 'America/Chicago'),
    'GS':  ('San Francisco, CA',   37.7680, -122.3877, 'hardwood', True, 'America/Los_Angeles'),
    'IND': ('Indianapolis, IN',    39.7640,  -86.1555, 'hardwood', True, 'America/Indiana/Indianapolis'),
    'LV':  ('Las Vegas, NV',       36.0908, -115.1719, 'hardwood', True, 'America/Los_Angeles'),
    'LA':  ('Los Angeles, CA',     34.0430, -118.2673, 'hardwood', True, 'America/Los_Angeles'),
    'MIN': ('Minneapolis, MN',     44.9795,  -93.2761, 'hardwood', True, 'America/Chicago'),
    'NY':  ('Brooklyn, NY',        40.6826,  -73.9754, 'hardwood', True, 'America/New_York'),
    'PHX': ('Phoenix, AZ',         33.4457, -112.0712, 'hardwood', True, 'America/Phoenix'),
    'POR': ('Portland, OR',        45.5316, -122.6668, 'hardwood', True, 'America/Los_Angeles'),
    'SEA': ('Seattle, WA',         47.6221, -122.3540, 'hardwood', True, 'America/Los_Angeles'),
    'TOR': ('Toronto, ON',         43.6359,  -79.3617, 'hardwood', True, 'America/Toronto'),
    'WSH': ('Washington, DC',      38.8277,  -76.9856, 'hardwood', True, 'America/New_York'),
}

_BY_SPORT = {'nfl': NFL_TEAMS, 'mlb': MLB_TEAMS, 'nba': NBA_TEAMS, 'nhl': NHL_TEAMS, 'wnba': WNBA_TEAMS}


def get(abbr, sport='nfl'):
    """Returns a dict for a team abbreviation, or None if not in the table
    (e.g. an unrecognized/renamed abbreviation — callers should hide the
    situation fields that need this rather than guess)."""
    key = (abbr or '').upper()
    if sport == 'nba':
        key = _NBA_ALIASES.get(key, key)
    row = _BY_SPORT.get(sport, NFL_TEAMS).get(key)
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
