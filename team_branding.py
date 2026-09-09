"""
team_branding.py — Hardcoded MLB team colors + logo source, for the Discord
score-movement image cards (see discord_cards.py).

Keyed by MLB Stats API abbreviation (e.g. 'COL') — NOT by full team name.
Full team names are inconsistent across seasons for at least one franchise
(the Athletics show up as both "Oakland Athletics" and "Athletics" in
GamePrediction rows depending on which season a game is from, since the
franchise dropped its city name mid-relocation), while the abbreviation is
stable and is already what game['home']['abbr'] / game['away']['abbr']
contain everywhere else in this codebase. Verified against the live MLB
Stats API (/api/v1/teams) — these are the exact current abbreviations, not
guesses (a few are non-obvious: Arizona is 'AZ' not 'ARI', Chicago White
Sox is 'CWS' not 'CHW').

logo_slug is for ESPN's team-logo CDN:
    https://a.espncdn.com/i/teamlogos/mlb/500/{logo_slug}.png
Chosen over MLB's own static CDN (mlbstatic.com, SVG-only) specifically to
avoid needing an SVG rasterizer (cairosvg) as a dependency — that pulls in
system-level cairo/pango libraries that don't come with the python:3.11-slim
Docker base image and would complicate the build. Plain PNG + Pillow is a
much smaller footprint. All 30 slugs below were verified live (HTTP 200).

Colors are each team's official primary/secondary brand colors (widely
documented, stable — these don't change season to season the way rosters
or odds do).
"""

TEAM_BRANDING = {
    'AZ':  {'name': 'Arizona Diamondbacks',  'logo_slug': 'ari', 'primary': '#A71930', 'secondary': '#000000', 'accent': '#E3D4AD'},
    'ATH': {'name': 'Athletics',              'logo_slug': 'ath', 'primary': '#003831', 'secondary': '#EFB21E', 'accent': '#EFB21E'},
    'ATL': {'name': 'Atlanta Braves',         'logo_slug': 'atl', 'primary': '#13274F', 'secondary': '#CE1141', 'accent': '#CE1141'},
    'BAL': {'name': 'Baltimore Orioles',      'logo_slug': 'bal', 'primary': '#DF4601', 'secondary': '#000000', 'accent': '#DF4601'},
    'BOS': {'name': 'Boston Red Sox',         'logo_slug': 'bos', 'primary': '#BD3039', 'secondary': '#0C2340', 'accent': '#BD3039'},
    'CHC': {'name': 'Chicago Cubs',           'logo_slug': 'chc', 'primary': '#0E3386', 'secondary': '#CC3433', 'accent': '#CC3433'},
    'CWS': {'name': 'Chicago White Sox',      'logo_slug': 'cws', 'primary': '#27251F', 'secondary': '#C4CED4', 'accent': '#C4CED4'},
    'CIN': {'name': 'Cincinnati Reds',        'logo_slug': 'cin', 'primary': '#C6011F', 'secondary': '#000000', 'accent': '#C6011F'},
    'CLE': {'name': 'Cleveland Guardians',    'logo_slug': 'cle', 'primary': '#0C2340', 'secondary': '#E31937', 'accent': '#E31937'},
    'COL': {'name': 'Colorado Rockies',       'logo_slug': 'col', 'primary': '#33006F', 'secondary': '#000000', 'accent': '#C4CED4'},
    'DET': {'name': 'Detroit Tigers',         'logo_slug': 'det', 'primary': '#0C2340', 'secondary': '#FA4616', 'accent': '#FA4616'},
    'HOU': {'name': 'Houston Astros',         'logo_slug': 'hou', 'primary': '#002D62', 'secondary': '#EB6E1F', 'accent': '#EB6E1F'},
    'KC':  {'name': 'Kansas City Royals',     'logo_slug': 'kc',  'primary': '#004687', 'secondary': '#BD9B60', 'accent': '#BD9B60'},
    'LAA': {'name': 'Los Angeles Angels',     'logo_slug': 'laa', 'primary': '#BA0021', 'secondary': '#003263', 'accent': '#BA0021'},
    'LAD': {'name': 'Los Angeles Dodgers',    'logo_slug': 'lad', 'primary': '#005A9C', 'secondary': '#FFFFFF', 'accent': '#A5ACAF'},
    'MIA': {'name': 'Miami Marlins',          'logo_slug': 'mia', 'primary': '#00A3E0', 'secondary': '#000000', 'accent': '#EF3340'},
    'MIL': {'name': 'Milwaukee Brewers',      'logo_slug': 'mil', 'primary': '#12284B', 'secondary': '#FFC52F', 'accent': '#FFC52F'},
    'MIN': {'name': 'Minnesota Twins',        'logo_slug': 'min', 'primary': '#002B5C', 'secondary': '#D31145', 'accent': '#D31145'},
    'NYM': {'name': 'New York Mets',          'logo_slug': 'nym', 'primary': '#002D72', 'secondary': '#FF5910', 'accent': '#FF5910'},
    'NYY': {'name': 'New York Yankees',       'logo_slug': 'nyy', 'primary': '#003087', 'secondary': '#FFFFFF', 'accent': '#C4CED3'},
    'PHI': {'name': 'Philadelphia Phillies',  'logo_slug': 'phi', 'primary': '#E81828', 'secondary': '#002D72', 'accent': '#002D72'},
    'PIT': {'name': 'Pittsburgh Pirates',     'logo_slug': 'pit', 'primary': '#27251F', 'secondary': '#FDB827', 'accent': '#FDB827'},
    'SD':  {'name': 'San Diego Padres',       'logo_slug': 'sd',  'primary': '#2F241D', 'secondary': '#FFC425', 'accent': '#FFC425'},
    'SF':  {'name': 'San Francisco Giants',   'logo_slug': 'sf',  'primary': '#FD5A1E', 'secondary': '#27251F', 'accent': '#FD5A1E'},
    'SEA': {'name': 'Seattle Mariners',       'logo_slug': 'sea', 'primary': '#0C2C56', 'secondary': '#005C5C', 'accent': '#005C5C'},
    'STL': {'name': 'St. Louis Cardinals',    'logo_slug': 'stl', 'primary': '#C41E3A', 'secondary': '#0C2340', 'accent': '#C41E3A'},
    'TB':  {'name': 'Tampa Bay Rays',         'logo_slug': 'tb',  'primary': '#092C5C', 'secondary': '#8FBCE6', 'accent': '#8FBCE6'},
    'TEX': {'name': 'Texas Rangers',          'logo_slug': 'tex', 'primary': '#003278', 'secondary': '#C0111F', 'accent': '#C0111F'},
    'TOR': {'name': 'Toronto Blue Jays',      'logo_slug': 'tor', 'primary': '#134A8E', 'secondary': '#E8291C', 'accent': '#E8291C'},
    'WSH': {'name': 'Washington Nationals',   'logo_slug': 'wsh', 'primary': '#AB0003', 'secondary': '#14225A', 'accent': '#14225A'},
}

_DEFAULT = {'name': None, 'logo_slug': None, 'primary': '#2C2C2A', 'secondary': '#6B7280', 'accent': '#6B7280'}


def get(abbr):
    """Returns the branding dict for an MLB abbreviation, or a neutral gray
    fallback (never None/KeyError) for anything unrecognized — a typo'd or
    future-renamed abbreviation shouldn't crash card generation, just look
    generic."""
    return TEAM_BRANDING.get((abbr or '').upper(), _DEFAULT)
