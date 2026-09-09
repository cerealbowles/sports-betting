"""
Weather for outdoor MLB and NFL stadiums via OpenWeatherMap free tier.
Set WEATHER_API_KEY env var to enable. Silently returns None if not set.
"""
import os
import time
import requests

OWM = "https://api.openweathermap.org/data/2.5/weather"

_cache = {}
_TTL   = 1800  # 30 min

# Outdoor MLB venues: substring of venue name → (lat, lon)
MLB_OUTDOOR = {
    'fenway':           (42.3467, -71.0972),
    'wrigley':          (41.9484, -87.6553),
    'pnc park':         (40.4468, -80.0057),
    'busch':            (38.6226, -90.1928),
    'coors':            (39.7559, -104.9942),
    'dodger':           (34.0739, -118.2400),
    'oracle park':      (37.7786, -122.3893),
    'great american':   (39.0979, -84.5082),
    'progressive field':(41.4962, -81.6852),
    'comerica':         (42.3390, -83.0485),
    'kauffman':         (39.0517, -94.4803),
    'target field':     (44.9817, -93.2784),
    'yankee':           (40.8296, -73.9262),
    'citi field':       (40.7571, -73.8458),
    'citizens bank':    (39.9057, -75.1665),
    'nationals park':   (38.8730, -77.0074),
    'truist':           (33.8908, -84.4679),
    'camden':           (39.2838, -76.6219),
    'guaranteed rate':  (41.8300, -87.6339),
    'rate field':       (41.8300, -87.6339),
    'angel':            (33.8003, -117.8827),
    'petco':            (32.7073, -117.1572),
    'sutter health':    (38.5872, -121.4993),
    'oakland':          (37.7516, -122.2007),
}

# Outdoor NFL venues: substring → (lat, lon)
NFL_OUTDOOR = {
    'soldier':          (41.8623, -87.6167),
    'lambeau':          (44.5013, -88.0622),
    'metlife':          (40.8135, -74.0745),
    'gillette':         (42.0909, -71.2643),
    'highmark':         (42.7738, -78.7870),
    'firstenergy':      (41.4960, -81.6976),
    'paycor':           (39.0954, -84.5160),
    'nissan':           (36.1665, -86.7713),
    'bank of america':  (35.2258, -80.8530),
    'empower':          (39.7439, -105.0201),
    'arrowhead':        (39.0489, -94.4839),
    'levi':             (37.4033, -121.9693),
    'tiaa bank':        (30.3239, -81.6373),
    'raymond james':    (27.9759, -82.5033),
    'tottenham':        (51.6044, -0.0665),
    'wembley':          (51.5560, -0.2796),
    'sofi':             (33.9535, -118.3392),   # retractable but weather-sensitive
}


def _get_weather(lat, lon):
    api_key = os.environ.get('WEATHER_API_KEY', '')
    if not api_key:
        return None

    cache_key = f'wx_{lat:.2f}_{lon:.2f}'
    now = time.time()
    if cache_key in _cache:
        data, ts = _cache[cache_key]
        if now - ts < _TTL:
            return data

    try:
        r = requests.get(OWM, params={
            'lat':   lat,
            'lon':   lon,
            'appid': api_key,
            'units': 'imperial',
        }, timeout=10)
        r.raise_for_status()
        raw = r.json()
    except Exception:
        return None

    wind = raw.get('wind', {})
    deg  = wind.get('deg', 0)
    dirs = ['N','NE','E','SE','S','SW','W','NW']
    wind_dir = dirs[round(deg / 45) % 8]

    data = {
        'temp':        round(raw['main']['temp']),
        'feels_like':  round(raw['main']['feels_like']),
        'conditions':  raw['weather'][0]['description'].title(),
        'wind_mph':    round(wind.get('speed', 0)),
        'wind_dir':    wind_dir,
        'humidity':    raw['main']['humidity'],
    }
    _cache[cache_key] = (data, now)
    return data


def get_game_weather(venue_name, sport='mlb'):
    """
    Returns weather dict for an outdoor venue, or None for domes / missing key.
    venue_name: string from the schedule API (e.g. 'PNC Park').
    """
    vl = venue_name.lower()
    lookup = MLB_OUTDOOR if sport == 'mlb' else NFL_OUTDOOR
    for fragment, (lat, lon) in lookup.items():
        if fragment in vl:
            return _get_weather(lat, lon)
    return None  # dome, retractable roof, or not in list
