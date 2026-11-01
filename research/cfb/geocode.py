"""Geocode every venue city in data/games.json via Open-Meteo (free, no key).
Writes data/geo.json: {"City|ST": {"lat", "lon", "tz"}}. Re-runs only look up
cities not already cached."""
import json, os, time, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES, GEO = os.path.join(HERE, 'data', 'games.json'), os.path.join(HERE, 'data', 'geo.json')
STATES = {'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado',
          'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia', 'FL': 'Florida', 'GA': 'Georgia',
          'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
          'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts',
          'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana',
          'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
          'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
          'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
          'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
          'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}


def lookup(city, state):
    q = urllib.parse.urlencode({'name': city, 'count': 20, 'country_code': 'US'})
    try:
        with urllib.request.urlopen(f'https://geocoding-api.open-meteo.com/v1/search?{q}', timeout=20) as r:
            res = json.load(r).get('results') or []
    except Exception:
        return None
    want = STATES.get(state, state)
    hit = next((x for x in res if x.get('admin1') == want), None)
    return hit and {'lat': hit['latitude'], 'lon': hit['longitude'], 'tz': hit['timezone']}


geo = json.load(open(GEO)) if os.path.exists(GEO) else {}
places = {(g['city'], g['state']) for g in json.load(open(GAMES)) if g.get('city') and g.get('state')}
for city, state in sorted(places):
    key = f'{city}|{state}'
    if geo.get(key):
        continue
    geo[key] = lookup(city, state)
    time.sleep(0.05)
json.dump(geo, open(GEO, 'w'), indent=0)
missing = sorted(k for k, v in geo.items() if v is None)
print(f'{len(geo)} places, {len(missing)} unresolved: {missing}')
