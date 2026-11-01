"""Pull completed FBS games (regular + postseason) from ESPN's scoreboard,
with kickoff timestamp, neutral-site flag, conference, and venue city/state.
Usage: python3 research/cfb/fetch_games.py 2023 2024 2025 2026"""
import json, os, sys, time, urllib.request, urllib.parse
URL = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard'
CONF = {'151': 'American', '1': 'ACC', '4': 'Big 12', '5': 'Big Ten', '12': 'C-USA', '18': 'Independent',
        '15': 'MAC', '17': 'Mountain West', '9': 'Pac-12', '8': 'SEC', '37': 'Sun Belt'}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'games.json')


def get(params):
    for i in range(3):
        try:
            with urllib.request.urlopen(URL + '?' + urllib.parse.urlencode(params), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 ** i)
    return {}


def parse(ev, season, stype):
    comp = ev['competitions'][0]
    if comp['status']['type']['state'] != 'post':
        return None
    t = {c['homeAway']: c for c in comp['competitors']}
    h, a = t['home'], t['away']
    venue = comp.get('venue') or {}
    addr = venue.get('address') or {}
    return dict(id=ev['id'], season=season, stype=stype, date=ev['date'][:10], ts=ev['date'],
                neutral=bool(comp.get('neutralSite')), venue=venue.get('fullName'),
                city=addr.get('city'), state=addr.get('state'),
                home=h['team']['displayName'], away=a['team']['displayName'],
                home_conf=CONF.get(str(h['team'].get('conferenceId')), 'FCS'),
                away_conf=CONF.get(str(a['team'].get('conferenceId')), 'FCS'),
                hs=int(h.get('score') or 0), as_=int(a.get('score') or 0))


out = {}
for season in map(int, sys.argv[1:]):
    for stype, weeks in ((2, range(1, 17)), (3, [1])):
        for wk in weeks:
            d = get({'seasontype': stype, 'week': wk, 'dates': season, 'groups': 80, 'limit': 400})
            for ev in d.get('events', []):
                g = parse(ev, season, stype)
                if g:
                    out[g['id']] = g
            time.sleep(0.1)
    print(season, sum(1 for g in out.values() if g['season'] == season), flush=True)
json.dump(list(out.values()), open(OUT, 'w'))
