"""Tests for slate_overview.py — run with `python3 -m pytest test_slate_overview.py`
(or plain `python3 test_slate_overview.py`)."""
import slate_overview as so
import team_venues


def _team(abbr, name, w, l, **kw):
    t = {'abbrev': abbr, 'abbr': abbr, 'name': name, 'wins': w, 'losses': l,
         'form': [], 'rest_days': None, 'games_played': w + l}
    t.update(kw)
    return t


def _game(gid, away, home, utc, status='Preview'):
    return {'game_id': gid, 'game_key': f'k{gid}', 'away': away, 'home': home,
            'game_time_utc': utc, 'status': status, 'venue': ''}


# Tuesday Oct 27 2026 slate as it appeared on the /nba page.
SLATE = [
    _game('1', _team('BKN', 'Brooklyn Nets', 0, 3), _team('BOS', 'Boston Celtics', 2, 1), '2026-10-27T23:30Z'),
    _game('2', _team('DET', 'Detroit Pistons', 3, 0), _team('NY', 'New York Knicks', 0, 3), '2026-10-28T00:00Z'),
    _game('3', _team('SAC', 'Sacramento Kings', 0, 3), _team('SA', 'San Antonio Spurs', 1, 2), '2026-10-28T00:00Z'),
    _game('4', _team('POR', 'Portland Trail Blazers', 2, 0), _team('LAL', 'Los Angeles Lakers', 3, 0), '2026-10-28T03:00Z'),
]


def test_projection_matches_d3_geoalbers():
    # Reference values from d3-geo geoAlbers() (the projection the SVG was drawn with).
    for (lat, lon), (ex, ey) in {
        (42.3662, -71.0621): (826.5876, 136.3856),
        (34.0430, -118.2673): (149.8104, 297.9206),
        (43.6435, -79.3791): (711.1585, 137.4845),
    }.items():
        x, y = so.project(lat, lon)
        assert abs(x - ex) < 0.01 and abs(y - ey) < 0.01


def test_every_nba_arena_lands_inside_the_map():
    for abbr in team_venues.NBA_TEAMS:
        v = team_venues.get(abbr, 'nba')
        x, y = so._to_pct(*so.project(v['lat'], v['lon']))
        assert 0 < x < 100 and 0 < y < 100, abbr


def test_aliases_resolve():
    assert team_venues.get('NYK', 'nba') == team_venues.get('NY', 'nba')
    assert team_venues.get('GSW', 'nba')['city'].startswith('San Francisco')


def test_markers_and_facts_for_real_slate():
    s = so.build(SLATE, 'nba')
    assert [m['home']['abbr'] for m in s['markers']] == ['BOS', 'NY', 'SA', 'LAL']
    labels = {f['label']: f for f in s['facts']}
    assert labels['GAME OF THE NIGHT']['title'] == 'POR @ LAL'
    assert 'Two unbeatens' in labels['GAME OF THE NIGHT']['detail']
    assert labels['BIGGEST MISMATCH ON PAPER']['title'] == 'DET @ NY'
    assert labels['STILL PERFECT']['title'] == 'DET 3-0 · LAL 3-0 · POR 2-0'
    assert labels['STILL WINLESS']['title'].startswith('BKN 0-3')
    assert s['facts'][-1]['kind'] == 'window'
    assert len(s['facts']) <= so.MAX_FACTS


def test_quiet_night_has_no_filler():
    g = [_game('1', _team('ATL', 'Hawks', 1, 1), _team('CHA', 'Hornets', 1, 1), '2026-10-27T23:00Z')]
    s = so.build(g, 'nba')
    assert len(s['markers']) == 1
    assert s['facts'] == []


def test_back_to_back_and_trip():
    g = [_game('1', _team('POR', 'Blazers', 2, 2, rest_days=1), _team('MIA', 'Heat', 2, 2, rest_days=3),
               '2026-10-27T23:30Z')]
    labels = {f['label']: f for f in so.build_facts(g, 'nba')}
    assert labels['BACK-TO-BACK']['title'] == 'POR'
    assert 'MIA is rested' in labels['BACK-TO-BACK']['detail']
    assert '3 time zones east' in labels['LONGEST TRIP']['detail']


def test_unknown_teams_and_empty_slate():
    assert so.build([], 'nba') is None
    g = [_game('1', _team('XXX', 'Nobody', 0, 0), _team('YYY', 'Nobody', 0, 0), '2026-10-27T23:00Z')]
    assert so.build(g, 'nba') is None


def test_same_city_arenas_do_not_stack():
    g = [_game('1', _team('BOS', 'Celtics', 1, 1), _team('LAL', 'Lakers', 1, 1), '2026-10-28T02:00Z'),
         _game('2', _team('MIA', 'Heat', 1, 1), _team('LAC', 'Clippers', 1, 1), '2026-10-28T03:00Z')]
    a, b = so.build(g, 'nba')['markers']
    assert abs(a['y'] - b['y']) >= 1.8 or abs(a['x'] - b['x']) >= 1.2


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
