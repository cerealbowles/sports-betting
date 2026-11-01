"""Tests for nfl_matchup_notes.py — run with `python3 -m pytest test_nfl_matchup_notes.py`
(or plain `python3 test_nfl_matchup_notes.py`)."""
import nfl_matchup_notes as mn

N = 32
CHI = {'id': '3', 'abbr': 'CHI', 'rest_days': 7, 'injuries': []}
SEA = {'id': '26', 'abbr': 'SEA', 'rest_days': 7, 'injuries': []}


def _ranks(**over):
    """Middle-of-the-pack everything, then per-team overrides as
    {'CHI': {'rush_ypc_allowed': (5.1, 29)}, ...}."""
    base = {
        'games': (8, None), 'rush_ypc': (4.3, 16), 'rush_ypg': (115.0, 16),
        'pass_any_a': (6.0, 16), 'pass_ypg': (220.0, 16),
        'rush_ypc_allowed': (4.3, 16), 'rush_ypg_allowed': (115.0, 16),
        'pass_any_a_allowed': (6.0, 16), 'pass_ypg_allowed': (220.0, 16),
        'sacks_pg': (2.4, 16), 'long_plays_allowed_pg': (3.5, 16),
 'turnover_diff': (0, 16),
    }
    out = {}
    for team in (CHI, SEA):
        d = dict(base)
        d.update(over.get(team['abbr'], {}))
        out[team['id']] = d
    return out


def _texts(notes):
    return [n['text'] for n in notes]


def test_average_matchup_produces_no_unit_notes():
    notes = mn.build_notes(SEA, CHI, _ranks(), N)
    assert not [n for n in notes if n['kind'] in ('run', 'pass', 'lean')]


def test_run_edge_uses_real_numbers():
    r = _ranks(SEA={'rush_ypc': (5.0, 3), 'rush_ypg': (140.2, 4)},
               CHI={'rush_ypc_allowed': (5.1, 29), 'rush_ypg_allowed': (138.0, 28)})
    notes = mn.build_notes(SEA, CHI, r, N)
    run = [n for n in notes if n['kind'] == 'run']
    assert len(run) == 1 and run[0]['team'] == 'SEA' and run[0]['tone'] == 'good'
    assert '5.0 YPC' in run[0]['text'] and '#29' in run[0]['text'] and '140.2 yds/g' in run[0]['text']


def test_users_example_weak_run_d_but_weaker_pass_d_leans_air():
    # CHI gives up a lot on the ground, but even more through the air; SEA's
    # run game is poor (4.0 YPC) — the path of least resistance is passing.
    # (Mild enough on the pass side not to trip the separate "big edge" note.)
    r = _ranks(SEA={'rush_ypc': (4.0, 26)},
               CHI={'rush_ypc_allowed': (4.4, 18), 'pass_any_a_allowed': (7.4, 27)})
    notes = mn.build_notes(SEA, CHI, r, N)
    lean = [n for n in notes if n['kind'] == 'lean']
    assert lean and 'through the air' in lean[0]['text'] and lean[0]['team'] == 'SEA'


def test_tough_spot_flagged_bad():
    r = _ranks(SEA={'pass_any_a': (4.9, 30)}, CHI={'pass_any_a_allowed': (4.8, 2)})
    notes = mn.build_notes(SEA, CHI, r, N)
    p = [n for n in notes if n['kind'] == 'pass' and n['team'] == 'SEA']
    assert p and p[0]['tone'] == 'bad'


def test_qb_out_is_top_note():
    home = dict(CHI, injuries=[{'name': 'Caleb Williams', 'pos': 'QB', 'status': 'Out'}])
    r = _ranks(SEA={'rush_ypc': (5.0, 3)}, CHI={'rush_ypc_allowed': (5.1, 29)})
    notes = mn.build_notes(SEA, home, r, N)
    assert notes[0]['kind'] == 'injury' and 'starting QB Caleb Williams is Out' in notes[0]['text']


def test_injuries_use_usage_to_skip_backups():
    usage = {CHI['id']: {'qb_id': '100', 'rush_shares': {'200': 0.55, '201': 0.03},
                         'rec_shares': {'300': 0.24, '301': 0.04}}}
    home = dict(CHI, injuries=[
        {'id': '101', 'name': 'Backup QB', 'pos': 'QB', 'status': 'Questionable'},
        {'id': '102', 'name': 'Third QB', 'pos': 'QB', 'status': 'Out'},
        {'id': '201', 'name': 'RB3', 'pos': 'RB', 'status': 'Out'},
        {'id': '301', 'name': 'WR5', 'pos': 'WR', 'status': 'Out'},
        {'id': '300', 'name': 'Top WR', 'pos': 'WR', 'status': 'Out'},
    ])
    t = ' '.join(_texts(mn.build_notes(SEA, home, _ranks(), N, usage=usage)))
    assert 'Backup QB' not in t and 'Third QB' not in t and 'RB3' not in t and 'WR5' not in t
    assert 'Top WR (WR, 24% of targets)' in t


def test_starting_qb_questionable_with_usage():
    usage = {CHI['id']: {'qb_id': '100', 'rush_shares': {}, 'rec_shares': {}}}
    home = dict(CHI, injuries=[{'id': '100', 'name': 'QB1', 'pos': 'QB', 'status': 'Questionable'}])
    notes = mn.build_notes(SEA, home, _ranks(), N, usage=usage)
    assert notes[0]['text'].startswith('CHI starting QB QB1 is questionable')


def test_lean_not_repeated_when_edge_already_noted():
    r = _ranks(SEA={'pass_any_a': (8.3, 2)}, CHI={'pass_any_a_allowed': (10.2, 31)})
    kinds = [n['kind'] for n in mn.build_notes(SEA, CHI, r, N) if n['team'] == 'SEA']
    assert 'pass' in kinds and 'lean' not in kinds


def test_rest_travel_weather():
    away = dict(SEA, rest_days=14)
    notes = mn.build_notes(away, CHI, _ranks(), N, weather={'temp': 18, 'gust_mph': 28},
                           tz_change=2, kickoff_et_hour=13)
    t = ' '.join(_texts(notes))
    assert 'Rest edge to SEA' in t
    assert '2 time zones east' in t and '11:00 AM body clock' in t
    assert 'Gusts to 28 mph' in t and '18°F' in t


def test_streak_and_turnovers():
    recent = {'away': [{'result': 'W'}] * 4 + [{'result': 'L'}], 'home': [{'result': 'L'}, {'result': 'W'}]}
    r = _ranks(SEA={'turnover_diff': (7.0, 2)}, CHI={'turnover_diff': (-3.0, 28)})
    notes = mn.build_notes(SEA, CHI, r, N, recent=recent)
    t = ' '.join(_texts(notes))
    assert 'SEA has won 4 straight' in t
    assert 'SEA is +7' in t and '-3 for CHI' in t


def test_small_sample_caveat():
    r = _ranks(SEA={'games': (2, None), 'rush_ypc': (5.0, 3)}, CHI={'rush_ypc_allowed': (5.1, 29)})
    notes = mn.build_notes(SEA, CHI, r, N)
    assert notes[-1]['kind'] == 'sample'


def test_missing_data_is_safe():
    assert mn.build_notes(SEA, CHI, {}, 0) == []
    notes = mn.build_notes(SEA, CHI, {}, 32, weather=None, recent=None)
    assert notes == []


def test_capped_and_weight_stripped():
    r = _ranks(SEA={'rush_ypc': (5.0, 1), 'pass_any_a': (8.0, 1), 'turnover_diff': (9.0, 1)},
               CHI={'rush_ypc_allowed': (5.5, 32), 'pass_any_a_allowed': (8.0, 32), 'sacks_pg': (4.0, 1),
                    'long_plays_allowed_pg': (6.0, 32),
                    'rush_ypc': (3.5, 32), 'pass_any_a': (4.0, 32), 'turnover_diff': (-8.0, 32)})
    home = dict(CHI, injuries=[{'name': 'A', 'pos': 'QB', 'status': 'Out'},
                               {'name': 'B', 'pos': 'WR', 'status': 'Out'}])
    notes = mn.build_notes(SEA, home, r, N, weather={'gust_mph': 30, 'temp': 10})
    assert len(notes) <= mn._MAX_NOTES
    assert all('weight' not in n for n in notes)


if __name__ == '__main__':
    import sys
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('ok  ', name)
            except AssertionError as e:
                fails += 1
                print('FAIL', name, e)
    sys.exit(1 if fails else 0)
