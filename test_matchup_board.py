import unittest

import matchup_board as mb


class ReadGameTests(unittest.TestCase):
    def test_target_when_all_three_agree_big(self):
        # Market BUF -3. Model -7, sim BUF by 8, form BUF by ~6.5 -> all say home.
        r = mb.read_game(-3.0, model_h=-7.0, sim_home=28, sim_away=20, home_diff_pg=8, away_diff_pg=-2)
        self.assertEqual(r['verdict'], 'TARGET')
        self.assertEqual(r['side'], 'home')
        self.assertEqual(r['agree'], 3)
        self.assertAlmostEqual(r['edges']['model'], 4.0)
        self.assertAlmostEqual(r['edges']['sim'], 5.0)
        self.assertAlmostEqual(r['edges']['form'], 3.5)  # form line = -(5 + 1.5) = -6.5

    def test_mock_tb_atl_matches_design(self):
        # From the design mock: ATL @ TB, market TB -2.5, model -1.0, sim 23-22 ATL,
        # ATL -2/7, TB 11/7. Model and sim point ATL, form is ~flat -> LEAN ATL +2.5.
        r = mb.read_game(-2.5, model_h=-1.0, sim_home=22, sim_away=23, home_diff_pg=11 / 7, away_diff_pg=-2 / 7)
        self.assertEqual(r['side'], 'away')
        self.assertEqual(r['agree'], 2)
        self.assertEqual(r['verdict'], 'LEAN')
        self.assertAlmostEqual(r['side_line'], 2.5)
        self.assertTrue(r['is_dog'])

    def test_stay_away_when_split(self):
        # Model says home by 4, sim says away by 4.
        r = mb.read_game(-3.0, model_h=-7.0, sim_home=20, sim_away=21, home_diff_pg=0, away_diff_pg=0)
        self.assertEqual(r['verdict'], 'STAY AWAY')
        self.assertGreaterEqual(r['oppose'], 1)

    def test_watch_when_small(self):
        r = mb.read_game(-3.0, model_h=-3.2, sim_home=23, sim_away=20, home_diff_pg=1.5, away_diff_pg=0)
        self.assertEqual(r['verdict'], 'WATCH')

    def test_missing_signal_cannot_target(self):
        r = mb.read_game(-3.0, model_h=-9.0, sim_home=30, sim_away=20)
        self.assertEqual(r['n_signals'], 2)
        self.assertEqual(r['verdict'], 'LEAN')

    def test_no_market_no_read(self):
        self.assertIsNone(mb.read_game(None, model_h=-3))

    def test_fmt_line(self):
        self.assertEqual(mb.fmt_line(-3.0), '-3')
        self.assertEqual(mb.fmt_line(2.5), '+2.5')
        self.assertEqual(mb.fmt_line(0), 'PK')


class ScriptTests(unittest.TestCase):
    def test_fast_and_trailing(self):
        va = {'pass_att_pg': 40, 'rush_att_pg': 25}
        vh = {'pass_att_pg': 35, 'rush_att_pg': 28}
        s = mb.script_read('ARI', 'DAL', va, vh, 20, 27)
        self.assertTrue(s['pace'].startswith('FAST'))
        self.assertIn('ARI likely trailing', s['text'])

    def test_slow(self):
        v = {'pass_att_pg': 28, 'rush_att_pg': 26}
        s = mb.script_read('A', 'B', v, v, 20, 21)
        self.assertTrue(s['pace'].startswith('SLOW'))
        self.assertIn('balanced', s['text'])


if __name__ == '__main__':
    unittest.main()
