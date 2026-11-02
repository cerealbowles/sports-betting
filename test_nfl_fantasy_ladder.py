import unittest

import nfl_fantasy as nf


class LadderTests(unittest.TestCase):
    def test_factor_matches_design_examples(self):
        # Design copy: Kyler Murray's +52% lifts 18.2 to ~22.0; Goff's -64% drops 16.7 to ~12.4.
        self.assertAlmostEqual(18.2 * nf.matchup_factor(52), 21.99, places=1)
        self.assertAlmostEqual(16.7 * nf.matchup_factor(-64), 12.42, places=1)
        self.assertEqual(nf.matchup_factor(None), 1.0)
        self.assertEqual(nf.matchup_factor(-500), 0.0)

    def test_tiers(self):
        self.assertEqual(nf.start_sit_tier('QB', 19), 'MUST START')
        self.assertEqual(nf.start_sit_tier('QB', 18.9), 'SOLID')
        self.assertEqual(nf.start_sit_tier('TE', 7.5), 'SOLID')
        self.assertEqual(nf.start_sit_tier('WR', 6.9), 'BENCH')
        self.assertIsNone(nf.start_sit_tier('DST', 10))

    def test_ladder_moves(self):
        ps = [{'name': 'A', 'position': 'RB', 'sim_median': 15, 'adj_median': 13},
              {'name': 'B', 'position': 'RB', 'sim_median': 12, 'adj_median': 17}]
        ladder = nf.start_sit_ladder(ps, 'RB')
        self.assertEqual([t['name'] for t in ladder], ['MUST START', 'SOLID', 'RISKY', 'BENCH'])
        self.assertEqual(ladder[0]['players'][0]['name'], 'B')
        self.assertEqual(ladder[0]['players'][0]['move'], 1)
        self.assertEqual(ladder[1]['players'][0]['move'], -1)


class CalibrateDrawsTests(unittest.TestCase):
    def test_stretches_around_median_and_clamps(self):
        d = nf.calibrate_draws([2, 6, 10, 14, 18], 'WR', median=10)
        self.assertEqual(d[2], 10)                      # median unchanged
        self.assertAlmostEqual(d[1], 10 - 1.75 * 4)     # below stretched by 1.75
        self.assertAlmostEqual(d[3], 10 + 1.15 * 4)     # above stretched by 1.15
        self.assertEqual(d[0], 0.0)                     # 10 - 1.75*8 = -4 -> clamped to 0

    def test_unknown_position_untouched(self):
        self.assertEqual(nf.calibrate_draws([3, 1, 2], 'DST'), [1, 2, 3])


class CombinedProjectionTests(unittest.TestCase):
    def test_implied_points(self):
        self.assertEqual(nf.implied_team_points(-3.0, 47.0, True), 25.0)
        self.assertEqual(nf.implied_team_points(-3.0, 47.0, False), 22.0)
        self.assertIsNone(nf.implied_team_points(None, 47.0, True))

    def test_qb_uses_market(self):
        c = nf.combined_projection('QB', 20.0, 27.0)
        self.assertTrue(c['uses_market'])
        self.assertAlmostEqual(c['point'], round(-4.771 + 0.528 * 20 + 0.578 * 27, 1))
        self.assertLess(c['floor'], c['point'])
        self.assertGreater(c['ceiling'], c['point'])

    def test_qb_without_line_falls_back(self):
        c = nf.combined_projection('QB', 18.0, None)
        self.assertFalse(c['uses_market'])
        self.assertEqual(c['point'], 18.0)

    def test_rec_is_projection_and_floor_clamped(self):
        c = nf.combined_projection('TE', 2.0, 30.0)
        self.assertEqual(c['point'], 2.0)
        self.assertEqual(c['floor'], 0.0)
        self.assertIsNone(nf.combined_projection('DST', 8.0, 20.0))


if __name__ == '__main__':
    unittest.main()
