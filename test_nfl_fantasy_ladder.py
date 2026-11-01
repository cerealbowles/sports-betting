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


if __name__ == '__main__':
    unittest.main()
