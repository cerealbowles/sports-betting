import unittest

import sim_math as sm


class TestSimMath(unittest.TestCase):
    def test_stevenson_under_sample(self):
        # Spec: Stevenson U 44.5 at -114 with mean 37, sd 30 ->
        # p(under) ~= 59.9%, EV ~= +12.4%.
        p_over = sm.p_over_normal(line=44.5, mean=37, sd=30)
        p_under = 1.0 - p_over
        self.assertAlmostEqual(p_under * 100, 59.9, delta=0.2)
        _, ev_under = sm.ev_over_under(p_over, over_price=-114, under_price=-114)
        self.assertAlmostEqual(ev_under * 100, 12.4, delta=0.2)

    def test_pass_tds_poisson_sample(self):
        # Spec: pass TDs lambda=1.6, O 1.5 -> p ~= 47.5%.
        p_over = sm.p_over_poisson(line=1.5, lam=1.6)
        self.assertAlmostEqual(p_over * 100, 47.5, delta=0.2)

    def test_anytime_td(self):
        p = sm.p_anytime_td(lam=0.0)
        self.assertAlmostEqual(p, 0.0)
        self.assertGreater(sm.p_anytime_td(lam=0.5), 0.0)

    def test_p_over_empirical(self):
        draws = [10, 20, 30, 40, 50]
        self.assertAlmostEqual(sm.p_over_empirical(25, draws), 0.6)
        self.assertIsNone(sm.p_over_empirical(25, []))

    def test_win_prob_excl_ties(self):
        self.assertAlmostEqual(sm.win_prob_excl_ties(52, 48), 0.52)
        self.assertIsNone(sm.win_prob_excl_ties(0, 0))

    def test_cover_prob_empirical(self):
        # Home favored by 3 (-3): margins > 3 cover, == 3 push, < 3 don't.
        margins = [10, 3, 3, -1, -20]
        out = sm.cover_prob_empirical(margins, line=-3.0)
        self.assertAlmostEqual(out['home_covers'], 1 / 5)
        self.assertAlmostEqual(out['push'], 2 / 5)
        self.assertAlmostEqual(out['away_covers'], 2 / 5)
        self.assertIsNone(sm.cover_prob_empirical([], line=-3.0))

    def test_histogram_bins_and_empty_bins(self):
        bins = sm.histogram([1, 1, 2, 9], bin_width=1, lo=0, hi=10)
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[1]['count'], 2)  # [1,2) has the two 1s
        self.assertEqual(bins[5]['count'], 0)  # empty bin stays present

    def test_pearson_r(self):
        self.assertAlmostEqual(sm.pearson_r([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(sm.pearson_r([1, 2, 3], [6, 4, 2]), -1.0)
        self.assertIsNone(sm.pearson_r([1], [2]))
        self.assertIsNone(sm.pearson_r([1, 1, 1], [1, 2, 3]))  # zero variance

    def test_percentile(self):
        vals = list(range(1, 11))  # 1..10
        self.assertEqual(sm.percentile(vals, 0), 1)
        self.assertEqual(sm.percentile(vals, 100), 10)
        self.assertIsNone(sm.percentile([], 50))


if __name__ == '__main__':
    unittest.main()
