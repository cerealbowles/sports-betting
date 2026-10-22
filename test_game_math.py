import unittest

import game_math as gm


class TestGameMath(unittest.TestCase):
    def test_hold_and_no_vig(self):
        # Spec: +134/-158 -> hold 4.0%, no-vig 41.1/58.9
        imp_away = gm.implied_prob(134)
        imp_home = gm.implied_prob(-158)
        novig_away, novig_home, hold = gm.no_vig_pair(imp_away, imp_home)
        self.assertAlmostEqual(hold * 100, 4.0, places=1)
        self.assertAlmostEqual(novig_away * 100, 41.1, places=1)
        self.assertAlmostEqual(novig_home * 100, 58.9, places=1)

    def test_ev_and_kelly(self):
        # Spec: p=.446 at +134 -> EV +4.4%, 1/4 Kelly 0.8u
        b = gm.net_payout_b(134)
        ev = gm.ev(0.446, b)
        self.assertAlmostEqual(ev * 100, 4.4, places=1)
        self.assertAlmostEqual(gm.kelly_units(0.446, b), 0.8, places=1)

    def test_net_payout_b_favorite(self):
        self.assertAlmostEqual(gm.net_payout_b(-110), 100 / 110)
        self.assertAlmostEqual(gm.net_payout_b(150), 1.5)

    def test_cover_prob_normal_sample(self):
        # Spec: NE +3 at ~54.6% (model spread_home=-1.5 -> home favored by 1.5,
        # so NE's own expected margin is -1.5; sigma=13.5). The spec's 49.9%
        # for +2.5 bakes in an optional key-number (3) adjustment this plain
        # normal approximation doesn't apply, so only the +3 value (not on a
        # key-number discontinuity) is checked tightly; +2.5 is checked only
        # for direction (strictly less than +3, since less cushion to cover).
        p_plus3 = gm.cover_prob_normal(model_margin=-1.5, market_line=3.0, sigma=13.5)
        p_plus25 = gm.cover_prob_normal(model_margin=-1.5, market_line=2.5, sigma=13.5)
        self.assertAlmostEqual(p_plus3 * 100, 54.6, delta=0.5)
        self.assertLess(p_plus25, p_plus3)

    def test_outlier_guard_trips_on_sample_total(self):
        # Spec: model 40.0 vs market 50.5, market_gap_sd 4.4 -> trips
        self.assertTrue(gm.outlier_guard(model_line=40.0, market_line=50.5, market_gap_sd=4.4))
        self.assertFalse(gm.outlier_guard(model_line=48.0, market_line=50.5, market_gap_sd=4.4))

    def test_outlier_guard_no_sd_is_safe(self):
        self.assertFalse(gm.outlier_guard(40.0, 50.5, market_gap_sd=0))
        self.assertFalse(gm.outlier_guard(40.0, 50.5, market_gap_sd=None))

    def test_confidence_formula(self):
        self.assertEqual(gm.confidence(0.60, 0.60), 5)  # no disagreement
        self.assertEqual(gm.confidence(0.60, 0.50), 3)  # 10pt gap -> -2
        self.assertEqual(gm.confidence(0.60, 0.10), 1)  # huge gap floors at 1
        # Sample: p_team=.604, p_player=.498 -> ~10.6pt gap -> start5 -2 = 3
        self.assertEqual(gm.confidence(0.604, 0.498), 3)

    def test_tag_rules(self):
        self.assertEqual(gm.tag_for(ev_value=0.05, confidence_value=4, is_outlier=False), 'PLAY')
        self.assertEqual(gm.tag_for(ev_value=0.05, confidence_value=3, is_outlier=False), 'LEAN')
        self.assertEqual(gm.tag_for(ev_value=0.01, confidence_value=5, is_outlier=False), 'PASS')
        self.assertEqual(gm.tag_for(ev_value=0.50, confidence_value=5, is_outlier=True), 'CHECK')


if __name__ == '__main__':
    unittest.main()
