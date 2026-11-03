"""Tests for cbb_rating.py / cbb_model.py (python3 -m unittest test_cbb_model)."""
import math
import random
import unittest

import cbb_model
import cbb_rating as R


def _synthetic_games(n_days=60, seed=7):
    """Round-robin-ish season with known strengths: each team's true margin
    vs average is its index-based rating, HFA = 3 pts, 68 possessions."""
    rng = random.Random(seed)
    confs = {'A': ['A1', 'A2', 'A3', 'A4'], 'B': ['B1', 'B2', 'B3', 'B4']}
    strength = {'A1': 12, 'A2': 6, 'A3': 2, 'A4': -2, 'B1': 4, 'B2': -4, 'B3': -8, 'B4': -12}
    conf_of = {t: c for c, ts in confs.items() for t in ts}
    teams = list(strength)
    games, gid = [], 0
    for day in range(n_days):
        rng.shuffle(teams)
        for h, a in zip(teams[::2], teams[1::2]):
            gid += 1
            neutral = rng.random() < 0.15
            margin = strength[h] - strength[a] + (0 if neutral else 3) + rng.gauss(0, 8)
            total = 140 + rng.gauss(0, 10)
            hs, as_ = round((total + margin) / 2), round((total - margin) / 2)
            if hs == as_:
                hs += 1
            games.append({'id': str(gid), 'season': 2026, 'season_type': 2,
                          'date': f'2026-01-{1 + day % 28:02d}' if day < 28 else f'2026-02-{1 + (day - 28) % 28:02d}',
                          'ts': '', 'neutral': neutral, 'home': h, 'away': a,
                          'home_conf': conf_of[h], 'away_conf': conf_of[a],
                          'home_score': hs, 'away_score': as_, 'poss': 68.0})
    return games, strength


class RatingTest(unittest.TestCase):
    def test_recovers_order_and_hfa(self):
        games, strength = _synthetic_games()
        f = R.fit(games, 2026)
        em = {t: R.team_ratings(f, t, ndigits=None)['adj_em'] for t in strength}
        self.assertEqual(sorted(strength, key=strength.get), sorted(em, key=em.get))
        hfa_pts = f['hfa'] * 68 / 100
        self.assertAlmostEqual(hfa_pts, 3.0, delta=1.5)

    def test_neutral_removes_home_court(self):
        games, _ = _synthetic_games()
        f = R.fit(games, 2026)
        home = R.predict(f, 'A2', 'B1', 'A', 'B', neutral=False)['margin']
        neutral = R.predict(f, 'A2', 'B1', 'A', 'B', neutral=True)['margin']
        swapped = R.predict(f, 'B1', 'A2', 'B', 'A', neutral=True)['margin']
        self.assertGreater(home, neutral)
        self.assertAlmostEqual(neutral, -swapped, places=6)


class ModelTest(unittest.TestCase):
    def setUp(self):
        games, _ = _synthetic_games()
        self.fit = R.fit(games, 2026)
        self.team = {'wins': 3, 'losses': 1, 'form': ['W', 'W', 'L', 'W']}

    def _logit_factors(self, m):
        return sum(c for label, c in m['factors'] if '(tracking)' not in label)

    def test_factors_sum_to_logit(self):
        for neutral in (False, True):
            rating = R.predict(self.fit, 'A1', 'B3', 'A', 'B', neutral=neutral)
            m = cbb_model.predict(self.team, self.team, rating=rating, neutral=neutral)
            self.assertAlmostEqual(self._logit_factors(m), m['logit'], delta=0.005)
            self.assertAlmostEqual(m['logit'], rating['margin'] / cbb_model.MARGIN_SCALE, places=3)

    def test_neutral_has_no_home_court_row(self):
        rating = R.predict(self.fit, 'A1', 'B3', 'A', 'B', neutral=True)
        m = cbb_model.predict(self.team, self.team, rating=rating, neutral=True)
        self.assertNotIn('Home court', [label for label, _ in m['factors']])

    def test_probability_matches_margin(self):
        rating = R.predict(self.fit, 'A4', 'A1', 'A', 'A')
        m = cbb_model.predict(self.team, self.team, rating=rating)
        expected = 1 / (1 + math.exp(-rating['margin'] / cbb_model.MARGIN_SCALE))
        self.assertAlmostEqual(m['home_prob'], round(expected, 4), places=4)

    def test_market_shrink_only_beyond_ten_points(self):
        rating = R.predict(self.fit, 'A1', 'B4', 'A', 'B')
        raw = cbb_model.predict(self.team, self.team, rating=rating)['home_prob']
        near = cbb_model.predict(self.team, self.team, rating=rating, market_home_prob=raw - 0.05)
        far = cbb_model.predict(self.team, self.team, rating=rating, market_home_prob=raw - 0.30)
        self.assertAlmostEqual(near['home_prob'], raw, places=4)
        self.assertLess(far['home_prob'], raw)

    def test_no_rating_falls_back_to_home_court(self):
        m = cbb_model.predict(self.team, self.team, rating=None)
        self.assertGreater(m['home_prob'], 0.5)
        self.assertEqual(cbb_model.predict(self.team, self.team, rating=None, neutral=True)['home_prob'], 0.5)


if __name__ == '__main__':
    unittest.main()
