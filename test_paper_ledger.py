import unittest
from datetime import datetime, timedelta, timezone

import paper_ledger as pl


class TestPaperLedger(unittest.TestCase):


  def test_grade_ml_and_spread(self):
    assert pl.grade('ml', 'home', None, -150, 4, 2) == ('win', round(100 / 150, 4))
    assert pl.grade('ml', 'away', None, 120, 4, 2) == ('loss', -1.0)
    # away +3.5, loses by 3 -> covers
    assert pl.grade('spread', 'away', 3.5, -110, 10, 7)[0] == 'win'
    # home -3, wins by 3 -> push
    assert pl.grade('spread', 'home', -3.0, -110, 10, 7) == ('push', 0.0)
    assert pl.grade('ml', 'home', None, -110, None, 3) == (None, None)


  def test_clv_ml_and_spread(self):
    # Took home at +100/-120; closed home -130/+110 -> market moved our way.
    cp, pts = pl.clv('ml', None, 100, -120, None, -130, 110)
    assert cp > 0 and pts is None
    # Took +3.5, closed +2.5 -> gained a point, prob CLV not comparable.
    assert pl.clv('spread', 3.5, -110, -110, 2.5, -110, -110) == (None, 1.0)
    # Same line, price moved against us.
    cp, pts = pl.clv('spread', -3.0, -105, -115, -3.0, -120, 100)
    assert pts == 0 and cp > 0  # our side got more expensive -> close prob higher -> positive


  def test_totals(self):
    # 24 + 20 = 44 vs 44.5: under wins, over loses; 44 vs 44 pushes.
    self.assertEqual(pl.grade('total', 'under', 44.5, -110, 24, 20)[0], 'win')
    self.assertEqual(pl.grade('total', 'over', 44.5, -110, 24, 20), ('loss', -1.0))
    self.assertEqual(pl.grade('total', 'over', 44.0, -110, 24, 20), ('push', 0.0))
    # Over at 44.5 that closes 46 gained 1.5; under at 44.5 closing 46 lost 1.5.
    self.assertEqual(pl.clv('total', 44.5, -110, -110, 46.0, -110, -110, side='over'), (None, 1.5))
    self.assertEqual(pl.clv('total', 44.5, -110, -110, 46.0, -110, -110, side='under'), (None, -1.5))

  def test_beat_close(self):
    assert pl.beat_close({'clv_pts': 1.0}) is True
    assert pl.beat_close({'clv_pts': -0.5}) is False
    assert pl.beat_close({'clv_pts': 0.0, 'clv_prob': 1.2}) is True
    assert pl.beat_close({'clv_prob': 0.0}) is None


  def test_lean_and_summary(self):
    lean = pl.lean('ml', 'home', 0.60, -110, -110)
    assert lean['edge'] == 10.0 and lean['kelly'] > 0
    rows = [
        {'result': 'win', 'profit': 1.0, 'edge': 5, 'clv_prob': 1.0},
        {'result': 'loss', 'profit': -1.0, 'edge': 5, 'clv_prob': -1.0},
        {'result': None, 'profit': None, 'edge': 5},
    ]
    s = pl.summarize(rows)
    assert (s['logged'], s['graded'], s['pending'], s['wins'], s['losses']) == (3, 2, 1, 1, 1)
    assert s['roi'] == 0.0 and s['beat_close_pct'] == 50.0


  def test_shadow_bankroll(self):
    rows = [
        {'id': 1, 'start': '2026-11-01T00:00Z', 'result': 'win', 'odds': 100, 'kelly': 0.1},
        {'id': 2, 'start': '2026-11-02T00:00Z', 'result': 'loss', 'odds': 100, 'kelly': 0.1},
        {'id': 3, 'start': '2026-11-03T00:00Z', 'result': 'win', 'odds': 100, 'kelly': -0.1},
    ]
    sh = pl.shadow_bankroll(rows, start=100, kelly_cap=0.25, flat_pct=0.01)
    assert sh['final']['full'] == round(100 * 1.1 * 0.9, 2)   # third pick skipped by Kelly
    assert sh['final']['half'] == round(100 * 1.05 * 0.95, 2)
    assert sh['final']['flat'] == 101.0
    assert len(sh['points']) == 4 and sh['max_drawdown']['full'] == 10.0


  def test_t60_due(self):
    now = datetime(2026, 11, 5, 18, 0, tzinfo=timezone.utc)
    assert pl.t60_due((now + timedelta(minutes=59)).isoformat(), now)
    assert not pl.t60_due((now + timedelta(minutes=61)).isoformat(), now)
    assert pl.t60_due('2026-11-05T18:30Z', now)
    assert pl.minutes_to_start('2026-11-05T18:30Z', now) == 30


if __name__ == '__main__':
  unittest.main()
