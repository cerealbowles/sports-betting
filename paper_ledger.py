"""Paper ledger: grading, CLV and shadow-bankroll math for PaperPick rows.

The app logs the model's side on every game it tracks (moneyline, spread
and total, not only the ones that clear the recommendation bar; totals are
never recommended) at two fixed moments:

  - 'first_rec': the first refresh where _best_market_pick recommends the
    game. Measures how early the edge shows; the hardest price to catch.
  - 't60': the first refresh at or after 60 minutes before start. Close to
    what a bettor checking the app pregame would actually get.

Each row is closed with the last pregame line (GamePrediction's market fields
stop refreshing once the game leaves Preview) and graded on the final score.
Everything here is pure, so it can be tested without the Flask app.
"""
from datetime import datetime, timezone

SNAPSHOTS = ('t60', 'first_rec')
MARKETS = ('ml', 'spread', 'total')
T60_MINUTES = 60


def payout(amer):
  """Profit per 1u staked at American odds, or None."""
  if amer is None:
    return None
  try:
    o = float(amer)
  except (TypeError, ValueError):
    return None
  if o > 0:
    return o / 100.0
  if o < 0:
    return 100.0 / -o
  return None


def _implied(amer):
  o = float(amer)
  return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def vig_free(side_odds, other_odds):
  """Vig-free probability of the side priced at side_odds, or None."""
  if side_odds is None or other_odds is None:
    return None
  try:
    a, b = _implied(side_odds), _implied(other_odds)
  except (TypeError, ValueError, ZeroDivisionError):
    return None
  return a / (a + b) if a + b else None


def side_line(home_spread, side):
  """Spread from the picked side's point of view (home -3.5 -> away +3.5)."""
  if home_spread is None:
    return None
  return home_spread if side == 'home' else -home_spread


def lean(market, side, model_prob, side_odds, other_odds, line=None):
  """One paper pick's pricing fields: market prob, edge (pts), Kelly fraction.
  Kelly uses the posted price (with vig), so it can be <= 0 even when the
  vig-free edge is positive; the shadow bankroll only stakes Kelly > 0."""
  mkt = vig_free(side_odds, other_odds)
  b = payout(side_odds)
  if model_prob is None or mkt is None or not b:
    return None
  return {
      'market': market, 'side': side, 'line': line,
      'odds': int(side_odds), 'other_odds': int(other_odds),
      'model_prob': round(model_prob, 4), 'market_prob': round(mkt, 4),
      'edge': round((model_prob - mkt) * 100, 2),
      'kelly': round((b * model_prob - (1 - model_prob)) / b, 4),
  }


def grade(market, side, line, odds, home_score, away_score):
  """('win'|'loss'|'push', profit per 1u) or (None, None) if ungradeable."""
  if home_score is None or away_score is None:
    return None, None
  b = payout(odds)
  if b is None:
    return None, None
  if market == 'total':
    if line is None:
      return None, None
    combined = home_score + away_score
    margin = (combined - line) if side == 'over' else (line - combined)
  else:
    margin = (home_score - away_score) if side == 'home' else (away_score - home_score)
  if market == 'spread':
    if line is None:
      return None, None
    margin += line
  if margin > 0:
    return 'win', round(b, 4)
  if margin < 0:
    return 'loss', -1.0
  return 'push', 0.0


def clv(market, line, odds, other_odds, close_line, close_odds, close_other_odds, side=None):
  """(clv_prob, clv_pts). clv_prob: closing vig-free prob of our side minus
  ours at pick time, in points (positive = the market moved our way). For a
  spread it is only comparable when the line didn't move, so it's None
  otherwise. clv_pts (spread only): points gained vs the closing line from
  our side's view (took +3.5, closed +2.5 -> +1.0). Totals likewise: an
  over taken at 44.5 that closes 46 gained 1.5."""
  clv_pts = None
  if market in ('spread', 'total'):
    if line is None or close_line is None:
      return None, None
    gained = (close_line - line) if (market == 'total' and side == 'over') else (line - close_line)
    clv_pts = round(gained, 1)
    if clv_pts != 0:
      return None, clv_pts
  p_open = vig_free(odds, other_odds)
  p_close = vig_free(close_odds, close_other_odds)
  if p_open is None or p_close is None:
    return None, clv_pts
  return round((p_close - p_open) * 100, 2), clv_pts


def beat_close(row):
  """True/False/None (None = no closing data, or exactly matched the close)."""
  if row.get('clv_pts'):
    return row['clv_pts'] > 0
  c = row.get('clv_prob')
  if c is None or abs(c) < 0.05:
    return None
  return c > 0


def summarize(rows):
  """Record/ROI/CLV block for a list of row dicts (graded or not)."""
  graded = [r for r in rows if r.get('result')]
  w = sum(1 for r in graded if r['result'] == 'win')
  l = sum(1 for r in graded if r['result'] == 'loss')
  p = sum(1 for r in graded if r['result'] == 'push')
  units = sum(r['profit'] for r in graded)
  closed = [r for r in rows if r.get('clv_prob') is not None or r.get('clv_pts') is not None]
  beats = [beat_close(r) for r in closed]
  decided = [x for x in beats if x is not None]
  probs = [r['clv_prob'] for r in closed if r.get('clv_prob') is not None]
  pts = [r['clv_pts'] for r in closed if r.get('clv_pts') is not None]
  return {
      'logged': len(rows), 'graded': len(graded), 'pending': len(rows) - len(graded),
      'wins': w, 'losses': l, 'pushes': p,
      'units': round(units, 2),
      'roi': round(100 * units / len(graded), 1) if graded else None,
      'win_rate': round(100 * w / (w + l), 1) if w + l else None,
      'avg_edge': round(sum(r['edge'] for r in rows) / len(rows), 1) if rows else None,
      'closed': len(closed),
      'beat_close_pct': round(100 * sum(decided) / len(decided), 1) if decided else None,
      'beat_close_n': len(decided),
      'avg_clv_prob': round(sum(probs) / len(probs), 2) if probs else None,
      'avg_clv_pts': round(sum(pts) / len(pts), 2) if pts else None,
  }


def shadow_bankroll(rows, start=100.0, kelly_cap=0.25, flat_pct=0.01):
  """Replay graded picks in start-time order under three staking plans.
  Each bet is sized off the bankroll as it stood after the previous one
  settled, an approximation for games that overlap in real time.
    full: Kelly fraction, capped at kelly_cap of bankroll per bet
    half: half that
    flat: flat_pct of the STARTING bankroll every bet (1u on 100)
  Kelly <= 0 picks are skipped by the Kelly plans, staked by flat.
  Returns {'points': [...], 'final': {...}, 'max_drawdown': {...}}."""
  plans = {'full': start, 'half': start, 'flat': start}
  peak = dict(plans)
  mdd = {k: 0.0 for k in plans}
  points = [{'label': 'Start', 'full': start, 'half': start, 'flat': start, 'pick': None}]
  flat_stake = start * flat_pct
  for r in sorted((r for r in rows if r.get('result')), key=lambda r: (r.get('start') or '', r.get('id') or 0)):
    k = max(0.0, min(r.get('kelly') or 0.0, kelly_cap))
    b = payout(r['odds']) or 0.0
    for plan, frac in (('full', k), ('half', k / 2)):
      stake = plans[plan] * frac
      plans[plan] += _settle(stake, b, r['result'])
    plans['flat'] += _settle(flat_stake, b, r['result'])
    for plan in plans:
      plans[plan] = max(plans[plan], 0.0)
      peak[plan] = max(peak[plan], plans[plan])
      if peak[plan]:
        mdd[plan] = max(mdd[plan], (peak[plan] - plans[plan]) / peak[plan])
    points.append({'label': r.get('date') or '', 'pick': r.get('label'),
                   **{p: round(v, 2) for p, v in plans.items()}})
  return {
      'points': points,
      'final': {p: round(v, 2) for p, v in plans.items()},
      'max_drawdown': {p: round(v * 100, 1) for p, v in mdd.items()},
      'start': start, 'kelly_cap': kelly_cap, 'flat_pct': flat_pct,
  }


def _settle(stake, b, result):
  if result == 'win':
    return stake * b
  if result == 'loss':
    return -stake
  return 0.0


def t60_due(game_time_utc, now=None):
  """True once now is within T60_MINUTES of the scheduled start."""
  if not game_time_utc:
    return False
  try:
    start = datetime.fromisoformat(game_time_utc.replace('Z', '+00:00'))
  except ValueError:
    return False
  if start.tzinfo is None:
    start = start.replace(tzinfo=timezone.utc)
  now = now or datetime.now(timezone.utc)
  return (start - now).total_seconds() <= T60_MINUTES * 60


def minutes_to_start(game_time_utc, now):
  try:
    start = datetime.fromisoformat((game_time_utc or '').replace('Z', '+00:00'))
  except ValueError:
    return None
  if start.tzinfo is None:
    start = start.replace(tzinfo=timezone.utc)
  return int(round((start - now).total_seconds() / 60))
