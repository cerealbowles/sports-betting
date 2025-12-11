from flask import Flask, request, redirect, url_for, render_template, abort, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from datetime import datetime
import os
import math

# app.py
# Simple Flask app for sports betting with Kelly criterion, open/closed bets and space to tweak formula using historical bets.

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bets.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class Setting(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  bankroll = db.Column(db.Float, default=20)
  percent_bankroll = db.Column(db.Float, default=0.25)  # fraction of bankroll

class OpenBet(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(200))
  odds = db.Column(db.Float)  # decimal odds
  prob = db.Column(db.Float)  # user's estimated win probability (0-1)
  stake = db.Column(db.Float)
  sport = db.Column(db.String(100), default='')     # NEW: sport text
  bet_type = db.Column(db.String(50), default='Moneyline')  # NEW: Moneyline/Spread/Over/Under/Player
  created_at = db.Column(db.DateTime, default=datetime.utcnow)
  eventstart = db.Column(db.DateTime, default=None)  # event start time

class ClosedBet(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(200))
  odds = db.Column(db.Float)
  prob = db.Column(db.Float)
  stake = db.Column(db.Float)
  sport = db.Column(db.String(100), default='')     # NEW
  bet_type = db.Column(db.String(50), default='Moneyline')  # NEW
  outcome = db.Column(db.String(20))  # 'win' or 'loss'
  profit = db.Column(db.Float)
  closed_at = db.Column(db.DateTime, default=datetime.utcnow)
  eventstart = db.Column(db.DateTime, default=None) 

# Create DB if missing
with app.app_context():
  if not os.path.exists('bets.db'):
    db.create_all()
    db.session.add(Setting(bankroll=50, percent_bankroll=0.25))
    db.session.commit()

def ensure_column_exists():
  with app.app_context():
    inspector = inspect(db.engine)
    columns = [col['name'] for col in inspector.get_columns('open_bet')]
    if 'eventstart' not in columns:
        with db.engine.begin() as conn:
            conn.execute(text('ALTER TABLE open_bet ADD COLUMN eventstart DATETIME DEFAULT NULL'))
            conn.execute(text('ALTER TABLE closed_bet ADD COLUMN eventstart DATETIME DEFAULT NULL'))
    
ensure_column_exists()

# Kelly calculation function with space to tweak using closed bets history
def compute_recommended_amount(bankroll, percent_bankroll, odds, prob, closed_bets):
  """
  Basic Kelly fraction for decimal odds, but adjust the user's supplied `prob`
  by incorporating recent closed bets that match the same sport and bet_type.

  Approach:
  - Find closed bets matching the same sport and bet_type (closed_bets are SQLAlchemy objects).
  - Compute an exponentially-decaying weight by recency (more recent = larger weight).
  - Compute weighted empirical win rate from those bets.
  - Blend the user's `prob` with the empirical win rate:
      adjusted_p = alpha * prob + (1 - alpha) * empirical_win_rate
    where alpha controls how much we trust the user's estimate (0..1).
  - Enforce a minimum probability of 0.5.
  - Use adjusted_p in the standard Kelly formula.
  """
  if odds is None or prob is None:
    return 0.0
  try:
    b = float(odds) - 1.0
    p = float(prob)
  except Exception:
    return 0.0
  if b <= 0:
    return 0.0

  # Parameters you can tweak:
  ALPHA = 0.6            # trust weight for user's supplied prob (higher = trust user more)
  TAU_DAYS = 30.0        # recency time-constant in days for exponential weighting
  MIN_PROB = 0.5         # minimum allowed probability after adjustment
  MAX_PROB = 0.95        # optional upper cap to avoid extreme overconfidence

  # If closed_bets provided, filter for same sport & bet_type (if those attrs exist)
  try:
    now = datetime.utcnow()
    weights_sum = 0.0
    weighted_wins = 0.0
    for cb in closed_bets or []:
      # only consider bets that have sport and bet_type attributes and match incoming context
      # if cb lacks sport/bet_type or incoming p doesn't include them, skip (caller must pass matching closed_bets)
      try:
        same_sport = getattr(cb, 'sport', None)
        same_type = getattr(cb, 'bet_type', None)
      except Exception:
        same_sport = same_type = None

      # We expect the caller to filter closed_bets to the relevant sport/type,
      # but in case they pass all closed_bets, only include matching ones.
      # If the incoming prob came from a bet context (not provided here), we cannot match; rely on user's p.
      # To keep this function self-contained we only use cb if its sport/type match the prob-bearing context:
      # We'll attempt to read sport/type from the closed bet objects; if they are empty we skip.
      # (The caller currently passes all closed_bets; we'll match to cb.sport/cb.bet_type if available.)
      # If no matching closed bets are present, we'll skip the empirical update below.
      # Note: The function call site should be updated to pass relevant sport/type in future for stronger effects.

      # Determine age in days
      try:
        age_days = max(0.0, (now - (cb.closed_at or now)).total_seconds() / 86400.0)
      except Exception:
        age_days = 0.0

      # Only include bets that have non-empty sport and bet_type and match p's context.
      # Here we don't have the incoming sport/type as parameters, so include all bets that have sport/type set.
      if not same_sport and not same_type:
        continue

      # Compute exponential decay weight
      tau = TAU_DAYS
      weight = math.exp(- age_days / tau)
      weights_sum += weight
      if getattr(cb, 'outcome', '') == 'win':
        weighted_wins += weight
  except Exception:
    weights_sum = 0.0
    weighted_wins = 0.0

  empirical = None
  if weights_sum > 0.0:
    empirical = weighted_wins / weights_sum

  # Blend user's prob with empirical recent performance when empirical exists
  if empirical is not None:
    adjusted_p = ALPHA * p + (1.0 - ALPHA) * empirical
  else:
    adjusted_p = p

  # Enforce minimum probability of MIN_PROB
  adjusted_p = max(adjusted_p, MIN_PROB)
  adjusted_p = min(adjusted_p, MAX_PROB)

  # Compute Kelly using adjusted probability
  f = (b * adjusted_p - (1 - adjusted_p)) / b
  f = max(0.0, f)  # no negative bets
  raw_stake = f * bankroll
  cap = percent_bankroll * bankroll
  recommended = min(raw_stake, cap)
  # prevent tiny fractional amounts (minimum $0.10)
  recommended = max(recommended, 0.10)
  return round(recommended, 2)



# Routes
@app.route('/')
def index():
  settings = Setting.query.first()
  open_bets = OpenBet.query.order_by(OpenBet.created_at.desc()).all()
  closed_bets = ClosedBet.query.order_by(ClosedBet.closed_at.desc()).all()
  return render_template('index.html', settings=settings, open_bets=open_bets, closed_bets=closed_bets, recommended=None)

@app.route('/save_settings', methods=['POST'])
def save_settings():
  s = Setting.query.first()
  if not s:
    s = Setting()
    db.session.add(s)
  try:
    s.bankroll = float(request.form.get('bankroll', s.bankroll))
    s.percent_bankroll = float(request.form.get('percent_bankroll', s.percent_bankroll))
  except Exception:
    pass
  db.session.commit()
  return redirect(url_for('index'))

# Calculation has been moved to client-side JavaScript.
# Provide a lightweight JSON API as an optional fallback for client-side fetch() calls.
@app.route('/api/calc', methods=['POST'])
def api_calc():
  settings = Setting.query.first()
  # Accept JSON or form-encoded payloads
  if request.is_json:
    data = request.get_json(silent=True) or {}
    odds = data.get('odds')
    prob = data.get('prob')
  else:
    odds = request.form.get('odds')
    prob = request.form.get('prob')

  try:
    odds = float(odds)
    prob = float(prob)
  except Exception:
    return {'recommended': 0.0}

  closed_bets = ClosedBet.query.all()
  recommended = compute_recommended_amount(settings.bankroll, settings.percent_bankroll, odds, prob, closed_bets)
  return {'recommended': recommended}

@app.route("/api/empirical_info", methods=["POST"])
def empirical_info():
  data = request.get_json()

  sport = data.get("sport", "")
  bet_type = data.get("bet_type", "")
  prob = float(data.get("prob", 0.5))

  # TODO: Replace with your real logic:
  # Here we just return some dummy values.
  empirical = 0.5
  adjusted = prob * 0.9
  alpha = 1.3
  matching_count = 12

  return jsonify({
      "empirical": empirical,
      "adjusted": adjusted,
      "alpha": alpha,
      "matching_count": matching_count
  })

@app.route('/add_open', methods=['POST'])
def add_open():
  try:
    name = request.form.get('name', 'Bet')
    eventstart_str = request.form.get('eventstart')
    odds = float(request.form.get('odds'))
    prob = float(request.form.get('prob'))
    stake = float(request.form.get('stake'))
    sport = request.form.get('sport', '')
    bet_type = request.form.get('bet_type', 'Moneyline')
  except Exception:
    return redirect(url_for('index'))
  
  eventstart_dt = datetime.fromisoformat(eventstart_str)
  b = OpenBet(name=name, odds=odds, prob=prob, stake=stake, sport=sport, bet_type=bet_type, eventstart=eventstart_dt)
  db.session.add(b)
  db.session.commit()
  return redirect(url_for('index'))

@app.route('/edit_open/<int:bet_id>', methods=['GET', 'POST'])
def edit_open(bet_id):
  b = OpenBet.query.get_or_404(bet_id)
  if request.method == 'POST':
    try:
      b.name = request.form.get('name', b.name)
      b.odds = float(request.form.get('odds', b.odds))
      b.prob = float(request.form.get('prob', b.prob))
      b.stake = float(request.form.get('stake', b.stake))
      b.sport = request.form.get('sport', b.sport)                # NEW
      b.bet_type = request.form.get('bet_type', b.bet_type)       # NEW
      db.session.commit()
      return redirect(url_for('index'))
    except Exception:
      pass
  return render_template('edit_open.html', b=b)

@app.route('/delete_open/<int:bet_id>', methods=['POST'])
def delete_open(bet_id):
  b = OpenBet.query.get_or_404(bet_id)
  db.session.delete(b)
  db.session.commit()
  return redirect(url_for('index'))

@app.route('/close_open/<int:bet_id>', methods=['POST'])
def close_open(bet_id):
  b = OpenBet.query.get_or_404(bet_id)
  outcome = request.form.get('outcome', 'loss')
  if outcome not in ('win', 'loss'):
    outcome = 'loss'
  profit = 0.0
  if outcome == 'win':
    profit = b.stake * (b.odds - 1.0)
  else:
    profit = -b.stake
  cb = ClosedBet(name=b.name, odds=b.odds, prob=b.prob, stake=b.stake,
                 sport=b.sport, bet_type=b.bet_type,
                 outcome=outcome, profit=profit, closed_at=datetime.utcnow())
  db.session.add(cb)
  db.session.delete(b)
  db.session.commit()
  # NOTE: Here is a good place to trigger re-calibration using closed bets in the future.
  return redirect(url_for('index'))

@app.route('/add_closed', methods=['POST'])
def add_closed():
  try:
    name = request.form.get('name', 'Bet')
    eventstart_str = request.form.get('eventstart')
    american_odds = float(request.form.get('american_odds'))
    # Convert American odds to decimal
    if american_odds > 0:
      odds = american_odds / 100.0 + 1.0
    else:
      odds = 100.0 / abs(american_odds) + 1.0
    prob = float(request.form.get('prob'))
    stake = float(request.form.get('stake'))
    sport = request.form.get('sport', '')
    bet_type = request.form.get('bet_type', 'Moneyline')
    outcome = request.form.get('outcome', 'loss')
    closed_at_raw = request.form.get('closed_at', '')
  except Exception:
    return redirect(url_for('index'))

  if outcome not in ('win', 'loss'):
    outcome = 'loss'

  if outcome == 'win':
    profit = stake * (odds - 1.0)
  else:
    profit = -stake

  eventstart_dt = datetime.fromisoformat(eventstart_str)

  closed_at = datetime.utcnow()
  if closed_at_raw:
    try:
      # input type="datetime-local" -> "YYYY-MM-DDTHH:MM"
      closed_at = datetime.strptime(closed_at_raw, "%Y-%m-%dT%H:%M")
    except Exception:
      pass

  cb = ClosedBet(
    name=name, odds=odds, prob=prob, stake=stake,
    sport=sport, bet_type=bet_type, outcome=outcome,
    profit=profit, closed_at=closed_at, eventstart=eventstart_dt
  )
  db.session.add(cb)
  db.session.commit()
  return redirect(url_for('index'))

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=5000, debug=True)