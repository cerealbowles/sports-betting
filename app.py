from flask import Flask, request, redirect, url_for, render_template_string, abort
from flask_sqlalchemy import SQLAlchemy
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
  bankroll = db.Column(db.Float, default=1000.0)
  percent_bankroll = db.Column(db.Float, default=0.02)  # fraction of bankroll as cap

class OpenBet(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(200))
  odds = db.Column(db.Float)  # decimal odds
  prob = db.Column(db.Float)  # user's estimated win probability (0-1)
  stake = db.Column(db.Float)
  sport = db.Column(db.String(100), default='')     # NEW: sport text
  bet_type = db.Column(db.String(50), default='Moneyline')  # NEW: Moneyline/Spread/Over/Under/Player
  created_at = db.Column(db.DateTime, default=datetime.utcnow)

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

# Create DB if missing
with app.app_context():
  if not os.path.exists('bets.db'):
    db.create_all()
    db.session.add(Setting(bankroll=1000.0, percent_bankroll=0.02))
    db.session.commit()

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

# Templates (single-file approach)
BASE = """
<!doctype html>
<title>Spooky Sports</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
input, select { padding: 6px; margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin-top: 10px; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background: #f2f2f2; }
button { padding: 6px 10px; }
.form-row { margin-bottom: 8px; }
.small { font-size: 0.9em; color: #555; }
</style>
<h1>Kelly Betting Manager</h1>

<script>
// Convert American odds (e.g. -120, +150) to decimal odds
function americanToDecimal(a) {
  var n = Number(a);
  if (!isFinite(n)) return null;
  if (n > 0) {
    return +(n / 100 + 1).toFixed(4);
  } else {
    return +(100 / Math.abs(n) + 1).toFixed(4);
  }
}

// Kelly calculation in JavaScript
function computeKelly(bankroll, percentCap, odds, prob) {
  if (!odds || !prob || odds <= 0 || prob <= 0 || prob >= 1) {
    return 0.0;
  }
  var b = odds - 1.0;
  if (b <= 0) return 0.0;
  var f = (b * prob - (1 - prob)) / b;
  f = Math.max(0.0, f);
  var rawStake = f * bankroll;
  var cap = percentCap * bankroll;
  var recommended = Math.min(rawStake, cap);
  // Set minimum to $0.10
  recommended = Math.max(recommended, 0.10);
  return Math.round(recommended * 100) / 100;
}

document.addEventListener('DOMContentLoaded', function(){
  var sportInput = document.getElementById('sport');
  var typeInput = document.getElementById('bet_type');
  var probInput = document.getElementById('prob');
  var aInput = document.getElementById('american_odds');
  var outSpan = document.getElementById('converted_decimal');
  var recommendedSpan = document.getElementById('recommended_stake');
  var oddsHidden = document.getElementById('odds_hidden');
  
  var bankroll = parseFloat(document.getElementById('bankroll_value').textContent);
  var percentCap = parseFloat(document.getElementById('percent_cap_value').textContent);

  function updateProbFromHistory() {
    var sport = sportInput.value.trim();
    var betType = typeInput.value;
    
    if (!sport || !betType) {
      probInput.value = '0.5';
      return;
    }

    // Fetch empirical win rate from server
    fetch('/api/empirical_prob', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sport: sport, bet_type: betType })
    })
    .then(r => r.json())
    .then(data => {
      if (data.empirical_prob !== undefined) {
        probInput.value = Math.max(0.5, data.empirical_prob).toFixed(4);
        updateRecommended();
      }
    })
    .catch(err => console.log('Error fetching empirical prob:', err));
  }

  function updateRecommended() {
    var dec = americanToDecimal(aInput.value);
    var p = parseFloat(probInput.value);
    if (dec === null || !isFinite(p)) {
      recommendedSpan.textContent = '—';
      oddsHidden.value = '';
      return;
    }
    outSpan.textContent = dec;
    var rec = computeKelly(bankroll, percentCap, dec, p);
    recommendedSpan.textContent = rec.toFixed(2);
    oddsHidden.value = dec;
  }

  if (sportInput) {
    sportInput.addEventListener('input', updateProbFromHistory);
  }
  if (typeInput) {
    typeInput.addEventListener('change', updateProbFromHistory);
  }
  if (aInput) {
    aInput.addEventListener('input', updateRecommended);
  }
  if (probInput) {
    probInput.addEventListener('input', updateRecommended);
  }
});
</script>

<div>
  <h2>Account Settings</h2>
  <form method="post" action="{{ url_for('save_settings') }}">
  <div class="form-row">
    Bankroll: <input name="bankroll" type="number" step="0.01" value="{{ settings.bankroll }}">
    Percent cap of bankroll (0-1): <input name="percent_bankroll" type="number" step="0.0001" value="{{ settings.percent_bankroll }}">
    <button type="submit">Save</button>
  </div>
  </form>
  <p class="small">Percent cap is a hard cap on stake (e.g., 0.02 = 2% of bankroll)</p>
  <!-- Hidden spans to pass settings to JavaScript -->
  <span id="bankroll_value" style="display:none;">{{ settings.bankroll }}</span>
  <span id="percent_cap_value" style="display:none;">{{ settings.percent_bankroll }}</span>
</div>

<hr>
<div>
  <h2>Kelly Calculator</h2>
  <div class="form-row">
    Bet name: <input id="bet_name" name="name" required>
  </div>
  <div class="form-row">
    Sport: <input id="sport" name="sport" type="text" placeholder="e.g. NBA, NFL">
  </div>
  <div class="form-row">
    Type:
    <select id="bet_type" name="bet_type">
      <option value="Moneyline">Moneyline</option>
      <option value="Spread">Spread</option>
      <option value="Over/Under">Over/Under</option>
      <option value="Player">Player</option>
    </select>
  </div>
  <div class="form-row">
    American odds (e.g. -120 or 150): <input id="american_odds" name="american_odds" type="number" step="1" required>
    <span class="small">Converted decimal: <strong id="converted_decimal">—</strong></span>
  </div>
  <div class="form-row">
    Your estimated win probability (0-1): <input id="prob" name="prob" type="number" step="0.0001" value="0.5" required>
  </div>
  <div class="form-row">
    <strong>Recommended stake: <span id="recommended_stake">—</span></strong>
  </div>
  <div class="form-row">
    Actual stake to place: <input id="actual_stake" name="stake" type="number" step="0.01" required>
  </div>
  <input type="hidden" id="odds_hidden" name="odds" value="">
  <div class="form-row">
    <form id="betForm" method="post" action="{{ url_for('add_open') }}">
      <input type="hidden" id="form_name" name="name">
      <input type="hidden" id="form_odds" name="odds">
      <input type="hidden" id="form_prob" name="prob">
      <input type="hidden" id="form_stake" name="stake">
      <input type="hidden" id="form_sport" name="sport">
      <input type="hidden" id="form_type" name="bet_type">
      <button type="submit">Place Bet</button>
    </form>
  </div>
  <p class="small">Kelly recommends the stake above; you can adjust it manually and place a different amount.</p>
</div>

<script>
document.getElementById('betForm').addEventListener('submit', function(e){
  var name = document.getElementById('bet_name').value;
  var odds = document.getElementById('odds_hidden').value;
  var prob = document.getElementById('prob').value;
  var stake = document.getElementById('actual_stake').value;
  var sport = document.getElementById('sport').value || '';
  var type = document.getElementById('bet_type').value || 'Moneyline';
  
  if (!name || !odds || !prob || !stake) {
    alert('Please fill in all fields.');
    e.preventDefault();
    return;
  }
  
  document.getElementById('form_name').value = name;
  document.getElementById('form_odds').value = odds;
  document.getElementById('form_prob').value = prob;
  document.getElementById('form_stake').value = stake;
  document.getElementById('form_sport').value = sport;
  document.getElementById('form_type').value = type;
});
</script>

<hr>
<div>
  <h2>Open Bets</h2>
  {% if open_bets %}
  <table>
  <tr>
    <th>When</th><th>Sport</th><th>Type</th><th>Name</th><th>Odds</th><th>Prob</th><th>Stake</th>
    <th>Winnings</th><th>Total Return</th><th>Actions</th>
  </tr>
  {% for b in open_bets %}
    <tr>
      <td>{{ b.created_at }}</td>
      <td>{{ b.sport }}</td>
      <td>{{ b.bet_type }}</td>
      <td>{{ b.name }}</td>
      <td>{{ "%.2f"|format(b.odds) }}</td>
      <td>{{ "%.4f"|format(b.prob) }}</td>
      <td>{{ "%.2f"|format(b.stake) }}</td>
      <td>{{ "%.2f"|format(b.stake * (b.odds - 1.0)) }}</td>
      <td>{{ "%.2f"|format(b.stake * b.odds) }}</td>
      <td>
        <a href="{{ url_for('edit_open', bet_id=b.id) }}">Edit</a> |
        <form style="display:inline" method="post" action="{{ url_for('close_open', bet_id=b.id) }}">
        Outcome:
        <select name="outcome">
          <option value="win">win</option>
          <option value="loss">loss</option>
        </select>
        <button type="submit">Close</button>
        </form>
        |
        <form style="display:inline" method="post" action="{{ url_for('delete_open', bet_id=b.id) }}" onsubmit="return confirm('Delete this open bet?');">
        <button type="submit">Delete</button>
        </form>
      </td>
    </tr>
  {% endfor %}
  </table>
  {% else %}
  <p>No open bets.</p>
  {% endif %}
</div>

<hr>
<div>
  <h2>Closed Bets (History)</h2>
  <p class="small">Closed bets can be used to refine the Kelly parameters. See compute_recommended_amount() and TODO in the code.</p>

  <!-- Add existing closed bet form -->
  <h3>Add closed bet</h3>
  <form method="post" action="{{ url_for('add_closed') }}">
    <div class="form-row">Name: <input name="name" required></div>
    <div class="form-row">Sport: <input name="sport" placeholder="e.g. NBA, NFL"></div>
    <div class="form-row">Type:
      <select name="bet_type">
        <option value="Moneyline">Moneyline</option>
        <option value="Spread">Spread</option>
        <option value="Over/Under">Over/Under</option>
        <option value="Player">Player</option>
      </select>
    </div>
    <div class="form-row">American odds (e.g. -120 or 150): <input name="american_odds" type="number" step="1" required></div>
    <div class="form-row">Prob (0-1): <input name="prob" type="number" step="0.0001" value="0.5" required></div>
    <div class="form-row">Stake: <input name="stake" type="number" step="0.01" required></div>
    <div class="form-row">Outcome:
      <select name="outcome">
        <option value="win">win</option>
        <option value="loss">loss</option>
      </select>
    </div>
    <div class="form-row">Closed at (optional): <input name="closed_at" type="datetime-local"></div>
    <button type="submit">Add Closed Bet</button>
  </form>

  {% if closed_bets %}
  <table>
    <tr>
      <th>When</th><th>Sport</th><th>Type</th><th>Name</th><th>Odds</th><th>Prob</th><th>Stake</th><th>Outcome</th><th>Winnings</th>
    </tr>
    {% for cb in closed_bets %}
    <tr>
      <td>{{ cb.closed_at }}</td>
      <td>{{ cb.sport }}</td>
      <td>{{ cb.bet_type }}</td>
      <td>{{ cb.name }}</td>
      <td>{{ "%.2f"|format(cb.odds) }}</td>
      <td>{{ "%.4f"|format(cb.prob) }}</td>
      <td>{{ "%.2f"|format(cb.stake) }}</td>
      <td>{{ cb.outcome }}</td>
      <td>{{ "%.2f"|format(cb.profit) }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p>No closed bets yet.</p>
  {% endif %}
</div>
"""

# Routes
@app.route('/')
def index():
  settings = Setting.query.first()
  open_bets = OpenBet.query.order_by(OpenBet.created_at.desc()).all()
  closed_bets = ClosedBet.query.order_by(ClosedBet.closed_at.desc()).all()
  return render_template_string(BASE, settings=settings, open_bets=open_bets, closed_bets=closed_bets, recommended=None)

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

@app.route('/add_open', methods=['POST'])
def add_open():
  try:
    name = request.form.get('name', 'Bet')
    odds = float(request.form.get('odds'))
    prob = float(request.form.get('prob'))
    stake = float(request.form.get('stake'))
    sport = request.form.get('sport', '')                  # NEW
    bet_type = request.form.get('bet_type', 'Moneyline')  # NEW
  except Exception:
    return redirect(url_for('index'))
  b = OpenBet(name=name, odds=odds, prob=prob, stake=stake, sport=sport, bet_type=bet_type)
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
  # simple edit form
  form = """
  <h2>Edit Open Bet</h2>
  <form method="post">
    Name: <input name="name" value="{{ b.name }}"><br>
    Sport: <input name="sport" value="{{ b.sport }}"><br>
    Type:
    <select name="bet_type">
      <option value="Moneyline" {% if b.bet_type == 'Moneyline' %}selected{% endif %}>Moneyline</option>
      <option value="Spread" {% if b.bet_type == 'Spread' %}selected{% endif %}>Spread</option>
      <option value="Over/Under" {% if b.bet_type == 'Over/Under' %}selected{% endif %}>Over/Under</option>
    </select><br>
    Odds: <input name="odds" type="number" step="0.01" value="{{ b.odds }}"><br>
    Prob: <input name="prob" type="number" step="0.0001" value="{{ b.prob }}"><br>
    Stake: <input name="stake" type="number" step="0.01" value="{{ b.stake }}"><br>
    <button type="submit">Save</button>
    <a href="{{ url_for('index') }}">Cancel</a>
  </form>
  """
  return render_template_string(form, b=b)

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
                 sport=b.sport, bet_type=b.bet_type,            # NEW: carry sport/type
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
    profit=profit, closed_at=closed_at
  )
  db.session.add(cb)
  db.session.commit()
  return redirect(url_for('index'))

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=5000, debug=True)