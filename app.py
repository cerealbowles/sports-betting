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
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spooky Sports</title>
<style>
/* Reset / basics */
* { box-sizing: border-box; margin: 0; padding: 0; }
html,body { height: 100%; }
body {
  background: #11121a;
  color: #e6e6e6;
  font-family: Georgia, 'Times New Roman', serif;
  padding: 14px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Container */
.container {
  max-width: 1100px;
  margin: 0 auto;
}

/* Header */
h1 {
  color: #ff6b35;
  text-align: center;
  font-size: 1.9rem;
  margin-bottom: 10px;
  text-shadow: 0 0 10px rgba(255,107,53,0.35);
}
h2 { color: #ff6b35; margin-bottom: 8px; font-size: 1.05rem; }
h3 { color: #ffd93d; margin-bottom: 8px; font-size: 0.95rem; }

/* Card styles */
.card {
  background: linear-gradient(180deg, #1f1f2e 0%, #262636 100%);
  border: 1px solid rgba(255,107,53,0.08);
  padding: 12px;
  border-radius: 8px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.45);
  margin-bottom: 12px;
}

/* Settings bar (compact) */
.settings-bar {
  display: flex;
  gap: 8px;
  align-items: center;
  justify-content: space-between;
  padding: 10px;
  border-radius: 8px;
  background: #252535;
  margin-bottom: 12px;
  border: 1px solid rgba(255,107,53,0.06);
}
.settings-left { display:flex; gap:8px; align-items:center; flex:1; }
.settings-left .form-row { display:flex; gap:6px; align-items:center; }
.settings-left label { color:#ffd93d; font-weight:bold; font-size:0.9rem; white-space:nowrap; }
.settings-left input[type="text"], .settings-left input[type="number"] {
  padding:6px 8px; border-radius:6px; border:1px solid #3a3a4a; background:#151522; color:#e6e6e6; width:110px;
}

/* Grid layout */
.main-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: 1fr; /* mobile first */
}

/* Calculator (left) */
.calculator {
  padding: 10px;
}

/* Dashboard (right) */
.dashboard {
  display:flex; flex-direction:column; gap:8px;
}
.chart {
  padding:10px;
  border-radius:8px;
  background: linear-gradient(180deg,#23232e,#2b2b40);
  border:1px solid rgba(255,107,53,0.06);
}

/* Forms */
.form-row { margin-bottom:10px; display:flex; flex-direction:column; gap:6px; }
.form-row label { color:#ffd93d; font-weight:bold; font-size:0.95rem; }
input[type="text"], input[type="number"], select, input[type="datetime-local"] {
  padding:8px 10px; border-radius:6px; border:1px solid #3a3a4a; background:#151522; color:#e6e6e6; font-size:0.95rem;
  -webkit-appearance:none;
}
input:focus, select:focus {
  outline: none; box-shadow: 0 0 8px rgba(255,107,53,0.12); border-color: #ff6b35;
}

/* Buttons */
button {
  background: #ff6b35; color: #14141a; border: none; padding: 8px 12px; border-radius:6px; cursor:pointer; font-weight:700;
}
button.secondary { background:#33343f; color:#ffd93d; border:1px solid rgba(255,217,61,0.06); }

/* Tables */
.table {
  width:100%; border-collapse:collapse; margin-top:8px; font-size:0.93rem;
}
.table th {
  text-align:left; padding:8px; background:#13131a; color:#ffd93d; font-size:0.78rem; border-bottom:1px solid rgba(255,107,53,0.07);
}
.table td { padding:8px; border-bottom:1px solid rgba(255,255,255,0.03); color:#e6e6e6; vertical-align:middle; }
.small { color:#9aa0a6; font-size:0.85rem; }

/* Status */
.status-win { color:#22c55e; font-weight:700; }
.status-loss { color:#ff4757; font-weight:700; }

/* Actions compact */
.actions { display:flex; gap:6px; align-items:center; }

/* Responsive rules */
@media (min-width: 768px) {
  .main-grid { grid-template-columns: 1fr 1fr; }
  .calculator { grid-column: 1 / 2; }
  .dashboard { grid-column: 2 / 3; }
  .settings-bar { padding:12px; }
}

/* Mobile-specific simplifications */
@media (max-width: 767px) {
  body { padding: 12px; }
  h1 { font-size: 1.4rem; }
  .settings-left input[type="text"] { width: 100px; }
  .hide-mobile { display: none !important; }
  .actions { flex-direction: column; gap:6px; }
}

/* Subtle spooky accent */
.spooky { text-shadow: 0 0 8px rgba(255,107,53,0.25); color:#ff8a50; }
</style>
</head>
<body>
<div class="container">
  <h1>👻 Spooky Sports 🎃</h1>

  <div class="settings-bar card">
    <div class="settings-left">
      <form method="post" action="{{ url_for('save_settings') }}" style="display:flex; gap:8px; align-items:center;">
        <div class="form-row" style="margin:0;">
          <label style="margin-right:6px;">Bankroll</label>
          <input name="bankroll" type="text" inputmode="decimal" value="{{ '%.2f' % settings.bankroll }}">
        </div>
        <div class="form-row" style="margin:0;">
          <label style="margin-right:6px;">Cap %</label>
          <input name="percent_bankroll" type="number" step="0.0001" value="{{ settings.percent_bankroll }}">
        </div>
        <button type="submit">Save</button>
      </form>
    </div>
    <div class="small">Bankroll is updated automatically when placing/closing bets.</div>
    <span id="bankroll_value" style="display:none;">{{ settings.bankroll }}</span>
    <span id="percent_cap_value" style="display:none;">{{ settings.percent_bankroll }}</span>
  </div>

  <div class="main-grid">
    <div class="calculator card calculator">
      <h2>🔮 New Bet</h2>
      <div class="form-row">
        <label>Bet name</label>
        <input id="bet_name" name="name" type="text" required>
      </div>
      <div class="form-row">
        <label>Sport</label>
        <input id="sport" name="sport" type="text" placeholder="NBA, NFL...">
      </div>
      <div class="form-row">
        <label>Type</label>
        <select id="bet_type" name="bet_type">
          <option value="Moneyline">Moneyline</option>
          <option value="Spread">Spread</option>
          <option value="Over/Under">Over/Under</option>
          <option value="Player">Player</option>
        </select>
      </div>
      <div class="form-row">
        <label>American odds (e.g. -120 or 150)</label>
        <input id="american_odds" name="american_odds" type="number" step="1" required placeholder="-120 or 150">
        <div class="small">Decimal: <strong id="converted_decimal">—</strong></div>
      </div>
      <div class="form-row">
        <label>Win probability (0-1)</label>
        <div style="display:flex; gap:8px; align-items:center;">
          <input id="prob" name="prob" type="number" step="0.0001" value="0.5" required style="flex:0 0 120px;">
          <div id="empirical_info" class="small" style="flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">Calc: —</div>
        </div>
      </div>
      <div class="form-row">
        <label>Recommended stake</label>
        <div><strong id="recommended_stake">—</strong></div>
      </div>
      <div class="form-row">
        <label>Actual stake</label>
        <input id="actual_stake" name="stake" type="number" step="0.01" required>
      </div>
      <input type="hidden" id="odds_hidden" name="odds" value="">
      <div style="margin-top:8px;">
        <form id="betForm" method="post" action="{{ url_for('add_open') }}">
          <input type="hidden" id="form_name" name="name">
          <input type="hidden" id="form_odds" name="odds">
          <input type="hidden" id="form_prob" name="prob">
          <input type="hidden" id="form_stake" name="stake">
          <input type="hidden" id="form_sport" name="sport">
          <input type="hidden" id="form_type" name="bet_type">
          <button type="submit" style="width:100%;">Place Bet</button>
        </form>
      </div>
    </div>

    <div class="dashboard">
      <div class="chart card chart">
        <h3>Bets (7D)</h3>
        <div class="small">Bar chart disabled in this lightweight view.</div>
      </div>
      <div class="chart card chart">
        <h3>Success Rate</h3>
        <div class="small">Bar chart disabled in this lightweight view.</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>🔓 Open Bets</h2>
    {% if open_bets %}
      <table class="table">
        <thead>
          <tr>
            <th class="hide-mobile">When</th>
            <th>Name</th>
            <th>Stake</th>
            <th>Return</th>
            <th>⚙️</th>
          </tr>
        </thead>
        <tbody>
          {% for b in open_bets %}
          <tr>
            <td class="hide-mobile">{{ b.created_at.strftime('%m/%d %H:%M') }}</td>
            <td>{{ b.name }}</td>
            <td>${{ "%.2f"|format(b.stake) }}</td>
            <td>${{ "%.2f"|format(b.stake * b.odds) }}</td>
            <td class="actions">
              <a href="{{ url_for('edit_open', bet_id=b.id) }}" class="small">Edit</a>
              <form method="post" action="{{ url_for('close_open', bet_id=b.id) }}" style="display:inline-flex; gap:6px; align-items:center;">
                <select name="outcome" style="padding:6px; border-radius:6px; background:#151522; color:#e6e6e6;">
                  <option value="win">Win</option>
                  <option value="loss">Loss</option>
                </select>
                <button type="submit" class="small">OK</button>
              </form>
              <form method="post" action="{{ url_for('delete_open', bet_id=b.id) }}" onsubmit="return confirm('Delete?');" style="display:inline-block;">
                <button type="submit" class="small" style="background:#ff4757;">Del</button>
              </form>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    {% else %}
      <div class="small">No open bets yet.</div>
    {% endif %}
  </div>

  <div class="card">
    <h2>🔒 Closed Bets</h2>
    <div style="margin-bottom:8px;"><button onclick="document.getElementById('addClosedBtn').scrollIntoView({behavior:'smooth'});">+ Add Bet</button></div>
    {% if closed_bets %}
      <table class="table">
        <thead>
          <tr>
            <th class="hide-mobile">When</th>
            <th>Name</th>
            <th>Stake</th>
            <th>Result</th>
            <th>P&L</th>
          </tr>
        </thead>
        <tbody>
          {% for cb in closed_bets %}
          <tr>
            <td class="hide-mobile">{{ cb.closed_at.strftime('%m/%d %H:%M') }}</td>
            <td>{{ cb.name }}</td>
            <td>${{ "%.2f"|format(cb.stake) }}</td>
            <td class="{% if cb.outcome == 'win' %}status-win{% else %}status-loss{% endif %}">{{ cb.outcome.upper() }}</td>
            <td class="{% if cb.profit >= 0 %}status-win{% else %}status-loss{% endif %}">${{ "%.2f"|format(cb.profit) }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    {% else %}
      <div class="small">No closed bets yet.</div>
    {% endif %}
  </div>

  <div id="addClosedBtn" class="card" style="margin-top:12px;">
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
      <div class="form-row">American odds: <input name="american_odds" type="number" step="1" required></div>
      <div class="form-row">Prob (0-1): <input name="prob" type="number" step="0.0001" value="0.5" required></div>
      <div class="form-row">Stake: <input name="stake" type="number" step="0.01" required></div>
      <div class="form-row">Outcome:
        <select name="outcome">
          <option value="win">win</option>
          <option value="loss">loss</option>
        </select>
      </div>
      <div class="form-row">Closed at (optional): <input name="closed_at" type="datetime-local"></div>
      <button type="submit" style="width:100%;">Add Closed Bet</button>
    </form>
  </div>
</div>

<script>
/* Utility: convert American -> decimal */
function americanToDecimal(a) {
  var n = Number(a);
  if (!isFinite(n)) return null;
  if (n > 0) return +(n/100 + 1).toFixed(4);
  return +(100/Math.abs(n) + 1).toFixed(4);
}

/* compute recommended stake (client mirror of server logic) */
function computeKelly(bankroll, percentCap, odds, prob) {
  if (!odds || !prob || odds <= 0 || prob <= 0 || prob >= 1) return 0.0;
  var b = odds - 1.0;
  if (b <= 0) return 0.0;
  var f = (b * prob - (1 - prob)) / b;
  f = Math.max(0.0, f);
  var rawStake = f * bankroll;
  var cap = percentCap * bankroll;
  var recommended = Math.min(rawStake, cap);
  recommended = Math.max(recommended, 0.10);
  return Math.round(recommended * 100) / 100;
}

/* Empirical info fetch (small summary next to probability) */
function updateEmpiricalInfo() {
  var sport = (document.getElementById('sport')||{}).value || '';
  var betType = (document.getElementById('bet_type')||{}).value || '';
  var prob = (document.getElementById('prob')||{}).value || '0.5';
  fetch('/api/empirical_info', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ sport: sport, bet_type: betType, prob: prob })
  }).then(r=>r.json()).then(function(data){
    var el = document.getElementById('empirical_info');
    if(!el) return;
    if(data.empirical === null || data.matching_count === 0) {
      el.textContent = 'Empirical: —   Adjusted: ' + (data.adjusted!==undefined?Number(data.adjusted).toFixed(4):'—');
    } else {
      el.textContent = 'Empirical: ' + Number(data.empirical).toFixed(4) +
        '  Adjusted: ' + Number(data.adjusted).toFixed(4) +
        '  (α=' + Number(data.alpha).toFixed(2) + ', n=' + (data.matching_count||0) + ')';
    }
  }).catch(function(){});
}

/* Wire up inputs */
document.addEventListener('DOMContentLoaded', function(){
  var aInput = document.getElementById('american_odds');
  var probInput = document.getElementById('prob');
  var outSpan = document.getElementById('converted_decimal');
  var recommendedSpan = document.getElementById('recommended_stake');
  var bankroll = parseFloat(document.getElementById('bankroll_value').textContent || '0');
  var percentCap = parseFloat(document.getElementById('percent_cap_value').textContent || '0.02');

  function updateRecommended() {
    var dec = americanToDecimal(aInput.value);
    var p = parseFloat(probInput.value);
    if (dec === null || !isFinite(p)) {
      recommendedSpan.textContent = '—';
      document.getElementById('odds_hidden').value = '';
      return;
    }
    outSpan.textContent = dec;
    var rec = computeKelly(bankroll, percentCap, dec, p);
    recommendedSpan.textContent = '$' + rec.toFixed(2);
    document.getElementById('odds_hidden').value = dec;
  }

  if (aInput) aInput.addEventListener('input', updateRecommended);
  if (probInput) probInput.addEventListener('input', updateRecommended);

  /* empirical info updates */
  var sportEl = document.getElementById('sport');
  var typeEl = document.getElementById('bet_type');
  if (sportEl) sportEl.addEventListener('input', updateEmpiricalInfo);
  if (typeEl) typeEl.addEventListener('change', updateEmpiricalInfo);
  if (probInput) probInput.addEventListener('input', updateEmpiricalInfo);
  updateEmpiricalInfo();

  /* bet form submission fills hidden fields */
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
});
</script>
</body>
</html>
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