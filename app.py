from flask import Flask, request, redirect, url_for, render_template_string, abort
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
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
  bankroll = db.Column(db.Float, default=1000)
  percent_bankroll = db.Column(db.Float, default=0.25)  # fraction of bankroll as cap

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
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spooky Sports Betting</title>
<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Georgia', serif;
  background-color: #1a1a2e;
  background-image: 
    repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255,255,255,.02) 10px, rgba(255,255,255,.02) 20px),
    repeating-linear-gradient(-45deg, transparent, transparent 10px, rgba(255,255,255,.02) 10px, rgba(255,255,255,.02) 20px);
  color: #e0e0e0;
  min-height: 100vh;
  padding: 12px;
}

.container {
  max-width: 1400px;
  margin: 0 auto;
}

h1 {
  color: #ff6b35;
  margin-bottom: 12px;
  font-size: 1.8em;
  text-align: center;
  text-shadow: 0 0 10px rgba(255, 107, 53, 0.5);
  letter-spacing: 1px;
}

h2 {
  color: #ff6b35;
  margin-bottom: 12px;
  font-size: 1.2em;
  border-bottom: 2px solid #ff6b35;
  padding-bottom: 6px;
  text-shadow: 0 0 5px rgba(255, 107, 53, 0.3);
}

h3 {
  color: #ffd93d;
  margin-bottom: 10px;
  font-size: 0.95em;
}

/* Settings Bar */
.settings-bar {
  background-color: #2d2d44;
  border: 2px solid #444;
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 15px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
}

.settings-bar form {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  width: 100%;
}

.settings-bar .form-row {
  margin: 0;
  display: flex;
  gap: 4px;
  align-items: center;
  flex-direction: row;
  flex: 1;
  min-width: 120px;
}

.settings-bar label {
  color: #ffd93d;
  font-size: 0.8em;
  white-space: nowrap;
  font-weight: bold;
}

.settings-bar input {
  padding: 5px 8px;
  background-color: #1a1a2e;
  border: 2px solid #444;
  border-radius: 4px;
  color: #e0e0e0;
  font-size: 0.85em;
  flex: 1;
  min-width: 60px;
}

.settings-bar button {
  padding: 5px 12px;
  font-size: 0.8em;
  white-space: nowrap;
}

/* Main Layout */
.main-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 15px;
  margin-bottom: 15px;
}

/* Calculator Card */
.calculator-card {
  background-color: #2d2d44;
  border: 2px solid #444;
  border-radius: 8px;
  padding: 14px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

/* Dashboard Card */
.dashboard {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.chart-card {
  background-color: #2d2d44;
  border: 2px solid #ff6b35;
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 0 15px rgba(255, 107, 53, 0.2);
}

.chart-card h3 {
  margin-bottom: 8px;
}

.chart-card canvas {
  max-height: 180px;
}

.chart-card:hover {
  box-shadow: 0 0 25px rgba(255, 107, 53, 0.4);
}

/* Forms */
.form-row {
  margin-bottom: 10px;
  display: flex;
  flex-direction: column;
}

.form-row label {
  font-weight: bold;
  color: #ffd93d;
  margin-bottom: 4px;
  font-size: 0.9em;
}

input[type="text"],
input[type="number"],
input[type="datetime-local"],
select {
  padding: 8px 10px;
  background-color: #1a1a2e;
  border: 2px solid #444;
  border-radius: 4px;
  color: #e0e0e0;
  font-size: 0.95em;
  transition: border-color 0.3s ease;
  font-family: 'Georgia', serif;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
}

input[type="text"]:focus,
input[type="number"]:focus,
input[type="datetime-local"]:focus,
select:focus {
  outline: none;
  border-color: #ff6b35;
  box-shadow: 0 0 8px rgba(255, 107, 53, 0.3);
  background-color: #252540;
}

/* Buttons */
button {
  padding: 10px 16px;
  background-color: #ff6b35;
  color: #1a1a2e;
  border: none;
  border-radius: 4px;
  font-size: 0.95em;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  text-transform: uppercase;
  letter-spacing: 1px;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
}

button:active {
  background-color: #ffd93d;
  box-shadow: 0 0 15px rgba(255, 217, 61, 0.5);
  transform: scale(0.98);
}

a {
  color: #ff6b35;
  text-decoration: none;
  font-weight: bold;
  transition: color 0.3s ease;
}

a:active {
  color: #ffd93d;
  text-shadow: 0 0 5px rgba(255, 217, 61, 0.5);
}

/* Full Width Sections */
.full-width-card {
  background-color: #2d2d44;
  border: 2px solid #444;
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 15px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
  font-size: 0.85em;
}

th {
  background-color: #1a1a2e;
  color: #ffd93d;
  padding: 8px;
  text-align: left;
  font-weight: bold;
  border-bottom: 2px solid #ff6b35;
  text-transform: uppercase;
  font-size: 0.75em;
}

td {
  padding: 7px;
  border-bottom: 1px solid #444;
  font-size: 0.9em;
  word-break: break-word;
}

tr:hover {
  background-color: rgba(255, 107, 53, 0.1);
}

tr:last-child td {
  border-bottom: none;
}

.small {
  font-size: 0.75em;
  color: #999;
  margin-top: 4px;
  display: block;
}

/* Modal Styles */
.modal {
  display: none;
  position: fixed;
  z-index: 1000;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.7);
  animation: fadeIn 0.3s ease;
}

.modal.active {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 12px;
  overflow-y: auto;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.modal-content {
  background-color: #2d2d44;
  padding: 20px;
  border-radius: 8px;
  border: 2px solid #ff6b35;
  width: 100%;
  max-width: 500px;
  box-shadow: 0 0 30px rgba(255, 107, 53, 0.4);
  animation: slideIn 0.3s ease;
  margin-top: 20px;
}

@keyframes slideIn {
  from {
    transform: translateY(-50px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  border-bottom: 2px solid #ff6b35;
  padding-bottom: 10px;
}

.modal-header h2 {
  margin: 0;
  border: none;
  padding: 0;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: #ff6b35;
  padding: 0;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: none;
  transition: all 0.2s ease;
  -webkit-appearance: none;
}

.close-btn:active {
  color: #ffd93d;
  text-shadow: 0 0 8px rgba(255, 217, 61, 0.5);
}

/* Status Indicators */
.status-win {
  color: #22c55e;
  font-weight: bold;
}

.status-loss {
  color: #ff4757;
  font-weight: bold;
}

/* Actions Column */
.actions {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.85em;
}

.actions form {
  display: flex;
  gap: 2px;
  align-items: center;
  flex-wrap: wrap;
}

.actions select {
  padding: 4px 6px;
  font-size: 0.8em;
  flex: 1;
  min-width: 70px;
}

.actions button {
  padding: 4px 8px;
  font-size: 0.75em;
  white-space: nowrap;
}

.actions a {
  display: inline-block;
  padding: 4px 8px;
  border: 1px solid #ff6b35;
  border-radius: 3px;
  font-size: 0.75em;
  text-align: center;
}

@media (min-width: 768px) {
  h1 {
    font-size: 2.5em;
    margin-bottom: 15px;
  }

  .main-grid {
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  .settings-bar {
    padding: 12px 15px;
    margin-bottom: 25px;
    gap: 20px;
  }

  .settings-bar input {
    width: 120px;
    flex: none;
  }

  button {
    padding: 9px 18px;
    font-size: 0.95em;
  }

  button:hover {
    background-color: #ffd93d;
    box-shadow: 0 0 15px rgba(255, 217, 61, 0.5);
    transform: translateY(-2px);
  }

  a:hover {
    color: #ffd93d;
    text-shadow: 0 0 5px rgba(255, 217, 61, 0.5);
  }

  .calculator-card {
    padding: 18px;
  }

  .chart-card {
    padding: 15px;
  }

  .chart-card canvas {
    max-height: 200px;
  }

  table {
    font-size: 0.95em;
  }

  th, td {
    padding: 10px 12px;
  }

  th {
    font-size: 0.9em;
  }

  td {
    font-size: 0.95em;
  }

  .small {
    font-size: 0.85em;
    margin-top: 8px;
  }

  .modal-content {
    padding: 30px;
  }

  .actions {
    flex-direction: row;
    gap: 8px;
  }
}

/* Add mobile-specific hide rules */
@media (max-width: 767px) {
  .hide-mobile { display: none !important; }
  /* make action column a bit wider on mobile */
  .actions { min-width: 120px; }
}

.empty-state {
  text-align: center;
  padding: 30px 20px;
  color: #999;
}

.empty-state p {
  font-size: 1em;
  margin-bottom: 15px;
}
</style>
</head>
<body>

<div class="container">
  <h1>👻 Spooky Sports 🎃</h1>

  <!-- Settings Bar -->
  <div class="settings-bar">
    <form method="post" action="{{ url_for('save_settings') }}" style="display: flex; gap: 8px; align-items: center; width: 100%;">
      <div class="form-row" style="margin: 0; flex: 1; min-width: 100px;">
        <label>Bankroll:</label>
        <!-- use text + inputmode for reliable mobile decimal input; display formatted value with 2 decimals -->
        <input name="bankroll" type="text" inputmode="decimal" pattern="^\$?\s*\d{1,3}(?:[,\d{3}]*)(?:\.\d{1,2})?$" value="{{ '%.2f' % settings.bankroll }}">
      </div>
      <div class="form-row" style="margin: 0; flex: 1; min-width: 100px;">
        <label>Cap %:</label>
        <input name="percent_bankroll" type="number" step="0.0001" value="{{ settings.percent_bankroll }}">
      </div>
      <button type="submit" style="flex: 0 0 auto;">Save</button>
    </form>
    <span id="bankroll_value" style="display:none;">{{ settings.bankroll }}</span>
    <span id="percent_cap_value" style="display:none;">{{ settings.percent_bankroll }}</span>
  </div>

  <!-- Main Grid: Calculator (Left on desktop, stacked on mobile) + Charts (Right on desktop, below on mobile) -->
  <div class="main-grid">
    <!-- Kelly Calculator Card -->
    <div class="calculator-card">
      <h2>🔮 Kelly</h2>
      <div class="form-row">
        <label>Bet name</label>
        <input id="bet_name" name="name" type="text" required>
      </div>
      <div class="form-row">
        <label>Sport</label>
        <input id="sport" name="sport" type="text" placeholder="NBA, NFL, etc">
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
        <label>American odds</label>
        <input id="american_odds" name="american_odds" type="number" step="1" required placeholder="-120 or 150">
        <span class="small">Decimal: <strong id="converted_decimal">—</strong></span>
      </div>
      <div class="form-row">
        <label>Win probability</label>
        <div style="display:flex; gap:8px; align-items:center; width:100%;">
          <input id="prob" name="prob" type="number" step="0.0001" value="0.5" required style="flex:0 0 120px;">
          <!-- non-editable calculation summary shown here -->
          <div id="empirical_info" class="small" aria-live="polite" style="flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            Calc: — 
          </div>
        </div>
      </div>
      <div class="form-row">
        <label style="font-size: 1em;">💰 Recommended: <strong id="recommended_stake">—</strong></label>
      </div>
      <div class="form-row">
        <label>Actual stake</label>
        <input id="actual_stake" name="stake" type="number" step="0.01" required>
      </div>
      <input type="hidden" id="odds_hidden" name="odds" value="">
      <button type="button" onclick="openAddBetModal()" style="width: 100%;">Place Bet</button>
    </div>

    <!-- Dashboard Charts -->
    <div class="dashboard">
      <div class="chart-card">
        <h3>Bets (7D)</h3>
        <canvas id="betsChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Success Rate</h3>
        <canvas id="successChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Open Bets (Full Width) -->
  <div class="full-width-card">
    <h2>🔓 Open Bets</h2>
    {% if open_bets %}
    <table>
      <thead>
        <tr>
          <th class="hide-mobile">Sport</th>
          <th class="hide-mobile">Type</th>
          <th>Name</th>
          <th class="hide-mobile">Odds</th>
          <th>Stake</th>
          <th>Return</th>
          <th>⚙️</th>
        </tr>
      </thead>
      <tbody>
        {% for b in open_bets %}
        <tr>
          <td class="hide-mobile">{{ b.sport }}</td>
          <td class="hide-mobile">{{ b.bet_type }}</td>
          <td>{{ b.name }}</td>
          <td class="hide-mobile">{{ "%.2f"|format(b.odds) }}</td>
          <td>${{ "%.2f"|format(b.stake) }}</td>
          <td>${{ "%.2f"|format(b.stake * b.odds) }}</td>
          <td class="actions">
            <a href="{{ url_for('edit_open', bet_id=b.id) }}">Edit</a>
            <form method="post" action="{{ url_for('close_open', bet_id=b.id) }}" style="display: inline-flex; gap: 4px; align-items: center;">
              <select name="outcome" style="padding: 3px; font-size: 0.75em; min-width: 70px;">
                <option value="win">Win</option>
                <option value="loss">Loss</option>
              </select>
              <button type="submit" style="padding: 4px 8px; font-size: 0.75em;">OK</button>
            </form>
            <form method="post" action="{{ url_for('delete_open', bet_id=b.id) }}" onsubmit="return confirm('Delete?');" style="display:inline-block; margin-top:6px;">
              <button type="submit" style="padding: 4px 8px; font-size: 0.75em; background-color: #ff4757;">Del</button>
            </form>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <div class="empty-state">
      <p>No open bets yet.</p>
    </div>
    {% endif %}
  </div>

  <!-- Closed Bets (Full Width) -->
  <div class="full-width-card">
    <h2>🔒 Closed Bets</h2>
    <button type="button" onclick="openAddClosedModal()" style="width: 100%; margin-bottom: 10px;">+ Add Bet</button>
    {% if closed_bets %}
    <table>
      <thead>
        <tr>
          <th class="hide-mobile">Sport</th>
          <th class="hide-mobile">Type</th>
          <th>Name</th>
          <th class="hide-mobile">Odds</th>
          <th>Stake</th>
          <th>Result</th>
          <th>P&L</th>
        </tr>
      </thead>
      <tbody>
        {% for cb in closed_bets %}
        <tr>
          <td class="hide-mobile">{{ cb.sport }}</td>
          <td class="hide-mobile">{{ cb.bet_type }}</td>
          <td>{{ cb.name }}</td>
          <td class="hide-mobile">{{ "%.2f"|format(cb.odds) }}</td>
          <td>${{ "%.2f"|format(cb.stake) }}</td>
          <td class="{% if cb.outcome == 'win' %}status-win{% else %}status-loss{% endif %}">{{ cb.outcome.upper() }}</td>
          <td class="{% if cb.profit >= 0 %}status-win{% else %}status-loss{% endif %}">${{ "%.2f"|format(cb.profit) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <div class="empty-state">
      <p>No closed bets yet.</p>
    </div>
    {% endif %}
  </div>
</div>

<!-- Add Bet Modal -->
<div id="addBetModal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      <h2>Place Bet</h2>
      <button class="close-btn" onclick="closeAddBetModal()">&times;</button>
    </div>
    <form id="betForm" method="post" action="{{ url_for('add_open') }}">
      <input type="hidden" id="form_name" name="name">
      <input type="hidden" id="form_odds" name="odds">
      <input type="hidden" id="form_prob" name="prob">
      <input type="hidden" id="form_stake" name="stake">
      <input type="hidden" id="form_sport" name="sport">
      <input type="hidden" id="form_type" name="bet_type">
      <div class="form-row">
        <label>Name</label>
        <input type="text" id="modal_name" required>
      </div>
      <div class="form-row">
        <label>Sport</label>
        <input type="text" id="modal_sport">
      </div>
      <div class="form-row">
        <label>Type</label>
        <select id="modal_type">
          <option value="Moneyline">Moneyline</option>
          <option value="Spread">Spread</option>
          <option value="Over/Under">Over/Under</option>
          <option value="Player">Player</option>
        </select>
      </div>
      <div class="form-row">
        <label>Odds</label>
        <input type="number" id="modal_odds" step="1" required>
      </div>
      <div class="form-row">
        <label>Probability</label>
        <input type="number" id="modal_prob" step="0.0001" required>
      </div>
      <div class="form-row">
        <label>Stake</label>
        <input type="number" id="modal_stake" step="0.01" required>
      </div>
      <button type="submit" style="width: 100%;">Confirm</button>
    </form>
  </div>
</div>

<!-- Add Closed Bet Modal -->
<div id="addClosedModal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      <h2>Add Closed Bet</h2>
      <button class="close-btn" onclick="closeAddClosedModal()">&times;</button>
    </div>
    <form method="post" action="{{ url_for('add_closed') }}">
      <div class="form-row">
        <label>Name</label>
        <input name="name" type="text" required>
      </div>
      <div class="form-row">
        <label>Sport</label>
        <input name="sport" type="text" placeholder="NBA, NFL, etc">
      </div>
      <div class="form-row">
        <label>Type</label>
        <select name="bet_type">
          <option value="Moneyline">Moneyline</option>
          <option value="Spread">Spread</option>
          <option value="Over/Under">Over/Under</option>
          <option value="Player">Player</option>
        </select>
      </div>
      <div class="form-row">
        <label>American odds</label>
        <input name="american_odds" type="number" step="1" required placeholder="-120 or 150">
      </div>
      <div class="form-row">
        <label>Probability</label>
        <input name="prob" type="number" step="0.0001" value="0.5" required>
      </div>
      <div class="form-row">
        <label>Stake</label>
        <input name="stake" type="number" step="0.01" required>
      </div>
      <div class="form-row">
        <label>Outcome</label>
        <select name="outcome">
          <option value="win">Win</option>
          <option value="loss">Loss</option>
        </select>
      </div>
      <div class="form-row">
        <label>Closed at</label>
        <input name="closed_at" type="datetime-local">
      </div>
      <button type="submit" style="width: 100%;">Add Bet</button>
    </form>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>
function americanToDecimal(a) {
  var n = Number(a);
  if (!isFinite(n)) return null;
  if (n > 0) {
    return +(n / 100 + 1).toFixed(4);
  } else {
    return +(100 / Math.abs(n) + 1).toFixed(4);
  }
}

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
  recommended = Math.max(recommended, 0.10);
  return Math.round(recommended * 100) / 100;
}

function openAddBetModal() {
  document.getElementById('modal_name').value = document.getElementById('bet_name').value;
  document.getElementById('modal_sport').value = document.getElementById('sport').value;
  document.getElementById('modal_type').value = document.getElementById('bet_type').value;
  document.getElementById('modal_odds').value = document.getElementById('american_odds').value;
  document.getElementById('modal_prob').value = document.getElementById('prob').value;
  document.getElementById('modal_stake').value = document.getElementById('actual_stake').value;
  document.getElementById('addBetModal').classList.add('active');
}

function closeAddBetModal() {
  document.getElementById('addBetModal').classList.remove('active');
}

function openAddClosedModal() {
  document.getElementById('addClosedModal').classList.add('active');
}

function closeAddClosedModal() {
  document.getElementById('addClosedModal').classList.remove('active');
}

window.onclick = function(event) {
  var addBetModal = document.getElementById('addBetModal');
  var addClosedModal = document.getElementById('addClosedModal');
  if (event.target == addBetModal) {
    addBetModal.classList.remove('active');
  }
  if (event.target == addClosedModal) {
    addClosedModal.classList.remove('active');
  }
}

document.getElementById('betForm').addEventListener('submit', function(e){
  document.getElementById('form_name').value = document.getElementById('modal_name').value;
  document.getElementById('form_odds').value = americanToDecimal(document.getElementById('modal_odds').value);
  document.getElementById('form_prob').value = document.getElementById('modal_prob').value;
  document.getElementById('form_stake').value = document.getElementById('modal_stake').value;
  document.getElementById('form_sport').value = document.getElementById('modal_sport').value;
  document.getElementById('form_type').value = document.getElementById('modal_type').value;
});

document.addEventListener('DOMContentLoaded', function(){
  var ctx1 = document.getElementById('betsChart');
  if (ctx1) {
    new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: {{ chart_labels | tojson }},
        datasets: [{
          label: 'Bets',
          data: {{ bets_by_day | tojson }},
          backgroundColor: 'rgba(255, 107, 53, 0.6)',
          borderColor: 'rgba(255, 107, 53, 1)',
          borderWidth: 2,
          borderRadius: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: { 
          legend: { display: false },
          tooltip: { 
            backgroundColor: 'rgba(0,0,0,0.8)',
            borderColor: '#ff6b35',
            borderWidth: 1
          }
        },
        scales: { 
          y: { 
            beginAtZero: true, 
            ticks: { stepSize: 1, color: '#ffd93d', font: { size: 10 } },
            grid: { color: '#444' }
          },
          x: {
            ticks: { color: '#ffd93d', font: { size: 9 } },
            grid: { display: false }
          }
        }
      }
    });
  }

  var ctx2 = document.getElementById('successChart');
  if (ctx2) {
    new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: {{ sport_labels | tojson }},
        datasets: [{
          label: 'Win %',
          data: {{ sport_success_rates | tojson }},
          backgroundColor: 'rgba(34, 197, 94, 0.6)',
          borderColor: 'rgba(34, 197, 94, 1)',
          borderWidth: 2,
          borderRadius: 3
        }]
      },
      options: {
        indexAxis: 'x',
        responsive: true,
        maintainAspectRatio: true,
        plugins: { 
          legend: { display: false },
          tooltip: { 
            backgroundColor: 'rgba(0,0,0,0.8)',
            borderColor: '#22c55e',
            borderWidth: 1,
            callbacks: {
              label: function(context) {
                return context.parsed.y + '%';
              }
            }
          }
        },
        scales: { 
          y: { 
            beginAtZero: true,
            max: 100,
            ticks: { color: '#ffd93d', font: { size: 10 } },
            grid: { color: '#444' }
          },
          x: {
            ticks: { color: '#ffd93d', font: { size: 9 } },
            grid: { display: false }
          }
        }
      }
    });
  }

  var aInput = document.getElementById('american_odds');
  var probInput = document.getElementById('prob');
  var outSpan = document.getElementById('converted_decimal');
  var recommendedSpan = document.getElementById('recommended_stake');
  var bankroll = parseFloat(document.getElementById('bankroll_value').textContent);
  var percentCap = parseFloat(document.getElementById('percent_cap_value').textContent);

  function updateRecommended() {
    var dec = americanToDecimal(aInput.value);
    var p = parseFloat(probInput.value);
    if (dec === null || !isFinite(p)) {
      recommendedSpan.textContent = '—';
      return;
    }
    outSpan.textContent = dec;
    var rec = computeKelly(bankroll, percentCap, dec, p);
    recommendedSpan.textContent = '$' + rec.toFixed(2);
  }

  if (aInput) aInput.addEventListener('input', updateRecommended);
  if (probInput) probInput.addEventListener('input', updateRecommended);

  // Fetch and display empirical/adjusted probability info for the current sport + bet type
  function updateEmpiricalInfo() {
    var sport = (document.getElementById('sport') || {}).value || '';
    var betType = (document.getElementById('bet_type') || {}).value || '';
    var prob = (document.getElementById('prob') || {}).value || '0.5';

    fetch('/api/empirical_info', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sport: sport, bet_type: betType, prob: prob })
    })
    .then(function(res){ return res.json(); })
    .then(function(data){
      var el = document.getElementById('empirical_info');
      if (!el) return;
      if (data.empirical === null || data.matching_count === 0) {
        el.textContent = 'Empirical: —   Adjusted: ' + (data.adjusted !== undefined ? Number(data.adjusted).toFixed(4) : '—');
      } else {
        el.textContent = 'Empirical: ' + Number(data.empirical).toFixed(4) +
                         '   Adjusted: ' + Number(data.adjusted).toFixed(4) +
                         '   (α=' + Number(data.alpha).toFixed(2) + ', n=' + (data.matching_count||0) + ')';
      }
    })
    .catch(function(err){
      console.log('empirical_info error', err);
    });
  }

  // wire inputs to update the info live
  document.addEventListener('DOMContentLoaded', function(){
    var sportEl = document.getElementById('sport');
    var typeEl = document.getElementById('bet_type');
    var probEl = document.getElementById('prob');
    if (sportEl) sportEl.addEventListener('input', updateEmpiricalInfo);
    if (typeEl) typeEl.addEventListener('change', updateEmpiricalInfo);
    if (probEl) probEl.addEventListener('input', updateEmpiricalInfo);
    // initial populate
    updateEmpiricalInfo();
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
  if not settings:
    settings = Setting(bankroll=1000.0, percent_bankroll=0.02)
    db.session.add(settings)
    db.session.commit()
  
  open_bets = OpenBet.query.order_by(OpenBet.created_at.desc()).all()
  closed_bets = ClosedBet.query.order_by(ClosedBet.closed_at.desc()).all()
  
  # Generate chart data - use closed_bets for historical data
  now = datetime.utcnow()
  chart_labels = []
  bets_by_day = []
  for i in range(6, -1, -1):
    day = now - timedelta(days=i)
    day_str = day.strftime('%a')
    chart_labels.append(day_str)
    # Count closed bets from this day (combining open+closed for complete picture)
    start_of_day = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    
    closed_count = ClosedBet.query.filter(
      ClosedBet.closed_at >= start_of_day,
      ClosedBet.closed_at < end_of_day
    ).count()
    open_count = OpenBet.query.filter(
      OpenBet.created_at >= start_of_day,
      OpenBet.created_at < end_of_day
    ).count()
    bets_by_day.append(closed_count + open_count)
  
  # Success rate by sport - convert to percentages for bar chart
  sports = {}
  for cb in closed_bets:
    sport = cb.sport or 'Unknown'
    if sport not in sports:
      sports[sport] = {'wins': 0, 'total': 0}
    sports[sport]['total'] += 1
    if cb.outcome == 'win':
      sports[sport]['wins'] += 1
  
  sport_labels = list(sports.keys())
  sport_success_rates = [int((sports[s]['wins'] / sports[s]['total'] * 100)) if sports[s]['total'] > 0 else 0 for s in sport_labels]
  
  return render_template_string(BASE, 
    settings=settings, 
    open_bets=open_bets, 
    closed_bets=closed_bets,
    chart_labels=chart_labels,
    bets_by_day=bets_by_day,
    sport_labels=sport_labels,
    sport_success_rates=sport_success_rates
  )

@app.route('/save_settings', methods=['POST'])
def save_settings():
  s = Setting.query.first()
  if not s:
    s = Setting()
    db.session.add(s)
  try:
    # Accept user-friendly currency input (e.g. "1,000.00" or "$1000.00") and normalize it.
    def parse_money(val, fallback):
      if val is None:
        return fallback
      try:
        v = str(val).strip()
        # remove common non-numeric characters
        v = v.replace('$', '').replace(',', '').replace(' ', '')
        return round(float(v), 2)
      except Exception:
        return fallback

    s.bankroll = parse_money(request.form.get('bankroll'), s.bankroll)
    # percent_bankroll should be a fraction (e.g. 0.02). allow either "2" (percent) or "0.02"
    try:
      raw_pct = request.form.get('percent_bankroll', None)
      if raw_pct is not None:
        p = float(str(raw_pct).replace('%', '').replace(' ', ''))
        if p > 1:
          p = p / 100.0
        s.percent_bankroll = float(p)
    except Exception:
      pass
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

  # create open bet and subtract stake from bankroll
  b = OpenBet(name=name, odds=odds, prob=prob, stake=stake, sport=sport, bet_type=bet_type)
  db.session.add(b)

  # Update bankroll: subtract stake immediately so user doesn't have to update manually
  s = Setting.query.first()
  if s:
    try:
      s.bankroll = float(s.bankroll) - float(stake)
    except Exception:
      pass

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
      <option value="Player" {% if b.bet_type == 'Player' %}selected{% endif %}>Player</option>
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
  # refund stake to bankroll if present (cancelled bet)
  s = Setting.query.first()
  if s:
    try:
      s.bankroll = float(s.bankroll) + float(b.stake)
    except Exception:
      pass
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
  # If the stake was already deducted at placement, on win we must return stake + profit.
  if outcome == 'win':
    profit = b.stake * (b.odds - 1.0)
  else:
    profit = -b.stake

  # create ClosedBet record
  cb = ClosedBet(name=b.name, odds=b.odds, prob=b.prob, stake=b.stake,
                 sport=b.sport, bet_type=b.bet_type,
                 outcome=outcome, profit=profit, closed_at=datetime.utcnow())
  db.session.add(cb)

  # Update bankroll:
  s = Setting.query.first()
  if s:
    try:
      if outcome == 'win':
        # stake was subtracted at placement; add back total return (stake + profit)
        s.bankroll = float(s.bankroll) + float(b.stake * b.odds)
      else:
        # loss: stake already removed at placement, no change
        s.bankroll = float(s.bankroll)
    except Exception:
      pass

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

@app.route('/api/empirical_info', methods=['POST'])
@app.route('/api/empirical_info', methods=['POST'])
def empirical_info():
  """
  Return empirical win-rate and blended (adjusted) probability for a given sport + bet_type.
  Accepts JSON with keys: sport, bet_type, prob (user estimate).
  Response:
    { empirical: 0.6 | null, adjusted: 0.61, alpha: 0.6, tau_days: 30.0, matching_count: N, weights_sum: W }
  """
  data = request.get_json(silent=True) or {}
  sport = (data.get('sport') or '').strip()
  bet_type = (data.get('bet_type') or '').strip()
  try:
    user_prob = float(data.get('prob', 0.5))
  except Exception:
    user_prob = 0.5

  ALPHA = 0.6
  TAU_DAYS = 30.0
  MIN_PROB = 0.5
  MAX_PROB = 0.95

  # If sport or bet_type missing, return adjusted = user_prob (bounded) but no empirical
  if not sport or not bet_type:
    adjusted = max(MIN_PROB, min(user_prob, MAX_PROB))
    return {
      'empirical': None,
      'adjusted': adjusted,
      'alpha': ALPHA,
      'tau_days': TAU_DAYS,
      'matching_count': 0,
      'weights_sum': 0.0
    }

  matching = ClosedBet.query.filter(ClosedBet.sport == sport, ClosedBet.bet_type == bet_type).all()
  if not matching:
    adjusted = max(MIN_PROB, min(user_prob, MAX_PROB))
    return {
      'empirical': None,
      'adjusted': adjusted,
      'alpha': ALPHA,
      'tau_days': TAU_DAYS,
      'matching_count': 0,
      'weights_sum': 0.0
    }

  now = datetime.utcnow()
  weights_sum = 0.0
  weighted_wins = 0.0
  for cb in matching:
    try:
      age_days = max(0.0, (now - (cb.closed_at or now)).total_seconds() / 86400.0)
    except Exception:
      age_days = 0.0
    weight = math.exp(- age_days / TAU_DAYS)
    weights_sum += weight
    if getattr(cb, 'outcome', '') == 'win':
      weighted_wins += weight

  empirical = (weighted_wins / weights_sum) if weights_sum > 0 else None

  if empirical is not None:
    adjusted = ALPHA * user_prob + (1.0 - ALPHA) * empirical
  else:
    adjusted = user_prob

  adjusted = max(MIN_PROB, min(adjusted, MAX_PROB))

  return {
    'empirical': empirical,
    'adjusted': adjusted,
    'alpha': ALPHA,
    'tau_days': TAU_DAYS,
    'matching_count': len(matching),
    'weights_sum': weights_sum
  }

if __name__ == '__main__':
  app.run(debug=True)


