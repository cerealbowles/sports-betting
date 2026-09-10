import warnings
warnings.filterwarnings('ignore', message='.*timezone.*')

from flask import Flask, request, redirect, url_for, render_template, abort, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import csv
import io
import json
import os
import math
import re
import threading
import atexit
import mlb_api
import mlb_model
import nhl_api
import nfl_api
import cfb_api

# app.py
# Simple Flask app for sports betting with Kelly criterion, open/closed bets and space to tweak formula using historical bets.

app = Flask(__name__)

# DB_PATH can be overridden via env var — used to mount a persistent Docker volume.
# Defaults to ./instance/bets.db for local dev (matches Flask's default instance folder).
_DB_PATH = os.environ.get('DB_PATH', os.path.join(app.instance_path, 'bets.db'))
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{_DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class Setting(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  bankroll = db.Column(db.Float, default=20)
  percent_bankroll = db.Column(db.Float, default=0.25)  # fraction of bankroll
  discord_webhook_url = db.Column(db.Text, default='')

class OpenBet(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(200))
  odds = db.Column(db.Float)  # decimal odds
  prob = db.Column(db.Float)  # user's estimated win probability (0-1)
  stake = db.Column(db.Float)
  sport = db.Column(db.String(100), default='')
  bet_type = db.Column(db.String(50), default='Moneyline')
  created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
  eventstart = db.Column(db.DateTime, default=None)
  notes = db.Column(db.Text, default='')
  game_key = db.Column(db.String(300), default='')  # odds_history lookup key
  bet_side = db.Column(db.String(10), default='')   # 'home' or 'away'
  is_paper = db.Column(db.Boolean, default=False)   # paper/sim bet — no bankroll impact

class ClosedBet(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(200))
  odds = db.Column(db.Float)
  prob = db.Column(db.Float)
  stake = db.Column(db.Float)
  sport = db.Column(db.String(100), default='')
  bet_type = db.Column(db.String(50), default='Moneyline')
  outcome = db.Column(db.String(20))  # 'win' or 'loss'
  profit = db.Column(db.Float)
  closed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
  eventstart = db.Column(db.DateTime, default=None)
  notes = db.Column(db.Text, default='')
  closing_line   = db.Column(db.Float, default=None)  # decimal odds at settlement
  is_paper       = db.Column(db.Boolean, default=False)
  cashout_amount = db.Column(db.Float, default=None)   # for outcome='cashout': amount received

  @property
  def clv(self):
    """Closing line value % — positive means you got better odds than closing."""
    if self.closing_line and self.closing_line > 0 and self.odds:
      return round((self.odds / self.closing_line - 1) * 100, 1)
    return None

class GamePrediction(db.Model):
  __tablename__ = 'game_predictions'
  id             = db.Column(db.Integer, primary_key=True)
  sport          = db.Column(db.String(10), nullable=False, default='MLB')
  game_date      = db.Column(db.String(10), nullable=False)   # YYYY-MM-DD ET
  game_time_utc  = db.Column(db.String(30), default='')
  home_team      = db.Column(db.String(80), nullable=False)
  away_team      = db.Column(db.String(80), nullable=False)
  home_prob      = db.Column(db.Float)                        # model win prob for home
  away_prob      = db.Column(db.Float)
  home_odds      = db.Column(db.Integer)                      # American odds at prediction time
  away_odds      = db.Column(db.Integer)
  factors_json   = db.Column(db.Text, default='[]')           # [(label, contrib), ...]
  home_score     = db.Column(db.Integer)                      # null until Final
  away_score     = db.Column(db.Integer)
  home_won          = db.Column(db.Boolean)                   # null until Final
  outcome_set_at    = db.Column(db.DateTime(timezone=True))
  edge_grade        = db.Column(db.String(5))                 # A/B/C/– composite grade
  edge_pct          = db.Column(db.Float)                     # model edge % vs vig-free market
  composite_score   = db.Column(db.Float)                     # raw composite score (0-1)
  opening_home_odds = db.Column(db.Integer)                   # first observed odds for this game (never overwritten)
  opening_away_odds = db.Column(db.Integer)
  closing_home_odds = db.Column(db.Integer)                   # odds captured at game-time (CLV)
  closing_away_odds = db.Column(db.Integer)
  pick_clv          = db.Column(db.Float)                     # CLV %: closing_implied - opening_implied for model pick
  pick_roi          = db.Column(db.Float)                     # ROI per unit for model pick (profit or -1)
  daily_rank        = db.Column(db.Integer)                   # recommendation rank on game day (1 = top pick)
  movement_profile  = db.Column(db.String(20))                 # early_sharp/late_sharp/sustained/reversal/flat
  movement_pct      = db.Column(db.Float)                      # total pre-game move on pick side, percentage points
  wind_mph       = db.Column(db.Float)                        # at game time, null for domes
  wind_dir       = db.Column(db.String(4))
  created_at     = db.Column(db.DateTime(timezone=True),
                             default=lambda: datetime.now(timezone.utc))

# Create tables for any new models on first run
with app.app_context():
  db.create_all()

def ensure_column_exists():
  with app.app_context():
    inspector = inspect(db.engine)

    def _add(table, col, col_type):
      existing = [c['name'] for c in inspector.get_columns(table)]
      if col not in existing:
        with db.engine.begin() as conn:
          conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col} {col_type}'))

    _add('open_bet',    'eventstart',   'DATETIME DEFAULT NULL')
    _add('closed_bet',  'eventstart',   'DATETIME DEFAULT NULL')
    _add('open_bet',    'notes',        'TEXT DEFAULT NULL')
    _add('closed_bet',  'notes',        'TEXT DEFAULT NULL')
    _add('closed_bet',  'closing_line', 'FLOAT DEFAULT NULL')
    _add('open_bet',    'game_key',     'TEXT DEFAULT NULL')
    _add('open_bet',    'bet_side',     'TEXT DEFAULT NULL')
    _add('open_bet',         'is_paper', 'BOOLEAN DEFAULT 0')
    _add('closed_bet',       'is_paper', 'BOOLEAN DEFAULT 0')
    _add('game_predictions', 'wind_mph',          'FLOAT DEFAULT NULL')
    _add('game_predictions', 'wind_dir',          'TEXT DEFAULT NULL')
    _add('game_predictions', 'edge_grade',        'TEXT DEFAULT NULL')
    _add('game_predictions', 'edge_pct',          'FLOAT DEFAULT NULL')
    _add('game_predictions', 'composite_score',   'FLOAT DEFAULT NULL')
    _add('game_predictions', 'opening_home_odds', 'INTEGER DEFAULT NULL')
    _add('game_predictions', 'opening_away_odds', 'INTEGER DEFAULT NULL')
    _add('game_predictions', 'closing_home_odds', 'INTEGER DEFAULT NULL')
    _add('game_predictions', 'closing_away_odds', 'INTEGER DEFAULT NULL')
    _add('game_predictions', 'pick_clv',          'FLOAT DEFAULT NULL')
    _add('game_predictions', 'pick_roi',          'FLOAT DEFAULT NULL')
    _add('game_predictions', 'daily_rank',        'INTEGER DEFAULT NULL')
    _add('game_predictions', 'movement_profile',  'TEXT DEFAULT NULL')
    _add('game_predictions', 'movement_pct',      'FLOAT DEFAULT NULL')
    _add('setting',          'discord_webhook_url', 'TEXT DEFAULT NULL')
    _add('closed_bet',       'cashout_amount',     'FLOAT DEFAULT NULL')

# Migrations must run before any Setting.query access
ensure_column_exists()

with app.app_context():
  if not Setting.query.first():
    db.session.add(Setting(bankroll=50, percent_bankroll=0.25))
    db.session.commit()


def _vig_free_implied(home_odds, away_odds):
    """Vig-free implied probabilities from American odds. Returns (home, away) or (None, None)."""
    try:
        def _raw(o):
            o = int(o)
            return 100 / (100 + o) if o > 0 else (-o) / (-o + 100)
        h, a = _raw(home_odds), _raw(away_odds)
        t = h + a
        return h / t, a / t
    except Exception:
        return None, None


def _pick_edge_pct(home_prob, home_odds, away_odds):
    """Model pick edge vs vig-free market implied. Returns edge_pct or None."""
    if home_prob is None or not home_odds or not away_odds:
        return None
    fav_home  = home_prob >= 0.5
    pick_prob = home_prob if fav_home else 1.0 - home_prob
    vf_h, vf_a = _vig_free_implied(home_odds, away_odds)
    if vf_h is None:
        return None
    return round((pick_prob - (vf_h if fav_home else vf_a)) * 100, 1)


def _line_move_pct(home_prob, current_home_odds, current_away_odds,
                   opening_home_odds, opening_away_odds):
    """Opening → current implied prob shift on the model pick side.
    Positive = market moved toward our pick (sharp agreement). Returns None if missing data."""
    if not all([home_prob, current_home_odds, current_away_odds,
                opening_home_odds, opening_away_odds]):
        return None
    fav_home = home_prob >= 0.5
    vf_curr_h, vf_curr_a = _vig_free_implied(current_home_odds, current_away_odds)
    vf_open_h, vf_open_a = _vig_free_implied(opening_home_odds, opening_away_odds)
    if not vf_curr_h or not vf_open_h:
        return None
    curr = vf_curr_h if fav_home else vf_curr_a
    opn  = vf_open_h if fav_home else vf_open_a
    return round((curr - opn) * 100, 2)


def _consensus_bucket(factors_json, fav_home):
    """Fraction of significant-factor weight aligned with pick → bucket label."""
    try:
        sig = [(l, c) for l, c in json.loads(factors_json or '[]') if abs(c) > 0.03]
        if sig:
            tw = sum(abs(c) for _, c in sig)
            aw = sum(abs(c) for _, c in sig if (c > 0) == fav_home)
            r  = aw / tw
            if   r < 0.20: return '0–20%'
            elif r < 0.40: return '20–40%'
            elif r < 0.60: return '40–60%'
            elif r < 0.80: return '60–80%'
            else:          return '80–100%'
    except Exception:
        pass
    return '40–60%'


# ── Grade-conditional probability calibration ────────────────────────────────
# Second-stage calibration layer applied AFTER the composite grade is known.
# The global Platt scaler in mlb_model.py corrects overall probability scale.
# This layer corrects for the residual per-grade bias: e.g. if grade-A picks
# actually win at 71% when the model says 58%, we lift those probabilities
# before they reach Kelly so bet sizing reflects the empirical win rate.
#
# Stored as logit-space additive corrections so the adjustment is symmetric
# and doesn't push probabilities past 0/1.
# Applied with 60% shrinkage × sample-size credibility (full at n ≥ 50).

_CONS_CALIB: dict = {}   # consensus bucket -> logit-space correction (float)
_CONS_CALIB_LOCK = threading.Lock()


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def _safe_logit(p):
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def _consensus_adjusted_prob(pick_prob, factors_json, fav_home):
    """Apply consensus-bucket calibration correction to model pick probability."""
    ck = _consensus_bucket(factors_json, fav_home)
    with _CONS_CALIB_LOCK:
        correction = _CONS_CALIB.get(ck, 0.0)
    if correction == 0.0:
        return pick_prob
    return _sigmoid(_safe_logit(pick_prob) + correction)


def _recompute_consensus_calibration(sport='MLB'):
    """
    Fit per-consensus-bucket logit-space probability corrections from resolved predictions.

    Measures the gap between model's average pick probability and actual win rate
    within each factor-consensus bucket, then stores a shrunk logit correction.
    Called at startup and after each nightly outcome resolution.
    """
    global _CONS_CALIB

    with app.app_context():
        resolved = GamePrediction.query.filter(
            GamePrediction.sport == sport,
            GamePrediction.home_won.isnot(None),
            GamePrediction.home_odds.isnot(None),
            GamePrediction.away_odds.isnot(None),
        ).all()

    cons_data: dict = {k: [] for k in ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%']}
    for p in resolved:
        fav_home  = (p.home_prob or 0.5) >= 0.5
        pick_prob = p.home_prob if fav_home else 1.0 - p.home_prob
        model_won = fav_home == bool(p.home_won)
        ck = _consensus_bucket(p.factors_json, fav_home)
        cons_data[ck].append((pick_prob, model_won))

    SHRINKAGE = 0.60
    MIN_N     = 15
    new_calib = {}

    for ck, records in cons_data.items():
        n = len(records)
        if n < MIN_N:
            continue
        avg_model = sum(p for p, _ in records) / n
        actual_wr = sum(1 for _, w in records if w) / n
        raw_corr  = _safe_logit(actual_wr) - _safe_logit(avg_model)
        alpha     = min(n / 50.0, 1.0)
        correction = raw_corr * SHRINKAGE * alpha
        new_calib[ck] = round(correction, 5)
        print(
            f'[cons-calib] {sport} cons {ck}: model={avg_model:.1%} '
            f'actual={actual_wr:.1%} raw_corr={raw_corr:+.3f} '
            f'applied={correction:+.3f} (n={n})',
            flush=True,
        )

    with _CONS_CALIB_LOCK:
        _CONS_CALIB.update(new_calib)


# ── Trust Score weight cache ──────────────────────────────────────────────────
# Prior weights (fixed starting point before history accumulates).
# Blend toward empirical ROI-derived weights as resolved game count grows.
# Weights: 50% consensus · 25% edge zone · 15% fav/dog · 10% calibration
_PRIOR_CONS   = {'0–20%': 0.20, '20–40%': 0.35, '40–60%': 0.50,
                 '60–80%': 0.65, '80–100%': 1.00}
_PRIOR_EDGE   = {'< -6%': 0.70, '-6 to -2%': 0.05, '-2 to +5%': 0.75,
                 '+5 to +10%': 1.00, '> +10%': 0.40}
# Fav/Dog: 239-game data shows Fav +3.8% ROI / 60.7% WR vs Dog −9.0% / 43.8%.
# Our dog picks tend to be contrarian — market has it right more often.
_PRIOR_FAVDOG = {'Fav': 0.65, 'Dog': 0.25}
# Calibration bucket priors: neutral 0.5 except known danger zones.
# 56-58% is a consistent danger zone (−17.9% ROI, n=27); 65-70% overconfidence trap.
_PRIOR_CALIB  = {
    '50–52%': 0.40, '52–54%': 0.55, '54–56%': 0.60,
    '56–58%': 0.20, '58–60%': 0.70, '60–65%': 0.60,
    '65–70%': 0.30, '70–75%': 0.50,
}

_TRUST_WEIGHTS = {
    'cons':   dict(_PRIOR_CONS),
    'edge':   dict(_PRIOR_EDGE),
    'favdog': dict(_PRIOR_FAVDOG),
    'calib':  dict(_PRIOR_CALIB),
    'n': 0,
    'baseline_roi': 0.0,
    'computed_at': None,
}
_TRUST_LOCK = threading.Lock()


# Where the live-recomputed team-bias dict is persisted so offline scripts
# (backfill_2026.py) can pick up the latest snapshot without triggering a
# live recompute themselves. Written every time _recompute_team_bias() runs
# in this (the real app) process — i.e. at most once per nightly resolve /
# daily pick generation, not once per backfill invocation.
_TEAM_BIAS_SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'team_bias_snapshot.json')


def _save_team_bias_snapshot(sport, bias):
    try:
        payload = {}
        if os.path.exists(_TEAM_BIAS_SNAPSHOT_PATH):
            try:
                with open(_TEAM_BIAS_SNAPSHOT_PATH) as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
        payload[sport] = {
            'bias': bias,
            'computed_at': datetime.now(timezone.utc).isoformat(),
        }
        with open(_TEAM_BIAS_SNAPSHOT_PATH, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f'[team-bias] failed to write snapshot: {e}', flush=True)


def _recompute_team_bias(sport='MLB'):
    """Auto-recalibrate per-team historical bias offsets from all resolved games.

    Replaces the hardcoded _TEAM_BIAS dict in mlb_model with dynamically
    derived values — same 30% shrinkage, same 'positive = model underestimates
    this team' convention — but computed from current resolved data rather than
    a manually-locked snapshot.

    Also persists the result to _TEAM_BIAS_SNAPSHOT_PATH so that offline
    scripts (backfill_2026.py) can load a recent, stable snapshot instead of
    recomputing live from whatever's in the DB at the moment they happen to
    run — recomputing live on every backfill run made the bias term (and
    therefore every historical factors_json) a moving target that changed
    based on when you reran the backfill, not on anything that happened by
    each game's date. This function itself still recomputes live — that's
    correct, since it's only called from the real app on a fixed cadence
    (startup, nightly outcome resolution, daily pick generation) — it's only
    backfill's ad hoc reruns that needed to stop triggering it.

    Context: _TEAM_BIAS was last manually recalibrated 2026-06-08. After that
    date, games are genuinely out-of-sample relative to those bias values. An
    out-of-sample test showed +0.1 correlation on in-sample data dropping to
    ~0.003 on the clean held-out post-recalibration window — suggesting the
    manually-derived values carry in-sample inflation. Auto-recalibrating nightly
    on the full running dataset keeps the corrections fresh and reduces the gap
    between fitting window and current games.

    Credibility blend: alpha = min(n_team / 150, 1.0) — more conservative than
    trust-weight buckets (n/50) since team-level tendencies should be stable
    across multiple seasons, not reactive to a 10-game hot/cold streak.
    """
    import mlb_model as _mlb_m

    with app.app_context():
        resolved = GamePrediction.query.filter(
            GamePrediction.sport == sport,
            GamePrediction.home_won.isnot(None),
            GamePrediction.home_prob.isnot(None),
        ).all()

    team_records = defaultdict(list)
    for p in resolved:
        pred_home = p.home_prob or 0.5
        won_home  = bool(p.home_won)
        team_records[p.home_team].append((pred_home,      won_home))
        team_records[p.away_team].append((1.0 - pred_home, not won_home))

    MIN_N    = 30
    SHRINK   = 0.30
    new_bias = {}
    for team, recs in team_records.items():
        n = len(recs)
        if n < MIN_N:
            continue
        avg_pred  = sum(p for p, _ in recs) / n
        actual_wr = sum(1 for _, w in recs if w) / n
        raw_bias  = actual_wr - avg_pred
        alpha     = min(n / 150.0, 1.0)
        shrunk    = round(raw_bias * SHRINK * alpha, 4)
        if abs(shrunk) >= 0.005:
            new_bias[team] = shrunk

    _mlb_m._TEAM_BIAS = new_bias
    _save_team_bias_snapshot(sport, new_bias)

    n_teams = len(new_bias)
    total   = len(resolved)
    print(f'[team-bias] recomputed from {total} {sport} games: '
          f'{n_teams} teams with |bias|≥0.005 (n≥{MIN_N} per team)', flush=True)


def _recompute_trust_weights(sport='MLB'):
    """
    Derive per-bucket ROI weights from resolved game_predictions and update
    _TRUST_WEIGHTS in place. Called at startup and after each nightly resolve.

    Credibility blend: alpha = min(n_bucket / 50, 1.0).
      - n < 5  → 100% prior
      - n = 25 → 50% empirical
      - n >= 50 → 100% empirical
    Normalization: ±30% ROI above/below baseline maps to the full 0–1 scale.
    """
    global _TRUST_WEIGHTS

    with app.app_context():
        resolved = GamePrediction.query.filter(
            GamePrediction.sport == sport,
            GamePrediction.home_won.isnot(None),
            GamePrediction.home_odds.isnot(None),
            GamePrediction.away_odds.isnot(None),
        ).all()

    n_total = len(resolved)
    if n_total < 10:
        return  # too few games for any empirical signal

    # Baseline ROI across all picks (denominator for lift)
    base_vals = []
    for p in resolved:
        fav_home  = (p.home_prob or 0.5) >= 0.5
        model_won = fav_home == bool(p.home_won)
        pick_odds = p.home_odds if fav_home else p.away_odds
        if not pick_odds:
            continue
        try:
            o = int(pick_odds)
            profit = o / 100.0 if o > 0 else 100.0 / (-o)
            base_vals.append(profit if model_won else -1.0)
        except (TypeError, ValueError):
            pass
    if not base_vals:
        return
    baseline_roi = sum(base_vals) / len(base_vals) * 100  # %

    cons_bkts   = {k: [] for k in _PRIOR_CONS}
    edge_bkts   = {k: [] for k in _PRIOR_EDGE}
    favdog_bkts = {k: [] for k in _PRIOR_FAVDOG}
    calib_bkts  = {k: [] for k in _PRIOR_CALIB}

    for p in resolved:
        fav_home  = (p.home_prob or 0.5) >= 0.5
        model_won = fav_home == bool(p.home_won)
        pick_odds = p.home_odds if fav_home else p.away_odds
        if not pick_odds:
            continue
        try:
            o = int(pick_odds)
            profit = o / 100.0 if o > 0 else 100.0 / (-o)
            roi_pct = (profit if model_won else -1.0) * 100
        except (TypeError, ValueError):
            continue

        # Consensus bucket
        ck = _consensus_bucket(p.factors_json, fav_home)
        cons_bkts[ck].append(roi_pct)

        # Edge zone bucket
        edge = _pick_edge_pct(p.home_prob, p.home_odds, p.away_odds)
        if edge is not None:
            e = edge
            if e < -6:    ek = '< -6%'
            elif e < -2:  ek = '-6 to -2%'
            elif e <= 5:  ek = '-2 to +5%'
            elif e <= 10: ek = '+5 to +10%'
            else:         ek = '> +10%'
            edge_bkts[ek].append(roi_pct)

        # Fav/Dog bucket (is our pick the market favorite or underdog?)
        vf_h, vf_a = _vig_free_implied(p.home_odds, p.away_odds)
        if vf_h is not None:
            mkt_prob = vf_h if fav_home else vf_a
            favdog_bkts['Fav' if mkt_prob >= 0.5 else 'Dog'].append(roi_pct)

        # Calibration bucket (pick_prob range → historical ROI)
        pick_prob_pct = (p.home_prob if fav_home else 1.0 - p.home_prob) * 100
        if pick_prob_pct is not None:
            if pick_prob_pct < 52:   cbk = '50–52%'
            elif pick_prob_pct < 54: cbk = '52–54%'
            elif pick_prob_pct < 56: cbk = '54–56%'
            elif pick_prob_pct < 58: cbk = '56–58%'
            elif pick_prob_pct < 60: cbk = '58–60%'
            elif pick_prob_pct < 65: cbk = '60–65%'
            elif pick_prob_pct < 70: cbk = '65–70%'
            else:                    cbk = '70–75%'
            calib_bkts[cbk].append(roi_pct)

    NORM = 60.0   # ±30% relative ROI → full 0–1 range (30% × 2 = 60)
    MIN_N = 5

    def _blend(vals, prior):
        n = len(vals)
        if n < MIN_N:
            return prior
        avg = sum(vals) / n
        empirical = max(0.05, min(1.0, 0.5 + (avg - baseline_roi) / NORM))
        alpha = min(n / 50.0, 1.0)
        return round(alpha * empirical + (1 - alpha) * prior, 4)

    new_weights = {
        'cons':   {k: _blend(cons_bkts[k],   _PRIOR_CONS[k])   for k in cons_bkts},
        'edge':   {k: _blend(edge_bkts[k],   _PRIOR_EDGE[k])   for k in edge_bkts},
        'favdog': {k: _blend(favdog_bkts[k], _PRIOR_FAVDOG[k]) for k in favdog_bkts},
        'calib':  {k: _blend(calib_bkts[k],  _PRIOR_CALIB[k])  for k in calib_bkts},
        'n': n_total,
        'baseline_roi': round(baseline_roi, 2),
        'computed_at': datetime.now(timezone.utc).isoformat(),
    }

    with _TRUST_LOCK:
        _TRUST_WEIGHTS.update(new_weights)

    print(f'[trust-weights] recomputed from {n_total} {sport} games '
          f'(baseline ROI {baseline_roi:+.1f}%). '
          f'Cons: {new_weights["cons"]}', flush=True)


def _trust_score(home_prob, home_odds, away_odds, factors_json, detail=False):
    """
    0-100 composite trust score for ranking recommendations.

    Components (50/25/15/10 split; per-bucket weights adapt from history):
      50% factor consensus   — % of significant factors aligned with pick
      25% edge zone          — market agreement bucket, calibrated from outcomes
      15% fav/dog            — is pick on market fav or dog? (dog picks −9% ROI)
      10% calibration bucket — pick_prob range → historical ROI (56-58% danger zone)

    Weights start as hand-tuned priors and blend toward empirical ROI-derived
    values as resolved game count grows (credibility: full empirical at n ≥ 50).

    When detail=True, returns a dict with per-component values for radar chart display.
    Always pass home_prob (not pick_prob) — fav_home is derived internally.
    """
    edge = _pick_edge_pct(home_prob, home_odds, away_odds)
    if edge is None:
        return ({} if detail else 0)

    with _TRUST_LOCK:
        w = dict(_TRUST_WEIGHTS)

    fav_home  = (home_prob or 0.5) >= 0.5
    pick_prob = (home_prob or 0.5) if fav_home else 1.0 - (home_prob or 0.5)

    # Factor consensus
    cons_val = 0.5
    ck = '40–60%'
    try:
        factors = json.loads(factors_json or '[]')
        sig = [(l, c) for l, c in (factors or []) if abs(c) > 0.03]
        if sig:
            total_w  = sum(abs(c) for _, c in sig)
            align_w  = sum(abs(c) for _, c in sig if (c > 0) == fav_home)
            ratio    = align_w / total_w if total_w > 0 else 0.5
            if ratio < 0.20:   ck = '0–20%'
            elif ratio < 0.40: ck = '20–40%'
            elif ratio < 0.60: ck = '40–60%'
            elif ratio < 0.80: ck = '60–80%'
            else:              ck = '80–100%'
            # Blend calibrated bucket weight (70%) with raw ratio (30%)
            cons_val = 0.70 * w['cons'].get(ck, 0.5) + 0.30 * ratio
    except Exception:
        pass

    # Edge zone
    e = edge or 0
    if e < -6:    ek = '< -6%'
    elif e < -2:  ek = '-6 to -2%'
    elif e <= 5:  ek = '-2 to +5%'
    elif e <= 10: ek = '+5 to +10%'
    else:         ek = '> +10%'
    edge_val = w['edge'].get(ek, 0.5)

    # Fav/Dog: is our pick the market favorite or underdog?
    favdog_val = 0.5
    fl = '—'
    vf_h, vf_a = _vig_free_implied(home_odds, away_odds)
    if vf_h is not None:
        mkt_prob = vf_h if fav_home else vf_a
        fl = 'Fav' if mkt_prob >= 0.5 else 'Dog'
        favdog_val = w['favdog'].get(fl, 0.5)

    # Calibration bucket: pick_prob range → historical ROI signal
    pp = pick_prob * 100
    if pp < 52:   cbk = '50–52%'
    elif pp < 54: cbk = '52–54%'
    elif pp < 56: cbk = '54–56%'
    elif pp < 58: cbk = '56–58%'
    elif pp < 60: cbk = '58–60%'
    elif pp < 65: cbk = '60–65%'
    elif pp < 70: cbk = '65–70%'
    else:         cbk = '70–75%'
    calib_val = w['calib'].get(cbk, 0.5)

    raw = (0.50 * cons_val + 0.40 * edge_val + 0.00 * favdog_val + 0.10 * calib_val)
    score = round(raw * 100)

    if detail:
        return {
            'score': score,
            'c':  round(cons_val,   3), 'cl': ck,
            'e':  round(edge_val,   3), 'el': ek,
            'f':  round(favdog_val, 3), 'fl': fl,
            'k':  round(calib_val,  3), 'kl': cbk,
        }
    return score


def _experimental_score(td, clv_pct=None, line_move_pct=None):
    """Experimental ranking formula: 0.4*edge + 0.3*market_signal + 0.2*consensus + 0.1*calib.

    market_signal uses the best available market movement source:
      - line_move_pct: opening → current/close (preferred; works pre-game and post-game)
      - clv_pct: entry → close fallback for resolved games without opening odds stored
      Both normalised to 0-1: 0.5 + pct/10, capped at [0, 1]. Returns 0 if neither available.
    """
    if not td:
        return 0.0
    signal_pct = line_move_pct if line_move_pct is not None else clv_pct
    if signal_pct is None:
        return 0.0
    signal_val = max(0.0, min(1.0, 0.5 + signal_pct / 10.0))
    return (0.40 * (td.get('e') or 0) +
            0.30 * signal_val +
            0.20 * (td.get('c') or 0) +
            0.10 * (td.get('k') or 0))


def _sharp_score(td, line_move_val=0.5):
    """Pre-game sharp score: 30% edge + 30% consensus + 20% calibration + 20% line movement.

    line_move_val: 0-1 normalized; 0.5 = neutral (no opening line).
      Derived from (current_pick_implied − opening_pick_implied) / 10%, capped [0, 1].
      Positive = market moved toward our pick (sharp agreement signal).
    """
    if not td:
        return 0
    raw = (0.30 * (td.get('e') or 0) +
           0.30 * (td.get('c') or 0) +
           0.20 * (td.get('k') or 0) +
           0.20 * line_move_val)
    return round(raw * 100)


# ── Unified score: single data-fit weight vector over the same components ────
# Trust/Sharp/Experimental are all linear blends of the same {c, e, k, market}
# components with hand-set splits. Unified replaces the splits with weights
# grid-searched against resolved games, blended toward the prior as empirical
# n grows — same credibility-blend pattern as _recompute_trust_weights.
#
# Two parallel tracks, same component family, different objectives AND
# (as of 2026-07-07) different component sets:
#   _UNIFIED_WEIGHTS      — grid-searches (e, f, m): edge, fav/dog, line move.
#                            Optimized for WIN RATE of the top-3 picks/day.
#                            Powers the main recommendation table (all ranks).
#                            A broad, "good across several bets" ranking.
#   _PICK_OF_DAY_WEIGHTS  — grid-searches (e, k, m): edge, calibration, line move.
#                            Optimized for ROI of the #1 pick/day only.
#                            Powers the single featured Pick of the Day slot.
#                            Tuned to find the one standout, not a portfolio.
# Calibration ('k') was replaced with fav/dog ('f') in the unified track after
# validation on 493 resolved MLB games: chronological train/test splits at
# 50/60/70/80% train each showed e/f/m beating e/k/m out-of-sample on every
# split (avg win rate 60.3% vs 57.3%, avg ROI +12.4% vs +5.0%), and a 4-variable
# search adding 'k' back alongside 'f' underperformed the 3-variable e/f/m
# blend (57.7%/+6.8%) — calibration doesn't pull weight even in combination.
# Pick-of-day keeps 'k' — untested with this swap, deliberately not merged
# since the two tracks solve different problems on purpose.
#
# Consensus ('c') was dropped after validation across multiple train/test
# splits showed it consistently overfits in-sample and hurts held-out ROI
# (e.g. 50/50 split: 4.0% test ROI with it vs 15.0% without). Movement
# profile ('mv') replaces it at a fixed 10% — not grid-searched, since only
# ~9% of games have a real (non-default-neutral) classification right now,
# too thin to trust a fitted weight. 10% mirrors the existing market-signal
# ('m') weight class since both track the same family of signal (CLV/line
# movement); the remaining 90% is grid-searched over each track's key set.
_MOVEMENT_FIXED_WEIGHT = 0.10
_PRIOR_UNIFIED = {'e': 0.45, 'f': 0.45, 'm': 0.00}
_UNIFIED_WEIGHTS = dict(_PRIOR_UNIFIED, mv=_MOVEMENT_FIXED_WEIGHT, n=0, computed_at=None)
_UNIFIED_LOCK = threading.Lock()

_PRIOR_PICK_OF_DAY = {'e': 0.45, 'k': 0.45, 'm': 0.00}
_PICK_OF_DAY_WEIGHTS = dict(_PRIOR_PICK_OF_DAY, mv=_MOVEMENT_FIXED_WEIGHT, n=0, computed_at=None)
_PICK_OF_DAY_LOCK = threading.Lock()


def _build_unified_rows(sport='MLB'):
    """Shared row-building for both weight-fit tracks — same component inputs,
    just fed into two different objective functions below."""
    with app.app_context():
        resolved = GamePrediction.query.filter(
            GamePrediction.sport == sport,
            GamePrediction.home_won.isnot(None),
            GamePrediction.home_odds.isnot(None),
            GamePrediction.away_odds.isnot(None),
            GamePrediction.pick_roi.isnot(None),
        ).all()

    rows = []
    for g in resolved:
        td = _trust_score(g.home_prob, g.home_odds, g.away_odds, g.factors_json, detail=True)
        if not td:
            continue
        lm = _line_move_pct(g.home_prob, g.closing_home_odds or g.home_odds,
                             g.closing_away_odds or g.away_odds,
                             g.opening_home_odds, g.opening_away_odds)
        signal_pct = lm if lm is not None else g.pick_clv
        m = max(0.0, min(1.0, 0.5 + signal_pct / 10.0)) if signal_pct is not None else 0.5
        mv = _movement_value(g.movement_profile)
        fav_home = (g.home_prob or 0.5) >= 0.5
        won = fav_home == bool(g.home_won)
        rows.append({'date': g.game_date, 'roi': g.pick_roi, 'won': won,
                     'e': td.get('e') or 0, 'k': td.get('k') or 0, 'f': td.get('f') or 0,
                     'm': m, 'mv': mv})
    return rows


# Key sets for the two tracks — deliberately different (see module comment
# above). Validated 2026-07-07 on 493 resolved MLB games via chronological
# train/test splits (50/60/70/80% train, held-out eval each time): swapping
# calibration ('k') for fav/dog ('f') in the win-rate objective raised average
# held-out top-3 win rate 57.3%→60.3% and ROI +5.0%→+12.4%, consistently across
# every split point tested — not a one-split fluke. A 4-variable grid search
# adding 'k' back alongside 'f' underperformed the 3-variable e/f/m blend
# (57.7%/+6.8%), confirming calibration isn't pulling weight even in
# combination. Pick-of-day (ROI-for-#1 objective) keeps 'k' — untested with
# this swap, and the two tracks solve different problems on purpose.
_UNIFIED_KEYS = ('e', 'f', 'm')
_PICK_OF_DAY_KEYS = ('e', 'k', 'm')


_CV_TEST_FRACS = (0.5, 0.6, 0.7, 0.8)  # chronological train fractions for held-out cut points


_GRID_MAX_WEIGHT = 0.55  # no single component may exceed this share — see docstring


def _grid_search_weights(rows, objective_fn, keys, test_fracs=_CV_TEST_FRACS,
                          max_weight=_GRID_MAX_WEIGHT):
    """Grid search over 3 components (from `keys`) summing to
    1 - _MOVEMENT_FIXED_WEIGHT, choosing the candidate with the best AVERAGE
    held-out objective across several chronological train/test cut points —
    not the raw full-sample argmax.

    Why the CV averaging: top3-win-rate (and top1-ROI) are noisy,
    discontinuous functions of ~3 picks/day. A plain full-sample argmax has
    no defense against overfitting that noise, and the alpha credibility
    blend below only guards against small-n noise (it's 1.0 — full trust,
    zero shrinkage — for any n ≥ 50). Observed live: going from 493→533
    resolved MLB games flipped the raw argmax from a balanced ~40/20/30 e/f/m
    split to a degenerate 0/0/90 one that underperformed the fixed-weight
    Sharp Score. Scoring each candidate by its average performance on several
    held-out windows (rather than the single window it was fit on) directly
    penalizes weight vectors that only work in-sample.

    Why max_weight too: CV averaging alone wasn't sufficient — at n=709
    (2026-07-30) the unified track still degenerated to e=0.15/f=0.70/m=0.05,
    concentrating the whole score on the single blunt Fav/Dog signal. Rerunning
    the same search capped at 0.55 landed on a much more diversified
    e=0.25/f=0.35/m=0.30 with held-out win rate 60.2% vs 61.0% uncapped — a
    <1pp difference, well within noise, for a materially more stable and
    interpretable weight vector. Since the CV mechanism only measures held-out
    fit and has no preference for diversification, and near-tied candidates are
    common with this few grid points, an explicit cap is needed to actually rule
    out the degenerate solutions rather than hope CV disfavors them.
    """
    from itertools import product as _iproduct
    dates = sorted(set(r['date'] for r in rows))
    cut_points = []
    for frac in test_fracs:
        split_idx = int(len(dates) * frac)
        train_dates = set(dates[:split_idx])
        test_rows = [r for r in rows if r['date'] not in train_dates]
        if test_rows:
            cut_points.append(test_rows)
    if not cut_points:
        cut_points = [rows]  # too few distinct days to split — fall back to in-sample

    remaining = round(1 - _MOVEMENT_FIXED_WEIGHT, 2)
    step = 0.05
    grid = [round(i * step, 2) for i in range(0, int(remaining / step) + 1)]
    best_w, best_val = None, -999
    for w1, w2 in _iproduct(grid, grid):
        w3 = round(remaining - w1 - w2, 2)
        if w3 < 0 or w3 > remaining:
            continue
        if max(w1, w2, w3, _MOVEMENT_FIXED_WEIGHT) > max_weight:
            continue
        weights = (w1, w2, w3)
        scores = [objective_fn(test_rows, weights, keys) for test_rows in cut_points]
        scores = [s for s in scores if s > -999]
        if not scores:
            continue
        val = sum(scores) / len(scores)
        if val > best_val:
            best_val, best_w = val, weights
    return best_w, best_val


def _ranked_day_groups(rows, weights, keys, top_n):
    from itertools import groupby as _igroupby
    w1, w2, w3 = weights
    k1, k2, k3 = keys
    rows_sorted = sorted(rows, key=lambda r: r['date'])
    out = []
    for _, grp in _igroupby(rows_sorted, key=lambda r: r['date']):
        day = sorted(grp, key=lambda r: -(
            w1 * r[k1] + w2 * r[k2] + w3 * r[k3] + _MOVEMENT_FIXED_WEIGHT * r['mv']))[:top_n]
        out.extend(day)
    return out


def _objective_top3_winrate(rows, weights, keys):
    day_picks = _ranked_day_groups(rows, weights, keys, 3)
    if not day_picks:
        return -999
    return sum(1 for r in day_picks if r['won']) / len(day_picks)


def _objective_top1_roi(rows, weights, keys):
    day_picks = _ranked_day_groups(rows, weights, keys, 1)
    if not day_picks:
        return -999
    return sum(r['roi'] for r in day_picks) / len(day_picks)


def _recompute_unified_weights(sport='MLB'):
    """Grid-search (e, f, m) weights maximizing WIN RATE of the top-3 picks/day,
    blended toward _PRIOR_UNIFIED by credibility alpha = min(n/50, 1.0). Powers
    the main recommendation table. Called at startup and after each nightly
    resolve, alongside _recompute_trust_weights."""
    global _UNIFIED_WEIGHTS

    rows = _build_unified_rows(sport)
    n = len(rows)
    if n < 10:
        return  # too few resolved games for any empirical signal

    best_w, best_val = _grid_search_weights(rows, _objective_top3_winrate, _UNIFIED_KEYS)

    alpha = min(n / 50.0, 1.0)
    fitted = dict(zip(_UNIFIED_KEYS, best_w))
    blended = {k: round(alpha * fitted[k] + (1 - alpha) * _PRIOR_UNIFIED[k], 4) for k in _PRIOR_UNIFIED}

    with _UNIFIED_LOCK:
        _UNIFIED_WEIGHTS = dict(blended, mv=_MOVEMENT_FIXED_WEIGHT, n=n,
                                 computed_at=datetime.now(timezone.utc).isoformat())

    print(f'[unified-weights] recomputed from {n} {sport} games: {blended} '
          f'(mv fixed at {_MOVEMENT_FIXED_WEIGHT}, objective=top3 win rate)', flush=True)


def _recompute_pick_of_day_weights(sport='MLB'):
    """Grid-search (e, k, m) weights maximizing ROI of the #1 pick/day only,
    blended toward _PRIOR_PICK_OF_DAY by credibility alpha = min(n/50, 1.0).
    Powers the featured Pick of the Day slot — deliberately a different
    objective than the main table's win-rate track (see module comment above)."""
    global _PICK_OF_DAY_WEIGHTS

    rows = _build_unified_rows(sport)
    n = len(rows)
    if n < 10:
        return

    best_w, best_val = _grid_search_weights(rows, _objective_top1_roi, _PICK_OF_DAY_KEYS)

    alpha = min(n / 50.0, 1.0)
    fitted = dict(zip(_PICK_OF_DAY_KEYS, best_w))
    blended = {k: round(alpha * fitted[k] + (1 - alpha) * _PRIOR_PICK_OF_DAY[k], 4) for k in _PRIOR_PICK_OF_DAY}

    with _PICK_OF_DAY_LOCK:
        _PICK_OF_DAY_WEIGHTS = dict(blended, mv=_MOVEMENT_FIXED_WEIGHT, n=n,
                                     computed_at=datetime.now(timezone.utc).isoformat())

    print(f'[pick-of-day-weights] recomputed from {n} {sport} games: {blended} '
          f'(mv fixed at {_MOVEMENT_FIXED_WEIGHT}, objective=top1 ROI)', flush=True)


def _odds_multiplier(pick_odds):
    """Post-score adjustment surfacing underdog value the weighted formula
    can't see on its own — none of the grid-searched components (edge,
    calibration/fav-dog, line-move, movement) distinguish a +120 dog from
    a -130 favorite at equivalent edge. Historically (n=463 resolved MLB
    games, 2026-06-29 analysis) dogs at +110-or-worse returned +54.3% ROI vs
    -13.7% for heavy favorites at -150-or-better, with the bulk of volume
    (favorites + pick'em, ~92% of games) flat-to-negative. Validated via
    train/test split before adopting: held up with consistent improvement
    across Top1-5 on 2 of 3 splits; mixed on the third (the same chronological
    split that's been a noisy outlier for every other change validated this
    session — not a multiplier-specific weakness).

    Deliberately applied AFTER the weighted sum, not folded into it — these
    values are hand-set from the bucket breakdown, not grid-searched, so they
    shouldn't be treated as equally validated as e/k/m. Revisit if the
    underdog sample (currently n=37) starts moving these numbers meaningfully.
    """
    if pick_odds is None:
        return 1.0
    try:
        odds = int(pick_odds)
    except (TypeError, ValueError):
        return 1.0
    if odds <= -150:
        return 0.70
    if odds <= -110:
        return 0.85
    if odds < 110:
        return 1.0
    return 1.2


def _unified_score(td, market_val=0.5, movement_val=0.5, pick_odds=None):
    """0-100 composite score using the win-rate-optimized (e, f, m) weights
    plus a fixed 10% movement-profile weight, then an odds-bucket multiplier
    (see _odds_multiplier) to surface underdog value the weighted sum can't
    see. Powers the main recommendation table (all ranks)."""
    if not td:
        return 0
    with _UNIFIED_LOCK:
        w = dict(_UNIFIED_WEIGHTS)
    raw = (w['mv'] * movement_val +
           w['e'] * (td.get('e') or 0) +
           w['f'] * (td.get('f') or 0) +
           w['m'] * market_val)
    return round(raw * 100 * _odds_multiplier(pick_odds))


# Strong-bet floor for Pick of the Day — without this, the ROI-optimized track
# (e=65% weight on raw edge, plus the underdog multiplier) can surface a thin,
# barely-above-coinflip edge on a big payout as the "best" ROI score, even
# though that's a worse real-world bet than a more confident pick with smaller
# theoretical ROI. Two checks, both must pass:
#   pick_prob >= 0.55  — meaningfully favored, not just nominally above 50%
#   calib component (td['k']) >= 0.45 — bucket isn't a known danger zone
#                                        (e.g. the 56-58% trap), independent
#                                        of the ROI-track's own k weight
_POD_MIN_PROB = 0.55
_POD_MIN_CALIB = 0.45


def _is_strong_bet(pick_prob, td):
    if pick_prob is None or pick_prob < _POD_MIN_PROB:
        return False
    if not td or (td.get('k') or 0) < _POD_MIN_CALIB:
        return False
    return True


def _pick_of_day_score(td, market_val=0.5, movement_val=0.5, pick_odds=None):
    """Same formula as _unified_score, using the ROI-for-#1-pick-optimized
    weight track instead. Only meaningful for candidates that already passed
    _is_strong_bet — this function doesn't apply that floor itself."""
    if not td:
        return 0
    with _PICK_OF_DAY_LOCK:
        w = dict(_PICK_OF_DAY_WEIGHTS)
    raw = (w['mv'] * movement_val +
           w['e'] * (td.get('e') or 0) +
           w['k'] * (td.get('k') or 0) +
           w['m'] * market_val)
    return round(raw * 100 * _odds_multiplier(pick_odds))


_UNIFIED_RANK_BUCKETS = [('#1', 1, 1), ('#2-3', 2, 3), ('#4-6', 4, 6), ('#7+', 7, 999)]
_UNIFIED_RANK_STATS = {}   # {sport: {bucket_label: {'n': int, 'wr': float}}}
_UNIFIED_RANK_LOCK = threading.Lock()


def _unified_rank_bucket(rank):
    for label, lo, hi in _UNIFIED_RANK_BUCKETS:
        if lo <= rank <= hi:
            return label
    return None


def _recompute_unified_rank_stats(sport='MLB'):
    """Retroactive win-rate per Unified-Score daily-rank bucket (#1, #2-3, #4-6, #7+),
    from resolved games with odds. Same ranking logic as the Rank Comparison table
    in model_performance(), cached here so /api/empirical_info can look it up cheaply.
    """
    from itertools import groupby as _igroupby

    with app.app_context():
        resolved = GamePrediction.query.filter(
            GamePrediction.sport == sport,
            GamePrediction.home_won.isnot(None),
            GamePrediction.home_odds.isnot(None),
            GamePrediction.away_odds.isnot(None),
        ).all()

    games = [p for p in resolved if p.pick_roi is not None and p.home_odds and p.away_odds]
    bucket_games = {label: [] for label, _, _ in _UNIFIED_RANK_BUCKETS}

    for _dt, _grp in _igroupby(sorted(games, key=lambda x: x.game_date), key=lambda x: x.game_date):
        day = list(_grp)
        scored = []
        for g in day:
            td = _trust_score(g.home_prob, g.home_odds, g.away_odds, g.factors_json, detail=True)
            lm = _line_move_pct(g.home_prob, g.closing_home_odds or g.home_odds,
                                 g.closing_away_odds or g.away_odds,
                                 g.opening_home_odds, g.opening_away_odds)
            signal_pct = lm if lm is not None else g.pick_clv
            m_val = max(0.0, min(1.0, 0.5 + signal_pct / 10.0)) if signal_pct is not None else 0.5
            mv_val = _movement_value(g.movement_profile)
            fav_home = (g.home_prob or 0.5) >= 0.5
            pick_odds = g.home_odds if fav_home else g.away_odds
            scored.append((g, _unified_score(td, m_val, mv_val, pick_odds)))
        scored.sort(key=lambda x: -x[1])
        for rank, (g, _) in enumerate(scored, 1):
            label = _unified_rank_bucket(rank)
            if label:
                bucket_games[label].append(g)

    stats = {}
    for label, _, _ in _UNIFIED_RANK_BUCKETS:
        bkt = bucket_games[label]
        if not bkt:
            continue
        fav_home = {g.id: (g.home_prob or 0.5) >= 0.5 for g in bkt}
        wins = sum(1 for g in bkt if fav_home[g.id] == bool(g.home_won))
        stats[label] = {'n': len(bkt), 'wr': round(100 * wins / len(bkt), 1)}

    with _UNIFIED_RANK_LOCK:
        _UNIFIED_RANK_STATS[sport] = stats

    print(f'[unified-rank-stats] recomputed from {len(games)} {sport} games: {stats}', flush=True)


# Direction-split: 'for' = line moved toward our pick, 'against' = toward the
# opponent. 'flat'/'static'/'no_data' aren't split — their whole definition is
# "no meaningful net move," so a direction would just be noise. Splitting the
# other 5 shapes roughly halves an already-thin sample per bucket, but lumping
# "Late Sharp toward us" with "Late Sharp against us" was hiding the one
# distinction that actually matters for whether a sharp move is good or bad news.
_MOVEMENT_SHAPES_DIRECTIONAL = ['early_sharp', 'late_sharp', 'sustained', 'reversal', 'minimal']
_MOVEMENT_PROFILES = (
    [f'{s}_for' for s in _MOVEMENT_SHAPES_DIRECTIONAL] +
    [f'{s}_against' for s in _MOVEMENT_SHAPES_DIRECTIONAL] +
    ['flat', 'static', 'no_data']
)
_PRIOR_MOVEMENT = {p: 0.5 for p in _MOVEMENT_PROFILES}
_MOVEMENT_WEIGHTS = dict(_PRIOR_MOVEMENT, n=0, computed_at=None)
_MOVEMENT_LOCK = threading.Lock()


def _recompute_movement_profiles(sport='MLB'):
    """Classify and store the pre-game movement profile for any game missing
    one yet, using whatever odds_snapshot history is available. Most games
    will return None (no snapshots, or fewer than 3) until the snapshot table
    builds up real history — that's expected, not an error.

    Also reclassifies any game still on the old undirected scheme (e.g. plain
    'late_sharp' from before the for/against split) — self-healing so a
    one-time DB migration isn't needed when the label scheme changes."""
    import odds_history
    _old_scheme_labels = _MOVEMENT_SHAPES_DIRECTIONAL  # e.g. 'late_sharp' with no _for/_against suffix
    with app.app_context():
        preds = GamePrediction.query.filter(
            GamePrediction.sport == sport,
            db.or_(
                GamePrediction.movement_profile.is_(None),
                GamePrediction.movement_profile.in_(_old_scheme_labels),
            ),
        ).all()
        odds_sport_key = 'baseball_mlb' if sport == 'MLB' else sport.lower()
        n_classified = 0
        for p in preds:
            fav_home = (p.home_prob or 0.5) >= 0.5
            game_start = None
            if p.game_time_utc:
                try:
                    game_start = datetime.fromisoformat(p.game_time_utc.replace('Z', '+00:00'))
                except Exception:
                    game_start = None
            profile, move_pct = odds_history.classify_movement(
                odds_sport_key, p.home_team, p.away_team, fav_home, game_start)
            if profile:
                p.movement_profile = profile
                p.movement_pct = move_pct
                n_classified += 1
        if n_classified:
            db.session.commit()
    print(f'[movement-profiles] classified {n_classified} new {sport} games '
          f'(of {len(preds)} missing a profile)', flush=True)


def _recompute_movement_weights(sport='MLB'):
    """ROI-derived weight per movement profile, same credibility-blend pattern
    as _recompute_trust_weights — blends toward 0.5 (neutral prior) until a
    profile has enough resolved games to trust empirically."""
    global _MOVEMENT_WEIGHTS

    with app.app_context():
        resolved = GamePrediction.query.filter(
            GamePrediction.sport == sport,
            GamePrediction.home_won.isnot(None),
            GamePrediction.movement_profile.isnot(None),
            GamePrediction.pick_roi.isnot(None),
        ).all()

    n_total = len(resolved)
    base_vals = [p.pick_roi for p in resolved]
    baseline_roi = (sum(base_vals) / len(base_vals) * 100) if base_vals else 0.0

    bkts = {p: [] for p in _MOVEMENT_PROFILES}
    for p in resolved:
        if p.movement_profile in bkts:
            bkts[p.movement_profile].append(p.pick_roi * 100)

    NORM = 60.0
    MIN_N = 5

    def _blend(vals, prior):
        n = len(vals)
        if n < MIN_N:
            return prior
        avg = sum(vals) / n
        empirical = max(0.05, min(1.0, 0.5 + (avg - baseline_roi) / NORM))
        alpha = min(n / 50.0, 1.0)
        return round(alpha * empirical + (1 - alpha) * prior, 4)

    new_weights = {p: _blend(bkts[p], _PRIOR_MOVEMENT[p]) for p in _MOVEMENT_PROFILES}
    new_weights['n'] = n_total
    new_weights['computed_at'] = datetime.now(timezone.utc).isoformat()

    with _MOVEMENT_LOCK:
        _MOVEMENT_WEIGHTS.update(new_weights)

    print(f'[movement-weights] recomputed from {n_total} classified {sport} games: {new_weights}', flush=True)


def _movement_value(profile):
    """0-1 weight for a stored movement profile label, or 0.5 (neutral) if
    the game has no profile yet (insufficient snapshot history)."""
    if not profile:
        return 0.5
    with _MOVEMENT_LOCK:
        return _MOVEMENT_WEIGHTS.get(profile, 0.5)


def _upsert_predictions(schedule, sport='MLB'):
  """Save model predictions and record outcomes for completed games."""
  now = datetime.now(timezone.utc)
  changed = False
  for day in (schedule or []):
    for game in day.get('games', []):
      # NFL preseason (nfl_api.build_schedule_context tags is_preseason from
      # ESPN's season.type) is still shown live but never tracked here —
      # backup-heavy, min-effort results would corrupt regular-season
      # accuracy/calibration on the Model Performance page. .get() is False
      # for MLB/NHL games, which don't set this key.
      if game.get('is_preseason'):
        continue
      model     = game.get('model')
      wx        = game.get('weather') or {}
      has_wind  = wx.get('wind_mph') is not None
      # Skip games we have nothing to record for
      if not model and not has_wind:
        continue
      status    = game.get('status', 'Preview')
      game_date = (game.get('game_time_utc') or '')[:10]
      home_name = (game.get('home') or {}).get('name', '')
      away_name = (game.get('away') or {}).get('name', '')
      if not game_date or not home_name or not away_name:
        continue

      odds = game.get('odds') or {}
      pred = GamePrediction.query.filter_by(
          sport=sport, game_date=game_date,
          home_team=home_name, away_team=away_name,
      ).first()

      if pred is None:
        pred = GamePrediction(
            sport=sport, game_date=game_date,
            game_time_utc=game.get('game_time_utc', ''),
            home_team=home_name, away_team=away_name,
        )
        db.session.add(pred)

      # Refresh prediction while game hasn't started
      if pred.home_won is None and status == 'Preview':
        if model:
          pred.home_prob    = model.get('home_prob')
          pred.away_prob    = model.get('away_prob')
          new_h_odds        = odds.get('home_best')
          new_a_odds        = odds.get('away_best')
          # Capture opening odds the first time we see valid odds for this game.
          # Prefer the true market opening from odds_history snapshot; fall back to current.
          if new_h_odds and pred.opening_home_odds is None:
            pred.opening_home_odds = odds.get('opening_home') or new_h_odds
            pred.opening_away_odds = odds.get('opening_away') or new_a_odds
          pred.home_odds    = new_h_odds
          pred.away_odds    = new_a_odds
          pred.factors_json = json.dumps(model.get('factors', []))
          edge = _pick_edge_pct(pred.home_prob, pred.home_odds, pred.away_odds)
          if edge is not None:
            pred.edge_pct = edge
        if has_wind and pred.wind_mph is None:
          pred.wind_mph = wx['wind_mph']
          pred.wind_dir = wx.get('wind_dir')
        changed = True

      # Capture closing odds once game goes Live — best approximation of closing line
      if pred.home_won is None and status == 'Live' and pred.closing_home_odds is None:
        if odds.get('home_best') is not None:
          pred.closing_home_odds = odds.get('home_best')
          pred.closing_away_odds = odds.get('away_best')
          # CLV: how much closing implied prob moved toward our pick vs opening
          if pred.home_odds and pred.away_odds:
            vf_open_h, vf_open_a   = _vig_free_implied(pred.home_odds, pred.away_odds)
            vf_close_h, vf_close_a = _vig_free_implied(pred.closing_home_odds, pred.closing_away_odds)
            if vf_open_h and vf_close_h:
              fav_home   = (pred.home_prob or 0.5) >= 0.5
              open_pick  = vf_open_h  if fav_home else vf_open_a
              close_pick = vf_close_h if fav_home else vf_close_a
              pred.pick_clv = round((close_pick - open_pick) * 100, 2)
          changed = True

      # Record outcome once game is Final
      if status == 'Final' and pred.home_won is None:
        try:
          h = int(game['home'].get('score') or -1)
          a = int(game['away'].get('score') or -1)
          if h >= 0 and a >= 0:
            pred.home_won       = (h > a)
            pred.home_score     = h
            pred.away_score     = a
            pred.outcome_set_at = now
            # ROI per unit for the model's pick
            fav_home   = (pred.home_prob or 0.5) >= 0.5
            pick_odds  = pred.home_odds if fav_home else pred.away_odds
            if pick_odds:
              try:
                o = int(pick_odds)
                profit = o / 100.0 if o > 0 else 100.0 / (-o)
                model_won = fav_home == pred.home_won
                pred.pick_roi = round(profit if model_won else -1.0, 4)
              except (TypeError, ValueError):
                pass
            changed = True
        except (TypeError, ValueError):
          pass

  if changed:
    try:
      db.session.commit()
    except Exception:
      db.session.rollback()


# Kelly calculation function with space to tweak using closed bets history
def compute_recommended_amount(bankroll, percent_bankroll, odds, prob, closed_bets, sport=None, bet_type=None):
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
    now = datetime.now(timezone.utc)
    weights_sum = 0.0
    weighted_wins = 0.0
    for cb in closed_bets or []:
      cb_sport = (getattr(cb, 'sport', '') or '').strip()
      cb_type = (getattr(cb, 'bet_type', '') or '').strip()
      if sport and cb_sport.lower() != sport.strip().lower():
        continue
      if bet_type and cb_type != bet_type.strip():
        continue

      try:
        age_days = max(0.0, (now - (cb.closed_at or now)).total_seconds() / 86400.0)
      except Exception:
        age_days = 0.0

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



def _naive(dt):
  """Strip timezone so naive and aware datetimes compare cleanly."""
  if dt is None:
    return datetime(1970, 1, 1)
  return dt.replace(tzinfo=None) if dt.tzinfo else dt


def compute_stats(open_bets, closed_bets):
  # Exclude paper bets from all financial stats
  real_open   = [b for b in open_bets   if not b.is_paper]
  real_closed  = [b for b in closed_bets if not b.is_paper]
  decided      = [b for b in real_closed if b.outcome in ('win', 'loss')]
  total_staked = sum(cb.stake for cb in real_closed)
  total_profit = sum(cb.profit for cb in real_closed)
  wins         = sum(1 for cb in decided if cb.outcome == 'win')
  cashouts     = sum(1 for cb in real_closed if cb.outcome == 'cashout')
  n = len(real_closed)
  nd = len(decided)
  return {
    'total_pl':     round(total_profit, 2),
    'win_rate':     round(wins / nd * 100, 1) if nd else 0,
    'roi':          round(total_profit / total_staked * 100, 1) if total_staked else 0,
    'total_staked': round(total_staked, 2),
    'wins':         wins,
    'losses':       nd - wins,
    'cashouts':     cashouts,
    'total_closed': n,
    'open_count':   len(real_open),
    'open_staked':  round(sum(b.stake for b in real_open), 2),
  }


def compute_chart_data(closed_bets):
  closed_bets = [b for b in closed_bets if not b.is_paper]
  sorted_bets = sorted(closed_bets, key=lambda b: _naive(b.closed_at))

  # 1. Cumulative P&L over time
  cum_pl, running = [], 0.0
  for b in sorted_bets:
    running += b.profit
    cum_pl.append({'x': _naive(b.closed_at).strftime('%m/%d'), 'y': round(running, 2)})

  # 2 & 3. By sport and by bet type
  by_sport = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profit': 0.0})
  by_type  = defaultdict(lambda: {'wins': 0, 'losses': 0})
  for b in closed_bets:
    s = (b.sport    or 'Other').strip() or 'Other'
    t = (b.bet_type or 'Other').strip() or 'Other'
    key = 'wins' if b.outcome == 'win' else 'losses'
    by_sport[s][key] += 1
    by_sport[s]['profit'] = round(by_sport[s]['profit'] + b.profit, 2)
    by_type[t][key] += 1

  # 4. 7-day activity
  now = datetime.now(timezone.utc).replace(tzinfo=None)
  day_labels = [(now - timedelta(days=i)).strftime('%m/%d') for i in range(6, -1, -1)]
  daily = {d: 0 for d in day_labels}
  for b in closed_bets:
    label = _naive(b.closed_at).strftime('%m/%d')
    if label in daily:
      daily[label] += 1

  # 5. Expected vs actual win rate by sport
  ev = defaultdict(lambda: {'prob_sum': 0.0, 'count': 0, 'wins': 0})
  for b in closed_bets:
    s = (b.sport or 'Other').strip() or 'Other'
    ev[s]['prob_sum'] += (b.prob or 0.0)
    ev[s]['count']    += 1
    if b.outcome == 'win':
      ev[s]['wins'] += 1
  ev_cmp = {
    s: {
      'expected': round(v['prob_sum'] / v['count'] * 100, 1) if v['count'] else 0,
      'actual':   round(v['wins']     / v['count'] * 100, 1) if v['count'] else 0,
      'count':    v['count'],
    }
    for s, v in ev.items()
  }

  # 6. Calibration: predicted probability bucket → actual win rate
  calib_buckets = defaultdict(lambda: {'total': 0, 'wins': 0})
  for b in closed_bets:
    prob = b.prob or 0.5
    bucket = round(round(prob / 0.05) * 0.05, 2)
    bucket = max(0.05, min(0.95, bucket))
    calib_buckets[bucket]['total'] += 1
    if b.outcome == 'win':
      calib_buckets[bucket]['wins'] += 1

  calibration = sorted([
    {
      'prob':   round(k * 100, 0),
      'actual': round(v['wins'] / v['total'] * 100, 1),
      'count':  v['total'],
    }
    for k, v in calib_buckets.items() if v['total'] >= 2
  ], key=lambda x: x['prob'])

  # 7. CLV over time — individual values + running average
  clv_points = []
  running_clv = 0.0
  for b in sorted_bets:
    if b.clv is not None:
      running_clv += b.clv
      n = len(clv_points) + 1
      clv_points.append({
        'x':    _naive(b.closed_at).strftime('%m/%d'),
        'clv':  b.clv,
        'avg':  round(running_clv / n, 2),
        'name': b.name,
      })

  return {
    'cumulative_pl': cum_pl,
    'by_sport':      {s: dict(v) for s, v in by_sport.items()},
    'by_type':       {t: dict(v) for t, v in by_type.items()},
    'daily':         {'labels': day_labels, 'counts': [daily[d] for d in day_labels]},
    'ev_comparison': ev_cmp,
    'calibration':   calibration,
    'clv':           clv_points,
  }


# Routes
_SPORT_ODDS_KEY = {'MLB': 'baseball_mlb', 'NHL': 'icehockey_nhl', 'NFL': 'americanfootball_nfl', 'CFB': 'americanfootball_ncaaf'}


def _annotate_open_bets(open_bets):
    """
    Attaches live CLV, line movement, and closing-line-suggestion display
    fields to each OpenBet in place (transient — computed for this render
    only, never persisted to the DB). Shared by the Dashboard (all sports)
    and any sport-scoped page (e.g. /mlb) that wants to show its own open
    bets without duplicating this ~40-line block per route.

    Returns closing_suggestions: {bet_id: american_odds}, for the close-bet
    modal's auto-fill — the caller passes this to its own template.
    """
    import odds_history as _oh

    def _to_american(decimal):
        if not decimal or decimal <= 1:
            return None
        if decimal >= 2.0:
            return int(round((decimal - 1) * 100))
        return int(round(-100 / (decimal - 1)))

    closing_suggestions = {}
    now_utc = datetime.now(timezone.utc)
    for bet in open_bets:
        bet.live_clv          = None
        bet.original_american = _to_american(bet.odds)
        bet.current_american  = None
        bet.line_move         = None
        bet.closing_american  = None  # pre-game closing line (American odds)
        bet.closing_clv_pos   = None  # True = beat the close, False = didn't
        if not (bet.game_key and bet.bet_side and bet.sport):
            continue
        sk = _SPORT_ODDS_KEY.get(bet.sport.upper(), '')
        if not sk:
            continue
        # Determine whether the game has started (with timezone-safe comparison)
        game_started = False
        es = None
        if bet.eventstart:
            es = bet.eventstart if bet.eventstart.tzinfo else bet.eventstart.replace(tzinfo=timezone.utc)
            game_started = now_utc >= es

        bet.cashout_value = None  # estimated cashout amount at current market

        if not game_started:
            # Pre-game: show live pre-game line movement
            latest = _oh.get_latest(sk, bet.game_key)
            if latest:
                ca  = latest[f'{bet.bet_side}_odds']
                mkt = ca / 100.0 + 1.0 if ca > 0 else 100.0 / abs(ca) + 1.0
                bet.current_american = ca
                bet.live_clv  = round((bet.odds / mkt - 1) * 100, 1)
                if bet.original_american is not None:
                    bet.line_move = ca - bet.original_american
                # Cashout = fair value of position at current market price
                if mkt > 1 and bet.odds and bet.stake:
                    bet.cashout_value = round(bet.stake * bet.odds / mkt, 2)
        else:
            # Post-start: show closing line — ignore in-game odds entirely
            if es:
                snap = _oh.get_closing_line(sk, bet.game_key, es)
                if snap:
                    ca = snap[f'{bet.bet_side}_odds']
                    closing_suggestions[bet.id] = ca
                    bet.closing_american = ca
                    cl_dec = ca / 100.0 + 1.0 if ca > 0 else 100.0 / abs(ca) + 1.0
                    closing_clv = (bet.odds / cl_dec - 1) * 100 if cl_dec > 1 else 0
                    bet.closing_clv_pos = closing_clv >= 0

    return closing_suggestions


def _tier_open_bets(real_open):
    """
    Per-bet book-implied prob / edge / EV / tier grading for non-paper open
    bets. Split out from _annotate_open_bets so callers needing just the
    CLV/line-move side (or just the tier side) don't have to run both, but
    in practice the Dashboard and /mlb use both together.
    """
    for bet in real_open:
        implied = 1.0 / bet.odds if bet.odds and bet.odds > 1 else None
        bet.book_implied = round(implied * 100, 1) if implied else None
        bet.edge_pct     = round((bet.prob - implied) * 100, 1) if (bet.prob and implied) else None
        bet.ev_pct       = round((bet.prob * bet.odds - 1) * 100, 1) if (bet.prob and bet.odds and bet.odds > 1) else None
        ep = bet.edge_pct
        if ep is None:
            bet.tier, bet.tier_cls = None, ''
        elif ep >= 7:
            bet.tier, bet.tier_cls = 'A+', 'tier-aplus'
        elif ep >= 4:
            bet.tier, bet.tier_cls = 'B',  'tier-b'
        elif ep >= 2:
            bet.tier, bet.tier_cls = 'Lean', 'tier-lean'
        else:
            bet.tier, bet.tier_cls = 'Pass', 'tier-pass'


def _open_bets_summary_stats(real_open):
    """Aggregate summary row (avg edge / avg CLV / % positive CLV / expected
    ROI) over a set of already-annotated-and-tiered non-paper open bets.
    Used for both the Dashboard's all-sports summary row and a sport-scoped
    page's own summary row (e.g. /mlb, MLB-only)."""
    bets_with_edge = [b for b in real_open if b.edge_pct is not None]
    bets_with_clv  = [b for b in real_open if b.live_clv is not None]
    return {
        'avg_edge':         round(sum(b.edge_pct for b in bets_with_edge) / len(bets_with_edge), 1) if bets_with_edge else None,
        'avg_clv':          round(sum(b.live_clv for b in bets_with_clv) / len(bets_with_clv), 1) if bets_with_clv else None,
        'positive_clv_pct': round(sum(1 for b in bets_with_clv if b.live_clv > 0) / len(bets_with_clv) * 100) if bets_with_clv else None,
        'expected_roi':     round(sum(b.ev_pct for b in bets_with_edge) / len(bets_with_edge), 1) if bets_with_edge else None,
    }


def _match_open_bets_to_games(days, sport=None):
    """
    Attaches game['open_bets'] = [OpenBet, ...] to each game in `days` (a
    list of {'games': [...]} dicts, as produced by each sport's
    build_schedule_context()/build_week_schedule_context()) so a placed bet
    is visible — and closeable — right on its game card. Matches on
    game_key first, falling back to a leading team-abbreviation match in the
    bet name for bets placed manually or before game_key existed. Scoped to
    `sport` when given so an abbreviation collision can't attach a bet to
    the wrong sport's game.
    """
    from odds_api import _normalize
    import re as _re

    query = OpenBet.query
    if sport:
        query = query.filter_by(sport=sport)
    all_open = query.all()

    by_key = {}
    for bet in all_open:
        if bet.game_key:
            by_key.setdefault(bet.game_key, []).append(bet)

    by_abbr = {}
    for bet in all_open:
        if bet.game_key:
            continue
        m = _re.match(r'^([A-Z]{2,4})\b', bet.name or '')
        if m:
            by_abbr.setdefault(m.group(1).upper(), []).append(bet)

    for day in days:
        for game in day.get('games', []):
            home, away = game.get('home') or {}, game.get('away') or {}
            gk = f"{_normalize(home.get('name', ''))}_{_normalize(away.get('name', ''))}"
            matched = list(by_key.get(gk, []))
            utc_raw = game.get('game_time_utc', '')
            if utc_raw:
                try:
                    utc_dt = datetime.fromisoformat(utc_raw.replace('Z', '+00:00'))
                    gk_ts = f"{gk}_{utc_dt.strftime('%Y-%m-%dT%H')}"
                    for bet in by_key.get(gk_ts, []):
                        if bet not in matched:
                            matched.append(bet)
                except Exception:
                    pass
            for abbr in (away.get('abbrev') or away.get('abbr'), home.get('abbrev') or home.get('abbr')):
                if not abbr:
                    continue
                for bet in by_abbr.get(abbr.upper(), []):
                    if bet not in matched:
                        matched.append(bet)
            game['open_bets'] = matched


@app.context_processor
def _inject_global_exposure():
    """
    Bankroll/exposure summary shown in a persistent strip on every page
    (base.html) regardless of which sport you're looking at.

    Why this exists as a context processor (auto-injected everywhere) rather
    than a section on one page: bankroll and Kelly % are global settings,
    not per-sport, so a sport-scoped page's own open-bet total can't tell
    you whether you're overextended once positions exist in more than one
    sport at once (real overlap window: MLB regular season runs into NFL's
    opening weeks each September). This keeps that total visible everywhere
    without needing a dedicated cross-sport Dashboard as the landing page.

    Wrapped in try/except: context processors run on every render, including
    templates rendered before the DB is fully migrated/seeded — a failure
    here shouldn't take down every page.
    """
    try:
        settings = Setting.query.first()
        if not settings:
            return {'global_exposure': None}
        open_bets   = OpenBet.query.filter_by(is_paper=False).all()
        open_staked = sum((b.stake or 0.0) for b in open_bets)
        # Total working capital = uncommitted bankroll + whatever's already
        # staked (matches the _adj_bankroll convention used for unit sizing
        # elsewhere — settings.bankroll alone is the UNCOMMITTED cash figure).
        total_capital = (settings.bankroll or 0.0) + open_staked
        pct_committed = round(open_staked / total_capital * 100, 1) if total_capital else None
        return {'global_exposure': {
            'bankroll':      round(settings.bankroll or 0.0, 2),
            'open_staked':   round(open_staked, 2),
            'open_count':    len(open_bets),
            'pct_committed': pct_committed,
        }}
    except Exception:
        return {'global_exposure': None}


@app.route('/')
def index():
  """Retired as a distinct cross-sport Dashboard — MLB is the effective
  landing page now. Kept as a redirect (not deleted) rather than renamed,
  since a lot of other routes call redirect(url_for('index')) as their
  generic "go home" behavior after a bet action; changing the destination
  here is far less invasive than updating every one of those call sites.
  The old all-sports bet list this used to render is gone — the persistent
  bankroll/exposure strip (see _inject_global_exposure) covers the
  cross-sport visibility it provided. Known gap: NHL/NFL open bets have no
  dedicated view now (their schedule pages don't have a live-bets section
  yet, unlike /mlb) — fine for now since neither has real betting volume,
  worth revisiting once either does.
  """
  return redirect(url_for('mlb_schedule'))

@app.route('/new-bet')
def new_bet():
  settings = Setting.query.first()
  if not settings:
    settings = Setting(bankroll=50, percent_bankroll=0.25)
  return render_template('new_bet.html', settings=settings)

@app.route('/history')
def history():
  open_bets   = OpenBet.query.all()
  closed_bets = ClosedBet.query.order_by(ClosedBet.closed_at.desc()).all()
  stats       = compute_stats(open_bets, closed_bets)
  chart_data  = compute_chart_data(closed_bets)

  settings  = Setting.query.first()
  # Add open stakes back so unit_size isn't deflated by active positions
  _adj_bankroll = (settings.bankroll + stats['open_staked']) if settings else 1.0
  unit_size = max(0.01, round(_adj_bankroll * settings.percent_bankroll, 4))

  real_closed = [b for b in closed_bets if not b.is_paper]

  # CLV summary across all closed bets with a closing line
  clv_vals = [b.clv for b in real_closed if b.clv is not None]
  clv_stats = {
      'avg':          round(sum(clv_vals) / len(clv_vals), 1) if clv_vals else None,
      'positive_pct': round(sum(1 for v in clv_vals if v > 0) / len(clv_vals) * 100)
                      if clv_vals else None,
      'count':        len(clv_vals),
  }

  # Units profit
  units_profit = round(stats['total_pl'] / unit_size, 2) if unit_size else 0

  # Recent trend — last 10 real closed bets
  recent = sorted(real_closed, key=lambda b: _naive(b.closed_at), reverse=True)[:10]
  recent_wins = sum(1 for b in recent if b.outcome == 'win')
  trend = {
      'wins':   recent_wins,
      'losses': len(recent) - recent_wins,
      'n':      len(recent),
      'up':     (recent_wins / len(recent)) > (stats['win_rate'] / 100) if recent else None,
  }

  # Sport list for filter dropdown
  sports_list = sorted({(b.sport or '').strip() for b in real_closed if b.sport and b.sport.strip()})

  imported = request.args.get('imported', type=int)
  skipped  = request.args.get('skipped',  type=int)
  # Optional ?sport=MLB — pre-applies the existing client-side sport filter
  # so a link from a sport-scoped page (e.g. /mlb) lands already filtered,
  # without needing a separate server-rendered history-per-sport route.
  initial_sport = request.args.get('sport', '').strip().upper()
  return render_template('history.html', closed_bets=closed_bets, stats=stats,
                         chart_data=chart_data, imported=imported, skipped=skipped,
                         unit_size=unit_size, clv_stats=clv_stats,
                         units_profit=units_profit, trend=trend,
                         sports_list=sports_list, initial_sport=initial_sport,
                         subnav_sport=(initial_sport or None))


@app.route('/import_bets', methods=['POST'])
def import_bets():
  raw = request.form.get('csv_data', '').strip().lstrip('﻿')
  if not raw:
    return redirect(url_for('history'))

  imported = 0
  skipped  = 0

  def _get(row, *keys):
    """Case/whitespace-insensitive multi-key lookup."""
    for k in keys:
      v = row.get(k) or row.get(k.replace('_', ' '))
      if v and str(v).strip():
        return str(v).strip()
    return ''

  try:
    try:
      dialect = csv.Sniffer().sniff(raw[:2048], delimiters=',\t;|')
    except csv.Error:
      dialect = csv.excel  # fallback to comma
    reader = csv.DictReader(io.StringIO(raw), dialect=dialect)
    for row in reader:
      try:
        # Normalize all keys: strip whitespace + lowercase
        row = {(k or '').strip().lower(): (v or '').strip()
               for k, v in row.items() if k is not None}

        # ── Required fields ──────────────────────────────
        odds_raw    = _get(row, 'odds', 'american_odds', 'line', 'price')
        stake_raw   = _get(row, 'stake', 'amount', 'wager', 'bet')
        outcome_raw = _get(row, 'outcome', 'result').lower()

        if not odds_raw or not stake_raw:
          if imported == 0 and skipped == 0:
            print(f"[import] first row keys={list(row.keys())} odds={odds_raw!r} stake={stake_raw!r} outcome={outcome_raw!r}", flush=True)
          skipped += 1; continue

        # Normalize outcome: accept win/loss/w/l
        if outcome_raw in ('w', 'win'):
          outcome = 'win'
        elif outcome_raw in ('l', 'loss'):
          outcome = 'loss'
        else:
          skipped += 1; continue

        american_odds = float(odds_raw)
        stake         = float(stake_raw)
        odds = american_odds / 100.0 + 1.0 if american_odds > 0 else 100.0 / abs(american_odds) + 1.0
        profit = stake * (odds - 1.0) if outcome == 'win' else -stake

        # ── Optional fields ──────────────────────────────
        name     = _get(row, 'name', 'bet_name', 'bet') or 'Bet'
        sport    = _get(row, 'sport')
        bet_type = _get(row, 'bet_type', 'type') or 'Moneyline'
        notes    = _get(row, 'notes', 'note', 'comment')
        date_str = _get(row, 'date', 'closed_at', 'settled')

        # Prob: accept 0.55 or 55 or 55%
        prob_raw = _get(row, 'prob', 'probability', 'win_prob') or '0.5'
        try:
          prob = float(prob_raw.rstrip('%'))
          if prob > 1.0:
            prob /= 100.0
          prob = max(0.01, min(0.99, prob))
        except (ValueError, TypeError):
          prob = 0.5

        # Date: try several common formats
        closed_at = datetime.now(timezone.utc)
        if date_str:
          for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y', '%Y/%m/%d', '%d-%m-%Y'):
            try:
              closed_at = datetime.strptime(date_str, fmt); break
            except ValueError:
              continue

        db.session.add(ClosedBet(
          name=name, odds=odds, prob=prob, stake=stake,
          sport=sport, bet_type=bet_type, outcome=outcome,
          profit=profit, closed_at=closed_at, notes=notes,
        ))
        imported += 1
      except Exception:
        skipped += 1
        continue

    db.session.commit()
  except Exception:
    skipped += 1

  return redirect(url_for('history', imported=imported, skipped=skipped))

@app.route('/settings')
def settings_page():
  s = Setting.query.first() or Setting()
  import odds_history
  odds_stats = odds_history.get_storage_stats()
  return render_template('settings.html', settings=s, odds_stats=odds_stats)

@app.route('/save_settings', methods=['POST'])
def save_settings():
  s = Setting.query.first()
  if not s:
    s = Setting()
    db.session.add(s)
  try:
    s.bankroll = float(request.form.get('bankroll', s.bankroll))
    raw_pct = float(request.form.get('percent_bankroll', s.percent_bankroll * 100))
    s.percent_bankroll = raw_pct / 100.0
  except Exception:
    pass
  s.discord_webhook_url = request.form.get('discord_webhook_url', '').strip()
  db.session.commit()
  return redirect(url_for('settings_page'))

@app.route('/api/send-daily-picks', methods=['POST'])
def api_send_daily_picks():
    """Manually trigger the daily MLB picks Discord message using the current schedule cache."""
    s       = Setting.query.first()
    webhook = (s.discord_webhook_url or '').strip() if s else ''
    if not webhook:
        return jsonify({'ok': False, 'error': 'No webhook URL configured'})
    try:
        schedule = mlb_api.build_schedule_context()
        _send_daily_mlb_recommendation(schedule, webhook)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/test-webhook', methods=['POST'])
def api_test_webhook():
  import requests as _req
  webhook = request.form.get('webhook_url', '').strip()
  if not webhook:
    return jsonify({'ok': False, 'error': 'No URL provided'})
  try:
    r = _req.post(webhook, json={'embeds': [{
        'title':       '✅ Spooky Sports · Connection test',
        'description': 'Discord notifications are configured correctly.',
        'color':       0x4ade80,
    }]}, timeout=5)
    return jsonify({'ok': r.status_code < 300})
  except Exception as e:
    return jsonify({'ok': False, 'error': str(e)})

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
    sport = data.get('sport', '')
    bet_type = data.get('bet_type', '')
  else:
    odds = request.form.get('odds')
    prob = request.form.get('prob')
    sport = request.form.get('sport', '')
    bet_type = request.form.get('bet_type', '')

  try:
    odds = float(odds)
    prob = float(prob)
  except Exception:
    return {'recommended': 0.0}

  closed_bets = [b for b in ClosedBet.query.all() if not b.is_paper]
  recommended = compute_recommended_amount(settings.bankroll, settings.percent_bankroll, odds, prob, closed_bets, sport=sport, bet_type=bet_type)
  return {'recommended': recommended}

@app.route("/api/empirical_info", methods=["POST"])
def empirical_info():
  data = request.get_json() or {}
  sport = (data.get("sport") or "").strip()
  bet_type = (data.get("bet_type") or "").strip()
  prob = float(data.get("prob") or 0.5)
  unf_rank = data.get("unf_rank")

  ALPHA = 0.6
  TAU_DAYS = 30.0

  # If this bet is tied to today's Unified Score rank, use that rank-bucket's
  # retroactive win rate (model-derived) instead of the bettor's own closed-bet
  # history — same #1/#2-3/#4-6/#7+ buckets as the Rank Comparison table.
  if unf_rank:
    try:
      rank = int(unf_rank)
    except (TypeError, ValueError):
      rank = None
    if rank:
      label = _unified_rank_bucket(rank)
      bkt = _UNIFIED_RANK_STATS.get(sport.upper() or 'MLB', {}).get(label) if label else None
      if bkt:
        empirical = round(bkt['wr'] / 100.0, 4)
        adjusted = round(ALPHA * prob + (1.0 - ALPHA) * empirical, 4)
        return jsonify({
            "empirical": empirical,
            "adjusted": adjusted,
            "alpha": ALPHA,
            "matching_count": bkt['n'],
            "source": "unified_rank",
            "bucket_label": label,
        })

  closed_bets = [b for b in ClosedBet.query.all() if not b.is_paper]
  now = datetime.now(timezone.utc)
  weights_sum = 0.0
  weighted_wins = 0.0
  matching_count = 0

  for cb in closed_bets:
    cb_sport = (cb.sport or '').strip()
    cb_type = (cb.bet_type or '').strip()
    if sport and cb_sport.lower() != sport.lower():
      continue
    if bet_type and cb_type != bet_type:
      continue
    try:
      age_days = max(0.0, (now - (cb.closed_at or now)).total_seconds() / 86400.0)
    except Exception:
      age_days = 0.0
    weight = math.exp(-age_days / TAU_DAYS)
    weights_sum += weight
    if cb.outcome == 'win':
      weighted_wins += weight
    matching_count += 1

  empirical = None
  adjusted = prob
  if weights_sum > 0.0:
    empirical = round(weighted_wins / weights_sum, 4)
    adjusted = round(ALPHA * prob + (1.0 - ALPHA) * empirical, 4)

  return jsonify({
      "empirical": empirical,
      "adjusted": adjusted,
      "alpha": ALPHA,
      "matching_count": matching_count,
      "source": "personal_history",
  })

@app.route('/api/place-top3', methods=['POST'])
def place_top3():
  data     = request.get_json(silent=True) or {}
  bets     = data.get('bets', [])
  is_paper = bool(data.get('is_paper', False))
  if not bets or len(bets) > 5:
    return jsonify({'error': 'Invalid bets list'}), 400
  settings = Setting.query.first()
  if not settings:
    return jsonify({'error': 'No settings'}), 400
  placed = []
  try:
    for bd in bets:
      odds_dec = float(bd.get('odds', 0))
      stake    = float(bd.get('stake', 0))
      if odds_dec <= 0 or stake <= 0:
        continue
      prob      = float(bd.get('prob', 0.5))
      name      = str(bd.get('name', 'Bet'))[:100]
      sport     = str(bd.get('sport', 'MLB'))
      bet_type  = str(bd.get('bet_type', 'Moneyline'))
      side      = str(bd.get('side', ''))
      home_name = str(bd.get('home_name', ''))
      away_name = str(bd.get('away_name', ''))
      utc_raw   = str(bd.get('game_time_utc', ''))
      eventstart = None
      if utc_raw:
        try:
          eventstart = datetime.fromisoformat(utc_raw.replace('Z', '+00:00'))
          if not eventstart.tzinfo:
            eventstart = eventstart.replace(tzinfo=timezone.utc)
        except Exception:
          pass
      game_key = ''
      if home_name and away_name:
        from odds_api import _normalize
        h_norm = _normalize(home_name)
        a_norm = _normalize(away_name)
        if eventstart:
          utc_dt = eventstart if getattr(eventstart, 'tzinfo', None) else eventstart.replace(tzinfo=timezone.utc)
          game_key = f"{h_norm}_{a_norm}_{utc_dt.strftime('%Y-%m-%dT%H')}"
        else:
          game_key = f"{h_norm}_{a_norm}"
      b = OpenBet(name=name, odds=odds_dec, prob=prob,
                  stake=0.0 if is_paper else stake,
                  sport=sport, bet_type=bet_type, eventstart=eventstart,
                  game_key=game_key, bet_side=side, is_paper=is_paper)
      db.session.add(b)
      if not is_paper:
        settings.bankroll = round(settings.bankroll - stake, 2)
      placed.append({'name': name, 'stake': stake})
    db.session.commit()
    return jsonify({'ok': True, 'placed': placed, 'count': len(placed)})
  except Exception as e:
    db.session.rollback()
    return jsonify({'error': str(e)}), 500

@app.route('/add_open', methods=['POST'])
def add_open():
  is_paper = request.form.get('is_paper') == '1'
  try:
    name     = request.form.get('name', 'Bet')
    odds     = float(request.form.get('odds'))
    prob     = float(request.form.get('prob'))
    stake    = 0.0 if is_paper else float(request.form.get('stake'))
    sport    = request.form.get('sport', '')
    bet_type = request.form.get('bet_type', 'Moneyline')
  except Exception:
    return redirect(url_for('index'))
  eventstart = None
  # Prefer eventstartutc (ISO UTC string from schedule page) — unambiguous timezone.
  # Fall back to eventstart (datetime-local, browser local time, treated as naive).
  utc_raw = request.form.get('eventstartutc', '').strip()
  if utc_raw:
    try:
      eventstart = datetime.fromisoformat(utc_raw.replace('Z', '+00:00'))
      if not eventstart.tzinfo:
        eventstart = eventstart.replace(tzinfo=timezone.utc)
    except Exception:
      pass
  if eventstart is None:
    eventstart_raw = request.form.get('eventstart', '')
    if eventstart_raw:
      try:
        eventstart = datetime.strptime(eventstart_raw, "%Y-%m-%dT%H:%M")
      except Exception:
        pass
  notes     = request.form.get('notes', '')
  bet_side  = request.form.get('bet_side', '')
  home_name = request.form.get('home_name', '')
  away_name = request.form.get('away_name', '')
  game_key  = ''
  if home_name and away_name:
    from odds_api import _normalize
    h_norm = _normalize(home_name)
    a_norm = _normalize(away_name)
    # Include hour-precision UTC timestamp so same teams playing consecutive days
    # (e.g. a 3-game series) each get a distinct key that matches the odds snapshot exactly.
    if eventstart:
      utc_dt = eventstart if getattr(eventstart, 'tzinfo', None) else eventstart.replace(tzinfo=timezone.utc)
      game_key = f"{h_norm}_{a_norm}_{utc_dt.strftime('%Y-%m-%dT%H')}"
    else:
      game_key = f"{h_norm}_{a_norm}"
  b = OpenBet(name=name, odds=odds, prob=prob, stake=stake, sport=sport, bet_type=bet_type,
              eventstart=eventstart, notes=notes, game_key=game_key, bet_side=bet_side,
              is_paper=is_paper)
  db.session.add(b)
  if not is_paper:
    settings = Setting.query.first()
    if settings:
      settings.bankroll = round(settings.bankroll - stake, 2)
  db.session.commit()
  next_url = request.form.get('next', '').strip()
  if next_url and next_url.startswith('/') and not next_url.startswith('//'):
    return redirect(next_url)
  return redirect(url_for('index'))

@app.route('/edit_open/<int:bet_id>', methods=['GET', 'POST'])
def edit_open(bet_id):
  b = OpenBet.query.get_or_404(bet_id)
  if request.method == 'POST':
    try:
      old_stake = b.stake
      b.name     = request.form.get('name', b.name)
      b.odds     = float(request.form.get('odds', b.odds))
      b.prob     = float(request.form.get('prob', b.prob))
      b.stake    = float(request.form.get('stake', b.stake))
      b.sport    = request.form.get('sport', b.sport)
      b.bet_type = request.form.get('bet_type', b.bet_type)
      b.notes    = request.form.get('notes', b.notes or '')
      if not b.is_paper:
        settings = Setting.query.first()
        if settings:
          settings.bankroll = round(settings.bankroll + old_stake - b.stake, 2)
      db.session.commit()
      return redirect(url_for('index'))
    except Exception:
      pass
  return render_template('edit_open.html', b=b)

@app.route('/delete_open/<int:bet_id>', methods=['POST'])
def delete_open(bet_id):
  b = OpenBet.query.get_or_404(bet_id)
  if not b.is_paper:
    settings = Setting.query.first()
    if settings:
      settings.bankroll = round(settings.bankroll + b.stake, 2)
  db.session.delete(b)
  db.session.commit()
  return redirect(url_for('index'))

@app.route('/api/cashout/<int:bet_id>', methods=['POST'])
def api_cashout(bet_id):
  b = OpenBet.query.get_or_404(bet_id)
  data = request.get_json(silent=True) or {}
  cashout_amount = float(data.get('cashout_amount', 0))
  if cashout_amount <= 0:
    return jsonify({'error': 'Invalid cashout amount'}), 400
  profit = round(cashout_amount - b.stake, 2)
  cb = ClosedBet(
    name=b.name, odds=b.odds, prob=b.prob, stake=b.stake,
    sport=b.sport, bet_type=b.bet_type, eventstart=b.eventstart,
    outcome='cashout', profit=profit, cashout_amount=cashout_amount,
    closed_at=datetime.now(timezone.utc), notes=b.notes or '',
    closing_line=None, is_paper=b.is_paper,
  )
  db.session.add(cb)
  db.session.delete(b)
  if not b.is_paper:
    settings = Setting.query.first()
    if settings:
      settings.bankroll = round(settings.bankroll + cashout_amount, 2)
  db.session.commit()
  return jsonify({'ok': True, 'profit': profit})

@app.route('/close_open/<int:bet_id>', methods=['POST'])
def close_open(bet_id):
  b = OpenBet.query.get_or_404(bet_id)
  outcome = request.form.get('outcome', 'loss')
  if outcome not in ('win', 'loss'):
    outcome = 'loss'
  profit = b.stake * (b.odds - 1.0) if outcome == 'win' else -b.stake

  # Closing line: use manually entered value, else auto-lookup from FanDuel history
  closing_line = None
  cl_raw = request.form.get('closing_line', '').strip()
  if cl_raw:
    try:
      ca = float(cl_raw)
      closing_line = ca / 100 + 1 if ca > 0 else 100 / abs(ca) + 1
    except Exception:
      pass
  if closing_line is None and b.game_key and b.bet_side and b.eventstart:
    try:
      import odds_history as _oh
      _SPORT_KEY = {'MLB': 'baseball_mlb', 'NHL': 'icehockey_nhl', 'NFL': 'americanfootball_nfl', 'CFB': 'americanfootball_ncaaf'}
      sk = _SPORT_KEY.get((b.sport or '').upper(), 'baseball_mlb')
      es = b.eventstart if b.eventstart.tzinfo else b.eventstart.replace(tzinfo=timezone.utc)
      snap = _oh.get_closing_line(sk, b.game_key, es)
      if snap:
        ca = snap[f'{b.bet_side}_odds']
        closing_line = ca / 100.0 + 1.0 if ca > 0 else 100.0 / abs(ca) + 1.0
    except Exception:
      pass

  notes = request.form.get('notes', b.notes or '')

  cb = ClosedBet(
    name=b.name, odds=b.odds, prob=b.prob, stake=b.stake,
    sport=b.sport, bet_type=b.bet_type, eventstart=b.eventstart,
    outcome=outcome, profit=profit, closed_at=datetime.now(timezone.utc),
    notes=notes, closing_line=closing_line, is_paper=b.is_paper,
  )
  db.session.add(cb)
  db.session.delete(b)
  # Paper bets never affect bankroll — they were never deducted on placement
  if not b.is_paper and outcome == 'win':
    settings = Setting.query.first()
    if settings:
      settings.bankroll = round(settings.bankroll + b.stake + profit, 2)
  db.session.commit()
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

  eventstart = None
  eventstart_raw = request.form.get('eventstart', '')
  if eventstart_raw:
    try:
      eventstart = datetime.strptime(eventstart_raw, "%Y-%m-%dT%H:%M")
    except Exception:
      pass

  closed_at = datetime.now(timezone.utc)
  if closed_at_raw:
    try:
      # input type="datetime-local" -> "YYYY-MM-DDTHH:MM"
      closed_at = datetime.strptime(closed_at_raw, "%Y-%m-%dT%H:%M")
    except Exception:
      pass

  notes = request.form.get('notes', '')
  cb = ClosedBet(
    name=name, odds=odds, prob=prob, stake=stake,
    sport=sport, bet_type=bet_type, outcome=outcome,
    profit=profit, closed_at=closed_at, eventstart=eventstart, notes=notes,
  )
  db.session.add(cb)
  db.session.commit()
  return redirect(url_for('history'))

@app.route('/mlb')
def mlb_schedule():
  schedule = mlb_api.build_schedule_context()
  from odds_api import _normalize
  import odds_history as _oh_mv
  import re as _re

  all_open = OpenBet.query.all()

  # Primary index: game_key (set when bet was placed from schedule page)
  by_key = {}
  for bet in all_open:
    if bet.game_key:
      by_key.setdefault(bet.game_key, []).append(bet)

  # Fallback index: leading team abbreviation in the bet name
  # Matches bets like "PIT ML", "CWS ML" placed manually or pre-game_key
  by_abbr = {}
  for bet in all_open:
    if bet.game_key:
      continue  # already covered above
    m = _re.match(r'^([A-Z]{2,4})\b', bet.name or '')
    if m:
      by_abbr.setdefault(m.group(1).upper(), []).append(bet)

  for day in schedule:
    for game in day['games']:
      gk = f"{_normalize(game['home']['name'])}_{_normalize(game['away']['name'])}"
      matched = list(by_key.get(gk, []))
      # Also try the hour-precision timestamped key that add_open stores
      _utc_raw = game.get('game_time_utc', '')
      if _utc_raw:
        try:
          from datetime import datetime as _dt
          _utc = _dt.fromisoformat(_utc_raw.replace('Z', '+00:00'))
          gk_ts = f"{gk}_{_utc.strftime('%Y-%m-%dT%H')}"
          for bet in by_key.get(gk_ts, []):
            if bet not in matched:
              matched.append(bet)
        except Exception:
          pass
      # Add any fallback matches for either team abbreviation
      for abbr in (game['away']['abbr'], game['home']['abbr']):
        for bet in by_abbr.get(abbr.upper(), []):
          if bet not in matched:
            matched.append(bet)
      game['open_bets'] = matched

  _upsert_predictions(schedule, 'MLB')

  # Build candidate list.
  # Preview games: use live schedule data (model + odds) for a fresh trust score.
  # Live/Final games: use stored pre-game data from GamePrediction so trust scores
  # reflect the board state at game start and never shift after the game begins.
  all_candidates = []
  for day in (schedule or []):
    for game in day.get('games', []):
      status    = game.get('status', 'Preview')
      gdate     = (game.get('game_time_utc') or '')[:10]
      home_obj  = game.get('home') or {}
      away_obj  = game.get('away') or {}
      home_name = home_obj.get('name', '')
      away_name = away_obj.get('name', '')

      if status == 'Preview':
        model = game.get('model')
        odds  = game.get('odds')
        if not model or not odds:
          continue
        hp, ap       = model.get('home_prob', 0.5), model.get('away_prob', 0.5)
        h_odds       = odds.get('home_best', 0)
        a_odds       = odds.get('away_best', 0)
        h_imp        = odds.get('home_implied', 0.5)
        a_imp        = odds.get('away_implied', 0.5)
        factors_json = model.get('factors_json') or json.dumps(model.get('factors', []))
      else:
        # Live / Final: reconstruct from stored pre-game snapshot
        stored = GamePrediction.query.filter_by(
            sport='MLB', game_date=gdate,
            home_team=home_name, away_team=away_name,
        ).first()
        if not stored or stored.home_prob is None or not stored.home_odds:
          continue
        hp, ap = stored.home_prob, stored.away_prob or (1.0 - stored.home_prob)
        h_odds = stored.home_odds
        a_odds = stored.away_odds or 0
        vf_h, vf_a = _vig_free_implied(h_odds, a_odds)
        h_imp        = vf_h or 0.5
        a_imp        = vf_a or 0.5
        factors_json = stored.factors_json or '[]'

      if hp >= ap:
        pick_abbr   = home_obj.get('abbr', '')
        pick_prob   = hp
        mkt_implied = h_imp
        amer_odds   = h_odds
        bet_side    = 'home'
      else:
        pick_abbr   = away_obj.get('abbr', '')
        pick_prob   = ap
        mkt_implied = a_imp
        amer_odds   = a_odds
        bet_side    = 'away'
      edge = round((pick_prob - mkt_implied) * 100, 1)

      adj_prob = _consensus_adjusted_prob(pick_prob, factors_json, hp >= ap)
      adj_edge = round((adj_prob - mkt_implied) * 100, 1)

      k_b  = (amer_odds / 100.0) if amer_odds > 0 else ((-100.0 / amer_odds) if amer_odds < 0 else 0)
      k_f  = ((k_b * adj_prob - (1 - adj_prob)) / k_b) if k_b > 0 else 0
      ev_pct = round((k_b * adj_prob - (1 - adj_prob)) * 100, 1) if k_b > 0 else 0.0
      div  = abs(adj_edge)
      if div >= 15:    k_f_adj = k_f * 0.25
      elif div >= 10:  k_f_adj = k_f * 0.5
      else:            k_f_adj = k_f
      if k_f_adj >= 0.10:   k_label, k_cls = 'A', 'kelly-strong'
      elif k_f_adj >= 0.05: k_label, k_cls = 'B', 'kelly-value'
      elif k_f_adj > 0:     k_label, k_cls = 'C', 'kelly-lean'
      else:                  k_label, k_cls = '–', 'edge-neg'

      # Line movement: pick's vig-free implied prob vs opening line (Preview only)
      line_move_val = 0.5
      line_move_pct = None
      if status == 'Preview':
        open_h = odds.get('opening_home')
        open_a = odds.get('opening_away')
        if open_h and open_a:
          vf_h_open, vf_a_open = _vig_free_implied(open_h, open_a)
          if vf_h_open is not None:
            pick_open = vf_h_open if bet_side == 'home' else vf_a_open
            pick_curr = h_imp    if bet_side == 'home' else a_imp
            line_move_pct = round((pick_curr - pick_open) * 100, 2)
            line_move_val = max(0.0, min(1.0, 0.5 + line_move_pct / 10.0))

      ts_detail = _trust_score(hp, h_odds, a_odds, factors_json, detail=True)
      trust = ts_detail.get('score', 0)

      # Movement profile — computed live so it reflects partial-day snapshot
      # history as it accumulates; feeds into the Unified Score at a fixed 10%.
      game_start = None
      _utc_raw = game.get('game_time_utc', '')
      if _utc_raw:
        try:
          game_start = datetime.fromisoformat(_utc_raw.replace('Z', '+00:00'))
        except Exception:
          game_start = None
      movement_profile, movement_pct = _oh_mv.classify_movement(
          'baseball_mlb', home_name, away_name, bet_side == 'home', game_start)
      movement_val = _movement_value(movement_profile)

      unified = _unified_score(ts_detail, line_move_val, movement_val, amer_odds)
      pod_score = _pick_of_day_score(ts_detail, line_move_val, movement_val, amer_odds)

      # Both probable starters must be officially announced. Until then, the
      # model is filling SP-dependent factors (SIERA, K%, BB%, Barrel%, Whiff%)
      # with team-average fallbacks instead of the actual starter — any edge
      # computed off that is comparing a generic placeholder against a market
      # that may already know who's pitching. Deprioritize rather than hide:
      # still shown, just sorted below every confirmed-SP game regardless of
      # score, and never eligible for the higher-bar Pick of the Day slot.
      sp_confirmed = bool(home_obj.get('pitcher')) and bool(away_obj.get('pitcher'))

      pod_eligible = status == 'Preview' and sp_confirmed and _is_strong_bet(adj_prob, ts_detail)

      outcome = None
      if status == 'Final':
        ls  = (game.get('linescore') or {}).get('totals', {})
        h_r = ls.get('home', {}).get('r', 0) or 0
        a_r = ls.get('away', {}).get('r', 0) or 0
        if h_r or a_r:
          outcome = 'W' if ((hp >= ap) == (h_r > a_r)) else 'L'

      all_candidates.append({
        'away_abbr':     away_obj.get('abbr', ''),
        'home_abbr':     home_obj.get('abbr', ''),
        'home_name':     home_name,
        'away_name':     away_name,
        'game_date':     gdate,
        'pick_abbr':     pick_abbr,
        'bet_side':      bet_side,
        'edge':          edge,
        'adj_edge':      adj_edge,
        'model_prob':    round(pick_prob * 100, 1),
        'adj_prob':      round(adj_prob * 100, 1),
        'mkt_implied':   round(mkt_implied * 100, 1),
        'amer_odds':     amer_odds,
        'k_label':       k_label,
        'k_cls':         k_cls,
        'k_f_adj':       k_f_adj,
        'trust_score':   trust,
        'ts_detail':     ts_detail,
        'ev_pct':        ev_pct,
        'unified_score': unified,
        'pod_score':     pod_score,
        'pod_eligible':  pod_eligible,
        'sp_confirmed':  sp_confirmed,
        'line_move_pct': line_move_pct,
        'movement_profile': movement_profile,
        'movement_pct':     movement_pct,
        'game_time_utc': game.get('game_time_utc', ''),
        'pick_name':     home_name if bet_side == 'home' else away_name,
        'has_open_bet':  bool(game.get('open_bets')),
        'status':        status,
        'outcome':       outcome,
      })

  # Unconfirmed-SP games sort below every confirmed-SP game regardless of
  # score (see sp_confirmed comment above) — Unified Score order still applies
  # within each group.
  all_candidates.sort(key=lambda x: (not x['sp_confirmed'], -x['unified_score']))
  for i, b in enumerate(all_candidates):
    b['rank'] = i + 1

  # Pick of the Day: highest ROI-optimized score among today's Preview games
  # that also clear the strong-bet floor (_is_strong_bet) — a separate
  # objective from the main table's win-rate-optimized ranking above, so this
  # is not just "whoever's #1 in the table." None if nothing qualifies; an
  # empty slot is the correct outcome on a night with no strong, high-ROI pick.
  _pod_candidates = [b for b in all_candidates if b['pod_eligible']]
  pick_of_day = max(_pod_candidates, key=lambda b: b['pod_score']) if _pod_candidates else None

  # Persist ranks: pre-game ranks update on every load (shift as games start);
  # Live games get their rank-at-start captured once and never overwritten.
  # Final games are skipped — their outcome is already recorded by _upsert_predictions.
  try:
    for b in all_candidates:
      if b['status'] not in ('Preview', 'Live'):
        continue
      pred = GamePrediction.query.filter_by(
          sport='MLB', game_date=b['game_date'],
          home_team=b['home_name'], away_team=b['away_name'],
      ).first()
      if not pred:
        continue
      if b['status'] == 'Preview':
        pred.daily_rank = b['rank']
      elif pred.daily_rank is None and pred.home_won is None:
        pred.daily_rank = b['rank']
    db.session.commit()
  except Exception:
    db.session.rollback()

  best_bets = all_candidates

  # Stamp each game with its rec rank, trust score, and adj_prob for bet URLs
  _rec_idx = {f"{b['away_abbr']}@{b['home_abbr']}": b for b in best_bets}
  for day in (schedule or []):
    for game in day.get('games', []):
      key = f"{game['away']['abbr']}@{game['home']['abbr']}"
      if key in _rec_idx:
        b = _rec_idx[key]
        game['rec'] = {
            'rank':        b['rank'],
            'k_label':     b['k_label'],
            'k_cls':       b['k_cls'],
            'adj_edge':    b['adj_edge'],
            'trust_score': b['trust_score'],
        }
        # Inject adj_prob into the model dict so bet URLs can use it
        if game.get('model'):
          fav_home = b['bet_side'] == 'home'
          adj = b['adj_prob'] / 100.0
          game['model']['adj_home_prob'] = round(adj if fav_home else 1.0 - adj, 4)
          game['model']['adj_away_prob'] = round(adj if not fav_home else 1.0 - adj, 4)
      else:
        game['rec'] = None

  settings = Setting.query.first()
  kelly_cap = settings.percent_bankroll if settings else 0.05
  bankroll  = settings.bankroll if settings else 0.0

  # Live MLB bets — same CLV/tier annotation as the Dashboard, scoped to
  # this sport only, via the shared helpers (see _annotate_open_bets) so
  # this doesn't duplicate ~70 lines of logic a second time.
  mlb_bets       = OpenBet.query.filter_by(sport='MLB').order_by(
      OpenBet.eventstart.asc().nulls_last(), OpenBet.created_at.asc()).all()
  closing_suggestions = _annotate_open_bets(mlb_bets)
  mlb_real_open  = [b for b in mlb_bets if not b.is_paper]
  mlb_paper_open = [b for b in mlb_bets if b.is_paper]
  _tier_open_bets(mlb_real_open)
  open_stats = _open_bets_summary_stats(mlb_real_open)

  # Unit size uses ALL sports' open stake (not just MLB's) added back to
  # bankroll — Kelly sizing is bankroll-wide, not per-sport (same reasoning
  # as the persistent exposure strip in base.html), so a bet placed from
  # this page needs to size against total exposure, not just MLB's.
  _all_open_stats = compute_stats(OpenBet.query.all(), ClosedBet.query.all())
  _adj_bankroll = bankroll + _all_open_stats['open_staked']
  unit_size = max(0.01, round(_adj_bankroll * (settings.percent_bankroll if settings else 0.25), 4))

  import odds_api as _oa
  odds_last_fetch = _oa.get_last_fetch_time()
  odds_next_fetch = _oa.get_next_fetch_time()

  return render_template('mlb_schedule.html', schedule=schedule, best_bets=best_bets,
                         bankroll=bankroll, kelly_cap=kelly_cap, pick_of_day=pick_of_day,
                         open_bets=mlb_real_open, paper_bets=mlb_paper_open,
                         unit_size=unit_size, closing_suggestions=closing_suggestions,
                         open_stats=open_stats, odds_last_fetch=odds_last_fetch,
                         odds_next_fetch=odds_next_fetch,
                         heading=f"{len(mlb_real_open)} Open MLB Bet{'s' if len(mlb_real_open) != 1 else ''}",
                         sync_next='/mlb', subnav_sport='MLB')

_SPORT_META = {

  'MLB': {'emoji': '⚾', 'schedule_endpoint': 'mlb_schedule', 'baseline': '~54% (home field)'},
  'NHL': {'emoji': '🏒', 'schedule_endpoint': 'nhl_schedule', 'baseline': '~54% (home ice)'},
  'NFL': {'emoji': '🏈', 'schedule_endpoint': 'nfl_schedule', 'baseline': '~57% (home field)'},
  'CFB': {'emoji': '🎓', 'schedule_endpoint': 'cfb_schedule', 'baseline': '~59% (home field)'},

}
_MODEL_SPORTS = list(_SPORT_META.keys())

@app.route('/model')
def model_performance():
  from collections import defaultdict as _dd

  sport = request.args.get('sport', 'MLB').upper()
  if sport not in _MODEL_SPORTS:
    sport = 'MLB'

  preds = (GamePrediction.query
           .filter_by(sport=sport)
           .order_by(GamePrediction.game_date.desc(), GamePrediction.id.desc())
           .all())

  resolved = [p for p in preds if p.home_won is not None]
  n = len(resolved)

  brier = accuracy = log_loss_val = None
  if n:
    b_sum = ll_sum = correct = 0.0
    for p in resolved:
      outcome = 1.0 if p.home_won else 0.0
      prob    = max(0.001, min(0.999, p.home_prob or 0.5))
      b_sum  += (prob - outcome) ** 2
      ll_sum += outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob)
      if (prob >= 0.5) == p.home_won:
        correct += 1
    brier        = round(b_sum / n, 4)
    log_loss_val = round(-ll_sum / n, 4)
    accuracy     = round(correct / n * 100, 1)

  # Calibration: bucket by favored-team probability, check if favored team won.
  # home_prob >= 0.5  → home team is favored; favored_prob = home_prob
  # home_prob <  0.5  → away team is favored; favored_prob = 1 - home_prob
  buckets = []
  for lo, hi in [(50, 52), (52, 54), (54, 56), (56, 58), (58, 60), (60, 65), (65, 70), (70, 75)]:
    mid = (lo + hi) / 200.0
    bucket_games = []
    for p in resolved:
      hp = p.home_prob or 0.5
      fav_prob = hp if hp >= 0.5 else 1.0 - hp
      if lo / 100 <= fav_prob < hi / 100:
        fav_home = hp >= 0.5
        fav_won  = fav_home == bool(p.home_won)
        pick_odds = p.home_odds if fav_home else p.away_odds
        roi = None
        if pick_odds and p.home_odds and p.away_odds:
          try:
            o = int(pick_odds)
            profit = o / 100.0 if o > 0 else 100.0 / (-o)
            roi = profit if fav_won else -1.0
          except (TypeError, ValueError):
            pass
        bucket_games.append({'won': fav_won, 'roi': roi})
    cnt  = len(bucket_games)
    wins = sum(1 for g in bucket_games if g['won'])
    roi_vals = [g['roi'] for g in bucket_games if g['roi'] is not None]
    actual = wins / cnt if cnt else None
    if cnt == 0:
      continue
    buckets.append({
        'label':   f'{lo}–{hi}%',
        'mid':     round(mid * 100, 1),
        'count':   cnt,
        'wins':    wins,
        'actual':  round(actual * 100, 1) if actual is not None else None,
        'error':   round((actual - mid) * 100, 1) if actual is not None else None,
        'avg_roi': round(sum(roi_vals) / len(roi_vals) * 100, 1) if roi_vals else None,
    })

  _team_prefix = re.compile(r'^(?:[A-Z]{2,4}|Hm|Aw) ')
  factor_data = _dd(list)
  for p in resolved:
    outcome = 1.0 if p.home_won else 0.0
    try:
      for label, contrib in (json.loads(p.factors_json or '[]') or []):
        norm_label = _team_prefix.sub('', label)
        factor_data[norm_label].append((float(contrib), outcome))
    except Exception:
      pass

  factor_stats = []
  for label, pts in factor_data.items():
    if len(pts) < 5:
      continue
    contribs = [c for c, _ in pts]
    outcomes = [o for _, o in pts]
    nc       = len(pts)
    # Skip factors that are effectively a fixed constant (e.g. Home Field is
    # hardcoded to 0.11 every game) — a single stray historical value from
    # before some past recalibration is enough to make naive variance nonzero,
    # producing a "correlation" that's really just whether that one outlier
    # game's home team happened to win. Require real spread, not just any spread.
    most_common_frac = max(Counter(contribs).values()) / nc
    if most_common_frac >= 0.95:
      continue  # >=95% identical — not a real distribution, skip
    mc, mo   = sum(contribs) / nc, sum(outcomes) / nc
    dc       = sum((c - mc) ** 2 for c in contribs) ** 0.5
    if dc < 1e-9:
      continue  # factor always contributes 0 — no longer in model, skip
    do       = sum((o - mo) ** 2 for o in outcomes) ** 0.5
    num      = sum((c - mc) * (o - mo) for c, o in pts)
    corr     = (num / (dc * do)) if do > 0 else 0.0
    factor_stats.append({
        'label': label,
        'corr':  round(corr, 3),
        'avg':   round(mc, 3),
        'n':     nc,
    })
  factor_stats.sort(key=lambda x: abs(x['corr']), reverse=True)

  # Baseline: win rate + ROI across all resolved picks with odds (denominator for lift)
  _base_wins = _base_total = 0
  _base_roi = []
  for p in resolved:
    if not p.home_odds or not p.away_odds:
      continue
    fav_home  = (p.home_prob or 0.5) >= 0.5
    model_won = fav_home == bool(p.home_won)
    _base_total += 1
    if model_won:
      _base_wins += 1
    pick_odds = p.home_odds if fav_home else p.away_odds
    if pick_odds:
      try:
        o = int(pick_odds)
        profit = o / 100.0 if o > 0 else 100.0 / (-o)
        _base_roi.append(profit if model_won else -1.0)
      except (TypeError, ValueError):
        pass
  baseline_wr  = (_base_wins / _base_total * 100) if _base_total else None
  baseline_roi = (sum(_base_roi) / len(_base_roi) * 100) if _base_roi else None

  grade_stats = []  # kept for template compatibility; grade system removed

  # Rank performance: does pick #1 outperform #2, #3 etc.?
  rank_buckets = {}  # label -> {'won', 'roi', 'clv', 'total'}
  for p in resolved:
    r = p.daily_rank
    if r is None or not p.home_odds or not p.away_odds:
      continue
    if r <= 3:
      lbl = str(r)         # show #1, #2, #3 individually
    elif r <= 5:
      lbl = '4–5'
    else:
      lbl = '6+'
    fav_home  = (p.home_prob or 0.5) >= 0.5
    model_won = fav_home == bool(p.home_won)
    pick_odds = p.home_odds if fav_home else p.away_odds
    roi = None
    if pick_odds:
      try:
        o = int(pick_odds)
        profit = o / 100.0 if o > 0 else 100.0 / (-o)
        roi = round(profit if model_won else -1.0, 4)
      except (TypeError, ValueError):
        pass
    clv = p.pick_clv
    if clv is None and p.closing_home_odds and p.closing_away_odds:
      vf_open_h, vf_open_a   = _vig_free_implied(p.home_odds, p.away_odds)
      vf_close_h, vf_close_a = _vig_free_implied(p.closing_home_odds, p.closing_away_odds)
      if vf_open_h and vf_close_h:
        open_pick  = vf_open_h  if fav_home else vf_open_a
        close_pick = vf_close_h if fav_home else vf_close_a
        clv = round((close_pick - open_pick) * 100, 2)
    if lbl not in rank_buckets:
      rank_buckets[lbl] = {'wins': 0, 'total': 0, 'roi': [], 'clv': []}
    rank_buckets[lbl]['total'] += 1
    if model_won:
      rank_buckets[lbl]['wins'] += 1
    if roi is not None:
      rank_buckets[lbl]['roi'].append(roi)
    if clv is not None:
      rank_buckets[lbl]['clv'].append(clv)

  rank_stats = []
  for lbl in ['1', '2', '3', '4–5', '6+']:
    b = rank_buckets.get(lbl)
    if not b or b['total'] == 0:
      continue
    roi_vals = b['roi']
    clv_vals = b['clv']
    wr  = round(b['wins'] / b['total'] * 100, 1)
    avg_roi = round(sum(roi_vals) / len(roi_vals) * 100, 1) if roi_vals else None
    rank_stats.append({
        'rank':     f'#{lbl}',
        'count':    b['total'],
        'win_rate': wr,
        'lift_wr':  round(wr - baseline_wr, 1) if baseline_wr is not None else None,
        'lift_roi': round(avg_roi - baseline_roi, 1) if (avg_roi is not None and baseline_roi is not None) else None,
        'avg_roi':  avg_roi,
        'avg_clv':  round(sum(clv_vals) / len(clv_vals), 2) if clv_vals else None,
        'clv_n':    len(clv_vals),
    })

  # ── Trust score buckets ────────────────────────────────────────────────────
  _score_order = ['70+', '55–70', '42–55', '28–42', '< 28']
  _score_bkts  = {k: {'wins': 0, 'total': 0, 'roi': []} for k in _score_order}
  for p in resolved:
    if not p.home_odds or not p.away_odds:
      continue
    ts = _trust_score(p.home_prob, p.home_odds, p.away_odds, p.factors_json)
    if ts == 0:
      continue
    if   ts >= 70: slbl = '70+'
    elif ts >= 55: slbl = '55–70'
    elif ts >= 42: slbl = '42–55'
    elif ts >= 28: slbl = '28–42'
    else:          slbl = '< 28'
    fav_home  = (p.home_prob or 0.5) >= 0.5
    model_won = fav_home == bool(p.home_won)
    pick_odds = p.home_odds if fav_home else p.away_odds
    roi = None
    if pick_odds:
      try:
        o = int(pick_odds)
        profit = o / 100.0 if o > 0 else 100.0 / (-o)
        roi = profit if model_won else -1.0
      except (TypeError, ValueError):
        pass
    bkt = _score_bkts[slbl]
    bkt['total'] += 1
    if model_won:
      bkt['wins'] += 1
    if roi is not None:
      bkt['roi'].append(roi)

  score_bucket_stats = []
  for lbl in _score_order:
    bkt = _score_bkts[lbl]
    if bkt['total'] == 0:
      continue
    roi_vals = bkt['roi']
    wr  = round(bkt['wins'] / bkt['total'] * 100, 1)
    avg_roi = round(sum(roi_vals) / len(roi_vals) * 100, 1) if roi_vals else None
    score_bucket_stats.append({
        'label':    lbl,
        'count':    bkt['total'],
        'win_rate': wr,
        'avg_roi':  avg_roi,
        'lift_wr':  round(wr - baseline_wr, 1) if baseline_wr is not None else None,
        'lift_roi': round(avg_roi - baseline_roi, 1) if (avg_roi is not None and baseline_roi is not None) else None,
    })

  # ── Market agreement (model vs opening market) ──────────────────────────────
  _mkt_order = ['< -6%', '-6 to -2%', '-2 to +5%', '+5 to +10%', '> +10%']
  _mkt_bkts  = {k: {'wins': 0, 'total': 0, 'roi': []} for k in _mkt_order}
  for p in resolved:
    if not p.home_prob or not p.home_odds or not p.away_odds:
      continue
    vf_h, vf_a = _vig_free_implied(p.home_odds, p.away_odds)
    if vf_h is None:
      continue
    fav_home  = (p.home_prob or 0.5) >= 0.5
    pick_prob = p.home_prob if fav_home else 1.0 - p.home_prob
    mkt_prob  = vf_h if fav_home else vf_a
    diff = (pick_prob - mkt_prob) * 100
    if diff < -6:    mlbl = '< -6%'
    elif diff < -2:  mlbl = '-6 to -2%'
    elif diff <= 5:  mlbl = '-2 to +5%'
    elif diff <= 10: mlbl = '+5 to +10%'
    else:            mlbl = '> +10%'
    bkt = _mkt_bkts[mlbl]
    bkt['total'] += 1
    fav_home  = (p.home_prob or 0.5) >= 0.5
    model_won = fav_home == bool(p.home_won)
    if model_won:
      bkt['wins'] += 1
    pick_odds = p.home_odds if fav_home else p.away_odds
    if pick_odds:
      try:
        o = int(pick_odds)
        profit = o / 100.0 if o > 0 else 100.0 / (-o)
        bkt['roi'].append(profit if model_won else -1.0)
      except (TypeError, ValueError):
        pass

  market_agree_stats = []
  for lbl in _mkt_order:
    bkt = _mkt_bkts[lbl]
    if bkt['total'] == 0:
      continue
    roi_vals = bkt['roi']
    wr  = round(bkt['wins'] / bkt['total'] * 100, 1)
    avg_roi = round(sum(roi_vals) / len(roi_vals) * 100, 1) if roi_vals else None
    market_agree_stats.append({
        'label':    lbl,
        'count':    bkt['total'],
        'win_rate': wr,
        'avg_roi':  avg_roi,
        'lift_wr':  round(wr - baseline_wr, 1) if baseline_wr is not None else None,
    })

  # ── Factor consensus (fraction of significant factors aligning with pick) ───
  _cons_order = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%']
  _cons_bkts  = {k: {'wins': 0, 'total': 0, 'roi': []} for k in _cons_order}
  for p in resolved:
    if not p.home_odds or not p.away_odds:
      continue
    try:
      factors = json.loads(p.factors_json or '[]')
    except Exception:
      continue
    sig = [(l, c) for l, c in (factors or []) if abs(c) > 0.03]
    if not sig:
      continue
    fav_home = (p.home_prob or 0.5) >= 0.5
    total_w  = sum(abs(c) for _, c in sig)
    align_w  = sum(abs(c) for _, c in sig if (c > 0) == fav_home)
    ratio    = align_w / total_w if total_w > 0 else 0.5
    if ratio < 0.20:   clbl = '0–20%'
    elif ratio < 0.40: clbl = '20–40%'
    elif ratio < 0.60: clbl = '40–60%'
    elif ratio < 0.80: clbl = '60–80%'
    else:              clbl = '80–100%'
    bkt = _cons_bkts[clbl]
    bkt['total'] += 1
    model_won = fav_home == bool(p.home_won)
    if model_won:
      bkt['wins'] += 1
    pick_odds = p.home_odds if fav_home else p.away_odds
    if pick_odds:
      try:
        o = int(pick_odds)
        profit = o / 100.0 if o > 0 else 100.0 / (-o)
        bkt['roi'].append(profit if model_won else -1.0)
      except (TypeError, ValueError):
        pass

  consensus_stats = []
  for lbl in _cons_order:
    bkt = _cons_bkts[lbl]
    if bkt['total'] == 0:
      continue
    roi_vals = bkt['roi']
    wr  = round(bkt['wins'] / bkt['total'] * 100, 1)
    avg_roi = round(sum(roi_vals) / len(roi_vals) * 100, 1) if roi_vals else None
    consensus_stats.append({
        'label':    lbl,
        'count':    bkt['total'],
        'win_rate': wr,
        'avg_roi':  avg_roi,
        'lift_wr':  round(wr - baseline_wr, 1) if baseline_wr is not None else None,
    })

  # ── Favorite vs Underdog ────────────────────────────────────────────────────
  _fv_keys = ['Fav', 'Dog']
  _fv_bkts = {k: {'wins': 0, 'total': 0, 'roi': []} for k in _fv_keys}
  for p in resolved:
    if not p.home_odds or not p.away_odds:
      continue
    vf_h, vf_a = _vig_free_implied(p.home_odds, p.away_odds)
    if vf_h is None:
      continue
    fav_home  = (p.home_prob or 0.5) >= 0.5
    mkt_prob  = vf_h if fav_home else vf_a
    fv_key    = 'Fav' if mkt_prob >= 0.5 else 'Dog'
    model_won = fav_home == bool(p.home_won)
    pick_odds = p.home_odds if fav_home else p.away_odds
    roi = None
    if pick_odds:
      try:
        o = int(pick_odds)
        profit = o / 100.0 if o > 0 else 100.0 / (-o)
        roi = profit if model_won else -1.0
      except (TypeError, ValueError):
        pass
    bkt = _fv_bkts[fv_key]
    bkt['total'] += 1
    if model_won:
      bkt['wins'] += 1
    if roi is not None:
      bkt['roi'].append(roi)

  fav_dog_stats = []
  for fv in _fv_keys:
    bkt = _fv_bkts[fv]
    if bkt['total'] == 0:
      continue
    roi_vals = bkt['roi']
    wr  = round(bkt['wins'] / bkt['total'] * 100, 1)
    avg_roi = round(sum(roi_vals) / len(roi_vals) * 100, 1) if roi_vals else None
    fav_dog_stats.append({
        'type':     fv,
        'count':    bkt['total'],
        'win_rate': wr,
        'avg_roi':  avg_roi,
        'lift_wr':  round(wr - baseline_wr, 1) if baseline_wr is not None else None,
        'by_grade': [],
    })

  # ── Closing line distance (model vs closing market) ─────────────────────────
  _cld_order = ['< -5%', '-5 to 0%', '0 to +5%', '> +5%']
  _cld_bkts  = {k: {'wins': 0, 'total': 0, 'roi': []} for k in _cld_order}
  for p in resolved:
    if not p.home_prob or not p.home_odds or not p.away_odds:
      continue
    if not p.closing_home_odds or not p.closing_away_odds:
      continue
    vf_ch, vf_ca = _vig_free_implied(p.closing_home_odds, p.closing_away_odds)
    if vf_ch is None:
      continue
    fav_home   = (p.home_prob or 0.5) >= 0.5
    pick_prob  = p.home_prob if fav_home else 1.0 - p.home_prob
    close_prob = vf_ch if fav_home else vf_ca
    diff = (pick_prob - close_prob) * 100
    if diff < -5:   cdlbl = '< -5%'
    elif diff < 0:  cdlbl = '-5 to 0%'
    elif diff <= 5: cdlbl = '0 to +5%'
    else:           cdlbl = '> +5%'
    bkt = _cld_bkts[cdlbl]
    bkt['total'] += 1
    model_won = fav_home == bool(p.home_won)
    if model_won:
      bkt['wins'] += 1
    pick_odds = p.home_odds if fav_home else p.away_odds
    if pick_odds:
      try:
        o = int(pick_odds)
        profit = o / 100.0 if o > 0 else 100.0 / (-o)
        bkt['roi'].append(profit if model_won else -1.0)
      except (TypeError, ValueError):
        pass

  closing_dist_stats = []
  for lbl in _cld_order:
    bkt = _cld_bkts[lbl]
    if bkt['total'] == 0:
      continue
    roi_vals = bkt['roi']
    wr  = round(bkt['wins'] / bkt['total'] * 100, 1)
    avg_roi = round(sum(roi_vals) / len(roi_vals) * 100, 1) if roi_vals else None
    closing_dist_stats.append({
        'label':    lbl,
        'count':    bkt['total'],
        'win_rate': wr,
        'avg_roi':  avg_roi,
        'lift_wr':  round(wr - baseline_wr, 1) if baseline_wr is not None else None,
    })

  # ── Rank comparison: Trust Score vs Sharp Score vs Experimental ─────────────
  # Use all resolved games with odds + pick_roi. For the market signal we use:
  #   line_move: opening_home_odds → closing_home_odds (preferred going forward)
  #   pick_clv:  entry → close fallback for games without opening_home_odds stored
  # The Experimental formula requires at least one market signal; Sharp Score and
  # retroactive Trust Score run on all games with odds.
  from itertools import groupby as _igroupby
  _ranked = [p for p in resolved if p.pick_roi is not None and p.home_odds and p.away_odds]

  # Compute per-game retroactive Trust Score and market signal
  _retro_ts_map  = {}   # {id: retro_ts_rank}
  _exp_rank_map  = {}   # {id: experimental_rank}
  _sharp_rank_map = {}  # {id: sharp_score_rank}
  _unified_rank_map = {}  # {id: unified_score_rank}

  for _dt, _grp in _igroupby(
      sorted(_ranked, key=lambda x: x.game_date), key=lambda x: x.game_date
  ):
    _day = list(_grp)

    # Retroactive Trust Score rank (current formula, no stale daily_rank)
    _ts_scored = []
    for _g in _day:
      _td = _trust_score(_g.home_prob, _g.home_odds, _g.away_odds, _g.factors_json, detail=True)
      _ts_scored.append((_g.id, _td.get('score', 0), _td))
    _ts_scored.sort(key=lambda x: -x[1])
    for _r, (_gid, _, __) in enumerate(_ts_scored, 1):
      _retro_ts_map[_gid] = _r

    # Sharp Score rank: 30%E + 30%C + 20%K + 20%LM
    # line_move for resolved: opening → close (or CLV as fallback)
    _sh_scored = []
    for _g in _day:
      _td = next((t for gid, _, t in _ts_scored if gid == _g.id), {})
      lm = _line_move_pct(_g.home_prob, _g.closing_home_odds or _g.home_odds,
                          _g.closing_away_odds or _g.away_odds,
                          _g.opening_home_odds, _g.opening_away_odds)
      lm_fallback = _g.pick_clv if lm is None else lm
      lm_val = max(0.0, min(1.0, 0.5 + lm_fallback / 10.0)) if lm_fallback is not None else 0.5
      sh = _sharp_score(_td, lm_val)
      _sh_scored.append((_g.id, sh))
    _sh_scored.sort(key=lambda x: -x[1])
    for _r, (_gid, _) in enumerate(_sh_scored, 1):
      _sharp_rank_map[_gid] = _r

    # Experimental rank: requires at least one market signal
    _exp_scored = []
    for _g in _day:
      _td = next((t for gid, _, t in _ts_scored if gid == _g.id), {})
      lm = _line_move_pct(_g.home_prob, _g.closing_home_odds or _g.home_odds,
                          _g.closing_away_odds or _g.away_odds,
                          _g.opening_home_odds, _g.opening_away_odds)
      sc = _experimental_score(_td, clv_pct=_g.pick_clv, line_move_pct=lm)
      if sc > 0:
        _exp_scored.append((_g.id, sc))
    _exp_scored.sort(key=lambda x: -x[1])
    for _r, (_gid, _) in enumerate(_exp_scored, 1):
      _exp_rank_map[_gid] = _r

    # Unified rank: fitted (e, f, m) weights + fixed 10% movement — see _recompute_unified_weights
    _un_scored = []
    for _g in _day:
      _td = next((t for gid, _, t in _ts_scored if gid == _g.id), {})
      lm = _line_move_pct(_g.home_prob, _g.closing_home_odds or _g.home_odds,
                          _g.closing_away_odds or _g.away_odds,
                          _g.opening_home_odds, _g.opening_away_odds)
      lm_fallback = _g.pick_clv if lm is None else lm
      lm_val = max(0.0, min(1.0, 0.5 + lm_fallback / 10.0)) if lm_fallback is not None else 0.5
      mv_val = _movement_value(_g.movement_profile)
      _fav_home = (_g.home_prob or 0.5) >= 0.5
      _pick_odds = _g.home_odds if _fav_home else _g.away_odds
      un = _unified_score(_td, lm_val, mv_val, _pick_odds)
      _un_scored.append((_g.id, un))
    _un_scored.sort(key=lambda x: -x[1])
    for _r, (_gid, _) in enumerate(_un_scored, 1):
      _unified_rank_map[_gid] = _r

  # Games eligible for each system
  _ts_eligible  = _ranked                                   # all with odds
  _sh_eligible  = _ranked                                   # all with odds
  _exp_eligible = [g for g in _ranked if _exp_rank_map.get(g.id)]  # need signal
  _un_eligible  = _ranked                                   # all with odds

  _RANK_BUCKETS_DEF = [('#1', 1, 1), ('#2-3', 2, 3), ('#4-6', 4, 6), ('#7+', 7, 999)]

  def _rank_bucket_stats(games, rank_fn):
    rows = []
    for label, lo, hi in _RANK_BUCKETS_DEF:
      bkt = [g for g in games if rank_fn(g) is not None and lo <= rank_fn(g) <= hi]
      if not bkt:
        rows.append({'label': label, 'n': 0, 'wr': None, 'roi': None})
        continue
      _fh = {g.id: (g.home_prob or 0.5) >= 0.5 for g in bkt}
      wins = sum(1 for g in bkt if (_fh[g.id] == bool(g.home_won)))
      roi_vals = [g.pick_roi for g in bkt if g.pick_roi is not None]
      rows.append({
        'label': label,
        'n':     len(bkt),
        'wr':    round(100 * wins / len(bkt), 1),
        'roi':   round(100 * sum(roi_vals) / len(roi_vals), 1) if roi_vals else None,
      })
    return rows

  rank_cmp_ts      = _rank_bucket_stats(_ts_eligible,  lambda g: _retro_ts_map.get(g.id))
  rank_cmp_sharp   = _rank_bucket_stats(_sh_eligible,  lambda g: _sharp_rank_map.get(g.id))
  rank_cmp_exp     = _rank_bucket_stats(_exp_eligible, lambda g: _exp_rank_map.get(g.id))
  rank_cmp_unified = _rank_bucket_stats(_un_eligible,  lambda g: _unified_rank_map.get(g.id))
  rank_comparison = [{'label': b['label'], 'ts': b, 'sharp': s, 'exp': e, 'unified': u}
                     for b, s, e, u in zip(rank_cmp_ts, rank_cmp_sharp, rank_cmp_exp, rank_cmp_unified)]
  rank_comparison_n    = len(_ts_eligible)
  rank_comparison_n_exp = len(_exp_eligible)
  unified_weights = dict(_UNIFIED_WEIGHTS)
  pick_of_day_weights = dict(_PICK_OF_DAY_WEIGHTS)

  # ── Movement Profile breakdown — monitoring sample size before this feeds
  # into the Unified Score (currently 0.5 neutral until each bucket hits MIN_N) ─
  _mv_games = [p for p in resolved if p.movement_profile]
  movement_stats = []
  for label in _MOVEMENT_PROFILES:
    bkt = [p for p in _mv_games if p.movement_profile == label]
    if not bkt:
      movement_stats.append({'label': label, 'n': 0, 'wr': None, 'roi': None, 'roi_n': 0})
      continue
    wins = sum(1 for p in bkt if ((p.home_prob or 0.5) >= 0.5) == bool(p.home_won))
    roi_vals = [p.pick_roi for p in bkt if p.pick_roi is not None]
    movement_stats.append({
      'label':  label,
      'n':      len(bkt),
      'wr':     round(100 * wins / len(bkt), 1),
      'roi':    round(100 * sum(roi_vals) / len(roi_vals), 1) if roi_vals else None,
      'roi_n':  len(roi_vals),
    })
  movement_weights = dict(_MOVEMENT_WEIGHTS)
  movement_n_total = len(_mv_games)

  # ── Firing Efficiency Analysis ───────────────────────────────────────────────
  # Retroactively rank all resolved games with odds by Trust Score within each day.
  # "Recommended N" = top-N picks per day by Trust Score.
  _fire_games = []
  for p in resolved:
    if not p.home_odds or not p.away_odds or not p.home_prob:
      continue
    fav_home  = (p.home_prob or 0.5) >= 0.5
    model_won = fav_home == bool(p.home_won)
    pick_odds = p.home_odds if fav_home else p.away_odds
    roi = None
    if pick_odds:
      try:
        o = int(pick_odds)
        profit = o / 100.0 if o > 0 else 100.0 / (-o)
        roi = profit if model_won else -1.0
      except (TypeError, ValueError):
        pass
    td   = _trust_score(p.home_prob, p.home_odds, p.away_odds, p.factors_json, detail=True)
    ts   = td.get('score', 0)
    lm   = _line_move_pct(p.home_prob, p.closing_home_odds or p.home_odds,
                          p.closing_away_odds or p.away_odds,
                          p.opening_home_odds, p.opening_away_odds)
    signal_pct = lm if lm is not None else p.pick_clv
    m_val = max(0.0, min(1.0, 0.5 + signal_pct / 10.0)) if signal_pct is not None else 0.5
    mv_val = _movement_value(p.movement_profile)
    unf  = _unified_score(td, m_val, mv_val, pick_odds)
    edge = _pick_edge_pct(p.home_prob, p.home_odds, p.away_odds)
    _fire_games.append({
        'date': p.game_date,
        'ts':   ts,
        'unf':  unf,
        'won':  model_won,
        'roi':  roi,
        'edge': edge,
    })

  # Rank within each day (highest TS = rank 1), stable sort preserves order on ties
  _fire_games.sort(key=lambda x: (x['date'], -x['ts']))
  _fdate = None; _frank = 0
  for g in _fire_games:
    if g['date'] != _fdate:
      _fdate = g['date']; _frank = 1
    else:
      _frank += 1
    g['rec_rank'] = _frank

  # Separate rank within each day by Unified Score (highest = rank 1)
  from itertools import groupby as _fire_groupby
  for _dt, _grp in _fire_groupby(
      sorted(_fire_games, key=lambda x: (x['date'], -x['unf'])), key=lambda x: x['date']
  ):
    for _urank, g in enumerate(_grp, 1):
      g['unf_rank'] = _urank

  def _fire_row(games, label, n_pick=None):
    if not games:
      return None
    total = len(games)
    wins  = sum(1 for g in games if g['won'])
    prof  = sum(1 for g in games if g['roi'] is not None and g['roi'] > 0)
    roi_v = [g['roi']  for g in games if g['roi']  is not None]
    edg_v = [g['edge'] for g in games if g['edge'] is not None]
    roi_avg = round(sum(roi_v) / len(roi_v) * 100, 1) if roi_v else None
    return {
        'label':      label,
        'n_pick':     n_pick,
        'count':      total,
        'win_rate':   round(wins  / total * 100, 1),
        'profitable': round(prof  / total * 100, 1),
        'avg_roi':    roi_avg,
        'avg_edge':   round(sum(edg_v) / len(edg_v), 1) if edg_v else None,
    }

  _all_row = _fire_row(_fire_games, 'All picks')
  firing_stats = []
  for _n, _lbl in [(1,'Top 1'),(2,'Top 2'),(3,'Top 3'),(5,'Top 5')]:
    row = _fire_row([g for g in _fire_games if g['rec_rank'] <= _n], _lbl, _n)
    if row:
      row['lift_roi'] = round((row['avg_roi'] or 0) - (_all_row['avg_roi'] or 0), 1) if _all_row else None
      row['lift_wr']  = round(row['win_rate'] - _all_row['win_rate'], 1) if _all_row else None
      row['lift_eff'] = round(row['profitable'] - _all_row['profitable'], 1) if _all_row else None
      firing_stats.append(row)
  if _all_row:
    _all_row['lift_roi'] = None; _all_row['lift_wr'] = None; _all_row['lift_eff'] = None
    firing_stats.append(_all_row)

  # Same Top-N breakdown, ranked by Unified Score instead of Trust Score
  _all_row_unf = _fire_row(_fire_games, 'All picks')
  firing_stats_unified = []
  for _n, _lbl in [(1,'Top 1'),(2,'Top 2'),(3,'Top 3'),(5,'Top 5')]:
    row = _fire_row([g for g in _fire_games if g['unf_rank'] <= _n], _lbl, _n)
    if row:
      row['lift_roi'] = round((row['avg_roi'] or 0) - (_all_row_unf['avg_roi'] or 0), 1) if _all_row_unf else None
      row['lift_wr']  = round(row['win_rate'] - _all_row_unf['win_rate'], 1) if _all_row_unf else None
      row['lift_eff'] = round(row['profitable'] - _all_row_unf['profitable'], 1) if _all_row_unf else None
      firing_stats_unified.append(row)
  if _all_row_unf:
    _all_row_unf['lift_roi'] = None; _all_row_unf['lift_wr'] = None; _all_row_unf['lift_eff'] = None
    firing_stats_unified.append(_all_row_unf)

  # ROI by Trust Score bucket: recommended (top 3) vs all games with odds
  _fts_order = ['70+', '55–70', '42–55', '28–42', '< 28']
  def _fts_bkt(ts):
    if   ts >= 70: return '70+'
    elif ts >= 55: return '55–70'
    elif ts >= 42: return '42–55'
    elif ts >= 28: return '28–42'
    else:          return '< 28'
  _fbkt_all = {k: [] for k in _fts_order}
  _fbkt_rec = {k: [] for k in _fts_order}
  for g in _fire_games:
    bk = _fts_bkt(g['ts'])
    if g['roi'] is not None:
      _fbkt_all[bk].append(g['roi'])
      if g['rec_rank'] <= 3:
        _fbkt_rec[bk].append(g['roi'])
  firing_bucket_stats = []
  for lbl in _fts_order:
    av = _fbkt_all[lbl]; rv = _fbkt_rec[lbl]
    firing_bucket_stats.append({
        'label':   lbl,
        'all_n':   len(av),
        'all_roi': round(sum(av)/len(av)*100,1) if av else None,
        'rec_n':   len(rv),
        'rec_roi': round(sum(rv)/len(rv)*100,1) if rv else None,
    })

  _fire_days = len(set(g['date'] for g in _fire_games))

  # ── Enriched recent predictions (resolved only, excluding today) ────────────
  from datetime import timezone as _tz
  _today_et = datetime.now(timezone(timedelta(hours=-4))).strftime('%Y-%m-%d')  # ET (DST)
  _team_pfx = re.compile(r'^(?:Hm|Aw) ')
  enriched_preds = []
  for p in resolved[:40]:
    if p.game_date == _today_et:
      continue
    ts = _trust_score(p.home_prob, p.home_odds, p.away_odds, p.factors_json)
    fav_home = (p.home_prob or 0.5) >= 0.5
    actual_home_won = bool(p.home_won)
    pick_won = (fav_home == actual_home_won)
    pick_team = p.home_team if fav_home else p.away_team
    opp_team  = p.away_team if fav_home else p.home_team
    pick_prob = (p.home_prob if fav_home else 1.0 - p.home_prob) or 0.5
    _edge = _pick_edge_pct(p.home_prob, p.home_odds, p.away_odds)

    factors_display = []
    try:
      for label, contrib in (json.loads(p.factors_json or '[]') or []):
        if abs(contrib) < 0.01:
          continue
        points_home = contrib > 0
        clean = _team_pfx.sub('', label)
        helped = (points_home == actual_home_won)
        factors_display.append({
            'label':       clean,
            'contrib':     round(contrib, 3),
            'points_home': points_home,
            'helped':      helped,
        })
      factors_display.sort(key=lambda x: abs(x['contrib']), reverse=True)
    except Exception:
      pass

    enriched_preds.append({
        'game_date':  p.game_date,
        'away_team':  p.away_team,
        'home_team':  p.home_team,
        'home_prob':  round(p.home_prob * 100, 1) if p.home_prob else None,
        'away_prob':  round((1.0 - p.home_prob) * 100, 1) if p.home_prob else None,
        'pick_team':  pick_team,
        'opp_team':   opp_team,
        'pick_prob':  round(pick_prob * 100, 1),
        'pick_won':   pick_won,
        'actual_winner': p.home_team if actual_home_won else p.away_team,
        'home_score': p.home_score,
        'away_score': p.away_score,
        'trust_score': ts,
        'edge':       _edge,
        'daily_rank': p.daily_rank,
        'exp_rank':   _exp_rank_map.get(p.id),
        'factors':    factors_display,
    })

  # ── Group recent predictions by date for the drill-down view ────────────
  pred_days = []
  _day_map = {}
  for p in enriched_preds:
    d = p['game_date']
    if d not in _day_map:
      _day_map[d] = {'date': d, 'games': [], 'wins': 0, 'total': 0}
      pred_days.append(_day_map[d])
    grp = _day_map[d]
    grp['games'].append(p)
    grp['total'] += 1
    if p['pick_won']:
      grp['wins'] += 1
  for grp in pred_days:
    grp['losses'] = grp['total'] - grp['wins']
    grp['win_pct'] = round(grp['wins'] / grp['total'] * 100) if grp['total'] else 0
    try:
      _dt_obj = datetime.strptime(grp['date'], '%Y-%m-%d')
      grp['date_label'] = f"{_dt_obj.strftime('%a')} {_dt_obj.month}/{_dt_obj.day}"
    except ValueError:
      grp['date_label'] = grp['date']

  return render_template('model_performance.html',
      preds=preds[:40],
      total=len(preds),
      resolved=n,
      brier=brier,
      accuracy=accuracy,
      log_loss=log_loss_val,
      buckets=buckets,
      factor_stats=factor_stats,
      grade_stats=grade_stats,
      rank_stats=rank_stats,
      score_bucket_stats=score_bucket_stats,
      market_agree_stats=market_agree_stats,
      consensus_stats=consensus_stats,
      fav_dog_stats=fav_dog_stats,
      closing_dist_stats=closing_dist_stats,
      baseline_wr=round(baseline_wr, 1) if baseline_wr is not None else None,
      baseline_roi=round(baseline_roi, 1) if baseline_roi is not None else None,
      sport=sport,
      subnav_sport=sport,
      sports=_MODEL_SPORTS,
      sport_meta=_SPORT_META[sport],
      enriched_preds=enriched_preds,
      pred_days=pred_days,
      rank_comparison=rank_comparison,
      rank_comparison_n=rank_comparison_n,
      rank_comparison_n_exp=rank_comparison_n_exp,
      unified_weights=unified_weights,
      pick_of_day_weights=pick_of_day_weights,
      movement_stats=movement_stats,
      movement_weights=movement_weights,
      movement_n_total=movement_n_total,
      firing_stats=firing_stats,
      firing_stats_unified=firing_stats_unified,
      firing_bucket_stats=firing_bucket_stats,
      fire_days=_fire_days,
      fire_total=len(_fire_games),
  )

@app.route('/nhl')
def nhl_schedule():
  schedule = nhl_api.build_schedule_context()
  _upsert_predictions(schedule, 'NHL')
  return render_template('nhl_schedule.html', schedule=schedule, subnav_sport='NHL')

@app.route('/nfl')
def nfl_schedule():
  week = request.args.get('week', type=int)
  week_ctx = nfl_api.build_week_schedule_context(week)
  _upsert_predictions(week_ctx['days'], 'NFL')
  _match_open_bets_to_games(week_ctx['days'], sport='NFL')

  settings = Setting.query.first()
  bankroll = settings.bankroll if settings else 0.0

  nfl_bets       = OpenBet.query.filter_by(sport='NFL').order_by(
      OpenBet.eventstart.asc().nulls_last(), OpenBet.created_at.asc()).all()
  closing_suggestions = _annotate_open_bets(nfl_bets)
  nfl_real_open  = [b for b in nfl_bets if not b.is_paper]
  nfl_paper_open = [b for b in nfl_bets if b.is_paper]
  _tier_open_bets(nfl_real_open)
  open_stats = _open_bets_summary_stats(nfl_real_open)

  _all_open_stats = compute_stats(OpenBet.query.all(), ClosedBet.query.all())
  _adj_bankroll = bankroll + _all_open_stats['open_staked']
  unit_size = max(0.01, round(_adj_bankroll * (settings.percent_bankroll if settings else 0.25), 4))

  import odds_api as _oa
  odds_last_fetch = _oa.get_last_fetch_time()
  odds_next_fetch = _oa.get_next_fetch_time()

  return render_template('nfl_schedule.html', week_ctx=week_ctx, subnav_sport='NFL',
                         open_bets=nfl_real_open, paper_bets=nfl_paper_open,
                         unit_size=unit_size, closing_suggestions=closing_suggestions,
                         open_stats=open_stats, odds_last_fetch=odds_last_fetch,
                         odds_next_fetch=odds_next_fetch,
                         heading=f"{len(nfl_real_open)} Open NFL Bet{'s' if len(nfl_real_open) != 1 else ''}",
                         sync_next='/nfl')

@app.route('/cfb')
def cfb_schedule():
  from urllib.parse import urlencode as _urlencode

  week = request.args.get('week', type=int)
  week_ctx = cfb_api.build_week_schedule_context(week)
  _upsert_predictions(week_ctx['days'], 'CFB')
  _match_open_bets_to_games(week_ctx['days'], sport='CFB')

  # Today's Recommendations: a lightweight edge-ranked list, not MLB's full
  # trust-score/unified-score system — CFB has no season of backtested
  # GamePrediction history to calibrate that against yet. Ranks by model
  # edge vs. the vig-free market price and requires a positive Kelly
  # fraction, which naturally pushes lopsided ranked-vs-cupcake games (where
  # the market is already priced near-certain and there's no real edge left)
  # to the bottom instead of needing an explicit odds cutoff. Scoped to the
  # displayed week (not "today") to match the week-view page below.
  conferences = set()
  recommended = []
  for day in week_ctx['days']:
    for game in day.get('games', []):
      home_t, away_t = game.get('home') or {}, game.get('away') or {}
      if home_t.get('conference'):
        conferences.add(home_t['conference'])
      if away_t.get('conference'):
        conferences.add(away_t['conference'])

      if game.get('status') != 'Preview':
        continue
      model, odds = game.get('model'), game.get('odds')
      if not model or not odds:
        continue
      hp, ap = model.get('home_prob', 0.5), model.get('away_prob', 0.5)
      h_imp, a_imp = odds.get('home_implied'), odds.get('away_implied')
      if h_imp is None or a_imp is None:
        continue

      if (hp - h_imp) >= (ap - a_imp):
        pick_side, pick_team, pick_prob, mkt_implied, amer_odds = 'home', home_t, hp, h_imp, odds.get('home_best')
      else:
        pick_side, pick_team, pick_prob, mkt_implied, amer_odds = 'away', away_t, ap, a_imp, odds.get('away_best')
      if amer_odds is None:
        continue

      k_b = (amer_odds / 100.0) if amer_odds > 0 else (-100.0 / amer_odds if amer_odds < 0 else 0)
      k_f = ((k_b * pick_prob - (1 - pick_prob)) / k_b) if k_b > 0 else 0
      if k_f <= 0:
        continue
      ev_pct = round((k_b * pick_prob - (1 - pick_prob)) * 100, 1)
      edge = round((pick_prob - mkt_implied) * 100, 1)

      if k_f >= 0.10:   k_label, k_cls = 'Strong', 'edge-pos'
      elif k_f >= 0.05: k_label, k_cls = 'Value',  'edge-pos'
      else:             k_label, k_cls = 'Lean',   'edge-neutral'

      qs = _urlencode({
          'name':          f"{pick_team.get('abbrev', '')} ML",
          'sport':         'CFB',
          'eventstartutc': game.get('game_time_utc', ''),
          'odds':          amer_odds,
          'implied':       mkt_implied,
          'model_prob':    pick_prob,
          'home_name':     home_t.get('name', ''),
          'away_name':     away_t.get('name', ''),
          'bet_side':      pick_side,
      })
      recommended.append({
          'game':         game,
          'pick_side':    pick_side,
          'pick_abbr':    pick_team.get('abbrev', ''),
          'pick_rank':    pick_team.get('rank_display'),
          'amer_odds':    amer_odds,
          'edge':         edge,
          'ev_pct':       ev_pct,
          'k_label':      k_label,
          'k_cls':        k_cls,
          'bet_url':      f"{url_for('new_bet')}?{qs}",
      })

  recommended.sort(key=lambda r: -r['edge'])
  recommended = recommended[:15]

  settings = Setting.query.first()
  bankroll = settings.bankroll if settings else 0.0

  cfb_bets       = OpenBet.query.filter_by(sport='CFB').order_by(
      OpenBet.eventstart.asc().nulls_last(), OpenBet.created_at.asc()).all()
  closing_suggestions = _annotate_open_bets(cfb_bets)
  cfb_real_open  = [b for b in cfb_bets if not b.is_paper]
  cfb_paper_open = [b for b in cfb_bets if b.is_paper]
  _tier_open_bets(cfb_real_open)
  open_stats = _open_bets_summary_stats(cfb_real_open)

  _all_open_stats = compute_stats(OpenBet.query.all(), ClosedBet.query.all())
  _adj_bankroll = bankroll + _all_open_stats['open_staked']
  unit_size = max(0.01, round(_adj_bankroll * (settings.percent_bankroll if settings else 0.25), 4))

  import odds_api as _oa
  odds_last_fetch = _oa.get_last_fetch_time()
  odds_next_fetch = _oa.get_next_fetch_time()

  return render_template('cfb_schedule.html', week_ctx=week_ctx, subnav_sport='CFB',
                         recommended=recommended, conferences=sorted(conferences),
                         open_bets=cfb_real_open, paper_bets=cfb_paper_open,
                         unit_size=unit_size, closing_suggestions=closing_suggestions,
                         open_stats=open_stats, odds_last_fetch=odds_last_fetch,
                         odds_next_fetch=odds_next_fetch,
                         heading=f"{len(cfb_real_open)} Open CFB Bet{'s' if len(cfb_real_open) != 1 else ''}",
                         sync_next='/cfb')

@app.route('/api/refresh-stats/stream')
def api_refresh_stats_stream():
    """SSE stream: clears the MLB stats cache, fetches every data source in order,
    and reports each step to the browser in real time."""
    from flask import stream_with_context, Response as _Resp
    import mlb_api as _mlb
    import statcast_api as _sc
    import fangraphs_api as _fg
    import json as _json

    def _sse(step, status):
        return f"data: {_json.dumps({'step': step, 'status': status})}\n\n"

    def generate():
        try:
            yield _sse('Clearing stats cache', 'running')
            _mlb._cache.clear()
            yield _sse('Stats cache cleared', 'done')

            yield _sse('Fetching today\'s MLB schedule', 'running')
            _mlb._get_schedule_raw()
            yield _sse('Schedule loaded', 'done')

            yield _sse('Fetching recent team data & standings', 'running')
            _mlb._get_recent_data()
            _mlb._get_team_era_map()
            _mlb._get_standings_splits()
            yield _sse('Team data & standings loaded', 'done')

            yield _sse('Fetching Statcast metrics', 'running')
            _sc.get_pitcher_statcast()
            _sc.get_team_statcast()
            _sc.get_team_batting_splits()
            try:
                _sc.get_pitcher_metrics()
            except Exception:
                pass
            yield _sse('Statcast metrics loaded', 'done')

            yield _sse('Fetching FanGraphs pitcher data', 'running')
            try:
                _fg.get_pitcher_xfip()
                yield _sse('FanGraphs data loaded', 'done')
            except Exception:
                yield _sse('FanGraphs unavailable — skipped', 'done')

            yield _sse('Building game models', 'running')
            schedule = _mlb.build_schedule_context()
            yield _sse('Game models computed', 'done')

            yield _sse('Saving predictions to database', 'running')
            with app.app_context():
                _upsert_predictions(schedule, 'MLB')
                _recompute_trust_weights('MLB')
                _recompute_team_bias('MLB')
                _recompute_unified_weights('MLB')
                _recompute_pick_of_day_weights('MLB')
                _recompute_unified_rank_stats('MLB')
                _recompute_movement_profiles('MLB')
                _recompute_movement_weights('MLB')
            yield _sse('Database updated', 'done')

            yield _sse('', 'complete')
        except Exception as exc:
            yield _sse(f'Error: {exc}', 'error')

    return _Resp(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/refresh-odds', methods=['POST'])
def api_refresh_odds():
    """Quick POST used by the sync button (Dashboard and sport pages) — clears
    odds cache and redirects back to wherever the button was clicked from."""
    import odds_api as _oa
    _oa._cache.clear()
    next_url = request.form.get('next', '').strip()
    if next_url and next_url.startswith('/') and not next_url.startswith('//'):
        return redirect(next_url)
    return redirect(url_for('index'))


@app.route('/api/refresh-odds/stream')
def api_refresh_odds_stream():
    """SSE stream: clears the odds cache, re-fetches from The Odds API,
    rebuilds recommendations, and reports each step in real time."""
    from flask import stream_with_context, Response as _Resp
    import mlb_api as _mlb
    import odds_api as _oa
    import json as _json, os as _os

    def _sse(step, status):
        return f"data: {_json.dumps({'step': step, 'status': status})}\n\n"

    def generate():
        try:
            yield _sse('Clearing odds cache', 'running')
            for k in list(_oa._cache.keys()):
                if k.startswith('odds_'):
                    del _oa._cache[k]
            try:
                fc = _oa._fc_load()
                for k in list(fc.keys()):
                    if k.startswith('odds_'):
                        fc[k]['ts'] = 0
                _os.makedirs(_os.path.dirname(_oa._CACHE_FILE), exist_ok=True)
                with open(_oa._CACHE_FILE, 'w') as f:
                    _json.dump(fc, f)
            except Exception:
                pass
            yield _sse('Odds cache cleared', 'done')

            yield _sse('Fetching odds from The Odds API', 'running')
            _oa.get_odds_map('mlb')
            yield _sse('Odds fetched', 'done')

            yield _sse('Rebuilding game recommendations', 'running')
            schedule = _mlb.build_schedule_context()
            with app.app_context():
                _upsert_predictions(schedule, 'MLB')
            yield _sse('Recommendations updated', 'done')

            yield _sse('', 'complete')
        except Exception as exc:
            yield _sse(f'Error: {exc}', 'error')

    return _Resp(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/backfill/stream')
def api_backfill_stream():
    """SSE stream: runs the 2026 backfill inline and reports each phase in real time."""
    from flask import stream_with_context, Response as _Resp
    import json as _json

    def _sse(step, status):
        return f"data: {_json.dumps({'step': step, 'status': status})}\n\n"

    def generate():
        try:
            import backfill_2026 as _bf
            from bootstrap_2024 import (
                fetch_team_pitching, fetch_team_batting,
                fetch_team_xwoba, fetch_team_abbrevs,
                fetch_team_xfip,
                build_fatigue_map,
            )

            yield _sse('Fetching completed 2026 games from MLB API', 'running')
            games = _bf.fetch_completed_games(2026)
            yield _sse(f'Fetched {len(games)} completed games', 'done')

            yield _sse('Fetching team pitching stats', 'running')
            team_pitch = fetch_team_pitching(2026)
            yield _sse(f'Team pitching loaded ({len(team_pitch)} teams)', 'done')

            yield _sse('Fetching team batting stats', 'running')
            team_bat = fetch_team_batting(2026)
            yield _sse(f'Team batting loaded ({len(team_bat)} teams)', 'done')

            yield _sse('Fetching xwOBA & advanced metrics', 'running')
            team_xwoba     = fetch_team_xwoba(2026)
            team_abbrevs   = fetch_team_abbrevs()
            xfip_by_abbrev = fetch_team_xfip(2026)
            xwoba_by_id = {tid: team_xwoba[abbr] for tid, abbr in team_abbrevs.items() if abbr in team_xwoba}
            # Translate through _FG_ABBREV_ALIAS — 7 teams (sd/sf/tb/cws/az/kc/wsh) use a
            # different abbreviation on FanGraphs than MLB Stats API, so a plain lookup
            # silently drops xFIP/SIERA/K%/BB%/SwStr% for those teams' games.
            xfip_by_id = {}
            for tid, abbr in team_abbrevs.items():
                fg_abbrev = _bf._FG_ABBREV_ALIAS.get(abbr, abbr)
                if fg_abbrev in xfip_by_abbrev:
                    xfip_by_id[tid] = xfip_by_abbrev[fg_abbrev]
            degraded = len(xfip_by_id) == 0
            yield _sse(f'xwOBA: {len(xwoba_by_id)} teams · xFIP: {len(xfip_by_id)} teams', 'done')
            if degraded:
                yield _sse('! FanGraphs fetch returned no data — existing predictions will keep their stored prob/factors', 'done')

            yield _sse('Building fatigue map', 'running')
            fatigue_map = build_fatigue_map(games)
            yield _sse('Fatigue map built', 'done')

            yield _sse(f'Scoring & writing {len(games)} predictions to database', 'running')
            import mlb_model as _mlb_model
            inserted = updated = skipped_err = 0
            from datetime import datetime as _dt, timezone as _tz
            now = _dt.now(_tz.utc)

            with app.app_context():
                for g in games:
                    try:
                        pred = GamePrediction.query.filter_by(
                            sport='MLB', game_date=g['game_date'],
                            home_team=g['home_name'], away_team=g['away_name'],
                        ).first()
                        is_new = pred is None
                        if is_new:
                            pred = GamePrediction(
                                sport='MLB', game_date=g['game_date'],
                                game_time_utc=g['game_time'],
                                home_team=g['home_name'], away_team=g['away_name'],
                            )
                            db.session.add(pred)
                            inserted += 1
                        else:
                            updated += 1

                        # Only recompute prob/factors when inputs are trustworthy — see
                        # the matching guard in backfill_2026.py's run() for why.
                        if is_new or not degraded:
                            fat_key = (g['game_date'], g['home_id'], g['away_id'])
                            home_ctx = _bf.build_team_dict(
                                g['home_id'], g['home_name'], team_pitch, team_bat, xwoba_by_id,
                                fatigue_map.get(fat_key, {}).get('home'), xfip_by_id)
                            away_ctx = _bf.build_team_dict(
                                g['away_id'], g['away_name'], team_pitch, team_bat, xwoba_by_id,
                                fatigue_map.get(fat_key, {}).get('away'), xfip_by_id)
                            result = _mlb_model.predict(home_ctx, away_ctx, game_time_utc=g['game_time'])
                            pred.home_prob    = result['home_prob']
                            pred.away_prob    = result['away_prob']
                            pred.factors_json = _json.dumps(result['factors'])

                        pred.home_won      = g['home_won']
                        pred.home_score    = g['home_score']
                        pred.away_score    = g['away_score']
                        pred.outcome_set_at = now
                        # Keep pick_roi in sync with whichever home_prob is now
                        # authoritative (freshly computed, or preserved above).
                        _fav = (pred.home_prob or 0.5) >= 0.5
                        _po  = pred.home_odds if _fav else pred.away_odds
                        if _po:
                            try:
                                _o = int(_po)
                                _pft = _o / 100.0 if _o > 0 else 100.0 / (-_o)
                                pred.pick_roi = round(_pft if (_fav == g['home_won']) else -1.0, 4)
                            except (TypeError, ValueError):
                                pass
                    except Exception as e:
                        skipped_err += 1
                db.session.commit()

            yield _sse(
                f'Database updated — {inserted} inserted, {updated} re-scored'
                + (f', {skipped_err} errors' if skipped_err else ''),
                'done'
            )

            yield _sse('Refitting Platt scaler & trust weights', 'running')
            with app.app_context():
                _refit_mlb_platt()
                _recompute_trust_weights('MLB')
                _recompute_team_bias('MLB')
                _recompute_unified_weights('MLB')
                _recompute_pick_of_day_weights('MLB')
                _recompute_unified_rank_stats('MLB')
                _recompute_movement_profiles('MLB')
                _recompute_movement_weights('MLB')
                _recompute_consensus_calibration('MLB')
            yield _sse('Model calibration updated', 'done')

            yield _sse('', 'complete')
        except Exception as exc:
            yield _sse(f'Error: {exc}', 'error')

    return _Resp(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/repair/pick-roi', methods=['POST'])
def api_repair_pick_roi():
    """Recompute pick_roi for all resolved games where it's wrong or missing."""
    from flask import jsonify
    repaired = skipped = 0
    preds = GamePrediction.query.filter(
        GamePrediction.home_won.isnot(None),
        GamePrediction.home_odds.isnot(None),
        GamePrediction.away_odds.isnot(None),
        GamePrediction.home_prob.isnot(None),
    ).all()
    for pred in preds:
        fav_home  = pred.home_prob >= 0.5
        pick_odds = pred.home_odds if fav_home else pred.away_odds
        try:
            o = int(pick_odds)
            profit = o / 100.0 if o > 0 else 100.0 / (-o)
            model_won  = fav_home == bool(pred.home_won)
            correct    = round(profit if model_won else -1.0, 4)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if pred.pick_roi is None or abs(pred.pick_roi - correct) > 0.0001:
            pred.pick_roi = correct
            repaired += 1
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return jsonify({'error': 'commit failed'}), 500
    return jsonify({'repaired': repaired, 'skipped': skipped, 'total': len(preds)})


@app.route('/api/live-scores')
def api_live_scores():
    """
    Returns {bet_id: score_dict} for all open bets that have a live or final score today.
    Called by the dashboard every 2 minutes to update score badges.
    """
    from odds_api import _normalize
    import mlb_api as _mlb, nhl_api as _nhl, nfl_api as _nfl, cfb_api as _cfb
    from flask import jsonify

    open_bets = OpenBet.query.filter(OpenBet.game_key != '').all()
    if not open_bets:
        return jsonify({})

    sports_needed = {(b.sport or '').upper() for b in open_bets if b.game_key}
    score_map = {}

    if 'MLB' in sports_needed:
        try:
            score_map.update(_mlb.get_live_scores())
        except Exception:
            pass
    if 'NHL' in sports_needed:
        try:
            score_map.update(_nhl.get_live_scores())
        except Exception:
            pass
    if 'NFL' in sports_needed:
        try:
            score_map.update(_nfl.get_live_scores())
        except Exception:
            pass
    if 'CFB' in sports_needed:
        try:
            score_map.update(_cfb.get_live_scores())
        except Exception:
            pass

    import odds_history as _oh

    def _amer_to_implied(amer):
        try:
            o = int(amer)
            if o > 0:
                return round(100 / (100 + o) * 100, 1)
            else:
                return round(-o / (-o + 100) * 100, 1)
        except Exception:
            return None

    import re as _re
    _ts_strip = _re.compile(r'_\d{4}-\d{2}-\d{2}T\d{2}$')

    result = {}
    for bet in open_bets:
        score_key = _ts_strip.sub('', bet.game_key)  # strip _YYYY-MM-DDTHH suffix
        info = score_map.get(score_key)
        if not info or info.get('status') == 'Preview':
            continue
        entry = dict(info)
        # Attach live implied win probability from current FanDuel odds
        if bet.bet_side in ('home', 'away'):
            snap = _oh.get_latest((bet.sport or '').upper(), bet.game_key)
            if snap:
                live_amer = snap.get(f'{bet.bet_side}_odds')
                if live_amer is not None:
                    entry['live_odds']    = live_amer
                    entry['live_implied'] = _amer_to_implied(live_amer)
        result[str(bet.id)] = entry

    return jsonify(result)


@app.route('/api/mlb/game/<int:game_pk>/boxscore')
def api_mlb_game_boxscore(game_pk):
    """Trimmed current-game batting/pitching lines for the schedule page's per-card
    'Stats & analysis' panel — fetched on demand when a card is expanded, not on
    every page load."""
    box = mlb_api.get_game_boxscore(game_pk)
    if box is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(box)


@app.route('/api/lifeos/mlb/game/<int:game_pk>/boxscore')
def api_lifeos_mlb_game_boxscore(game_pk):
    """Same trimmed boxscore as /api/mlb/game/<id>/boxscore, gated for LifeOS the
    same way /api/lifeos/games is (shared-secret LIFEOS_API_TOKEN header)."""
    expected_token = os.environ.get('LIFEOS_API_TOKEN', '')
    provided_token = request.headers.get('X-LifeOS-Token', '')
    if not expected_token or provided_token != expected_token:
        return jsonify({'error': 'unauthorized'}), 401

    box = mlb_api.get_game_boxscore(game_pk)
    if box is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(box)


@app.route('/api/lifeos/games')
def api_lifeos_games():
    """
    Read-only feed for LifeOS (a separate app, /home/spooky/lifedb) — today's MLB/NFL
    games with live score, status, and FanDuel odds where available, so its Sports
    page can show a card per game instead of re-implementing its own score/odds
    fetching. Reuses the same cached get_live_scores()/get_odds_map() this app's own
    dashboard already calls, so it adds no extra load on MLB/ESPN/Odds API beyond
    what's already happening — LifeOS just gets to read the result.

    No other endpoint in this app requires auth, but this one is reachable from
    outside this app's own dashboard, so it's gated on a shared-secret header
    (LIFEOS_API_TOKEN, set identically in both apps' .env — see docker-compose.yml).
    """
    expected_token = os.environ.get('LIFEOS_API_TOKEN', '')
    provided_token = request.headers.get('X-LifeOS-Token', '')
    if not expected_token or provided_token != expected_token:
        return jsonify({'error': 'unauthorized'}), 401

    import mlb_api as _mlb, nfl_api as _nfl
    import odds_api as _odds
    from odds_api import _normalize

    sport_param = request.args.get('sport', 'mlb,nfl')
    requested = {s.strip().lower() for s in sport_param.split(',') if s.strip()}

    games = []

    def _mlb_start_times():
        """
        get_live_scores() doesn't expose gameDate, but LifeOS's Today page needs a start
        time to rank an upcoming (not-yet-started) game — so this re-reads the same
        schedule payload get_live_scores() already fetched and cached (_get_schedule_raw
        is keyed by date + a 2-min TTL, so this is a cache hit, not a second live call).
        """
        times = {}
        try:
            data = _mlb._get_schedule_raw()
        except Exception:
            return times
        for date_obj in (data or {}).get('dates', []):
            for game in date_obj.get('games', []):
                h_name = game.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
                a_name = game.get('teams', {}).get('away', {}).get('team', {}).get('name', '')
                if not h_name or not a_name:
                    continue
                times[f'{_normalize(h_name)}_{_normalize(a_name)}'] = game.get('gameDate')
        return times

    def _nfl_start_times():
        """Same idea as _mlb_start_times() — reuses nfl_api's own cache key so this is a
        cache hit, not a second call to ESPN."""
        times = {}
        try:
            today_str = _nfl._today_et()
            data = _nfl._cached_get(_nfl.ESPN_NFL, {}, f'nfl_scores_{today_str}', 120)
        except Exception:
            return times
        for event in (data or {}).get('events', []):
            comp = event.get('competitions', [{}])[0]
            names = {}
            for competitor in comp.get('competitors', []):
                side = competitor.get('homeAway', 'home')
                names[side] = competitor.get('team', {}).get('displayName', '')
            home, away = names.get('home', ''), names.get('away', '')
            if not home or not away:
                continue
            times[f'{_normalize(home)}_{_normalize(away)}'] = event.get('date')
        return times

    def _collect(sport_key, score_map, start_times):
        odds_map = {}
        try:
            odds_map = _odds.get_odds_map(sport_key)
        except Exception as e:
            print(f'[lifeos] odds fetch failed for {sport_key}: {e}', flush=True)

        for game_key, info in score_map.items():
            # game_key is "{normalized_home}_{normalized_away}" (odds_api._normalize
            # lowercases/strips accents but keeps spaces, so a single "_" cleanly
            # separates the two halves) — reused here only to look up odds/start time by
            # name, not for display (home_abbr/away_abbr below are the display identifiers).
            home_norm, _, away_norm = game_key.partition('_')
            odds = None
            try:
                odds = _odds.lookup_game_odds(odds_map, home_norm, away_norm)
            except Exception:
                pass

            games.append({
                'sport': sport_key,
                'homeTeam': info.get('home_abbr') or home_norm,
                'awayTeam': info.get('away_abbr') or away_norm,
                'status': info.get('status'),
                'homeScore': info.get('home_score'),
                'awayScore': info.get('away_score'),
                'period': info.get('period'),
                'startAt': start_times.get(game_key),
                'gamePk': info.get('game_pk') if sport_key == 'mlb' else None,
                'odds': {
                    'homeMoneyline': odds.get('home_best'),
                    'awayMoneyline': odds.get('away_best'),
                    'totalLine': odds.get('total_line'),
                    'overOdds': odds.get('over_odds'),
                    'underOdds': odds.get('under_odds'),
                } if odds else None,
            })

    if 'mlb' in requested:
        try:
            _collect('mlb', _mlb.get_live_scores(), _mlb_start_times())
        except Exception as e:
            print(f'[lifeos] mlb fetch failed: {e}', flush=True)
    if 'nfl' in requested:
        try:
            _collect('nfl', _nfl.get_live_scores(), _nfl_start_times())
        except Exception as e:
            print(f'[lifeos] nfl fetch failed: {e}', flush=True)

    return jsonify({'games': games, 'generatedAt': datetime.now(timezone.utc).isoformat()})


# ── Background cache warmer ───────────────────────────────────────────────────
# Keeps all three schedule caches pre-populated so page loads are instant.
# ── Discord score alerts ───────────────────────────────────────────────────────
# Trust Score alerting was removed here (Trust Score itself is still computed
# internally — Unified Score is partly derived from its component values —
# but it's no longer surfaced, tracked, or alerted on anywhere in this file).
_PREV_SCORES: dict = {}    # {game_key: {...snapshot...}} from last warmer run
_PREV_SCORES_LOCK = threading.Lock()
_SCORE_ALERT_THRESHOLD = 10   # minimum Unified Score delta to fire a movement alert
_UNF_HIGH_THRESHOLD    = 60   # unified score "high confidence" crossing alert


def _snapshot_reason_bits(prev, curr):
    """
    Build a human-readable list of "what actually changed" between two
    Unified Score snapshots of the same game, using the concrete underlying
    values (not the abstracted 0-1 component weights) — so a Discord alert
    can say *why* the score moved, not just that it did.

    Compares, in the order most likely to matter:
      1. Edge zone bucket   (e.g. '-2 to +5%' -> '+5 to +10%')
      2. Favorite/Dog status (pick flipped from market favorite to underdog, or back)
      3. Line movement % vs opening (continuous, always shown if available)
      4. Movement profile pattern (e.g. 'flat' -> 'late_sharp_for')
      5. Odds payout bucket (crossed a Kelly-multiplier threshold, e.g. -150)

    Returns a list of strings, already ordered; empty if nothing meaningfully
    changed (can happen if Unified Score moved from rounding/threshold noise
    alone, e.g. crossing bucket bounds by <1 point).
    """
    bits = []

    if prev.get('el') != curr.get('el') and curr.get('el'):
        bits.append(f"Edge zone: {prev.get('el', '—')} → **{curr.get('el')}**")

    if prev.get('fl') != curr.get('fl') and curr.get('fl') not in (None, '—'):
        bits.append(f"Market status: {prev.get('fl', '—')} → **{curr.get('fl')}**")

    plm, clm = prev.get('lm_pct'), curr.get('lm_pct')
    if clm is not None and (plm is None or abs(clm - plm) >= 0.5):
        prev_str = f"{plm:+.1f}%" if plm is not None else "—"
        bits.append(f"Line move vs open: {prev_str} → **{clm:+.1f}%**")

    if prev.get('mp') != curr.get('mp') and curr.get('mp'):
        bits.append(f"Movement pattern: {(prev.get('mp') or '—').replace('_', ' ')} → "
                    f"**{curr.get('mp').replace('_', ' ')}**")

    p_odds, c_odds = prev.get('odds'), curr.get('odds')
    if p_odds is not None and c_odds is not None:
        p_mult, c_mult = _odds_multiplier(p_odds), _odds_multiplier(c_odds)
        if p_mult != c_mult:
            bits.append(f"Odds: {p_odds:+d} → **{c_odds:+d}** (crossed a payout-weighting threshold)")

    return bits


def _check_unified_score_alerts(schedule, webhook_url: str) -> None:
    """Compare pre-game Unified Score to the previous warmer snapshot.
    Fires Discord embeds for:
      • Unified Score movement ±10+ (with a "why" breakdown — see
        _snapshot_reason_bits)
      • Unified Score first crossing 60 (high-confidence alert)
    """
    import requests as _req
    import odds_history
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')

    if not webhook_url:
        return

    new_scores: dict = {}
    alerts: list = []

    for day in (schedule or []):
        for game in day.get('games', []):
            if game.get('status') != 'Preview':
                continue
            model = game.get('model')
            odds  = game.get('odds')
            if not model or not odds:
                continue

            home_obj  = game.get('home') or {}
            away_obj  = game.get('away') or {}
            hp, ap    = model.get('home_prob', 0.5), model.get('away_prob', 0.5)
            bet_side  = 'home' if hp >= ap else 'away'
            pick_abbr = home_obj.get('abbr', '') if bet_side == 'home' else away_obj.get('abbr', '')

            fj  = model.get('factors_json') or json.dumps(model.get('factors', []))
            h_odds = odds.get('home_best')
            a_odds = odds.get('away_best')
            # Trust Score is computed here purely as an internal input to
            # Unified Score (which is partly derived from its edge/fav-dog
            # component values) — the composite number itself is never
            # stored, compared, or surfaced below.
            td = _trust_score(hp, h_odds, a_odds, fj, detail=True)

            # Compute Unified Score — needs opening → current line move
            line_move_val = 0.5  # neutral default
            lm_pct = None
            open_h = odds.get('opening_home')
            open_a = odds.get('opening_away')
            if open_h and open_a and h_odds and a_odds:
                vf_h_open, vf_a_open = _vig_free_implied(open_h, open_a)
                vf_h_curr, vf_a_curr = _vig_free_implied(h_odds, a_odds)
                if vf_h_open and vf_h_curr:
                    pick_open = vf_h_open if bet_side == 'home' else vf_a_open
                    pick_curr = vf_h_curr if bet_side == 'home' else vf_a_curr
                    lm_pct    = (pick_curr - pick_open) * 100
                    line_move_val = max(0.0, min(1.0, 0.5 + lm_pct / 10.0))

            game_start = None
            _utc_raw = game.get('game_time_utc', '')
            if _utc_raw:
                try:
                    game_start = datetime.fromisoformat(_utc_raw.replace('Z', '+00:00'))
                except Exception:
                    game_start = None
            movement_profile, _ = odds_history.classify_movement(
                'baseball_mlb', home_obj.get('name', ''), away_obj.get('name', ''),
                bet_side == 'home', game_start)
            movement_val = _movement_value(movement_profile)
            pick_odds = h_odds if bet_side == 'home' else a_odds
            unified = _unified_score(td, line_move_val, movement_val, pick_odds)

            game_key = f"{away_obj.get('abbr', '')}@{home_obj.get('abbr', '')}"
            snapshot = {
                'unf': unified, 'el': td.get('el'), 'fl': td.get('fl'),
                'lm_pct': round(lm_pct, 1) if lm_pct is not None else None,
                'mp': movement_profile, 'odds': pick_odds,
            }
            new_scores[game_key] = snapshot

            try:
                dt = datetime.fromisoformat(
                    game.get('game_time_utc', '').replace('Z', '+00:00'))
                time_str = dt.astimezone(_ET).strftime('%-I:%M %p ET')
            except Exception:
                time_str = ''

            with _PREV_SCORES_LOCK:
                prev = _PREV_SCORES.get(game_key)

            if prev is not None:
                prev_unf  = prev.get('unf', unified)
                unf_delta = unified - prev_unf

                if abs(unf_delta) >= _SCORE_ALERT_THRESHOLD:
                    alerts.append({
                        'type': 'unf', 'game': game_key, 'pick': pick_abbr,
                        'prev': prev_unf, 'curr': unified, 'delta': unf_delta,
                        'why': _snapshot_reason_bits(prev, snapshot), 'time': time_str,
                        'odds': pick_odds,
                    })
                elif (prev_unf < _UNF_HIGH_THRESHOLD <= unified):
                    # Crossed into high-confidence zone without triggering movement alert
                    alerts.append({
                        'type': 'unf_high', 'game': game_key, 'pick': pick_abbr,
                        'prev': prev_unf, 'curr': unified, 'delta': unf_delta,
                        'why': _snapshot_reason_bits(prev, snapshot), 'time': time_str,
                        'odds': pick_odds,
                    })
            else:
                # First observation — fire if already high confidence
                if unified >= _UNF_HIGH_THRESHOLD:
                    alerts.append({
                        'type': 'unf_high', 'game': game_key, 'pick': pick_abbr,
                        'prev': None, 'curr': unified, 'delta': 0,
                        'why': [], 'time': time_str,
                        'odds': pick_odds,
                    })

    with _PREV_SCORES_LOCK:
        _PREV_SCORES.update(new_scores)

    for a in alerts:
        up = a['delta'] > 0
        t  = a['type']
        why_str = '\n'.join(f"• {b}" for b in a['why']) if a['why'] else ''

        if t == 'unf':
            title = f"{'📈' if up else '📉'} Unified Score Movement · MLB"
            desc  = (f"**{a['game']}** · Pick: **{a['pick']}**\n"
                     f"Unified Score: {a['prev']} → **{a['curr']}** ({a['delta']:+d}) {'↑' if up else '↓'}")
            if why_str:
                desc += f"\n\n**Why:**\n{why_str}"
            color = 0x4ade80 if up else 0xf87171

        else:  # unf_high
            prev_str = f"{a['prev']} → " if a['prev'] is not None else ''
            title = f"🎯 High Confidence Pick · MLB"
            desc  = (f"**{a['game']}** · Pick: **{a['pick']}**\n"
                     f"Unified Score: {prev_str}**{a['curr']}** (crossed {_UNF_HIGH_THRESHOLD})")
            if why_str:
                desc += f"\n\n**Why:**\n{why_str}"
            color = 0xfb923c  # orange

        if a['time']:
            desc += f"\n\nGame: {a['time']}"

        # High Confidence Pick alerts get a team-branded image card attached
        # (see discord_cards.py) — deliberately not the regular movement
        # alerts too, which would spam the channel with an image on every
        # ±10 move on a busy slate. Falls back to the plain text embed (still
        # sent either way) if image generation fails for any reason — a
        # broken card should never mean a missed alert.
        image_bytes = None
        if t == 'unf_high':
            try:
                import discord_cards
                matchup_line = a['game'].replace('@', ' @ ') + f", Pick: {a['pick']} ML"
                odds = a.get('odds')
                if odds:
                    try:
                        matchup_line += f" · {int(odds):+d}"
                    except (TypeError, ValueError):
                        pass
                image_bytes = discord_cards.render_score_move_card(
                    a['pick'], a['prev'], a['curr'], matchup_line, a['why'])
            except Exception as e:
                print(f'[unified-score-alert] card render failed: {e}', flush=True)

        try:
            import requests as _req
            embed = {'title': title, 'description': desc, 'color': color}
            if image_bytes:
                embed['image'] = {'url': 'attachment://score_move.png'}
                _req.post(webhook_url,
                          data={'payload_json': json.dumps({'embeds': [embed]})},
                          files={'files[0]': ('score_move.png', image_bytes, 'image/png')},
                          timeout=10)
            else:
                _req.post(webhook_url, json={'embeds': [embed]}, timeout=5)
        except Exception:
            pass


# Runs in a daemon thread — won't block shutdown.
# NOTE: works correctly with a single Gunicorn worker (1 process = 1 shared cache).

def _send_daily_mlb_recommendation(schedule, webhook_url: str) -> None:
    """Post the full ranked MLB recommendation list to Discord at noon ET."""
    import requests as _req
    import odds_history
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
    print('[daily-picks] triggered', flush=True)
    if not webhook_url:
        print('[daily-picks] no webhook URL configured — skipping', flush=True)
        return

    candidates = []
    for day in (schedule or []):
        for game in day.get('games', []):
            if game.get('status', 'Preview') != 'Preview':
                continue
            model = game.get('model')
            odds  = game.get('odds')
            if not model or not odds:
                continue
            home_obj  = game.get('home') or {}
            away_obj  = game.get('away') or {}
            home_abbr = home_obj.get('abbr') or home_obj.get('name', '?')[:3].upper()
            away_abbr = away_obj.get('abbr') or away_obj.get('name', '?')[:3].upper()
            hp    = model.get('home_prob', 0.5)
            ap    = model.get('away_prob', 0.5)
            h_odds = odds.get('home_best', 0)
            a_odds = odds.get('away_best', 0)
            bet_side  = 'home' if hp >= ap else 'away'
            fj        = model.get('factors_json') or json.dumps(model.get('factors', []))
            td        = _trust_score(hp, h_odds, a_odds, fj, detail=True)
            lm = _line_move_pct(hp, h_odds, a_odds,
                                odds.get('opening_home'), odds.get('opening_away'))
            lm_val = max(0.0, min(1.0, 0.5 + lm / 10.0)) if lm is not None else 0.5
            game_start = None
            _utc_raw = game.get('game_time_utc', '')
            if _utc_raw:
                try:
                    game_start = datetime.fromisoformat(_utc_raw.replace('Z', '+00:00'))
                except Exception:
                    game_start = None
            movement_profile, _ = odds_history.classify_movement(
                'baseball_mlb', home_obj.get('name', ''), away_obj.get('name', ''),
                bet_side == 'home', game_start)
            movement_val = _movement_value(movement_profile)
            pick_odds = h_odds if bet_side == 'home' else a_odds
            unf       = _unified_score(td, lm_val, movement_val, pick_odds)
            pick_abbr = home_abbr if bet_side == 'home' else away_abbr
            opp_abbr  = away_abbr if bet_side == 'home' else home_abbr
            pick_odds = h_odds if bet_side == 'home' else a_odds
            gt_utc    = game.get('game_time_utc') or ''
            gt_str    = ''
            if gt_utc:
                try:
                    from datetime import datetime as _dt
                    gt_str = _dt.fromisoformat(gt_utc.replace('Z', '+00:00')) \
                               .astimezone(_ET).strftime('%-I:%M %p')
                except Exception:
                    pass
            candidates.append({
                'pick':  pick_abbr,
                'opp':   opp_abbr,
                'odds':  f'{pick_odds:+d}' if pick_odds else '—',
                'unf':   unf,
                'time':  gt_str,
            })

    if not candidates:
        print('[daily-picks] no Preview games with unified scores — message not sent', flush=True)
        return

    print(f'[daily-picks] building embed with {len(candidates)} games', flush=True)
    candidates.sort(key=lambda x: -x['unf'])

    # Monospace table — one line per game, all columns aligned
    header = f"{'#':>2}  {'PICK':<4}  {'OPP':<4}  {'ODDS':>5}  {'US':>2}  TIME"
    sep    = '─' * len(header)
    rows   = [header, sep]
    for i, c in enumerate(candidates, 1):
        rows.append(
            f"{i:>2}  {c['pick']:<4}  {c['opp']:<4}  {c['odds']:>5}  {c['unf']:>2}  {c['time']}"
        )

    today_str   = datetime.now(_ET).strftime('%A, %b %-d')
    description = '```\n' + '\n'.join(rows) + '\n```'
    # Discord embed description limit is 4096 chars
    if len(description) > 3900:
        description = description[:3900] + '\n…```'

    try:
        r = _req.post(webhook_url, json={'embeds': [{
            'title':       f'⚾ MLB Daily Picks · {today_str}',
            'description': description,
            'color':       0xFB4F14,
            'footer':      {'text': 'Sorted by Unified Score · noon ET snapshot'},
        }]}, timeout=10)
        print(f'[daily-picks] Discord response: {r.status_code}', flush=True)
    except Exception as e:
        print(f'[daily-picks] Discord POST failed: {e}', flush=True)


def _warm_all_caches(send_daily: bool = False):
    # Snapshot purging disabled — keeping all odds_snapshot rows to build a real
    # sample for line-movement-profile analysis. Revisit once that data is old
    # enough to roll up into a per-game category instead of raw odds.
    for name, fn in [('MLB', mlb_api.build_schedule_context),
                     ('NHL', nhl_api.build_schedule_context),
                     ('NFL', nfl_api.build_schedule_context),
                     ('CFB', cfb_api.build_schedule_context)]:
        try:
            result = fn()
            if result:
                with app.app_context():
                    _upsert_predictions(result, name)
                    if name == 'MLB':
                        s       = Setting.query.first()
                        webhook = (s.discord_webhook_url or '').strip() if s else ''
                        _check_unified_score_alerts(result, webhook)
                        if send_daily:
                            _send_daily_mlb_recommendation(result, webhook)
        except Exception:
            pass


def _warm_all_caches_noon():
    """Warmer run at noon — same as normal but also fires the daily picks message."""
    print('[scheduler] noon warmer fired — will send daily picks', flush=True)
    _warm_all_caches(send_daily=True)


# ── Nightly outcome resolver ──────────────────────────────────────────────────
# Fills in home_won / scores for GamePrediction rows that are still pending.
# Runs at 5 am ET — after the latest West Coast games (~midnight PT) are Final.
# Only resolves outcomes; use backfill_2026.py to re-score with new model weights.

def _resolve_pending_outcomes():
    """Fetch the last 3 days of Final MLB games and close any pending predictions."""
    import requests as _req
    from datetime import date, timedelta
    from zoneinfo import ZoneInfo

    _MLB_API = 'https://statsapi.mlb.com/api/v1'
    _ET = ZoneInfo('America/New_York')
    today_et = datetime.now(_ET).date()
    start    = (today_et - timedelta(days=3)).isoformat()
    end      = today_et.isoformat()

    try:
        r = _req.get(f'{_MLB_API}/schedule', params={
            'sportId': 1, 'gameType': 'R',
            'startDate': start, 'endDate': end,
            'hydrate': 'linescore,teams',
        }, timeout=15)
        r.raise_for_status()
        dates = r.json().get('dates', [])
    except Exception:
        return

    now = datetime.now(timezone.utc)
    resolved = 0
    with app.app_context():
        for day in dates:
            for game in day.get('games', []):
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                home = game.get('teams', {}).get('home', {})
                away = game.get('teams', {}).get('away', {})
                h_name   = home.get('team', {}).get('name', '')
                a_name   = away.get('team', {}).get('name', '')
                game_date = day.get('date', '')
                try:
                    h_score = int(home.get('score') or -1)
                    a_score = int(away.get('score') or -1)
                except (TypeError, ValueError):
                    continue
                if h_score < 0 or a_score < 0 or not game_date:
                    continue

                pred = GamePrediction.query.filter_by(
                    sport='MLB',
                    game_date=game_date,
                    home_team=h_name,
                    away_team=a_name,
                    home_won=None,
                ).first()
                if pred is not None:
                    pred.home_won       = (h_score > a_score)
                    pred.home_score     = h_score
                    pred.away_score     = a_score
                    pred.outcome_set_at = now
                    resolved += 1

        # Purge predictions that are still pending after 1+ days — these are
        # postponed, cancelled, or otherwise never going Final.
        cutoff = (today_et - timedelta(days=1)).isoformat()
        stale = (GamePrediction.query
                 .filter(GamePrediction.sport == 'MLB',
                         GamePrediction.home_won.is_(None),
                         GamePrediction.game_date < cutoff)
                 .all())
        purged = len(stale)
        for p in stale:
            db.session.delete(p)

        if resolved or purged:
            try:
                db.session.commit()
                if resolved:
                    print(f'[outcome-resolver] resolved {resolved} pending predictions', flush=True)
                    _refit_mlb_platt()
                    _recompute_trust_weights('MLB')
                    _recompute_team_bias('MLB')
                    _recompute_unified_weights('MLB')
                    _recompute_pick_of_day_weights('MLB')
                    _recompute_unified_rank_stats('MLB')
                    _recompute_movement_profiles('MLB')
                    _recompute_movement_weights('MLB')
                    _recompute_consensus_calibration('MLB')
                if purged:
                    print(f'[outcome-resolver] purged {purged} stale pending predictions', flush=True)
            except Exception:
                db.session.rollback()


def _refit_mlb_platt():
    """Load resolved MLB predictions from DB and refit Platt scaler."""
    with app.app_context():
        records = [
            (p.home_prob, p.home_won)
            for p in GamePrediction.query.filter_by(sport='MLB').all()
            if p.home_prob is not None and p.home_won is not None
        ]
    mlb_model.fit_platt(records)


def _start_cache_warmer():
    from apscheduler.schedulers.background import BackgroundScheduler
    from zoneinfo import ZoneInfo

    scheduler = BackgroundScheduler(daemon=True)
    # Non-noon runs: 8,10,14,16,18,20 ET — TS alerts only
    scheduler.add_job(_warm_all_caches, 'cron',
                      hour='8,10,14,16,18,20', minute=0,
                      timezone='America/New_York', id='warm_caches')
    # Noon run: TS alerts + daily picks message
    scheduler.add_job(_warm_all_caches_noon, 'cron',
                      hour=12, minute=0,
                      timezone='America/New_York', id='warm_caches_noon')
    # Outcome resolver — 5 am ET daily (after all West Coast games are Final)
    scheduler.add_job(_resolve_pending_outcomes, 'cron',
                      hour=5, minute=0,
                      timezone=ZoneInfo('America/New_York'),
                      id='resolve_outcomes')
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))

    # Calibrations are fast (DB-only) — run synchronously so the first page load
    # sees correct Platt scaling, grade adjustments, and Trust Score weights.
    _refit_mlb_platt()
    _recompute_trust_weights('MLB')
    _recompute_team_bias('MLB')
    _recompute_unified_weights('MLB')
    _recompute_pick_of_day_weights('MLB')
    _recompute_unified_rank_stats('MLB')
    _recompute_movement_profiles('MLB')
    _recompute_movement_weights('MLB')
    _recompute_consensus_calibration('MLB')

    # API cache warming and outcome resolution are slow (network calls) — run in background.
    def _startup():
        _warm_all_caches()
        _resolve_pending_outcomes()
        _refit_mlb_platt()
        _recompute_trust_weights('MLB')
        _recompute_team_bias('MLB')
        _recompute_unified_weights('MLB')
        _recompute_pick_of_day_weights('MLB')
        _recompute_unified_rank_stats('MLB')
        _recompute_movement_profiles('MLB')
        _recompute_movement_weights('MLB')
        _recompute_consensus_calibration('MLB')
    threading.Thread(target=_startup, daemon=True, name='warm-startup').start()

# Only start the warmer in the actual server process, not during testing or
# when Flask's dev-server reloader spawns a child process, and not when
# app.py is imported for its models/db by an offline script (e.g.
# backfill_2026.py, which sets DISABLE_STARTUP_TASKS before importing).
# Without this guard, importing app.py from a script spins up the full
# scheduler + live cache warming + a synchronous _recompute_team_bias() over
# whatever's currently in the DB — an expensive, unwanted side effect for a
# script that just wants the models, and the exact "recompute bias live on
# every backfill run" moving-target bug this guard exists to stop.
if (os.environ.get('WERKZEUG_RUN_MAIN') != 'false'
        and not os.environ.get('DISABLE_STARTUP_TASKS')):
    _start_cache_warmer()


if __name__ == '__main__':
  app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)), debug=True)