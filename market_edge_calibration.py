"""
Shared snapshot I/O for the model-vs-market disagreement shrink used by every
*_model.py (mlb/nfl/nba/nhl/cfb). The shrink pulls a model's probability back
toward the market once they disagree by more than a fixed 10pp threshold —
see the "Shrink large model-vs-market disagreements" block in each model.

The 10pp threshold stays a fixed, hand-chosen constant in each model file.
Only the *rate* (how much of the excess beyond 10pp gets pulled back) is
data-driven: app.py's _recompute_market_edge_shrink() refits it per sport
from resolved game_predictions and calls save_snapshot(); each model loads
its snapshot at import via load_snapshot(). Sports with too few resolved
large-edge games fall back to the hardcoded 25% prior (same one MLB's
initial one-off study produced) until they accumulate enough of their own.
"""
import json
import os
from datetime import datetime, timezone

_SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'market_edge_shrink_snapshot.json')

# Prior rate — from MLB's initial 50-game / 8-large-edge-game study (see
# mlb_model.py). Used for any sport until it has its own recomputed snapshot.
PRIOR_RATE = 0.25


def load_rate(sport):
    """Return the current shrink rate for `sport`, or PRIOR_RATE if no
    snapshot exists yet (first run, or sport hasn't hit the min-sample gate)."""
    try:
        if not os.path.exists(_SNAPSHOT_PATH):
            return PRIOR_RATE
        with open(_SNAPSHOT_PATH) as f:
            payload = json.load(f)
        entry = payload.get(sport)
        if not entry or 'rate' not in entry:
            return PRIOR_RATE
        return float(entry['rate'])
    except Exception:
        return PRIOR_RATE


def save_snapshot(sport, rate, n):
    try:
        payload = {}
        if os.path.exists(_SNAPSHOT_PATH):
            try:
                with open(_SNAPSHOT_PATH) as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
        payload[sport] = {
            'rate':        round(rate, 4),
            'n':           n,
            'computed_at': datetime.now(timezone.utc).isoformat(),
        }
        with open(_SNAPSHOT_PATH, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f'[market-edge] failed to write snapshot: {e}', flush=True)
