"""
news_sentiment.py — Sport-agnostic keyword/phrase sentiment scoring for
player-news text. Deliberately NOT an LLM call per article (cost/latency
at RSS-feed volume) and NOT a trained classifier (no labeled dataset
exists yet) — a small, hand-maintained lexicon, same spirit as
mlb_model._TRADE_DEADLINE_DISCOUNT's manually-curated, manually-updated
dict rather than a fitted model. Expand CORE_LEXICON/<SPORT>_LEXICON based
on what nfl_news_backtest.py's matched_terms actually show mattering, not
speculatively upfront.

score_text() is called per-article by nfl_news.py (and, later, an MLB
equivalent) — kept in its own module specifically so the lexicon/scoring
logic is reusable across sports without nfl_news.py owning it.
"""
import math
import re

# Status/trajectory language that means roughly the same thing regardless
# of sport — availability, role change, physical state.
CORE_LEXICON = [
    ('ruled out', -1.0),
    ('placed on ir', -1.0),
    ('placed on injured reserve', -1.0),
    ('will miss', -0.8),
    ('out indefinitely', -0.8),
    ('season-ending', -0.9),
    ('setback', -0.6),
    ('doubtful', -0.5),
    ('questionable', -0.3),
    ('limited practice', -0.4),
    ('did not practice', -0.5),
    ('benched', -0.7),
    ('demoted', -0.6),
    ('timeline unclear', -0.5),
    ('day-to-day', -0.2),
    ('full practice', 0.4),
    ('cleared to play', 0.6),
    ('no limitations', 0.5),
    ('full participant', 0.4),
    ('named starter', 0.7),
    ('returns from injury', 0.6),
    ('activated from ir', 0.6),
    ('best shape of his career', 0.5),
    ('breakout', 0.5),
    ('expected to play', 0.4),
    ('trending toward playing', 0.4),
]

# Sport-specific overlays, merged on top of CORE_LEXICON. NFL only for now
# (see this feature's plan — MLB/others add a sibling list here later,
# parameterized purely via the `sport` argument below, no rewrite needed).
NFL_LEXICON = [
    ('projected qb1', 0.6),
    ('depth chart move', 0.3),
    ('snap count increase', 0.4),
    ('elevated to', 0.4),
    ('workload increase', 0.4),
    ('bottom-5 run defense', 0.4),
    ('bottom-five run defense', 0.4),
    ('get-right spot', 0.3),
    ('get right spot', 0.3),
    ('plus matchup', 0.3),
    ('favorable matchup', 0.3),
    ('smash spot', 0.4),
]

_OVERLAYS = {
    'NFL': NFL_LEXICON,
}


def _lexicon_for(sport):
    return CORE_LEXICON + _OVERLAYS.get(sport, [])


def score_text(text, sport='NFL'):
    """Case-insensitive, word-boundary phrase scan against CORE_LEXICON
    plus `sport`'s overlay. Returns (score, matched_terms):
    - score: sum of matched phrase weights, tanh-squashed to roughly
      [-1, 1] — bounded units, same reasoning mlb_model.py uses for its
      probability-scale nudges, chosen so this is easy to fold into a
      logit-space factor later without a rescale.
    - matched_terms: [(phrase, weight), ...] for every phrase that hit,
      stored alongside the score (NewsSentiment.matched_terms_json) purely
      for auditability — same "visibly auditable" requirement
      _TRADE_DEADLINE_DISCOUNT documents for its own manual adjustment.
    Returns (0.0, []) for empty/falsy text.
    """
    if not text:
        return 0.0, []

    lowered = text.lower()
    matched = []
    raw_total = 0.0
    for phrase, weight in _lexicon_for(sport):
        if re.search(r'\b' + re.escape(phrase) + r'\b', lowered):
            matched.append((phrase, weight))
            raw_total += weight

    return math.tanh(raw_total), matched
