"""
news_sentiment.py — Sport-agnostic keyword/phrase sentiment scoring for
player-news text. Deliberately NOT an LLM call per article (cost/latency
at RSS-feed volume) and NOT a trained classifier (no labeled dataset
exists yet) — a small, hand-maintained lexicon, same spirit as
mlb_model._TRADE_DEADLINE_DISCOUNT's manually-curated, manually-updated
dict rather than a fitted model. Expand CORE_LEXICON/<SPORT>_LEXICON based
on what nfl_news_backtest.py's matched_terms actually show mattering, not
speculatively upfront.

score_text() is called per-article by nfl_news.py/cfb_news.py/nhl_news.py/
nba_news.py/wnba_news.py/mlb_news.py (via news_ingest.score_article()) —
kept in its own module specifically so the lexicon/scoring logic is
reusable across sports without any one of those owning it.
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

# Sport-specific overlays, merged on top of CORE_LEXICON, parameterized
# purely via the `sport` argument below — no rewrite needed to add one.
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

# CFB/NHL/NBA/WNBA/MLB overlays added alongside the news-ingest pipeline's
# expansion to those sports (nfl_news.py's engine was split into
# news_ingest.py; see cfb_news.py/nhl_news.py/nba_news.py/wnba_news.py/
# mlb_news.py). Deliberately sparse at launch, same reasoning as this
# module's own docstring — expand based on what matched_terms actually
# show mattering once each sport's own backtest exists, not speculatively.
CFB_LEXICON = [
    ('entered the transfer portal', -0.7),
    ('transfer portal', -0.4),
    ('suspended', -0.6),
    ('named the starter', 0.6),
    ('heisman', 0.4),
    ('decommitted', -0.5),
    ('committed to', 0.3),
]

NHL_LEXICON = [
    ('placed on injured reserve', -1.0),
    ('day-to-day', -0.2),
    ('upper-body injury', -0.4),
    ('lower-body injury', -0.4),
    ('recalled from', 0.3),
    ('sent down to', -0.5),
    ('scratched', -0.5),
    ('on the top line', 0.4),
    ('power play unit', 0.3),
    ('hot streak', 0.5),
]

NBA_LEXICON = [
    ('load management', -0.3),
    ('rest night', -0.4),
    ('starting lineup', 0.4),
    ('coming off the bench', -0.2),
    ('minutes restriction', -0.3),
    ('triple-double', 0.5),
    ('cleared for contact', 0.4),
]

WNBA_LEXICON = [
    ('starting lineup', 0.4),
    ('minutes restriction', -0.3),
    ('ruled out', -1.0),
    ('cleared to return', 0.6),
]

MLB_LEXICON = [
    ('placed on the il', -1.0),
    ('placed on the injured list', -1.0),
    ('activated from the il', 0.6),
    ('demoted to the minors', -0.6),
    ('called up', 0.5),
    ('batting leadoff', 0.3),
    ('moved up in the order', 0.4),
    ('dropped in the order', -0.3),
    ('scratched from the lineup', -0.6),
    ('tommy john', -0.9),
]

_OVERLAYS = {
    'NFL': NFL_LEXICON,
    'CFB': CFB_LEXICON,
    'NHL': NHL_LEXICON,
    'NBA': NBA_LEXICON,
    'WNBA': WNBA_LEXICON,
    'MLB': MLB_LEXICON,
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
