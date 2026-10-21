"""
nfl_news.py — Pulls NFL player/team news from RSS and JSON feeds, matches
each article to a player or team already known to this app (via
nfl_roster_api's rosters — there is no separate Player table), scores it
with news_sentiment.score_text(), and persists via
nfl_stats_db.ingest_news_sentiment().

Thin NFL-specific config layer over news_ingest.py, which holds the actual
fetch -> match -> score -> persist engine shared with cfb_news.py/
nhl_news.py/nba_news.py/wnba_news.py/mlb_news.py. The one genuinely new
piece of logic when this was NFL-only (no precedent anywhere in this app)
was matching free-text article mentions to a player_id — see
news_ingest.match_players()'s docstring for the confidence rules, still
sport-agnostic there.

build_indexes()/_fetch_espn_teams()/match_players()/score_article()/run()
below stay as module-level functions (not just re-exports) because
nfl_news_backfill.py calls them by these exact names/signatures — confirmed
via grep before this module was split from news_ingest.py.

Meant to run periodically purely to accumulate history going forward —
RSS/news endpoints are current-state-only, so there is no retroactive
backfill possible, same constraint InjuryStatus's docstring documents.
Wired into app.py's BackgroundScheduler as a daily 7am ET job
(_ingest_nfl_news) — see that job's docstring for the cron details.

Run: python3 nfl_news.py [--dry-run] [--lookback-hours N]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import news_ingest
import nfl_roster_api

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl'

# Plain dicts, not parsing logic — a dead/renamed feed URL should never
# require touching news_ingest.match_players()/score_article(). 'kind'
# picks which fetch function handles it. RSS URLs marked unverified were
# the best-known candidates at design time (see this feature's plan) —
# fantasy-site feed slugs drift, so run with --dry-run first to confirm
# each is still reachable before relying on it.
NEWS_SOURCES = [
    {'name': 'espn_json', 'kind': 'espn_json', 'url': f'{ESPN_SITE}/news'},
    {'name': 'espn_rss', 'kind': 'rss', 'url': 'https://www.espn.com/espn/rss/nfl/news'},
    {'name': 'nfl_rss', 'kind': 'rss', 'url': 'https://www.nfl.com/feeds/rss/news'},
    {'name': 'cbssports_rss', 'kind': 'rss', 'url': 'https://www.cbssports.com/rss/headlines/nfl/'},
    {'name': 'rotoballer_rss', 'kind': 'rss', 'url': 'https://www.rotoballer.com/nfl/feed'},
]


def _fetch_espn_teams():
    return news_ingest.fetch_espn_teams(ESPN_SITE)


def build_indexes():
    return news_ingest.build_indexes(ESPN_SITE, nfl_roster_api)


def match_players(text, name_index, team_index):
    return news_ingest.match_players(text, name_index, team_index)


def score_article(text):
    return news_ingest.score_article(text, 'NFL')


def run(lookback_hours=48, dry_run=False):
    return news_ingest.run('NFL', ESPN_SITE, nfl_roster_api, NEWS_SOURCES,
                            lookback_hours=lookback_hours, dry_run=dry_run)


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    lookback_hours = 48
    if '--lookback-hours' in sys.argv:
        idx = sys.argv.index('--lookback-hours')
        if idx + 1 < len(sys.argv):
            lookback_hours = int(sys.argv[idx + 1])
    run(lookback_hours=lookback_hours, dry_run=dry_run)
