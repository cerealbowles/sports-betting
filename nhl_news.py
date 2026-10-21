"""
nhl_news.py — NHL player/team news ingestion. Thin config layer over
news_ingest.py (the shared fetch -> match -> score -> persist engine
originally built for nfl_news.py) — see that module's docstring for the
mechanics, and nfl_news.py for the sibling this was split from.

CBS's "nhl" headlines RSS feed was confirmed live to mix in some other-
sport content (its own cross-sport betting roundups) — harmless here since
news_ingest.match_players() only ever matches against NHL rosters, so an
NFL/MLB mention in the feed just yields "no match", not a wrong one.

Wired into app.py's BackgroundScheduler — see _ingest_nhl_news's docstring
for the cron details.

Run: python3 nhl_news.py [--dry-run] [--lookback-hours N]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import news_ingest
import nhl_roster_api

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl'

NEWS_SOURCES = [
    {'name': 'espn_json', 'kind': 'espn_json', 'url': f'{ESPN_SITE}/news'},
    {'name': 'espn_rss', 'kind': 'rss', 'url': 'https://www.espn.com/espn/rss/nhl/news'},
    {'name': 'cbssports_rss', 'kind': 'rss', 'url': 'https://www.cbssports.com/rss/headlines/nhl/'},
]


def run(lookback_hours=48, dry_run=False):
    return news_ingest.run('NHL', ESPN_SITE, nhl_roster_api, NEWS_SOURCES,
                            lookback_hours=lookback_hours, dry_run=dry_run)


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    lookback_hours = 48
    if '--lookback-hours' in sys.argv:
        idx = sys.argv.index('--lookback-hours')
        if idx + 1 < len(sys.argv):
            lookback_hours = int(sys.argv[idx + 1])
    run(lookback_hours=lookback_hours, dry_run=dry_run)
