"""
mlb_news.py — MLB player/team news ingestion. Thin config layer over
news_ingest.py (the shared fetch -> match -> score -> persist engine
originally built for nfl_news.py) — see that module's docstring for the
mechanics, and nfl_news.py for the sibling this was split from.

Uses ESPN's own baseball/mlb teams/roster endpoints for matching (via
mlb_roster_api.py), separate from team_catalog.py's statsapi.mlb.com-based
MLB team-name lookup used elsewhere in this app — see that roster module's
docstring for why.

Wired into app.py's BackgroundScheduler — see _ingest_mlb_news's docstring
for the cron details.

Run: python3 mlb_news.py [--dry-run] [--lookback-hours N]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import news_ingest
import mlb_roster_api

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb'

NEWS_SOURCES = [
    {'name': 'espn_json', 'kind': 'espn_json', 'url': f'{ESPN_SITE}/news'},
    {'name': 'espn_rss', 'kind': 'rss', 'url': 'https://www.espn.com/espn/rss/mlb/news'},
    {'name': 'cbssports_rss', 'kind': 'rss', 'url': 'https://www.cbssports.com/rss/headlines/mlb/'},
]


def run(lookback_hours=48, dry_run=False):
    return news_ingest.run('MLB', ESPN_SITE, mlb_roster_api, NEWS_SOURCES,
                            lookback_hours=lookback_hours, dry_run=dry_run)


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    lookback_hours = 48
    if '--lookback-hours' in sys.argv:
        idx = sys.argv.index('--lookback-hours')
        if idx + 1 < len(sys.argv):
            lookback_hours = int(sys.argv[idx + 1])
    run(lookback_hours=lookback_hours, dry_run=dry_run)
