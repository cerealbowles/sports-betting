"""
cfb_news.py — CFB player/team news ingestion. Thin config layer over
news_ingest.py (the shared fetch -> match -> score -> persist engine
originally built for nfl_news.py) — see that module's docstring for the
mechanics, and nfl_news.py for the sibling this was split from.

~130 FBS teams vs NFL's 32 means far more name collisions across rosters
(see cfb_roster_api.py's docstring) — news_ingest.match_players()'s
last-name-only fallback will skip far more ambiguous matches here than it
does for NFL, by design (it never guesses).

Wired into app.py's BackgroundScheduler — see _ingest_cfb_news's docstring
for the cron details.

Run: python3 cfb_news.py [--dry-run] [--lookback-hours N]
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import news_ingest
import cfb_roster_api

ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football'

NEWS_SOURCES = [
    {'name': 'espn_json', 'kind': 'espn_json', 'url': f'{ESPN_SITE}/news'},
    {'name': 'espn_rss', 'kind': 'rss', 'url': 'https://www.espn.com/espn/rss/ncf/news'},
    {'name': 'cbssports_rss', 'kind': 'rss', 'url': 'https://www.cbssports.com/rss/headlines/college-football/'},
]


def run(lookback_hours=48, dry_run=False):
    fbs_ids = cfb_roster_api.get_fbs_team_ids() or None
    return news_ingest.run('CFB', ESPN_SITE, cfb_roster_api, NEWS_SOURCES,
                            lookback_hours=lookback_hours, dry_run=dry_run, team_ids=fbs_ids)


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    lookback_hours = 48
    if '--lookback-hours' in sys.argv:
        idx = sys.argv.index('--lookback-hours')
        if idx + 1 < len(sys.argv):
            lookback_hours = int(sys.argv[idx + 1])
    run(lookback_hours=lookback_hours, dry_run=dry_run)
