"""
nfl_news.py — Pulls NFL player/team news from RSS and JSON feeds, matches
each article to a player or team already known to this app (via
nfl_roster_api's rosters — there is no separate Player table), scores it
with news_sentiment.score_text(), and persists via
nfl_stats_db.ingest_news_sentiment().

Same fetch-JSON-via-requests -> parse -> persist shape as injuries_api.py/
weather_api.py/nfl_api.py. The one genuinely new piece of logic here (no
precedent anywhere in this app) is matching free-text article mentions to
a player_id — see match_players()'s docstring for the confidence rules.

Meant to run periodically purely to accumulate history going forward —
RSS/news endpoints are current-state-only, so there is no retroactive
backfill possible, same constraint InjuryStatus's docstring documents.
Wired into app.py's BackgroundScheduler as a daily 7am ET job
(_ingest_nfl_news) — see that job's docstring for the cron details.

Run: python3 nfl_news.py [--dry-run] [--lookback-hours N]
"""
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

try:
    import feedparser
except ImportError:
    feedparser = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import news_sentiment
import nfl_roster_api
import nfl_stats_db

_ET = ZoneInfo('America/New_York')
ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl'

# Plain dicts, not parsing logic — a dead/renamed feed URL should never
# require touching match_players()/score_article(). 'kind' picks which
# fetch function handles it. RSS URLs marked unverified were the best-
# known candidates at design time (see this feature's plan) — fantasy-
# site feed slugs drift, so run with --dry-run first to confirm each is
# still reachable before relying on it.
NEWS_SOURCES = [
    {'name': 'espn_json', 'kind': 'espn_json', 'url': f'{ESPN_SITE}/news'},
    {'name': 'espn_rss', 'kind': 'rss', 'url': 'https://www.espn.com/espn/rss/nfl/news'},
    {'name': 'nfl_rss', 'kind': 'rss', 'url': 'https://www.nfl.com/feeds/rss/news'},
    {'name': 'cbssports_rss', 'kind': 'rss', 'url': 'https://www.cbssports.com/rss/headlines/nfl/'},
    {'name': 'rotoballer_rss', 'kind': 'rss', 'url': 'https://www.rotoballer.com/nfl/feed'},
]

_SUFFIXES = re.compile(r'\b(jr|sr|ii|iii|iv)\.?\b', re.IGNORECASE)
_TAG = re.compile(r'<[^>]+>')


def _today_et():
    return datetime.now(_ET).strftime('%Y-%m-%d')


def _normalize(name):
    name = _SUFFIXES.sub('', name or '')
    name = re.sub(r'[^a-z0-9 ]', '', name.lower())
    return ' '.join(name.split())


def _strip_html(text):
    return _TAG.sub('', text or '').strip()


# ── Fetch ────────────────────────────────────────────────────────────────

def _fetch_espn_news():
    """[{'url', 'headline', 'summary', 'published_at'}]. ESPN's news JSON
    has historically carried per-article `categories` with an athleteId
    for player-tagged stories — confirmed live, not relied on blindly: if
    present it's the highest-confidence match (skips text-matching
    entirely for that article), otherwise this falls back to the same
    text-matching every other source needs."""
    try:
        r = requests.get(f'{ESPN_SITE}/news', params={'limit': 50}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    out = []
    for item in data.get('articles', []):
        published = None
        pub_str = item.get('published')
        if pub_str:
            try:
                published = datetime.fromisoformat(pub_str.replace('Z', '+00:00'))
            except ValueError:
                published = None
        athlete_ids = [str(c['athleteId']) for c in item.get('categories', [])
                       if c.get('type') == 'athlete' and c.get('athleteId')]
        out.append({
            'url': ((item.get('links') or {}).get('web') or {}).get('href', ''),
            'headline': item.get('headline', ''),
            'summary': item.get('description', ''),
            'published_at': published,
            'athlete_ids': athlete_ids,  # [] if ESPN didn't tag this one
        })
    return [e for e in out if e['url'] and e['headline']]


def _fetch_rss(url):
    """[{'url', 'headline', 'summary', 'published_at'}]. Fetches via
    `requests` with an explicit timeout first, THEN hands the bytes to
    feedparser — feedparser.parse(url) given a bare URL has no timeout of
    its own and was confirmed live to hang indefinitely against at least
    one of this module's configured feeds. feedparser still tolerates
    malformed feeds (sets .bozo rather than raising), so a feed that's
    drifted/half-broken yields whatever entries it can parse rather than
    failing the whole run."""
    if feedparser is None:
        return []
    try:
        r = requests.get(url, timeout=(5, 10), headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
    except Exception:
        return []
    try:
        parsed = feedparser.parse(r.content)
    except Exception:
        return []
    if not parsed.entries:
        return []

    out = []
    for e in parsed.entries:
        published = None
        if getattr(e, 'published_parsed', None):
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        out.append({
            'url': e.get('link', ''),
            'headline': e.get('title', ''),
            'summary': _strip_html(e.get('summary', '')),
            'published_at': published,
            'athlete_ids': [],
        })
    return [e for e in out if e['url'] and e['headline']]


def _fetch_espn_teams():
    """Raw ESPN team dicts (id, nickname, displayName, shortDisplayName,
    location) for all 32 teams — the source for both _build_team_index()
    (nickname/city -> team_id) and the team_id list _build_name_index()
    loops to pull rosters, since team_catalog.py only exposes displayName,
    not nickname."""
    try:
        r = requests.get(f'{ESPN_SITE}/teams', params={'limit': 50}, timeout=15)
        r.raise_for_status()
        leagues = (r.json().get('sports') or [{}])[0].get('leagues') or [{}]
        return [t['team'] for t in leagues[0].get('teams', []) if (t.get('team') or {}).get('id')]
    except Exception:
        return []


# ── Player/team matching ────────────────────────────────────────────────

def build_indexes():
    """(name_index, team_index):
    - name_index: {normalized_full_name: [(player_id, team_id, player_name)]}
      plus {'last:<normalized_last_name>': [...]} entries for the lower-
      confidence last-name-only fallback match() uses.
    - team_index: {normalized_nickname_or_city_or_displayname: team_id}
    Rebuilt once per run — nfl_roster_api.get_roster() is already
    24h-cached, so looping all 32 teams here is cheap."""
    teams = _fetch_espn_teams()
    name_index = defaultdict(list)
    team_index = {}

    for t in teams:
        team_id = t['id']
        for key in (t.get('nickname'), t.get('shortDisplayName'), t.get('displayName'), t.get('location')):
            if key:
                team_index[_normalize(key)] = team_id
        for p in nfl_roster_api.get_roster(team_id):
            if not p.get('name'):
                continue
            full = _normalize(p['name'])
            if full:
                name_index[full].append((p['id'], team_id, p['name']))
            last = p['name'].split()[-1] if p['name'].split() else None
            if last:
                name_index[f'last:{_normalize(last)}'].append((p['id'], team_id, p['name']))

    return name_index, team_index


def _team_mentioned(team_id, team_index, norm_text):
    return any(tid == team_id and re.search(r'\b' + re.escape(key) + r'\b', norm_text)
               for key, tid in team_index.items())


def match_players(text, name_index, team_index):
    """Matches free-text article content to players/teams already known
    via rosters. Confidence rules, deliberately conservative (never guess):
    - Exact normalized full-name match -> high confidence, always kept.
    - Last-name-only match -> only kept if exactly one full-name candidate
      for that last name also has their team's nickname/city mentioned in
      the same text (disambiguates "Jefferson" via "Vikings"); otherwise
      dropped and counted as ambiguous rather than guessed.
    - No player match at all -> falls back to a team-only mention
      (player_id=None), if any team name appears.
    Returns ([{'player_id', 'team_id', 'player_name'}, ...], ambiguous_count).
    """
    norm_text = _normalize(text)
    matches = []
    seen_ids = set()
    ambiguous = 0

    for key, entries in name_index.items():
        if key.startswith('last:'):
            continue
        if key and re.search(r'\b' + re.escape(key) + r'\b', norm_text):
            for pid, team_id, real_name in entries:
                if pid not in seen_ids:
                    matches.append({'player_id': pid, 'team_id': team_id, 'player_name': real_name})
                    seen_ids.add(pid)

    if not matches:
        for key, entries in name_index.items():
            if not key.startswith('last:'):
                continue
            last = key[len('last:'):]
            if not re.search(r'\b' + re.escape(last) + r'\b', norm_text):
                continue
            candidates = [e for e in entries if _team_mentioned(e[1], team_index, norm_text)]
            if len(candidates) == 1:
                pid, team_id, real_name = candidates[0]
                if pid not in seen_ids:
                    matches.append({'player_id': pid, 'team_id': team_id, 'player_name': real_name})
                    seen_ids.add(pid)
            elif len(entries) > 1:
                ambiguous += 1

    if not matches:
        for key, team_id in team_index.items():
            if re.search(r'\b' + re.escape(key) + r'\b', norm_text):
                matches.append({'player_id': None, 'team_id': team_id, 'player_name': None})
                break

    return matches, ambiguous


def _matches_from_athlete_ids(athlete_ids, name_index):
    """Resolves ESPN's own athleteId tags (when present) straight to a
    team_id via name_index's roster data, skipping text-matching
    entirely — see _fetch_espn_news()'s docstring."""
    by_id = {}
    for entries in name_index.values():
        for pid, team_id, real_name in entries:
            by_id[pid] = (team_id, real_name)
    out = []
    for aid in athlete_ids:
        hit = by_id.get(str(aid))
        if hit:
            team_id, real_name = hit
            out.append({'player_id': str(aid), 'team_id': team_id, 'player_name': real_name})
    return out


# ── Scoring + persistence ───────────────────────────────────────────────

def score_article(text):
    return news_sentiment.score_text(text, sport='NFL')


def run(lookback_hours=48, dry_run=False):
    name_index, team_index = build_indexes()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    today = _today_et()

    totals = defaultdict(int)
    for src in NEWS_SOURCES:
        entries = _fetch_espn_news() if src['kind'] == 'espn_json' else _fetch_rss(src['url'])
        totals[f"{src['name']}_fetched"] = len(entries)

        for e in entries:
            published = e.get('published_at')
            if published and published < cutoff:
                continue
            totals['fetched'] += 1

            text = f"{e.get('headline', '')}. {e.get('summary', '')}"
            matches = _matches_from_athlete_ids(e.get('athlete_ids') or [], name_index)
            ambiguous = 0
            if not matches:
                matches, ambiguous = match_players(text, name_index, team_index)
            totals['ambiguous'] += ambiguous

            if not matches:
                totals['no_match'] += 1
                continue
            totals['matched'] += 1

            score, matched_terms = score_article(text)
            if not dry_run:
                inserted = nfl_stats_db.ingest_news_sentiment(
                    source=src['name'], url=e['url'], headline=e['headline'],
                    raw_snippet=text.strip(), published_at=published, date_et=today,
                    matches=matches, score=score, matched_terms=matched_terms,
                )
                totals['inserted'] += inserted

    print(f"NFL news ingestion {'(DRY RUN) ' if dry_run else ''}— lookback {lookback_hours}h")
    for src in NEWS_SOURCES:
        fetched_count = totals[src['name'] + '_fetched']
        print(f"  {src['name']:<16} fetched: {fetched_count}")
    print(f"  total fetched (within lookback): {totals['fetched']}")
    print(f"  matched: {totals['matched']}  ambiguous (skipped): {totals['ambiguous']}  "
          f"no match: {totals['no_match']}")
    if not dry_run:
        print(f"  inserted: {totals['inserted']}")
    return dict(totals)


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    lookback_hours = 48
    if '--lookback-hours' in sys.argv:
        idx = sys.argv.index('--lookback-hours')
        if idx + 1 < len(sys.argv):
            lookback_hours = int(sys.argv[idx + 1])
    run(lookback_hours=lookback_hours, dry_run=dry_run)
