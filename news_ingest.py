"""
news_ingest.py — Sport-agnostic engine behind nfl_news.py and its siblings
(cfb_news.py, nhl_news.py, nba_news.py, wnba_news.py, mlb_news.py): fetch
news from ESPN's JSON/RSS feeds, match free-text articles to a player or
team via that sport's roster data, score with news_sentiment.score_text(),
and persist via nfl_stats_db.ingest_news_sentiment() (one shared
NewsSentiment table across sports — see that function's `sport` param).

Extracted from nfl_news.py once CFB/NHL/NBA/WNBA/MLB needed the exact same
fetch -> match -> score -> persist shape; nfl_news.py now configures this
module (its own ESPN_SITE/NEWS_SOURCES/roster module) and delegates to it
rather than duplicating the logic — see nfl_news.py for how a sport wires
itself up, and nfl_news_backfill.py for why build_indexes()/match_players()/
score_article() stayed as nfl_news.py-level wrappers (that module calls them
directly by name, confirmed via grep before this refactor).

match_players()'s confidence rules (exact full-name match always kept;
last-name-only kept only if exactly one candidate's team is also mentioned;
otherwise falls back to a team-only mention) are sport-agnostic by
construction — they only ever look at whatever name_index/team_index
build_indexes() built for that sport's own rosters.
"""
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

try:
    import feedparser
except ImportError:
    feedparser = None

import news_sentiment
import nfl_stats_db

_ET = ZoneInfo('America/New_York')
_SUFFIXES = re.compile(r'\b(jr|sr|ii|iii|iv)\.?\b', re.IGNORECASE)
_TAG = re.compile(r'<[^>]+>')


def today_et():
    return datetime.now(_ET).strftime('%Y-%m-%d')


def normalize(name):
    name = _SUFFIXES.sub('', name or '')
    name = re.sub(r'[^a-z0-9 ]', '', name.lower())
    return ' '.join(name.split())


def strip_html(text):
    return _TAG.sub('', text or '').strip()


# ── Fetch ────────────────────────────────────────────────────────────────

def fetch_espn_news(espn_site):
    """[{'url', 'headline', 'summary', 'published_at', 'athlete_ids'}].
    ESPN's news JSON has historically carried per-article `categories` with
    an athleteId for player-tagged stories — confirmed live for NFL, not
    relied on blindly: if present it's the highest-confidence match (skips
    text-matching entirely for that article), otherwise this falls back to
    the same text-matching every other source needs."""
    try:
        r = requests.get(f'{espn_site}/news', params={'limit': 50}, timeout=15)
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


def fetch_rss(url):
    """[{'url', 'headline', 'summary', 'published_at', 'athlete_ids'}].
    Fetches via `requests` with an explicit timeout first, THEN hands the
    bytes to feedparser — feedparser.parse(url) given a bare URL has no
    timeout of its own and was confirmed live to hang indefinitely against
    at least one NFL feed. feedparser still tolerates malformed feeds (sets
    .bozo rather than raising), so a feed that's drifted/half-broken yields
    whatever entries it can parse rather than failing the whole run."""
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
            'summary': strip_html(e.get('summary', '')),
            'published_at': published,
            'athlete_ids': [],
        })
    return [e for e in out if e['url'] and e['headline']]


def fetch_espn_teams(espn_site):
    """Raw ESPN team dicts (id, nickname, displayName, shortDisplayName,
    location) for every team in `espn_site`'s league — the source for both
    build_indexes()'s team_index (nickname/city -> team_id) and the team_id
    list it loops to pull rosters, since team_catalog.py only exposes
    displayName, not nickname."""
    try:
        r = requests.get(f'{espn_site}/teams', params={'limit': 350}, timeout=15)
        r.raise_for_status()
        leagues = (r.json().get('sports') or [{}])[0].get('leagues') or [{}]
        return [t['team'] for t in leagues[0].get('teams', []) if (t.get('team') or {}).get('id')]
    except Exception:
        return []


# ── Player/team matching ────────────────────────────────────────────────

def build_indexes(espn_site, roster_api, team_ids=None):
    """(name_index, team_index):
    - name_index: {normalized_full_name: [(player_id, team_id, player_name)]}
      plus {'last:<normalized_last_name>': [...]} entries for the lower-
      confidence last-name-only fallback match_players() uses.
    - team_index: {normalized_nickname_or_city_or_displayname: team_id}
    Rebuilt once per run — `roster_api`.get_roster() is already 24h-cached
    on every sport's roster module, so looping all of that league's teams
    here is cheap on a warm cache. `roster_api` must expose
    get_roster(team_id) -> [{'id', 'name'}, ...] (same shape
    nfl_roster_api/nba_roster_api/etc use).

    `team_ids` (optional): restricts which of fetch_espn_teams()'s teams
    get a roster pulled — cfb_news.py passes its FBS-only id set here,
    since ESPN's /teams endpoint has no server-side classification filter
    (confirmed live — 'groups'/'group' params are silently ignored there)
    and pulling rosters for every D1/D2/D3 program (800+) isn't workable
    on a cold cache. team_index still gets every team name regardless (a
    small-school team-only mention is still worth matching), only the
    roster/name_index build is restricted.

    Rosters are fetched in parallel (ThreadPoolExecutor) — same pattern as
    nba_roster_api.get_active_roster(), which found a full roster fetched
    sequentially took 23+ seconds per team; CFB's larger team count makes
    that multiply into minutes if done one at a time."""
    teams = fetch_espn_teams(espn_site)
    name_index = defaultdict(list)
    team_index = {}

    for t in teams:
        team_id = t['id']
        for key in (t.get('nickname'), t.get('shortDisplayName'), t.get('displayName'), t.get('location')):
            if key:
                team_index[normalize(key)] = team_id

    roster_teams = [t for t in teams if team_ids is None or t['id'] in team_ids]
    with ThreadPoolExecutor(max_workers=min(len(roster_teams), 16) or 1) as ex:
        futures = {ex.submit(roster_api.get_roster, t['id']): t['id'] for t in roster_teams}
        for f in as_completed(futures):
            team_id = futures[f]
            try:
                roster = f.result()
            except Exception:
                roster = []
            for p in roster:
                if not p.get('name'):
                    continue
                full = normalize(p['name'])
                if full:
                    name_index[full].append((p['id'], team_id, p['name']))
                last = p['name'].split()[-1] if p['name'].split() else None
                if last:
                    name_index[f'last:{normalize(last)}'].append((p['id'], team_id, p['name']))

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
    norm_text = normalize(text)
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


def matches_from_athlete_ids(athlete_ids, name_index):
    """Resolves ESPN's own athleteId tags (when present) straight to a
    team_id via name_index's roster data, skipping text-matching entirely —
    see fetch_espn_news()'s docstring."""
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

def score_article(text, sport):
    return news_sentiment.score_text(text, sport=sport)


def run(sport, espn_site, roster_api, news_sources, lookback_hours=48, dry_run=False, team_ids=None):
    """Shared ingestion loop for one sport: fetch every configured source,
    match, score, persist. `news_sources`: [{'name', 'kind', 'url'}, ...]
    same shape as nfl_news.NEWS_SOURCES — 'kind' is 'espn_json' or 'rss'.
    `team_ids`: see build_indexes()'s docstring."""
    name_index, team_index = build_indexes(espn_site, roster_api, team_ids=team_ids)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    today = today_et()

    totals = defaultdict(int)
    for src in news_sources:
        entries = fetch_espn_news(espn_site) if src['kind'] == 'espn_json' else fetch_rss(src['url'])
        totals[f"{src['name']}_fetched"] = len(entries)

        for e in entries:
            published = e.get('published_at')
            if published and published < cutoff:
                continue
            totals['fetched'] += 1

            text = f"{e.get('headline', '')}. {e.get('summary', '')}"
            matches = matches_from_athlete_ids(e.get('athlete_ids') or [], name_index)
            ambiguous = 0
            if not matches:
                matches, ambiguous = match_players(text, name_index, team_index)
            totals['ambiguous'] += ambiguous

            if not matches:
                totals['no_match'] += 1
                continue
            totals['matched'] += 1

            score, matched_terms = score_article(text, sport)
            if not dry_run:
                inserted = nfl_stats_db.ingest_news_sentiment(
                    source=src['name'], url=e['url'], headline=e['headline'],
                    raw_snippet=text.strip(), published_at=published, date_et=today,
                    matches=matches, score=score, matched_terms=matched_terms, sport=sport,
                )
                totals['inserted'] += inserted

    print(f"{sport} news ingestion {'(DRY RUN) ' if dry_run else ''}— lookback {lookback_hours}h")
    for src in news_sources:
        fetched_count = totals[src['name'] + '_fetched']
        print(f"  {src['name']:<16} fetched: {fetched_count}")
    print(f"  total fetched (within lookback): {totals['fetched']}")
    print(f"  matched: {totals['matched']}  ambiguous (skipped): {totals['ambiguous']}  "
          f"no match: {totals['no_match']}")
    if not dry_run:
        print(f"  inserted: {totals['inserted']}")
    return dict(totals)
