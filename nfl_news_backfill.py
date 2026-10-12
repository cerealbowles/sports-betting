"""
nfl_news_backfill.py — One-time historical backfill of NewsSentiment rows
via GDELT (the free, public, date-range-queryable global news database:
https://www.gdeltproject.org/), to give nfl_news_backtest.py real history
to correlate against immediately instead of waiting weeks for nfl_news.py's
daily 7am job to accumulate it forward from today.

Why this needs a different data source than nfl_news.py: confirmed live
before building this — ESPN's news JSON and the working RSS feeds (ESPN,
CBS Sports) are all rolling windows of roughly the last 24 hours, with no
date-range query parameter at all. There is nothing to "pull further back"
from them; that data was never retained anywhere queryable. GDELT's DOC
2.0 API (https://api.gdeltproject.org/api/v2/doc/doc), by contrast,
supports real startdatetime/enddatetime filtering years back — confirmed
live returning real dated August 2026 articles before this was built.

Known tradeoff vs. nfl_news.py's live pipeline: GDELT's ArtList mode gives
back {url, title, seendate, domain, ...} — a headline, not a headline +
summary. match_players() is run against the title alone here, so recall
is somewhat lower than the live pipeline's headline+summary matching —
acceptable for a one-time historical signal check, not worth the much
bigger lift of scraping full article bodies per URL.

Query shape: one GDELT query per team ("<Team Name>" NFL, quoted phrase +
league keyword to cut noise), restricted to TRUSTED_DOMAINS via GDELT's
domainis: operator — confirmed live before adding this that an
unrestricted query pulls from ~200 mostly low-quality domains (local
sports-radio affiliates, content aggregators) with ~42% of "distinct"
URLs actually being the same wire story reposted verbatim; filtering at
the query level avoids spending this script's already-scarce rate-limit
budget fetching content that would just get discarded anyway. Each
returned article's own `domain` field is also checked client-side
against the same list as a backstop, in case the operator doesn't
constrain results exactly as documented. Covers the whole backfill
window in one call per team, recursively halved if a response comes back
AT the 250-record cap (signals real coverage got truncated) down to a
1-day floor. Reuses nfl_news.py's build_indexes()/match_players()/
score_article() so a backfilled row is scored identically to a
live-ingested one — just tagged source='gdelt_backfill' and dated by the
article's own seendate (not "today", the live pipeline's convention) so
date-based joins/backtesting line up with when the news actually ran.
Also passes dedupe_by_headline=True to nfl_stats_db.ingest_news_sentiment
— even among trusted outlets, the same wire story can run on more than
one of them, and a plain (source, url, player_id) key can't catch that
since each repost is a genuinely distinct URL.

Rate-limited to GDELT's documented ask (≤1 request/5sec) and checkpointed
by team_id (nfl_news_backfill_checkpoint.json, gitignored-worthy scratch
state) so a stopped run resumes without re-fetching/re-scoring teams
already done — this is explicitly meant to be run over however long it
takes, not in one sitting.

Run: python3 nfl_news_backfill.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
     [--dry-run] [--reset]
Default --start: 7 days before this season's Week 1 (computed live from
ESPN's scoreboard unless overridden). Default --end: today (ET).
"""
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import nfl_news
import nfl_stats_db

_ET = ZoneInfo('America/New_York')
GDELT_DOC_API = 'https://api.gdeltproject.org/api/v2/doc/doc'

# National sports-media outlets only — confirmed live that an unrestricted
# GDELT query over this app's team/NFL terms pulls mostly from local
# sports-radio affiliate sites, hyperlocal papers, and content aggregators,
# not original reporting. Deliberately short and high-confidence rather
# than exhaustive: better to under-cover than let the radio-affiliate noise
# back in. Matched against GDELT's own `domain` field, which is the
# registrable domain (e.g. "espn.com"), not a full hostname.
TRUSTED_DOMAINS = [
    'espn.com', 'cbssports.com', 'nfl.com', 'si.com', 'bleacherreport.com',
    'nbcsports.com', 'foxsports.com', 'yahoo.com', 'usatoday.com',
    'apnews.com', 'theathletic.com', 'sportingnews.com', 'pff.com',
    'nytimes.com', 'washingtonpost.com',
]
_DOMAIN_QUERY_CLAUSE = '(' + ' OR '.join(f'domainis:{d}' for d in TRUSTED_DOMAINS) + ')'
RATE_LIMIT_SECONDS = 8             # GDELT asks for <=1 request/5sec, but confirmed live its real
                                    # throttling trips well after a 5.5s gap too — padded, plus
                                    # _gdelt_query()'s own 429 backoff loop as the real safety net
MAX_RECORDS = 250                 # GDELT DOC 2.0's per-query cap
MIN_SPLIT_HOURS = 24              # don't recursively split narrower than a day
MAX_SPLIT_DEPTH = 8               # safety bound, not expected to bind given the day floor
SOURCE_NAME = 'gdelt_backfill'
CHECKPOINT_PATH = os.path.join(_HERE, 'nfl_news_backfill_checkpoint.json')


def _domain_allowed(domain):
    """Client-side backstop for the query's own domainis: restriction, in
    case GDELT doesn't constrain results exactly as documented. GDELT's
    `domain` field is normally already the bare registrable domain (e.g.
    'espn.com'), but strips a leading 'www.' and checks by suffix anyway
    (catches a subdomain like 'sports.yahoo.com') rather than assuming
    that's guaranteed."""
    d = (domain or '').lower()
    if d.startswith('www.'):
        d = d[4:]
    return any(d == td or d.endswith('.' + td) for td in TRUSTED_DOMAINS)


def _fmt_gdelt(dt):
    return dt.strftime('%Y%m%d%H%M%S')


def _parse_seendate(s):
    """GDELT's seendate format: 'YYYYMMDDTHHMMSSZ'."""
    try:
        return datetime.strptime(s, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _gdelt_query(query, start_dt, end_dt):
    """One GDELT DOC 2.0 ArtList query -> [{'url','title','seendate'},
    ...]. GDELT's documented policy is <=1 request/5sec, but confirmed
    live that a single request can still come back 429 well after that —
    its real throttling is stricter/statefully-windowed than the docs
    say, not a fixed per-request interval. A 429 is retried with
    increasing backoff (20s/45s/90s/180s/300s) rather than treated as "zero
    articles" — silently swallowing it would corrupt the backfill with
    teams that look like they had no news purely because of throttling.
    Only gives up (returns []) after exhausting retries or on a non-429
    failure (network blip, malformed response, a genuine non-200 that
    isn't rate-limiting) — a single bad window should never kill the
    whole backfill, it just loses that slice."""
    params = {
        'query': query, 'mode': 'ArtList', 'maxrecords': MAX_RECORDS, 'format': 'json',
        'startdatetime': _fmt_gdelt(start_dt), 'enddatetime': _fmt_gdelt(end_dt),
    }
    backoffs = [20, 45, 90, 180, 300]  # up to 5 min — confirmed live that GDELT's real throttling
                                        # window outlasts a few short retries; this is meant to run
                                        # unattended over a long stretch, so patience costs nothing
    attempt = 0
    while True:
        try:
            r = requests.get(GDELT_DOC_API, params=params, timeout=30)
        except Exception as e:
            print(f"    GDELT request error ({e}) — giving up on this window")
            return []

        if r.status_code == 200:
            time.sleep(RATE_LIMIT_SECONDS)
            try:
                return r.json().get('articles') or []
            except ValueError:
                return []

        if r.status_code == 429 and attempt < len(backoffs):
            wait = backoffs[attempt]
            print(f"    GDELT rate-limited (429) — backing off {wait}s (attempt {attempt + 1}/{len(backoffs)})")
            time.sleep(wait)
            attempt += 1
            continue

        print(f"    GDELT returned {r.status_code} — giving up on this window")
        time.sleep(RATE_LIMIT_SECONDS)
        return []


def _fetch_team_window(query, start_dt, end_dt, depth=0):
    """Recursively halves the window if truncated at MAX_RECORDS (real
    coverage exceeded what one query could return), down to a 1-day floor
    or MAX_SPLIT_DEPTH, whichever binds first — over a ~6-week backfill
    window that floor caps recursion at ~6 levels regardless, so
    MAX_SPLIT_DEPTH is just a defensive bound, not expected to trigger."""
    articles = _gdelt_query(query, start_dt, end_dt)
    span = end_dt - start_dt
    if len(articles) >= MAX_RECORDS and span > timedelta(hours=MIN_SPLIT_HOURS) and depth < MAX_SPLIT_DEPTH:
        mid = start_dt + span / 2
        return (_fetch_team_window(query, start_dt, mid, depth + 1)
                + _fetch_team_window(query, mid, end_dt, depth + 1))
    return articles


def _week1_start_et(season=None):
    """ET calendar date of Week 1's earliest kickoff — used only to
    compute this script's default --start (7 days before). Live ESPN
    lookup (not nfl_api's cached version — this script has no Flask app
    context) so the default stays correct in future seasons without
    editing this file; returns None on any failure, caller must handle."""
    try:
        params = {'seasontype': 2, 'week': 1}
        if season is not None:
            params.update({'season': season, 'dates': season})
        r = requests.get(f'{nfl_news.ESPN_SITE}/scoreboard', params=params, timeout=15)
        events = r.json().get('events', [])
        dates = [e.get('date') for e in events if e.get('date')]
        if not dates:
            return None
        earliest = min(datetime.fromisoformat(d.replace('Z', '+00:00')) for d in dates)
        return earliest.astimezone(_ET).date()
    except Exception:
        return None


def _load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {'done_team_ids': []}


def _save_checkpoint(done_team_ids):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump({'done_team_ids': sorted(done_team_ids)}, f)


def run(start_date, end_date, dry_run=False, reset=False):
    """start_date/end_date: 'YYYY-MM-DD' ET calendar dates, inclusive.
    One GDELT query (whole window, recursively sub-divided as needed) per
    team. Checkpointed by team_id in CHECKPOINT_PATH — a dry run never
    advances the checkpoint, since nothing was actually persisted."""
    if reset and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    done = set(_load_checkpoint().get('done_team_ids', []))

    start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) + timedelta(days=1)

    print(f"NFL news backfill {'(DRY RUN) ' if dry_run else ''}— {start_date} through {end_date} "
          f"(ET), via GDELT, {len(done)} team(s) already checkpointed done")

    name_index, team_index = nfl_news.build_indexes()
    teams = nfl_news._fetch_espn_teams()
    if not teams:
        print("Couldn't fetch the team list from ESPN — aborting.")
        return

    totals = defaultdict(int)
    for t in teams:
        team_id, display_name = t.get('id'), t.get('displayName')
        if not team_id or not display_name:
            continue
        if team_id in done:
            print(f"  [{display_name}] already done, skipping")
            continue

        query = f'"{display_name}" NFL {_DOMAIN_QUERY_CLAUSE}'
        articles = _fetch_team_window(query, start_dt, end_dt)

        seen_urls = set()
        team_totals = defaultdict(int)
        for a in articles:
            url, title = a.get('url'), a.get('title')
            if not url or not title or url in seen_urls:
                continue
            seen_urls.add(url)
            if not _domain_allowed(a.get('domain')):
                team_totals['off_domain_skipped'] += 1
                continue
            team_totals['fetched'] += 1

            published = _parse_seendate(a.get('seendate'))
            date_et = (published.astimezone(_ET) if published else datetime.now(_ET)).strftime('%Y-%m-%d')

            matches, ambiguous = nfl_news.match_players(title, name_index, team_index)
            team_totals['ambiguous'] += ambiguous
            if not matches:
                team_totals['no_match'] += 1
                continue
            team_totals['matched'] += 1

            score, matched_terms = nfl_news.score_article(title)
            if not dry_run:
                inserted = nfl_stats_db.ingest_news_sentiment(
                    source=SOURCE_NAME, url=url, headline=title,
                    raw_snippet=title, published_at=published, date_et=date_et,
                    matches=matches, score=score, matched_terms=matched_terms,
                    dedupe_by_headline=True,
                )
                team_totals['inserted'] += inserted

        print(f"  [{display_name}] fetched {team_totals['fetched']} (skipped {team_totals['off_domain_skipped']} off-list)  "
              f"matched {team_totals['matched']}  ambiguous {team_totals['ambiguous']}  no_match {team_totals['no_match']}"
              + ('' if dry_run else f"  inserted {team_totals['inserted']}"))
        for k, v in team_totals.items():
            totals[k] += v

        if not dry_run:
            done.add(team_id)
            _save_checkpoint(done)

    print(f"\nBackfill {'(DRY RUN) ' if dry_run else ''}complete: {len(done)}/{len(teams)} teams checkpointed done")
    print(f"  total fetched: {totals['fetched']}  matched: {totals['matched']}  "
          f"ambiguous (skipped): {totals['ambiguous']}  no match: {totals['no_match']}")
    if not dry_run:
        print(f"  inserted: {totals['inserted']}")


def clean_existing(dry_run=False):
    """One-time cleanup over already-stored gdelt_backfill rows — for rows
    inserted before TRUSTED_DOMAINS/dedupe_by_headline existed, since new
    ingestion is already filtered/deduped going in and doesn't need this.
    Two passes: (1) removes rows whose URL's domain isn't in
    TRUSTED_DOMAINS, (2) among what's left, removes syndicated-duplicate
    rows (same player-or-team + normalized headline), keeping the
    earliest-dated copy of each."""
    from urllib.parse import urlparse
    from app import app as flask_app, db, NewsSentiment

    with flask_app.app_context():
        rows = NewsSentiment.query.filter_by(source=SOURCE_NAME).order_by(
            NewsSentiment.date.asc(), NewsSentiment.created_at.asc()).all()
        total = len(rows)

        off_domain = [r for r in rows if not _domain_allowed(urlparse(r.url).netloc)]
        off_domain_ids = {r.id for r in off_domain}
        kept = [r for r in rows if r.id not in off_domain_ids]

        seen = {}
        dup = []
        for r in kept:
            key = (r.player_id, r.team_id, nfl_stats_db._normalize_headline(r.headline))
            if key in seen:
                dup.append(r)
            else:
                seen[key] = r

        print(f"gdelt_backfill: {total} rows total")
        print(f"  off-trusted-domain: {len(off_domain)}")
        print(f"  syndicated duplicates (among the rest): {len(dup)}")
        print(f"  would remain: {total - len(off_domain) - len(dup)}")

        if not dry_run:
            for r in off_domain + dup:
                db.session.delete(r)
            db.session.commit()
            print("Deleted.")
    return len(off_domain), len(dup)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', default=None, help="YYYY-MM-DD (ET) — default: 7 days before this season's Week 1")
    parser.add_argument('--end', default=None, help='YYYY-MM-DD (ET) — default: today')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--reset', action='store_true', help='Clear the checkpoint and start over from the first team')
    parser.add_argument('--clean-existing', action='store_true',
                         help='Remove already-stored off-trusted-domain/duplicate rows and exit (no fetching)')
    args = parser.parse_args()

    if args.clean_existing:
        clean_existing(dry_run=args.dry_run)
        sys.exit(0)

    start = args.start
    if not start:
        week1 = _week1_start_et()
        if week1 is None:
            print("Couldn't determine Week 1's date live — pass --start YYYY-MM-DD explicitly.")
            sys.exit(1)
        start = (week1 - timedelta(days=7)).strftime('%Y-%m-%d')
    end = args.end or datetime.now(_ET).strftime('%Y-%m-%d')

    run(start, end, dry_run=args.dry_run, reset=args.reset)
