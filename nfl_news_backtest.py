"""
nfl_news_backtest.py — Does NFL player-news sentiment (nfl_news.py /
news_sentiment.py) actually correlate with real performance deviations?

For every NewsSentiment row matched to a specific player, finds that
player's next stored game on or after the article's date, computes
`performance_delta = actual_fantasy_points - trailing_average` (the
player's own rolling average over their games strictly before that one —
isolates "did coverage precede a deviation from this player's own normal"
rather than "did an already-good player play well"), and correlates
sentiment_score against performance_delta, bucketed by how many days
before the game the article ran.

There's no retroactive history — nfl_news.py only started accumulating
real data from whenever it first ran (same constraint InjuryStatus's
docstring documents) — so this will report "not enough data" until
nfl_news.py has run for at least several days/weeks. That's expected, not
a bug.

Run: python3 nfl_news_backtest.py [--min-articles N]
Output: a printed lag-bucket table, plus news_sentiment_backtest.json.
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import nfl_fantasy
import nfl_stats_db

LAG_BUCKETS = [(0, 1), (2, 3), (4, 7), (8, 9999)]


def _lag_bucket(days):
    for lo, hi in LAG_BUCKETS:
        if lo <= days <= hi:
            return f'{lo}-{hi if hi < 9999 else "+"}d'
    return 'unknown'


def _date(s):
    return datetime.strptime(s, '%Y-%m-%d')


def build_rows():
    """[{'player_id', 'sentiment_score', 'performance_delta', 'lag_days',
    'bucket', 'position'}, ...] — one row per (article, player) pair that
    has both a resolvable next game and a trailing average to compare
    against. See this module's docstring for why performance_delta is a
    delta against the player's own trailing average, not raw points."""
    from app import app as flask_app, NewsSentiment

    with flask_app.app_context():
        sentiment_rows = NewsSentiment.query.filter(
            NewsSentiment.sport == 'NFL', NewsSentiment.player_id.isnot(None)
        ).all()
        sentiment_rows = [{
            'player_id': r.player_id, 'date': r.date, 'sentiment_score': r.sentiment_score,
        } for r in sentiment_rows]

    by_player = defaultdict(list)
    for r in sentiment_rows:
        by_player[r['player_id']].append(r)

    rows = []
    for player_id, articles in by_player.items():
        gamelog = nfl_stats_db.get_player_gamelog_db(player_id)  # newest-first, all stored games
        if not gamelog:
            continue
        games_asc = sorted(gamelog, key=lambda g: g['date'])

        for art in articles:
            next_game = next((g for g in games_asc if g['date'] >= art['date']), None)
            if next_game is None:
                continue

            actual = nfl_fantasy.compute_fantasy_points(next_game['categories'])
            prior_games = nfl_stats_db.get_player_gamelog_db(player_id, before_date=next_game['date'])
            prior_points = [nfl_fantasy.compute_fantasy_points(g['categories']) for g in prior_games]
            trailing_avg = nfl_fantasy.rolling_average(prior_points)
            if trailing_avg is None:
                continue

            lag_days = max((_date(next_game['date']) - _date(art['date'])).days, 0)
            rows.append({
                'player_id': player_id,
                'position': next_game.get('position'),
                'sentiment_score': art['sentiment_score'],
                'performance_delta': actual - trailing_avg,
                'lag_days': lag_days,
                'bucket': _lag_bucket(lag_days),
            })
    return rows


def _pearson(xs, ys):
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def summarize(rows, min_articles=1):
    """{'by_lag_bucket': [...], 'by_surprise_slice': [...]}. Every slice
    reports n alongside its correlation — a correlation from a handful of
    rows is noise, not signal, and the report says so rather than hiding it."""
    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r['bucket']].append(r)

    by_lag_bucket = []
    for bucket in [f'{lo}-{hi if hi < 9999 else "+"}d' for lo, hi in LAG_BUCKETS]:
        bucket_rows = by_bucket.get(bucket, [])
        xs = [r['sentiment_score'] for r in bucket_rows]
        ys = [r['performance_delta'] for r in bucket_rows]
        by_lag_bucket.append({
            'bucket': bucket, 'n': len(bucket_rows),
            'correlation': _pearson(xs, ys) if len(bucket_rows) >= min_articles else None,
            'mean_score': float(np.mean(xs)) if xs else None,
            'mean_delta': float(np.mean(ys)) if ys else None,
        })

    # Sub-80th-percentile slice (see nfl_fantasy.SURPRISE_PERCENTILE_THRESHOLD):
    # approximated here as below-median trailing performance for that
    # player's position bucket across ALL rows, since reconstructing the
    # exact per-week field percentile a live pool would have shown
    # requires rebuilding that week's full slate — a documented
    # simplification, not a true per-week percentile (see this module's
    # docstring and the plan this was built from).
    positions = {r['position'] for r in rows}
    medians = {pos: float(np.median([r['performance_delta'] for r in rows if r['position'] == pos]))
               for pos in positions}

    slices = {'below_median': [], 'at_or_above_median': []}
    for r in rows:
        med = medians.get(r['position'], 0.0)
        key = 'below_median' if r['performance_delta'] < med else 'at_or_above_median'
        slices[key].append(r)

    by_surprise_slice = []
    for key, slice_rows in slices.items():
        xs = [r['sentiment_score'] for r in slice_rows]
        ys = [r['performance_delta'] for r in slice_rows]
        by_surprise_slice.append({
            'slice': key, 'n': len(slice_rows),
            'correlation': _pearson(xs, ys) if len(slice_rows) >= min_articles else None,
        })

    return {'by_lag_bucket': by_lag_bucket, 'by_surprise_slice': by_surprise_slice}


def run(min_articles=1):
    rows = build_rows()
    result = summarize(rows, min_articles=min_articles)

    print(f'NFL news-sentiment backtest — {len(rows)} (article, player) rows with a resolvable game\n')
    if not rows:
        print('  No usable rows yet. nfl_news.py needs to run for a while first — there is no')
        print('  retroactive article history, so this only gets real data from whenever')
        print('  ingestion first ran forward.')
    else:
        print('  By days-before-game lag bucket:')
        print(f"  {'bucket':<8}{'n':>6}{'correlation':>14}{'mean score':>14}{'mean delta':>14}")
        for b in result['by_lag_bucket']:
            corr = f"{b['correlation']:.3f}" if b['correlation'] is not None else 'n/a'
            mscore = f"{b['mean_score']:.3f}" if b['mean_score'] is not None else 'n/a'
            mdelta = f"{b['mean_delta']:.3f}" if b['mean_delta'] is not None else 'n/a'
            print(f"  {b['bucket']:<8}{b['n']:>6}{corr:>14}{mscore:>14}{mdelta:>14}")

        print('\n  By below/at-or-above this player\'s position-median performance delta')
        print('  (approximate stand-in for the Salary Cap Builder\'s sub-80th-percentile')
        print('  "surprise play" slice — see this module\'s docstring):')
        print(f"  {'slice':<22}{'n':>6}{'correlation':>14}")
        for s in result['by_surprise_slice']:
            corr = f"{s['correlation']:.3f}" if s['correlation'] is not None else 'n/a'
            print(f"  {s['slice']:<22}{s['n']:>6}{corr:>14}")

    out_path = os.path.join(_HERE, 'news_sentiment_backtest.json')
    with open(out_path, 'w') as f:
        json.dump({'row_count': len(rows), **result}, f, indent=2)
    print(f'\n  Full results written to {out_path}')
    return result


if __name__ == '__main__':
    min_articles = 1
    if '--min-articles' in sys.argv:
        idx = sys.argv.index('--min-articles')
        if idx + 1 < len(sys.argv):
            min_articles = int(sys.argv[idx + 1])
    run(min_articles=min_articles)
