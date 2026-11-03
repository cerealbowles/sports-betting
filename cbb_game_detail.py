"""
cbb_game_detail.py — Men's college basketball's thin wrapper around
basketball_game_detail.py's shared build() (same page NBA/WNBA use), plus
the CBB-only extras the template's _cbb_ratings_panel.html renders:
both teams' adjusted efficiency ratings and ESPN's BPI win probability.

find_game() first asks ESPN's summary for the event's tip time, so it only
has to build that one date's slate — scanning a ±3-day window like NBA's
find_game would mean building up to seven 100-game CBB slates.
"""
from datetime import datetime, timedelta

import basketball_game_detail as bgd
import cbb_api
import cbb_boxscore_api
import cbb_roster_api
import cbb_stats_db
import game_detail_common as gdc


def _event_date(game_id):
    data = cbb_boxscore_api._get(game_id) or {}
    comps = (data.get('header') or {}).get('competitions') or [{}]
    return cbb_api._event_date_et(comps[0].get('date', ''))


def find_game(game_id):
    """(game_dict, date_str) for a CBB event id, or (None, None)."""
    first = _event_date(game_id)
    today = datetime.strptime(cbb_api._today_et(), '%Y-%m-%d')
    candidates = [first] if first else []
    candidates += [(today + timedelta(days=d)).strftime('%Y-%m-%d') for d in (0, -1, 1)]
    seen = set()
    for date_str in candidates:
        if date_str in seen:
            continue
        seen.add(date_str)
        for day in cbb_api.build_schedule_context(target_date=date_str):
            for g in day.get('games', []):
                if str(g.get('game_id')) == str(game_id):
                    gdc.apply_schedule_corrections('CBB', g, date_str)
                    return g, date_str
    return None, None


def _rating_rows(g, date_str):
    """Opponent-adjusted efficiency rows for the Team Comparison card, in
    basketball_game_detail's stat-row shape, ranked nationally."""
    table = cbb_api.get_power_ratings(as_of=date_str)
    if not table:
        return []
    by_team = dict(table)
    n = len(table)
    rows = []
    for key, label, best_high in (('adj_em', 'Adj. efficiency margin', True),
                                  ('adj_o', 'Adj. offense (pts/100)', True),
                                  ('adj_d', 'Adj. defense (pts/100)', False)):
        order = sorted(by_team, key=lambda t: by_team[t][key], reverse=best_high)
        rank = {t: i for i, t in enumerate(order, 1)}
        a, h = g['away']['name'], g['home']['name']
        if a not in by_team or h not in by_team:
            return []
        rows.append({'label': label, 'n_teams': n,
                     'away': {'value': by_team[a][key], 'rank': rank[a]},
                     'home': {'value': by_team[h][key], 'rank': rank[h]}})
    return rows


def build_game_detail(game_id):
    g, date_str = find_game(game_id)
    if not g:
        return None
    season = cbb_api._get_cbb_season(date_str)
    detail = bgd.build('CBB', 'cbb', g, season, date_str, cbb_stats_db, cbb_boxscore_api, cbb_roster_api)
    detail['stats'] = _rating_rows(g, date_str) + (detail.get('stats') or [])
    situation = detail.setdefault('situation', {})
    situation['neutral'] = g.get('neutral')
    situation['conference_game'] = g.get('conference_game')
    detail['cbb'] = {
        'ratings': {'away': g['away'].get('ratings'), 'home': g['home'].get('ratings')},
        'projection': {k: (g.get('model') or {}).get(k)
                       for k in ('proj_home_pts', 'proj_away_pts', 'proj_poss', 'margin')},
        'bpi': cbb_boxscore_api.get_bpi_predictor(game_id),
        'neutral': g.get('neutral'),
        'game_type_label': g.get('game_type_label'),
    }
    return detail
