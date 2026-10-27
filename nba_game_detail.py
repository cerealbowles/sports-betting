"""
nba_game_detail.py — NBA's thin wrapper around basketball_game_detail.py's
shared build() (NBA and WNBA are structurally identical — see that
module's docstring). This file only has what's genuinely NBA-specific:
find_game()'s schedule lookup and which sport modules to pass in.
"""
from datetime import datetime, timedelta

import basketball_game_detail as bgd
import game_detail_common as gdc
import nba_api
import nba_stats_db
import nba_boxscore_api
import nba_roster_api


def find_game(game_id):
    """Returns (game_dict, date_str) for an NBA game_id — tries today, then
    walks a few days back/forward (NBA has no "week" cache to lean on the
    way NFL does; a single-day schedule fetch is cheap enough to try a
    handful of candidates). Returns (None, None) if not found."""
    today = datetime.strptime(nba_api._today_et(), '%Y-%m-%d')
    for delta in (0, -1, 1, -2, 2, -3, 3):
        date_str = (today + timedelta(days=delta)).strftime('%Y-%m-%d')
        ctx = nba_api.build_schedule_context(target_date=date_str)
        for day in ctx:
            for g in day.get('games', []):
                if str(g.get('game_id')) == str(game_id):
                    gdc.apply_schedule_corrections('NBA', g, date_str)
                    return g, date_str
    return None, None


def build_game_detail(game_id):
    g, date_str = find_game(game_id)
    if not g:
        return None
    season = nba_api._get_nba_season(date_str)
    return bgd.build('NBA', 'nba', g, season, date_str, nba_stats_db, nba_boxscore_api, nba_roster_api)
