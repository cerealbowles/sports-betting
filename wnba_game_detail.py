"""
wnba_game_detail.py — WNBA's thin wrapper around basketball_game_detail.py's
shared build() (NBA and WNBA are structurally identical — see that
module's docstring). This file only has what's genuinely WNBA-specific:
find_game()'s schedule lookup and which sport modules to pass in.

Note: WNBA's regular season runs May-Sept (playoffs into October) — this
will return (None, None) for every game id during the offseason, same as
wnba_api.build_schedule_context() returning [] for any date with no
scheduled games (see that function's own docstring).
"""
from datetime import datetime, timedelta

import basketball_game_detail as bgd
import wnba_api
import wnba_stats_db
import wnba_boxscore_api
import wnba_roster_api


def find_game(game_id):
    """Returns (game_dict, date_str) for a WNBA game_id — same date-window
    scan as nba_game_detail.find_game() (no week concept here either)."""
    today = datetime.strptime(wnba_api._today_et(), '%Y-%m-%d')
    for delta in (0, -1, 1, -2, 2, -3, 3):
        date_str = (today + timedelta(days=delta)).strftime('%Y-%m-%d')
        ctx = wnba_api.build_schedule_context(target_date=date_str)
        for day in ctx:
            for g in day.get('games', []):
                if str(g.get('game_id')) == str(game_id):
                    return g, date_str
    return None, None


def build_game_detail(game_id):
    g, date_str = find_game(game_id)
    if not g:
        return None
    season = wnba_api._get_wnba_season()
    return bgd.build('WNBA', 'wnba', g, season, date_str, wnba_stats_db, wnba_boxscore_api, wnba_roster_api)
