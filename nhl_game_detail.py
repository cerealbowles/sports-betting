"""
nhl_game_detail.py — Data assembly for the NHL game-detail page
(/nhl/game/<game_id>), same contract/approach as nfl_game_detail.py (read
that module's docstring first).

NHL is the most structurally different of the three sports ported after
NFL (confirmed against the live codebase before writing this, not
assumed):
  - No week concept — single ET calendar date schedule
    (nhl_api.build_schedule_context(target_date=...)); find_game() scans a
    date window like nba_game_detail.py/wnba_game_detail.py do.
  - Team dicts use 'abbrev', not 'abbr' (no alias key the way NFL/NBA/WNBA
    have both) — and crucially have NO 'id' field at all. NHL's own
    schedule API (nhl_api.py's NHL_API) is a different id space than the
    ESPN ids the local stats warehouse (nhl_stats_db.py) and injury
    pipeline are keyed by; nhl_api.py bridges the two internally via
    _get_espn_event_map() (abbrev -> ESPN team id) at its one call site
    that needs both. This module re-derives that same bridge for its own
    ranks/injuries lookups.
  - hockey_total_model.predict_total() takes four plain gf_pg/ga_pg
    scalars directly (no sport/team-id params the way football's/
    basketball's total models do).
  - Stats warehouse is thin: TeamGameStat only has points/points_allowed
    (goals for/against) for NHL — no shots/PP%/faceoff% columns, and
    PlayerGameStat stores NHL's per-player line as opaque stats_json, not
    flat columns, so "Team comparison" here is just goals-for/against per
    game, ranked — real, just narrower than NFL/NBA's version.
  - No "unit vs unit" matchup concept (same as basketball) — matchups
    stays [].
  - Injuries: real per-player status/name/position already exists via
    nhl_stats_db.get_latest_injury_rows() (richer than NBA's path — no
    separate roster-name lookup needed).
  - No travel/surface/weather — situation only has 'rest'.
"""
from datetime import datetime, timedelta

import game_math as gm
import game_detail_common as gdc
import nhl_api
import nhl_stats_db
import nhl_roster_api
import spread_model


def find_game(game_id):
    """Returns (game_dict, date_str) for an NHL game_id — date-window scan,
    same approach as nba_game_detail.py/wnba_game_detail.py (NHL has no
    week concept either)."""
    today = datetime.strptime(nhl_api._today_et(), '%Y-%m-%d')
    for delta in (0, -1, 1, -2, 2, -3, 3):
        date_str = (today + timedelta(days=delta)).strftime('%Y-%m-%d')
        ctx = nhl_api.build_schedule_context(target_date=date_str)
        for day in ctx:
            for g in day.get('games', []):
                if str(g.get('game_id')) == str(game_id):
                    return g, date_str
    return None, None


def _espn_ids(home_abbrev, away_abbrev, date_str):
    """Bridges NHL's own abbrevs to the ESPN team ids the stats warehouse
    is keyed by (see module docstring) — (home_espn_id, away_espn_id), or
    (None, None) if ESPN's scoreboard has nothing for this date/matchup."""
    espn_map = nhl_api._get_espn_event_map(date_str)
    info = espn_map.get(frozenset({home_abbrev, away_abbrev}))
    if not info:
        return None, None
    return info.get('home_id'), info.get('away_id')


def _league_rank_table(season, before_date):
    def _compute():
        teams = nhl_stats_db.get_teams_with_data_db(season)
        raw = {}
        for team_id, _name in teams:
            totals = nhl_stats_db.get_team_goal_totals_db(team_id, before_date=before_date, limit=82)
            if totals:
                raw[team_id] = totals

        lower_is_better = {'ga_pg'}
        n_teams = len(raw)
        ranked = {tid: {} for tid in raw}
        for stat in ('gf_pg', 'ga_pg'):
            vals = [(tid, v[stat]) for tid, v in raw.items() if v.get(stat) is not None]
            reverse = stat not in lower_is_better
            vals.sort(key=lambda x: x[1], reverse=reverse)
            for i, (tid, v) in enumerate(vals, start=1):
                ranked[tid][stat] = (v, i)
        return {'ranks': ranked, 'n_teams': n_teams}

    return gdc.cached_compute(f'nhl_detail_ranks_{season}_{before_date}', 3600, _compute)


def _recent_form(season, team_name, before_date, n=5):
    game_log = nhl_stats_db.get_season_game_log_db(season)
    games = []
    for g in game_log:
        if g['game_date'] >= before_date:
            continue
        if g['home_name'] == team_name:
            games.append((g['game_date'], True, g['away_name'], g['home_score'], g['away_score'], g['home_won']))
        elif g['away_name'] == team_name:
            games.append((g['game_date'], False, g['home_name'], g['away_score'], g['home_score'], not g['home_won']))
    games.sort(key=lambda x: x[0], reverse=True)

    rows = []
    for game_date, is_home, opp, own_score, opp_score, won in games[:n]:
        rows.append({
            'result': 'W' if won else 'L',
            'opp': f"{'vs' if is_home else '@'} {opp}",
            'score': f"{int(own_score)}-{int(opp_score)}" if own_score is not None and opp_score is not None else None,
            'ats': None, 'ou': None,
        })
    return rows


def build_game_detail(game_id):
    g, date_str = find_game(game_id)
    if not g:
        return None

    home, away = g['home'], g['away']
    home_abbrev, away_abbrev = home.get('abbrev', ''), away.get('abbrev', '')
    model = g.get('model') or {}
    factors = model.get('factors') or []
    odds = g.get('odds') or {}
    season = nhl_api._get_nhl_season_start_year()

    p_team = model.get('team_only_prob')
    p_player = model.get('player_prob')
    p_blend = model.get('home_prob')
    conf_ml = gm.confidence(p_team, p_player) if p_team is not None and p_player is not None else None

    home_novig = away_novig = hold_ml = None
    if odds.get('home_best') and odds.get('away_best'):
        imp_h = gm.implied_prob(odds['home_best'])
        imp_a = gm.implied_prob(odds['away_best'])
        home_novig, away_novig, hold_ml = gm.no_vig_pair(imp_h, imp_a)

    ml_verdict = None
    if p_blend is not None and home_novig is not None:
        home_edge = p_blend - home_novig
        away_edge = (1 - p_blend) - away_novig
        fav_home = home_edge >= away_edge
        edge_pts = (home_edge if fav_home else away_edge) * 100
        side_price = odds['home_best'] if fav_home else odds['away_best']
        b = gm.net_payout_b(side_price)
        ev = gm.ev(p_blend if fav_home else (1 - p_blend), b)
        units = gm.kelly_units(p_blend if fav_home else (1 - p_blend), b)
        tag = gm.tag_for(ev, conf_ml or 1, is_outlier=False)
        ml_verdict = {
            'market': 'MONEYLINE', 'tag': tag,
            'pick': f"{home_abbrev if fav_home else away_abbrev} ML",
            'price': side_price,
            'mkt': f"{(home_novig if fav_home else away_novig) * 100:.1f}%",
            'model': f"{(p_blend if fav_home else 1 - p_blend) * 100:.1f}%",
            'edge': f"{edge_pts:+.1f} pts", 'ev': f"{ev * 100:+.1f}%",
            'kelly': f"{units:.1f}u", 'confidence': conf_ml,
            'why': (f"Blended model has {'home' if fav_home else 'away'} side at "
                    f"{(p_blend if fav_home else 1 - p_blend) * 100:.1f}% vs FanDuel's "
                    f"{(home_novig if fav_home else away_novig) * 100:.1f}% no-vig."),
            'gate_note': gdc.recommendation_gate_note('NHL', 'MONEYLINE', tag, amer_odds=side_price, edge_pts=edge_pts),
        }

    spread_margin = spread_model.predict_margin('NHL', factors)
    gap_sd_spread = gdc.market_gap_sd('NHL', 'spread')
    spread_verdict = None
    home_spread_line = odds.get('home_spread')
    home_spread_price = odds.get('home_spread_price')
    away_spread_price = odds.get('away_spread_price')
    if spread_margin is not None and home_spread_line is not None and home_spread_price and away_spread_price:
        home_cover = spread_model.cover_prob('NHL', factors, home_spread_line)
        away_cover = (1 - home_cover) if home_cover is not None else None
        ev_home = gm.ev(home_cover, gm.net_payout_b(home_spread_price)) if home_cover is not None else None
        ev_away = gm.ev(away_cover, gm.net_payout_b(away_spread_price)) if away_cover is not None else None
        if ev_home is not None and ev_away is not None:
            fav_home = ev_home >= ev_away
            side_cover = home_cover if fav_home else away_cover
            side_line = home_spread_line if fav_home else -home_spread_line
            side_price = home_spread_price if fav_home else away_spread_price
            ev = ev_home if fav_home else ev_away
            is_outlier = gm.outlier_guard(spread_margin, -home_spread_line, gap_sd_spread) if gap_sd_spread else False
            units = 0.0 if is_outlier else gm.kelly_units(side_cover, gm.net_payout_b(side_price))
            tag = gm.tag_for(ev, 4, is_outlier)
            model_side_abbr = home_abbrev if spread_margin > 0 else away_abbrev
            model_side_line = -abs(round(spread_margin, 1))
            spread_verdict = {
                'market': 'SPREAD', 'tag': tag,
                'pick': f"{(home_abbrev if fav_home else away_abbrev)} {'+' if side_line > 0 else ''}{side_line:g}",
                'price': side_price,
                'mkt': f"{home_abbrev if home_spread_line < 0 else away_abbrev} {home_spread_line:g}",
                'model': f"{model_side_abbr} {model_side_line:g}",
                'edge': f"{abs(spread_margin - (-home_spread_line)):.1f} pts",
                'ev': f"{ev * 100:+.1f}%", 'kelly': ('—' if is_outlier else f"{units:.1f}u"),
                'why': ("Model is an outlier vs. its historical spread error — verify inputs before betting."
                        if is_outlier else
                        f"Model line {model_side_abbr} {model_side_line:g} vs market {home_abbrev} {home_spread_line:g}."),
                'gate_note': gdc.recommendation_gate_note('NHL', 'SPREAD', tag),
            }

    verdicts = [v for v in (ml_verdict, spread_verdict) if v]

    home_espn_id, away_espn_id = _espn_ids(home_abbrev, away_abbrev, date_str)

    stat_rows = []
    if home_espn_id and away_espn_id:
        rank_table = _league_rank_table(season, date_str)
        ranks = rank_table['ranks']
        for key, label in (('gf_pg', 'Goals/game'), ('ga_pg', 'Goals allowed/game')):
            a = ranks.get(away_espn_id, {}).get(key)
            h = ranks.get(home_espn_id, {}).get(key)
            if not a or not h:
                continue
            stat_rows.append({'label': label, 'away': {'value': a[0], 'rank': a[1]},
                               'home': {'value': h[0], 'rank': h[1]}, 'n_teams': rank_table['n_teams']})

    situation = {}
    if home.get('rest_days') is not None and away.get('rest_days') is not None:
        situation['rest'] = f"{away['rest_days']}d / {home['rest_days']}d"

    recent = {
        'away': _recent_form(season, away['name'], date_str),
        'home': _recent_form(season, home['name'], date_str),
    }

    injuries = {'away': [], 'home': []}
    if home_espn_id and away_espn_id:
        try:
            injuries = {
                'away': [{'id': r['id'], 'status': r['status'], 'name': r['name'], 'pos': r['pos'] or ''}
                         for r in nhl_stats_db.get_latest_injury_rows(away_espn_id, date_str)],
                'home': [{'id': r['id'], 'status': r['status'], 'name': r['name'], 'pos': r['pos'] or ''}
                         for r in nhl_stats_db.get_latest_injury_rows(home_espn_id, date_str)],
            }
        except Exception:
            pass
    # The goalie's own injury status is already real and already on the
    # team dict (nhl_api.py's build_schedule_context sets
    # team['goalie']['injury']) — surface it even if the skater-level
    # lookup above found nothing. This goalie dict already carries its own
    # sv_pct/gaa/record (from the NHL API, keyed by abbrev, no ESPN id
    # available here), so its "season" line is built from that directly
    # instead of going through _attach_season_stats()'s ESPN gamelog path.
    for side, team in (('away', away), ('home', home)):
        goalie = team.get('goalie')
        if goalie and goalie.get('injury'):
            w, l, otl = (goalie.get('record') or '0-0-0').split('-')[:3] if goalie.get('record') else ('0', '0', '0')
            injuries[side].insert(0, {
                'status': goalie['injury'], 'name': goalie.get('name', 'Goalie'), 'pos': 'G',
                'season': {
                    'is_goalie': True, 'wins': int(w) if w.isdigit() else 0,
                    'losses': int(l) if l.isdigit() else 0,
                    'otl': int(otl) if otl.isdigit() else 0,
                    'save_pct': float(goalie['sv_pct']) if goalie.get('sv_pct') else None,
                    'gaa': float(goalie['gaa']) if goalie.get('gaa') else None,
                    'games': None, 'starts': None, 'label': 'this season', 'is_current': True,
                },
            })

    # Real per-player season stats (G/A/P for skaters, record/SV%/GAA for
    # goalies) from ESPN's gamelog — turns an unfamiliar name into a real
    # sense of how much the team is missing, then sorts each team's list by
    # that contribution (points for skaters, starts for goalies) so the
    # players who matter most surface first instead of alphabetical/feed order.
    for side in ('away', 'home'):
        for p in injuries[side]:
            if 'season' not in p and p.get('id'):
                try:
                    p['season'] = nhl_roster_api.get_player_season_summary(p['id'], p['pos'])
                except Exception:
                    p['season'] = None

        def _impact(p):
            s = p.get('season')
            if not s:
                return -1
            if s.get('is_goalie'):
                return (s.get('starts') or 0) * 3 + (s.get('wins') or 0)
            return s.get('points', 0)

        injuries[side].sort(key=_impact, reverse=True)

    history = gdc.line_movement('nhl', home['name'], away['name'], g['game_time_utc'])

    return {
        'game': {
            'id': g['game_id'], 'league': 'NHL', 'kickoff_utc': g['game_time_utc'],
            'venue': g.get('venue'), 'status': g.get('status'), 'weather': None,
        },
        'teams': {
            'away': {**away, 'abbr': away_abbrev},
            'home': {**home, 'abbr': home_abbrev},
        },
        'fanduel': {
            'moneyline': {'away': {'price': odds.get('away_best'), 'open': odds.get('opening_away')},
                           'home': {'price': odds.get('home_best'), 'open': odds.get('opening_home')}},
            'spread': {'away': {'line': -home_spread_line if home_spread_line is not None else None,
                                  'price': odds.get('away_spread_price')},
                        'home': {'line': home_spread_line, 'price': odds.get('home_spread_price')}},
            'hold_ml': round(hold_ml * 100, 1) if hold_ml is not None else None,
            'novig_home': round(home_novig * 100, 1) if home_novig is not None else None,
            'novig_away': round(away_novig * 100, 1) if away_novig is not None else None,
            'history': history,
        },
        'model': {
            'p_team_home': p_team, 'p_player_home': p_player, 'p_blend_home': p_blend,
            'spread_home': -spread_margin if spread_margin is not None else None,
            'confidence': conf_ml,
            'factors': factors,
            'ats_record_when_edge': gdc.ats_record_when_edge('NHL', 1.5),
        },
        'verdicts': verdicts,
        'matchups': [],
        'stats': stat_rows,
        'recent': recent,
        'injuries': injuries,
        'situation': situation,
        'trends': [],
        'h2h': [],
    }
