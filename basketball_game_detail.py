"""
basketball_game_detail.py — shared build_game_detail() body for NBA and
WNBA (nba_game_detail.py / wnba_game_detail.py), which are structurally
identical: same ensemble-model shape (team_only_prob/player_prob already
split), same bball_total_model.py total model, same PlayerGameStat/
TeamGameStat column names, same injury-lookup shape
(get_injury_statuses(event_id) -> {player_id: status}), same lack of a
"unit vs unit" matchup concept or travel/surface/weather data. Only the
per-sport module references and sport string differ — those come in via
the `mods` dict rather than importing nba_api/wnba_api etc. directly here,
so this module has no sport-specific import of its own.
"""
import game_math as gm
import game_detail_common as gdc
import spread_model


def league_rank_table(sport, stats_db, season, before_date):
    def _compute():
        teams = stats_db.get_teams_with_data_db(season)
        raw = {}
        for team_id, _name in teams:
            totals = stats_db.get_team_stat_totals_db(team_id, before_date=before_date, limit=82)
            if totals:
                raw[team_id] = totals

        lower_is_better = {'ppg_allowed', 'topg'}
        n_teams = len(raw)
        ranked = {tid: {} for tid in raw}
        for stat in ('ppg', 'ppg_allowed', 'fg_pct', 'three_pct', 'rpg', 'apg', 'topg'):
            vals = [(tid, v[stat]) for tid, v in raw.items() if v.get(stat) is not None]
            reverse = stat not in lower_is_better
            vals.sort(key=lambda x: x[1], reverse=reverse)
            for i, (tid, v) in enumerate(vals, start=1):
                ranked[tid][stat] = (v, i)
        return {'ranks': ranked, 'n_teams': n_teams}

    return gdc.cached_compute(f'{sport.lower()}_detail_ranks_{season}_{before_date}', 3600, _compute)


def recent_form(stats_db, season, team_name, before_date, n=5):
    game_log = stats_db.get_season_game_log_db(season)
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


def injuries_for_team(boxscore_api, roster_api, event_id, team_id):
    try:
        statuses = boxscore_api.get_injury_statuses(event_id)
    except Exception:
        return []
    if not statuses:
        return []
    try:
        roster = {p['id']: p['name'] for p in roster_api.get_roster(team_id)}
    except Exception:
        roster = {}
    rows = []
    for pid, status in statuses.items():
        name = roster.get(pid)
        if not name:
            continue
        rows.append({'status': status, 'name': name, 'pos': ''})
    return rows


def build(sport, short_sport, g, season, before_date, stats_db, boxscore_api, roster_api):
    """sport: 'NBA'/'WNBA' (matches spread_model.COEFFS/bball_total_model.
    CALIBRATION/GamePrediction.sport keys). short_sport: 'nba'/'wnba'
    (matches odds_api.SPORT_KEYS). before_date: the ET calendar date string
    this game is on, used as the "as of" cutoff for ranks/recent-form so a
    future game doesn't leak its own (nonexistent) result into either."""
    home, away = g['home'], g['away']
    model = g.get('model') or {}
    factors = model.get('factors') or []
    odds = g.get('odds') or {}

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
            'pick': f"{home['abbr'] if fav_home else away['abbr']} ML",
            'price': side_price,
            'mkt': f"{(home_novig if fav_home else away_novig) * 100:.1f}%",
            'model': f"{(p_blend if fav_home else 1 - p_blend) * 100:.1f}%",
            'edge': f"{edge_pts:+.1f} pts", 'ev': f"{ev * 100:+.1f}%",
            'kelly': f"{units:.1f}u", 'confidence': conf_ml,
            'why': (f"Blended model has {'home' if fav_home else 'away'} side at "
                    f"{(p_blend if fav_home else 1 - p_blend) * 100:.1f}% vs FanDuel's "
                    f"{(home_novig if fav_home else away_novig) * 100:.1f}% no-vig."),
        }

    spread_margin = spread_model.predict_margin(sport, factors)
    gap_sd_spread = gdc.market_gap_sd(sport, 'spread')
    spread_verdict = None
    home_spread_line = odds.get('home_spread')
    home_spread_price = odds.get('home_spread_price')
    away_spread_price = odds.get('away_spread_price')
    if spread_margin is not None and home_spread_line is not None and home_spread_price and away_spread_price:
        home_cover = spread_model.cover_prob(sport, factors, home_spread_line)
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
            model_side_abbr = home['abbr'] if spread_margin > 0 else away['abbr']
            model_side_line = -abs(round(spread_margin, 1))
            spread_verdict = {
                'market': 'SPREAD', 'tag': tag,
                'pick': f"{(home['abbr'] if fav_home else away['abbr'])} {'+' if side_line > 0 else ''}{side_line:g}",
                'price': side_price,
                'mkt': f"{home['abbr'] if home_spread_line < 0 else away['abbr']} {home_spread_line:g}",
                'model': f"{model_side_abbr} {model_side_line:g}",
                'edge': f"{abs(spread_margin - (-home_spread_line)):.1f} pts",
                'ev': f"{ev * 100:+.1f}%", 'kelly': ('—' if is_outlier else f"{units:.1f}u"),
                'why': ("Model is an outlier vs. its historical spread error — verify inputs before betting."
                        if is_outlier else
                        f"Model line {model_side_abbr} {model_side_line:g} vs market {home['abbr']} {home_spread_line:g}."),
            }

    verdicts = [v for v in (ml_verdict, spread_verdict) if v]

    rank_table = league_rank_table(sport, stats_db, season, before_date)
    ranks = rank_table['ranks']
    stat_rows = []
    for key, label in (('ppg', 'Points/game'), ('ppg_allowed', 'Points allowed'),
                        ('fg_pct', 'FG%'), ('three_pct', '3PT%'),
                        ('rpg', 'Rebounds/game'), ('apg', 'Assists/game'), ('topg', 'Turnovers/game')):
        a = ranks.get(away['id'], {}).get(key)
        h = ranks.get(home['id'], {}).get(key)
        if not a or not h:
            continue
        stat_rows.append({'label': label, 'away': {'value': a[0], 'rank': a[1]},
                           'home': {'value': h[0], 'rank': h[1]}, 'n_teams': rank_table['n_teams']})

    situation = {}
    if home.get('rest_days') is not None and away.get('rest_days') is not None:
        situation['rest'] = f"{away['rest_days']}d / {home['rest_days']}d"

    recent = {
        'away': recent_form(stats_db, season, away['name'], before_date),
        'home': recent_form(stats_db, season, home['name'], before_date),
    }

    injuries = {
        'away': injuries_for_team(boxscore_api, roster_api, g['game_id'], away.get('id')),
        'home': injuries_for_team(boxscore_api, roster_api, g['game_id'], home.get('id')),
    }

    history = gdc.line_movement(short_sport, home['name'], away['name'], g['game_time_utc'])

    return {
        'game': {
            'id': g['game_id'], 'league': sport, 'kickoff_utc': g['game_time_utc'],
            'venue': g.get('venue'), 'status': g.get('status'), 'weather': None,
        },
        'teams': {'away': away, 'home': home},
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
            'ats_record_when_edge': gdc.ats_record_when_edge(sport, 1.5),
        },
        'verdicts': verdicts,
        'matchups': [],   # no clean off-unit/def-unit pairing in basketball
        'stats': stat_rows,
        'recent': recent,
        'injuries': injuries,
        'situation': situation,
        'trends': [],
        'h2h': [],
    }
