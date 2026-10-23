"""
nfl_game_detail.py — Data assembly for the NFL game-detail page
(/nfl/game/<game_id>), matching the design handoff's data contract
(game-page-handoff/HANDOFF.md section 3) as closely as real data allows.

Builds on top of nfl_api._build_game()'s already-assembled game dict (same
one the schedule page's game card renders from) rather than re-deriving
team/odds/model data from ESPN — this module only adds what that dict
doesn't already have: EV/Kelly/tags (game_math.py), league ranks, travel/
surface (team_venues.py), confidence, and the line-movement series.

What's deliberately NOT here, and why — see HANDOFF.md's "known gaps" and
the plan this was built from:
  - Referee crew: no data source anywhere, in or outside this codebase.
  - ATS/O-U on past games (ties into trends/h2h quality): `pregame_odds`
    only recently started capturing snapshots and the local stats
    warehouse (TeamGameStat) currently only has the 2026 season backfilled,
    so there's no real historical closing-line data to grade past games
    against yet. Trends/H2H below return [] until that exists; recent-form
    rows show real SU (straight-up) results only.
"""
from datetime import datetime

import game_math as gm
import game_detail_common as gdc
import nfl_api
import nfl_stats_db
import spread_model
import team_venues


# ── Locate the game ──────────────────────────────────────────────────────────

def find_game(game_id):
    """Returns (game_dict, season, week) for an NFL game_id by searching the
    current week and its immediate neighbors — build_week_schedule_context()
    is already cached (schedule-page page loads warm it), so this doesn't
    cost an extra ESPN fetch in the common case of a game the user just
    tapped from /nfl. Returns (None, None, None) if not found in that window."""
    season, current_week = nfl_api.get_current_week()
    tried = set()
    for w in (current_week, current_week - 1, current_week + 1):
        if w is None or w < 1 or w > 18 or w in tried:
            continue
        tried.add(w)
        ctx = nfl_api.build_week_schedule_context(w)
        for day in ctx.get('days', []):
            for g in day.get('games', []):
                if str(g.get('game_id')) == str(game_id):
                    return g, ctx.get('season', season), w
    return None, None, None


# ── Math helpers specific to this page (wrap game_math with real inputs) ────
# Sport-generic versions of these now live in game_detail_common.py.

def _hist_key(home_name, away_name, game_time_utc):
    return gdc.hist_key(home_name, away_name, game_time_utc)


def _market_gap_sd(market):
    return gdc.market_gap_sd('NFL', market)


def _ats_record_when_edge(threshold_pts=1.5):
    return gdc.ats_record_when_edge('NFL', threshold_pts)


# ── League ranks (real, computed from the local stats warehouse) ───────────

_RANK_TTL = 3600


def _league_rank_table(season, game_date_et):
    """{team_id: {stat_key: (value, rank)}} across every team with at least
    one stored game this season — real computation (no EPA/success-rate;
    see module docstring and HANDOFF.md's gaps table for why those two
    specifically are left out). Cached briefly since this walks every
    team's play-by-play aggregate once per call."""
    def _compute():
        teams = nfl_stats_db.get_teams_with_data_db(season)
        raw = {}
        for team_id, _name in teams:
            off_games, def_games = nfl_stats_db.get_team_game_aggregates_db(
                team_id, before_date=game_date_et, limit=20)
            allowed_games = nfl_stats_db.get_opponent_offense_allowed_db(
                team_id, before_date=game_date_et, limit=20)

            pass_att = sum(g.get('pass_att', 0) for g in off_games)
            pass_yds = sum(g.get('pass_yds', 0) for g in off_games)
            pass_td  = sum(g.get('pass_td', 0) for g in off_games)
            pass_int = sum(g.get('pass_int', 0) for g in off_games)
            rush_att = sum(g.get('rush_att', 0) for g in off_games)
            rush_yds = sum(g.get('rush_yds', 0) for g in off_games)
            sacks_for = sum(g.get('sacks', 0) for g in def_games)

            opp_pass_att = sum(g.get('pass_att', 0) for g in allowed_games)
            opp_pass_yds = sum(g.get('pass_yds', 0) for g in allowed_games)
            opp_pass_td  = sum(g.get('pass_td', 0) for g in allowed_games)
            opp_pass_int = sum(g.get('pass_int', 0) for g in allowed_games)
            opp_rush_att = sum(g.get('rush_att', 0) for g in allowed_games)
            opp_rush_yds = sum(g.get('rush_yds', 0) for g in allowed_games)
            giveaways = sum(g.get('giveaways', 0) for g in off_games)

            any_a = ((pass_yds + 20 * pass_td - 45 * pass_int) / pass_att) if pass_att else None
            ypc = (rush_yds / rush_att) if rush_att else None
            any_a_allowed = ((opp_pass_yds + 20 * opp_pass_td - 45 * opp_pass_int) / opp_pass_att) if opp_pass_att else None
            ypc_allowed = (opp_rush_yds / opp_rush_att) if opp_rush_att else None
            # Turnover diff = takeaways (via defense) minus giveaways (via offense).
            takeaways = sum(g.get('takeaways', 0) for g in def_games)
            turnover_diff = takeaways - giveaways

            raw[team_id] = {
                'pass_any_a':        any_a,
                'rush_ypc':          ypc,
                'pass_any_a_allowed': any_a_allowed,
                'rush_ypc_allowed':  ypc_allowed,
                'sacks':             sacks_for or None,
                'turnover_diff':     turnover_diff,
            }

        # Rank each stat across every team that has a value for it. Higher
        # is better except *_allowed (lower allowed = better defense).
        lower_is_better = {'pass_any_a_allowed', 'rush_ypc_allowed'}
        n_teams = len(raw)
        ranked = {tid: {} for tid in raw}
        for stat in ('pass_any_a', 'rush_ypc', 'pass_any_a_allowed',
                     'rush_ypc_allowed', 'sacks', 'turnover_diff'):
            vals = [(tid, v[stat]) for tid, v in raw.items() if v[stat] is not None]
            reverse = stat not in lower_is_better
            vals.sort(key=lambda x: x[1], reverse=reverse)
            for i, (tid, v) in enumerate(vals, start=1):
                ranked[tid][stat] = (round(v, 1) if isinstance(v, float) else v, i)
        return {'ranks': ranked, 'n_teams': n_teams}

    return nfl_api._cached_compute(f'nfl_detail_ranks_{season}_{game_date_et}', _RANK_TTL, _compute)


# ── Line movement series (section 5) ────────────────────────────────────────

def _line_movement(g):
    return gdc.line_movement('nfl', g['home']['name'], g['away']['name'], g['game_time_utc'])


# ── Recent form (section 9), real SU results, ATS/OU only when a closing
# line happens to be on record for that specific historical game ───────────

def _recent_form(season, team_name, before_date, n=5):
    game_log = nfl_api._get_season_game_log(season)
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
        row = {
            'result': 'W' if won else 'L',
            'opp': f"{'vs' if is_home else '@'} {opp}",
            'score': f"{int(own_score)}-{int(opp_score)}" if own_score is not None and opp_score is not None else None,
            'ats': None,   # only filled in below when a real closing line exists
            'ou': None,
        }
        rows.append(row)
    return rows


# ── Full assembly ────────────────────────────────────────────────────────────

def build_game_detail(game_id):
    g, season, _week = find_game(game_id)
    if not g:
        return None

    home, away = g['home'], g['away']
    model = g.get('model') or {}
    factors = model.get('factors') or []
    odds = g.get('odds') or {}
    game_date_et = (g.get('game_time_utc') or '')[:10]

    # ── model probabilities + confidence + bug fixes from the handoff ──────
    p_team = model.get('team_only_prob')
    p_player = model.get('player_prob')
    p_blend = model.get('home_prob')   # the ONE probability used everywhere below
    conf_ml = gm.confidence(p_team, p_player) if p_team is not None and p_player is not None else None

    home_novig = away_novig = hold_ml = None
    if odds.get('home_best') and odds.get('away_best'):
        imp_h = gm.implied_prob(odds['home_best'])
        imp_a = gm.implied_prob(odds['away_best'])
        home_novig, away_novig, hold_ml = gm.no_vig_pair(imp_h, imp_a)

    ml_verdict = None
    if p_blend is not None and home_novig is not None:
        # Pick derived from model-vs-no-vig, never from line-movement
        # direction — the handoff's bug #1.
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

    spread_margin = spread_model.predict_margin('NFL', factors)  # home margin, + = home favored
    gap_sd_spread = _market_gap_sd('spread')
    spread_verdict = None
    home_spread_line = odds.get('home_spread')
    home_spread_price = odds.get('home_spread_price')
    away_spread_price = odds.get('away_spread_price')
    if spread_margin is not None and home_spread_line is not None and home_spread_price and away_spread_price:
        home_cover = spread_model.cover_prob('NFL', factors, home_spread_line)
        away_cover = (1 - home_cover) if home_cover is not None else None
        # Pick whichever side has the higher EV (not just whichever the
        # model "likes" by margin) — same value-side-first principle as the
        # moneyline verdict above.
        ev_home = gm.ev(home_cover, gm.net_payout_b(home_spread_price)) if home_cover is not None else None
        ev_away = gm.ev(away_cover, gm.net_payout_b(away_spread_price)) if away_cover is not None else None
        if ev_home is not None and ev_away is not None:
            fav_home = ev_home >= ev_away
            side_cover = home_cover if fav_home else away_cover
            side_line = home_spread_line if fav_home else -home_spread_line
            side_price = home_spread_price if fav_home else away_spread_price
            ev = ev_home if fav_home else ev_away
            # Outlier guard on a single consistent axis: model's home margin
            # vs. the market's implied home margin (-home_spread_line), e.g.
            # a -3 home line implies the market expects home by ~3.
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

    # ── ranks / unit matchups (sections 6-7), real data only ───────────────
    rank_table = _league_rank_table(season, game_date_et)
    unit_rows = []
    ranks = rank_table['ranks']
    for off_id, off_abbr, off_key, off_label, off_unit, def_id, def_abbr, def_key, def_label, def_unit in (
        (away['id'], away['abbr'], 'pass_any_a', 'ANY/A', 'Pass offense',
         home['id'], home['abbr'], 'pass_any_a_allowed', 'ANY/A', 'Pass defense'),
        (away['id'], away['abbr'], 'rush_ypc', 'YPC', 'Rush offense',
         home['id'], home['abbr'], 'rush_ypc_allowed', 'YPC', 'Rush defense'),
        (home['id'], home['abbr'], 'pass_any_a', 'ANY/A', 'Pass offense',
         away['id'], away['abbr'], 'pass_any_a_allowed', 'ANY/A', 'Pass defense'),
        (home['id'], home['abbr'], 'rush_ypc', 'YPC', 'Rush offense',
         away['id'], away['abbr'], 'rush_ypc_allowed', 'YPC', 'Rush defense'),
    ):
        off = ranks.get(off_id, {}).get(off_key)
        deff = ranks.get(def_id, {}).get(def_key)
        if not off or not deff:
            continue
        off_val, off_rank = off
        def_val, def_rank = deff
        edge = off_abbr if off_rank < def_rank else (def_abbr if def_rank < off_rank else 'even')
        unit_rows.append({
            'off_team': off_abbr, 'off_unit': off_unit, 'off_stat': f'{off_val} {off_label}',
            'off_rank': off_rank, 'def_team': def_abbr, 'def_unit': def_unit,
            'def_stat': f'{def_val} {def_label} allowed', 'def_rank': def_rank, 'edge': edge,
        })

    stat_rows = []
    for key, label in (('pass_any_a', 'Pass ANY/A'), ('rush_ypc', 'Rush YPC'),
                        ('pass_any_a_allowed', 'Pass ANY/A allowed'), ('rush_ypc_allowed', 'Rush YPC allowed'),
                        ('sacks', 'Sacks'), ('turnover_diff', 'Turnover diff')):
        a = ranks.get(away['id'], {}).get(key)
        h = ranks.get(home['id'], {}).get(key)
        if not a or not h:
            continue
        stat_rows.append({'label': label, 'away': {'value': a[0], 'rank': a[1]},
                           'home': {'value': h[0], 'rank': h[1]}, 'n_teams': rank_table['n_teams']})

    # ── situation: rest (real), weather (real), travel/surface (real,
    # computed from team_venues.py), next week/ref crew hidden (no data) ───
    away_venue = team_venues.get(away['abbr'])
    home_venue = team_venues.get(home['abbr'])
    situation = {}
    if home.get('rest_days') is not None and away.get('rest_days') is not None:
        situation['rest'] = f"{away['rest_days']}d / {home['rest_days']}d"
    if away_venue and home_venue:
        miles = team_venues.travel_miles(away['abbr'], home['abbr'])
        try:
            kickoff_dt = datetime.fromisoformat((g['game_time_utc'] or '').replace('Z', '+00:00'))
            tz_change = team_venues.timezone_change_hours(away['abbr'], home['abbr'], kickoff_dt)
        except Exception:
            tz_change = None
        if miles is not None:
            tz_note = ''
            if tz_change:
                # tz_change = home_offset - away_offset; negative means the
                # home venue sits further west (more negative UTC offset)
                # than the away team's home tz, i.e. the away team is
                # traveling west.
                tz_note = f", {abs(tz_change)} time zone{'s' if abs(tz_change) != 1 else ''} {'west' if tz_change < 0 else 'east'}"
            situation['travel'] = f"{away['abbr']} {miles} mi{tz_note}"
        situation['surface'] = home_venue['surface'].capitalize()
    if g.get('weather'):
        situation['weather'] = f"{g['weather']['temp']}°F" + (
            f", gust {g['weather']['gust_mph']} mph" if g['weather'].get('gust_mph') else '')

    # ── recent form (section 9) — real SU results only ──────────────────────
    recent = {
        'away': _recent_form(season, away['name'], game_date_et),
        'home': _recent_form(season, home['name'], game_date_et),
    }

    return {
        'game': {
            'id': g['game_id'], 'league': 'NFL', 'kickoff_utc': g['game_time_utc'],
            'venue': g.get('venue'), 'status': g.get('status'),
            'weather': g.get('weather'),
        },
        'teams': {'away': away, 'home': home},
        'fanduel': {
            'updated_at': None,
            'moneyline': {'away': {'price': odds.get('away_best'), 'open': odds.get('opening_away')},
                           'home': {'price': odds.get('home_best'), 'open': odds.get('opening_home')}},
            'spread': {'away': {'line': -home_spread_line if home_spread_line is not None else None,
                                  'price': odds.get('away_spread_price')},
                        'home': {'line': home_spread_line, 'price': odds.get('home_spread_price')}},
            'hold_ml': round(hold_ml * 100, 1) if hold_ml is not None else None,
            'novig_home': round(home_novig * 100, 1) if home_novig is not None else None,
            'novig_away': round(away_novig * 100, 1) if away_novig is not None else None,
            'history': _line_movement(g),
        },
        'model': {
            'p_team_home': p_team, 'p_player_home': p_player, 'p_blend_home': p_blend,
            'spread_home': -spread_margin if spread_margin is not None else None,
            'confidence': conf_ml,
            'factors': factors,
            'ats_record_when_edge': _ats_record_when_edge(),
        },
        'verdicts': verdicts,
        'matchups': unit_rows,
        'stats': stat_rows,
        'recent': recent,
        'injuries': {'away': away.get('injuries') or [], 'home': home.get('injuries') or []},
        'situation': situation,
        'trends': [],   # hidden — see module docstring
        'h2h': [],      # hidden — see module docstring
    }
