"""
cfb_game_detail.py — Data assembly for the CFB game-detail page
(/cfb/game/<game_id>), same contract/approach as nfl_game_detail.py (read
that module's docstring first) — CFB is structurally the closest of the
ported sports to NFL (confirmed against the live codebase before writing
this, not assumed):
  - Week-based schedule (cfb_api.build_week_schedule_context(week=...)),
    same find_game() week-window scan as NFL's.
  - Team dicts already carry 'id', 'abbrev'/'abbr' (alias), and 'rest_days'
    directly — no separate id-bridge or schedule_fatigue sub-dict needed.
  - Single model, no team/player blend (cfb_model.predict() returns one
    home_prob) — conf_ml stays None and tag_for() falls back to its
    default confidence, same degrade as NHL's/MLB's pages.
  - No per-team stats warehouse the way nfl_stats_db.py is for NFL (CFB's
    own cfb_stats_db.py is thin — points for/against only, not pass/rush
    splits) — the league rank table below reuses cfb_api's own
    _get_season_game_log()/_compute_team_season_stats() (the SAME function
    that already computes each team's own ppg/ppg_allowed for the card),
    an opponent-strength-weighted PPG/PPG-allowed read across every FBS
    team that's played this season, keyed by team NAME (cfb_api has no
    universal team-id space the way NFL's ESPN ids are).
  - No travel/surface situation — no stadium coordinate table exists for
    130+ FBS programs (team_venues.py's NFL/MLB tables are each a fixed,
    ~30-team pro league; CFB's roster of venues is an order of magnitude
    larger and turns over with realignment) — situation stays rest +
    weather only, same honest-omission approach as NFL's referee crew.
"""
from datetime import datetime

import game_math as gm
import game_detail_common as gdc
import cfb_api
import spread_model


# ── Locate the game ──────────────────────────────────────────────────────────

def find_game(game_id):
    """Returns (game_dict, season, week) for a CFB game_id by searching the
    current week and its immediate neighbors — same approach as
    nfl_game_detail.py's find_game()."""
    season, current_week = cfb_api.get_current_week()
    tried = set()
    for w in (current_week, current_week - 1, current_week + 1):
        if w is None or w < 1 or w > cfb_api._REG_SEASON_WEEKS or w in tried:
            continue
        tried.add(w)
        ctx = cfb_api.build_week_schedule_context(w)
        for day in ctx.get('days', []):
            for g in day.get('games', []):
                if str(g.get('game_id')) == str(game_id):
                    return g, ctx.get('season', season), w
    return None, None, None


# ── Math helpers specific to this page (wrap game_math with real inputs) ────

def _market_gap_sd(market):
    return gdc.market_gap_sd('CFB', market)


def _ats_record_when_edge(threshold_pts=3.0):
    return gdc.ats_record_when_edge('CFB', threshold_pts)


# ── League ranks — reuses cfb_api's own opponent-weighted PPG computation,
# the same one each team's own card stat already comes from ──────────────

_RANK_TTL = 3600


def _league_rank_table(season):
    def _compute():
        game_log = cfb_api._get_season_game_log(season)
        team_stats = cfb_api._compute_team_season_stats(game_log)

        lower_is_better = {'ppg_allowed'}
        n_teams = len(team_stats)
        ranked = {name: {} for name in team_stats}
        for stat in ('ppg', 'ppg_allowed'):
            vals = [(name, v[stat]) for name, v in team_stats.items() if v.get(stat) is not None]
            reverse = stat not in lower_is_better
            vals.sort(key=lambda x: x[1], reverse=reverse)
            for i, (name, v) in enumerate(vals, start=1):
                ranked[name][stat] = (v, i)
        return {'ranks': ranked, 'n_teams': n_teams}

    return gdc.cached_compute(f'cfb_detail_ranks_{season}', _RANK_TTL, _compute)


# ── Line movement series (section 5) ────────────────────────────────────────

def _line_movement(g):
    return gdc.line_movement('cfb', g['home']['name'], g['away']['name'], g['game_time_utc'])


# ── Recent form (section 9), real SU results only ───────────────────────────

def _recent_form(season, team_name, before_date, n=5):
    game_log = cfb_api._get_season_game_log(season)
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
            'opp':    f"{'vs' if is_home else '@'} {opp}",
            'score':  f"{int(own_score)}-{int(opp_score)}" if own_score is not None and opp_score is not None else None,
            'ats':    None,   # only filled in below when a real closing line exists
            'ou':     None,
        })
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

    # ── model probability — single model, no team/player blend (CFB has no
    # player-level model), same degrade pattern as NHL's/MLB's pages ──────
    # final_home_prob (calibration-blend + line-movement adjusted) is what
    # _best_market_pick uses to decide the ★ recommendation — read it here
    # too so this page's edge/Kelly agree with that decision instead of
    # silently recomputing off the raw, unadjusted model prob.
    p_blend = model.get('final_home_prob', model.get('home_prob'))
    conf_ml = None

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
            'why': (f"Model has {'home' if fav_home else 'away'} side at "
                    f"{(p_blend if fav_home else 1 - p_blend) * 100:.1f}% vs DraftKings's "
                    f"{(home_novig if fav_home else away_novig) * 100:.1f}% no-vig."),
            'gate_note': gdc.recommendation_gate_note('CFB', 'MONEYLINE', tag, amer_odds=side_price, edge_pts=edge_pts),
        }

    spread_margin = spread_model.predict_margin('CFB', factors)  # home margin, + = home favored
    gap_sd_spread = _market_gap_sd('spread')
    spread_verdict = None
    home_spread_line = odds.get('home_spread')
    home_spread_price = odds.get('home_spread_price')
    away_spread_price = odds.get('away_spread_price')
    if spread_margin is not None and home_spread_line is not None and home_spread_price and away_spread_price:
        home_cover = spread_model.cover_prob('CFB', factors, home_spread_line)
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
                'gate_note': gdc.recommendation_gate_note('CFB', 'SPREAD', tag),
            }

    verdicts = [v for v in (ml_verdict, spread_verdict) if v]

    # ── league ranks (sections 6-7) — PPG/PPG-allowed in place of NFL's
    # pass/rush ANY/A split (CFB has no local play-by-play warehouse to
    # support that finer split) — one "offense vs opposing defense" row per
    # side, same shape as NHL's simpler 2-stat version ─────────────────────
    rank_table = _league_rank_table(season)
    ranks = rank_table['ranks']
    unit_rows = []
    for off_name, off_abbr, def_name, def_abbr in (
        (away['name'], away['abbr'], home['name'], home['abbr']),
        (home['name'], home['abbr'], away['name'], away['abbr']),
    ):
        off = ranks.get(off_name, {}).get('ppg')
        deff = ranks.get(def_name, {}).get('ppg_allowed')
        if not off or not deff:
            continue
        off_val, off_rank = off
        def_val, def_rank = deff
        edge = off_abbr if off_rank < def_rank else (def_abbr if def_rank < off_rank else 'even')
        unit_rows.append({
            'off_team': off_abbr, 'off_unit': 'Offense', 'off_stat': f'{off_val:.1f} PPG',
            'off_rank': off_rank, 'def_team': def_abbr, 'def_unit': 'Defense',
            'def_stat': f'{def_val:.1f} PPG allowed', 'def_rank': def_rank, 'edge': edge,
        })

    stat_rows = []
    for key, label in (('ppg', 'Points/G'), ('ppg_allowed', 'Points allowed/G')):
        a = ranks.get(away['name'], {}).get(key)
        h = ranks.get(home['name'], {}).get(key)
        if not a or not h:
            continue
        stat_rows.append({'label': label, 'away': {'value': a[0], 'rank': a[1]},
                           'home': {'value': h[0], 'rank': h[1]}, 'n_teams': rank_table['n_teams']})

    # ── situation: rest (real, team dict's own rest_days), weather (real,
    # already on g) — no travel/surface (see module docstring) ────────────
    situation = {}
    if home.get('rest_days') is not None and away.get('rest_days') is not None:
        situation['rest'] = f"{away['rest_days']}d / {home['rest_days']}d"
    if g.get('weather'):
        situation['weather'] = f"{g['weather']['temp']}°F" + (
            f", gust {g['weather']['gust_mph']} mph" if g['weather'].get('gust_mph') else '')

    recent = {
        'away': _recent_form(season, away['name'], game_date_et),
        'home': _recent_form(season, home['name'], game_date_et),
    }

    return {
        'game': {
            'id': g['game_id'], 'league': 'CFB', 'kickoff_utc': g['game_time_utc'],
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
            'p_team_home': None, 'p_player_home': None, 'p_blend_home': p_blend,
            'spread_home': -spread_margin if spread_margin is not None else None,
            'confidence': conf_ml,
            'factors': factors,
            'ats_record_when_edge': _ats_record_when_edge(),
        },
        'verdicts': verdicts,
        'matchups': unit_rows,
        'stats': stat_rows,
        'recent': recent,
        'injuries': {'away': [], 'home': []},   # no injury data source for CFB today
        'situation': situation,
        'trends': [],
        'h2h': [],
    }
