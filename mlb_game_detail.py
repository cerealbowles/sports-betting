"""
mlb_game_detail.py — Data assembly for the MLB game-detail page
(/mlb/game/<game_pk>), same contract/approach as nfl_game_detail.py/
nhl_game_detail.py (read nfl_game_detail.py's module docstring first).

MLB structural notes (confirmed against the live codebase before writing
this, not assumed):
  - No week concept — single ET calendar date schedule
    (mlb_api.build_schedule_context(target_date=...)); find_game() scans a
    date window like nhl_game_detail.py/nba_game_detail.py do. The id is
    MLB Stats API's own gamePk (game['game_pk']), not an ESPN id.
  - Team dicts use 'abbr' (MLB Stats API's own abbreviation, e.g. 'AZ' not
    'ARI') and DO carry a numeric 'id' (MLB Stats API team id) — see
    mlb_api._build_team_info(). No separate id-bridge needed the way NHL's
    module requires one.
  - Single model, no team/player blend (mlb_model.predict() returns one
    home_prob) — conf_ml stays None and tag_for() falls back to its default
    confidence, same as NHL's page (NHL has no player model either).
  - No per-team stats warehouse the way nfl_stats_db.py is for NFL — the
    league rank table below queries the MLB Stats API's whole-league
    `/teams/stats` endpoint directly (one call for pitching, one for
    hitting; same shape mlb_api._get_team_era_map() already uses for ERA
    alone) rather than iterating 30 teams individually.
  - Team dicts (g['home']/g['away']) already carry 'form' (last-10 W/L),
    'ops', 'runs_pg', 'pitcher' (incl. 'era'), and 'schedule_fatigue' —
    computed by mlb_api._build_team_info() for the schedule card itself,
    reused directly here rather than re-derived.
  - Recent form (last 5, with real SU score) is built from the same raw
    MLB Stats API /schedule window mlb_api._get_recent_data() already
    fetches (identical url/params/cache-key, so this shares that cache
    instead of costing an extra request) — that helper's own return value
    doesn't carry opponent names/scores, which this page needs.
  - Travel/surface/timezone: team_venues.py's MLB_TEAMS table (added
    alongside this module).
"""
from datetime import datetime, timedelta

import game_math as gm
import game_detail_common as gdc
import mlb_api
import spread_model
import team_venues


# ── Locate the game ──────────────────────────────────────────────────────────

def find_game(game_id):
    """Returns (game_dict, date_str) for an MLB game_pk — date-window scan,
    same approach as nhl_game_detail.py/nba_game_detail.py (MLB has no week
    concept either)."""
    today = datetime.strptime(mlb_api._today_et(), '%Y-%m-%d')
    for delta in (0, -1, 1, -2, 2, -3, 3):
        date_str = (today + timedelta(days=delta)).strftime('%Y-%m-%d')
        ctx = mlb_api.build_schedule_context(target_date=date_str)
        for day in ctx:
            for g in day.get('games', []):
                if str(g.get('game_pk')) == str(game_id):
                    return g, date_str
    return None, None


# ── Math helpers specific to this page (wrap game_math with real inputs) ────

def _market_gap_sd(market):
    return gdc.market_gap_sd('MLB', market)


def _ats_record_when_edge(threshold_pts=1.0):
    return gdc.ats_record_when_edge('MLB', threshold_pts)


# ── League ranks — whole-league MLB Stats API calls, not a per-team loop ───

_RANK_TTL = 3600


def _league_rank_table(season):
    """{team_id: {stat_key: (value, rank)}} across every team with a value
    for that stat this season. Two whole-league API calls (pitching,
    hitting) rather than NFL's per-team stats-warehouse walk — MLB Stats
    API's own /teams/stats endpoint already aggregates every team in one
    response, same shape mlb_api._get_team_era_map() uses for ERA alone."""
    def _compute():
        pitching = mlb_api._cached_get(
            f"{mlb_api.MLB_API}/teams/stats",
            {'stats': 'season', 'season': season, 'group': 'pitching', 'sportId': 1},
            f'detail_team_pitching_{season}', mlb_api._TTL['stats'])
        hitting = mlb_api._cached_get(
            f"{mlb_api.MLB_API}/teams/stats",
            {'stats': 'season', 'season': season, 'group': 'hitting', 'sportId': 1},
            f'detail_team_hitting_{season}', mlb_api._TTL['stats'])

        raw = {}
        for split in ((pitching or {}).get('stats') or [{}])[0].get('splits', []):
            tid = (split.get('team') or {}).get('id')
            st = split.get('stat', {})
            if not tid:
                continue
            raw.setdefault(tid, {})
            try:
                raw[tid]['era'] = float(st.get('era'))
            except (TypeError, ValueError):
                pass
            try:
                raw[tid]['whip'] = float(st.get('whip'))
            except (TypeError, ValueError):
                pass
        for split in ((hitting or {}).get('stats') or [{}])[0].get('splits', []):
            tid = (split.get('team') or {}).get('id')
            st = split.get('stat', {})
            if not tid:
                continue
            raw.setdefault(tid, {})
            try:
                raw[tid]['ops'] = float(st.get('ops'))
            except (TypeError, ValueError):
                pass
            gp = st.get('gamesPlayed') or 0
            runs = st.get('runs')
            if gp and runs is not None:
                raw[tid]['runs_pg'] = round(runs / gp, 2)

        lower_is_better = {'era', 'whip'}
        n_teams = len(raw)
        ranked = {tid: {} for tid in raw}
        for stat in ('era', 'whip', 'ops', 'runs_pg'):
            vals = [(tid, v[stat]) for tid, v in raw.items() if v.get(stat) is not None]
            reverse = stat not in lower_is_better
            vals.sort(key=lambda x: x[1], reverse=reverse)
            for i, (tid, v) in enumerate(vals, start=1):
                ranked[tid][stat] = (round(v, 3) if isinstance(v, float) else v, i)
        return {'ranks': ranked, 'n_teams': n_teams}

    return gdc.cached_compute(f'mlb_detail_ranks_{season}', _RANK_TTL, _compute)


# ── Line movement series (section 5) ────────────────────────────────────────

def _line_movement(g):
    return gdc.line_movement('mlb', g['home']['name'], g['away']['name'], g['game_time_utc'])


# ── Recent form (section 9) — real SU results, from the same raw schedule
# window mlb_api._get_recent_data() fetches (shared cache key/TTL, so this
# costs no extra request in the common case of that function already having
# run for today's build_schedule_context() call). ──────────────────────────

def _recent_games_raw(days_back=20):
    end   = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    start = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    return mlb_api._cached_get(
        f"{mlb_api.MLB_API}/schedule",
        {'sportId': 1, 'startDate': start, 'endDate': end,
         'hydrate': 'linescore', 'gameType': 'R'},
        f'recent_{start}', mlb_api._TTL['recent'])


def _recent_form(team_name, before_date, n=5):
    data = _recent_games_raw()
    games = []
    for date_obj in (data or {}).get('dates', []):
        for game in date_obj.get('games', []):
            if game.get('status', {}).get('abstractGameState') != 'Final':
                continue
            game_date = date_obj['date']
            if game_date >= before_date:
                continue
            away, home = game['teams']['away'], game['teams']['home']
            a_name, h_name = away['team'].get('name', ''), home['team'].get('name', '')
            a_s, h_s = away.get('score', 0) or 0, home.get('score', 0) or 0
            if a_name == team_name:
                games.append((game_date, False, h_name, a_s, h_s, a_s > h_s))
            elif h_name == team_name:
                games.append((game_date, True, a_name, h_s, a_s, h_s > a_s))
    games.sort(key=lambda x: x[0], reverse=True)

    rows = []
    for game_date, is_home, opp, own_score, opp_score, won in games[:n]:
        rows.append({
            'result': 'W' if won else 'L',
            'opp':    f"{'vs' if is_home else '@'} {opp}",
            'score':  f"{int(own_score)}-{int(opp_score)}",
            'ats':    None,   # only filled in below when a real closing line exists
            'ou':     None,
        })
    return rows


# ── Full assembly ────────────────────────────────────────────────────────────

def build_game_detail(game_id):
    g, game_date_et = find_game(game_id)
    if not g:
        return None

    home, away = g['home'], g['away']
    home['split_label'], away['split_label'] = 'Home', 'Away'
    model = g.get('model') or {}
    factors = model.get('factors') or []
    odds = g.get('odds') or {}
    season = datetime.strptime(game_date_et, '%Y-%m-%d').year

    # ── model probability — single model, no team/player blend (MLB has
    # no player-level model), same degrade pattern as NHL's page ──────────
    # final_home_prob (calibration-blend + line-movement adjusted) is what
    # _best_market_pick/_mlb_candidates use to decide the ★ recommendation —
    # read it here too so this page's edge/Kelly agree with that decision
    # instead of silently recomputing off the raw, unadjusted model prob.
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
            'gate_note': gdc.recommendation_gate_note('MLB', 'MONEYLINE', tag, amer_odds=side_price, edge_pts=edge_pts),
        }

    spread_margin = spread_model.predict_margin('MLB', factors)  # home margin, + = home favored
    gap_sd_spread = _market_gap_sd('spread')
    spread_verdict = None
    home_spread_line = odds.get('home_spread')
    home_spread_price = odds.get('home_spread_price')
    away_spread_price = odds.get('away_spread_price')
    if spread_margin is not None and home_spread_line is not None and home_spread_price and away_spread_price:
        home_cover = spread_model.cover_prob('MLB', factors, home_spread_line)
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
                'edge': f"{abs(spread_margin - (-home_spread_line)):.1f} runs",
                'ev': f"{ev * 100:+.1f}%", 'kelly': ('—' if is_outlier else f"{units:.1f}u"),
                'why': ("Model is an outlier vs. its historical run-line error — verify inputs before betting."
                        if is_outlier else
                        f"Model line {model_side_abbr} {model_side_line:g} vs market {home['abbr']} {home_spread_line:g}."),
                'gate_note': gdc.recommendation_gate_note('MLB', 'SPREAD', tag),
            }

    verdicts = [v for v in (ml_verdict, spread_verdict) if v]

    # ── league ranks (sections 6-7) — ERA/WHIP (pitching) vs OPS/Runs-per-G
    # (hitting) in place of NFL's pass/rush ANY/A split; one "offense vs
    # opposing pitching" row per side (baseball's natural matchup unit,
    # narrower than NFL's 4 but real for both) ─────────────────────────────
    rank_table = _league_rank_table(season)
    ranks = rank_table['ranks']
    unit_rows = []
    for off_id, off_abbr, def_id, def_abbr in (
        (away['id'], away['abbr'], home['id'], home['abbr']),
        (home['id'], home['abbr'], away['id'], away['abbr']),
    ):
        off = ranks.get(off_id, {}).get('ops')
        deff = ranks.get(def_id, {}).get('era')
        if not off or not deff:
            continue
        off_val, off_rank = off
        def_val, def_rank = deff
        edge = off_abbr if off_rank < def_rank else (def_abbr if def_rank < off_rank else 'even')
        unit_rows.append({
            'off_team': off_abbr, 'off_unit': 'Offense', 'off_stat': f'{off_val:.3f} OPS',
            'off_rank': off_rank, 'def_team': def_abbr, 'def_unit': 'Pitching',
            'def_stat': f'{def_val:.2f} ERA', 'def_rank': def_rank, 'edge': edge,
        })

    stat_rows = []
    for key, label, fmt in (('era', 'Team ERA', '{:.2f}'), ('whip', 'WHIP', '{:.2f}'),
                             ('ops', 'OPS', '{:.3f}'), ('runs_pg', 'Runs/G', '{:.2f}')):
        a = ranks.get(away['id'], {}).get(key)
        h = ranks.get(home['id'], {}).get(key)
        if not a or not h:
            continue
        stat_rows.append({'label': label, 'away': {'value': fmt.format(a[0]), 'rank': a[1]},
                           'home': {'value': fmt.format(h[0]), 'rank': h[1]}, 'n_teams': rank_table['n_teams']})

    # ── situation: rest (real, via mlb_api's own schedule_fatigue), travel
    # (real, team_venues.py's MLB table), weather (real, already on g) ──────
    away_venue = team_venues.get(away['abbr'], sport='mlb')
    home_venue = team_venues.get(home['abbr'], sport='mlb')
    situation = {}
    away_fatigue = away.get('schedule_fatigue') or {}
    home_fatigue = home.get('schedule_fatigue') or {}
    if away_fatigue.get('rest_days') is not None and home_fatigue.get('rest_days') is not None:
        situation['rest'] = {'away': away_fatigue['rest_days'], 'home': home_fatigue['rest_days']}
    if away_venue and home_venue:
        miles = team_venues.travel_miles(away['abbr'], home['abbr'], sport='mlb')
        try:
            first_pitch = datetime.fromisoformat((g['game_time_utc'] or '').replace('Z', '+00:00'))
            tz_change = team_venues.timezone_change_hours(away['abbr'], home['abbr'], first_pitch, sport='mlb')
        except Exception:
            tz_change = None
        if miles is not None:
            tz_note = ''
            if tz_change:
                tz_note = f", {abs(tz_change)} time zone{'s' if abs(tz_change) != 1 else ''} {'west' if tz_change < 0 else 'east'}"
            situation['travel'] = f"{away['abbr']} {miles} mi{tz_note}"
        if not home_venue['dome']:
            situation['surface'] = home_venue['surface'].capitalize()
    if g.get('weather'):
        situation['weather'] = f"{g['weather']['temp']}°F" + (
            f", gust {g['weather']['gust_mph']} mph" if g['weather'].get('gust_mph') else '')

    recent = {
        'away': _recent_form(away['name'], game_date_et),
        'home': _recent_form(home['name'], game_date_et),
    }

    return {
        'game': {
            'id': g['game_pk'], 'league': 'MLB', 'kickoff_utc': g['game_time_utc'],
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
        'injuries': {'away': [], 'home': []},   # no injury data source for MLB today
        'situation': situation,
        'trends': [],
        'h2h': [],
    }
