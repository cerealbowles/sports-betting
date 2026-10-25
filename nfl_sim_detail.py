"""
nfl_sim_detail.py — Data assembly for the NFL simulation/player-projections
page (/nfl/simulations/<game_id>), matching game-page-handoff/
SIM-HANDOFF.md as closely as real data allows.

Builds on nfl_game_detail.py (for the real betting model's blend
probability, the FanDuel moneyline/spread lines, and the team/game meta —
no reason to re-derive any of that) plus the daily simulation background
job's own stored output (nfl_stats_db.get_sim_results_db()/
get_sim_game_summary_db(), written by nfl_game_simulator.simulate_week()).

What's deliberately NOT here, and why:
  - FanDuel player PROPS (pass/rush/rec yards, anytime TD, ...): this app
    has never ingested a player-props feed (odds_api.py only ever
    requests h2h/spreads/totals markets) — the "Top FanDuel prop" column,
    the prop-EV sort, the per-prop histogram/line-marker, and the
    "Best prop edges" rail all need that feed and are omitted rather than
    shown with fabricated prices. sim_math.py's p_over_normal/
    p_over_poisson/p_anytime_td/ev_over_under are still here, ready for
    whenever that feed exists.
  - Totals (O/U) — this app removed Total modeling/UI everywhere on
    2026-10-23 (spreads + moneyline only going forward), so the agreement
    table and the game-sim row both carry only moneyline/spread, never a
    total row/histogram, even though the design mockup has one.
  - "Sim correlations" uses Pearson r across each pair of players' own
    per-run FANTASY-POINT draws (already stored, NflSimResult.
    sim_draws_json) rather than per-category stat pairs (pass yds vs.
    rec yds, ...) — those raw per-category arrays aren't persisted (only
    each field's median/mean is, in sim_stats_json), and persisting them
    too is real future work, not something to fake here.
"""
import nfl_game_detail
import nfl_stats_db
import sim_math as sm
import spread_model

MARGIN_BIN_WIDTH = 4
MARGIN_RANGE = (-28, 28)
FANTASY_FLOOR_PCT = 10
FANTASY_CEILING_PCT = 90
CORR_MIN_ABS_R = 0.3
CORR_MAX_PAIRS = 8


def _agreement_rows(detail, summary, margins):
    """Rows: ['home win prob', 'spread'] x {market, sim, blend, fanduel,
    read, note}. Hidden (omitted) entirely for a market with no FanDuel
    line stored yet, or no blend-model probability for it — same
    "hide the section rather than fake it" rule as the rest of this page."""
    rows = []

    p_blend_home = detail['model']['p_blend_home']
    novig_home = detail['fanduel']['novig_home']
    if p_blend_home is not None and novig_home is not None and (summary['home_wins'] + summary['away_wins']):
        sim_home_pct = sm.win_prob_excl_ties(summary['home_wins'], summary['away_wins']) * 100
        blend_home_pct = p_blend_home * 100
        agree = (sim_home_pct >= 50) == (blend_home_pct >= 50)
        rows.append({
            'market': 'Home win prob', 'sim': f'{sim_home_pct:.1f}%', 'blend': f'{blend_home_pct:.1f}%',
            'fanduel': f'{novig_home:.1f}%', 'read': 'AGREE' if agree else 'SPLIT',
            'note': (f"Sim and the blend model both favor the "
                     f"{'home' if sim_home_pct >= 50 else 'away'} side."
                     if agree else
                     f"Sim favors {'home' if sim_home_pct >= 50 else 'away'}, "
                     f"the blend model favors {'home' if blend_home_pct >= 50 else 'away'}."),
        })

    home_spread_line = detail['fanduel']['spread']['home']['line']
    factors = detail['model']['factors']
    if home_spread_line is not None and margins and factors:
        cover = sm.cover_prob_empirical(margins, home_spread_line)
        blend_home_cover = spread_model.cover_prob('NFL', factors, home_spread_line)
        if cover and blend_home_cover is not None:
            sim_home_pct = cover['home_covers'] * 100
            blend_home_pct = blend_home_cover * 100
            agree = (sim_home_pct >= 50) == (blend_home_pct >= 50)
            rows.append({
                'market': 'Spread', 'sim': f'{sim_home_pct:.1f}%', 'blend': f'{blend_home_pct:.1f}%',
                'fanduel': f"{home_spread_line:+g}", 'read': 'AGREE' if agree else 'SPLIT',
                'note': (f"Sim and the blend model both lean the "
                         f"{'home' if sim_home_pct >= 50 else 'away'} side of the spread."
                         if agree else
                         f"Sim leans {'home' if sim_home_pct >= 50 else 'away'} against the spread, "
                         f"the blend model leans {'home' if blend_home_pct >= 50 else 'away'}."),
            })
    return rows


def _fantasy_range(sim_draws, sim_median):
    if not sim_draws:
        return None
    return {
        'floor': round(sm.percentile(sim_draws, FANTASY_FLOOR_PCT), 1),
        'median': round(sim_median, 1),
        'ceiling': round(sm.percentile(sim_draws, FANTASY_CEILING_PCT), 1),
    }


def _player_rows(rows, team_abbr):
    from app import _nfl_sim_stat_compare  # module-level helper, avoid a circular import at load time

    out = []
    for r in rows:
        if r['player_id'].startswith('DST_'):
            continue  # no fantasy range/props/correlations story for a DST entry here
        _nfl_sim_stat_compare(r)
        out.append({
            'player_id': r['player_id'], 'name': r['player_name'] or r['player_id'],
            'team': team_abbr, 'position': r['position'] or '?',
            'stat_chips': [c for c in (r.get('stat_compare') or []) if c['pred']],  # hide zero stats
            'fantasy': _fantasy_range(r['sim'], r['sim_median']),
            'actual': r.get('actual'),
            'sim_draws': r['sim'],
        })
    return out


def _correlations(players):
    """Top |r| pairs across each pair of players' own fantasy-point sim
    draws (see module docstring for why this isn't per-stat-category)."""
    pairs = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            a, b = players[i], players[j]
            r = sm.pearson_r(a['sim_draws'], b['sim_draws'])
            if r is None or abs(r) < CORR_MIN_ABS_R:
                continue
            pairs.append({
                'a': f"{a['name']} ({a['team']})", 'b': f"{b['name']} ({b['team']})",
                'r': round(r, 2),
                'note': ('tend to post big fantasy games in the same simulated runs'
                         if r > 0 else 'rarely both have a big fantasy game in the same simulated run'),
            })
    pairs.sort(key=lambda p: -abs(p['r']))
    return pairs[:CORR_MAX_PAIRS]


def build_sim_detail(game_id):
    g, season, week = nfl_game_detail.find_game(game_id)
    if not g:
        return None

    summary_by_game = nfl_stats_db.get_sim_game_summary_db(season, week)
    summary = summary_by_game.get(game_id)
    if not summary:
        return None  # background job hasn't simulated this game yet

    detail = nfl_game_detail.build_game_detail(game_id)
    results_by_game = nfl_stats_db.get_sim_results_db(season, week)
    teams = results_by_game.get(game_id, {})
    home_id, away_id = g['home'].get('id'), g['away'].get('id')

    from app import _attach_actual_to_nfl_sim_game  # see _player_rows() for why this import is lazy
    home_rows = teams.get(home_id, [])
    away_rows = teams.get(away_id, [])
    for row in home_rows + away_rows:
        row.setdefault('actual', None)
        row.setdefault('actual_stats', None)
    _attach_actual_to_nfl_sim_game(g, home_rows, away_rows)

    home_players = _player_rows(home_rows, g['home']['abbr'])
    away_players = _player_rows(away_rows, g['away']['abbr'])

    margins = summary.get('margins') or []
    n_ties_pct = round(100 * summary['ties'] / summary['n_sims'], 1) if summary['n_sims'] else 0.0

    qa, qh = summary['away_quarter_avg'], summary['home_quarter_avg']
    quarters = {
        'away': qa, 'home': qh,
        'away_1h': round(qa[0] + qa[1], 1), 'away_2h': round(qa[2] + qa[3], 1),
        'home_1h': round(qh[0] + qh[1], 1), 'home_2h': round(qh[2] + qh[3], 1),
    }

    return {
        'game': detail['game'], 'teams': detail['teams'],
        'run': {'n_sims': summary['n_sims'], 'ran_at': summary.get('computed_at')},
        'outcome': {
            'away_avg_score': summary['away_avg_score'], 'home_avg_score': summary['home_avg_score'],
            'margin': round(summary['home_avg_score'] - summary['away_avg_score'], 1),
            'total': round(summary['home_avg_score'] + summary['away_avg_score'], 1),
            'away_win_pct': summary['away_win_pct'], 'home_win_pct': summary['home_win_pct'],
            'tie_pct': n_ties_pct, 'tie_warning': n_ties_pct > 2.0,
        },
        'agreement': _agreement_rows(detail, summary, margins),
        'players': {'away': away_players, 'home': home_players},
        'correlations': _correlations(away_players + home_players),
        'margin_hist': sm.histogram(margins, MARGIN_BIN_WIDTH, *MARGIN_RANGE) if margins else [],
        'margin_cover': (sm.cover_prob_empirical(margins, detail['fanduel']['spread']['home']['line'])
                          if margins and detail['fanduel']['spread']['home']['line'] is not None else None),
        'spread_line': detail['fanduel']['spread']['home']['line'],
        'quarters': quarters,
    }
