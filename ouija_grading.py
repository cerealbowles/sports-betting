"""Weekly grading for the Ouija Board (/stats/matchups/record).

Rebuilds every past NFL week's reads as they would have looked at kickoff —
nothing from that game or later — and grades them against what happened.
Nothing is persisted: every input is already stored point-in-time, so a
rebuild is reproducible and can't drift from a snapshot table.

  player lines   NflSimResult draws (raw and nfl_fantasy.calibrate_draws)
                 vs PlayerGameStat fantasy points
  matchup chips  defense-vs-position grade from TeamGameStat rows strictly
                 before the slate date vs what that defense allowed in the
                 game (both relative to league average)
  game reads     matchup_board.read_game() on GamePrediction's frozen
                 kickoff line + stored model factors, the week's sim, and
                 point differential before the date

Leakage guard: a week only counts when its NflSimGame rows were computed
before that week's first kickoff (weeks 1-6 of 2026 were backfilled on
Oct 14, after they'd been played, and are reported as excluded).

The matchup-chip grade here is rebuilt with a date cutoff; the live board's
own grade (app._nfl_league_defense_profile) is the same last-5 average with
no cutoff, which is correct pregame.
"""
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import matchup_board as mb

POSITIONS = ('QB', 'RB', 'WR', 'TE')
CELL_POS = (('QB', 'fpts_allowed_qb'), ('RB', 'fpts_allowed_rb'), ('WR', 'fpts_allowed_rec'))
TONES = ('smash', 'good', 'even', 'bad', 'wall')


def _aware(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _q(sorted_draws, f):
    return sorted_draws[int(round(f * (len(sorted_draws) - 1)))]


def _record(results):
    w = sum(1 for r in results if r is True)
    l = sum(1 for r in results if r is False)
    p = sum(1 for r in results if r is None)
    return {'w': w, 'l': l, 'p': p, 'n': w + l,
            'pct': round(w / (w + l) * 100) if w + l else None,
            'roi': round((w * (100 / 110) - l) / (w + l) * 100, 1) if w + l else None}


def _corr(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 2) if sx and sy else None


def grade_season(season):
    """Everything /stats/matchups/record shows, as plain data."""
    import spread_model
    import nfl_fantasy
    from app import (app as flask_app, TeamGameStat, GamePrediction, NflSimGame, NflSimResult, PlayerGameStat,
                     NflProjectionSnapshot)

    with flask_app.app_context():
        tgs = TeamGameStat.query.filter_by(sport='NFL').all()
        preds = {(p.game_date, p.home_team, p.away_team): p
                 for p in GamePrediction.query.filter_by(sport='NFL').all()}
        sim_games = NflSimGame.query.filter_by(season=season).all()
        sim_rows = NflSimResult.query.filter_by(season=season).all()
        snaps = NflProjectionSnapshot.query.filter_by(season=season).all()
        game_ids = {s.game_id for s in sim_games} | {s.game_id for s in snaps}
        stat_rows = (PlayerGameStat.query.filter(PlayerGameStat.sport == 'NFL',
                                                 PlayerGameStat.game_id.in_(game_ids)).all()
                     if game_ids else [])
        actual_pts = {}
        for r in stat_rows:
            try:
                cats = json.loads(r.stats_json) if r.stats_json else {}
            except (TypeError, ValueError):
                cats = {}
            actual_pts[(r.game_id, r.player_id)] = nfl_fantasy.compute_fantasy_points(cats)

    by_game = defaultdict(dict)
    team_rows = defaultdict(list)
    for r in tgs:
        by_game[r.game_id][bool(r.is_home)] = r
        team_rows[r.team_id].append(r)
    for rows in team_rows.values():
        rows.sort(key=lambda r: r.game_date, reverse=True)

    def kickoff(p):
        try:
            return datetime.fromisoformat(p.game_time_utc.replace('Z', '+00:00')) if p and p.game_time_utc else None
        except ValueError:
            return None

    # Which weeks are clean: sims computed before the week's first kickoff.
    week_games = defaultdict(list)
    for s in sim_games:
        h, a = by_game[s.game_id].get(True), by_game[s.game_id].get(False)
        if h and a:
            week_games[s.week].append((s, h, a, preds.get((h.game_date, h.team_name, a.team_name))))
    weeks = {}
    for wk, items in week_games.items():
        kicks = [k for k in (kickoff(p) for *_, p in items) if k]
        computed = [_aware(s.computed_at) for s, *_ in items if s.computed_at]
        first_kick = min(kicks) if kicks else None
        last_run = max(computed) if computed else None
        finals = sum(1 for _, h, a, _ in items if h.points is not None and a.points is not None)
        weeks[wk] = {'week': wk, 'games': len(items), 'finals': finals,
                     'clean': bool(first_kick and last_run and last_run < first_kick),
                     'sim_run': last_run.isoformat() if last_run else None}

    def diff_pg(team_id, before):
        rows = [r for r in team_rows[team_id] if r.game_date < before and r.season == season
                and r.points is not None and r.points_allowed is not None
                and (r.game_type or 'regular') == 'regular']
        return sum(r.points - r.points_allowed for r in rows) / len(rows) if rows else None

    profiles = {}

    def cell_pct(def_id, pos, before):
        if before not in profiles:
            prof = {}
            for tid, rows in team_rows.items():
                recent = [r for r in rows if r.game_date < before][:5]
                e = {}
                for p, col in CELL_POS:
                    vals = [getattr(r, col) for r in recent if getattr(r, col) is not None]
                    if vals:
                        e[p] = sum(vals) / len(vals)
                prof[tid] = e
            avgs = {}
            for p, _ in CELL_POS:
                vals = [e[p] for e in prof.values() if p in e]
                avgs[p] = sum(vals) / len(vals) if len(vals) >= 2 else None
            profiles[before] = (prof, avgs)
        prof, avgs = profiles[before]
        val, avg = prof.get(def_id, {}).get(pos), avgs.get(pos)
        if val is None or not avg:
            return None, avg
        return (val / avg - 1) * 100, avg

    games, cells = [], []
    for wk, items in week_games.items():
        if not weeks[wk]['clean']:
            continue
        for s, h, a, p in items:
            if h.points is None or a.points is None:
                continue
            date = h.game_date
            for off, deff in ((a, h), (h, a)):
                for pos, col in CELL_POS:
                    pct, avg = cell_pct(deff.team_id, pos, date)
                    allowed = getattr(deff, col)
                    if pct is None or allowed is None:
                        continue
                    cells.append({'week': wk, 'off': off.team_name, 'def': deff.team_name,
                                  'pos': 'WR/TE' if pos == 'WR' else pos, 'pct': round(pct),
                                  'actual_pct': round((allowed / avg - 1) * 100),
                                  'tone': mb.grade_style(pct)})
            mkt = p.spread_pick_line if p else None
            model_h = None
            if p and p.home_prob is not None:
                try:
                    factors = [tuple(f) for f in json.loads(p.factors_json or '[]')]
                except (TypeError, ValueError):
                    factors = []
                m = spread_model.best_margin('NFL', factors, p.home_prob)
                model_h = None if m is None else -m
            read = mb.read_game(mkt, model_h, s.home_avg_score, s.away_avg_score,
                                diff_pg(h.team_id, date), diff_pg(a.team_id, date))
            margin = h.points - a.points

            def covered(side):
                x = margin + mkt
                return None if x == 0 else (x > 0 if side == 'home' else x < 0)

            g = {'week': wk, 'date': date, 'away': a.team_name, 'home': h.team_name,
                 'away_pts': a.points, 'home_pts': h.points, 'margin': margin, 'mkt': mkt,
                 'sim_margin': s.home_avg_score - s.away_avg_score,
                 'model_margin': None if model_h is None else -model_h}
            if read:
                g.update({'verdict': read['verdict'], 'side': read['side'],
                          'side_team': h.team_name if read['side'] == 'home' else a.team_name,
                          'side_line': read['side_line'], 'comp': round(read['comp'], 1),
                          'agree': read['agree'], 'covered': covered(read['side']),
                          'form_margin': None if read['form_line'] is None else -read['form_line'],
                          'signal_cover': {k: covered('home' if v > 0 else 'away')
                                           for k, v in read['edges'].items()
                                           if v is not None and abs(v) >= mb.AGREE_PTS}})
            games.append(g)

    # Player lines.
    clean_weeks = {wk for wk, w in weeks.items() if w['clean']}
    players = []
    for r in sim_rows:
        if r.week not in clean_weeks or r.player_id.startswith('DST_') or r.position not in POSITIONS:
            continue
        actual = actual_pts.get((r.game_id, r.player_id))
        if actual is None or not r.sim_draws_json:
            continue
        raw = sorted(json.loads(r.sim_draws_json))
        cal = nfl_fantasy.calibrate_draws(raw, r.position, r.sim_median)
        players.append({'week': r.week, 'name': r.player_name, 'pos': r.position, 'med': r.sim_median,
                        'actual': actual, 'raw': (_q(raw, 0.1), _q(raw, 0.9)), 'cal': (_q(cal, 0.1), _q(cal, 0.9))})

    def band_stats(ps):
        if not ps:
            return None
        n = len(ps)
        out = {'n': n, 'bias': round(sum(p['actual'] - p['med'] for p in ps) / n, 1),
               'mae': round(sum(abs(p['actual'] - p['med']) for p in ps) / n, 1)}
        for key in ('raw', 'cal'):
            out[key] = {'inside': round(sum(p[key][0] <= p['actual'] <= p[key][1] for p in ps) / n * 100),
                        'below': round(sum(p['actual'] < p[key][0] for p in ps) / n * 100),
                        'above': round(sum(p['actual'] > p[key][1] for p in ps) / n * 100)}
        return out

    band_by_pos = [{'pos': pos, **band_stats([p for p in players if p['pos'] == pos])}
                   for pos in POSITIONS if any(p['pos'] == pos for p in players)]
    band_by_week = [{'week': wk, **band_stats([p for p in players if p['week'] == wk])}
                    for wk in sorted(clean_weeks) if any(p['week'] == wk for p in players)]

    # "Players of the day": each week's top 5 sim medians per position.
    pod = []
    for wk in sorted(clean_weeks):
        for pos in POSITIONS:
            top = sorted((p for p in players if p['week'] == wk and p['pos'] == pos and p['med'] >= 2.5),
                         key=lambda p: -p['med'])[:5]
            pod.extend(top)
    pod_stats = {'n': len(pod),
                 'avg_med': round(sum(p['med'] for p in pod) / len(pod), 1) if pod else None,
                 'avg_actual': round(sum(p['actual'] for p in pod) / len(pod), 1) if pod else None,
                 'beat_med': round(sum(p['actual'] >= p['med'] for p in pod) / len(pod) * 100) if pod else None}

    # Matchup chips.
    tone_rows = []
    for t in TONES:
        v = [c['actual_pct'] for c in cells if c['tone'] == t]
        if v:
            tone_rows.append({'tone': t, 'n': len(v), 'avg': round(sum(v) / len(v), 1),
                              'beat': round(sum(1 for x in v if x > 0) / len(v) * 100)})
    extreme = [c for c in cells if c['tone'] in ('smash', 'wall')]
    chip_hit = (round(sum((c['pct'] > 0) == (c['actual_pct'] > 0) for c in extreme) / len(extreme) * 100)
                if extreme else None)

    # Game reads.
    read_games = [g for g in games if g.get('verdict') and g['mkt'] is not None]
    verdicts = [{'verdict': v, **_record([g['covered'] for g in read_games if g['verdict'] == v])}
                for v in ('TARGET', 'LEAN', 'WATCH', 'STAY AWAY')]
    signals = [{'signal': label, **_record([g['signal_cover'][k] for g in read_games if k in g['signal_cover']])}
               for k, label in (('model', 'Model'), ('sim', 'Simulator'), ('form', 'Point diff'))]

    def mae(key):
        e = [abs(g[key] - g['margin']) for g in games if g.get(key) is not None]
        return round(sum(e) / len(e), 1) if e else None

    margin_err = {'market': (round(sum(abs(-g['mkt'] - g['margin']) for g in games if g['mkt'] is not None)
                                   / max(1, sum(1 for g in games if g['mkt'] is not None)), 1)),
                  'model': mae('model_margin'), 'sim': mae('sim_margin'), 'form': mae('form_margin')}

    # Combined projection vs the old projection vs the sim, from pregame
    # snapshots (NflProjectionSnapshot exists from 2026 week 9 on).
    proj_rows = []
    for sn in snaps:
        a = actual_pts.get((sn.game_id, sn.player_id))
        fin = by_game.get(sn.game_id, {})
        if a is None or not fin.get(True) or fin[True].points is None or sn.combined is None:
            continue
        proj_rows.append({'week': sn.week, 'pos': 'REC' if sn.position in ('WR', 'TE') else sn.position,
                          'actual': a, 'projection': sn.projection, 'combined': sn.combined,
                          'sim': sn.sim_median, 'lo': sn.comb_floor, 'hi': sn.comb_ceiling})
    proj_table = []
    for pos in ('QB', 'RB', 'REC'):
        rs = [r for r in proj_rows if r['pos'] == pos]
        if not rs:
            continue
        with_sim = [r for r in rs if r['sim'] is not None]
        m = lambda key, xs: round(sum(abs(r['actual'] - r[key]) for r in xs) / len(xs), 2) if xs else None
        proj_table.append({'pos': 'WR/TE' if pos == 'REC' else pos, 'n': len(rs),
                           'projection': m('projection', rs), 'combined': m('combined', rs),
                           'sim': m('sim', with_sim), 'combined_same': m('combined', with_sim), 'n_sim': len(with_sim),
                           'inside': round(sum(r['lo'] <= r['actual'] <= r['hi'] for r in rs) / len(rs) * 100)})
    proj_weeks = sorted({r['week'] for r in proj_rows})

    latest = max(clean_weeks) if clean_weeks else None
    return {
        'season': season,
        'weeks': [weeks[w] for w in sorted(weeks)],
        'clean_weeks': sorted(clean_weeks),
        'n_games': len(games),
        'players': {'by_pos': band_by_pos, 'by_week': band_by_week, 'overall': band_stats(players),
                    'pod': pod_stats, 'scale': nfl_fantasy.SIM_BAND_SCALE,
                    'fit_weeks': list(nfl_fantasy.SIM_BAND_FIT_WEEKS.get(season, ()))},
        'chips': {'n': len(cells), 'r': _corr([c['pct'] for c in cells], [c['actual_pct'] for c in cells]),
                  'tones': tone_rows, 'extreme_hit': chip_hit,
                  'latest': sorted((c for c in cells if c['week'] == latest), key=lambda c: -abs(c['pct']))[:12]},
        'games': {'verdicts': verdicts, 'signals': signals, 'margin_err': margin_err,
                  'agree3': _record([g['covered'] for g in read_games if g['agree'] == 3]),
                  'latest': sorted((g for g in games if g['week'] == latest), key=lambda g: g['date'])},
        'latest_week': latest,
        'projection': {'table': proj_table, 'weeks': proj_weeks, 'n': len(proj_rows)},
    }
