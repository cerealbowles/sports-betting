"""The Ouija Board (/stats/matchups): read each game three ways against the
market's home spread and say whether the reads agree.

Pure functions only. app.py's matchups_page() gathers the inputs (market
line, the model's margin, the daily game sim, each team's point
differential, the defense-vs-position grades and last-5 volume) and calls in
here, so the verdict math stays testable without a Flask context.

Sign convention throughout: every edge is `market home spread - signal home
spread`, so a positive edge means that signal thinks the HOME side is
underpriced and a negative edge means the AWAY side is.

The three signals:
  model     the betting model's projected home margin (spread_model.
            best_margin), turned into a home spread.
  sim       the daily drive-resampling game sim's average score
            (nfl_game_simulator / NflSimGame).
  form      half the per-game point-differential gap, plus 1.5 for home
            field. Deliberately crude: it's the "what has each team actually
            done" sanity check, not a model.

These are reads, not picks. None of the three has a validated ATS edge for
NFL (see spread_model._ATS_PCT and the CFB conference-rating backtest), so
the page shows them as context and never feeds a recommendation.
"""

AGREE_PTS = 0.5        # a signal "agrees" when it points comp's way by at least this
TARGET_PTS = 3.0       # all three agree and the average edge is at least this
LEAN_PTS = 1.5         # two or more agree and the average edge is at least this
HOME_FIELD_PTS = 1.5   # added to the point-diff line, per the design spec

FAST_PLAYS = 125       # combined last-5 pass+rush attempts per game
SLOW_PLAYS = 112
SCRIPT_MARGIN = 3      # sim margin at which we name a trailing team

VERDICT_ORDER = {'TARGET': 0, 'LEAN': 1, 'WATCH': 2, 'STAY AWAY': 3}


def spread_from_margin(home_margin):
    """Projected home margin (home pts minus away pts) -> home spread line."""
    if home_margin is None:
        return None
    return -home_margin


def form_line(home_diff_pg, away_diff_pg):
    """Point-differential spread for the home team, or None when either side
    is missing."""
    if home_diff_pg is None or away_diff_pg is None:
        return None
    return -((home_diff_pg - away_diff_pg) / 2 + HOME_FIELD_PTS)


def read_game(mkt_h, model_h=None, sim_home=None, sim_away=None,
              home_diff_pg=None, away_diff_pg=None):
    """Score one game. Returns None without a market line (nothing to read
    against). Any signal can be missing; the composite averages whichever are
    present, and TARGET still needs all three, so a game missing one signal
    can lean at most."""
    if mkt_h is None:
        return None
    sim_line = spread_from_margin(sim_home - sim_away) if sim_home is not None and sim_away is not None else None
    f_line = form_line(home_diff_pg, away_diff_pg)
    edges = {
        'model': None if model_h is None else mkt_h - model_h,
        'sim': None if sim_line is None else mkt_h - sim_line,
        'form': None if f_line is None else mkt_h - f_line,
    }
    present = [v for v in edges.values() if v is not None]
    if not present:
        return None
    comp = sum(present) / len(present)
    direction = 1 if comp >= 0 else -1
    agree = sum(1 for v in present if v * direction >= AGREE_PTS)
    oppose = sum(1 for v in present if v * direction <= -AGREE_PTS)

    if agree == 3 and abs(comp) >= TARGET_PTS:
        verdict = 'TARGET'
    elif oppose >= 1 and agree >= 1:
        verdict = 'STAY AWAY'
    elif agree >= 2 and abs(comp) >= LEAN_PTS:
        verdict = 'LEAN'
    else:
        verdict = 'WATCH'

    side = 'home' if direction > 0 else 'away'
    side_line = mkt_h if side == 'home' else -mkt_h
    return {
        'edges': edges, 'n_signals': len(present), 'comp': comp, 'direction': direction,
        'agree': agree, 'oppose': oppose, 'verdict': verdict,
        'side': side, 'side_line': side_line, 'is_dog': side_line > 0,
        'model_line': model_h, 'sim_line': sim_line, 'form_line': f_line,
    }


def fmt_line(x):
    """-3.0 -> '-3', 2.5 -> '+2.5', 0 -> 'PK'."""
    if x is None:
        return '—'
    if abs(x) < 0.05:
        return 'PK'
    s = f'{x:+.1f}'
    return s[:-2] if s.endswith('.0') else s


def half(x):
    """Round to the nearest half point, the way lines are quoted."""
    return None if x is None else round(x * 2) / 2


def script_read(away, home, vol_away, vol_home, sim_away, sim_home):
    """Pace tag + one-line game-script read. vol_* are last-5 offense dicts
    with pass_att_pg / rush_att_pg (or None)."""
    pace = None
    plays = None
    if vol_away and vol_home:
        plays = (vol_away['pass_att_pg'] + vol_away['rush_att_pg']
                 + vol_home['pass_att_pg'] + vol_home['rush_att_pg'])
        tag = 'FAST' if plays >= FAST_PLAYS else 'SLOW' if plays <= SLOW_PLAYS else None
        pace = f'{tag} · {round(plays)} PLAYS/G' if tag else f'{round(plays)} PLAYS/G'
    if sim_away is None or sim_home is None:
        return {'pace': pace, 'plays': plays, 'text': None}
    margin = sim_home - sim_away
    if abs(margin) >= SCRIPT_MARGIN:
        lead, trail = (home, away) if margin > 0 else (away, home)
        text = (f'{trail} likely trailing (sim {sim_away:.0f}–{sim_home:.0f}): expect extra {trail} '
                f'passing; {lead} leans on the run late.')
    else:
        text = f'Sim has it within {abs(margin):.0f}, a balanced script. Volume follows each team’s usual mix.'
    return {'pace': pace, 'plays': plays, 'text': text}


def why_text(home, away, r, move_text=None):
    """Plain-language reason for the verdict."""
    side_team = home if r['side'] == 'home' else away
    other = away if r['side'] == 'home' else home

    def leans(x):
        return f"{side_team if x * r['direction'] > 0 else other} by {abs(x):.1f}"

    e = r['edges']
    if r['verdict'] == 'STAY AWAY':
        parts = [f'{label} says {leans(e[k])}' for k, label in
                 (('model', 'model'), ('sim', 'the sim'), ('form', 'point diff')) if e[k] is not None]
        return 'The signals disagree: ' + ', '.join(parts) + '. No clean side.'
    if r['verdict'] == 'WATCH':
        return (f"Gaps are small (average {abs(r['comp']):.1f} pts). "
                'Worth watching for a line move, not worth forcing.')
    bits = []
    if r['model_line'] is not None:
        bits.append(f"your model makes it {fmt_line(half(r['model_line']))}")
    if r['sim_line'] is not None:
        bits.append(f"the sim {fmt_line(half(r['sim_line']))}")
    if r['form_line'] is not None:
        bits.append(f"point differential {fmt_line(half(r['form_line']))}")
    text = f"Market has {home} at {fmt_line(r['side_line'] if r['side'] == 'home' else -r['side_line'])}. "
    text += (bits[0][0].upper() + bits[0][1:] + (', ' + ', '.join(bits[1:]) if bits[1:] else '') + '.') if bits else ''
    if move_text:
        text += f' {move_text}.'
    return text


def grade_style(pct):
    """Matchup pct (positive = easier for the offense) -> tone class used by
    the template's chips."""
    if pct is None:
        return 'none'
    if pct >= 15:
        return 'smash'
    if pct >= 5:
        return 'good'
    if pct <= -15:
        return 'wall'
    if pct <= -5:
        return 'bad'
    return 'even'


def env_tag(total):
    if total is None:
        return ''
    if total >= 49:
        return 'SHOOTOUT'
    if total <= 44:
        return 'GRINDER'
    return 'NEUTRAL'
