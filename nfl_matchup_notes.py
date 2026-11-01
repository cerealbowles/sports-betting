"""
nfl_matchup_notes.py — Rule-based matchup commentary for the NFL game-detail
page ("MATCHUP NOTES" card). Turns the numbers nfl_game_detail.py already
computes (league rank table, weather, travel, injuries, recent form) into
short plain-English notes like:

    "SEA's run game is just 4.0 YPC (#24) and CHI allows 5.1 (#29) — but
     CHI is stingier through the air, so the ground is SEA's path of least
     resistance."

Deliberately deterministic, no LLM: every number in a note comes straight
from the inputs, the same game always produces the same notes, and nothing
can be invented. This is CONTEXT, not a pick — none of these rules have been
backtested for ATS edge (see the Recommendation Philosophy: only validated
signals become recommendations), so the page labels the card accordingly and
nothing here feeds the model or the Recommendations list.

Wording is chosen to describe the matchup ("path of least resistance",
"clear edge on the ground") rather than predict play-calling ("they WILL
pass more") — whether teams actually shift run/pass mix against a given
defense hasn't been measured in this app's data.

Pure functions, no DB/network access — build_notes() takes plain dicts so
test_nfl_matchup_notes.py can exercise every rule directly.
"""

# Tier cutoffs as a fraction of the league: top/bottom quarter (8 of 32).
_STRONG = 0.25
_WEAK = 0.75

# Below this many games played, rank-based notes get a sample-size caveat.
_MIN_GAMES = 3

# Favorability score thresholds (see _favor()): 0 = worst possible matchup
# for the offense, 2 = best possible.
_FAVOR_BIG = 1.45
_FAVOR_TOUGH = 0.55
_LEAN_GAP = 0.6

_MAX_NOTES = 7

_SKILL_POS = ('QB', 'RB', 'WR', 'TE')
_OL_POS = ('OL', 'OT', 'OG', 'T', 'G', 'C')


def _stat(ranks, team_id, key):
    """(value, rank) or (None, None)."""
    v = (ranks.get(team_id) or {}).get(key)
    return v if v else (None, None)


def _pct(rank, n):
    """0.0 = best in league, 1.0 = worst."""
    if rank is None or not n or n < 2:
        return None
    return (rank - 1) / (n - 1)


def _ord(n):
    return f"#{n}"


def _fmt(v, nd=1):
    if v is None:
        return '—'
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _favor(off_rank, def_rank, n):
    """How good a spot this is for the offense: offense quality (0..1, 1 =
    best) plus defense weakness (0..1, 1 = worst defense). None if either
    side is missing."""
    po, pd = _pct(off_rank, n), _pct(def_rank, n)
    if po is None or pd is None:
        return None
    return (1 - po) + pd


def _note(kind, team, tone, text, weight):
    return {'kind': kind, 'team': team, 'tone': tone, 'text': text, 'weight': weight}


# ── Unit-vs-unit (the core of it) ──────────────────────────────────────────

def _unit_notes(off, deff, ranks, n):
    """Notes for ONE offense (off) facing ONE defense (deff). off/deff are
    the team dicts (need 'id' and 'abbr')."""
    out = []
    O, D = off['abbr'], deff['abbr']

    ypc, ypc_r = _stat(ranks, off['id'], 'rush_ypc')
    rypg, _ = _stat(ranks, off['id'], 'rush_ypg')
    ypc_a, ypc_a_r = _stat(ranks, deff['id'], 'rush_ypc_allowed')
    rypg_a, _ = _stat(ranks, deff['id'], 'rush_ypg_allowed')
    anya, anya_r = _stat(ranks, off['id'], 'pass_any_a')
    pypg, _ = _stat(ranks, off['id'], 'pass_ypg')
    anya_a, anya_a_r = _stat(ranks, deff['id'], 'pass_any_a_allowed')
    pypg_a, _ = _stat(ranks, deff['id'], 'pass_ypg_allowed')

    rush_f = _favor(ypc_r, ypc_a_r, n)
    pass_f = _favor(anya_r, anya_a_r, n)

    rush_tone = pass_tone = None

    if rush_f is not None:
        rush_vol = f", {_fmt(rypg)} yds/g" if rypg is not None else ''
        rush_vol_a = f", {_fmt(rypg_a)} yds/g" if rypg_a is not None else ''
        if rush_f >= _FAVOR_BIG:
            rush_tone = 'good'
            out.append(_note('run', O, 'good',
                f"{O}'s run game ({_fmt(ypc)} YPC, {_ord(ypc_r)}{rush_vol}) draws a {D} run defense "
                f"allowing {_fmt(ypc_a)} YPC ({_ord(ypc_a_r)}{rush_vol_a}) — clear edge on the ground.",
                rush_f))
        elif rush_f <= _FAVOR_TOUGH:
            rush_tone = 'bad'
            out.append(_note('run', O, 'bad',
                f"Tough sledding for {O} on the ground: {_fmt(ypc)} YPC ({_ord(ypc_r)}) into a {D} run "
                f"defense allowing just {_fmt(ypc_a)} ({_ord(ypc_a_r)}).",
                2 - rush_f))

    if pass_f is not None:
        if pass_f >= _FAVOR_BIG:
            pass_tone = 'good'
            vol = f", {_fmt(pypg)} pass yds/g" if pypg is not None else ''
            out.append(_note('pass', O, 'good',
                f"{O}'s passing attack ({_fmt(anya)} ANY/A, {_ord(anya_r)}{vol}) faces a {D} secondary "
                f"allowing {_fmt(anya_a)} ANY/A ({_ord(anya_a_r)}) — big edge through the air.",
                pass_f))
        elif pass_f <= _FAVOR_TOUGH:
            pass_tone = 'bad'
            out.append(_note('pass', O, 'bad',
                f"{O}'s passing game ({_fmt(anya)} ANY/A, {_ord(anya_r)}) runs into a {D} pass defense "
                f"holding opponents to {_fmt(anya_a)} ({_ord(anya_a_r)}).",
                2 - pass_f))

    # Lean: the defense is clearly softer on one side than the other.
    # Skipped when it would just repeat a "big edge" note on that same side.
    # Driven by the DEFENSE's split (what the opponent gives up), with the
    # offense's own strength as a tiebreak via the favorability scores.
    if rush_f is not None and pass_f is not None:
        gap = pass_f - rush_f
        d_run_p, d_pass_p = _pct(ypc_a_r, n), _pct(anya_a_r, n)
        if abs(gap) >= _LEAN_GAP and d_run_p is not None and d_pass_p is not None:
            if gap > 0 and d_pass_p > d_run_p and pass_tone != 'good':
                out.append(_note('lean', O, 'neutral',
                    f"Path of least resistance for {O} is through the air: {D} is {_ord(ypc_a_r)} vs the run "
                    f"({_fmt(ypc_a)} YPC allowed) but {_ord(anya_a_r)} vs the pass ({_fmt(anya_a)} ANY/A allowed"
                    + (f", {_fmt(pypg_a)} yds/g" if pypg_a is not None else '') + ").",
                    1.0 + abs(gap) / 2))
            elif gap < 0 and d_run_p > d_pass_p and rush_tone != 'good':
                out.append(_note('lean', O, 'neutral',
                    f"Path of least resistance for {O} is on the ground: {D} allows {_fmt(ypc_a)} YPC "
                    f"({_ord(ypc_a_r)}"
                    + (f", {_fmt(rypg_a)} yds/g" if rypg_a is not None else '')
                    + f") but is {_ord(anya_a_r)} vs the pass ({_fmt(anya_a)} ANY/A allowed).",
                    1.0 + abs(gap) / 2))

    # Pass rush: only call out a genuinely elite one.
    spg, spg_r = _stat(ranks, deff['id'], 'sacks_pg')
    if spg_r is not None and _pct(spg_r, n) is not None and spg_r <= max(3, round(n * 0.12)):
        out.append(_note('rush', D, 'good',
            f"{D}'s pass rush is {_ord(spg_r)} in the league at {_fmt(spg)} sacks/game — "
            f"a test for {O}'s protection.", 1.1))

    # Explosive plays allowed: only the leakiest defenses. (No red-zone
    # note: TeamGameStat.redzone_trips/tds_allowed currently read ~3x too
    # many trips with a ~25% TD rate vs the real ~55% — not trustworthy yet.)
    lp, lp_r = _stat(ranks, deff['id'], 'long_plays_allowed_pg')
    if lp_r is not None and _pct(lp_r, n) is not None and _pct(lp_r, n) >= 0.85:
        out.append(_note('explosive', O, 'good',
            f"{D} has given up {_fmt(lp)} plays of 10+ yards per game ({_ord(lp_r)}) — "
            f"chunk-play opportunities for {O}.", 1.05))
    return out


# ── Situational / availability / form ──────────────────────────────────────

_KEY_SHARE = 0.15      # touch share (carries or targets) that makes a skill player "key"


def _injury_notes(team, usage=None):
    """usage: get_team_usage_shares_db(injury_aware=False) for this team —
    {'qb_id', 'rush_shares', 'rec_shares'} — so only players who actually
    get the ball are called out (an Out backup QB or a WR4 isn't news).
    Without usage data, falls back to position-only for Out/Doubtful and
    skips the Questionable-QB note (can't tell starter from backup)."""
    out = []
    T = team.get('abbr', '')
    inj = team.get('injuries') or []
    status = lambda p: (p.get('status') or '').lower()
    pos = lambda p: (p.get('pos') or '').upper()
    is_out = lambda p: status(p) in ('out', 'doubtful', 'injured reserve')

    if usage:
        qb_id = str(usage.get('qb_id') or '')
        rush, rec = usage.get('rush_shares') or {}, usage.get('rec_shares') or {}

        def share(p):
            pid = str(p.get('id') or '')
            return max(rush.get(pid, 0), rec.get(pid, 0)), (
                'carries' if rush.get(pid, 0) >= rec.get(pid, 0) else 'targets')

        is_qb1 = lambda p: pos(p) == 'QB' and str(p.get('id') or '') == qb_id
    else:
        share = lambda p: (None, None)
        is_qb1 = lambda p: pos(p) == 'QB'

    qbs = [p for p in inj if is_qb1(p) and is_out(p)]
    if qbs:
        p = qbs[0]
        out.append(_note('injury', T, 'bad', f"{T} starting QB {p['name']} is {p['status']}.", 2.5))
    elif usage:
        q = [p for p in inj if is_qb1(p) and status(p) == 'questionable']
        if q:
            out.append(_note('injury', T, 'neutral',
                f"{T} starting QB {q[0]['name']} is questionable — watch the final report.", 1.6))

    def _label(p):
        sh, kind = share(p)
        return f"{p['name']} ({p['pos']}, {sh * 100:.0f}% of {kind})" if sh else f"{p['name']} ({p['pos']})"

    skill = [p for p in inj if is_out(p) and pos(p) in _SKILL_POS[1:]]
    if usage:
        skill = sorted((p for p in skill if share(p)[0] >= _KEY_SHARE), key=lambda p: -share(p)[0])
    if skill:
        names = ', '.join(_label(p) for p in skill[:3])
        more = f" +{len(skill) - 3} more" if len(skill) > 3 else ''
        weight = 1.2 + (sum(share(p)[0] for p in skill) if usage else 0.1 * len(skill))
        out.append(_note('injury', T, 'bad', f"{T} without {names}{more}.", weight))

    if usage:
        q_skill = [p for p in inj if status(p) == 'questionable' and pos(p) in _SKILL_POS[1:]
                   and share(p)[0] >= 0.2]
        if q_skill:
            out.append(_note('injury', T, 'neutral',
                f"{T} questionable: {', '.join(_label(p) for p in q_skill[:2])}.", 1.0))

    ol = [p for p in inj if is_out(p) and pos(p) in _OL_POS]
    if len(ol) >= 2:
        out.append(_note('injury', T, 'bad',
            f"{T} is down {len(ol)} offensive linemen ({', '.join(p['name'] for p in ol[:3])}).", 1.3))
    return out


def _situation_notes(away, home, weather, tz_change, kickoff_et_hour):
    out = []
    A, H = away.get('abbr', ''), home.get('abbr', '')
    ra, rh = away.get('rest_days'), home.get('rest_days')
    if ra is not None and rh is not None and abs(ra - rh) >= 3:
        more, less = (A, H) if ra > rh else (H, A)
        out.append(_note('rest', more, 'good',
            f"Rest edge to {more}: {max(ra, rh)} days vs {min(ra, rh)} for {less}.", 1.15))
    if tz_change is not None and abs(tz_change) >= 2:
        direction = 'east' if tz_change > 0 else 'west'
        text = f"{A} travels {abs(tz_change)} time zones {direction}"
        weight = 0.9
        if tz_change > 0 and kickoff_et_hour is not None and kickoff_et_hour < 14:
            body = kickoff_et_hour - tz_change
            ampm = 'AM' if body < 12 else 'PM'
            text += f" for an early kickoff ({(body - 1) % 12 + 1}:00 {ampm} body clock)"
            weight = 1.1
        out.append(_note('travel', A, 'bad', text + '.', weight))
    if weather:
        gust = weather.get('gust_mph')
        temp = weather.get('temp')
        precip = weather.get('precip_pct')
        if gust is not None and gust >= 20:
            out.append(_note('weather', None, 'neutral',
                f"Gusts to {gust} mph — could limit deep passing and kicking for both sides.",
                1.0 + min(gust - 20, 15) / 30))
        if temp is not None and temp <= 25:
            out.append(_note('weather', None, 'neutral', f"Cold game: {temp}°F at kickoff.", 0.8))
        if precip is not None and precip >= 60:
            out.append(_note('weather', None, 'neutral',
                f"{precip}% chance of precipitation — ball security and footing in play.", 0.85))
    return out


def _form_notes(team, rows):
    """rows: newest-first [{'result': 'W'|'L', ...}] from _recent_form()."""
    if not rows:
        return []
    T = team.get('abbr', '')
    first = rows[0]['result']
    streak = 0
    for r in rows:
        if r['result'] != first:
            break
        streak += 1
    if streak >= 3:
        word = 'won' if first == 'W' else 'lost'
        return [_note('form', T, 'good' if first == 'W' else 'bad',
                      f"{T} has {word} {streak} straight.", 0.7 + 0.05 * streak)]
    return []


def _turnover_note(away, home, ranks):
    ta, _ = _stat(ranks, away['id'], 'turnover_diff')
    th, _ = _stat(ranks, home['id'], 'turnover_diff')
    if ta is None or th is None or abs(ta - th) < 6:
        return []
    better, worse = (away, home) if ta > th else (home, away)
    bv, wv = (int(round(ta)), int(round(th))) if ta > th else (int(round(th)), int(round(ta)))
    return [_note('turnovers', better['abbr'], 'good',
                  f"Turnover battle: {better['abbr']} is {bv:+d} on the season vs {wv:+d} for {worse['abbr']}.",
                  0.95)]


# ── Entry point ─────────────────────────────────────────────────────────────

def build_notes(away, home, ranks, n_teams, weather=None, tz_change=None,
                kickoff_et_hour=None, recent=None, usage=None):
    """Returns up to _MAX_NOTES notes, most important first:
    [{'kind', 'team', 'tone': 'good'|'bad'|'neutral', 'text'}].

    away/home: team dicts with 'id', 'abbr', optional 'rest_days',
    'injuries' ([{'name','pos','status'}]).
    ranks: nfl_game_detail._league_rank_table()['ranks'] —
    {team_id: {stat_key: (value, rank)}}.
    recent: {'away': rows, 'home': rows} from _recent_form().
    usage: {team_id: get_team_usage_shares_db(injury_aware=False)} —
    optional; makes injury notes starter/usage-aware."""
    notes = []
    if n_teams and n_teams >= 8:
        notes += _unit_notes(away, home, ranks, n_teams)
        notes += _unit_notes(home, away, ranks, n_teams)
        notes += _turnover_note(away, home, ranks)
    usage = usage or {}
    notes += _injury_notes(away, usage.get(away['id'])) + _injury_notes(home, usage.get(home['id']))
    notes += _situation_notes(away, home, weather, tz_change, kickoff_et_hour)
    recent = recent or {}
    notes += _form_notes(away, recent.get('away')) + _form_notes(home, recent.get('home'))

    notes.sort(key=lambda x: x['weight'], reverse=True)
    notes = notes[:_MAX_NOTES]

    games = [g for g in (_stat(ranks, away['id'], 'games')[0], _stat(ranks, home['id'], 'games')[0])
             if g is not None]
    if games and min(games) < _MIN_GAMES and any(x['kind'] in ('run', 'pass', 'lean') for x in notes):
        notes.append(_note('sample', None, 'neutral',
            f"Small sample: season stats cover only {min(games)} game(s) for one of these teams.", 0))

    for x in notes:
        x.pop('weight', None)
    return notes
