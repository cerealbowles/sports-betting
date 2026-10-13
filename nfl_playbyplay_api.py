"""
nfl_playbyplay_api.py — Play-by-play parsing for NFL, sourced from the
SAME ESPN `summary?event={id}` response nfl_boxscore_api.py already reads
(`drives.previous[].plays[]` — the box score only reads `boxscore`, this
reads a different key off the identical payload, so this costs zero extra
API calls when a caller already has `summary` in hand, same reuse pattern
as nfl_boxscore_api.py's own docstring describes).

Why this exists: PlayerGameStat's existing category totals (rushing
yards, receptions, etc. — see nfl_boxscore_api.py) answer "how much did
this player produce," not "how" — a bellcow averaging 4.5 YPC on 20
carries and a boom-bust back averaging 4.5 YPC on 8 explosive runs + 12
stuffed ones look identical in the existing data. This derives the
how: long vs. short runs, pass depth (short/deep — ESPN's own play-by-play
tag, not a yardage threshold this module invents), red-zone usage/
efficiency, and — mirrored to the DEFENSIVE side — which plays/zones a
defense actually gives up vs. stuffs.

GRANULARITY: returns BOTH levels — the per-player/per-game AGGREGATES
(long vs. short runs, pass depth splits, red-zone usage) this module was
first built for, AND the individual parsed PLAYS themselves (raw_plays),
stored separately (see nfl_stats_db.ingest_play_by_play()) rather than
discarded after aggregation — a deliberate departure from this app's
usual "aggregate only" convention (see get_team_game_aggregates_db()'s
docstring for that convention elsewhere), made because ESPN's play-by-
play, unlike its box score, isn't guaranteed to stay available forever
(weather already disappears off completed games — see nfl_api._parse_
weather's docstring), and because storing the raw plays means any future
metric idea can be recomputed from what's already in the DB instead of
re-fetching from ESPN.

PLAYER ATTRIBUTION — the real limitation: ESPN's free play-by-play has no
structured athlete-id field on a play, just prose ("D.Henry left guard
for 26 yards", or sometimes "Tank Bigsby 65 Yd Rush" for some scoring
plays). This module extracts the name token(s) via regex and resolves
them against THIS GAME's own box score athlete list (parse_player_
boxscore's output — always accurate for the game being parsed, unlike
nfl_roster_api's CURRENT-roster snapshot, which drifts for old games as
players get traded/cut — see nfl_fantasy.build_player_pool's docstring
for that same documented gap elsewhere). Validated against 2,556 real
plays across 20 games spanning 2024-2025: 99.7-99.9% of rush/pass/sack
plays parse cleanly. The remainder (rare formatting outliers — a
substitution note stacked with a fumble-then-replay-reversal in the same
play text, etc.) are SKIPPED, not guessed — same "a real gap beats a
wrong guess" convention as every other silent-exclusion case in this
app's stats pipeline.

KNOWN GAP — two-point conversions: real box scores count a 2-point
conversion rush/pass as a real rushing/passing attempt, but ESPN's
`drives.plays[]` never gives one its own play entry at all — it's just
prose tacked onto the preceding touchdown's text ("...TOUCHDOWN. TWO-
POINT CONVERSION ATTEMPT. C.Williams pass to K.Allen is incomplete.
ATTEMPT FAILS."), with no structured down/distance/yardage object to
parse. So in a game with 2-point attempts, this module's per-player
attempt/yardage totals can run 0-2 below the real box score for the
players involved — a real, bounded, structural gap in the data source,
not a parsing bug, and not something resolvable without a different
source. Same category of gap as compute_fantasy_points()'s own documented
"no stat key for 2-point conversions at all" — this module inherits it
rather than introducing a new one.

Three real bugs found and fixed during validation, each confirmed against
a specific live game before shipping: (1) a penalty that replays the down
("- No Play") was double-crediting the nullified play's own yardage to
whoever carried/threw it — confirmed against Derrick Henry's real 25-for-
151 box score line, which an unfixed version miscounted as 26-for-141.
(2) a short-abbreviation collision (two "Williams"es on the same 2024
Bears roster, both reducing to "C.Williams") was dropping BOTH players'
attribution instead of favoring the more-featured one the way ESPN's own
text actually does. (3) ESPN duplicate-logs a play when a penalty on it
is DECLINED (a normal entry, then a second "administrative" one with
identical down/distance/yardLine) — confirmed live (Washington @
Cleveland 2024, a 13-yard scramble logged twice) and deduped on exact
start-position match.
"""
import re
from collections import defaultdict

LONG_RUN_YARDS = 10       # "explosive run" — standard NFL analytics threshold
REDZONE_YARDS_TO_GOAL = 20

_NAME = r"[A-Z][A-Za-z.]*\.[A-Za-z'\-]+(?:\s(?:Jr|Sr|II|III|IV)\.?)?"
_FULLNAME = r"[A-Z][\w'.\-]*(?:\s+[A-Z][\w'.\-]*){1,2}"
_PRE_CLAUSES = (
    r"(?:\([^)]*\)\s*)*"
    r"(?:[A-Za-z.]+(?:\s+and\s+[A-Za-z.]+)?\sreported in as eligible\.\s*)*"
    r"(?:\d+-[A-Za-z.]+,?\s+in at \w+\.\s*)*"
)

_RUSH_RE = re.compile(rf"^{_PRE_CLAUSES}(?:Direct snap to\s+)?({_NAME})\b")
_PASS_RE = re.compile(
    rf"^{_PRE_CLAUSES}({_NAME})\s+pass\s+(?:incomplete\s+)?(short|deep)\s+"
    rf"(?:left|right|middle)(?:\s+to\s+({_NAME}))?"
)
_SACK_RE = re.compile(rf"^{_PRE_CLAUSES}({_NAME})\s+sacked")
_ALT_PASS_RE = re.compile(rf"^({_FULLNAME})\s+(?:\d+\s+Yd\s+)?[Pp]ass\s+[Ff]rom\s+({_FULLNAME}?)(?:\s*\(|\s+for\s+\d|$)")
_ALT_RUSH_RE = re.compile(rf"^({_FULLNAME})\s+\d+\s+Yd\s+(?:Rush|Run)\b")
_SKIP_RE = re.compile(r"\bkneels\b|\bspiked the ball\b", re.IGNORECASE)

# A play's own structured `penalty` dict (yards/type/status) turned out to
# be a newer ESPN addition — confirmed ABSENT entirely on 2024-season
# games, present on 2025 ones — so this can't be relied on and text
# parsing is the only consistent path for ALL seasons. Two distinct
# situations share this same clause shape:
#   (a) a standalone 'Penalty'-type play: "PENALTY on GB-J.Jacobs, False
#       Start, 5 yards, enforced at WAS 40 - No Play."
#   (b) a penalty EMBEDDED at the end of an otherwise-normal Rush/Pass/
#       Sack play's text: "(Shotgun) D.Henry left tackle ... for 3 yards
#       (...). PENALTY on BLT-D.Faalele, Offensive Holding, 10 yards,
#       enforced at BLT 48 - No Play." — critically, when this ends in
#       "- No Play", the down was REPLAYED and doesn't count: the rush/
#       pass/sack this clause is attached to must NOT be added to anyone's
#       stats (confirmed live: this exact case gave Derrick Henry a
#       phantom 26th carry at a penalty-adjusted -10 yards instead of his
#       real 25-for-151 box score line, before this check existed). A
#       "declined" penalty changes nothing and the attached play DOES
#       still count.
# Team-level penalties (delay of game, too many men) have no player
# clause at all: "PENALTY on NE, Delay of Game, ...".
_PENALTY_RE = re.compile(
    r"PENALTY on ([A-Z]+)(?:-([A-Za-z.'\-]+(?:\s[A-Za-z.'\-]+)?(?:\s(?:Jr|Sr|II|III|IV)\.?)?))?,"
    r"\s*([^,]+?),\s*(?:(\d+)\s*yards?)?",
    re.IGNORECASE,
)
_NO_PLAY_RE = re.compile(r"-\s*No Play\.?\s*$")
_PENALTY_DECLINED_RE = re.compile(r"\bdeclined\b", re.IGNORECASE)
# ESPN's own team-header abbreviation doesn't always match the one its
# play-by-play prose uses for the same team. Confirmed live by scanning
# every "PENALTY on XXX" team token across 20 real games (2024-2025)
# against each game's header abbreviations — every mismatch found is
# listed here; an abbreviation with no alias just falls back to itself.
_TEAM_ABBREV_ALIASES = {
    'WSH': 'WAS', 'WAS': 'WSH',
    'BAL': 'BLT', 'BLT': 'BAL',
    'HOU': 'HST', 'HST': 'HOU',
    'LAR': 'LA',  'LA':  'LAR',
    'ARI': 'ARZ', 'ARZ': 'ARI',
    'CLE': 'CLV', 'CLV': 'CLE',
}

_SUFFIX_RE = re.compile(r"\s+(?:Jr|Sr|II|III|IV)\.?$")


def _abbrev_candidates(full_name):
    """A roster player's displayName (e.g. 'Ty Johnson', 'Derrick Henry')
    -> the set of abbreviated forms ESPN's play text might use for them.
    ESPN extends the first-name prefix (T. -> Ty.) to break ties when two
    players on the same team share a first initial + last name — rather
    than reverse-engineer that tie-breaking rule, this just generates
    every plausible prefix length and lets the caller match whichever one
    the text actually contains, scoped to one team's roster at a time
    (see _build_name_index) so cross-team collisions don't matter."""
    name = _SUFFIX_RE.sub('', full_name.strip())
    parts = name.split()
    if len(parts) < 2:
        return set()
    first, last = parts[0], ' '.join(parts[1:])
    return {f"{first[:n]}.{last}" for n in range(1, min(len(first), 3) + 1)}


def _stat_volume(categories):
    """Rough 'how featured is this player in this game' proxy — sum of
    every numeric-parseable stat value across their box-score categories.
    Used only to break a short-abbreviation collision (see
    _build_name_index) — not stored or exposed anywhere."""
    total = 0.0
    for stats in (categories or {}).values():
        for v in stats.values():
            try:
                total += abs(float(v))
            except (TypeError, ValueError):
                continue  # non-numeric (e.g. "19/24" completions ratio) — skip
    return total


def _build_name_index(team_players):
    """team_players: [{id, name, categories}, ...] (one team's box-score
    athletes, from parse_player_boxscore's output). Returns {abbrev_lower:
    player_id} for ABBREVIATED-format matching, plus {fullname_lower:
    player_id} for the alt full-name format.

    A short abbreviation (e.g. "C.Williams") claimed by more than one
    player on the same team is a REAL collision — but confirmed live
    (Chicago 2024: Caleb Williams AND Chris Williams both reduce to
    "C.Williams") that ESPN's own play text doesn't split these evenly:
    it keeps the short form for whichever player is more featured and
    quietly extends the OTHER one's prefix instead (confirmed: that game's
    text used "C.Williams" for the starting QB throughout and "Ch.Williams"
    for the depth player, never the reverse). So the short form is
    resolved to whichever colliding player has the larger _stat_volume()
    in this game, rather than dropped outright — the longer, auto-
    disambiguated candidates (ESPN's own extension, generated below too)
    still correctly resolve the other player independently."""
    abbrev_counts = defaultdict(set)
    volume_by_id = {}
    full_index = {}
    for p in team_players:
        volume_by_id[p['id']] = _stat_volume(p.get('categories'))
        for cand in _abbrev_candidates(p['name']):
            abbrev_counts[cand.lower()].add(p['id'])
        full_index[_SUFFIX_RE.sub('', p['name']).strip().lower()] = p['id']
    abbrev_index = {
        k: (next(iter(v)) if len(v) == 1 else max(v, key=lambda pid: volume_by_id.get(pid, 0.0)))
        for k, v in abbrev_counts.items()
    }
    return abbrev_index, full_index


def _resolve(name_token, abbrev_index, full_index):
    if not name_token:
        return None
    key = name_token.strip().lower()
    return abbrev_index.get(key) or full_index.get(key)


def _is_redzone(play):
    return (play.get('start') or {}).get('yardsToEndzone', 999) <= REDZONE_YARDS_TO_GOAL


def _is_no_play(text):
    """True if a penalty on this play replayed the down (the play's own
    rush/pass/sack result — and penalty-adjusted statYardage — must NOT
    be credited to anyone's stats). False for a declined penalty, which
    changes nothing about the original play's result."""
    return bool(_NO_PLAY_RE.search(text))


def _parse_penalty_clause(text, abbrev_to_team_id, name_index_by_team):
    """Finds a 'PENALTY on TEAM[-PLAYER], TYPE, [N yards]' clause
    anywhere in `text` (standalone Penalty-type plays, or embedded at the
    end of another play's text) and resolves it to {team_id, player_id,
    type, yards, status}. None if no such clause is present. No reliance
    on the play's own structured `penalty` field — confirmed absent
    entirely for 2024-season games (present for 2025), so text is the
    only consistent source across seasons this app has data for."""
    m = _PENALTY_RE.search(text)
    if not m:
        return None
    team_id = abbrev_to_team_id.get(m.group(1).upper())
    player_id = None
    if team_id and m.group(2):
        abbrev_idx, full_idx = name_index_by_team.get(team_id, ({}, {}))
        player_id = _resolve(m.group(2), abbrev_idx, full_idx)
    status = 'Declined' if _PENALTY_DECLINED_RE.search(text) else ('No Play' if _is_no_play(text) else 'Accepted')
    yards = int(m.group(4)) if m.group(4) else None
    return {
        'team_id': team_id, 'player_id': player_id,
        'type': m.group(3).strip() if m.group(3) else None,
        'yards': yards, 'status': status,
    }


def _new_player_row():
    return {
        'rush_att': 0, 'rush_yds': 0.0, 'rush_td': 0,
        'long_rush_att': 0, 'long_rush_yds': 0.0,
        'redzone_rush_att': 0, 'redzone_rush_td': 0,
        'targets': 0, 'receptions': 0, 'rec_yds': 0.0, 'rec_td': 0, 'yac': 0.0,
        'deep_tgt': 0, 'deep_rec': 0, 'deep_rec_yds': 0.0,
        'short_tgt': 0, 'short_rec': 0, 'short_rec_yds': 0.0,
        'redzone_tgt': 0, 'redzone_rec_td': 0,
        'pass_att': 0, 'pass_comp': 0, 'pass_yds': 0.0, 'pass_td': 0, 'sacks_taken': 0,
        'deep_pass_att': 0, 'deep_pass_comp': 0, 'deep_pass_yds': 0.0,
        'short_pass_att': 0, 'short_pass_comp': 0, 'short_pass_yds': 0.0,
        'redzone_pass_att': 0, 'redzone_pass_td': 0,
    }


def _build_abbrev_to_team_id(summary):
    """{abbrev: team_id} (both the real one and any _TEAM_ABBREV_ALIASES
    variant) from the header's competitor list — the one place in this
    payload that reliably pairs a team's id with its abbreviation (the
    boxscore section, used for name resolution elsewhere in this module,
    only has ids, no abbreviations)."""
    out = {}
    competitors = (((summary or {}).get('header') or {}).get('competitions') or [{}])[0].get('competitors', [])
    for c in competitors:
        team = c.get('team') or {}
        abbrev, team_id = team.get('abbreviation'), team.get('id')
        if abbrev and team_id:
            out[abbrev] = team_id
            alias = _TEAM_ABBREV_ALIASES.get(abbrev)
            if alias:
                out[alias] = team_id
    return out


def parse_play_splits(summary):
    """summary: a full `summary?event=` response (same input as
    nfl_boxscore_api.parse_player_boxscore()). Returns {'player_splits':
    {player_id: {...see _new_player_row()}}, 'team_allowed': {team_id:
    {'long_plays_allowed', 'redzone_trips_allowed', 'redzone_tds_allowed'}},
    'raw_plays': [...], 'skipped': int} — {} shape with empty dicts/lists
    if there's no play-by-play yet (pre-game, or this endpoint simply
    didn't carry it for an old enough game)."""
    from nfl_boxscore_api import parse_player_boxscore

    drives = ((summary or {}).get('drives') or {}).get('previous') or []
    if not drives:
        return {'player_splits': {}, 'team_allowed': {}, 'raw_plays': [], 'skipped': 0}

    boxscore_by_team = parse_player_boxscore(summary)
    name_index_by_team = {
        team_id: _build_name_index(players) for team_id, players in boxscore_by_team.items()
    }
    abbrev_to_team_id = _build_abbrev_to_team_id(summary)

    player_splits = defaultdict(_new_player_row)
    team_allowed = defaultdict(lambda: {'long_plays_allowed': 0, 'redzone_trips_allowed': 0, 'redzone_tds_allowed': 0})
    raw_plays = []
    skipped = 0

    for drive in drives:
        off_team = (drive.get('team') or {}).get('id')
        plays = drive.get('plays') or []
        if not off_team or not plays:
            continue
        # Defense = whichever other team appears as a play's own team ref;
        # simplest robust source is the drive's own opposing side, which
        # isn't directly given — derive from any play's start.team (offense)
        # vs the two team ids already known from name_index_by_team.
        other_teams = [t for t in name_index_by_team if t != off_team]
        def_team = other_teams[0] if len(other_teams) == 1 else None

        abbrev_idx, full_idx = name_index_by_team.get(off_team, ({}, {}))
        drive_redzone = False
        prev_start = None

        for play in plays:
            text = (play.get('text') or '').strip()
            ptype = (play.get('type') or {}).get('text', '')
            start = play.get('start') or {}
            if not text or _SKIP_RE.search(text):
                continue
            # ESPN duplicate-logs a play when a penalty on it is DECLINED:
            # a normal entry, then a second "administrative" one with the
            # same down/distance/yardLine (confirmed live — Washington
            # @ Cleveland 2024, a 13-yard J.Daniels scramble logged twice,
            # second one tagged "declined", identical start position).
            # Since nothing actually changed, the dupe is dropped outright
            # rather than counted as a second play.
            same_start = (prev_start is not None and start.get('down') == prev_start.get('down')
                          and start.get('distance') == prev_start.get('distance')
                          and start.get('yardLine') == prev_start.get('yardLine'))
            prev_start = start
            if same_start and _PENALTY_DECLINED_RE.search(text):
                continue
            is_rz = _is_redzone(play)
            if is_rz:
                drive_redzone = True
            yards = play.get('statYardage') or 0
            is_td = bool(play.get('scoringPlay')) and ptype in ('Rushing Touchdown', 'Passing Touchdown')
            no_play = ptype != 'Penalty' and _is_no_play(text)
            base_record = {
                'play_id': play.get('id'), 'drive_id': drive.get('id'),
                'sequence_number': play.get('sequenceNumber'),
                'offense_team_id': off_team, 'defense_team_id': def_team,
                'play_type': ptype, 'text': text,
                'down': start.get('down'), 'distance': start.get('distance'),
                'yards_to_endzone': start.get('yardsToEndzone'),
                'yards_gained': yards, 'is_scoring_play': bool(play.get('scoringPlay')),
                'is_redzone': is_rz, 'is_turnover': bool(play.get('isTurnover')),
                'is_no_play': no_play,
                'period': (play.get('period') or {}).get('number'),
                'clock': (play.get('clock') or {}).get('displayValue'),
                'pass_depth': None, 'yards_after_catch': None,
                'passer_id': None, 'rusher_id': None, 'receiver_id': None,
                'penalty_team_id': None, 'penalty_player_id': None,
                'penalty_type': None, 'penalty_yards': None, 'penalty_status': None,
            }

            if ptype == 'Penalty':
                pen = _parse_penalty_clause(text, abbrev_to_team_id, name_index_by_team)
                if pen:
                    base_record['penalty_team_id'] = pen['team_id']
                    base_record['penalty_player_id'] = pen['player_id']
                    base_record['penalty_type'] = pen['type']
                    base_record['penalty_yards'] = pen['yards']
                    base_record['penalty_status'] = pen['status']
                    base_record['is_no_play'] = pen['status'] == 'No Play'
                else:
                    skipped += 1
                raw_plays.append(base_record)
                continue

            if no_play:
                # The down was replayed — this play's own rush/pass/sack
                # result must not be credited to anyone (see
                # _parse_penalty_clause's docstring). Still parse out the
                # embedded penalty clause for the raw record.
                pen = _parse_penalty_clause(text, abbrev_to_team_id, name_index_by_team)
                if pen:
                    base_record['penalty_team_id'] = pen['team_id']
                    base_record['penalty_player_id'] = pen['player_id']
                    base_record['penalty_type'] = pen['type']
                    base_record['penalty_yards'] = pen['yards']
                    base_record['penalty_status'] = pen['status']

            if ptype in ('Rush', 'Rushing Touchdown'):
                m = _RUSH_RE.match(text)
                pid = _resolve(m.group(1), abbrev_idx, full_idx) if m else None
                if not pid:
                    m2 = _ALT_RUSH_RE.match(text)
                    pid = _resolve(m2.group(1), {}, full_idx) if m2 else None
                base_record['rusher_id'] = pid
                raw_plays.append(base_record)
                if no_play:
                    continue
                if not pid:
                    skipped += 1
                    if def_team and yards >= LONG_RUN_YARDS:
                        team_allowed[def_team]['long_plays_allowed'] += 1
                    continue
                row = player_splits[pid]
                row['rush_att'] += 1
                row['rush_yds'] += yards
                if is_td:
                    row['rush_td'] += 1
                if yards >= LONG_RUN_YARDS:
                    row['long_rush_att'] += 1
                    row['long_rush_yds'] += yards
                    if def_team:
                        team_allowed[def_team]['long_plays_allowed'] += 1
                if is_rz:
                    row['redzone_rush_att'] += 1
                    if is_td:
                        row['redzone_rush_td'] += 1

            elif ptype in ('Pass Reception', 'Pass Incompletion', 'Passing Touchdown'):
                m = _PASS_RE.match(text)
                passer_pid = receiver_pid = None
                depth = None
                if m:
                    passer_pid = _resolve(m.group(1), abbrev_idx, full_idx)
                    depth = m.group(2)
                    receiver_pid = _resolve(m.group(3), abbrev_idx, full_idx) if m.group(3) else None
                else:
                    m2 = _ALT_PASS_RE.match(text)
                    if m2:
                        receiver_pid = _resolve(m2.group(1), {}, full_idx)
                        passer_pid = _resolve(m2.group(2), {}, full_idx) if m2.group(2) else None
                        depth = 'deep' if yards >= 20 else 'short'  # alt format carries no explicit depth tag

                is_complete = ptype in ('Pass Reception', 'Passing Touchdown')
                yac = play.get('yardsAfterCatch') or 0.0
                base_record['passer_id'] = passer_pid
                base_record['receiver_id'] = receiver_pid
                base_record['pass_depth'] = depth
                base_record['yards_after_catch'] = yac
                raw_plays.append(base_record)

                if no_play:
                    continue
                if passer_pid is None and receiver_pid is None:
                    skipped += 1
                    if def_team and yards >= LONG_RUN_YARDS:
                        team_allowed[def_team]['long_plays_allowed'] += 1
                    continue

                if passer_pid:
                    prow = player_splits[passer_pid]
                    prow['pass_att'] += 1
                    if is_complete:
                        prow['pass_comp'] += 1
                        prow['pass_yds'] += yards
                    if is_td:
                        prow['pass_td'] += 1
                    if depth == 'deep':
                        prow['deep_pass_att'] += 1
                        if is_complete:
                            prow['deep_pass_comp'] += 1
                            prow['deep_pass_yds'] += yards
                    elif depth == 'short':
                        prow['short_pass_att'] += 1
                        if is_complete:
                            prow['short_pass_comp'] += 1
                            prow['short_pass_yds'] += yards
                    if is_rz:
                        prow['redzone_pass_att'] += 1
                        if is_td:
                            prow['redzone_pass_td'] += 1

                if receiver_pid:
                    rrow = player_splits[receiver_pid]
                    rrow['targets'] += 1
                    if is_complete:
                        rrow['receptions'] += 1
                        rrow['rec_yds'] += yards
                        rrow['yac'] += yac
                    if is_td:
                        rrow['rec_td'] += 1
                    if depth == 'deep':
                        rrow['deep_tgt'] += 1
                        if is_complete:
                            rrow['deep_rec'] += 1
                            rrow['deep_rec_yds'] += yards
                    elif depth == 'short':
                        rrow['short_tgt'] += 1
                        if is_complete:
                            rrow['short_rec'] += 1
                            rrow['short_rec_yds'] += yards
                    if is_rz:
                        rrow['redzone_tgt'] += 1
                        if is_td:
                            rrow['redzone_rec_td'] += 1

                if def_team and is_complete and yards >= LONG_RUN_YARDS:
                    team_allowed[def_team]['long_plays_allowed'] += 1

            elif ptype == 'Sack':
                m = _SACK_RE.match(text)
                pid = _resolve(m.group(1), abbrev_idx, full_idx) if m else None
                base_record['passer_id'] = pid
                raw_plays.append(base_record)
                if no_play:
                    continue
                if pid:
                    player_splits[pid]['sacks_taken'] += 1
                else:
                    skipped += 1

        if drive_redzone and def_team:
            team_allowed[def_team]['redzone_trips_allowed'] += 1
            if drive.get('result') == 'TD':
                team_allowed[def_team]['redzone_tds_allowed'] += 1

    return {
        'player_splits': dict(player_splits),
        'team_allowed': dict(team_allowed),
        'raw_plays': raw_plays,
        'skipped': skipped,
    }
