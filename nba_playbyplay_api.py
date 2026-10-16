"""
nba_playbyplay_api.py — Play-by-play parsing for NBA, sourced from the
SAME ESPN `summary?event={id}` response nba_boxscore_api.py already reads
(`plays[]` — the box score only reads `boxscore`, this reads a different
key off the identical payload, so this costs zero extra API calls when a
caller already has `summary` in hand — see nba_boxscore_api.get_live_
summary()).

Unlike nfl_playbyplay_api.py, NBA's free play-by-play carries a real,
structured `participants[].athlete.id` list on every play — no prose-
regex name resolution needed, and no games where attribution silently
fails. A play can have multiple participants (e.g. a made shot: shooter
+ assister; a rebound: the reboundee only; a block: shooter + blocker),
stored as the raw ordered id list rather than this module guessing which
slot means what — ESPN doesn't label the slots, so splitting them into
separate shooter_id/assist_id/etc. columns the way NflPlay does for
passer_id/rusher_id/receiver_id would be inventing structure that isn't
actually there.

Stores every play verbatim (see NbaPlay's docstring for why raw events are
kept rather than just aggregated splits) — including shot location
(coordinate.x/y) and the point value attempted, which together make a
real shot chart possible later without re-fetching anything.
"""


def parse_plays(summary, game_id):
    """Returns a list of play dicts matching NbaPlay's columns, parsed from
    summary['plays']. [] if summary has no plays (pre-game, or a fetch
    failure upstream already returned {})."""
    plays = (summary or {}).get('plays') or []
    out = []
    for p in plays:
        play_id = p.get('id')
        if not play_id:
            continue
        ptype = p.get('type') or {}
        period = (p.get('period') or {}).get('number')
        clock = (p.get('clock') or {}).get('displayValue')
        coord = p.get('coordinate') or {}
        shot_x, shot_y = coord.get('x'), coord.get('y')
        # ESPN returns a garbage sentinel (~-214748340/-214748365, an
        # overflow-looking placeholder, not a real court position) instead
        # of omitting the key on non-shooting plays (jump balls, timeouts,
        # etc.) — confirmed live on a Jumpball play. A real court coordinate
        # is always a small number (roughly 0-94), so anything with an
        # absolute value this large is that sentinel, not data.
        if shot_x is not None and abs(shot_x) > 1000:
            shot_x = None
        if shot_y is not None and abs(shot_y) > 1000:
            shot_y = None
        participant_ids = [
            part.get('athlete', {}).get('id')
            for part in (p.get('participants') or [])
            if part.get('athlete', {}).get('id')
        ]
        out.append({
            'game_id':           game_id,
            'play_id':           play_id,
            'sequence_number':   p.get('sequenceNumber'),
            'team_id':           (p.get('team') or {}).get('id'),
            'play_type_id':      ptype.get('id'),
            'play_type':         ptype.get('text'),
            'text':              p.get('text'),
            'short_description': p.get('shortDescription'),
            'period':            period,
            'clock':             clock,
            'home_score':        p.get('homeScore'),
            'away_score':        p.get('awayScore'),
            'score_value':       p.get('scoreValue'),
            'is_scoring_play':   bool(p.get('scoringPlay')),
            'is_shooting_play':  bool(p.get('shootingPlay')),
            'points_attempted':  p.get('pointsAttempted'),
            'shot_x':            shot_x,
            'shot_y':            shot_y,
            'participant_ids':   participant_ids,
        })
    return out
