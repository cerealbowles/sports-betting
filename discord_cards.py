"""
discord_cards.py — Team-branded PNG cards for Discord alerts.

Currently used only for "High Confidence Pick" alerts (Unified Score
crossing 60) — see _check_unified_score_alerts in app.py. Regular Unified
Score movement alerts stay plain text embeds; per-alert images for every
movement would spam the channel on a busy slate.

Uses Pillow + ESPN's PNG team-logo CDN (see team_branding.py for why PNG
over MLB's own SVG CDN — avoids needing an SVG rasterizer as a system
dependency). Font is bundled in assets/fonts/ rather than relying on the
Docker base image (python:3.11-slim) having any fonts installed, which it
doesn't by default.

Logos are cached to disk on first fetch (assets/logos/) — they never
change, so there's no reason to hit ESPN's CDN more than once per team ever.
"""
import io
import os

import requests
from PIL import Image, ImageDraw, ImageFont

import team_branding

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONT_DIR  = os.path.join(_HERE, 'assets', 'fonts')
_LOGO_DIR  = os.path.join(_HERE, 'assets', 'logos')
_LOGO_URL  = 'https://a.espncdn.com/i/teamlogos/mlb/500/{}.png'

_FONT_BOLD = os.path.join(_FONT_DIR, 'DejaVuSans-Bold.ttf')
_FONT_REG  = os.path.join(_FONT_DIR, 'DejaVuSans.ttf')

W, H = 900, 320


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _hex(c):
    c = c.lstrip('#')
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _get_logo(abbr):
    """Returns a PIL Image (RGBA) for the team's logo, disk-cached. None on
    any failure — callers must render a text-only fallback badge in that case,
    not crash the whole alert over a CDN hiccup."""
    b = team_branding.get(abbr)
    slug = b.get('logo_slug')
    if not slug:
        return None

    os.makedirs(_LOGO_DIR, exist_ok=True)
    cache_path = os.path.join(_LOGO_DIR, f'{slug}.png')

    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert('RGBA')
        except Exception:
            pass  # corrupt cache file — fall through and re-fetch

    try:
        r = requests.get(_LOGO_URL.format(slug), timeout=8)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert('RGBA')
        try:
            img.save(cache_path)
        except Exception:
            pass  # caching is an optimization, not a requirement
        return img
    except Exception as e:
        print(f'[discord_cards] logo fetch failed for {abbr} ({slug}): {e}', flush=True)
        return None


def _circular(img, size):
    """Resize to a size x size square (cover-crop, not stretch) and mask to
    a circle."""
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    img = img.resize((size, size), Image.LANCZOS)
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def _strip_md(text):
    """_snapshot_reason_bits() formats its output for Discord's markdown
    renderer (**bold** around the new value) — Pillow just draws literal
    asterisk characters, so they need stripping before they hit the image."""
    return text.replace('**', '')


def _wrap(draw, text, font, max_width):
    """Greedy word-wrap using actual glyph measurement (not a fixed char
    count) — team names and reasoning text vary too much in width to guess."""
    words = text.split()
    lines, cur = [], ''
    for w in words:
        trial = f'{cur} {w}'.strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_score_move_card(abbr, prev_score, curr_score, matchup_line, why_lines=None,
                           label='HIGH CONFIDENCE PICK'):
    """
    Returns PNG bytes for a team-branded score-movement card.

    abbr          – MLB abbreviation (e.g. 'COL') — looked up in team_branding.
    prev_score    – prior Unified Score, or None on a first-observation alert
                    (no meaningful "previous" value to show — renders as
                    just the current score, no arrow).
    curr_score    – current Unified Score.
    matchup_line  – e.g. "COL @ SD · Pick: COL ML · -108"
    why_lines     – list[str] from _snapshot_reason_bits, or None/[] to omit
                    the section entirely (e.g. first-observation alerts have
                    no prior snapshot to diff against).
    """
    b = team_branding.get(abbr)
    primary   = _hex(b['primary'])
    secondary = _hex(b['secondary'])
    accent    = _hex(b['accent'])
    team_name = b.get('name') or abbr

    img  = Image.new('RGB', (W, H), primary)
    draw = ImageDraw.Draw(img)

    # Subtle diagonal darker panel for depth — flat color blocks, no
    # gradients (keeps this looking like a sports scoreboard, not a poster).
    dark = tuple(max(0, c - 30) for c in primary)
    draw.polygon([(0, H), (W, H), (W, H * 0.35), (0, H)], fill=dark)

    # Logo badge
    badge_cx, badge_cy, badge_r = 150, H // 2, 108
    draw.ellipse((badge_cx - badge_r, badge_cy - badge_r, badge_cx + badge_r, badge_cy + badge_r),
                 fill=secondary)
    logo = _get_logo(abbr)
    if logo is not None:
        logo_size = int(badge_r * 1.5)
        circ = _circular(logo, logo_size)
        img.paste(circ, (badge_cx - logo_size // 2, badge_cy - logo_size // 2), circ)
    else:
        # Fallback: team abbreviation lettering, still on-brand via the
        # badge's secondary-color fill — never block the alert on a CDN miss.
        f = _font(_FONT_BOLD, 40)
        tw = draw.textlength(abbr, font=f)
        draw.text((badge_cx - tw / 2, badge_cy - 24), abbr, font=f, fill=primary)

    draw.text((badge_cx, badge_cy + badge_r + 16), team_name.upper(),
              font=_font(_FONT_BOLD, 15), fill=(245, 246, 248), anchor='ma')

    # Right column
    x0 = 330
    draw.text((x0, 46), label + ' · MLB', font=_font(_FONT_REG, 15), fill=accent)

    score_font = _font(_FONT_BOLD, 58)
    if prev_score is not None:
        score_text = f'{prev_score} → {curr_score}'
    else:
        score_text = f'{curr_score}'
    draw.text((x0, 78), score_text, font=score_font, fill=(245, 246, 248))

    if prev_score is not None:
        delta = curr_score - prev_score
        up = delta > 0
        pill_color = (23, 52, 4) if up else (74, 19, 19)
        text_color = (151, 196, 89) if up else (224, 122, 122)
        arrow = '▲' if up else '▼'
        pill_text = f'{arrow} {delta:+d}'
        pf = _font(_FONT_BOLD, 20)
        tw = draw.textlength(pill_text, font=pf)
        pill_w = int(tw + 36)
        pill_y0 = 148
        draw.rounded_rectangle((x0, pill_y0, x0 + pill_w, pill_y0 + 38), radius=19, fill=pill_color)
        draw.text((x0 + pill_w / 2, pill_y0 + 19), pill_text, font=pf, fill=text_color, anchor='mm')

    draw.text((x0, 210), matchup_line, font=_font(_FONT_REG, 18), fill=(230, 220, 242))

    if why_lines:
        wf = _font(_FONT_REG, 14)
        max_w = W - x0 - 40
        y = 244
        # Cap at 3 lines total across all reasons so the card never runs off
        # the bottom — the full breakdown is still in the text portion of
        # the Discord message, this is just the visual highlight.
        rendered = 0
        for reason in why_lines:
            for line in _wrap(draw, f'• {_strip_md(reason)}', wf, max_w):
                if rendered >= 3:
                    break
                draw.text((x0, y), line, font=wf, fill=(196, 206, 212))
                y += 20
                rendered += 1
            if rendered >= 3:
                break

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()
