"""Discord alert routing — which channel (webhook) each alert type posts to,
and whether the user has that alert switched on. Settings → Notifications
edits both; everything that posts to Discord should resolve its webhook
through webhook_for()/should_send() here rather than reading
Setting.discord_webhook_url directly, so per-channel routing and the
on/off toggles apply everywhere.

Storage (both on the single Setting row, JSON like favorite_teams_json):
  discord_channels_json  {channel_key: webhook URL}
  alert_prefs_json       {alert_key: true/false}
Setting.discord_webhook_url stays the "Default" channel — any alert whose
channel has no URL of its own falls back to it, so nothing stops posting
until every channel exists.

Set DISCORD_DRY_RUN=1 to print payloads instead of POSTing them.
"""
import json
import os

import requests

# Order here is the order Settings shows them in.
CHANNELS = {
    'default':  {'label': 'Default',           'hint': 'Catch-all. Anything without its own channel lands here.'},
    'my_bets':  {'label': 'My bets',           'hint': 'Line moves, injuries, lead changes and finals on games you have money on.'},
    'model':    {'label': 'Model',             'hint': "Tonight's card, Omen Score moves, picks added or dropped."},
    'scores':   {'label': 'Scores',            'hint': 'Final scores across the board.'},
    'injuries': {'label': 'Injuries & lineups', 'hint': "Starters ruled out on tonight's slate."},
}

# `live` = a sender is actually wired up today. The rest are defined so the
# toggles and routing exist now; their triggers land with the Phase 6 build.
ALERTS = {
    'daily_card':     {'label': "The model posts tonight's card",     'channel': 'model',    'default': True,  'live': True},
    'score_moves':    {'label': 'Omen Score moves ±10 / crosses 60',  'channel': 'model',    'default': True,  'live': True},
    'picks_changed':  {'label': 'A pick is added to or drops off the list', 'channel': 'model', 'default': True, 'live': True},
    'bet_line_move':  {'label': 'A line moves 1+ point on my bet',     'channel': 'my_bets',  'default': True,  'live': False},
    'bet_starter_out': {'label': 'A starter is ruled out in my game',  'channel': 'my_bets',  'default': True,  'live': False},
    'bet_lead_change': {'label': 'The lead changes in my game',        'channel': 'my_bets',  'default': True,  'live': False},
    'any_final':      {'label': 'Any final score',                     'channel': 'scores',   'default': False, 'live': False},
    'starter_out':    {'label': "A starter is ruled out anywhere on tonight's slate", 'channel': 'injuries', 'default': False, 'live': False},
}


def _load(raw):
    try:
        v = json.loads(raw or '{}')
        return v if isinstance(v, dict) else {}
    except (TypeError, ValueError):
        return {}


def channel_urls(settings):
    """{channel_key: url} with Default filled from discord_webhook_url."""
    urls = {k: (v or '').strip() for k, v in _load(getattr(settings, 'discord_channels_json', '')).items()
            if k in CHANNELS}
    urls['default'] = (getattr(settings, 'discord_webhook_url', '') or '').strip()
    return urls


def prefs(settings):
    """{alert_key: bool}, defaults filled in for anything never saved."""
    saved = _load(getattr(settings, 'alert_prefs_json', ''))
    return {k: bool(saved.get(k, a['default'])) for k, a in ALERTS.items()}


def webhook_for(settings, alert_key):
    """Webhook URL this alert should post to ('' if none configured) —
    its channel's own URL, else Default."""
    if not settings:
        return ''
    urls = channel_urls(settings)
    channel = ALERTS.get(alert_key, {}).get('channel', 'default')
    return urls.get(channel) or urls.get('default', '')


def should_send(settings, alert_key):
    """Webhook URL if this alert is switched on and has somewhere to go,
    else ''. The one call senders need: `if (url := should_send(...)):`."""
    if not settings or not prefs(settings).get(alert_key, False):
        return ''
    return webhook_for(settings, alert_key)


def post(url, payload, timeout=10):
    """POST a Discord webhook payload; returns the status code (0 on
    failure, 204 when DISCORD_DRY_RUN printed it instead)."""
    if not url:
        return 0
    if os.environ.get('DISCORD_DRY_RUN') == '1':
        print(f'[discord dry-run] → {url[:60]}…\n{json.dumps(payload)[:2000]}', flush=True)
        return 204
    try:
        return requests.post(url, json=payload, timeout=timeout).status_code
    except Exception:
        return 0
