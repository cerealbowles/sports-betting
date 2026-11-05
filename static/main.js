/* ── Toast ─────────────────────────────────────────────
   Fading bottom-center popup. showToast() displays immediately on the current page; stashToast()
   persists a message in sessionStorage for the page we're about to
   navigate to, which picks it up via the DOMContentLoaded check below. */
var _toastTimer = null;
function showToast(message) {
var el = document.getElementById('bmToast');
if (!el) return;
el.textContent = message;
el.classList.add('show');
clearTimeout(_toastTimer);
_toastTimer = setTimeout(function () { el.classList.remove('show'); }, 2600);
}
function stashToast(message) {
try { sessionStorage.setItem('bmPendingToast', message); } catch (e) {}
}
document.addEventListener('DOMContentLoaded', function () {
var msg;
try { msg = sessionStorage.getItem('bmPendingToast'); } catch (e) { msg = null; }
if (!msg) return;
try { sessionStorage.removeItem('bmPendingToast'); } catch (e) {}
showToast(msg);
});

/* ── Lines toggle (header) ────────────────────────────── */
function toggleLinesVisibility(btn) {
var hide = !document.body.classList.contains('hide-lines');
document.body.classList.toggle('hide-lines', hide);
btn.classList.toggle('off', hide);
btn.setAttribute('aria-pressed', hide ? 'false' : 'true');
fetch('/api/toggle-lines', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({show_lines: !hide}),
}).catch(function () {});
}

/* ── Header sport switcher dropdown ───────────────────── */
function toggleSportSwitch(e) {
e.stopPropagation();
var trigger = e.currentTarget;
var menu = trigger.nextElementSibling;
var open = menu.classList.toggle('open');
trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
}
document.addEventListener('click', function (e) {
document.querySelectorAll('.sport-switch-menu.open').forEach(function (menu) {
    if (!menu.parentElement.contains(e.target)) {
        menu.classList.remove('open');
        var trigger = menu.previousElementSibling;
        if (trigger) trigger.setAttribute('aria-expanded', 'false');
    }
});
});
document.addEventListener('keydown', function (e) {
if (e.key !== 'Escape') return;
document.querySelectorAll('.sport-switch-menu.open').forEach(function (menu) {
    menu.classList.remove('open');
    var trigger = menu.previousElementSibling;
    if (trigger) trigger.setAttribute('aria-expanded', 'false');
});
});

/* ── Game details bottom sheet ────────────────────────
   Tapping a card's .game-card-summary slides up its .game-card-details node
   (full stats + bet-type chips) as a bottom sheet. The node is moved into
   the shared sheet body, not cloned or fetched — a placeholder marks its
   original spot in the card so it can be moved back unchanged on close. */
var openDetailsKey = null;
var gameDetailsCloseTimer = null;

function openGameDetails(key) {
if (openDetailsKey === key) return;
if (openDetailsKey) restoreGameDetails(openDetailsKey);
var details = document.getElementById('gd-' + key);
var body    = document.getElementById('gameDetailsSheetBody');
var overlay = document.getElementById('gameDetailsOverlay');
if (!details || !body || !overlay) return;
clearTimeout(gameDetailsCloseTimer);
var placeholder = document.createElement('div');
placeholder.id = 'gd-ph-' + key;
placeholder.style.display = 'none';
details.parentNode.insertBefore(placeholder, details);
details.hidden = false;
body.appendChild(details);
openDetailsKey = key;
overlay.style.display = 'flex';
overlay.offsetHeight; // force reflow so the slide-up transition runs
overlay.classList.add('open');
document.body.style.overflow = 'hidden';
// Lets per-sport scripts (e.g. MLB's lazy live-boxscore fetch) react to a
// details panel becoming visible without main.js knowing sport specifics.
document.dispatchEvent(new CustomEvent('gamedetailsopen', { detail: { key: key, el: details } }));
}

function restoreGameDetails(key) {
var details = document.getElementById('gd-' + key);
var placeholder = document.getElementById('gd-ph-' + key);
if (details && placeholder) {
    details.hidden = true;
    placeholder.replaceWith(details);
}
}

function closeGameDetails() {
var overlay = document.getElementById('gameDetailsOverlay');
if (!overlay || !overlay.classList.contains('open')) return;
overlay.classList.remove('open');
document.body.style.overflow = '';
clearTimeout(gameDetailsCloseTimer);
var key = openDetailsKey;
openDetailsKey = null;
gameDetailsCloseTimer = setTimeout(function () {
    overlay.style.display = 'none';
    if (key) restoreGameDetails(key);
}, 300);
}

function closeGameDetailsOutside(e) {
if (e.target === e.currentTarget) closeGameDetails();
}

/* Switches which of ML/Spread/Total is visible inside the single shared
   factors panel (_factors_panel.html) — all three render through the same
   chart (_factor_diverge_panel.html), overlaid in the same spot rather
   than stacked as separate sections. The market chips themselves
   (_market_chips.html) are the tab switcher now — there's no separate
   tab row — so this also mirrors the "active" state onto whichever chip
   matches `kind`. `container` is the `.factors-panel` div (there's
   exactly one per game, a sibling of `.mc-chips` inside `.mc-wrap`). */
function switchFactorsTab(container, kind) {
container.querySelectorAll('.mf-panel').forEach(function (p) {
    p.hidden = p.getAttribute('data-mf-panel') !== kind;
});
var wrap = container.closest('.mc-wrap');
if (wrap) {
    wrap.querySelectorAll('.mc-chip[data-mc-kind]').forEach(function (c) {
    c.classList.toggle('mc-chip-active', c.getAttribute('data-mc-kind') === kind);
    });
}
}

/* Recap lineup tabs (nfl_recap.html) — Recommended/Best possible/Sim
   Median/Sim Ceiling used to render as a cramped 2x2 grid of four full
   tables per week; now only the picked tab's table is visible. */
function switchRecapTab(container, kind) {
container.querySelectorAll('.recap-tab').forEach(function (b) {
    b.classList.toggle('active', b.getAttribute('data-recap-tab') === kind);
});
container.querySelectorAll('.recap-panel').forEach(function (p) {
    p.hidden = p.getAttribute('data-recap-panel') !== kind;
});
}
document.addEventListener('click', function (e) {
var tab = e.target.closest('.recap-tab');
if (!tab) return;
var container = tab.closest('.recap-subtables-wrap');
if (container) switchRecapTab(container, tab.getAttribute('data-recap-tab'));
});

/* Recap "Top by actual points, per position" tabs — QB/RB/WR/TE/DST used
   to render as five tables side by side in an auto-fit grid; now tabbed
   the same way as the lineup tables above (shares the .recap-tab/
   .recap-panel classes for styling, keyed off data-recap-postab instead
   of data-recap-tab so the two click handlers don't fight over which
   panel is "active"). */
function switchRecapPosTab(container, kind) {
container.querySelectorAll('.recap-tab').forEach(function (b) {
    b.classList.toggle('active', b.getAttribute('data-recap-postab') === kind);
});
container.querySelectorAll('.recap-panel').forEach(function (p) {
    p.hidden = p.getAttribute('data-recap-pospanel') !== kind;
});
}
document.addEventListener('click', function (e) {
var tab = e.target.closest('[data-recap-postab]');
if (!tab) return;
var container = tab.closest('.recap-postabs-wrap');
if (container) switchRecapPosTab(container, tab.getAttribute('data-recap-postab'));
});

/* Market-chips row (ML/Spread, _market_chips.html) — tapping a chip opens
   the details sheet, opens the shared factors panel (_factors_panel.html)
   and switches it to the tapped chip's tab. */
function openModelFactors(gameKey, kind) {
openGameDetails(gameKey);
var details = document.getElementById('gd-' + gameKey);
if (!details) return;
setTimeout(function () {
    // getElementById, not querySelector('#...') — game_key routinely
    // contains spaces/dots (e.g. "st. louis cardinals_..."), which are
    // valid in an id attribute but break an unescaped CSS #id selector.
    var target = document.getElementById('factors-' + gameKey);
    if (!target || !details.contains(target)) return;
    switchFactorsTab(target, kind);
    target.scrollIntoView({behavior: 'smooth', block: 'start'});
}, 350);
}

document.addEventListener('click', function (e) {
var summary = e.target.closest('.game-card-summary');
if (!summary) return;
var card = summary.closest('.game-card');
if (!card || !card.id) return;
// NFL cards link to a full game-detail page instead of opening the
// bottom sheet (see _nfl_game_card.html's data-detail-href and
// game_detail.html) — every other sport still uses the sheet.
if (card.dataset.detailHref) { window.location.href = card.dataset.detailHref; return; }
openGameDetails(card.id.replace(/^gc-/, ''));
});

document.addEventListener('keydown', function (e) {
if (e.key !== 'Enter' && e.key !== ' ') return;
if (!e.target.classList || !e.target.classList.contains('game-card-summary')) return;
e.preventDefault();
e.target.click();
});

// ── Header search (base.html #siteSearch) ─────────────────────────────────
// Typeahead over /api/search; Enter without picking a result submits to
// the full /search page. On phones the field is collapsed behind the icon
// button until toggled open.
function toggleSiteSearch(btn) {
    var form = document.getElementById('siteSearch');
    var open = form.classList.toggle('open');
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) document.getElementById('siteSearchInput').focus();
}
(function () {
    var input = document.getElementById('siteSearchInput');
    var list = document.getElementById('siteSearchResults');
    if (!input || !list) return;
    var timer = null, seq = 0, active = -1;
    function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }
    function close() { list.hidden = true; list.innerHTML = ''; active = -1; }
    function setActive(i) {
        var items = list.querySelectorAll('a');
        items.forEach(function (a, j) { a.classList.toggle('active', j === i); a.setAttribute('aria-selected', j === i ? 'true' : 'false'); });
        active = i;
    }
    input.addEventListener('input', function () {
        clearTimeout(timer);
        var q = input.value.trim();
        if (q.length < 2) { close(); return; }
        timer = setTimeout(function () {
            var mine = ++seq;
            fetch('/api/search?q=' + encodeURIComponent(q)).then(function (r) { return r.json(); }).then(function (rows) {
                if (mine !== seq) return;
                if (!rows.length) { list.innerHTML = '<li class="site-search-empty">Nothing turned up.</li>'; list.hidden = false; return; }
                list.innerHTML = rows.map(function (e) {
                    return '<li role="option"><a href="' + esc(e.href) + '">' +
                        (e.logo ? '<img src="' + esc(e.logo) + '" alt="" loading="lazy" onerror="this.style.visibility=\'hidden\'">' : '<span class="search-noimg"></span>') +
                        '<span class="search-main"><strong>' + esc(e.label) + '</strong><span>' + esc(e.kind) + ' · ' + esc(e.sub) + '</span></span></a></li>';
                }).join('') + '<li><a class="site-search-all" href="/search?q=' + encodeURIComponent(q) + '">All results for “' + esc(q) + '” →</a></li>';
                list.hidden = false; active = -1;
            }).catch(close);
        }, 160);
    });
    input.addEventListener('keydown', function (e) {
        var items = list.querySelectorAll('a');
        if (list.hidden || !items.length) return;
        if (e.key === 'ArrowDown') { e.preventDefault(); setActive(Math.min(active + 1, items.length - 1)); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(Math.max(active - 1, 0)); }
        else if (e.key === 'Enter' && active >= 0) { e.preventDefault(); window.location = items[active].href; }
        else if (e.key === 'Escape') { close(); }
    });
    document.addEventListener('click', function (e) { if (!e.target.closest('#siteSearch')) close(); });
})();
