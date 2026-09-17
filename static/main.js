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

/* ── Modal: Add Past Bet ─────────────────────────────── */
function openAddClosedModal() {
document.getElementById('addClosedModal').classList.add('open');
document.body.style.overflow = 'hidden';
}
function closeAddClosedModal() {
document.getElementById('addClosedModal').classList.remove('open');
document.body.style.overflow = '';
}
function closeModalOutside(e) {
if (e.target === e.currentTarget) {
    closeAddClosedModal();
    closeCloseBetModal();
}
}
document.addEventListener('keydown', function(e) {
if (e.key === 'Escape') { closeAddClosedModal(); closeCloseBetModal(); closeBetSheet(); closeGameDetails(); }
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

document.addEventListener('click', function (e) {
var summary = e.target.closest('.game-card-summary');
if (!summary) return;
var card = summary.closest('.game-card');
if (!card || !card.id) return;
openGameDetails(card.id.replace(/^gc-/, ''));
});

document.addEventListener('keydown', function (e) {
if (e.key !== 'Enter' && e.key !== ' ') return;
if (!e.target.classList || !e.target.classList.contains('game-card-summary')) return;
e.preventDefault();
e.target.click();
});

/* ── Bet bottom sheet ─────────────────────────────────
   Tapping any "bet" link (.btn-bet-team, present on every sport's game
   card) opens the existing /new-bet form in a sheet that slides up over
   the current page instead of navigating away. The sheet just hosts an
   iframe pointed at /new-bet?...&embed=1 — the form's own pre-fill/Kelly
   JS (below) runs inside that iframe unmodified. */
var betSheetCloseTimer = null;

function openBetSheet(url) {
var overlay = document.getElementById('betSheetOverlay');
var frame   = document.getElementById('betSheetFrame');
if (!overlay || !frame) { window.location.href = url; return; }
clearTimeout(betSheetCloseTimer);
var sep = url.indexOf('?') === -1 ? '?' : '&';
frame.src = url + sep + 'embed=1&next=' + encodeURIComponent(window.location.pathname + window.location.search);
overlay.style.display = 'flex';
overlay.offsetHeight; // force reflow so the slide-up transition runs
overlay.classList.add('open');
document.body.style.overflow = 'hidden';
}

function closeBetSheet() {
var overlay = document.getElementById('betSheetOverlay');
if (!overlay || !overlay.classList.contains('open')) return;
overlay.classList.remove('open');
document.body.style.overflow = '';
clearTimeout(betSheetCloseTimer);
betSheetCloseTimer = setTimeout(function () {
    overlay.style.display = 'none';
    var frame = document.getElementById('betSheetFrame');
    if (frame) frame.src = 'about:blank';
}, 300);
}

function closeBetSheetOutside(e) {
if (e.target === e.currentTarget) closeBetSheet();
}

document.addEventListener('click', function (e) {
var link = e.target.closest('.btn-bet-team');
// Buttons (not <a> links) are the new inline Moneyline chips on CFB/NFL
// cards — they open the in-sheet bet slip themselves (see below), not the
// iframe sheet, so leave them alone here.
if (!link || link.tagName !== 'A') return;
e.preventDefault();
closeGameDetails();
openBetSheet(link.href);
});

/* Successful submission redirects the iframe away from /new-bet (add_open's
   `next` param, pre-filled above with the current page's URL) — detect that
   and refresh the underlying page so open-bet indicators pick up the new
   position, since the iframe itself is about to be torn down. */
document.addEventListener('DOMContentLoaded', function () {
var frame = document.getElementById('betSheetFrame');
if (!frame) return;
frame.addEventListener('load', function () {
    try {
    var loc = frame.contentWindow.location;
    if (loc.href === 'about:blank') return;
    if (loc.pathname !== '/new-bet') {
        closeBetSheet();
        setTimeout(function () { window.location.reload(); }, 280);
    }
    } catch (err) { /* cross-origin — ignore */ }
});
});

/* ── Modal: Close Bet ────────────────────────────────── */
function openCloseBetModal(betId, betName) {
var modal = document.getElementById('closeBetModal');
if (!modal) return;
var nameEl = document.getElementById('close_bet_name');
if (nameEl) nameEl.textContent = betName;
var form = modal.querySelector('form');
if (form) form.action = '/close_open/' + betId;
modal.classList.add('open');
document.body.style.overflow = 'hidden';
}
function closeCloseBetModal() {
var modal = document.getElementById('closeBetModal');
if (modal) modal.classList.remove('open');
document.body.style.overflow = '';
}

/* ── Utility: American → decimal ────────────────────── */
function americanToDecimal(a) {
var n = Number(a);
if (!isFinite(n)) return null;
if (n > 0) return +(n/100 + 1).toFixed(4);
return +(100/Math.abs(n) + 1).toFixed(4);
}

/* ── Kelly calc ──────────────────────────────────────── */
var KELLY_MIN_P = 0.5;
var KELLY_MAX_P = 0.95;

function blendProb(userProb) {
return Math.max(KELLY_MIN_P, Math.min(KELLY_MAX_P, userProb));
}

function computeKelly(bankroll, percentCap, odds, prob) {
if (!odds || !prob || odds <= 0 || prob <= 0 || prob >= 1) return {amount: 0.0, negativeEV: true};
var b = odds - 1.0;
if (b <= 0) return {amount: 0.0, negativeEV: true};
var f = (b * prob - (1 - prob)) / b;
var negativeEV = f <= 0;
f = Math.max(0.0, f);
var recommended = Math.min(f * bankroll, percentCap * bankroll);
recommended = Math.max(recommended, 0.10);
return {amount: Math.round(recommended * 100) / 100, negativeEV: negativeEV};
}

/* ── Empirical win rate ──────────────────────────────── */
var currentEmpirical = null;
var probSetFromHistory = false;

function clearProbHistoryIndicator() {
var probEl = document.getElementById('prob');
var sourceEl = document.getElementById('prob_source');
if (probEl) probEl.classList.remove('from-history');
if (sourceEl) sourceEl.textContent = '';
probSetFromHistory = false;
}

function updateEmpiricalInfo(autoFill) {
var sport   = (document.getElementById('sport')    || {}).value || '';
var betType = (document.getElementById('bet_type') || {}).value || '';
var prob    = parseFloat((document.getElementById('prob') || {}).value) || 0.5;
var unfRank = window.currentUnfRank || null;
fetch('/api/empirical_info', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sport: sport, bet_type: betType, prob: prob, unf_rank: unfRank})
}).then(function(r) { return r.json(); }).then(function(data) {
    var el       = document.getElementById('empirical_info');
    var sourceEl = document.getElementById('prob_source');
    var probEl   = document.getElementById('prob');
    if (!el) return;
    if (!data.empirical || data.matching_count === 0) {
    currentEmpirical = null;
    el.textContent = sport || betType ? 'No history yet for this sport/type' : 'Calc: —';
    if (sourceEl) sourceEl.textContent = '';
    if (probSetFromHistory && probEl) { probEl.classList.remove('from-history'); probSetFromHistory = false; }
    } else {
    currentEmpirical = data.empirical;
    if (data.source === 'unified_rank') {
        el.textContent =
            'Unified Score Rank ' + data.bucket_label + ' win rate: ' + (Number(data.empirical) * 100).toFixed(1) + '%'
            + '  (n=' + data.matching_count + ' ' + (sport || 'bets') + ')';
    } else {
        el.textContent =
            'Historical win rate: ' + (Number(data.empirical) * 100).toFixed(1) + '%'
            + '  (n=' + data.matching_count + ' ' + (sport || 'bets') + ')';
    }

    // Blend into the Win Probability field — only on explicit autofill triggers
    // (initial load, sport/type change). Never on the prob-input listener's own
    // recalc (autoFill=false there), so it doesn't fight manual edits.
    if (probEl && data.adjusted != null && autoFill) {
        probEl.value = Number(data.adjusted).toFixed(2);
        probEl.classList.add('from-history');
        probSetFromHistory = true;
        if (sourceEl) {
            sourceEl.textContent = data.source === 'unified_rank'
                ? 'Blended ' + Math.round(data.alpha * 100) + '% model / ' + Math.round((1 - data.alpha) * 100) + '% rank ' + data.bucket_label + ' history'
                : 'Blended ' + Math.round(data.alpha * 100) + '% model / ' + Math.round((1 - data.alpha) * 100) + '% your history';
        }
    }

    // Always re-run Kelly after empirical loads
    if (typeof updateRecommended === 'function') updateRecommended();
    }
}).catch(function() {});
}

/* ── Bet form wiring — only active on new_bet page ──── */
document.addEventListener('DOMContentLoaded', function() {
var aInput          = document.getElementById('american_odds');
if (!aInput) return;
var probInput       = document.getElementById('prob');
var outSpan         = document.getElementById('converted_decimal');
var edgeEl          = document.getElementById('edge_display');
var recommendedSpan = document.getElementById('recommended_stake');
var bankroll   = parseFloat((document.getElementById('bankroll_value')    || {}).textContent || '0');
var percentCap = parseFloat((document.getElementById('percent_cap_value') || {}).textContent || '0.02');

// Vig-free implied from URL param (passed from schedule page)
var _urlImplied = parseFloat(new URLSearchParams(window.location.search).get('implied') || '');
var vigFreeImplied = isFinite(_urlImplied) ? _urlImplied * 100 : null;

// Hide the raw-implied edge line when context banner already shows vig-free edge
var contextBannerVisible = !isNaN(_urlImplied) && isFinite(_urlImplied);

function updateRecommended() {
    var dec = americanToDecimal(aInput.value);
    var p   = parseFloat(probInput.value);

    if (dec === null || !isFinite(p)) {
    recommendedSpan.textContent = '—';
    if (outSpan) outSpan.textContent = '—';
    if (edgeEl)  edgeEl.textContent  = '';
    document.getElementById('odds_hidden').value = '';
    return;
    }

    if (outSpan) outSpan.textContent = dec;
    document.getElementById('odds_hidden').value = dec;

    // Edge display: use vig-free implied when available; hide when banner shows it
    if (edgeEl) {
    if (contextBannerVisible) {
        edgeEl.textContent = '';
        edgeEl.className   = 'small';
    } else {
        var rawImplied = (1 / dec) * 100;
        var edge       = (p * 100) - rawImplied;
        edgeEl.textContent = 'Implied: ' + rawImplied.toFixed(1) + '% (raw) · Edge: ' + (edge >= 0 ? '+' : '') + edge.toFixed(1) + '%';
        edgeEl.className   = 'small ' + (edge >= 0 ? 'status-win' : 'status-loss');
    }
    }

    // Kelly with blended prob
    var adjP = blendProb(p);
    var result = computeKelly(bankroll, percentCap, dec, adjP);
    if (result.negativeEV) {
    recommendedSpan.innerHTML = '$' + result.amount.toFixed(2) +
        ' <span class="small status-loss">(negative EV — Kelly says skip)</span>';
    } else {
    recommendedSpan.textContent = '$' + result.amount.toFixed(2);
    }
}

window.updateRecommended = updateRecommended;

aInput.addEventListener('input', updateRecommended);
probInput.addEventListener('input', function() {
    if (probSetFromHistory) clearProbHistoryIndicator();
    updateRecommended();
    updateEmpiricalInfo(false);
});

var sportEl = document.getElementById('sport');
var typeEl  = document.getElementById('bet_type');
if (sportEl) sportEl.addEventListener('input',  function() { updateEmpiricalInfo(true); });
if (typeEl)  typeEl.addEventListener('change',  function() { updateEmpiricalInfo(true); });

updateEmpiricalInfo(false);

/* Bet form submission — collect all visible fields into hidden inputs */
document.getElementById('betForm').addEventListener('submit', function(e) {
    var name       = (document.getElementById('bet_name')    || {}).value;
    var odds       = (document.getElementById('odds_hidden') || {}).value;
    var prob       = (document.getElementById('prob')        || {}).value;
    var stake      = (document.getElementById('actual_stake')|| {}).value;
    var sport      = (document.getElementById('sport')       || {}).value || '';
    var type       = (document.getElementById('bet_type')    || {}).value || 'Moneyline';
    var eventstart = (document.getElementById('eventstart')  || {}).value || '';
    var notes      = (document.getElementById('notes')       || {}).value || '';
    if (!name || !odds || !prob || !stake) {
    alert('Please fill in all fields.');
    e.preventDefault();
    return;
    }
    document.getElementById('form_name').value       = name;
    document.getElementById('form_eventstart').value = eventstart;
    document.getElementById('form_odds').value       = odds;
    document.getElementById('form_prob').value       = prob;
    document.getElementById('form_stake').value      = stake;
    document.getElementById('form_sport').value      = sport;
    document.getElementById('form_type').value       = type;
    document.getElementById('form_notes').value      = notes;
});
});

/* ── Inline Moneyline bet slip (CFB/NFL game cards) ───────────────────
   Tapping a .btn-bet-team *button* (as opposed to the <a> links other
   sports still use) opens the .bet-slip already sitting in that game's
   .game-card-details, pre-filled with the recommended Kelly stake, and
   submits straight to /add_open via fetch — no navigation, no iframe.
   Elements are found via closest()/querySelector from the clicked chip's
   own .game-card-details ancestor rather than global IDs, since every
   game's (hidden) details node carries an identically-classed bet-slip. */
function openBetSlip(btnEl) {
var details = btnEl.closest('.game-card-details');
var slip    = details && details.querySelector('.bet-slip');
if (!slip) return;

details.querySelectorAll('.btn-bet-team.selected').forEach(function (el) { el.classList.remove('selected'); });
btnEl.classList.add('selected');

var d = btnEl.dataset;
slip.dataset.team          = d.team;
slip.dataset.odds          = d.odds;
slip.dataset.modelProb     = d.modelProb;
slip.dataset.impliedProb   = d.impliedProb;
slip.dataset.sport         = d.sport;
slip.dataset.betType       = d.betType || 'Moneyline';
slip.dataset.eventstartutc = d.eventstartutc;
slip.dataset.homeName      = d.homeName;
slip.dataset.awayName      = d.awayName;
slip.dataset.betSide       = d.betSide;
slip.dataset.gameKey       = d.gameKey;

var logo = slip.querySelector('.bet-slip-logo');
if (logo) { logo.src = d.logo; logo.alt = d.team; }
var nameEl = slip.querySelector('.bet-slip-pick-name');
if (nameEl) nameEl.textContent = d.team + ' ' + (d.betType || 'Moneyline');
var subEl = slip.querySelector('.bet-slip-pick-sub');
if (subEl) subEl.textContent = d.oddsDisplay + ' vs ' + d.opp;

var modelProb   = parseFloat(d.modelProb);
var impliedProb = parseFloat(d.impliedProb);
setText(slip, '.bss-model',   isFinite(modelProb)   ? (modelProb * 100).toFixed(1) + '%' : '—');
setText(slip, '.bss-implied', isFinite(impliedProb) ? (impliedProb * 100).toFixed(1) + '%' : '—');
var edgeEl = slip.querySelector('.bss-edge');
if (edgeEl) {
    if (isFinite(modelProb) && isFinite(impliedProb)) {
    var edge = (modelProb - impliedProb) * 100;
    edgeEl.textContent = (edge > 0 ? '+' : '') + edge.toFixed(1) + '%';
    edgeEl.className   = 'bss-val ' + (edge > 0 ? 'edge-pos' : 'edge-neg');
    } else {
    edgeEl.textContent = '—';
    edgeEl.className   = 'bss-val';
    }
}
setText(slip, '.bss-odds', d.oddsDisplay);

var recommended = recommendedStakeFor(d.odds, d.modelProb);
slip.dataset.recommended = recommended.toFixed(2);
setText(slip, '.bet-slip-recommended', '$' + recommended.toFixed(2));

var input = slip.querySelector('.stake-input-wrap input');
if (input) input.value = recommended.toFixed(2);

var err = slip.querySelector('.bet-slip-error');
if (err) err.classList.remove('show');
var confirm = slip.querySelector('.bet-slip-confirm');
if (confirm) confirm.classList.remove('show');

slip.classList.add('open');
updateBetSlipReturn(slip);
slip.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeBetSlip(btnEl) {
var slip = btnEl.closest('.bet-slip');
if (!slip) return;
slip.classList.remove('open');
var details = slip.closest('.game-card-details');
if (details) details.querySelectorAll('.btn-bet-team.selected').forEach(function (el) { el.classList.remove('selected'); });
}

function setStakeQuick(btnEl, mode) {
var slip  = btnEl.closest('.bet-slip');
var input = slip && slip.querySelector('.stake-input-wrap input');
if (!input) return;
input.value = (mode === 'rec' ? parseFloat(slip.dataset.recommended) || 0 : Number(mode)).toFixed(2);
updateBetSlipReturn(slip);
}

function setText(root, selector, text) {
var el = root.querySelector(selector);
if (el) el.textContent = text;
}

function recommendedStakeFor(americanOdds, modelProb) {
var bankroll   = parseFloat((document.getElementById('global_bankroll_value')    || {}).textContent || '0');
var percentCap = parseFloat((document.getElementById('global_percent_cap_value') || {}).textContent || '0.02');
var dec = americanToDecimal(americanOdds);
var p   = parseFloat(modelProb);
if (dec === null || !isFinite(p)) return 0;
var result = computeKelly(bankroll, percentCap, dec, blendProb(p));
return result.negativeEV ? 0 : result.amount;
}

function updateBetSlipReturn(slip) {
if (!slip) return;
var input = slip.querySelector('.stake-input-wrap input');
var stake = parseFloat(input && input.value) || 0;
var dec   = americanToDecimal(slip.dataset.odds);
var toWin = dec ? stake * (dec - 1) : 0;
setText(slip, '.bet-slip-to-win', '$' + toWin.toFixed(2));
var placeBtn = slip.querySelector('.btn-place-bet');
if (placeBtn) placeBtn.textContent = 'Bet $' + stake.toFixed(2) + ' on ' + slip.dataset.team;
}

document.addEventListener('input', function (e) {
if (!e.target.matches('.stake-input-wrap input')) return;
updateBetSlipReturn(e.target.closest('.bet-slip'));
});

function placeBet(btnEl) {
var slip = btnEl.closest('.bet-slip');
if (!slip) return;
var input = slip.querySelector('.stake-input-wrap input');
var stake = parseFloat(input && input.value) || 0;
var err   = slip.querySelector('.bet-slip-error');
if (err) err.classList.remove('show');
if (!stake || stake <= 0) {
    if (err) { err.textContent = 'Enter a stake amount.'; err.classList.add('show'); }
    return;
}

btnEl.disabled = true;
var d = slip.dataset;
var fd = new FormData();
fd.append('name',          d.team + ' ' + (d.betType || 'Moneyline'));
fd.append('odds',          d.odds);
fd.append('prob',          d.modelProb);
fd.append('stake',         stake.toFixed(2));
fd.append('sport',         d.sport || '');
fd.append('bet_type',      d.betType || 'Moneyline');
fd.append('home_name',     d.homeName || '');
fd.append('away_name',     d.awayName || '');
fd.append('bet_side',      d.betSide || '');
fd.append('eventstartutc', d.eventstartutc || '');
fd.append('is_paper',      '0');

fetch('/add_open', { method: 'POST', body: fd })
    .then(function (r) {
    if (r.status === 409) return r.text().then(function () { throw new Error('unsettled'); });
    var confirm = slip.querySelector('.bet-slip-confirm');
    if (confirm) {
        confirm.textContent = '✓ Bet placed — $' + stake.toFixed(2) + ' on ' + d.team + ' logged to Open Bets';
        confirm.classList.add('show');
    }
    setTimeout(function () {
        closeGameDetails();
        window.location.reload();
    }, 900);
    })
    .catch(function () {
    btnEl.disabled = false;
    if (err) {
        err.textContent = 'Close out your finished bet(s) before placing a new one.';
        err.classList.add('show');
    }
    });
}
