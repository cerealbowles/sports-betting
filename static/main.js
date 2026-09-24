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
}
}
document.addEventListener('keydown', function(e) {
if (e.key === 'Escape') { closeAddClosedModal(); closeBetSheet(); closeGameDetails(); }
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

/* "Bet" links on the recommendations table and the market-chips row (ML/
   Spread/Total, _market_chips.html) used to jump straight to /new-bet —
   now they all open the same game-card details sheet a tap on the card
   would, pre-selecting the matching chip so the inline bet slip pops open
   exactly as if the user had picked it there. betType defaults to
   'Moneyline' so the original 2-arg call sites (recommendations table)
   keep working unchanged; Spread/Total callers pass it explicitly since
   a game can have an ML button and a Spread button that share the same
   data-bet-side ('home'/'away') — betType disambiguates between them.
   Total buttons don't have a home/away side, so their data-bet-side is
   'over'/'under' instead. */
function openRecommendedBet(gameKey, betSide, betType) {
betType = betType || 'Moneyline';
openGameDetails(gameKey);
var details = document.getElementById('gd-' + gameKey);
if (!details) return;
var btn = details.querySelector('.btn-bet-team[data-bet-side="' + betSide + '"][data-bet-type="' + betType + '"]');
if (btn && btn.tagName === 'BUTTON') btn.click();
}

/* Market-chips row (ML/Spread/Total, _market_chips.html) — tapping a chip
   used to jump straight into placing that bet (openRecommendedBet above).
   Now it opens the details sheet and surfaces the model reasoning behind
   that chip's number instead: ML and Spread both point at the same
   "Model Factors" panel (Spread's proxy is derived straight from that
   win-probability model, so it's the same factors), Total points at its
   own "Total Model" panel (_total_factors.html) since totals don't share
   the win-prob model's factor list. Placing the bet is still one tap away
   from there — this just stops assuming that's what a chip tap means. */
function openModelFactors(gameKey, kind) {
openGameDetails(gameKey);
var details = document.getElementById('gd-' + gameKey);
if (!details) return;
setTimeout(function () {
    // getElementById, not querySelector('#...') — game_key routinely
    // contains spaces/dots (e.g. "st. louis cardinals_..."), which are
    // valid in an id attribute but break an unescaped CSS #id selector.
    var target = document.getElementById((kind === 'total' ? 'total-factors-' : 'factors-') + gameKey);
    if (!target || !details.contains(target)) return;
    target.open = true;
    target.scrollIntoView({behavior: 'smooth', block: 'start'});
}, 350);
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
// Team-side bets (Moneyline/Spread) carry data-team + data-bet-type and the
// full pick name is built as "<team> <betType>" (e.g. "NYY Moneyline").
// Total bets (Over/Under) have no team, so the button supplies the full
// name directly via data-name (e.g. "Over 8.5") instead.
var pickName = d.name || (d.team + ' ' + (d.betType || 'Moneyline'));
slip.dataset.team          = d.team || '';
slip.dataset.name          = pickName;
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
if (logo) { logo.src = d.logo || ''; logo.alt = d.team || ''; logo.style.display = d.logo ? '' : 'none'; }
var nameEl = slip.querySelector('.bet-slip-pick-name');
if (nameEl) nameEl.textContent = pickName;
var subEl = slip.querySelector('.bet-slip-pick-sub');
if (subEl) subEl.textContent = d.sub || (d.oddsDisplay + ' vs ' + d.opp);

var modelProb   = parseFloat(d.modelProb);
// Break-even for the price actually offered (vig included), so the edge shown
// here agrees with the Kelly stake — d.impliedProb is the vig-free market prob.
var slipDec     = americanToDecimal(d.odds);
var impliedProb = slipDec ? 1 / slipDec : parseFloat(d.impliedProb);
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
fd.append('name',          d.name || (d.team + ' ' + (d.betType || 'Moneyline')));
fd.append('odds',          d.odds);
fd.append('prob',          d.modelProb);
fd.append('stake',         stake.toFixed(2));
fd.append('sport',         d.sport || '');
fd.append('bet_type',      d.betType || 'Moneyline');
fd.append('home_name',     d.homeName || '');
fd.append('away_name',     d.awayName || '');
fd.append('bet_side',      d.betSide || '');
fd.append('eventstartutc', d.eventstartutc || '');

fetch('/add_open', { method: 'POST', body: fd })
    .then(function (r) {
    if (r.status === 409) return r.text().then(function () { throw new Error('unsettled'); });
    var confirm = slip.querySelector('.bet-slip-confirm');
    if (confirm) {
        confirm.textContent = '✓ Bet placed — $' + stake.toFixed(2) + ' on ' + d.name + ' logged to Open Bets';
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
