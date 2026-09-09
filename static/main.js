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
if (e.key === 'Escape') { closeAddClosedModal(); closeCloseBetModal(); }
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
