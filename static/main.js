/* Utility: convert American -> decimal */
function americanToDecimal(a) {
var n = Number(a);
if (!isFinite(n)) return null;
if (n > 0) return +(n/100 + 1).toFixed(4);
return +(100/Math.abs(n) + 1).toFixed(4);
}

/* compute recommended stake (client mirror of server logic) */
function computeKelly(bankroll, percentCap, odds, prob) {
if (!odds || !prob || odds <= 0 || prob <= 0 || prob >= 1) return 0.0;
var b = odds - 1.0;
if (b <= 0) return 0.0;
var f = (b * prob - (1 - prob)) / b;
f = Math.max(0.0, f);
var rawStake = f * bankroll;
var cap = percentCap * bankroll;
var recommended = Math.min(rawStake, cap);
recommended = Math.max(recommended, 0.10);
return Math.round(recommended * 100) / 100;
}

/* Empirical info fetch (small summary next to probability) */
function updateEmpiricalInfo() {
var sport = (document.getElementById('sport')||{}).value || '';
var betType = (document.getElementById('bet_type')||{}).value || '';
var prob = (document.getElementById('prob')||{}).value || '0.5';
fetch('/api/empirical_info', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ sport: sport, bet_type: betType, prob: prob })
}).then(r=>r.json()).then(function(data){
    var el = document.getElementById('empirical_info');
    if(!el) return;
    if(data.empirical === null || data.matching_count === 0) {
    el.textContent = 'Empirical: —   Adjusted: ' + (data.adjusted!==undefined?Number(data.adjusted).toFixed(4):'—');
    } else {
    el.textContent = 'Empirical: ' + Number(data.empirical).toFixed(4) +
        '  Adjusted: ' + Number(data.adjusted).toFixed(4) +
        '  (α=' + Number(data.alpha).toFixed(2) + ', n=' + (data.matching_count||0) + ')';
    }
}).catch(function(){});
}

/* Wire up inputs */
document.addEventListener('DOMContentLoaded', function(){
var aInput = document.getElementById('american_odds');
var probInput = document.getElementById('prob');
var outSpan = document.getElementById('converted_decimal');
var recommendedSpan = document.getElementById('recommended_stake');
var bankroll = parseFloat(document.getElementById('bankroll_value').textContent || '0');
var percentCap = parseFloat(document.getElementById('percent_cap_value').textContent || '0.02');

function updateRecommended() {
    var dec = americanToDecimal(aInput.value);
    var p = parseFloat(probInput.value);
    if (dec === null || !isFinite(p)) {
    recommendedSpan.textContent = '—';
    document.getElementById('odds_hidden').value = '';
    return;
    }
    outSpan.textContent = dec;
    var rec = computeKelly(bankroll, percentCap, dec, p);
    recommendedSpan.textContent = '$' + rec.toFixed(2);
    document.getElementById('odds_hidden').value = dec;
}

if (aInput) aInput.addEventListener('input', updateRecommended);
if (probInput) probInput.addEventListener('input', updateRecommended);

/* empirical info updates */
var sportEl = document.getElementById('sport');
var typeEl = document.getElementById('bet_type');
if (sportEl) sportEl.addEventListener('input', updateEmpiricalInfo);
if (typeEl) typeEl.addEventListener('change', updateEmpiricalInfo);
if (probInput) probInput.addEventListener('input', updateEmpiricalInfo);
updateEmpiricalInfo();

/* bet form submission fills hidden fields */
document.getElementById('betForm').addEventListener('submit', function(e){
    var name = document.getElementById('bet_name').value;
    var odds = document.getElementById('odds_hidden').value;
    var prob = document.getElementById('prob').value;
    var stake = document.getElementById('actual_stake').value;
    var sport = document.getElementById('sport').value || '';
    var type = document.getElementById('bet_type').value || 'Moneyline';
    if (!name || !odds || !prob || !stake) {
    alert('Please fill in all fields.');
    e.preventDefault();
    return;
    }
    document.getElementById('form_name').value = name;
    document.getElementById('form_odds').value = odds;
    document.getElementById('form_prob').value = prob;
    document.getElementById('form_stake').value = stake;
    document.getElementById('form_sport').value = sport;
    document.getElementById('form_type').value = type;
});
});