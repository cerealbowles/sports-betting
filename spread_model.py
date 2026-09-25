"""
spread_model.py — Real margin/spread prediction, built from scratch per
sport, replacing spread_proxy.py's crude derivation (implied_margin was
just a linear fit of actual margin against logit(home_prob), NOT its own
model). This one is an independent Ridge regression of actual final-score
margin against each sport's own win-probability factor decomposition
(factors_json / game.model.factors — the same [(label, contribution), ...]
list the "Model Factors" UI already renders), fit and cross-validated by
spread_model_fit.py against instance/bets.db's resolved game_predictions
history. Re-run that script and paste its COEFFS output here to refit as
more graded games accumulate.

Honest result from the 2026-09-25 fit (chronological 70/30 split, Ridge
alpha picked per sport by 5-fold CV, graded against real ESPN-backfilled
market spread lines, break-even at standard -110 is 52.4%):

    MLB   58.6% ATS (n=694 test)   NHL   64.8% ATS (n=860)  — beat break-even
    NFL   46.2% ATS (n=106)   CFB   44.1% (n=118)
    NBA   46.5% ATS (n=740)   WNBA  51.9% (n=322)            — below break-even

For MLB and NHL specifically, this model's picks are IDENTICAL to
spread_proxy's on every graded held-out game — both are ultimately
monotonic in the same underlying win-probability signal, and regressing on
the decomposed factors instead of the single logit didn't change which
side either favors here, just the margin number attached to it. It's
shipped anyway because it's a genuinely independent fit (not a derivation)
that can keep improving as more graded games accumulate history the old
proxy had no mechanism to use, and because VALIDATED_SPORTS lands on the
same two sports spread_proxy already found — a real cross-check, not
coincidence dressed up as one. For the other four sports the picks do
differ from spread_proxy's, but neither approach clears break-even, so
mc-confidence-unproven styling applies same as before.

Label canonicalization: MLB's factor labels carry a leading team
abbreviation (e.g. "BOS SP Barrel%", "TB OPS" — mlb_model.py's `_add`
always labels with whichever team's stat drove the differential). This
module strips that prefix the same way spread_model_fit.py's training data
was canonicalized, so a live factors list from any team matches the shared
COEFFS keys below (see _canonicalize, kept in sync by hand with
spread_model_fit.py's identical logic — duplicated rather than imported
since the fit script imports app.py/Flask at module scope and shouldn't be
a runtime dependency of the live model).
"""
import math

COEFFS = {
    "MLB": {
        "intercept": 0.083,
        "weights": {
            "Bullpen Fatigue": 0.5847,
            "H2H Record": -1.9614,
            "Hist Adj": 4.7411,
            "Home Field": -0.0957,
            "OPS": 0.9092,
            "Off K%": 0.0711,
            "R/G": 4.0576,
            "SP BB%": 2.206,
            "SP Barrel%": 3.8835,
            "SP K%": 0.4954,
            "SP SIERA": -1.1818,
            "SP Whiff%": -1.6929,
            "SP+BP ERA": 0.3281,
            "SP+BP SIERA": 2.9747,
            "Trade Deadline": -0.0999,
        },
        "sigma": 4.525,
    },
    "NFL": {
        "intercept": 1.8336,
        "weights": {
            "Bye week (away)": 0.2393,
            "Bye week (home)": 0.1787,
            "Home field": -0.0,
            "Home/road record": 0.386,
            "Points allowed/G": 3.0921,
            "Points per game": 4.8061,
            "Recent form (L3)": 1.9019,
            "Short week (away)": -0.2205,
            "Short week (home)": 0.2373,
            "Win percentage": 3.5707,
        },
        "sigma": 13.364,
    },
    "CFB": {
        "intercept": 10.0752,
        "weights": {
            "Bye week (away)": -2.6606,
            "Bye week (home)": -6.905,
            "Home field": -0.0,
            "Home/road record": -8.426,
            "Points allowed/G": 8.6054,
            "Points per game": 6.4568,
            "Ranking": 11.9844,
            "Recent form (L3)": 4.248,
            "Short week (away)": -0.7914,
            "Short week (home)": -0.9731,
            "Win percentage": 3.9077,
        },
        "sigma": 19.409,
    },
    "NBA": {
        "intercept": 1.5276,
        "weights": {
            "Back-to-back": 10.1025,
            "Home court": -0.0,
            "Home/road record": 2.1329,
            "Points allowed/G": 21.7645,
            "Points per game": 20.5715,
            "Recent form (L5)": 8.4843,
            "Win percentage": -4.5821,
        },
        "sigma": 14.495,
    },
    "NHL": {
        "intercept": 0.1855,
        "weights": {
            "Back-to-back (away)": 1.5514,
            "Back-to-back (home)": 1.3381,
            "Goals against/G": 1.2354,
            "Goals for/G": 1.3459,
            "Home ice": -0.0,
            "Home/road record": 1.7769,
            "Points percentage": -1.492,
            "Recent form (L5)": 2.2797,
        },
        "sigma": 2.578,
    },
    "WNBA": {
        "intercept": 1.5518,
        "weights": {
            "Back-to-back": 3.7089,
            "Home court": 0.0,
            "Home/road record": 2.8838,
            "Points allowed/G": 7.502,
            "Points per game": 7.2717,
            "Recent form (L5)": 0.605,
            "Win percentage": 4.7956,
        },
        "sigma": 12.955,
    },
}

# Out-of-sample ATS accuracy per sport, from the fit run documented above.
# Same standard as spread_proxy.VALIDATED_SPORTS — only sports that actually
# beat break-even (52.4% at -110) out-of-sample count as validated; being
# fit-and-graded on the same data a sport's coefficients came from doesn't.
VALIDATED_SPORTS = {'MLB', 'NHL'}


# mlb_model.py falls back to these when a team dict has no 'abbr' key
# (ha = home.get('abbr', 'Hm'), aa = away.get('abbr', 'Aw')) — not
# themselves valid team codes but mean exactly the same thing a real team
# abbreviation would in this position.
_FALLBACK_ABBR_TOKENS = {'Hm', 'Aw'}


def _looks_like_team_code(token):
    """See spread_model_fit.py's _looks_like_team_code — kept in sync by
    hand, duplicated rather than imported (that script pulls in Flask/app.py
    at module scope, which shouldn't be a runtime dependency of live
    request handling)."""
    if token in _FALLBACK_ABBR_TOKENS:
        return True
    return token.isalpha() and token.isupper() and 2 <= len(token) <= 4


# mlb_model.py's _era_estimator falls back through SIERA -> xFIP -> xERA ->
# ERA depending on data availability, labeling the SP+BP blend factor with
# whichever metric it used (era_label = f'{ha} SP+BP {h_era_lbl}'), and
# groups xFIP with SIERA and xERA with ERA for its own coefficient (era_coeff
# = 0.37 for SIERA/xFIP, 0.63 for xERA/ERA — mlb_model.py's _era_estimator
# call site). COEFFS only has fitted weights for "SP+BP SIERA" and "SP+BP
# ERA" (spread_model_fit.py's training data only ever saw those two labels),
# so this alias maps a live xFIP/xERA game onto the weight for whichever of
# the two it's grouped with above, instead of silently dropping the factor.
#
# Same reasoning applies to team offense: mlb_model.py's section 7 prefers
# handedness-split xwOBA, falling back to overall xwOBA, falling back to
# OPS, labeling whichever tier it used ("xwOBA Split" / "xwOBA" / "OPS") —
# but spread_model_fit.py's training data only ever saw "OPS" (the other
# two tiers are newer/rarer), so COEFFS has no fitted weight for them. All
# three tiers measure the same offense-differential concept at comparable
# contribution scale (mlb_model.py's per-tier coefficients — 2.5 for
# xwOBA[-split], 0.22 for OPS — are chosen so the logit contribution is
# similar-sized regardless of which data tier was available), so aliasing
# onto OPS's weight is a reasonable stopgap rather than dropping the whole
# offense factor whenever the higher-quality xwOBA data happens to be there.
_LABEL_ALIASES = {
    'SP+BP xFIP': 'SP+BP SIERA',
    'SP+BP xERA': 'SP+BP ERA',
    'xwOBA Split': 'OPS',
    'xwOBA': 'OPS',
}


def _canonicalize(factors):
    """Strip team-abbreviation prefixes from a live factors list so its
    labels match COEFFS' sport-level (not team-level) keys."""
    canon = []
    for label, contrib in factors:
        if ' ' in label:
            first, suffix = label.split(' ', 1)
            if _looks_like_team_code(first):
                label = suffix
        canon.append((_LABEL_ALIASES.get(label, label), contrib))
    return canon


def weighted_factors(sport, factors):
    """Returns {'margin': predicted home margin, 'rows': [(label,
    weighted_contribution), ...] sorted by |weighted_contribution| desc} or
    None if this sport has no fitted model. `factors` is game.model.factors
    (raw win-prob logit contributions) — this reweights them with spread_
    model's own fitted coefficients, it does not reuse the win-prob values
    directly. Powers both predict_margin() below and the "Spread Model"
    factors panel (_spread_factors.html)."""
    coeffs = COEFFS.get(sport)
    if not coeffs:
        return None
    weights = coeffs['weights']
    margin = coeffs['intercept']
    rows = []
    for label, contrib in _canonicalize(factors):
        w = weights.get(label)
        if w is None:
            continue
        weighted = contrib * w
        margin += weighted
        if abs(weighted) > 1e-6:
            rows.append((label, round(weighted, 3) + 0.0))  # +0.0 folds -0.0 to 0.0
    rows.sort(key=lambda r: -abs(r[1]))
    return {'margin': round(margin, 2), 'rows': rows}


def predict_margin(sport, factors):
    """Model-implied home-team margin (positive = home wins by that much),
    or None if this sport has no fitted model or factors is empty."""
    if not factors:
        return None
    wf = weighted_factors(sport, factors)
    return wf['margin'] if wf else None


def _phi(x):
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cover_prob(sport, factors, home_spread_line):
    """P(home team covers home_spread_line), or None if unavailable.
    home_spread_line follows market convention (negative = home favored) —
    home covers when actual_margin > -home_spread_line."""
    coeffs = COEFFS.get(sport)
    margin = predict_margin(sport, factors)
    if margin is None or home_spread_line is None or not coeffs:
        return None
    sigma = coeffs['sigma']
    if sigma <= 0:
        return None
    return _phi((margin - (-home_spread_line)) / sigma)


def is_validated(sport):
    """Whether this sport's spread model has actually beaten break-even
    out-of-sample, vs. just being fit-and-graded on the same historical
    data — see spread_proxy.is_validated, same reasoning applies here."""
    return sport in VALIDATED_SPORTS
