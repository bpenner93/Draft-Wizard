"""
bestball.py  --  the best-ball value model (pure math, no I/O, no UI)
=====================================================================
Redraft and best ball are not the same game, and the difference is not the
roster template. In redraft you start a lineup you CHOOSE, so a player you can
never start is worth nothing and the engine is right to price him at
BLOCKED_VALUE. In best ball the lineup is chosen for you, every week, from
whoever scored most -- so the 12 bench spots on a 20-man DraftKings roster are
not insurance, they are 12 more tickets that get played automatically in any
week they hit.

That single fact inverts the redraft roster logic. `roster_factor` x
`effective_vor` prices a 7th WR at roughly -32 (rf 0.2 interpolated toward
BLOCKED_VALUE -40); DK best ball wants 6-8 of them. Left alone, the redraft
engine builds a redraft roster in a format that punishes it.

────────────────────────────────────────────────────────────────────────────
THE MODEL: MARGINAL OPTION VALUE
────────────────────────────────────────────────────────────────────────────
A player is worth what he ADDS to your optimal weekly lineup, which is not his
mean. Adding a player at position p who scores `x` in some week changes that
week's optimal total by exactly

        gain = max(0, x - B_p)

where B_p is the bar a new p-player must beat to displace someone. This is
exact, not an approximation, for the DK slot structure. Derivation, with A =
your scores at p sorted descending, n_p = dedicated slots at p, F = the score
of your current FLEX occupant:

  * if x makes the dedicated cut (x >= A[n_p-1]) it takes that slot and the
    displaced A[n_p-1] cascades down to compete for FLEX, so the week gains
    (x - A[n_p-1]) + max(0, A[n_p-1] - F)  =  x - min(A[n_p-1], F)
  * if it misses the cut it can still take FLEX, gaining max(0, x - F)

Both collapse to max(0, x - B_p) with **B_p = min(A[n_p-1], F)** (and B_QB =
A[n_QB-1], since QB is not flex-eligible here). So the bar depends only on the
roster, never on the candidate -- which means it can be sampled ONCE per pick
and reused for every candidate. That is what makes this fast enough to run live.

Everything the format rewards falls out of that one equation rather than being
bolted on as separate heuristics:

  * DEPTH decays but never goes negative. Each player you add raises B_p
    stochastically, so the next one is worth less -- asymptotically zero, from
    ABOVE. It can never invert into a penalty, which is the redraft bug this
    replaces.
  * CEILING is rewarded automatically. max(0, x - B) is convex in x, so for two
    players with the same mean the higher-variance one is worth strictly more.
    No explicit ceiling term is needed.
  * BYE COVERAGE falls out of week-aligned sampling. Stack three WRs on the
    same bye and B_WR collapses that week, so the fourth WR on a different bye
    picks up real value exactly when you are thin.
  * AVAILABILITY is priced harder than in redraft, because there is no waiver
    wire: a missed week is a hole nothing can fill.

⚠ WHAT IS NOT PLAYER-SPECIFIC YET (read before trusting the ceiling story).
The weekly spread uses a POSITION-CONSTANT coefficient of variation, because
that is all the board actually carries. `weekly_floor`/`weekly_ceiling` in
season_rankings.py are not fitted -- they are proj_pg * (1 -/+ cv) with
cv fixed per position (QB .30 / RB .55 / WR .62 / TE .66), so every WR's
"ceiling" is his mean x 1.775. Weighting by that column would be arithmetically
identical to weighting the mean: a no-op wearing a ceiling costume.
So the convexity above is real and correctly ORDERED for players of different
means, but it does NOT yet distinguish a boom/bust deep threat from a
PPR-steady slot receiver at the same projection. Fixing that needs a real
per-player weekly variance fitted on history -- see WEEKLY_CV below, which is
the single place to swap it in.
"""
from __future__ import annotations

import numpy as np

# Weekly coefficient of variation of DK points -- the biggest lever in this
# model, because E[max(0, x - B)] is convex in x (a ~2x swing between cv .35 and
# cv .95 at a fixed mean).
#
# ⭐ FITTED, not assumed: `fit_weekly_cv.py` over 2015-2025 weekly_stats, median
# per-player CV across weeks PLAYED (availability is modelled separately in
# sample_weekly, so folding in zeros would double-count injuries as dispersion),
# restricted to each season's fantasy-relevant players (top 24 QB / 60 RB /
# 84 WR / 24 TE) because that is the population a draft board ranks.
#
# The values season_rankings.py uses for its floor/ceiling display band --
# QB .30 / RB .55 / WR .62 / TE .66 -- turn out to be close at RB/WR/TE but
# BADLY wrong at QB: real QB weekly CV is .446, understated by 49%. In a convex
# value function that silently deflated every quarterback.
WEEKLY_CV = {"QB": 0.446, "RB": 0.614, "WR": 0.661, "TE": 0.651}
DEFAULT_CV = 0.614

# ⛔⭐ CV IS NOT A POSITION CONSTANT -- AND FIXING THAT STILL DOES NOT PAY.
# Kept, fully wired, DEFAULT OFF (USE_RANK_CV / draft_engine.BB_RANK_CV = False).
# The measurement below is the whole lesson: it is a textbook case of an INPUT
# improving while the PRODUCT regresses.
#
# INPUT-LEVEL: rank CV beats flat CV out of sample by +7.5% MAE on
# E[max(0, weekly - bar)] -- see below. Genuinely better calibrated.
# BOARD-LEVEL: `bestball_cv_duel.py`, both arms best-ball, 24 paired
# leagues/season, seats swapped, true DK points:
#       2025   -51.1  t -0.87        2024  -176.6  t -3.83
#       COMBINED  -113.9  t -3.05, sign CONSISTENT
# ⭐ And that duel IS interpretable, unlike bestball_duel.py: both arms build the
# same shape (max positional-mix gap 0.3 players), so the seasonal positional bet
# that flipped the earlier test cancels in the pairing.
#
# WHY BOTH CAN BE TRUE: the pick is a RANKING across candidates, not an absolute
# value. Rank CV raises deep players' CV, and in a convex functional that lifts
# their value relative to studs -- so the wizard reaches for high-variance depth
# earlier than it should. Better absolute numbers, worse ordering. Testing
# E[max(0, x - B)] felt like "testing the consumer", but the real consumer is the
# comparison BETWEEN players, and a uniformly better-calibrated input can still
# order them worse.
# ⚠ Do not re-ship this on the strength of the +7.5% alone.
#
# ⛔ SIX ARMS HAVE NOW BEEN TRIED. None wins. Same duel, 24 paired leagues/
# season (v0 reproduces the number above exactly, so the arms are comparable):
#       v0  shipped table, RB/WR/TE, full-sample medians   -113.9  t -3.05
#       v1  + drop TE, + the OOS-VALIDATED train fit       -116.8  t -3.44
#       v2  + renormalise to preserve mean CV              -100.3  t -2.82
#       v3  zeros-clean refit (below), RB+WR, raw          -141.6  t -4.88
#       v3n + renormalised                                 -123.4  t -3.55
#       v3w WR ONLY, zeros-clean, renormalised              -54.2  t -1.47
# The best arm merely stops losing significantly. Sign consistent both seasons
# in every arm. Mix gap <=0.6 players throughout, so none of this is the
# position-mix confound that ruined bestball_duel.py.
#
# ⭐ WHY -- THE BIAS IS THE WHOLE STORY, AND MAE HID IT. cv_ci-style MAE says
# rank CV is better by 7.5-8%. SIGNED bias by tier says the opposite where it
# counts. Predicted vs realised E[max(0, week - B)] at B=16, OOS, ex-ante
# population (+ = OVER-values):
#       RB top12   flat +29%   rank  +6%
#       RB 13-24   flat +45%   rank +17%
#       RB 25-48   flat +18%   rank +39%
#       RB   49+   flat  -1%   rank +87%     <-- the entire problem
# MAE still improves because the top tiers carry far more absolute value
# (3.46 vs 0.53 pts), so halving a +29% error on a stud outweighs creating a
# +87% error on a scrub. The board does not care about aggregate MAE -- it cares
# about the COMPARISON, and that is exactly what this breaks.
#
# ⭐⛔ ROOT CAUSE OF THE STEEPNESS: ZEROS. test_rank_dependent_cv.py reads
# player_history.db weekly_snapshots, which carries 0.0 rows for weeks a player
# did not produce (22.9% of all rows). CV was computed over them. This module
# gates availability SEPARATELY in sample_weekly, so those zeros are
# DOUBLE-COUNTED as dispersion -- and they land hardest on deep players, who
# miss the most games. Refitting on produced weeks only:
#       WR spread top12->49+   .257 -> .149   (42% of the gradient was availability)
#       RB 49+                 1.080 -> .862
# and the bias at B=16 improves to RB 49+ +42% (from +87%) and WR 49+ +7% (from
# +33%). Genuinely better -- and STILL loses the duel. That is v3 above.
#
# ✅ THE FLAT CONSTANTS ARE NOT AFFECTED. fit_weekly_cv.py's source is 72.4%
# zero rows, but its relevance filter removes almost all of them: inside the
# population that actually set WEEKLY_CV only 2.5% of rows are 0.0, and
# refitting on produced weeks moves QB -.001 / RB -.009 / WR -.033 / TE -.059.
# The zeros problem is specific to a GRADIENT, whose deep bucket is deliberately
# populated with the very players the relevance filter was excluding.
# (TE -.059 is the one non-trivial cell and has not been duelled on its own.)
#
# ⛔ TREAT THIS LEVER AS CLOSED AT THE CV LEVEL. Six arms, two of them correctly
# specified, all negative. If it is ever revisited the target is the FUNCTIONAL,
# not the numbers: the gamma is still too fat in the far tail (RB 49+ empirical
# P(week>30) is 0.8% against 2.4% implied), so a saturating tail, an empirical
# marginal, or valuing a pick by win-probability rather than summed option value
# are the live ideas. Re-tuning CV constants is not.
#
# ---- the fit itself, which stands on its own merits ----------------------
# `test_rank_dependent_cv.py` on 12 seasons (2014-2025, weekly_snapshots; PPR
# stands in for DK because on the 1,096 overlapping player-seasons the two CVs
# correlate at 1.0000, median abs diff 0.000).
#
# Two objections had to fall first, and both did:
#  1. LEAK. The original gradient was bucketed by REALIZED season rank, which
#     you do not know on draft day and which shares its weeks with the variance
#     being measured. Re-fitted by EX-ANTE rank (prior season points/game) the
#     gradient SURVIVES nearly intact -- RB spread .372 -> .305, WR .238 -> .205
#     -- so the leak was worth only ~15-18% of it.
#  2. "IT IS JUST ARITHMETIC" (CV = sd/mean rises as the mean falls). True and
#     IRRELEVANT for a predictive model: what this needs is the right sd, and
#     sd = mu * CV(mu). Whether the relationship comes from Poisson-like
#     counting or from role instability does not change which number predicts
#     better. The arbiter is the CONSUMER functional, not the CV itself.
#
# Out-of-sample (fit 2014-2019, test 2020-2025), MAE against realised
# E[max(0, weekly - bar)] -- the exact quantity this module computes, with both
# arms fitted on the SAME relevant-player population so the flat arm is the best
# single constant rather than a strawman:
#       bar   0    -3.1%   (flat wins: at bar 0 the functional IS the mean, so
#                            dispersion cannot matter and the extra parameter
#                            only adds noise -- a good sign, not a bad one)
#       bar   5   +10.8%      bar  10   +6.1%
#       bar  15    +7.4%      bar  20   +9.2%
#       ALL       +7.5%
# The gain lands where the bar is non-trivial, i.e. on DEPTH picks -- rounds
# 8-20, which is where best ball is won and where the flat constant was
# under-pricing the high-variance lottery tickets the format hunts.
#
# ⛔ QB IS DELIBERATELY EXCLUDED. Its ex-ante spread is .019 (top12 .397 vs
# 13-24 .416) -- indistinguishable from noise, unlike RB .305 and WR .205.
# Applying a gradient there would be fitting a bucket boundary to nothing.
WEEKLY_CV_BY_RANK = {
    "RB": ((12, 0.524), (24, 0.554), (48, 0.671), (10**6, 0.829)),
    "WR": ((12, 0.506), (24, 0.564), (48, 0.600), (10**6, 0.711)),
    "TE": ((12, 0.601), (24, 0.650), (10**6, 0.724)),
}
# ⚠ Buckets were fitted against rank-by-PRIOR-SEASON-points, but production keys
# them on `_posrank` (our projection's within-position rank), because that is
# the only ex-ante rank available for a rookie or a changed role. The projection
# is the better estimator of the two, so this should if anything sharpen the
# split -- but the boundaries were not fitted on it, which is the one untested
# join in this change.


# ⛔ OFF: better-calibrated input, worse board (-113.9, t -3.05, both seasons).
USE_RANK_CV = False     # A/B switch; draft_engine mirrors BB_RANK_CV onto this


def rank_cv(pos: str, posrank) -> float:
    """Weekly CV for a player at `posrank` within `pos`. Falls back to the flat
    position constant when the position has no fitted gradient (QB) or the rank
    is unknown -- an unranked player must not silently inherit the deep-bucket
    CV, which is the highest one and would inflate his value."""
    table = WEEKLY_CV_BY_RANK.get(pos) if USE_RANK_CV else None
    if table is None or posrank is None:
        return WEEKLY_CV.get(pos, DEFAULT_CV)
    try:
        r = float(posrank)
    except (TypeError, ValueError):
        return WEEKLY_CV.get(pos, DEFAULT_CV)
    for hi, cv in table:
        if r <= hi:
            return cv
    return table[-1][1]

WEEKS = 17
BYE_WEEKS = range(4, 15)          # the window real NFL byes fall in

# DK best ball: QB/2RB/3WR/TE + 1 FLEX(RB/WR/TE), 20 roster spots, no K/DST.
DK_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
DK_FLEX = ("RB", "WR", "TE")


def weekly_params(p: dict) -> tuple[float, float, int]:
    """(per-game mean, weekly CV, bye week) for one board player.

    `pts` is a SEASON total over `games` games, so the per-game mean is the
    conditional mean given he plays -- which is what we sample, gated by an
    availability draw. A player with no bye on the board (rookie/UDFA rows, or
    an export predating the bye column) gets bye 0, i.e. no bye week; his
    missed time still shows up through the availability rate.
    """
    pts = float(p.get("pts") or 0.0)
    games = float(p.get("games") or 17.0)
    games = min(max(games, 1.0), float(WEEKS))
    mu = pts / games
    # `_posrank` is attached by draft_engine.compute_values (per-league, from our
    # projection); `pos_rank` is the board's own column, used when this module is
    # driven directly. Either is ex-ante. Neither present -> flat position CV.
    cv = rank_cv(p.get("pos"), p.get("_posrank", p.get("pos_rank")))
    bye = int(p.get("bye") or 0)
    return mu, cv, bye


def sample_weekly(p: dict, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """(n_sims, WEEKS) of simulated weekly points, zero when he does not play.

    Gamma rather than Normal: weekly fantasy scores are non-negative and
    right-skewed, and a Normal would hand out negative points and understate the
    upper tail -- the tail is the whole point in best ball.
    """
    mu, cv, bye = weekly_params(p)
    if mu <= 0:
        return np.zeros((n_sims, WEEKS))
    shape = 1.0 / (cv * cv)
    draws = rng.gamma(shape, mu / shape, size=(n_sims, WEEKS))

    # availability: `games` counts games PLAYED, out of the non-bye weeks
    playable = WEEKS - (1 if 1 <= bye <= WEEKS else 0)
    q = float(p.get("games") or 17.0) / max(playable, 1)
    q = min(max(q, 0.0), 1.0)
    live = rng.random((n_sims, WEEKS)) < q
    if 1 <= bye <= WEEKS:
        live[:, bye - 1] = False
    return draws * live


def roster_bars(roster: list[dict], n_sims: int, rng: np.random.Generator,
                slots: dict | None = None, flex: tuple = DK_FLEX
                ) -> dict[str, np.ndarray]:
    """Per-position (n_sims, WEEKS) samples of the bar B_p a NEW player at p
    must beat. See the module docstring for why B_p = min(A[n_p-1], F).

    Sampled once per pick and reused for every candidate, which is what keeps
    this live-draft fast: the bar is a property of YOUR roster, not of the
    player you are considering.
    """
    slots = slots or DK_SLOTS
    by_pos: dict[str, list[np.ndarray]] = {}
    for p in roster:
        by_pos.setdefault(p.get("pos"), []).append(sample_weekly(p, n_sims, rng))

    # kth[pos][k] = the k-th best score at pos in each (sim, week); 0 if absent
    def kth(pos: str, k: int) -> np.ndarray:
        arrs = by_pos.get(pos) or []
        if len(arrs) < k:
            return np.zeros((n_sims, WEEKS))
        stack = np.stack(arrs, axis=2)                     # (sims, weeks, n)
        return np.partition(stack, -k, axis=2)[:, :, -k]

    # F = best flex-eligible player NOT already in a dedicated slot = the max
    # over flex positions of that position's (n_q + 1)-th best.
    F = np.zeros((n_sims, WEEKS))
    for q in flex:
        F = np.maximum(F, kth(q, slots.get(q, 0) + 1))

    bars: dict[str, np.ndarray] = {}
    for pos, n_p in slots.items():
        cut = kth(pos, n_p)                                # A[n_p - 1]
        bars[pos] = np.minimum(cut, F) if pos in flex else cut
    return bars


def marginal_value(cand: dict, bars: dict[str, np.ndarray], n_sims: int,
                   rng: np.random.Generator) -> float:
    """Expected FULL-SEASON points this player adds to your optimal lineups.

    Week-aligned against the bar, so a candidate whose bye collides with the
    week your roster is already thin is correctly worth less than one whose
    does not.
    """
    b = bars.get(cand.get("pos"))
    if b is None:
        return 0.0
    x = sample_weekly(cand, n_sims, rng)
    return float(np.maximum(0.0, x - b).sum(axis=1).mean())


def value_board(remaining: list[dict], roster: list[dict], n_sims: int = 300,
                seed: int = 0, slots: dict | None = None) -> dict[str, float]:
    """{player_id: marginal best-ball value} for every remaining player.

    One shared RNG stream and one shared bar sample, so candidates are compared
    against the IDENTICAL simulated roster rather than each getting its own
    draw -- otherwise the ranking would carry sampling noise between players
    that has nothing to do with their value.
    """
    rng = np.random.default_rng(seed)
    bars = roster_bars(roster, n_sims, rng, slots=slots)
    return {p["id"]: marginal_value(p, bars, n_sims, rng) for p in remaining}
