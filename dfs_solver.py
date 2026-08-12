"""
dfs_solver.py -- portable Monte-Carlo DFS SOLVER for the app (numpy + pandas only,
no pipeline imports, so it runs on Streamlit Cloud and headless in tests).

Port of the pipeline's `dfs_sim.py` (correlated outcome sim + realistic field +
candidate generation + coverage-greedy portfolio) and `optimizer.py`'s payout
synthesis, extended with the things the interactive tool needs:

  * per-lineup EQUITY METRICS (EV / ROI / P(cash) / P(top 1%) / P(win) / ceiling)
    so every candidate can be ranked and browsed, not just the chosen portfolio
  * locks / bans / exposure caps threaded through candidate generation
  * DK SHOWDOWN (1 CPT @1.5x pts & salary + 5 FLEX, single game)
  * game board + stack board scored off the SAME sim (best games, best stacks)

⚠ HONESTY NOTE (load-bearing -- see the DFS backtests): the simulated field is
built from OUR projections, so a candidate's SIM ROI is optimistic by
construction -- the field cannot disagree with us the way the real field does.
Sim ROI is a RANKING device (the bias is common to every candidate); it is NOT a
forecast. `TIER_BASELINE` carries the measured real-field ROI per contest tier
for the reality anchor, and the UI shows both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SALARY_CAP = 50_000
CLASSIC_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}      # + 1 FLEX (RB/WR/TE)
FLEX_ELIGIBLE = ("RB", "WR", "TE")
DK_COLUMNS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
SD_COLUMNS = ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"]
SD_SIZE = 6
CPT_MULT = 1.5                       # DK showdown captain: 1.5x points AND 1.5x salary

# Position CV (std/mean of startable real DK scores), calibrated on 2023-24.
CEIL_CV = {"QB": 0.49, "RB": 0.65, "WR": 0.68, "TE": 0.63, "DST": 0.50, "K": 0.45}

# MEASURED role-pair DK correlations (2023-24). Same-team and opposing-team pairs;
# cross-game = 0. K rows are a prior (kickers weren't in the measured corpus).
POS_SAME = {frozenset(["QB", "WR"]): 0.38, frozenset(["QB", "TE"]): 0.26, frozenset(["QB", "RB"]): 0.10,
            frozenset(["WR", "WR"]): 0.05, frozenset(["RB", "RB"]): -0.08, frozenset(["WR", "RB"]): 0.02,
            frozenset(["WR", "TE"]): 0.05, frozenset(["TE", "RB"]): 0.02, frozenset(["TE", "TE"]): 0.05,
            frozenset(["QB", "DST"]): 0.08, frozenset(["WR", "DST"]): 0.03, frozenset(["RB", "DST"]): 0.05,
            frozenset(["TE", "DST"]): 0.03,
            frozenset(["K", "QB"]): 0.15, frozenset(["K", "WR"]): 0.05, frozenset(["K", "RB"]): 0.08,
            frozenset(["K", "TE"]): 0.05, frozenset(["K", "DST"]): 0.10, frozenset(["K", "K"]): 0.05}
POS_OPP = {frozenset(["QB", "WR"]): 0.06, frozenset(["QB", "TE"]): 0.06, frozenset(["WR", "WR"]): 0.03,
           frozenset(["RB", "RB"]): -0.13, frozenset(["QB", "RB"]): 0.01, frozenset(["WR", "RB"]): -0.02,
           frozenset(["WR", "TE"]): 0.03,
           frozenset(["QB", "DST"]): -0.20, frozenset(["WR", "DST"]): -0.10, frozenset(["RB", "DST"]): -0.08,
           frozenset(["TE", "DST"]): -0.08,
           frozenset(["K", "QB"]): 0.02, frozenset(["K", "DST"]): -0.12, frozenset(["K", "K"]): 0.02}


# ── pool adapter ──────────────────────────────────────────────────────────────

def _num(v, default: float = 0.0) -> float:
    """Bundle value -> a real float. None, NaN and junk all collapse to `default`
    (see the NaN-is-truthy note in prepare_pool)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if (np.isnan(f) or np.isinf(f)) else f


def prepare_pool(players: list[dict], showdown: bool = False) -> pd.DataFrame:
    """Bundle player records -> the frame the sim expects. Keeps only rows that
    can actually be rostered (salary + projection). `game_id` is derived from the
    unordered team/opp pair so same-game correlation works without a bundle change.

    Classic drops K (not a DK-classic position); showdown keeps it.
    """
    ok_pos = ("QB", "RB", "WR", "TE", "DST") + (("K",) if showdown else ())
    rows = []
    for p in players:
        # NB: `x or 0` is WRONG here -- the bundle carries missing numbers as NaN
        # (players_df runs pd.to_numeric), and NaN is TRUTHY, so `or 0` passes it
        # straight through and `NaN <= 0` is False. That let unsalaried players
        # into the pool and turned every downstream salary sum into NaN.
        sal = _num(p.get("salary"))
        proj = _num(p.get("proj"))
        pos = str(p.get("pos") or "").upper()
        if sal <= 0 or proj <= 0 or pos not in ok_pos:
            continue
        team, opp = str(p.get("team")), str(p.get("opp") or "?")
        rows.append({
            "id": p["id"], "name": p.get("name"), "position": pos, "team": team, "opp": opp,
            "home": bool(p.get("home")), "game_id": "@".join(sorted([team, opp])),
            "salary": sal, "proj": proj,
            "ceil": _num(p.get("ceil"), proj * 1.4) or proj * 1.4,
            "floor": _num(p.get("floor"), proj * 0.6) or proj * 0.6,
            "est_ownership": _num(p.get("own"), 8.0),
            "imp": _num(p.get("imp"), np.nan) or np.nan,
            "value": round(proj / (sal / 1000.0), 2),
        })
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


def pool_report(players: list[dict], showdown: bool = False) -> dict:
    """What got dropped and why -- surfaced in the UI so an empty/odd pool is
    never a silent failure."""
    ok_pos = ("QB", "RB", "WR", "TE", "DST") + (("K",) if showdown else ())
    return {"n_in": len(players),
            "no_salary": sum(1 for p in players if _num(p.get("salary")) <= 0),
            "no_proj": sum(1 for p in players if _num(p.get("proj")) <= 0),
            "wrong_pos": sum(1 for p in players
                             if str(p.get("pos") or "").upper() not in ok_pos)}


# ── 1. correlated outcome simulation ──────────────────────────────────────────

def _corr_matrix(pool: pd.DataFrame) -> np.ndarray:
    """Player x player correlation from the measured position-pair structure,
    projected to the nearest PSD with a unit diagonal (a valid copula)."""
    pos = pool["position"].astype(str).str.upper().to_numpy()
    teams = pool["team"].astype(str).to_numpy()
    games = pool["game_id"].astype(str).to_numpy()
    n = len(pool)
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if games[i] != games[j]:
                c = 0.0
            elif teams[i] == teams[j]:
                c = POS_SAME.get(frozenset([pos[i], pos[j]]), 0.0)
            else:
                c = POS_OPP.get(frozenset([pos[i], pos[j]]), 0.0)
            C[i, j] = C[j, i] = c
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-4, None)
    C = (V * w) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def sim_scores(pool: pd.DataFrame, n_sims: int, seed: int, corr: str = "matrix",
               a_team: float = 0.68, a_game: float = 0.38) -> np.ndarray:
    """[n_sims, n_players] correlated DK draws: lognormal marginals (mean = proj,
    position CV) through a copula.

    corr='matrix' (DEFAULT) = the measured Pearson pair structure (POS_SAME/POS_OPP).

    ⛔⭐ 'factor' WAS THE DEFAULT AND IS FALSIFIED (measured 2026-08-12, 9,049
    player-weeks 2023-25, real DK points, latent = empirical normal score within
    position x salary quintile). It imposes rho = a_team^2 + a_game^2 = 0.607 on
    EVERY same-team pair. Realized:

        same-team   QB-WR 0.329   QB-TE 0.317   QB-RB 0.085
                    WR-WR 0.014   RB-WR -0.008  RB-RB -0.043

    Team co-movement in DK points is ONLY the QB<->pass-catcher link; there is no
    general "the offense booms together" factor (WR-WR is 0.014). A single team
    loading cannot even fit that shape -- b_QB*b_WR = .33 with b_WR^2 = .014 needs
    b_QB = 2.8. Its co-boom claim fails the same way: measured P(both above their own
    90th pct)/0.01 is 2.93x for QB-WR but 1.03x for WR-WR and 0.72x for RB-RB, where
    'factor' asserts 3.94x for all of them. The POS_SAME table it was supposed to
    improve on is well calibrated (.38/.329, .26/.317, .10/.085, -.08/-.043).

    Independent second confirmation, from a different statistic: against 311 real
    cached GPP fields (31.6M entries), 'matrix' reproduces the field's realized score
    dispersion (sd/mean 0.200 vs 0.190, p99/mean 1.48x vs 1.45x) while 'factor'
    over-disperses it by 23% (0.233, 1.60x). The old justification was a -58% vs -68%
    GPP ROI comparison, which cannot discriminate in a format whose median week is
    -100% and where one week has swung a variant by +3233%.

    ⚠ REMAINING, MEASURED, NOT YET BUILT: a Gaussian copula at the measured rho does
    under-model the QB-axis tail specifically -- QB-WR co-boom 2.29x simulated vs
    2.93x realized. A GLOBAL t-copula is the wrong repair (t5 gives 1.55x on WR-WR
    against a realized 1.03x, and 1.35x on RB-RB against 0.72x): it fixes the QB pairs
    and breaks every other pair. The right fix is tail dependence on the QB ->
    pass-catcher axis ONLY.

    corr='factor'  = the old uniform game+team factor, kept for A/B only.
    corr='tcopula' = measured structure plus GLOBAL tail dependence; needs scipy and
                     falls back to 'matrix' without it. See the caveat above before
                     using it.
    """
    rng = np.random.default_rng(seed)
    proj = np.maximum(pool["proj"].to_numpy(dtype=float), 0.1)
    cv = pool["position"].map(CEIL_CV).fillna(0.55).to_numpy(dtype=float)
    s2 = np.log(1 + cv ** 2)
    s = np.sqrt(s2)
    m = np.log(proj) - s2 / 2

    if corr == "tcopula":
        try:
            from scipy import stats
        except ImportError:
            corr = "matrix"                       # scipy isn't a Cloud dependency
    if corr == "tcopula":
        from scipy import stats
        df = 5
        L = np.linalg.cholesky(_corr_matrix(pool))
        z = rng.standard_normal((n_sims, len(pool))) @ L.T
        w = rng.chisquare(df, size=(n_sims, 1))
        tv = z * np.sqrt(df / w)
        latent = stats.norm.ppf(np.clip(stats.t.cdf(tv, df), 1e-6, 1 - 1e-6))
    elif corr == "matrix":
        L = np.linalg.cholesky(_corr_matrix(pool))
        latent = rng.standard_normal((n_sims, len(pool))) @ L.T
    else:
        teams = pd.factorize(pool["team"].astype(str))[0]
        games = pd.factorize(pool["game_id"].astype(str))[0]
        a_idio = np.sqrt(np.maximum(1 - a_team ** 2 - a_game ** 2, 1e-6))
        latent = (a_game * rng.standard_normal((n_sims, games.max() + 1))[:, games]
                  + a_team * rng.standard_normal((n_sims, teams.max() + 1))[:, teams]
                  + a_idio * rng.standard_normal((n_sims, len(pool))))
    return np.exp(m[None, :] + s[None, :] * latent)


# ── 2/3. lineup generation ────────────────────────────────────────────────────

# Game-key multiplier for FIELD generation. Calibrated by sweep against the real field's
# measured 2.6-players-from-one-game: (1.0,1.0)->2.16, (1.15,1.35)->2.53, (1.4,2.1)->4.68,
# (2.6,3.6)->7.76. Our candidates no longer use this mechanism at all (see build_classic).
FIELD_INTENSITY = (1.15, 1.35)


def _greedy(pos, sal, key, cap, rng, stack_game=None, games=None,
            intensity=(1.4, 2.3), second_game=None, must=None, banned=None):
    """9 player indices (DK-classic valid) chosen greedily by `key`; None if
    infeasible. `stack_game` biases toward one game (QB + pass-catchers +
    bring-back), `intensity` sets stack size (light -> balanced, heavy ->
    onslaught), `second_game` adds the double-game-stack archetype. `must` seeds
    locked players; `banned` is a boolean mask of excluded rows."""
    k = key.copy()
    if banned is not None:
        k = np.where(banned, -np.inf, k)
    if stack_game is not None and games is not None:
        k = k * np.where(games == stack_game, rng.uniform(*intensity), 1.0)
    if second_game is not None and games is not None:
        k = k * np.where(games == second_game, rng.uniform(1.3, 1.9), 1.0)
    order = np.argsort(-k)

    need = dict(CLASSIC_SLOTS)
    have = {p: 0 for p in need}
    flex = False
    chosen, stot = [], 0.0

    for j in (must or []):                                  # locks first
        p = pos[j]
        if p in need and have[p] < need[p]:
            have[p] += 1
        elif p in FLEX_ELIGIBLE and not flex:
            flex = True
        else:
            return None                                     # locks are infeasible together
        chosen.append(j)
        stot += sal[j]
    if stot > cap:
        return None
    seen = set(chosen)

    for j in order:
        if len(chosen) == 9:
            break
        if j in seen or not np.isfinite(k[j]):
            continue
        p = pos[j]
        left = 9 - len(chosen)
        if stot + sal[j] > cap - (left - 1) * 2500:          # keep the rest affordable
            continue
        if p in need and have[p] < need[p]:
            have[p] += 1
        elif p in FLEX_ELIGIBLE and not flex:
            flex = True
        else:
            continue
        chosen.append(j)
        seen.add(j)
        stot += sal[j]
    return chosen if (len(chosen) == 9 and stot >= 43_000) else None


# ── 2b. near-optimal, structure-explicit construction ─────────────────────────
#
# ⛔⭐ WHY THIS REPLACED THE GAME-KEY MULTIPLIER (measured 2026-08-12).
#
# `_greedy` above biases a whole GAME's key by `intensity` and reserves $2,500 per
# unfilled slot. Both were mis-scaled, and the two faults compounded:
#
#   * the archetype labelled "QB+2" (intensity 1.4-2.1) actually put **4.68** players
#     in the QB's game; "onslaught" (2.6-3.6) put **7.76** of 9. Real GPP winners
#     average 3.1 and the real field 2.6.
#   * keyed on raw projection, the greedy averaged **122.4** projected points against
#     a true MILP optimum of **134.3** on the same pool -- 11.9 points left on the
#     table every lineup.
#
# The second fault CAUSED the apparent contrarianism. The MILP max-projection lineup
# on that pool carries **165.4 cumulative ownership** and an ownership floor is
# non-binding all the way to 160 -- i.e. near-chalk costs ZERO projection; it is
# where the efficient frontier already sits. Our ~58 cum own was never a leverage
# decision, it was the 12-point deficit showing up as ownership. That is also why
# `own_band` no-opped: only 17 of 600 candidates could reach 110-175, because you
# cannot make chalk out of lineups that are 12 points short.
#
# So construction is split into the two things it was conflating:
#   1. get ON the frontier (`_optimize`: seed + local swap search, numpy only --
#      scipy is not a Streamlit Cloud dependency, so no MILP here)
#   2. buy differentiation DELIBERATELY as an explicit structure (`ARCHETYPES`),
#      whose projection cost is then reportable per slate (QB+2+bring-back measured
#      -7.0 points off the optimum, so ~5% of projection for the correlation).

def _pos_counts_ok(cnt: dict, total: int) -> bool:
    """DK classic feasibility on position counts (FLEX = the 9th of RB/WR/TE)."""
    if cnt.get("QB", 0) != 1 or cnt.get("DST", 0) != 1:
        return False
    rb, wr, te = cnt.get("RB", 0), cnt.get("WR", 0), cnt.get("TE", 0)
    return (2 <= rb <= 3 and 3 <= wr <= 4 and 1 <= te <= 2
            and rb + wr + te == 7 and total == 9)


def _best_at_or_below(sal, key, mask):
    """(salary_grid, running_max_key, argmax_idx) for the players in `mask`, sorted by
    salary -- so "best key affordable at <= X" is one searchsorted away. This is what
    makes the funded 2-swap below cheap enough to run per candidate lineup."""
    idx = np.where(mask)[0]
    if not len(idx):
        return None
    o = idx[np.argsort(sal[idx])]
    k = key[o]
    run = np.maximum.accumulate(k)
    arg = o[np.maximum.accumulate(np.where(k == run, np.arange(len(k)), -1))]
    return sal[o], run, arg


def _funded_swap(pos, sal, key, cap, cur, inl, bad, pinned, role_min):
    """One best FUNDED 2-swap: drop two players, add two, where the pair is affordable
    only together.

    A cap-binding lineup is a 1-swap local optimum by construction -- every single
    upgrade breaks the cap, so the search halts ~10 projected points short of the
    frontier even though a "downgrade one slot to fund an upgrade elsewhere" move
    exists. For each dropped pair the best partner is looked up in O(1) from
    `_best_at_or_below`, so the whole neighbourhood is ~36 x n and vectorized.
    """
    base = sal[cur].sum()
    movable = [i for i in cur if i not in pinned]
    best = (1e-9, None)
    avail = (~inl) & (~bad) & np.isfinite(key)
    # Tables depend only on `avail`, which is fixed for the whole call -- building them
    # per (pair, position-pair) instead made this ~20ms a lineup.
    POSN = ("QB", "RB", "WR", "TE", "DST")
    tabs = {p: _best_at_or_below(sal, key, avail & (pos == p)) for p in POSN}
    cand = {p: np.where(avail & (pos == p))[0] for p in POSN}
    for a in range(len(movable)):
        for b in range(a + 1, len(movable)):
            i1, i2 = movable[a], movable[b]
            rest = [j for j in cur if j not in (i1, i2)]
            budget = cap - (base - sal[i1] - sal[i2])
            # ⛔ STRUCTURE-SAFE BY CONSTRUCTION. An earlier version let a 2-swap drop
            # structural players and tried to force the replacements back into the
            # role masks. It only ever repaired the FIRST short mask (and never a
            # shortfall of 2), so "QB+2 with a bring-back" silently decayed to 1.46
            # pass-catchers and 36% bring-back. Now a pair is simply not considered
            # unless the remaining seven still satisfy every role_min -- the 1-swap
            # pass, which does handle masks correctly, covers structural upgrades.
            if any(int(mask[rest].sum()) < kmin for mask, kmin in role_min):
                continue
            c = {}
            for j in rest:
                c[pos[j]] = c.get(pos[j], 0) + 1
            legal = []
            for p1 in POSN:
                for p2 in POSN:
                    cc = dict(c)
                    cc[p1] = cc.get(p1, 0) + 1
                    cc[p2] = cc.get(p2, 0) + 1
                    if _pos_counts_ok(cc, 9):
                        legal.append((p1, p2))
            for p1, p2 in legal:
                j1s = cand[p1]
                tab = tabs[p2]
                if not len(j1s) or tab is None:
                    continue
                gs, run, arg = tab
                room = budget - sal[j1s]
                ok = room >= gs[0]
                if not ok.any():
                    continue
                j1s, room = j1s[ok], room[ok]
                slot = np.searchsorted(gs, room, side="right") - 1
                j2s = arg[slot]
                gain = key[j1s] + run[slot] - key[i1] - key[i2]
                valid = j2s != j1s                      # can't add the same player twice
                if not valid.any():
                    continue
                gain = np.where(valid, gain, -np.inf)
                t = int(np.argmax(gain))
                if gain[t] > best[0]:
                    best = (float(gain[t]), (i1, i2, int(j1s[t]), int(j2s[t])))
    return best[1]


def _optimize(pos, sal, key, cap, seed_idx, banned=None, pinned=None,
              role_min=None, max_passes: int = 12):
    """Best valid DK-classic lineup reachable from `seed_idx` by local swaps.

    Maximizes `key.sum()` subject to the slot rules, the salary cap, `banned`, and
    `role_min` -- a list of (mask, k) meaning "at least k players from this mask",
    which is how an archetype's structure is enforced. `pinned` players may never
    leave. Returns 9 indices, or None if no valid lineup exists.

    A single pass evaluates every (drop one, add one) move: 9 removals x n adds,
    vectorized per removal, so a pass is 9 numpy ops. Best-improvement, repeated
    until no move gains -- this closes essentially all of the 11.9-point greedy gap
    without an ILP.
    """
    n = len(pos)
    cur = list(dict.fromkeys(int(i) for i in seed_idx))
    if len(cur) != 9:
        return None
    inl = np.zeros(n, dtype=bool)
    inl[cur] = True
    pinned = set(int(p) for p in (pinned or []))
    role_min = role_min or []
    bad = np.zeros(n, dtype=bool) if banned is None else np.asarray(banned, dtype=bool)

    def counts(idx):
        c = {}
        for j in idx:
            c[pos[j]] = c.get(pos[j], 0) + 1
        return c

    if not _pos_counts_ok(counts(cur), 9) or sal[cur].sum() > cap:
        return None
    for mask, k in role_min:
        if int(mask[cur].sum()) < k:
            return None

    for _ in range(max_passes):
        best_gain, best_mv = 1e-9, None
        base_sal = sal[cur].sum()
        for i in cur:
            if i in pinned:
                continue
            rest = [j for j in cur if j != i]
            c = counts(rest)
            budget = cap - (base_sal - sal[i])
            # a candidate j is feasible iff it restores a legal position count,
            # fits the budget, isn't banned/already in, and keeps every role_min
            need_ok = np.zeros(n, dtype=bool)
            for p in ("QB", "RB", "WR", "TE", "DST"):
                cc = dict(c)
                cc[p] = cc.get(p, 0) + 1
                if _pos_counts_ok(cc, 9):
                    need_ok |= (pos == p)
            ok = need_ok & (sal <= budget) & (~inl) & (~bad) & np.isfinite(key)
            for mask, k in role_min:
                have = int(mask[rest].sum())
                if have < k:                      # this add MUST supply the shortfall
                    ok &= mask.astype(bool)
            if not ok.any():
                continue
            gains = np.where(ok, key - key[i], -np.inf)
            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_gain, best_mv = float(gains[j]), (i, j)
        if best_mv is None:
            # 1-swap is exhausted. On a cap-binding lineup that is a local optimum by
            # construction, so try the funded 2-swap before giving up.
            mv2 = _funded_swap(pos, sal, key, cap, cur, inl, bad, pinned, role_min)
            if mv2 is None:
                break
            i1, i2, j1, j2 = mv2
            for i in (i1, i2):
                cur.remove(i)
                inl[i] = False
            for j in (j1, j2):
                cur.append(j)
                inl[j] = True
            continue
        i, j = best_mv
        cur.remove(i)
        cur.append(j)
        inl[i] = False
        inl[j] = True
    return cur if len(cur) == 9 else None


# Archetype = the structure we are deliberately buying, stated as counts rather than
# implied by a key multiplier. `mates` = same-team WR/TE with the QB, `bring` = players
# from the OPPOSING team in that game.
#
# ⭐ WEIGHTS ARE CALIBRATED TO REALIZED ROI, not to how often a structure shows up among
# winners. Measured over 351 cached real contests / 31.8M entries (GPP tiers, entry-
# weighted within contest then contests equally weighted, bootstrap CI over contests).
# A team stack of n = QB + (n-1) same-team mates, so these map onto our archetypes:
#
#   team stack   ROI      95% CI            | bring-back   ROI      95% CI
#   0 (none)   -24.9%  [-38.1, -10.7]       | no         -18.6%  [-21.2, -15.9]
#   2 (QB+1)   -19.3%  [-23.8, -13.5]       | yes         +0.2%  [ -6.5,  +8.8]
#   3 (QB+2)    -7.5%  [-10.9,  -4.1]
#   4 (QB+3)   +19.6%  [ +3.8, +38.6]
#   5 (QB+4)    -8.1%  [-31.9, +21.3]
#
# So: QB+2 and QB+3 are where the money came out, a bring-back is worth ~19 points of
# ROI, and going to QB+4 gives it all back. Weighted accordingly. ⚠ OBSERVATIONAL --
# these are the field's own lineups, so a structure's ROI is confounded with the quality
# of players the people who chose it picked, and GPP means are jackpot-driven. Treated
# as a prior on where to spend candidates, NOT as an EV forecast.
ARCHETYPES = [
    # (label,             weight, mates, bring)
    ("balanced",            0.05,   0,     0),   # -24.9% alone; kept only for coverage
    ("qb1",                 0.05,   1,     0),
    ("qb1_bring",           0.10,   1,     1),
    ("qb2",                 0.10,   2,     0),
    ("qb2_bring",           0.35,   2,     1),   # the measured core
    ("qb3",                 0.15,   3,     0),
    ("qb3_bring",           0.20,   3,     1),
]

# ⭐⭐ STACK DEPTH IS CONDITIONAL ON THE GAME, NOT A CONSTANT.
#
# The flat mix above averaged a 5.0-player game stack on real slates against a real field
# of 2.6 and winners of 3.1, and on the real-field A/B it was the arm's leading weakness:
# it min-cashed 39% less often than gpp_builder's 3.35-stack build (3.67 vs 5.08 per 20)
# and its edge inverted in 2025. But replacing 5.0 with a flat 3.0 would also be wrong.
#
# MEASURED (816 game-weeks, 2023-25, market total pre-lock vs realized DK output):
# P(this game produces the week's best k-stack), by total quartile --
#       k=1     k=2     k=3     k=4     k=5
#   low   2.8%    2.8%    2.8%    1.2%    2.0%
#   high 13.2%   14.7%   15.7%   18.1%   16.7%
# The high/low ratio runs 4.7x at k=1 to ~15x at k=4, so THE RETURN TO DEPTH SCALES WITH
# THE ENVIRONMENT. corr(total, excess output) also rises monotonically in k, .298 -> .388.
#
# ⚠ BUT CONDITION ON PROJECTED GAME STRENGTH, NOT ON THE TOTAL ITSELF. Controlling for the
# game's PROJECTED k-stack, the total's partial correlation is only .153-.190 and it SHRINKS
# as k grows (.190 at k=1 -> .153 at k=5) -- i.e. the projection already carries most of
# what the total knows, which is unsurprising when player projections are built off team
# implied totals. So depth scales with projected strength and the total is a small tilt.
#
# ⛔ GAME SCRIPT DOES NOT ENTER *SIZE*. By implied-total gap the same table is flat at every
# k (close 6.5/5.6/6.2/5.9/6.5 vs blowout 6.1/6.9/6.5/6.5/6.9). That is a statement about
# HOW MANY, not about WHOM -- blowout risk may well decide which side of a game to stack,
# which this measurement cannot see and which stays open.
STACK_TOTAL_TILT = 0.25          # weight on the total's z-score in game strength
# How sharply candidates concentrate on the strongest games. This, not the depth table,
# is what sets the OVERALL mean game stack, since depth is conditional on the tier a
# candidate lands in. Calibrated by sweep to land the mean near the winners' 3.1.
GAME_SELECT_SHARPNESS = 0.45

# strength tercile -> (label, weight, mates, bring); game stack = 1 + mates + bring
DEPTH_BY_STRENGTH = {
    # game stack = 1 + mates + bring, so these tables average 3.30 / 2.74 / 2.40 and the
    # realized overall mean lands near the winners' 3.1 at GAME_SELECT_SHARPNESS.
    "strong": [("qb2_bring", 0.28, 2, 1), ("qb3_bring", 0.08, 3, 1),
               ("qb2", 0.24, 2, 0), ("qb1_bring", 0.26, 1, 1), ("qb1", 0.14, 1, 0)],
    "mid":    [("qb1_bring", 0.30, 1, 1), ("qb2", 0.20, 2, 0),
               ("qb2_bring", 0.12, 2, 1), ("qb1", 0.38, 1, 0)],
    "weak":   [("qb1", 0.60, 1, 0), ("qb1_bring", 0.28, 1, 1),
               ("qb2", 0.12, 2, 0)],
}

# Cumulative-ownership target band, replacing the old (110, 175).
#
# ⭐ MEASURED ROI BY CUMULATIVE OWNERSHIP (same corpus):
#     <100  -48.7%  [-55.7, -40.3]      140-160   +5.4%  [ -8.3, +25.4]
#  100-120  -28.6%  [-36.0, -20.8]      160-180   -3.5%  [-11.9,  +6.1]
#  120-140  -12.1%  [-19.5,  -3.9]         180+   -1.3%  [ -7.1,  +4.8]
#
# Monotone up to 140-160 and then it stalls -- and DUPLICATION is why. Mean duplicate
# entries per distinct lineup in a 100k+ field: 1.05 below 100 cum own, 1.50 at 140-160,
# 2.64 at 160-180, 6.97 at 180+ (worst single lineup: 409 copies). So the upside of chalk
# is real but gets diluted by dupes past ~170.
#
# ⛔ The old solver sat at 61 cum own -- squarely in the -48.7% bucket -- and the (110,175)
# band could not fix it, because only 17 of 600 candidates could even reach the band while
# construction was 12 projected points short of the frontier.
OWN_BAND_DEFAULT = (140, 170)


def _stack_scores(pos, key, team, opp, banned, mates_n, bring_n):
    """Per-QB score for the archetype's structure: the QB plus his best `mates_n`
    same-team pass-catchers plus his best bring-back.

    ⛔ WHY THIS EXISTS. Sampling the QB by his OWN key spreads candidates evenly over
    all 32 quarterbacks, so most lineups are built around a mediocre stack. Measured:
    a QB+2+bring-back build sampled that way averaged 117.2 projected points even with
    ZERO outcome jitter, against 127.3 for the best such structure -- the whole 10-point
    residual was stack SELECTION, not the optimizer and not the shrinkage. A stack is
    only worth its combined pieces, so score it that way.
    """
    n = len(pos)
    out = np.full(n, -np.inf)
    qbs = np.where((pos == "QB") & (~banned) & np.isfinite(key))[0]
    for q in qbs:
        s = key[q]
        if mates_n:
            m = (team == team[q]) & np.isin(pos, ["WR", "TE"]) & (~banned)
            mk = np.sort(key[m])[::-1]
            if len(mk) < mates_n:
                continue
            s += mk[:mates_n].sum()
        if bring_n:
            bmask = (team == opp[q]) & np.isin(pos, ["WR", "TE", "RB"]) & (~banned)
            bk = np.sort(key[bmask])[::-1]
            if len(bk) < bring_n:
                continue
            s += bk[:bring_n].sum()
        out[q] = s
    return out


def _sample_top(scores, rng, k=12, power=4.0):
    """Sample an index from the top-k by score, weighted by score**power. Keeps the
    portfolio spread over several good stacks instead of over every stack."""
    ok = np.where(np.isfinite(scores))[0]
    if not len(ok):
        return None
    top = ok[np.argsort(-scores[ok])[:k]]
    w = np.maximum(scores[top] - scores[top].min() * 0.0, 1e-6) ** power
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return int(top[0])
    return int(rng.choice(top, p=w / tot))


def _seed_structure(pool_pos, key, rng, qb, mates_n, bring_n, team, opp, banned):
    """Pick the structural players (QB + n mates + n bring-back), sampled by key so
    candidates differ run to run. Returns (indices, role_min) or (None, None) if the
    game cannot supply the structure."""
    picks = [int(qb)]
    role_min = []
    for mask_src, need, elig in (
        ((team == team[qb]) & np.isin(pool_pos, ["WR", "TE"]), mates_n, "mate"),
        ((team == opp[qb]) & np.isin(pool_pos, ["WR", "TE", "RB"]), bring_n, "bring"),
    ):
        m = mask_src & (~banned)
        m[qb] = False
        if need <= 0:
            continue
        cand = np.where(m)[0]
        if len(cand) < need:
            return None, None
        # Sample among the plausible ones so candidates differ run to run, but sharply
        # and from a SHORTLIST -- a flat key**2 over every pass-catcher on the roster
        # was a large part of the 10-point stack-selection deficit.
        short = cand[np.argsort(-key[cand])[:max(need + 2, 4)]]
        w = np.maximum(key[short], 1e-6) ** 4
        take = rng.choice(short, size=need, replace=False, p=w / w.sum())
        picks += [int(t) for t in take]
        role_min.append((m.astype(float), need))
    return picks, role_min


MIN_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}      # + 1 FLEX (RB/WR/TE)
MAX_SLOTS = {"QB": 1, "RB": 3, "WR": 4, "TE": 2, "DST": 1}


def _fill_to_nine(pos, sal, key, cap, picks, banned):
    """Complete a partial lineup to a VALID 9.

    Fills the MANDATORY minimums first (QB/RB2/WR3/TE1/DST) and only then the single
    FLEX. Filling by key alone can reach 9 players with no QB and no DST -- a legal-
    looking greedy that fails `_pos_counts_ok` and silently kills the yield.

    Only has to reach FEASIBILITY, cheaply: `_optimize` does the maximizing. Each pick
    reserves the cheapest real options for the slots still open (a flat per-slot
    reserve is what made hard structures infeasible).
    """
    n = len(pos)
    cur = list(picks)
    inl = np.zeros(n, dtype=bool)
    inl[cur] = True
    have = {p: 0 for p in MAX_SLOTS}
    for j in cur:
        have[pos[j]] = have.get(pos[j], 0) + 1

    # mandatory shortfalls, then the flex
    todo = []
    for p, k in MIN_SLOTS.items():
        todo += [p] * max(0, k - have.get(p, 0))
    n_flex = 9 - len(cur) - len(todo)
    if n_flex < 0:
        return None
    todo += [None] * n_flex                      # None = any FLEX-eligible

    for slot in todo:
        remaining = 9 - len(cur) - 1
        elig = np.isin(pos, FLEX_ELIGIBLE) if slot is None else (pos == slot)
        # respect the per-position ceiling for flex picks
        if slot is None:
            for p in FLEX_ELIGIBLE:
                if have.get(p, 0) >= MAX_SLOTS[p]:
                    elig &= (pos != p)
        ok = elig & (~inl) & (~banned) & np.isfinite(key)
        if not ok.any():
            return None
        spent = sal[cur].sum()
        cheap_rest = np.sort(sal[(~inl) & (~banned)])
        cands = np.where(ok)[0]
        # cheapest-first is the feasibility-safe order; _optimize upgrades afterwards
        for j in cands[np.argsort(sal[cands])]:
            floor_rest = cheap_rest[:remaining].sum() if remaining > 0 and len(cheap_rest) >= remaining else 0.0
            if spent + sal[j] + floor_rest <= cap:
                cur.append(int(j))
                inl[j] = True
                have[pos[j]] = have.get(pos[j], 0) + 1
                break
        else:
            return None

    c = {}
    for j in cur:
        c[pos[j]] = c.get(pos[j], 0) + 1
    return cur if (len(cur) == 9 and _pos_counts_ok(c, 9) and sal[cur].sum() <= cap) else None


def build_classic(pos, sal, key, cap, rng, team, opp, mates_n=0, bring_n=0,
                  qb=None, banned=None, must=None):
    """One near-optimal DK-classic lineup with an EXPLICIT structure.

    mates_n / bring_n state the structure directly, so "QB+2 with a bring-back" means
    exactly that -- unlike the old game-key multiplier, which produced 4.7-7.8 players
    from one game while claiming 2-3.
    """
    n = len(pos)
    banned = np.zeros(n, dtype=bool) if banned is None else np.asarray(banned, dtype=bool)
    k = np.where(banned, -np.inf, key)
    role_min, picks = [], list(must or [])
    pinned = set(int(x) for x in (must or []))

    if mates_n or bring_n:
        if qb is None:
            sc = _stack_scores(pos, k, team, opp, banned, mates_n, bring_n)
            qb = _sample_top(sc, rng)
            if qb is None:
                return None
        s, role_min = _seed_structure(pos, k, rng, qb, mates_n, bring_n, team, opp, banned)
        if s is None:
            return None
        picks = list(dict.fromkeys(picks + s))
        # ⛔ THE QB MUST BE PINNED. `role_min`'s masks are defined relative to THIS
        # quarterback's team and opponent, so if a swap replaces him with a different
        # QB the masks still validate while the structure has dissolved -- measured as
        # "qb2" lineups carrying 0.00 same-team pass-catchers.
        pinned.add(int(qb))
    elif must:
        picks = list(dict.fromkeys(picks))

    if len(picks) > 9:
        return None
    c = {}
    for j in picks:
        c[pos[j]] = c.get(pos[j], 0) + 1
    room = {"QB": 1, "DST": 1, "RB": 3, "WR": 4, "TE": 2}
    if any(v > room.get(p, 0) for p, v in c.items()) or sal[picks].sum() > cap:
        return None

    seed = _fill_to_nine(pos, sal, k, cap, picks, banned)
    if seed is None:
        return None
    return _optimize(pos, sal, k, cap, seed, banned=banned,
                     pinned=pinned, role_min=role_min)


def _build_sd(target, floor, make_key, build, relax=(1.0, 0.85, 0.70, 0.0)):
    """Build showdown lineups, relaxing the salary floor if the yield is too low.

    The floor is derived from the most EXPENSIVE six players, but the field's
    lineups are built on a value key that deliberately picks cheap ones -- so a
    single fixed floor can starve field generation to near zero while candidate
    generation (a projection key) sails through. Rather than fail the solve,
    step the floor down until enough lineups exist."""
    seen, lus = set(), []
    for factor in relax:
        if len(lus) >= target:
            break
        f = floor * factor
        for _ in range(target * 4):
            if len(lus) >= target:
                break
            lu = build(make_key(), f)
            if not lu:
                continue
            h = tuple(sorted(lu))
            if h in seen:
                continue
            seen.add(h)
            lus.append(lu)
    return lus


def sd_min_salary(sal: np.ndarray, cap: int = SALARY_CAP) -> float:
    """Minimum spend to accept a showdown lineup.

    DK prices showdown slates on their own scale, so six real showdown players
    fill the cap. Our bundle ships CLASSIC salaries, where the best possible six
    (CPT at 1.5x plus the five next-most-expensive) can land well under $50k --
    a fixed 86%-of-cap gate then rejects every lineup and the solver returns
    nothing. So the gate is relative to what this pool can actually REACH."""
    if len(sal) < 6:
        return 0.0
    top = np.sort(sal)[::-1]
    reachable = top[0] * CPT_MULT + top[1:6].sum()
    return min(0.86 * cap, 0.90 * reachable)


def _greedy_sd(pos, sal, key, cap, rng, teams, n_base, must=None, banned=None,
               min_salary=None):
    """DK SHOWDOWN: 1 CPT + 5 FLEX from a single game, both teams represented.
    Index space is doubled -- [0, n_base) = FLEX rows, [n_base, 2*n_base) = the
    same players as CPT (1.5x salary, 1.5x points). Returns 6 indices in that
    doubled space, or None."""
    k = key.copy()
    if banned is not None:
        k = np.where(banned, -np.inf, k)
    chosen, stot = [], 0.0
    used_base = set()
    cpt = None

    for j in (must or []):
        b = j % n_base
        if b in used_base:
            return None
        if j >= n_base:
            if cpt is not None:
                return None
            cpt = j
        chosen.append(j)
        used_base.add(b)
        stot += sal[j]

    if cpt is None:                                          # pick a captain by key
        cpt_order = [j for j in np.argsort(-k[n_base:]) + n_base
                     if (j % n_base) not in used_base and np.isfinite(k[j])]
        if not cpt_order:
            return None
        cpt = cpt_order[0]
        chosen.append(cpt)
        used_base.add(cpt % n_base)
        stot += sal[cpt]
    if stot > cap:
        return None

    for j in np.argsort(-k[:n_base]):
        if len(chosen) == SD_SIZE:
            break
        if j in used_base or not np.isfinite(k[j]):
            continue
        left = SD_SIZE - len(chosen)
        if stot + sal[j] > cap - (left - 1) * 1000:
            continue
        chosen.append(int(j))
        used_base.add(int(j))
        stot += sal[j]

    if len(chosen) != SD_SIZE:
        return None
    if len({teams[j % n_base] for j in chosen}) < 2:          # DK requires both teams
        return None
    floor = sd_min_salary(sal[:n_base], cap) if min_salary is None else min_salary
    return chosen if stot >= floor else None


def _to_binary(idx_lists, n):
    B = np.zeros((len(idx_lists), n), dtype=np.float64)
    for i, idx in enumerate(idx_lists):
        B[i, idx] = 1.0
    return B


def gen_field(pool, size, seed, sharp_frac=0.6, showdown=False, n_base=None, teams=None):
    """Realistic opponent field (binary [size, n_cols]): `sharp_frac` SHARP lineups
    built on market value (projection/$) + game stacks, the rest recreational
    (ownership-weighted). A purely ownership-driven field is too soft and lets the
    solver win by naive fading.

    The real field piles ~15-22% of ALL lineups onto the single most-popular game;
    independent sampling under-models that and leaves the leverage objective
    toothless, so game choice is concentrated (gw**3)."""
    rng = np.random.default_rng(seed)
    pos = pool["position"].to_numpy()
    sal = pool["salary"].to_numpy(dtype=float)
    proj = pool["proj"].to_numpy(dtype=float)
    value = proj / np.maximum(sal / 1000.0, 0.1)
    own = pd.to_numeric(pool["est_ownership"], errors="coerce").fillna(1).clip(lower=0.4).to_numpy()

    if showdown:
        n = len(pool)
        sal2 = np.concatenate([sal, sal * CPT_MULT])
        val2 = np.concatenate([value, value])
        own2 = np.concatenate([own, own * 0.35])              # CPT ownership is thinner
        pos2 = np.concatenate([pos, pos])
        tm = teams if teams is not None else pool["team"].to_numpy()
        state = {"i": 0}

        def make_key():
            sharp = state["i"] < int(size * sharp_frac)
            state["i"] += 1
            return (val2 if sharp else own2) * rng.uniform(0.55, 1.45, size=2 * n)

        lus = _build_sd(size, sd_min_salary(sal),
                        make_key,
                        lambda k, f: _greedy_sd(pos2, sal2, k, SALARY_CAP, rng, tm, n,
                                                min_salary=f))
        return _to_binary(lus, 2 * n)

    games = pd.factorize(pool["game_id"].astype(str))[0]
    ugames = np.unique(games)
    gw = pd.Series(proj).groupby(games).sum().reindex(ugames).to_numpy()
    gw = gw / gw.sum()
    gwc = gw ** 3
    gwc = gwc / gwc.sum()
    n_sharp = int(size * sharp_frac)

    # ⚠ REAL FIELDS SPEND THE CAP. Measured over 351 cached contests / 31.8M entries, the
    # real field averages $49,127-$49,239 in the GPP tiers; a value-keyed greedy averages
    # $46,244 because value/$1k rewards cheap players. An underspent field is a WEAK
    # benchmark, which flatters every candidate scored against it.
    # ⚠ The floor must be paired with the softened intensity: applied alone it selects for
    # the CONCENTRATED lineups (they were the ones spending up) and the field's game stack
    # went to 6.0 -- worse than the underspend it fixed.
    # ⛔ AND IT MUST RELAX. As a FIXED $48,000 this silently destroyed the field on smaller
    # slates -- 23 lineups built out of 800 on a 295-player pool, which trips solve()'s
    # 50-lineup minimum and errors the whole solve, so the arm just vanished from those
    # slates. Exactly the trap `sd_min_salary` already documents for showdown: gate on what
    # the pool can REACH, and step down rather than fail.
    def _pass(floor, target):
        out = []
        for _ in range(target * 3):
            if len(out) >= target:
                break
            sharp = len(out) < int(target * sharp_frac)
            sg = rng.choice(ugames, p=gwc) if rng.random() < (0.70 if sharp else 0.55) else None
            key = (value if sharp else own) * rng.uniform(0.5 if not sharp else 0.65,
                                                          1.5 if not sharp else 1.35,
                                                          size=len(pool))
            lu = _greedy(pos, sal, key, SALARY_CAP, rng, sg, games, intensity=FIELD_INTENSITY)
            if lu and sal[lu].sum() >= floor:
                out.append(lu)
        return out

    for floor in (48_000, 46_500, 45_000, 0):
        lus = _pass(floor, size)
        if len(lus) >= max(50, size * 0.5):
            break
    return _to_binary(lus, len(pool))


def _game_strength(pool, pos, proj, banned):
    """Per-game strength -> (tercile label, selection weight, QBs) for depth scaling.

    Strength is the game's PROJECTED best-5 output, z-scored, plus a small tilt on the
    game's market total (`imp` summed over its two teams). The tilt is deliberately small:
    controlling for projected output, the total's partial correlation with realized output
    is only .153-.190 and shrinks with stack depth -- the projection already carries most
    of what the total knows.
    """
    gid = pool["game_id"].astype(str).to_numpy()
    ok = ~np.asarray(banned, dtype=bool)
    strength, raw, qb_by_game = {}, {}, {}
    for g in pd.unique(gid):
        m = (gid == g) & ok
        if m.sum() < 5:
            continue
        top5 = float(np.sort(proj[m])[::-1][:5].sum())
        raw[g] = top5
        qb_by_game[g] = [int(i) for i in np.where(m & (pos == "QB"))[0]]
    if not raw:
        return {}, {}, {}
    keys = list(raw)
    v = np.array([raw[g] for g in keys], dtype=float)
    z = (v - v.mean()) / (v.std() or 1.0)
    # total tilt
    if "imp" in pool.columns:
        tot = []
        for g in keys:
            sub = pool[pool["game_id"].astype(str) == g]
            im = pd.to_numeric(sub.groupby("team")["imp"].first(), errors="coerce").dropna()
            tot.append(float(im.sum()) if len(im) >= 2 else np.nan)
        tot = np.array(tot, dtype=float)
        if np.isfinite(tot).sum() >= 3:
            tz = np.where(np.isfinite(tot), tot, np.nanmean(tot))
            tz = (tz - tz.mean()) / (tz.std() or 1.0)
            z = z + STACK_TOTAL_TILT * tz
    lo, hi = np.quantile(z, [1 / 3, 2 / 3])
    gwt = {}
    for g, s in zip(keys, z):
        strength[g] = "strong" if s >= hi else ("weak" if s < lo else "mid")
        # Selection weight: shootouts get picked more, but the sharpness is what sets the
        # OVERALL stack depth, because depth is conditional on the tier a candidate lands
        # in. At 1.1 it funnelled 504 of 600 candidates into "strong" games and the mean
        # game stack came out 3.78; see GAME_SELECT_SHARPNESS.
        gwt[g] = float(np.exp(GAME_SELECT_SHARPNESS * s))
    return strength, gwt, qb_by_game


def tune_chalk(pool, seed, mates_n=2, bring_n=1, target=OWN_BAND_DEFAULT, probe=28,
               grid=(0.0, 0.02, 0.05, 0.08, 0.12, 0.18), min_unique=0.60):
    """Pick the ownership bonus `mu` whose candidates land in the target cum-own band,
    and REPORT what was actually reachable.

    ⛔ THIS REPLACES A HARD BAND FILTER. The old design filtered candidates to a fixed
    (110, 175) cumulative-ownership window; on a live pool only 17 of 600 candidates
    could reach it, so the filter no-opped and the portfolio stayed at 58 cum own while
    the UI implied it had been steered to the winning range. How much ownership a lineup
    can carry depends on the slate AND on the structure -- a QB+2+bring-back commits four
    slots for correlation reasons, which caps reachable chalk (measured: 134 on a pool
    whose unstructured optimum reaches 165). So tune toward the target instead of
    filtering for it, and say plainly when the structure cannot get there.

    Returns (mu, achieved_own, note).
    """
    pos = pool["position"].to_numpy()
    sal = pool["salary"].to_numpy(dtype=float)
    proj = pool["proj"].to_numpy(dtype=float)
    own = pd.to_numeric(pool["est_ownership"], errors="coerce").fillna(0).to_numpy(float)
    team = pool["team"].astype(str).to_numpy()
    opp = pool["opp"].astype(str).to_numpy()
    mid = 0.5 * (target[0] + target[1])
    best = (None, None, 1e18)
    # ⚠ mu IS BOUNDED, AND NOT ONLY FOR TASTE. `own` and `proj` are the same order of
    # magnitude per player, so a large mu turns the objective into an ownership
    # maximizer: at 0.80 the build collapsed to 363 distinct lineups out of 600 and
    # spent $48.4k against a real-field $49.8k, because cheap chalk outranked points.
    # If the band is out of reach we take the best bounded mu and SAY it is
    # structure-limited, rather than buying ownership with everything else.
    for mu in grid:
        rng = np.random.default_rng(seed)
        got, seen = [], set()
        for t in range(probe):
            key = proj + mu * own
            lu = build_classic(pos, sal, key, SALARY_CAP, rng, team, opp,
                               mates_n=mates_n, bring_n=bring_n)
            if lu:
                got.append(own[lu].sum())
                seen.add(tuple(sorted(lu)))
        if not got or len(seen) < min_unique * len(got):
            continue                               # too repetitive to build a portfolio on
        a = float(np.mean(got))
        d = abs(a - mid)
        if d < best[2]:
            best = (mu, a, d)
        if a >= mid:
            break                                  # monotone in mu; no need to go higher
    if best[0] is None:
        return 0.0, None, "could not build probe lineups — ownership left untuned"
    mu, ach = best[0], best[1]
    if ach < target[0]:
        note = (f"target {target[0]}-{target[1]} NOT reachable with this structure — "
                f"best {ach:.0f} at chalk bonus {mu:.2f} (structure-limited)")
    elif ach > target[1]:
        note = f"above target — {ach:.0f} cum own at chalk bonus {mu:.2f}"
    else:
        note = f"in target {target[0]}-{target[1]} — {ach:.0f} cum own at chalk bonus {mu:.2f}"
    return mu, ach, note


def gen_candidates(pool, n, seed, stack=True, versatile=True, showdown=False,
                   locks=None, bans=None, teams=None, chalk=0.0, lam=0.10):
    """Diverse, projection-good candidate lineups (binary [m, n_cols]).

    stack=True forces a game stack (GPP ceiling); stack=False builds balanced
    no-stack lineups (CASH floor). versatile=True rolls a MIX of archetypes per
    candidate -- balanced / QB+2 / onslaught / double-game-stack -- so the
    portfolio spans outcome scenarios rather than one stack shape.

    locks/bans are ROW INDICES into `pool` (showdown: into the doubled space)."""
    rng = np.random.default_rng(seed)
    pos = pool["position"].to_numpy()
    sal = pool["salary"].to_numpy(dtype=float)
    proj = pool["proj"].to_numpy(dtype=float)
    must = list(locks or [])

    if showdown:
        nb = len(pool)
        sal2 = np.concatenate([sal, sal * CPT_MULT])
        proj2 = np.concatenate([proj, proj * CPT_MULT])
        pos2 = np.concatenate([pos, pos])
        tm = teams if teams is not None else pool["team"].to_numpy()
        banned = np.zeros(2 * nb, dtype=bool)
        for b in (bans or []):
            banned[b] = True
            banned[(b + nb) % (2 * nb)] = True                # ban both roles
        spread = 0.55 if stack else 0.30
        lus = _build_sd(n, sd_min_salary(sal),
                        lambda: proj2 * rng.uniform(1 - spread, 1 + spread, size=2 * nb),
                        lambda k, f: _greedy_sd(pos2, sal2, k, SALARY_CAP, rng, tm, nb,
                                                must=must, banned=banned, min_salary=f))
        return _to_binary(lus, 2 * nb)

    banned = np.zeros(len(pool), dtype=bool)
    for b in (bans or []):
        banned[b] = True
    team = pool["team"].astype(str).to_numpy()
    opp = pool["opp"].astype(str).to_numpy()

    # ⭐ DIVERSITY COMES FROM SIMULATED OUTCOMES, NOT ARBITRARY JITTER. The old key was
    # `proj * uniform(1-spread, 1+spread)` at spread=0.55 -- a +-55% shake with no
    # relation to how players actually vary or co-vary. Optimizing draws from the SAME
    # correlated sim the rest of the solve uses means every candidate is optimal in some
    # plausible world, and correlated worlds naturally produce correlated lineups. This
    # is what makes an exact optimizer safe to use: without it, `_optimize` would just
    # return the same frontier lineup every time.
    # `lam` SHRINKS the draw toward projection. A raw draw has position CV 0.49-0.71, so
    # optimizing one costs ~17 projected points against the frontier and reads as wildly
    # contrarian (measured: 102.9 proj / 50 cum own at lam=1.0, against a 134.3 optimum).
    # Small lam is enough -- structure sampling alone already yields ~140 distinct lineups
    # per 150 attempts, so diversity does not have to be bought with noise.
    draws = sim_scores(pool, max(n, 64), seed + 7)
    n_draw = len(draws)
    ownv = pd.to_numeric(pool["est_ownership"], errors="coerce").fillna(0).to_numpy(float)

    if not stack:
        # CASH: no structure, and no chalk bonus -- the disciplined max-projection lineup
        # IS the measured +EV corner, so leave the objective alone.
        seen, lus = set(), []
        for t in range(n * 3):
            if len(lus) >= n:
                break
            d = draws[t % n_draw]
            lu = build_classic(pos, sal, proj + 0.5 * lam * (d - proj), SALARY_CAP, rng,
                               team, opp, banned=banned, must=must)
            if lu:
                h = tuple(sorted(lu))
                if h not in seen:
                    seen.add(h)
                    lus.append(lu)
        return _to_binary(lus, len(pool))

    # GPP: choose the GAME first, then let depth scale with that game's strength -- the
    # order a human builds in, and the order the measurement supports.
    strength, gw, qb_by_game = _game_strength(pool, pos, proj, banned)
    if not len(gw):
        return _to_binary([], len(pool))
    games = list(gw.keys())
    gwt = np.array([gw[g] for g in games], dtype=float)
    gwt = gwt / gwt.sum()

    seen, lus = set(), []
    for t in range(n * 4):
        if len(lus) >= n:
            break
        d = draws[t % n_draw]
        key = proj + lam * (d - proj) + chalk * ownv
        if versatile and rng.random() < 0.05:
            lu = build_classic(pos, sal, key, SALARY_CAP, rng, team, opp,
                               banned=banned, must=must)              # unstacked coverage
        else:
            g = games[int(rng.choice(len(games), p=gwt))]
            qbs = qb_by_game.get(g, [])
            if not qbs:
                continue
            tier = strength[g]
            spec = DEPTH_BY_STRENGTH[tier] if versatile else DEPTH_BY_STRENGTH["strong"]
            w = np.array([s[1] for s in spec], dtype=float)
            w = w / w.sum()
            _, _, mates_n, bring_n = spec[int(rng.choice(len(spec), p=w))]
            qk = np.array([key[q] for q in qbs], dtype=float)
            qb = int(qbs[int(np.argmax(qk))]) if len(qbs) == 1 else int(
                rng.choice(qbs, p=np.maximum(qk, 1e-6) ** 4 / (np.maximum(qk, 1e-6) ** 4).sum()))
            lu = build_classic(pos, sal, key, SALARY_CAP, rng, team, opp,
                               mates_n=mates_n, bring_n=bring_n, qb=qb,
                               banned=banned, must=must)
        if lu:
            h = tuple(sorted(lu))
            if h not in seen:
                seen.add(h)
                lus.append(lu)
    return _to_binary(lus, len(pool))


# ── 4. contest payout curves ──────────────────────────────────────────────────

def synth_pay_by_rank(field_size: int, entry_fee: float, prize_pool: float | None = None,
                      kind: str = "gpp", cash_frac: float = 0.20) -> np.ndarray:
    """Approximate payout schedule by finishing rank (length = field_size, 0 for
    non-paying ranks).

    kind='cash'  -> binary double-up/50-50: everyone above the cash line gets the
                    same multiple (that IS the real structure, so this is exact
                    up to the paying fraction).
    kind='gpp'   -> top-heavy power curve: `cash_frac` of the field cashes, the
                    min-cash is ~2x entry, and the tail is fitted so the payouts
                    sum to the prize pool. A REAL DK payout table is more precise
                    but this carries the right SHAPE, which is what drives the
                    solve.
    prize_pool defaults to field * entry * 0.85 (~15% rake)."""
    n = max(int(field_size), 2)
    pool = float(prize_pool) if prize_pool else n * float(entry_fee) * 0.85
    pbr = np.zeros(n, dtype=float)

    if kind == "cash":
        n_pay = max(1, int(round(n * cash_frac)))
        pbr[:n_pay] = pool / n_pay
        return pbr

    n_pay = max(1, int(round(n * cash_frac)))
    m = 2.0 * float(entry_fee)                                # min-cash ~2x entry
    ranks = np.arange(1, n_pay + 1, dtype=float)
    # power law w/ min-cash floor: pay(r) = m + A * r**-alpha, A solved for the pool
    alpha = 1.0
    tail = ranks ** -alpha
    budget = pool - m * n_pay
    if budget <= 0:                                           # flat-ish (small/soft field)
        pbr[:n_pay] = pool / n_pay
        return pbr
    pbr[:n_pay] = m + budget * tail / tail.sum()
    return pbr


# Measured REAL-field ROI per contest tier -- the reality anchor for the sim's
# optimistic numbers. Sources: winnings_tally (41 slates, exact field + payout,
# 2023-25), cash_tally (20 slates / 21 double-ups), single_entry_gpp (42 slates).
# ⛔⭐ RE-MEASURED 2026-08-12. The previous GPP numbers here (-12/-19/-20/-63) were STALE
# and optimistic by 12-25 points on three of the four tiers. Re-running the very harness
# that produced them (`winnings_tally.py --seasons 2023 2024 2025`) on the same 41
# independent slates / 117 contest-entries now returns:
#
#   tier            was    now
#   milly >=100k    -12%   -30%
#   large 30-100k   -19%   -31%
#   mid 5-30k       -20%   -45%
#   small 1-5k      -63%   -62%     <- essentially unchanged
#   ALL                    -42%
#
# The small tier barely moves while the other three fall hard, which points at the two
# `_market_pool` corrections: the availability filter that was deleting 18.4% of every
# salary-eligible pool (mean salary $3,570), and "a missing stat line scores ZERO, not its
# projection". Together those removed a hindsight guarantee that every cheap punt play
# recorded something -- and punt-heavy construction is exactly what the big-field
# contrarian builds do most, while the small-field build punts least.
#
# Independently corroborated: a separate harness scoring gpp_builder over 46 slates lands
# at -41.3% aggregate against winnings_tally's -42%.
TIER_BASELINE = {
    "cash":       {"roi": +23, "note": "soft big double-ups, 3 seasons, cash rate 62% (CI straddles break-even; NOT re-measured 2026-08-12)"},
    "se_gpp":     {"roi":  -9, "note": "single-entry GPP, 42 slates -- predates the pool fixes, likely optimistic"},
    "gpp_milly":  {"roi": -30, "note": "Milly >=100k entries -- re-measured 2026-08-12 (was -12%, stale)"},
    "gpp_large":  {"roi": -31, "note": "30-100k entries -- re-measured 2026-08-12 (was -19%, stale)"},
    "gpp_mid":    {"roi": -45, "note": "5-30k entries -- re-measured 2026-08-12 (was -20%, stale)"},
    "gpp_small":  {"roi": -62, "note": "1-5k high-entry -- SHARP fields, avoid (re-measured, unchanged)"},
    "showdown":   {"roi": None, "note": "never backtested here -- no measured baseline"},
}

# Contest presets used when the live DK lobby isn't reachable.
PRESETS = [
    dict(key="cash",      label="Cash — big double-up",   kind="cash", size=6000,   fee=5.0,  entries=1,  cash_frac=0.44),
    dict(key="se_gpp",    label="Single-entry GPP",       kind="gpp",  size=5000,   fee=10.0, entries=1,  cash_frac=0.23),
    dict(key="gpp_small", label="Small GPP (1-5k)",       kind="gpp",  size=3000,   fee=20.0, entries=20, cash_frac=0.20),
    dict(key="gpp_mid",   label="Mid GPP (5-30k)",        kind="gpp",  size=20000,  fee=5.0,  entries=20, cash_frac=0.20),
    dict(key="gpp_large", label="Large GPP (30-100k)",    kind="gpp",  size=60000,  fee=5.0,  entries=20, cash_frac=0.20),
    dict(key="gpp_milly", label="Milly (100k+)",          kind="gpp",  size=200000, fee=20.0, entries=20, cash_frac=0.20),
    dict(key="showdown",  label="Showdown GPP",           kind="gpp",  size=20000,  fee=5.0,  entries=20, cash_frac=0.20),
]


def tier_of(size: int, kind: str, entries: int = 1, showdown: bool = False) -> str:
    """Classify a contest into the tier whose measured baseline applies."""
    if showdown:
        return "showdown"                 # no backtest exists for this format
    if kind == "cash":
        return "cash"
    if entries <= 1:
        return "se_gpp"
    if size >= 100_000:
        return "gpp_milly"
    if size >= 30_000:
        return "gpp_large"
    if size >= 5_000:
        return "gpp_mid"
    return "gpp_small"


# ── 5. equity: candidates vs the simulated field, through the payout curve ────

def cand_payouts(cand_scores: np.ndarray, field_scores: np.ndarray,
                 pay_by_rank: np.ndarray, field_size: int) -> np.ndarray:
    """[n_sims, n_cand] payout per candidate per sim.

    Each candidate's SIM score is ranked against the SIMULATED FIELD **in that
    same sim** -- so when the slate booms the field booms too, which is what
    removes the variance bias a fixed threshold would introduce -- then mapped
    through the contest's payout-by-rank."""
    M = len(pay_by_rank)
    n_sims, n_cand = cand_scores.shape
    nf = field_scores.shape[1]
    out = np.empty((n_sims, n_cand))
    ranks = np.empty((n_sims, n_cand))
    for s in range(n_sims):
        fs = np.sort(field_scores[s])
        above = nf - np.searchsorted(fs, cand_scores[s], side="left")
        r = np.clip((above / nf * field_size).astype(int), 0, M - 1)
        out[s] = pay_by_rank[r]
        ranks[s] = r + 1
    return out, ranks


def lineup_metrics(cand_pay, cand_ranks, cand_scores, cand_bin, pool, entry_fee,
                   field_size, showdown=False, n_base=None, field_entries=None):
    """Per-candidate equity table -- the numbers the UI ranks and shows.

    ev / roi           expected payout per entry, and ROI on the entry fee (SIM)
    p_cash             fraction of sims with a non-zero payout
    p_top1 / p_win     fraction finishing in the top 1% / winning outright
    med / p90          median and 90th-percentile lineup score
    own                summed projected ownership (cumulative %)
    dupes              rough expected duplicate entries in the field
    """
    ev = cand_pay.mean(axis=0)
    roi = (ev - entry_fee) / entry_fee * 100.0 if entry_fee else np.zeros_like(ev)
    p_cash = (cand_pay > 0).mean(axis=0) * 100.0
    p_top1 = (cand_ranks <= max(1, field_size * 0.01)).mean(axis=0) * 100.0
    p_win = (cand_ranks <= 1).mean(axis=0) * 100.0
    med = np.median(cand_scores, axis=0)
    p90 = np.percentile(cand_scores, 90, axis=0)

    own = pool["est_ownership"].to_numpy(dtype=float)
    sal = pool["salary"].to_numpy(dtype=float)
    if showdown:
        own = np.concatenate([own, own * 0.35])
        sal = np.concatenate([sal, sal * CPT_MULT])
    cum_own = cand_bin @ own
    salary = cand_bin @ sal

    # DUPLICATION: reported as a RELATIVE risk percentile, not a count. The
    # textbook estimate (product of ownership shares x field size) assumes players
    # are rostered independently, which the real field badly violates -- it
    # clusters on the same stacks -- and it comes out ~100x low. The ORDERING it
    # gives is still right, so rank it and leave the fake precision out.
    with np.errstate(divide="ignore"):
        logp = cand_bin @ np.log(np.clip(own / 100.0, 1e-4, 1.0))
    dup_rank = pd.Series(logp).rank(pct=True).to_numpy() * 100.0

    return pd.DataFrame({"ev": ev, "roi": roi, "p_cash": p_cash, "p_top1": p_top1,
                         "p_win": p_win, "med": med, "p90": p90, "own": cum_own,
                         "salary": salary, "dup_risk": dup_rank})


def solve_portfolio_sum(cand_pay, n_port, cand_bin=None, max_exposure=None):
    """The n_port portfolio maximizing E[TOTAL payout] -- i.e. plain expected value.

    In a multi-entry contest you are paid on EVERY entry, so portfolio EV is the SUM of
    per-entry payouts, and a sum is linear: correlation between your own lineups does not
    change it. Maximizing it therefore means taking the n highest-EV candidates, with no
    diversification at all. That is the honest EV benchmark against `solve_portfolio`'s
    E[max], which is a variance-seeking objective (it deliberately values a lineup that
    wins in scenarios the others miss, even when that lineup is individually poor).

    `max_exposure` is still honoured, which makes this the practitioner's version -- pure
    EV ranking plus a cap on any one player's share -- and lets the A/B separate "EV
    ranking" from "diversification" as the source of any difference.

    NB the SCORER applies displacement (our own entries occupy distinct finishing ranks),
    so if this objective over-picks near-duplicates the measurement will charge it for
    that; it is not being flattered by the linearity it assumes.
    """
    ev = cand_pay.mean(axis=0)
    order = np.argsort(-ev)
    if cand_bin is None or max_exposure is None:
        return [int(j) for j in order[:n_port]]
    chosen, usage = [], np.zeros(cand_bin.shape[1])
    for j in order:
        if len(chosen) >= n_port:
            break
        if (usage[cand_bin[j] > 0] >= max_exposure * n_port).any():
            continue
        chosen.append(int(j))
        usage += cand_bin[j]
    if len(chosen) < n_port:                       # cap too tight -- fill by EV
        for j in order:
            if len(chosen) >= n_port:
                break
            if int(j) not in chosen:
                chosen.append(int(j))
    return chosen


def solve_portfolio(cand_pay, n_port, cand_bin=None, max_exposure=None):
    """Coverage greedy: the n_port portfolio maximizing E[BEST entry payout].

    A candidate's marginal gain = mean over sims of how much it RAISES the
    portfolio's best payout that sim. That naturally diversifies -- it picks
    lineups that win in scenarios the current portfolio misses -- which is the
    right objective for a top-heavy GPP where one big hit drives ROI.
    max_exposure caps any player's share of the portfolio."""
    n_sims, n_cand = cand_pay.shape
    best = np.zeros(n_sims)
    chosen = []
    usage = np.zeros(cand_bin.shape[1]) if (cand_bin is not None and max_exposure is not None) else None
    for _ in range(n_port):
        marg = np.maximum(cand_pay - best[:, None], 0).mean(axis=0)
        if chosen:
            marg[chosen] = -1
        if usage is not None:
            over = usage >= max_exposure * n_port
            if over.any():
                marg[cand_bin[:, over].sum(axis=1) > 0] = -1
        j = int(np.argmax(marg))
        if marg[j] <= 0 and chosen:
            break
        chosen.append(j)
        best = np.maximum(best, cand_pay[:, j])
        if usage is not None:
            usage += cand_bin[j]
    return chosen


# ── 6. boards: best games, best stacks ────────────────────────────────────────

def game_board(pool: pd.DataFrame, scores: np.ndarray, field_bin: np.ndarray,
               showdown: bool = False) -> pd.DataFrame:
    """Per-game view combining Vegas, the sim, and the FIELD's attention.

    ceil85   85th-pct of the game's best-5 DFS total across sims -- how much
             fantasy the game can actually produce when it goes right
    field%   share of simulated field lineups with >=3 players in the game
    lev      ceiling rank minus field-attention rank: positive = an under-stacked
             ceiling game, which is exactly what the solver hunts
    """
    n = len(pool)
    games = pool["game_id"].to_numpy()
    fb = field_bin[:, :n] + (field_bin[:, n:] if showdown and field_bin.shape[1] == 2 * n else 0)
    rows = []
    for g in pd.unique(games):
        mask = games == g
        sub = pool[mask]
        gs = scores[:, mask]
        k = min(5, gs.shape[1])
        top5 = -np.sort(-gs, axis=1)[:, :k].sum(axis=1)
        share = float((fb[:, mask].sum(axis=1) >= 3).mean() * 100) if len(fb) else np.nan
        teams = sorted(sub["team"].unique())
        imps = [sub[sub["team"] == t]["imp"].dropna() for t in teams]
        imps = [float(x.iloc[0]) for x in imps if len(x)]
        rows.append({
            "game": g.replace("@", " vs "),
            "total": round(sum(imps), 1) if len(imps) == 2 else None,
            "imp": " / ".join(f"{t} {i:.1f}" for t, i in zip(teams, imps)) if len(imps) == 2 else "",
            "ceil85": round(float(np.percentile(top5, 85)), 1),
            "median": round(float(np.median(top5)), 1),
            "own": round(float(sub["est_ownership"].sum()), 0),
            "field%": round(share, 1),
            "n": int(mask.sum()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["lev"] = (df["ceil85"].rank(ascending=False) - df["field%"].rank(ascending=False)).astype(int) * -1
    return df.sort_values("ceil85", ascending=False).reset_index(drop=True)


def stack_board(pool: pd.DataFrame, scores: np.ndarray, max_bring: int = 3,
                top_n: int = 60) -> pd.DataFrame:
    """Rank QB stacks by simulated CEILING, with their cumulative ownership.

    Enumerates QB + 1 or 2 same-team pass-catchers, each with no bring-back or
    with one of the opponent's top pass-catchers. Scored on the SAME sim draws as
    everything else, so the ceiling numbers are comparable to the game board.

    lev = ceiling percentile - ownership percentile: the stack the field is
    under-rostering relative to what it can actually produce."""
    idx = {pid: i for i, pid in enumerate(pool["id"])}
    own = pool["est_ownership"].to_numpy(dtype=float)
    rows = []
    for _, qb in pool[pool["position"] == "QB"].iterrows():
        mates = pool[(pool["team"] == qb["team"]) & (pool["position"].isin(["WR", "TE"]))]
        mates = mates.nlargest(5, "proj")
        opp = pool[(pool["team"] == qb["opp"]) & (pool["position"].isin(["WR", "TE", "RB"]))]
        opp = opp.nlargest(max_bring, "proj")
        mi = list(mates.index)
        combos = [[a] for a in mi] + [[mi[i], mi[j]] for i in range(len(mi)) for j in range(i + 1, len(mi))]
        for c in combos:
            for bb in [None] + list(opp.index):
                members = [idx[qb["id"]]] + [idx[pool.loc[m, "id"]] for m in c]
                if bb is not None:
                    members.append(idx[pool.loc[bb, "id"]])
                tot = scores[:, members].sum(axis=1)
                names = [qb["name"]] + [pool.loc[m, "name"] for m in c]
                label = f"{qb['team']} {qb['name'].split()[-1]} + " + " + ".join(
                    n.split()[-1] for n in names[1:])
                if bb is not None:
                    label += f"  ⇄ {pool.loc[bb, 'team']} {pool.loc[bb, 'name'].split()[-1]}"
                rows.append({
                    "stack": label, "team": qb["team"], "qb": qb["name"],
                    "n": len(members), "bring": bb is not None,
                    "salary": int(sum(pool.iloc[m]["salary"] for m in members)),
                    "proj": round(float(sum(pool.iloc[m]["proj"] for m in members)), 1),
                    "ceil85": round(float(np.percentile(tot, 85)), 1),
                    "median": round(float(np.median(tot)), 1),
                    "own": round(float(own[members].sum()), 1),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["lev"] = (df["ceil85"].rank(pct=True) - df["own"].rank(pct=True)).mul(100).round(0).astype(int)
    return df.sort_values("ceil85", ascending=False).head(top_n).reset_index(drop=True)


# ── 7. end-to-end solve ───────────────────────────────────────────────────────

def solve(pool: pd.DataFrame, contest: dict, n_port: int = 20, n_sims: int = 800,
          n_cand: int = 600, field_sim: int = 1500, seed: int = 0, corr: str = "factor",
          locks: list[str] | None = None, bans: list[str] | None = None,
          max_exposure: float | None = 0.6, own_band: tuple | None = None,
          showdown: bool = False, versatile: bool = True,
          no_stack: bool = False) -> dict:
    """Full solve for ONE contest. Returns everything the UI needs:

      {pool, scores, cand_bin, metrics, chosen, field_bin, contest, n_base}

    `metrics` is one row per CANDIDATE (not just the chosen portfolio) -- that's
    the 'show me all the lineups' surface. `chosen` indexes the solved portfolio.

    objective is taken from contest['kind']: 'cash' -> no-stack floor candidates
    ranked by expected payout (maximize cashes); 'gpp' -> stacked candidates +
    coverage greedy (maximize E[best payout])."""
    pool = pool.reset_index(drop=True)
    n_base = len(pool)
    kind = contest.get("kind", "gpp")
    is_cash = kind == "cash"

    id2i = {pid: i for i, pid in enumerate(pool["id"])}
    lock_i = [id2i[p] for p in (locks or []) if p in id2i]
    ban_i = [id2i[p] for p in (bans or []) if p in id2i]

    scores = sim_scores(pool, n_sims, seed, corr=corr)
    if showdown:
        scores_full = np.concatenate([scores, scores * CPT_MULT], axis=1)
        teams = pool["team"].to_numpy()
    else:
        scores_full, teams = scores, None

    # `no_stack` is the SINGLE-ENTRY rule. Over 42 slates a max-projection build
    # returned -9% with one entry against -54% for the contrarian stacked
    # portfolio -- a 45-point gap, not noise. A portfolio exists to COVER outcomes
    # across many entries; with one entry you want the single highest-EV lineup,
    # and a forced QB stack just wrecks its floor.
    want_stack = not (is_cash or no_stack)

    # OWNERSHIP: tuned toward the measured band, never filtered to it (see tune_chalk).
    chalk, own_ach, own_note = 0.0, None, None
    if want_stack and not showdown:
        band = own_band if own_band is not None else OWN_BAND_DEFAULT
        chalk, own_ach, own_note = tune_chalk(pool, seed + 11, target=band)

    cand_bin = gen_candidates(pool, n_cand, seed + 1, stack=want_stack,
                              versatile=versatile and want_stack, showdown=showdown,
                              locks=lock_i, bans=ban_i, teams=teams, chalk=chalk)

    # FIELD REALISM: a field of purely RANDOM draws makes our candidates look
    # superhuman, because ours are the argmax of hundreds of draws while the
    # field's are not selected at all. Real opponents optimize too. So overbuild
    # the field and let the SHARP half be selection-biased toward high expected
    # score -- the same advantage we take. Without this the solver reports
    # several-hundred-percent ROI, which is pure self-reference.
    raw_field = gen_field(pool, int(field_sim * 1.8), seed + 2, showdown=showdown, teams=teams)
    if len(raw_field):
        mean_sc = (scores_full.mean(axis=0) @ raw_field.T)
        order = np.argsort(-mean_sc)
        n_sharp = int(len(raw_field) * 0.45)
        rng = np.random.default_rng(seed + 3)
        sharp = order[:n_sharp]
        rest = rng.permutation(order[n_sharp:])[:max(0, field_sim - n_sharp)]
        field_bin = raw_field[np.concatenate([sharp, rest])[:field_sim]]
    else:
        field_bin = raw_field
    if len(cand_bin) < 1 or len(field_bin) < 50:
        return {"error": f"could not build enough lineups (cands {len(cand_bin)}, "
                         f"field {len(field_bin)}) — check salaries / locks / cap"}

    band_note = own_note
    # NB guard on own_note, not just on length: in showdown `cand_bin` spans the DOUBLED
    # (FLEX|CPT) column space, so this matmul against a base-length ownership vector is a
    # dimension error. own_note is only ever set on the classic, stacked path.
    if own_note and len(cand_bin):
        cum = cand_bin @ pool["est_ownership"].to_numpy(dtype=float)
        band_note = f"{own_note}; built {cum.mean():.0f} mean / {cum.max():.0f} max"

    pbr = synth_pay_by_rank(contest["size"], contest["fee"], contest.get("prize_pool"),
                            kind=kind, cash_frac=contest.get("cash_frac", 0.20))
    cand_scores = scores_full @ cand_bin.T
    field_scores = scores_full @ field_bin.T
    cand_pay, cand_ranks = cand_payouts(cand_scores, field_scores, np.sort(pbr)[::-1],
                                        int(contest["size"]))

    # SIM SPLIT. Picking the portfolio by payout across the same draws it is then
    # scored on is hindsight selection -- the chosen lineups look brilliant because
    # they were chosen for those exact outcomes. Select on the first half of the
    # sims, report every metric on the held-out half.
    half = max(1, n_sims // 2)
    fit_pay, eval_pay = cand_pay[:half], cand_pay[half:]
    eval_ranks, eval_scores = cand_ranks[half:], cand_scores[half:]

    met = lineup_metrics(eval_pay, eval_ranks, eval_scores, cand_bin, pool,
                         float(contest["fee"]), int(contest["size"]),
                         showdown=showdown, n_base=n_base)

    if is_cash:
        # ⭐ CASH RANKS BY PROJECTION, NOT BY SIMULATED PAYOUT (measured 2026-08-12).
        # Ranking cash candidates by simulated payout lost to a plain max-projection ILP
        # over 35 real double-ups / 32 slates: McNemar p=0.031 on 0/6 discordant slates
        # (the ILP won six and lost none). The ILP was the only +EV arm there (+2.9% ROI,
        # 51.4% cash rate, above break-even in all three seasons) and it reproduces the
        # documented cash edge (p=.531 on the same 32-slate scale).
        # Cash is decided at the ~44th-percentile cash line, so the objective is simply the
        # highest expected score -- the payout curve is binary and adds only sampling noise
        # to the ranking. This aligns the Solver's cash arm with the build the +EV cash
        # result was actually measured on.
        proj_v = pool["proj"].to_numpy(dtype=float)
        chosen = list(np.argsort(-(cand_bin[:, :n_base] @ proj_v))[:n_port])
        alts = {}
    else:
        chosen = solve_portfolio(fit_pay, n_port, cand_bin=cand_bin, max_exposure=max_exposure)
        # Alternative selections from the SAME candidates and the SAME selection-half sims,
        # so an A/B between them isolates the objective and nothing else.
        alts = {
            "emax": chosen,
            "esum": solve_portfolio_sum(fit_pay, n_port),
            "esum_cap": solve_portfolio_sum(fit_pay, n_port, cand_bin=cand_bin,
                                            max_exposure=max_exposure),
        }

    return {"pool": pool, "scores": scores, "cand_bin": cand_bin, "metrics": met,
            "chosen": chosen, "alts": alts, "field_bin": field_bin, "contest": contest,
            "n_base": n_base, "showdown": showdown, "pay_by_rank": pbr,
            "band_note": band_note, "n_cand_built": len(cand_bin),
            "tier": tier_of(int(contest["size"]), kind, int(contest.get("entries", 1)),
                            showdown=showdown)}


# ── 8. what is safe to SHOW ───────────────────────────────────────────────────

def roi_is_meaningful(kind: str, showdown: bool = False) -> bool:
    """Whether the solver's absolute ROI can be shown as a number.

    SHOWDOWN: never, whatever the contest type. The cash validation below was
    measured on CLASSIC main-slate double-ups; showdown has no backtest here at
    all, and with only ~40 players in one game the simulated field is far too
    self-similar -- it reports things like a 100% cash rate, which is nonsense.

    CASH (classic): yes. The simulated cash arm independently reproduces the
    measured real-field result -- sim ROI +22% / cash rate 63% against a measured
    +23% / 62% over three seasons (cash_tally, 20 slates / 21 double-ups). Binary
    payouts depend on beating the median, which the sim gets right.

    GPP: NO. The simulated field is built from OUR projections, so it cannot
    disagree with us the way a real field does, and a top-heavy payout curve is
    convex in rank -- the small edge that gives compounds into ROI in the
    thousands of percent. The measured reality for the same builds is -12% to
    -63% by tier. The ORDERING the sim produces is still informative (every
    candidate carries the same bias), so GPP lineups are RANKED by simulated
    equity while the absolute number is withheld and TIER_BASELINE is shown
    instead."""
    return kind == "cash" and not showdown


def display_metrics(res: dict) -> pd.DataFrame:
    """Metrics table with a display policy already applied: `roi` is kept for
    cash and blanked for GPP (see roi_is_meaningful), and an `equity` column
    carries the relative ranking that IS valid in both."""
    m = res["metrics"].copy()
    m["equity"] = m["ev"].rank(pct=True).mul(100).round(0)
    if not roi_is_meaningful(res["contest"].get("kind", "gpp"), res.get("showdown", False)):
        m["roi"] = np.nan
    return m


def tier_note(res: dict) -> tuple[float | None, str]:
    """The measured real-field ROI for this contest's tier, for the UI headline."""
    t = TIER_BASELINE.get(res.get("tier", ""), {})
    return t.get("roi"), t.get("note", "")


# ── 9. presentation helpers ───────────────────────────────────────────────────

def lineup_rows(res: dict, ci: int) -> pd.DataFrame:
    """One candidate lineup as a display table, in DK slot order."""
    pool, cb, nb = res["pool"], res["cand_bin"], res["n_base"]
    idx = np.where(cb[ci] > 0)[0]
    if res["showdown"]:
        rows = []
        for j in idx:
            p = pool.iloc[j % nb].to_dict()
            cpt = j >= nb
            p["slot"] = "CPT" if cpt else "FLEX"
            p["salary"] = p["salary"] * (CPT_MULT if cpt else 1)
            p["proj"] = p["proj"] * (CPT_MULT if cpt else 1)
            p["est_ownership"] = p["est_ownership"] * (0.35 if cpt else 1)
            rows.append(p)
        rows.sort(key=lambda r: (r["slot"] != "CPT", -r["proj"]))
        return pd.DataFrame(rows)

    players = [pool.iloc[j].to_dict() for j in idx]
    rest = list(players)
    ordered = []

    def take(pos, k):
        got = sorted([p for p in rest if p["position"] == pos], key=lambda p: -p["proj"])[:k]
        for g in got:
            rest.remove(g)
        return got

    ordered += take("QB", 1) + take("RB", 2) + take("WR", 3) + take("TE", 1)
    dst = take("DST", 1)
    flex = [p for p in rest if p["position"] in FLEX_ELIGIBLE][:1]
    ordered = ordered + flex + dst
    labels = DK_COLUMNS if len(ordered) == 9 else [p["position"] for p in ordered]
    return pd.DataFrame([dict(p, slot=labels[i]) for i, p in enumerate(ordered)])


def stack_label(res: dict, ci: int) -> str:
    """Human-readable construction of a candidate ('KC QB+2 · ⇄ BUF')."""
    pool, cb, nb = res["pool"], res["cand_bin"], res["n_base"]
    idx = np.where(cb[ci] > 0)[0]
    if res["showdown"]:
        cpt = [j for j in idx if j >= nb]
        return f"CPT {pool.iloc[cpt[0] % nb]['name'].split()[-1]}" if cpt else "—"
    players = [pool.iloc[j] for j in idx]
    qb = next((p for p in players if p["position"] == "QB"), None)
    if qb is None:
        return "no QB"
    mates = sum(1 for p in players if p["team"] == qb["team"] and p["position"] in ("WR", "TE"))
    bb = sum(1 for p in players if p["team"] == qb["opp"] and p["position"] in ("WR", "TE", "RB"))
    s = f"{qb['team']} QB+{mates}"
    return s + (f" · ⇄ {qb['opp']}" if bb else "")


def exposure_table(res: dict, rows: list[int]) -> pd.DataFrame:
    """Player exposure across a set of candidate rows."""
    pool, cb, nb = res["pool"], res["cand_bin"], res["n_base"]
    if not rows:
        return pd.DataFrame()
    sub = cb[rows]
    if res["showdown"]:
        cnt = sub[:, :nb].sum(axis=0) + sub[:, nb:].sum(axis=0)
        cpt = sub[:, nb:].sum(axis=0)
    else:
        cnt = sub.sum(axis=0)
        cpt = np.zeros_like(cnt)
    out = pool.assign(lineups=cnt.astype(int), cpt=cpt.astype(int))
    out = out[out["lineups"] > 0].copy()
    out["expo%"] = (100 * out["lineups"] / len(rows)).round(0)
    cols = ["name", "position", "team", "salary", "proj", "est_ownership", "lineups", "expo%"]
    if res["showdown"]:
        cols.insert(-2, "cpt")
    return out[cols].sort_values("lineups", ascending=False)


def to_dk_rows(res: dict, rows: list[int]) -> pd.DataFrame:
    """DK upload grid (one row per lineup, names in slot order)."""
    out = []
    for ci in rows:
        t = lineup_rows(res, ci)
        if len(t) in (9, SD_SIZE):
            out.append(list(t["name"]))
    cols = SD_COLUMNS if res["showdown"] else DK_COLUMNS
    return pd.DataFrame(out, columns=cols) if out else pd.DataFrame(columns=cols)
