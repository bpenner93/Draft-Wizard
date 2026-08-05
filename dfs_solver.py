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


def sim_scores(pool: pd.DataFrame, n_sims: int, seed: int, corr: str = "factor",
               a_team: float = 0.68, a_game: float = 0.38) -> np.ndarray:
    """[n_sims, n_players] correlated DK draws: lognormal marginals (mean = proj,
    position CV) through a copula.

    corr='factor' (DEFAULT) = uniform game+team factor. It is 'wrong' on linear
    correlation yet performs BETTER on the real field, because its high same-team
    loading captures the TAIL co-movement (co-boom) GPP ceiling actually needs.
    corr='matrix' uses the measured Pearson structure (correct negative RB-RB) but
    a Gaussian copula under-models that tail. corr='tcopula' = measured structure
    PLUS tail dependence; needs scipy, and falls back to 'matrix' without it.
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
    lus = []
    for _ in range(size * 2):
        if len(lus) >= size:
            break
        sharp = len(lus) < n_sharp
        sg = rng.choice(ugames, p=gwc) if rng.random() < (0.70 if sharp else 0.55) else None
        key = (value if sharp else own) * rng.uniform(0.5 if not sharp else 0.65,
                                                      1.5 if not sharp else 1.35, size=len(pool))
        lu = _greedy(pos, sal, key, SALARY_CAP, rng, sg, games)
        if lu:
            lus.append(lu)
    return _to_binary(lus, len(pool))


def gen_candidates(pool, n, seed, stack=True, versatile=True, showdown=False,
                   locks=None, bans=None, teams=None):
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
    games = pd.factorize(pool["game_id"].astype(str))[0]
    ugames = np.unique(games)
    gw = pd.Series(proj).groupby(games).sum().reindex(ugames).to_numpy()
    gw = gw / gw.sum()                                        # shootouts get picked more
    seen, lus = set(), []
    for _ in range(n * 6):
        if len(lus) >= n:
            break
        if not stack:                                         # cash: balanced, no stack
            sg = sg2 = None
            inten = (1.0, 1.0)
            spread = 0.35
        elif versatile:                                       # GPP: roll an archetype
            roll = rng.random()
            sg2 = None
            spread = 0.55
            if roll < 0.15:
                sg, inten = None, (1.0, 1.0)                              # balanced
            elif roll < 0.55:
                sg, inten = rng.choice(ugames, p=gw), (1.4, 2.1)          # QB+2 stack
            elif roll < 0.80:
                sg, inten = rng.choice(ugames, p=gw), (2.6, 3.6)          # onslaught (4-5)
            else:
                sg, sg2 = rng.choice(ugames, size=2, replace=False, p=gw)  # double game stack
                inten = (1.6, 2.3)
        else:
            sg, sg2, inten, spread = rng.choice(ugames, p=gw), None, (1.4, 2.3), 0.55
        key = proj * rng.uniform(1 - spread, 1 + spread, size=len(pool))
        lu = _greedy(pos, sal, key, SALARY_CAP, rng, sg, games, intensity=inten,
                     second_game=sg2, must=must, banned=banned)
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
TIER_BASELINE = {
    "cash":       {"roi": +23, "note": "soft big double-ups, 3 seasons, cash rate 62% (CI straddles break-even)"},
    "se_gpp":     {"roi":  -9, "note": "single-entry GPP, 42 slates -- best GPP tier, still ~rake"},
    "gpp_milly":  {"roi": -12, "note": "Milly >=100k entries -- our contrarian build's best GPP fit"},
    "gpp_large":  {"roi": -19, "note": "30-100k entries"},
    "gpp_mid":    {"roi": -20, "note": "5-30k entries"},
    "gpp_small":  {"roi": -63, "note": "1-5k high-entry -- SHARP fields, avoid"},
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
    cand_bin = gen_candidates(pool, n_cand, seed + 1, stack=want_stack,
                              versatile=versatile and want_stack, showdown=showdown,
                              locks=lock_i, bans=ban_i, teams=teams)

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

    # OWN-BAND: left free the solver drifts over-contrarian (~95 cum own), but the
    # winning-lineup data says the optimum is NEAR-chalk (~150). Restricting to the
    # band makes it play near-chalk ON THE RIGHT (sim-optimal) game.
    band_note = None
    if own_band is not None and not showdown:
        own = pool["est_ownership"].to_numpy(dtype=float)
        cum = cand_bin @ own
        keep = (cum >= own_band[0]) & (cum <= own_band[1])
        if keep.sum() >= max(n_port, 20):
            cand_bin = cand_bin[keep]
            band_note = f"applied — {int(keep.sum())}/{len(keep)} candidates in band"
        else:
            # Do NOT fail silently. A no-op band reads as "the solver targeted the
            # winning ownership range" when it did nothing of the kind, and the
            # portfolio then drifts contrarian without saying so.
            band_note = (f"NOT applied — only {int(keep.sum())} of {len(keep)} candidates "
                         f"land in {own_band[0]}-{own_band[1]} cumulative ownership")

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
        chosen = list(np.argsort(-fit_pay.mean(axis=0))[:n_port])
    else:
        chosen = solve_portfolio(fit_pay, n_port, cand_bin=cand_bin, max_exposure=max_exposure)

    return {"pool": pool, "scores": scores, "cand_bin": cand_bin, "metrics": met,
            "chosen": chosen, "field_bin": field_bin, "contest": contest,
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
