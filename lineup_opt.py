"""
lineup_opt.py -- compact DK-classic lineup builder for the app (self-contained,
no pipeline imports; pure python so it runs on Streamlit Cloud and in tests).

Mirrors the VALIDATED live GPP construction (optimizer.py --engine profile /
gpp_builder.PROFILE, see [[field-gap-vs-top50]]):
  - stacks ONLY in high-total games:  QB + 1 or 2 same-team pass-catchers
    (stack2 ~40% -> ~1.4 avg), opponent bring-back ~55%
  - max_per_game cap so the roster spreads across games
  - band-targeted de-chalk: eff = ceiling - 0.10 * max(0, own - 8)
  - spend the cap (min salary ~49K), scores on CEILING for GPP
Cash mode = max(0.65*proj + 0.35*floor), no stack forcing, no de-chalk.

This is the phone/interactive tool; the heavy engines (sim solver, full profile
portfolio) still live in the private pipeline's optimizer.py.
"""
from __future__ import annotations

import random

SALARY_CAP = 50_000
ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}   # + 1 FLEX (RB/WR/TE)
FLEX_ELIGIBLE = ("RB", "WR", "TE")
DK_COLUMNS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]

GPP = dict(frac_stack2=0.40, frac_bring=0.55, max_per_game=4, max_te=2,
           own_lambda=0.10, own_thresh=8.0, min_salary=49_000,
           total_floor=44.0, jitter=0.06)
CASH = dict(frac_stack2=0.0, frac_bring=0.0, max_per_game=9, max_te=2,
            own_lambda=0.0, own_thresh=100.0, min_salary=48_000,
            total_floor=0.0, jitter=0.02)


def _eff(p: dict, mode: str, prm: dict) -> float:
    """Per-player effective score. GPP tilts to ceiling with the de-chalk hinge;
    cash blends proj+floor."""
    if mode == "cash":
        return 0.65 * (p.get("proj") or 0) + 0.35 * (p.get("floor") or 0)
    ceil = p.get("ceil") or (p.get("proj") or 0) * 1.4
    own = p.get("own") or 0.0
    return ceil - prm["own_lambda"] * max(0.0, own - prm["own_thresh"])


def usable(players: list[dict]) -> tuple[list[dict], int]:
    """Rows the builder can use (salary + projection present); returns (pool,
    n_dropped) so the page can surface what was excluded."""
    ok = [p for p in players
          if (p.get("salary") or 0) > 0 and (p.get("proj") or 0) > 0
          and p.get("pos") in ("QB", "RB", "WR", "TE", "DST")]
    return ok, len(players) - len(ok)


def _game_key(p: dict) -> str:
    a, b = str(p.get("team")), str(p.get("opp"))
    return "@".join(sorted([a, b]))


def _count_game(lineup: list[dict], gk: str) -> int:
    return sum(1 for q in lineup if _game_key(q) == gk)


def _n_te(lineup: list[dict]) -> int:
    return sum(1 for q in lineup if q["pos"] == "TE")


def _slots_open(lineup: list[dict]) -> dict:
    """Remaining roster needs. NEGATIVE values = overflow (invalid): only
    RB/WR/TE overflow may spill into FLEX -- a 2nd QB/DST goes negative on its
    own key so validate_lineup catches it (a max(0,..) here once let the swap
    pass sneak a 2nd QB into FLEX)."""
    have = {}
    for p in lineup:
        have[p["pos"]] = have.get(p["pos"], 0) + 1
    need, flex_used = {}, 0
    for k, v in ROSTER.items():
        h = have.get(k, 0)
        if k in FLEX_ELIGIBLE:
            flex_used += max(0, h - v)
            need[k] = max(0, v - h)
        else:
            need[k] = v - h                    # negative = too many QB/DST
    need["FLEX"] = 1 - flex_used               # negative = too many flex extras
    # anything not in ROSTER at all (shouldn't happen) counts as overflow
    for k, h in have.items():
        if k not in ROSTER:
            need[k] = -h
    return need


def _min_fill_cost(pool: list[dict], need: dict) -> int:
    """Cheapest possible way to fill the remaining slots (budget feasibility)."""
    cost = 0
    mins = {}
    for pos in ("QB", "RB", "WR", "TE", "DST"):
        sals = sorted(p["salary"] for p in pool if p["pos"] == pos)
        mins[pos] = sals
        cost += sum(sals[:need.get(pos, 0)]) if sals else 10**9 * need.get(pos, 0)
    if need.get("FLEX"):
        fx = sorted(p["salary"] for p in pool if p["pos"] in FLEX_ELIGIBLE)
        cost += fx[0] if fx else 10**9
    return cost


def validate_lineup(lineup: list[dict], partial: bool = False) -> list[str]:
    """DK-classic validity errors; partial=True skips 'incomplete' checks
    (for the manual builder as you go)."""
    errs = []
    ids = [p["id"] for p in lineup]
    if len(set(ids)) != len(ids):
        errs.append("duplicate player")
    sal = sum(p.get("salary") or 0 for p in lineup)
    if sal > SALARY_CAP:
        errs.append(f"over the cap: ${sal:,} > ${SALARY_CAP:,}")
    need = _slots_open(lineup)
    over = {k: -v for k, v in need.items() if v < 0}
    if over:
        errs.append(f"too many at {over}")
    if not partial:
        if len(lineup) != 9:
            errs.append(f"{len(lineup)}/9 players")
        elif any(v > 0 for v in need.values()):
            errs.append(f"missing: " + ", ".join(k for k, v in need.items() if v > 0))
    teams = {p.get("team") for p in lineup if p["pos"] != "DST"}
    if not partial and len(teams) < 2:
        errs.append("players from 2+ teams required")
    return errs


def _pick(cands: list[dict], rng: random.Random, jitter: float, top_k: int = 6):
    """Weighted-random among the top candidates (diversity source)."""
    if not cands:
        return None
    scored = sorted(cands, key=lambda p: -(p["_eff"] * (1 + rng.uniform(-jitter, jitter))))
    return scored[0] if top_k <= 1 else rng.choice(scored[:min(top_k, len(scored))])


def build_lineup(pool: list[dict], mode: str = "gpp", locks: list[str] | None = None,
                 params: dict | None = None, rng: random.Random | None = None,
                 stack_team: str | None = None) -> list[dict] | None:
    """One lineup. Returns list of 9 player dicts or None if infeasible."""
    rng = rng or random.Random()
    prm = dict(GPP if mode == "gpp" else CASH)
    prm.update(params or {})
    pool = [dict(p, _eff=_eff(p, mode, prm)) for p in pool]
    by_id = {p["id"]: p for p in pool}

    lineup = []
    for pid in (locks or []):
        p = by_id.get(pid)
        if p and p["id"] not in {q["id"] for q in lineup}:
            lineup.append(p)
    if validate_lineup(lineup, partial=True):
        return None

    # ── GPP stack skeleton (skip pieces already locked) ───────────────────────
    protected = set(q["id"] for q in lineup)          # locks are never swapped out
    if mode == "gpp" and stack_team is not None and not any(q["pos"] == "QB" for q in lineup):
        qbs = [p for p in pool if p["pos"] == "QB" and p["team"] == stack_team]
        qb = max(qbs, key=lambda p: p["_eff"]) if qbs else None
        if qb:
            lineup.append(qb)
            protected.add(qb["id"])
            n_stack = 2 if rng.random() < prm["frac_stack2"] else 1
            mates = [p for p in pool if p["team"] == stack_team and p["pos"] in ("WR", "TE")
                     and p["id"] not in {q["id"] for q in lineup}]
            for p in sorted(mates, key=lambda x: -x["_eff"])[:n_stack]:
                if p["pos"] == "TE" and _n_te(lineup) >= prm["max_te"]:
                    continue
                if _slots_open(lineup).get(p["pos"], 0) + _slots_open(lineup).get("FLEX", 0) > 0:
                    lineup.append(p)
                    protected.add(p["id"])
            if rng.random() < prm["frac_bring"]:
                bb = [p for p in pool if p["team"] == qb.get("opp") and p["pos"] in ("WR", "TE")
                      and p["id"] not in {q["id"] for q in lineup}]
                bb = [p for p in bb if not (p["pos"] == "TE" and _n_te(lineup) >= prm["max_te"])]
                p = _pick(bb, rng, prm["jitter"], top_k=3)
                if p and (_slots_open(lineup).get(p["pos"], 0) + _slots_open(lineup).get("FLEX", 0)) > 0:
                    lineup.append(p)
                    protected.add(p["id"])

    # ── greedy fill ───────────────────────────────────────────────────────────
    guard = 0
    while len(lineup) < 9 and guard < 60:
        guard += 1
        need = _slots_open(lineup)
        pos_next = next((k for k in ("QB", "DST", "RB", "WR", "TE") if need.get(k, 0) > 0), None)
        elig_pos = (pos_next,) if pos_next else FLEX_ELIGIBLE
        used = {q["id"] for q in lineup}
        budget = SALARY_CAP - sum(q["salary"] for q in lineup)
        cands = []
        for p in pool:
            if p["id"] in used or p["pos"] not in elig_pos or p["salary"] > budget:
                continue
            if p["pos"] == "TE" and _n_te(lineup) >= prm["max_te"]:
                continue
            if mode == "gpp" and p["pos"] != "DST" and _count_game(lineup, _game_key(p)) >= prm["max_per_game"]:
                continue
            # DST shouldn't face our QB's offense
            qb = next((q for q in lineup if q["pos"] == "QB"), None)
            if p["pos"] == "DST" and qb and p["team"] in (qb.get("team"), qb.get("opp")):
                continue
            # feasibility: after taking p, the rest must still be fillable
            rest_pool = [x for x in pool if x["id"] not in used and x["id"] != p["id"]]
            trial = lineup + [p]
            if _min_fill_cost(rest_pool, _slots_open(trial)) > SALARY_CAP - sum(q["salary"] for q in trial):
                continue
            cands.append(p)
        p = _pick(cands, rng, prm["jitter"])
        if p is None:
            return None
        lineup.append(p)

    if len(lineup) != 9 or validate_lineup(lineup):
        return None

    # ── improvement swaps: raise eff, keep constraints, spend toward the cap.
    # Stack pieces + locks are PROTECTED -- an unconstrained swap pass dismantles
    # the QB stack for higher-eff one-offs, which defeats the whole construction.
    for _ in range(2):
        for i, cur in enumerate(list(lineup)):
            if cur["id"] in protected:
                continue
            used = {q["id"] for q in lineup}
            others = [q for j, q in enumerate(lineup) if j != i]
            budget = SALARY_CAP - sum(q["salary"] for q in others)
            best, best_eff = None, cur["_eff"]
            for p in pool:
                if p["id"] in used or p["salary"] > budget or p["_eff"] <= best_eff:
                    continue
                trial = others + [p]
                if validate_lineup(trial):
                    continue
                if mode == "gpp" and p["pos"] != "DST" and _count_game(others, _game_key(p)) >= prm["max_per_game"]:
                    continue
                qb = next((q for q in trial if q["pos"] == "QB"), None)
                dst = next((q for q in trial if q["pos"] == "DST"), None)
                if qb and dst and dst["team"] in (qb.get("team"), qb.get("opp")):
                    continue
                best, best_eff = p, p["_eff"]
            if best is not None:
                lineup[i] = best

    if sum(q["salary"] for q in lineup) < prm["min_salary"]:
        # under-spent: try upgrading the cheapest slot once more with eff/$ ignored
        pass                                    # acceptable -- min_salary is a soft target here
    return lineup


def stack_order(pool: list[dict], prm: dict) -> list[str]:
    """QB teams ordered by GAME total (team implied + opponent implied,
    shootouts first), gated at total_floor when totals are known -- the profile
    builder's high-total targeting."""
    imp, opp_of = {}, {}
    for p in pool:
        t = p.get("team")
        if t and p.get("imp") is not None:
            imp[t] = float(p["imp"])
        if t and p.get("opp"):
            opp_of[t] = p["opp"]
    qb_teams = sorted({p["team"] for p in pool if p["pos"] == "QB" and p.get("team")})

    def game_total(t):
        return (imp.get(t) or 0.0) + (imp.get(opp_of.get(t)) or 0.0)

    ordered = sorted(qb_teams, key=lambda t: -game_total(t))
    if imp:                                      # gate only when totals exist
        gated = [t for t in ordered if game_total(t) >= prm.get("total_floor", 0.0)]
        if gated:
            return gated
    return ordered


def build_portfolio(players: list[dict], n: int = 5, mode: str = "gpp",
                    locks: list[str] | None = None, bans: list[str] | None = None,
                    max_exposure: float = 0.6, min_diff: int = 2,
                    params: dict | None = None, seed: int | None = None) -> list[dict]:
    """n lineups with exposure caps + roster-overlap diversity.
    Returns [{players, salary, proj, own, eff, stack}]."""
    rng = random.Random(seed)
    pool, _ = usable(players)
    bans = set(bans or [])
    pool = [p for p in pool if p["id"] not in bans]
    locks = [l for l in (locks or []) if any(p["id"] == l for p in pool)]
    prm = dict(GPP if mode == "gpp" else CASH)
    prm.update(params or {})

    order = stack_order(pool, prm) if mode == "gpp" else [None]
    counts: dict[str, int] = {}
    out, prior_sets = [], []
    qi, tries = 0, 0
    while len(out) < n and tries < n * 12:
        tries += 1
        # exposure: hide over-exposed players (locks exempt)
        cap = max(1, int(max_exposure * n)) if mode == "gpp" else n
        vis = [p for p in pool if counts.get(p["id"], 0) < cap or p["id"] in locks]
        stack_team = order[qi % len(order)] if order and mode == "gpp" else None
        qi += 1
        lu = build_lineup(vis, mode=mode, locks=locks, params=params, rng=rng,
                          stack_team=stack_team)
        if lu is None:
            continue
        ids = {p["id"] for p in lu}
        if any(len(ids & s) > 9 - min_diff for s in prior_sets):
            continue
        prior_sets.append(ids)
        for pid in ids:
            counts[pid] = counts.get(pid, 0) + 1
        qb = next((p for p in lu if p["pos"] == "QB"), None)
        n_mates = sum(1 for p in lu if qb and p["team"] == qb["team"] and p["pos"] in ("WR", "TE"))
        n_bb = sum(1 for p in lu if qb and p["team"] == qb.get("opp") and p["pos"] in ("WR", "TE"))
        out.append({
            "players": sort_slots(lu),
            "salary": sum(p["salary"] for p in lu),
            "proj": round(sum(p.get("proj") or 0 for p in lu), 1),
            "own": round(sum(p.get("own") or 0 for p in lu), 1),
            "stack": (f"{qb['team']} QB+{n_mates}" + (f" · BB {qb.get('opp')}" if n_bb else "")) if qb else "—",
        })
    if mode == "cash":                      # cash users play lineup #1 -- best first
        out.sort(key=lambda lu: -lu["proj"])
    return out


def sort_slots(lineup: list[dict]) -> list[dict]:
    """Order players into DK slots: QB, RB, RB, WR, WR, WR, TE, FLEX, DST.
    FLEX = the lowest-salary extra among RB/WR/TE beyond base counts."""
    rest = list(lineup)
    ordered = []

    def take(pos, k):
        nonlocal rest
        got = sorted([p for p in rest if p["pos"] == pos],
                     key=lambda p: -(p.get("proj") or 0))[:k]
        for g in got:
            rest.remove(g)
        return got

    qb = take("QB", 1); rb = take("RB", 2); wr = take("WR", 3); te = take("TE", 1)
    dst = take("DST", 1)
    flex = [p for p in rest if p["pos"] in FLEX_ELIGIBLE][:1]
    ordered = qb + rb + wr + te + flex + dst
    # slot labels for display
    labels = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    return [dict(p, slot=labels[i]) for i, p in enumerate(ordered)] if len(ordered) == 9 else \
           [dict(p, slot=p["pos"]) for p in lineup]


def to_dk_rows(portfolio: list[dict]) -> list[list[str]]:
    """DK upload-style grid, one name-row per lineup in DK_COLUMNS order --
    sort_slots already emits QB,RB,RB,WR,WR,WR,TE,FLEX,DST. Build the CSV with
    e.g. pd.DataFrame(rows, columns=DK_COLUMNS) (same header the pipeline's
    export_dk_csv writes)."""
    rows = []
    for lu in portfolio:
        names = [p["name"] for p in lu["players"]]
        if len(names) == 9:
            rows.append(names)
    return rows
