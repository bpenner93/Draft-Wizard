"""
draft_engine.py  --  the draft-wizard brain (pure logic, no I/O, no UI)
=======================================================================
One entry point, ``analyze(board, cfg, drafted_ids)``, recomputed after every
pick. It returns everything the UI renders:

  * per-league VALUE   -- points rescored for the league, VOR over replacement
  * TIERS              -- gap-based value clusters per position (the cliffs)
  * SURVIVAL           -- Monte-Carlo P(player is still there at YOUR next pick),
                          plus expected best-available-by-position at that pick
  * RECOMMENDATION     -- draft-now-vs-wait: the player whose value you'd lose by
                          waiting (tier about to break) x your roster need
  * OPPONENT ANALYSIS  -- each team's roster shape, the needs of the teams picking
                          before you, and likely position runs

Everything is plain dicts/lists + numpy. The board is a list of player dicts:
  {id, name, pos, team, pts (PPR), rec (season receptions), games,
   adp, adp_sd, adp_sf, ecr, our_rank, clay, floor, ceil, rookie, age}
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- config
# points to SUBTRACT per reception from the full-PPR baseline (pts is already PPR):
# a PPR league subtracts nothing; standard strips the whole reception point.
SCORING_PER_REC = {"ppr": 0.0, "half": 0.5, "standard": 1.0}
FLEX_POS = ("RB", "WR", "TE")
SUPERFLEX_POS = ("QB", "RB", "WR", "TE")
# how a FLEX slot is split across RB/WR/TE when setting replacement level
FLEX_SPLIT = {"RB": 0.35, "WR": 0.50, "TE": 0.15}
SUPERFLEX_SPLIT = {"QB": 0.80, "RB": 0.05, "WR": 0.12, "TE": 0.03}

# recommendation weights (VOR-point units) -- the knobs to tune the pick analyzer
W_SCARCITY = 1.0    # weight on cost-of-waiting (tier cliff before your next pick)
W_NEED     = 1.0    # weight on roster-need bonus
KDST_LATE_ROUNDS = 3   # only value K/DST when this many rounds (or fewer) remain
RUN_STRENGTH = 0.6     # how hard a LIVE positional run bends survival off ADP (0 = pure ADP)
RUN_MIN_PICKS = 4      # need this many picks on the board before trusting a run signal


@dataclass
class LeagueConfig:
    teams: int = 12
    scoring: str = "ppr"                 # ppr | half | standard
    superflex: bool = False
    starters: dict = field(default_factory=lambda: {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1})
    bench: int = 6
    my_slot: int = 1
    snake: bool = True
    rounds: int | None = None
    archetype: str = "value"             # value | zero_rb | hero_rb | robust_rb | hero_wr | elite_te

    def roster_size(self) -> int:
        return sum(self.starters.values()) + self.bench

    def total_rounds(self) -> int:
        return self.rounds or self.roster_size()


# ----------------------------------------------------------------------------- value
def league_points(p: dict, scoring: str) -> float:
    pts = p.get("pts") or 0.0
    if p["pos"] in FLEX_POS:                      # only pass-catchers lose PPR value
        return pts - SCORING_PER_REC[scoring] * (p.get("rec") or 0.0)
    return pts


def replacement_ranks(cfg: LeagueConfig) -> dict:
    """How many players at each position get drafted as league-wide starters --
    VOR is value over the player just past that line."""
    t = cfg.teams
    base = {pos: cfg.starters.get(pos, 0) * t for pos in ("QB", "RB", "WR", "TE", "K", "DST")}
    flex = cfg.starters.get("FLEX", 0) * t
    for pos, frac in FLEX_SPLIT.items():
        base[pos] += flex * frac
    sflex = (cfg.starters.get("SUPERFLEX", 0) + (1 if cfg.superflex else 0)) * t
    for pos, frac in SUPERFLEX_SPLIT.items():
        base[pos] += sflex * frac
    return {pos: max(1, int(round(v))) for pos, v in base.items()}


def compute_values(board: list[dict], cfg: LeagueConfig) -> None:
    """Attach _pts, _vor, _ovr (overall value rank), _posrank, _tier in place."""
    for p in board:
        p["_pts"] = round(league_points(p, cfg.scoring), 1)
    repl = replacement_ranks(cfg)
    by_pos: dict[str, list] = {}
    for p in board:
        by_pos.setdefault(p["pos"], []).append(p)
    repl_pts = {}
    for pos, plist in by_pos.items():
        plist.sort(key=lambda x: -x["_pts"])
        r = repl.get(pos, len(plist))
        repl_pts[pos] = plist[r - 1]["_pts"] if len(plist) >= r else (plist[-1]["_pts"] if plist else 0.0)
    for p in board:
        p["_vor"] = round(p["_pts"] - repl_pts.get(p["pos"], 0.0), 1)
    board.sort(key=lambda x: -x["_vor"])
    for i, p in enumerate(board, 1):
        p["_ovr"] = i
    posc: Counter = Counter()
    for p in board:
        posc[p["pos"]] += 1
        p["_posrank"] = posc[p["pos"]]
    _assign_tiers(by_pos)


def _assign_tiers(by_pos: dict[str, list]) -> None:
    """Gap-based value tiers per position: a break where the VOR drop to the next
    player exceeds max(3.0, 2.2 x typical gap)."""
    for pos, plist in by_pos.items():
        plist.sort(key=lambda x: -x["_vor"])
        vals = [p["_vor"] for p in plist]
        if not vals:
            continue
        gaps = [vals[i] - vals[i + 1] for i in range(min(len(vals) - 1, 30))]
        posgaps = [g for g in gaps if g > 0]
        typ = float(np.median(posgaps)) if posgaps else 1.0
        thr = max(3.0, 2.2 * typ)
        tier = 1
        plist[0]["_tier"] = 1
        for i in range(1, len(plist)):
            if vals[i - 1] - vals[i] > thr:
                tier += 1
            plist[i]["_tier"] = tier


# ----------------------------------------------------------------------------- snake math
def team_on_clock(cfg: LeagueConfig, overall: int) -> int:
    """1-based drafting slot that owns overall pick number `overall` (1-based)."""
    rnd = (overall - 1) // cfg.teams          # 0-based round
    idx = (overall - 1) % cfg.teams           # 0-based seat in the round
    if cfg.snake and rnd % 2 == 1:
        return cfg.teams - idx
    return idx + 1


def my_pick_numbers(cfg: LeagueConfig, upto_round: int | None = None) -> list[int]:
    t, s = cfg.teams, cfg.my_slot
    R = upto_round or cfg.total_rounds()
    out = []
    for r in range(1, R + 1):
        out.append((r - 1) * t + (s if (not cfg.snake or r % 2 == 1) else (t - s + 1)))
    return out


def next_pick_for_slot(cfg: LeagueConfig, current_overall: int) -> int | None:
    for n in my_pick_numbers(cfg):
        if n >= current_overall:
            return n
    return None


# ----------------------------------------------------------------------------- survival (Monte Carlo)
def _impute_adp(p: dict, rank_rem: int, current_overall: int) -> float:
    a = p.get("_adp_eff", p.get("adp"))
    if a is not None and not (isinstance(a, float) and math.isnan(a)):
        return float(a)
    # no market signal: assume he goes around where the board values him, but late
    return float(current_overall + rank_rem)


def _impute_sd(p: dict) -> float:
    sd = p.get("adp_sd")
    if sd is not None and not (isinstance(sd, float) and math.isnan(sd)) and sd > 0:
        return max(0.5, float(sd))
    a = p.get("adp") or 60.0
    return max(8.0, 0.35 * float(a))            # unknown -> wide


def monte_carlo(remaining: list[dict], k_before: int, n_sims: int = 1200,
                seed: int = 0, adp_bias: np.ndarray | None = None):
    """Latent draft-position model: each remaining player has a draft slot
    ~Normal(adp, sd); the `k_before` earliest go before your pick. Returns
    (survival_prob[n], ebest_vor_by_pos) where ebest is the expected best-
    available VOR per position AT your pick."""
    n = len(remaining)
    if n == 0:
        return np.array([]), {}
    order_rank = {p["id"]: i for i, p in enumerate(
        sorted(remaining, key=lambda x: -x.get("_vor", 0)))}
    adp = np.array([_impute_adp(p, order_rank[p["id"]], 0) for p in remaining], dtype=float)
    sd = np.array([_impute_sd(p) for p in remaining], dtype=float)
    if adp_bias is not None:
        adp = adp + adp_bias
    if k_before <= 0:                            # you're on the clock now
        return np.ones(n), {p["pos"]: max((q["_vor"] for q in remaining if q["pos"] == p["pos"]), default=0.0)
                            for p in remaining}
    rng = np.random.default_rng(seed)
    D = rng.normal(adp[None, :], sd[None, :], size=(n_sims, n))
    order = np.argsort(D, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(n)[None, :]
    survive = ranks >= k_before                  # not among the k_before earliest
    surv_prob = survive.mean(axis=0)
    vor = np.array([p.get("_vor", 0.0) for p in remaining])
    pos = np.array([p["pos"] for p in remaining])
    ebest = {}
    for pp in set(pos.tolist()):
        m = pos == pp
        v = np.where(survive[:, m], vor[m][None, :], -1e9)
        best = v.max(axis=1)
        best[best < -1e8] = 0.0
        ebest[pp] = float(best.mean())
    return surv_prob, ebest


# ----------------------------------------------------------------------------- roster need
def need_scores(cfg: LeagueConfig, counts: dict, rounds_left: int | None = None) -> dict:
    """Positive URGENCY to fill a starting slot (missing starter = urgent; flex/
    depth = mild; filled = 0). Overfill is handled multiplicatively by
    roster_factor, not here. K/DST only matter in the last few rounds."""
    need = {}
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        starters = cfg.starters.get(pos, 0)
        if cfg.superflex and pos == "QB":
            starters += 1
        have = counts.get(pos, 0)
        if pos in ("K", "DST"):
            if have >= 1:
                need[pos] = -50.0                                   # never a 2nd
            elif rounds_left is not None and rounds_left <= KDST_LATE_ROUNDS:
                need[pos] = 6.0                                     # last few rounds: grab one
            else:
                need[pos] = -30.0                                  # far too early
            continue
        gap = starters - have
        if gap >= 1:
            need[pos] = 5.0 + 2.0 * (gap - 1)            # missing a starter -> urgent
        else:
            startable = starters + (cfg.starters.get("FLEX", 0) if pos in FLEX_POS else 0)
            need[pos] = 2.0 if have < startable else 0.0  # flex/depth still worth a nudge
    return need


def roster_factor(cfg: LeagueConfig, pos: str, have: int) -> float:
    """Multiplier on a player's VOR reflecting roster reality: once you can't
    START (or reasonably bench) more of a position, extra copies are ~worthless.
    Stops the board from stacking 5 TEs just because TE VOR stays positive deep."""
    base = cfg.starters.get(pos, 0) + (1 if (cfg.superflex and pos == "QB") else 0)
    flex = cfg.starters.get("FLEX", 0)
    if pos in ("RB", "WR"):
        startable = base + flex
        return 1.0 if have < startable else (0.6 if have < startable + 3 else 0.2)
    if pos in ("QB", "TE"):
        return 1.0 if have < base else (0.5 if have < base + 1 else 0.03)
    if pos in ("K", "DST"):
        return 0.05        # a kicker/DST "VOR" isn't real value; the late need-gate handles timing
    return 1.0


# ----------------------------------------------------------------------------- strategy archetypes
ARCHETYPES = ("value", "zero_rb", "hero_rb", "robust_rb", "hero_wr", "elite_te")
ARCHETYPE_LABEL = {
    "value": "Best value (reactive)", "zero_rb": "Zero-RB", "hero_rb": "Hero-RB",
    "robust_rb": "Robust-RB", "hero_wr": "Hero-WR", "elite_te": "Elite-TE",
}


def archetype_bonus(name: str, pos: str, rnd: int, counts: dict, cfg: LeagueConfig) -> float:
    """A per-pick positional nudge (VOR-point units) layered on top of pure value
    when the user commits to a structural plan. `value` is 0 (fully reactive)."""
    if not name or name == "value":
        return 0.0
    if name == "zero_rb":
        if pos == "RB":
            return -30.0 if rnd <= 4 else (8.0 if rnd >= 6 else -8.0)
        if pos in ("WR", "TE"):
            return 6.0 if rnd <= 5 else 0.0
    elif name == "hero_rb":
        if pos == "RB":
            if counts.get("RB", 0) == 0 and rnd <= 2:
                return 8.0                       # lock ONE anchor RB, then pivot off RB
            return -20.0 if rnd <= 5 else 6.0
        if pos in ("WR", "TE") and 2 <= rnd <= 6:
            return 5.0
    elif name == "robust_rb":
        if pos == "RB":
            return 9.0 if rnd <= 3 else 0.0
    elif name == "hero_wr":
        if pos == "WR":
            if counts.get("WR", 0) == 0 and rnd <= 2:
                return 8.0
            return -18.0 if rnd <= 5 else 5.0
        if pos in ("RB", "TE") and 2 <= rnd <= 6:
            return 5.0
    elif name == "elite_te":
        if pos == "TE" and counts.get("TE", 0) == 0 and rnd <= 4:
            return 14.0
    return 0.0


# ----------------------------------------------------------------------------- opponent analysis
def opponent_analysis(cfg: LeagueConfig, picks: list[dict], current_overall: int, my_next: int) -> dict:
    """picks: [{overall, slot, pos, id, name}]. Roster shape per team + the needs
    of the teams on the clock before your pick + likely position runs."""
    counts = {s: Counter() for s in range(1, cfg.teams + 1)}
    for d in picks:
        counts[d["slot"]][d["pos"]] += 1
    upcoming = [team_on_clock(cfg, o) for o in range(current_overall, my_next or current_overall)]
    seat_need = []
    for s in upcoming:
        ns = need_scores(cfg, counts[s])
        toppos = max(("QB", "RB", "WR", "TE"), key=lambda p: ns.get(p, 0))
        seat_need.append({"slot": s, "top_need": toppos})
    demand = Counter(x["top_need"] for x in seat_need)
    return {"team_counts": {s: dict(c) for s, c in counts.items()},
            "upcoming_slots": upcoming, "seat_need": seat_need, "demand": dict(demand)}


# ----------------------------------------------------------------------------- live draft flow
def run_signal(picks: list[dict], board: list[dict], current_overall: int, window: int):
    """Detect a LIVE positional run: how much each position's recent pick share
    exceeds its ADP-neighborhood baseline. Positive = running hot (its players
    should survive LESS than ADP implies); negative = being skipped (value slides
    to you). This is what makes the tool react to THIS room, not the average draft."""
    core = ("QB", "RB", "WR", "TE")
    zero = {p: 0.0 for p in core}
    if window <= 0 or not picks:
        return dict(zero), dict(zero), dict(zero)
    recent = picks[-window:]
    ac = Counter(p["pos"] for p in recent if p["pos"] in core)
    na = sum(ac.values()) or 1
    act = {p: ac.get(p, 0) / na for p in core}
    lo, hi = current_overall - window, current_overall + window
    neigh = [p for p in board if p.get("adp") and lo <= p["adp"] <= hi and p["pos"] in core]
    nb = len(neigh) or 1
    bc = Counter(p["pos"] for p in neigh)
    base = {p: bc.get(p, 0) / nb for p in core}
    pressure = {p: round(act[p] - base[p], 3) for p in core}
    return pressure, act, base


def _strategy_summary(pressure: dict, act_share: dict, remaining: list[dict]) -> dict:
    core = ("QB", "RB", "WR", "TE")
    hot = sorted([p for p in core if pressure.get(p, 0) >= 0.15], key=lambda p: -pressure[p])
    cold = sorted([p for p in core if pressure.get(p, 0) <= -0.12], key=lambda p: pressure[p])
    bits = []
    for pos in hot:
        bits.append(f"{pos} run in progress ({act_share[pos]*100:.0f}% of recent picks) -- "
                    f"grab the one you want or it's gone")
    for pos in cold:
        topv = max((p["_vor"] for p in remaining if p["pos"] == pos), default=0.0)
        if topv > 0:
            bits.append(f"{pos} value sliding to you (best {pos} VOR {topv:.0f} still on the board)")
    return {"hot": hot, "cold": cold, "pressure": pressure, "note": "; ".join(bits)}


# ----------------------------------------------------------------------------- façade
def analyze(board_in: list[dict], cfg: LeagueConfig, drafted_ids: list[str],
            n_sims: int = 1200) -> dict:
    """Recompute the whole wizard state. `drafted_ids` is the ordered list of
    player ids taken so far (overall pick order). Returns a render-ready dict."""
    board = [dict(p) for p in board_in]                 # copy; we annotate
    by_id = {p["id"]: p for p in board}
    compute_values(board, cfg)

    drafted_set = set(drafted_ids)
    picks = []
    my_roster_ids = []
    for i, pid in enumerate(drafted_ids):
        overall = i + 1
        slot = team_on_clock(cfg, overall)
        p = by_id.get(pid)
        if p is None:
            continue
        p["_drafted"] = True
        p["_slot"] = slot
        picks.append({"overall": overall, "slot": slot, "pos": p["pos"], "id": pid, "name": p.get("name")})
        if slot == cfg.my_slot:
            my_roster_ids.append(pid)

    current_overall = len(drafted_ids) + 1
    my_next = next_pick_for_slot(cfg, current_overall)
    picks_until = (my_next - current_overall) if my_next else 0
    my_counts = Counter(by_id[i]["pos"] for i in my_roster_ids if i in by_id)
    rounds_left = cfg.total_rounds() - len(my_roster_ids)

    remaining = [p for p in board if not p.get("_drafted")]
    for p in remaining:                              # superflex leagues use SF ADP for timing
        sf = p.get("adp_sf")
        p["_adp_eff"] = sf if (cfg.superflex and sf) else p.get("adp")
    # live positional run -> bend survival off ADP toward what THIS room is doing
    pressure, act_share, _base = run_signal(picks, board, current_overall,
                                            window=min(len(picks), cfg.teams))
    if len(picks) >= RUN_MIN_PICKS and picks_until > 0:
        adp_bias = np.clip(
            np.array([-RUN_STRENGTH * pressure.get(p["pos"], 0.0) * picks_until for p in remaining]),
            -12.0, 12.0)
    else:
        adp_bias = None
    surv, ebest = monte_carlo(remaining, k_before=picks_until, n_sims=n_sims, adp_bias=adp_bias)
    surv_by_id = {p["id"]: float(s) for p, s in zip(remaining, surv)}

    # best-now VOR per position + cost of waiting
    bestnow: dict[str, float] = {}
    for p in remaining:
        if p["_vor"] > bestnow.get(p["pos"], -1e9):
            bestnow[p["pos"]] = p["_vor"]
    cow = {pos: round(bestnow[pos] - ebest.get(pos, 0.0), 1) for pos in bestnow}
    need = need_scores(cfg, my_counts, rounds_left)
    my_round = (((my_next or current_overall) - 1) // cfg.teams) + 1   # round you're drafting for

    for p in remaining:
        pos = p["pos"]
        p["_surv"] = round(surv_by_id.get(p["id"], 1.0), 3)
        p["_cow"] = cow.get(pos, 0.0)
        p["_need"] = round(need.get(pos, 0.0), 1)
        rf = roster_factor(cfg, pos, my_counts.get(pos, 0))
        ab = archetype_bonus(cfg.archetype, pos, my_round, my_counts, cfg)
        p["_rec_score"] = round(
            p["_vor"] * rf + W_SCARCITY * max(0.0, cow.get(pos, 0.0))
            + W_NEED * need.get(pos, 0.0) + ab, 1)

    remaining.sort(key=lambda x: -x["_rec_score"])
    best_available = sorted(remaining, key=lambda x: -x["_vor"])

    opp = opponent_analysis(cfg, picks, current_overall, my_next)

    # tier context for the recommendation's position
    def tier_left(pos, tier):
        return sum(1 for p in remaining if p["pos"] == pos and p.get("_tier") == tier)

    rec = remaining[0] if remaining else None
    reco = None
    if rec:
        reco = {
            "id": rec["id"], "name": rec["name"], "pos": rec["pos"], "team": rec.get("team"),
            "posrank": rec["_posrank"], "vor": rec["_vor"], "tier": rec.get("_tier"),
            "survival": rec["_surv"], "cost_of_waiting": rec["_cow"], "need": rec["_need"],
            "tier_left": tier_left(rec["pos"], rec.get("_tier")),
            "reason": _reason(rec, cow, need, opp["demand"]),
        }

    return {
        "current_overall": current_overall,
        "round": (current_overall - 1) // cfg.teams + 1,
        "on_clock": team_on_clock(cfg, current_overall),
        "my_next_pick": my_next,
        "picks_until_mine": picks_until,
        "my_roster": [_slim(by_id[i]) for i in my_roster_ids if i in by_id],
        "my_counts": dict(my_counts),
        "needs": need,
        "recommendation": reco,
        "alternates": [_slim(p, full=True) for p in remaining[1:6]],
        "best_available": [_slim(p, full=True) for p in best_available[:60]],
        "cost_of_waiting": cow,
        "ebest_next": {k: round(v, 1) for k, v in ebest.items()},
        "opponents": opp,
        "strategy": _strategy_summary(pressure, act_share, remaining),
        "board_size_remaining": len(remaining),
    }


# ----------------------------------------------------------------------------- forward planner
POS_ORDER = ("QB", "RB", "WR", "TE", "K", "DST")


def _bonus6(cfg: LeagueConfig, counts: dict, rnd: int, rounds_left: int) -> np.ndarray:
    need = need_scores(cfg, counts, rounds_left)
    return np.array([need.get(pp, 0.0) + archetype_bonus(cfg.archetype, pp, rnd, counts, cfg)
                     for pp in POS_ORDER])


def plan_draft(board_in: list[dict], cfg: LeagueConfig, drafted_ids: list[str],
               n_sims: int = 150, seed: int = 0) -> dict:
    """Simulate the REST of your draft many times (opponents pick by latent ADP,
    you pick by your value+archetype policy) and report the likely build path at
    each of your future picks + how each position's best-available value declines
    (the windows / dead zones)."""
    board = [dict(p) for p in board_in]
    by_id = {p["id"]: p for p in board}
    compute_values(board, cfg)
    current_overall = len(drafted_ids) + 1
    my_counts0: Counter = Counter()
    for i, pid in enumerate(drafted_ids):
        if team_on_clock(cfg, i + 1) == cfg.my_slot and pid in by_id:
            my_counts0[by_id[pid]["pos"]] += 1
    total = cfg.teams * cfg.total_rounds()
    my_future = [n for n in my_pick_numbers(cfg) if current_overall <= n <= total]
    if not my_future:
        return {"picks": [], "windows": {}, "note": "Draft complete.", "my_future_picks": []}

    remaining = [p for p in board if p["id"] not in set(drafted_ids)]
    for p in remaining:
        sf = p.get("adp_sf")
        p["_adp_eff"] = sf if (cfg.superflex and sf) else p.get("adp")
    n = len(remaining)
    order_rank = {p["id"]: i for i, p in enumerate(sorted(remaining, key=lambda x: -x["_vor"]))}
    adp = np.array([_impute_adp(p, order_rank[p["id"]], current_overall) for p in remaining])
    sd = np.array([_impute_sd(p) for p in remaining])
    vor = np.array([p["_vor"] for p in remaining])
    names = [p.get("name") for p in remaining]
    pcode = np.array([POS_ORDER.index(p["pos"]) if p["pos"] in POS_ORDER else 0 for p in remaining])
    rng = np.random.default_rng(seed)

    K = len(my_future)
    cpos = [Counter() for _ in range(K)]
    cname = [Counter() for _ in range(K)]
    svor = np.zeros(K)
    bypos = np.zeros((K, len(POS_ORDER)))
    my_total0 = sum(my_counts0.values())

    for _ in range(n_sims):
        slots = rng.normal(adp, sd)
        taken = np.zeros(n, dtype=bool)
        counts = Counter(my_counts0)
        for k, pickno in enumerate(my_future):
            rnd = ((pickno - 1) // cfg.teams) + 1
            rounds_left = cfg.total_rounds() - (my_total0 + k)
            avail = (slots >= pickno) & (~taken)
            if not avail.any():
                continue
            b6 = _bonus6(cfg, counts, rnd, rounds_left)
            rf6 = np.array([roster_factor(cfg, pp, counts.get(pp, 0)) for pp in POS_ORDER])
            masked = np.where(avail, vor * rf6[pcode] + b6[pcode], -1e18)
            j = int(np.argmax(masked))
            code = int(pcode[j])
            cpos[k][POS_ORDER[code]] += 1
            cname[k][names[j]] += 1
            svor[k] += vor[j]
            taken[j] = True
            counts[POS_ORDER[code]] += 1
            for c in range(len(POS_ORDER)):
                m = avail & (pcode == c)
                if m.any():
                    bypos[k][c] += float(vor[m].max())

    picks = []
    for k, pickno in enumerate(my_future):
        tot = sum(cpos[k].values()) or 1
        picks.append({
            "pick_no": pickno, "round": ((pickno - 1) // cfg.teams) + 1,
            "top_pos": (cpos[k].most_common(1)[0][0] if cpos[k] else None),
            "pos_probs": {p: round(cpos[k][p] / tot, 2) for p in cpos[k]},
            "exp_vor": round(svor[k] / n_sims, 1),
            "examples": [nm for nm, _ in cname[k].most_common(3)],
            "by_pos_vor": {POS_ORDER[c]: round(bypos[k][c] / n_sims, 1) for c in range(len(POS_ORDER))},
        })
    windows = {pp: [(picks[k]["pick_no"], picks[k]["by_pos_vor"][pp]) for k in range(K)]
               for pp in POS_ORDER if pp not in ("K", "DST")}
    path = " -> ".join(p["top_pos"] for p in picks[:8] if p["top_pos"])
    return {"picks": picks, "windows": windows, "my_future_picks": my_future,
            "note": (f"Likely build: {path}" if path else "")}


# ----------------------------------------------------------------------------- practice / mock draft
OPP_CAP = {"QB": 3, "RB": 8, "WR": 9, "TE": 3, "K": 1, "DST": 1}   # most an AI team drafts at a pos


def prep_valued(board_in: list[dict], cfg: LeagueConfig) -> list[dict]:
    """A per-league VALUED copy of the board (VOR/tiers + effective ADP) for the
    mock-draft AI to read once."""
    b = [dict(p) for p in board_in]
    compute_values(b, cfg)
    for p in b:
        sf = p.get("adp_sf")
        p["_adp_eff"] = sf if (cfg.superflex and sf) else p.get("adp")
    return b


def _needs_starter(cfg: LeagueConfig, pos: str, counts: dict) -> bool:
    starters = cfg.starters.get(pos, 0) + (1 if (cfg.superflex and pos == "QB") else 0)
    return counts.get(pos, 0) < starters


def _team_counts(cfg: LeagueConfig, drafted_ids: list[str], by_id: dict, slot: int) -> Counter:
    c: Counter = Counter()
    for i, pid in enumerate(drafted_ids):
        if team_on_clock(cfg, i + 1) == slot and pid in by_id:
            c[by_id[pid]["pos"]] += 1
    return c


def opponent_pick(remaining: list[dict], cfg: LeagueConfig, counts: dict,
                  rng, rounds_left: int = 99) -> dict:
    """One AI opponent pick: draft by ADP (the market) with noise, fill starting
    needs first, respect position caps, and hold K/DST until the last two rounds."""
    best, bkey = None, 1e18
    for p in remaining:
        pos = p["pos"]
        if counts.get(pos, 0) >= OPP_CAP.get(pos, 8):
            continue
        if pos in ("K", "DST") and rounds_left > 2:
            continue
        adp = p.get("_adp_eff")
        if adp is None or (isinstance(adp, float) and math.isnan(adp)):
            adp = p.get("adp")
        if adp is None or (isinstance(adp, float) and math.isnan(adp)):
            adp = float(p.get("_ovr", 200))                 # no market signal -> value rank
        sd = p.get("adp_sd")
        sd = float(sd) if (sd and not (isinstance(sd, float) and math.isnan(sd))) else max(6.0, 0.30 * float(adp))
        key = rng.normal(float(adp), sd)
        if _needs_starter(cfg, pos, counts):
            key -= 6.0                                      # draft a needed starter ~a round earlier
        if key < bkey:
            bkey, best = key, p
    if best is None:                                         # everyone capped -> best available by adp
        best = min(remaining, key=lambda x: (x.get("adp") or x.get("_ovr", 999)))
    return best


def mock_advance(valued_board: list[dict], cfg: LeagueConfig, drafted_ids: list[str], rng) -> list[str]:
    """Auto-draft for every team that isn't YOU until it's your pick (or the draft
    ends). Returns the extended drafted-id list. Pass a board from prep_valued()."""
    by_id = {p["id"]: p for p in valued_board}
    drafted = list(drafted_ids)
    total = cfg.teams * cfg.total_rounds()
    while len(drafted) < total:
        overall = len(drafted) + 1
        if team_on_clock(cfg, overall) == cfg.my_slot:
            break
        slot = team_on_clock(cfg, overall)
        counts = _team_counts(cfg, drafted, by_id, slot)
        rl = cfg.total_rounds() - sum(counts.values())
        taken = set(drafted)
        rem = [p for p in valued_board if p["id"] not in taken]
        if not rem:
            break
        drafted.append(opponent_pick(rem, cfg, counts, rng, rl)["id"])
    return drafted


def _letter(pct: float) -> str:
    """pct: 1.0 = best team in the league, 0.0 = worst."""
    for thr, g in ((0.92, "A+"), (0.80, "A"), (0.68, "B+"), (0.55, "B"),
                   (0.42, "C+"), (0.28, "C"), (0.15, "D"), (0.0, "F")):
        if pct >= thr:
            return g
    return "F"


def grade_draft(cfg: LeagueConfig, drafted_ids: list[str], board_in: list[dict]) -> dict:
    """Grade YOUR roster vs the rest of the league: total value banked (sum of each
    player's positive VOR) ranked against the other teams -> percentile -> letter."""
    board = [dict(p) for p in board_in]
    compute_values(board, cfg)
    by_id = {p["id"]: p for p in board}
    val = {s: 0.0 for s in range(1, cfg.teams + 1)}
    cnt = {s: Counter() for s in range(1, cfg.teams + 1)}
    best = {s: None for s in range(1, cfg.teams + 1)}
    for i, pid in enumerate(drafted_ids):
        p = by_id.get(pid)
        if not p:
            continue
        s = team_on_clock(cfg, i + 1)
        v = max(0.0, p.get("_vor", 0.0))
        val[s] += v
        cnt[s][p["pos"]] += 1
        if best[s] is None or v > best[s][1]:
            best[s] = (p.get("name"), v, p["pos"])
    my = cfg.my_slot
    order = sorted(val, key=lambda s: -val[s])
    my_rank = order.index(my) + 1
    pct = 1.0 - (my_rank - 1) / max(1, cfg.teams - 1)
    return {
        "grade": _letter(pct), "my_rank": my_rank, "teams": cfg.teams,
        "pctile": round(pct * 100), "my_value": round(val[my], 1),
        "league_avg": round(sum(val.values()) / cfg.teams, 1),
        "my_counts": dict(cnt[my]), "best_pick": best[my],
    }


def _reason(p: dict, cow: dict, need: dict, demand: dict) -> str:
    bits = []
    if p["_surv"] < 0.40:
        bits.append(f"won't last -- {p['_surv']*100:.0f}% to survive to your next pick")
    c = cow.get(p["pos"], 0.0)
    if c >= 6:
        bits.append(f"tier cliff: ~{c:.0f} VOR lost if you wait")
    if need.get(p["pos"], 0) >= 5:
        bits.append(f"fills a starting {p['pos']} hole")
    d = demand.get(p["pos"], 0)
    if d >= 2:
        bits.append(f"run risk: {d} of the picks ahead lean {p['pos']}")
    return "; ".join(bits) or "best value on the board"


def _slim(p: dict, full: bool = False) -> dict:
    out = {"id": p["id"], "name": p.get("name"), "pos": p["pos"], "team": p.get("team"),
           "posrank": p.get("_posrank"), "vor": p.get("_vor"), "pts": p.get("_pts"),
           "tier": p.get("_tier"), "adp": p.get("adp"), "rookie": p.get("rookie")}
    if full:
        out.update({"survival": p.get("_surv"), "cost_of_waiting": p.get("_cow"),
                    "rec_score": p.get("_rec_score"), "ecr": p.get("ecr"), "clay": p.get("clay"),
                    "need": p.get("_need")})
    return out


# ----------------------------------------------------------------------------- board loader
def load_board(path: str | Path = None) -> list[dict]:
    path = Path(path or (Path(__file__).parent / "data" / "draft_board.json"))
    with open(path, encoding="utf-8") as f:
        return json.load(f)["players"]


# ----------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    board = load_board()
    cfg = LeagueConfig(teams=12, scoring="ppr", my_slot=7)
    print(f"board: {len(board)} players; league: {cfg.teams}-team {cfg.scoring}, slot {cfg.my_slot}\n")

    # simulate the first 15 picks going strictly by ADP
    ranked = sorted([p for p in board if p.get("adp")], key=lambda x: x["adp"])
    drafted = [p["id"] for p in ranked[:15]]

    res = analyze(board, cfg, drafted)
    print(f"pick {res['current_overall']} (round {res['round']}), you're seat {cfg.my_slot}; "
          f"your next pick {res['my_next_pick']} ({res['picks_until_mine']} away)\n")
    r = res["recommendation"]
    print(f"RECOMMEND: {r['name']} {r['pos']}{r['posrank']} ({r['team']})  VOR {r['vor']}  "
          f"tier {r['pos']}T{r['tier']} ({r['tier_left']} left)  surv {r['survival']*100:.0f}%")
    print(f"  why: {r['reason']}\n")
    print("  alternates:")
    for a in res["alternates"]:
        print(f"    {a['name']:22} {a['pos']}{a['posrank']:<2} VOR {a['vor']:5.1f}  "
              f"surv {a['survival']*100:3.0f}%  cow {a['cost_of_waiting']:4.1f}  rec {a['rec_score']:5.1f}")
    print(f"\n  upcoming demand before your pick: {res['opponents']['demand']}")
    print(f"  cost of waiting by pos: {res['cost_of_waiting']}")
