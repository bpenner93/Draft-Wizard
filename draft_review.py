"""
draft_review.py -- post-mortem for a finished mock: a detailed grade, and the
picks where you actually left value on the board.
======================================================================

⭐ THE COUNTERFACTUAL IS EXACT, NOT ESTIMATED. "You should have taken Judkins
first" is only true if the player you *did* take would still have been there next
turn, and normally that is unknowable -- you removed him from the pool, so the
rest of the draft is not the draft that would have happened. Because the practice
room is seeded PER PICK (see mock_advance), we can replay it: put Y in your seat
at that pick instead of X, run the same opponents forward with the same seed, and
look at whether X survived. So every number here is observed in a real
alternative draft, not inferred from a survival model.

⚠ That is also the limit of it. It answers "against THIS room, what did that pick
cost?" It is not a claim about drafting in general, and it is hindsight -- the
alternative is only visible after you know who went where. Read it as a drill,
not a scorecard.
"""
from __future__ import annotations

from collections import defaultdict

from draft_engine import (LeagueConfig, compute_values, mock_advance, my_pick_numbers,
                          prep_valued, team_on_clock)
from draft_names import surname

# Only report a miss worth acting on. Below this the "better" pick is inside the
# noise of the projections themselves, and a review that flags twelve things
# teaches nothing.
MIN_GAIN = 12.0
MAX_MISSES = 6


def _vor(p):
    return float(p.get("_vor") or 0.0)


def _usable(p):
    """VOR floored at zero. ⭐ Below replacement a player adds NOTHING to a lineup,
    so the difference between -30 and -10 is 0, not 20. Without this floor the
    review rated swapping two last-round bench bodies as a "+40 miss" and told you
    to take a -10 VOR tight end. Same convention grade_draft already scores on."""
    return max(0.0, _vor(p))


def review(board_in: list[dict], cfg: LeagueConfig, drafted_ids: list[str],
           seed: int | None, min_gain: float = MIN_GAIN) -> dict:
    """Full post-draft review. `seed` is the practice room's seed; without it the
    counterfactual cannot be replayed and only the grade half is returned."""
    board = [dict(p) for p in board_in]
    compute_values(board, cfg)
    by_id = {p["id"]: p for p in board}
    valued = prep_valued(board_in, cfg)

    mine = [o for o in my_pick_numbers(cfg) if o <= len(drafted_ids)]
    my_ids = [drafted_ids[o - 1] for o in mine]
    roster = [by_id[i] for i in my_ids if i in by_id]

    # ---- grade detail -------------------------------------------------------
    teams = {}
    for s in range(1, cfg.teams + 1):
        ids = [drafted_ids[i] for i in range(len(drafted_ids))
               if team_on_clock(cfg, i + 1) == s]
        teams[s] = [by_id[i] for i in ids if i in by_id]

    def total(ps):
        return sum(max(0.0, _vor(p)) for p in ps)

    scores = {s: total(ps) for s, ps in teams.items()}
    mine_score = scores.get(cfg.my_slot, 0.0)
    rank = 1 + sum(1 for s, v in scores.items() if v > mine_score)
    league_avg = sum(scores.values()) / max(1, len(scores))

    # per-position, against the league's average at that position
    pos_rows = []
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        mine_p = [p for p in roster if p["pos"] == pos]
        if not mine_p and not cfg.starters.get(pos):
            continue
        others = [total([q for q in ps if q["pos"] == pos])
                  for s, ps in teams.items() if s != cfg.my_slot]
        avg = sum(others) / max(1, len(others))
        pos_rows.append({
            "pos": pos, "n": len(mine_p), "value": round(total(mine_p)),
            "league_avg": round(avg), "edge": round(total(mine_p) - avg),
            "best": max(mine_p, key=_vor)["name"] if mine_p else None,
        })

    # value vs the market: did you reach, or did the room let players fall?
    vs_adp = []
    for o, pid in zip(mine, my_ids):
        p = by_id.get(pid)
        if p and p.get("adp"):
            vs_adp.append(o - float(p["adp"]))
    reach = round(sum(vs_adp) / len(vs_adp), 1) if vs_adp else None

    out = {
        "grade_value": round(mine_score), "rank": rank, "teams": cfg.teams,
        "league_avg": round(league_avg), "edge": round(mine_score - league_avg),
        "positions": pos_rows, "avg_vs_adp": reach,
        "roster": [{"pick": o, "name": by_id[i]["name"], "pos": by_id[i]["pos"],
                    "vor": round(_vor(by_id[i]))}
                   for o, i in zip(mine, my_ids) if i in by_id],
        "misses": [], "replayable": seed is not None,
    }
    if seed is None:
        return out

    # ---- the misses ---------------------------------------------------------
    taken_at = {pid: i + 1 for i, pid in enumerate(drafted_ids)}
    misses = []
    for k, o in enumerate(mine[:-1]):
        nxt = mine[k + 1]
        x = by_id.get(drafted_ids[o - 1])
        z = by_id.get(drafted_ids[nxt - 1])
        if not x or not z:
            continue
        pool_ids = set(drafted_ids[: o - 1])
        # candidates: available when you picked, gone before your next turn, and
        # worth more than what you actually ended up with at that next turn
        cands = [p for p in board
                 if p["id"] not in pool_ids
                 and o < taken_at.get(p["id"], 10 ** 9) < nxt
                 and _usable(p) > _usable(z) and _vor(p) > 0]
        if not cands:
            continue
        y = max(cands, key=_usable)

        # ⭐ replay the room with Y in your seat instead of X, same seed
        alt = list(drafted_ids[: o - 1]) + [y["id"]]
        alt = mock_advance(valued, cfg, alt, None, seed=seed)
        gone = set(alt)
        x_survived = x["id"] not in gone
        if x_survived:
            got_back, gain = x, _usable(y) - _usable(z)
        else:
            left = [p for p in board if p["id"] not in gone]
            if not left:
                continue
            got_back = max(left, key=_usable)
            gain = (_usable(y) + _usable(got_back)) - (_usable(x) + _usable(z))
        if gain < min_gain:
            continue
        misses.append({
            "pick": o, "next_pick": nxt,
            "took": x["name"], "took_pos": x["pos"], "took_vor": round(_vor(x)),
            "instead": y["name"], "instead_pos": y["pos"], "instead_vor": round(_vor(y)),
            "instead_went": taken_at.get(y["id"]),
            "got_at_next": z["name"], "got_at_next_vor": round(_vor(z)),
            "would_have_got": got_back["name"], "would_have_vor": round(_vor(got_back)),
            "still_there": bool(x_survived), "gain": round(gain),
        })
    misses.sort(key=lambda m: -m["gain"])
    out["misses"] = misses[:MAX_MISSES]
    out["total_left"] = round(sum(m["gain"] for m in misses))
    return out


def miss_sentence(m: dict) -> str:
    """One line a human can act on."""
    if m["still_there"]:
        return (f"**{m['instead']}** ({m['instead_pos']}, {m['instead_vor']} VOR) went at "
                f"pick {m['instead_went']} — before your next turn. You took "
                f"**{m['took']}** at {m['pick']}, and he was **still on the board** at "
                f"{m['next_pick']}. Taking {surname(m['instead'])} first and "
                f"{surname(m['took'])} second was worth **+{m['gain']}** "
                f"(you got {m['got_at_next']}, {m['got_at_next_vor']} VOR, instead).")
    return (f"**{m['instead']}** ({m['instead_pos']}, {m['instead_vor']} VOR) went at pick "
            f"{m['instead_went']}. Taking him at {m['pick']} over **{m['took']}** "
            f"({m['took_vor']}) and falling back to {m['would_have_got']} "
            f"({m['would_have_vor']}) at {m['next_pick']} was worth **+{m['gain']}**.")


def headline(rv: dict) -> str:
    """The one thing to tell them above the detail."""
    if not rv.get("misses"):
        return ("No pick cost you more than "
                f"{int(MIN_GAIN)} VOR against this room — the board and your picks agreed.")
    m = rv["misses"][0]
    return (f"Biggest miss: pick **{m['pick']}** — {m['instead']} over {m['took']}, "
            f"**+{m['gain']} VOR**. {len(rv['misses'])} pick(s) cost "
            f"{rv.get('total_left', 0)} in total.")


def position_note(rv: dict) -> str | None:
    """The structural read: where you beat the room and where you lost it."""
    rows = [r for r in rv.get("positions", []) if r["pos"] not in ("K", "DST")]
    if not rows:
        return None
    best = max(rows, key=lambda r: r["edge"])
    worst = min(rows, key=lambda r: r["edge"])
    if best["edge"] <= 0 and worst["edge"] >= 0:
        return None
    bits = []
    if best["edge"] > 0:
        bits.append(f"You beat the room at **{best['pos']}** (+{best['edge']} vs their average)")
    if worst["edge"] < 0:
        bits.append(f"lost it at **{worst['pos']}** ({worst['edge']})")
    return " and ".join(bits) + "."
