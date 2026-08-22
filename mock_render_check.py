"""
mock_render_check.py -- run a full mock and assert on what the UI would RENDER.
==============================================================================
Every defect found on 2026-08-21 was a DISPLAY defect that the engine tests and
AppTest both passed: a chip labelled "Jr. · WR", ADP printing the literal string
"None", a constant Surv% column painted solid dark red, "ON THE CLOCK" on a
finished draft, "SEA DST" abbreviated to "S. DST".

None of those are visible from `analyze()`'s return value. They only exist once
the values are turned into a label, a Styler, or an HTML panel. So this walks a
real 15-round mock and, at every one of your picks, builds the SAME artifacts
draft_app builds and asserts on the rendered strings.

    python draft_wizard/mock_render_check.py
    python draft_wizard/mock_render_check.py --seeds 5

Exit 0 = green.
"""
from __future__ import annotations

import argparse
import io
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import draft_board as B          # noqa: E402
import draft_engine as E         # noqa: E402

FAILS: list[str] = []


def bad(msg):
    FAILS.append(msg)
    print(f"  ✗ {msg}")


def ok(msg):
    print(f"  ✓ {msg}")


# `surname` now lives in draft_names, where both the app and draft_review import
# it from -- it had been re-derived (wrongly) twice before it was shared.
from draft_names import surname          # noqa: E402


# draft_app is a Streamlit script and cannot be imported, so lift the remaining
# pure display helper out of it by source. If it is renamed this raises, which is
# the correct outcome -- a silently-skipped check is worse than a broken one.
# (It just raised, on exactly that: `surname` moved out of draft_app.)
def _from_app(name):
    src = (HERE / "draft_app.py").read_text(encoding="utf-8")
    seg = src[src.index("def arc_flag"):src.index("def build_rookie_board")]
    ns: dict = {"pd": pd}
    exec(seg, ns)
    return ns[name]


arc_flag = _from_app("arc_flag")


# --------------------------------------------------------------------------- the board table
def build_table(res, cur, bb_mode=False):
    """Byte-for-byte the row dicts + Styler that draft_app builds."""
    rows = []
    for p in res["ranked"][:60]:
        _adp = p.get("adp")
        rows.append({
            "Player": p["name"], "Pos": f"{p['pos']}{p['posrank']}", "Tm": p.get("team"),
            "Bye": p.get("bye"), "Score": p.get("rec_score"), "VOR": p["vor"],
            "Tier": f"{p['pos']}T{p['tier']}" if p.get("tier") else "",
            "Surv%": round((p.get("survival") or 1) * 100),
            "ADP": _adp, "vsADP": (None if _adp is None else round(cur - _adp)),
            "CoW": p.get("cost_of_waiting"), "Arc": arc_flag(p.get("arc")),
            "Decl%": p.get("decl"),
        })
    df = pd.DataFrame(rows)
    for c in ("Bye", "Score", "VOR", "Surv%", "ADP", "vsADP", "CoW", "Decl%"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    fmt = {"Score": "{:.0f}", "VOR": "{:.0f}", "ADP": "{:.0f}", "CoW": "{:.0f}",
           "Surv%": "{:.0f}", "Decl%": "{:.0f}", "vsADP": "{:+.0f}", "Bye": "{:.0f}"}

    def grad_ok(col):
        if col not in df.columns:
            return False
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        return len(v) > 1 and v.nunique() > 1

    sty = df.style
    for col, cmap in (("Score", "Greens"), ("Surv%", "RdYlGn"), ("vsADP", "PuOr")):
        if grad_ok(col):
            sty = sty.background_gradient(subset=[col], cmap=cmap)
    sty = sty.format({k: v for k, v in fmt.items() if k in df.columns}, na_rep="—")
    return df, sty


# --------------------------------------------------------------------------- run
def run(seed: int, cfg: E.LeagueConfig, board: list[dict]) -> dict:
    by_id = {p["id"]: p for p in board}
    rng = np.random.default_rng(seed)
    valued = E.prep_valued(board, cfg)
    drafted = E.mock_advance(valued, cfg, [], rng)
    total = cfg.teams * cfg.total_rounds()
    seen = {"chips": 0, "tables": 0, "grids": 0, "suffix_names": 0, "dst_cells": 0,
            "no_adp_rows": 0, "const_surv": 0}
    guard = 0
    while len(drafted) < total and guard < total + 10:
        res = E.analyze(board, cfg, drafted, n_sims=250)
        if res["complete"] or not res["recommendation"]:
            break
        cur = res["current_overall"]

        # ---- 1. quick-pick chip labels (the "Jr. · WR" bug)
        for a in res["ranked"][:5]:
            lab = f"{surname(a['name'])} · {a['pos']}"
            seen["chips"] += 1
            head = lab.split(" · ")[0]
            if head.lower().strip(".") in ("jr", "sr", "ii", "iii", "iv", "v"):
                bad(f"seed {seed} pick {cur}: chip label is a SUFFIX -- {a['name']!r} -> {lab!r}")
            if re.search(r"\b(Jr|Sr|II|III|IV)\.?$", a["name"]):
                seen["suffix_names"] += 1
            if a["pos"] == "DST" and head == "DST":
                bad(f"seed {seed} pick {cur}: DST chip lost its team -- {a['name']!r} -> {lab!r}")

        # ---- 2/3. the rendered table (literal "None", constant-column gradient)
        df, sty = build_table(res, cur)
        html = sty.to_html()
        seen["tables"] += 1
        if df["ADP"].isna().any():
            seen["no_adp_rows"] += 1
        for cell in re.findall(r"<td[^>]*>([^<]*)</td>", html):
            if cell.strip() in ("None", "nan", "NaN", "<NA>"):
                bad(f"seed {seed} pick {cur}: table cell rendered {cell.strip()!r} "
                    f"instead of an em-dash")
                break
        if df["Surv%"].dropna().nunique() == 1 and "background-color" in html:
            sub = html[html.index("Surv%"):] if "Surv%" in html else ""
            if "background-color" in sub:
                seen["const_surv"] += 1

        # ---- 5. the board grid (DST abbreviation, and it must show real picks)
        grid = B.draft_board_html(cfg, None, drafted, by_id, rounds_window=7)
        seen["grids"] += 1
        if re.search(r"\b[A-Z]\. (DST|DEF)\b", grid):
            bad(f"seed {seed} pick {cur}: a defense is abbreviated in the grid "
                f"({re.search(r'[A-Z]. (DST|DEF)', grid).group(0)!r})")
        if "DST" in grid:
            seen["dst_cells"] += 1

        rec = res["recommendation"]
        seen.setdefault("first_by_pos", {}).setdefault(
            rec["pos"], B.pick_label(cfg, cur))
        seen.setdefault("picks", []).append(
            (B.pick_label(cfg, cur), rec["name"], rec["pos"]))
        drafted.append(rec["id"])
        guard += 1
        drafted = E.mock_advance(valued, cfg, drafted, rng)

    # ---- 4. terminal state: nothing may claim to be on the clock
    fin = E.analyze(board, cfg, drafted)
    if not fin["complete"]:
        bad(f"seed {seed}: mock ended at {len(drafted)}/{total}, not complete")
    else:
        # draft_app only renders on_deck when `not done`; assert the panel itself
        # would also be harmless if it ever were rendered
        od = B.on_deck_html(cfg, None, fin["current_overall"], fin["my_next_pick"],
                            fin["opponents"].get("seat_need"), complete=fin["complete"])
        if "ON THE CLOCK" in od:
            bad(f"seed {seed}: on_deck_html still says ON THE CLOCK on a finished draft")
        df, sty = build_table(fin, fin["current_overall"])
        h = sty.to_html()
        if df["Surv%"].dropna().nunique() == 1:
            # a constant column must NOT be gradient-painted
            body = h[h.index("<tbody"):] if "<tbody" in h else h
            if "background-color" in body and df["Score"].nunique() == 1:
                bad(f"seed {seed}: constant column still painted post-draft")
        for cell in re.findall(r"<td[^>]*>([^<]*)</td>", h):
            if cell.strip() in ("None", "nan", "NaN"):
                bad(f"seed {seed}: post-draft table rendered {cell.strip()!r}")
                break
        # the final board's roster must be legal
        shape = {}
        for i, pid in enumerate(drafted):
            if E.team_on_clock(cfg, i + 1) == cfg.my_slot and pid in by_id:
                shape[by_id[pid]["pos"]] = shape.get(by_id[pid]["pos"], 0) + 1
        for pos, want in cfg.starters.items():
            if pos in ("FLEX", "SUPERFLEX") or not want:
                continue
            if shape.get(pos, 0) < want:
                bad(f"seed {seed}: illegal roster, {pos} {shape.get(pos,0)} < {want}")
        seen["shape"] = " ".join(f"{k}{v}" for k, v in sorted(shape.items()))
        seen["grade"] = E.grade_draft(cfg, drafted, board)["grade"]
    return seen


# ⚠ Do NOT rely on the random walk to hit these. Whether a Jr. lands in a given
# mock's top-5 chips is luck -- seed 1 produces none, seeds 2-3 produce seven --
# so the edge cases are asserted DIRECTLY here and the walk is used for breadth.
# A check whose condition never fires is worth nothing, and one that fires only
# sometimes is worse: it goes green for the wrong reason.
NAME_CASES = [
    # (raw name,              chip label,   grid label)
    ("Marvin Harrison Jr.",   "Harrison",   "M. Harrison"),
    ("Brian Thomas Jr.",      "Thomas",     "B. Thomas"),
    ("Travis Etienne Jr.",    "Etienne",    "T. Etienne"),
    ("Kenneth Walker III",    "Walker",     "K. Walker"),
    ("Odell Beckham Jr",      "Beckham",    "O. Beckham"),
    ("Amon-Ra St. Brown",     "Brown",      "A. St. Brown"),
    ("Ja'Marr Chase",         "Chase",      "J. Chase"),
    ("Bijan Robinson",        "Robinson",   "B. Robinson"),
    ("SEA DST",               "SEA DST",    "SEA DST"),
    ("LA DST",                "LA DST",     "LA DST"),
]


def check_names():
    print("  name rendering (deterministic):")
    bad_n = 0
    for raw, want_chip, want_grid in NAME_CASES:
        got_chip, got_grid = surname(raw), B._short(raw, 14)
        if got_chip != want_chip:
            bad(f"chip label for {raw!r}: got {got_chip!r}, want {want_chip!r}"); bad_n += 1
        if got_grid != want_grid:
            bad(f"grid label for {raw!r}: got {got_grid!r}, want {want_grid!r}"); bad_n += 1
    if not bad_n:
        ok(f"all {len(NAME_CASES)} name cases render correctly "
           f"(suffixes stripped, defenses kept whole)")


def live_cfg(league_id: str) -> E.LeagueConfig:
    """Build the config from the LIVE league, not from a constant.

    ⭐ This is the whole lesson of 2026-08-21: a hardcoded 12T/15rd/1QB here would
    keep passing after a commissioner edits the league, which is exactly how the
    Superflex drift survived for two weeks. Rehearse against what the league says
    TODAY."""
    from sleeper_api import league_state
    s = league_state(league_id=league_id)
    rc = s["roster_cfg"]
    print(f"  live: {(s['league_name'] or '').strip()} · {s['type']} · {s['teams']}T · "
          f"{s['rounds']} rds · slot {s['my_slot']} · "
          f"{'SUPERFLEX' if rc['superflex'] else '1QB'} {rc['scoring'].upper()} · "
          f"{len(s['picks'])} picks made")
    return E.LeagueConfig(
        teams=int(s["teams"]), scoring=rc["scoring"], superflex=rc["superflex"],
        my_slot=int(s["my_slot"] or 1), snake=bool(s["snake"]),
        rounds=int(s["rounds"]), bench=int(rc["bench"]), starters=dict(rc["starters"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--league", help="Sleeper league_id — read the settings LIVE "
                                     "instead of using the built-in default")
    a = ap.parse_args()

    board = E.load_board()
    if a.league:
        cfg = live_cfg(a.league)
    else:
        cfg = E.LeagueConfig(
            teams=12, scoring="ppr", superflex=False, my_slot=10, snake=True, rounds=15,
            bench=6, starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2,
                               "SUPERFLEX": 0, "K": 0, "DST": 1})
    print(f"=== FULL MOCKS, RENDERING THE UI AT EVERY PICK "
          f"({cfg.teams}T {cfg.scoring.upper()} slot {cfg.my_slot}, "
          f"{cfg.total_rounds()} rds) ===")
    check_names()
    tot = {}
    for s in range(1, a.seeds + 1):
        r = run(s, cfg, board)
        print(f"  seed {s}: {r['chips']} chip labels · {r['tables']} rendered tables "
              f"· {r['grids']} grids · roster {r.get('shape','?')} "
              f"· grade {r.get('grade','?')}")
        if r.get("first_by_pos"):
            print(f"          first at each position: "
                  + "  ".join(f"{k}@{v}" for k, v in r["first_by_pos"].items()))
        if s == 1 and r.get("picks"):
            print("          your board: "
                  + " → ".join(f"{lab} {nm.split()[-1]}({pos})"
                               for lab, nm, pos in r["picks"][:8]) + " …")
        for k, v in r.items():
            if isinstance(v, int):
                tot[k] = tot.get(k, 0) + v

    print()
    print(f"  suffix names (Jr./III) that passed through a chip: {tot.get('suffix_names', 0)}")
    print(f"  tables containing a player with NO ADP:            {tot.get('no_adp_rows', 0)}")
    print(f"  grids containing a drafted defense:                {tot.get('dst_cells', 0)}")
    # coverage NOTES, not failures: whether a suffixed name reaches the top-5
    # chips in a given mock is luck (0 at seed 1, 7 across seeds 1-3). The
    # deterministic NAME_CASES above are what actually guard it.
    if not tot.get("suffix_names"):
        print("      (no suffixed name reached a chip in this sample — covered by "
              "NAME_CASES above)")
    if not tot.get("no_adp_rows"):
        bad("no table ever contained a player without an ADP -- the 'None' check "
            "never exercised its condition")

    print()
    if FAILS:
        print(f"  ✗ {len(FAILS)} FAILURE(S)")
        sys.exit(1)
    print(f"  ✓ ALL GREEN across {a.seeds} full mocks")


if __name__ == "__main__":
    main()
