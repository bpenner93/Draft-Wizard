"""
mock_history.py -- a record of every practice mock you finish.
==============================================================

⚠ PERSISTENCE IS HONEST-BUT-LIMITED, AND THE UI SAYS SO. Streamlit Community
Cloud gives each run an EPHEMERAL filesystem: a write succeeds, and then vanishes
on the next restart or redeploy, and is not shared between devices. Silently
"saving" to it would be the worst kind of failure -- a history that looks kept and
is not. So: the session always holds the list, a disk write is attempted (which
genuinely persists when you run locally), and the real durable path is the
explicit Download / Restore pair, which works identically everywhere.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

STORE = Path(__file__).parent / "data" / "mock_history.json"
MAX_KEEP = 200


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def record(cfg, seed, grade_letter, rv, mode="Redraft") -> dict:
    """One row from a finished mock. Deliberately flat + small: this is a log you
    scan, and a fat record makes the download unwieldy for no gain."""
    top = rv["misses"][0] if rv.get("misses") else None
    shape: dict[str, int] = {}
    for r in rv.get("roster", []):
        shape[r["pos"]] = shape.get(r["pos"], 0) + 1
    return {
        "when": _now(), "mode": mode, "room": int(seed) if seed is not None else None,
        "teams": cfg.teams, "scoring": cfg.scoring, "slot": cfg.my_slot,
        "rounds": cfg.total_rounds(), "superflex": bool(cfg.superflex),
        "grade": grade_letter, "value": rv.get("grade_value"),
        "rank": rv.get("rank"), "edge": rv.get("edge"),
        "avg_vs_adp": rv.get("avg_vs_adp"),
        "shape": " ".join(f"{k}{v}" for k, v in sorted(shape.items())),
        "pos_edge": {r["pos"]: r["edge"] for r in rv.get("positions", [])},
        "n_misses": len(rv.get("misses", [])), "left_on_board": rv.get("total_left", 0),
        "top_miss": (f"{top['instead']} over {top['took']} at {top['pick']} "
                     f"(+{top['gain']})" if top else None),
    }


def load_disk() -> list[dict]:
    try:
        return json.loads(STORE.read_text(encoding="utf-8")).get("mocks", [])
    except Exception:
        return []


def save_disk(rows: list[dict]) -> bool:
    """Best effort. Returns whether it actually landed -- the caller must not
    claim the history is saved on the strength of having called this."""
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps({"mocks": rows[-MAX_KEEP:]}, indent=1),
                         encoding="utf-8")
        return True
    except Exception:
        return False


def merge(a: list[dict], b: list[dict]) -> list[dict]:
    """Union by (when, room), newest last. Restoring a download must not duplicate
    the rows already in the session."""
    seen, out = set(), []
    for r in list(a) + list(b):
        k = (r.get("when"), r.get("room"), r.get("slot"))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    out.sort(key=lambda r: r.get("when") or "")
    return out[-MAX_KEEP:]


def summarise(rows: list[dict]) -> dict:
    """Trends across mocks -- the reason to keep a history at all."""
    if not rows:
        return {}
    n = len(rows)
    fin = [r["rank"] for r in rows if r.get("rank")]
    left = [r.get("left_on_board") or 0 for r in rows]
    pe: dict[str, list] = {}
    for r in rows:
        for k, v in (r.get("pos_edge") or {}).items():
            pe.setdefault(k, []).append(v)
    return {
        "n": n,
        "avg_finish": round(sum(fin) / len(fin), 1) if fin else None,
        "avg_left": round(sum(left) / n),
        "pos_edge": {k: round(sum(v) / len(v)) for k, v in pe.items()
                     if k not in ("K", "DST")},
    }
