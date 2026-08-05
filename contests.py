"""
contests.py -- the contest menu the solver solves AGAINST.

Two sources:
  * LIVE DraftKings lobby -- draftkings.com/lobby/getcontests is public and
    keyless (the same feed the pipeline's dk_lobby.py uses), so the app can pull
    the real menu for the current slate: entry fee, field cap, prize pool, max
    entries. That makes "highest ROI for this tournament type" a question about
    contests that actually exist rather than a stylized guess.
  * PRESETS -- one per tier, used offline / out of season / if the lobby call
    fails. Sized to the middle of each tier.

Requests only (already an app dependency); every call degrades to presets.
"""
from __future__ import annotations

import pandas as pd
import requests

LOBBY = "https://www.draftkings.com/lobby/getcontests?sport={sport}"
_H = {"User-Agent": "Mozilla/5.0 (draft-wizard solver)", "Accept": "application/json"}

# DoubleUp pays ~top 44% (2x), FiftyFifty ~top 50%. cash_frac = paying fraction.
CASH_FRAC = {"du": 0.44, "5050": 0.50}


def fetch_dk_lobby(sport: str = "NFL", timeout: int = 20) -> pd.DataFrame:
    """Live DK contest menu. Returns [] on any failure -- never raises into the UI."""
    try:
        r = requests.get(LOBBY.format(sport=sport), headers=_H, timeout=timeout)
        r.raise_for_status()
        cons = r.json().get("Contests", []) or []
    except Exception:
        return pd.DataFrame()

    rows = []
    for c in cons:
        attr = c.get("attr") or {}
        fee = float(c.get("a") or 0)
        size = float(c.get("m") or 0)
        if fee <= 0 or size <= 0:
            continue
        name = str(c.get("n", ""))
        prize = float(c.get("po") or 0)
        is_du = attr.get("IsDoubleUp") == "true" or "Double Up" in name
        is_5050 = attr.get("IsFiftyfifty") == "true" or "50/50" in name or "50-50" in name
        if attr.get("IsQualifier") == "true" or attr.get("IsWinnerTakeAll") == "true":
            continue                                   # tickets / WTA: not a $ EV play
        if is_du or is_5050:
            kind = "cash"
            cash_frac = CASH_FRAC["5050"] if is_5050 else CASH_FRAC["du"]
        elif attr.get("Multiplier") == "true":
            continue                                   # top-heavy "cash", skip
        else:
            kind, cash_frac = "gpp", 0.20
        rake = 1 - prize / (size * fee) if size * fee else None
        rows.append({
            "contest_id": c.get("id"), "name": name, "kind": kind,
            "fee": fee, "size": int(size), "prize_pool": prize,
            "entries": int(c.get("mec") or 1), "filled": int(c.get("nt") or 0),
            "cash_frac": cash_frac,
            "rake%": round(100 * rake, 1) if rake is not None else None,
            "overlay$": round(prize - size * fee, 0),
            "start": c.get("sd"), "draft_group": c.get("dg"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["kind", "size"], ascending=[True, False]).reset_index(drop=True)


def label(row: dict) -> str:
    """Selectbox label. NOT markdown (Streamlit selectbox options render as plain
    text), so the $ signs are safe here -- but anything that ends up in st.markdown
    must escape them or "$5 ... $1M" is parsed as LaTeX."""
    o = row.get("overlay$") or 0
    bits = [f"${row['fee']:,.0f}", f"{row['size']:,} entries"]
    if row.get("rake%") is not None:
        bits.append(f"rake {row['rake%']:.0f}%")
    if o > 0:
        bits.append(f"⚠ OVERLAY ${o:,.0f}")
    return f"{row['name']}  ·  " + " · ".join(bits)


def preset_frame() -> pd.DataFrame:
    """Offline contest menu -- one representative contest per measured tier."""
    from dfs_solver import PRESETS
    rows = []
    for p in PRESETS:
        rows.append({"contest_id": p["key"], "name": p["label"], "kind": p["kind"],
                     "fee": p["fee"], "size": p["size"], "prize_pool": None,
                     "entries": p["entries"], "filled": 0, "cash_frac": p["cash_frac"],
                     "rake%": 15.0, "overlay$": 0, "start": None, "draft_group": None})
    return pd.DataFrame(rows)


def to_contest(row: dict) -> dict:
    """Contest row -> the dict dfs_solver.solve expects."""
    return {"key": row.get("contest_id"), "label": row.get("name"), "kind": row["kind"],
            "size": int(row["size"]), "fee": float(row["fee"]),
            "prize_pool": row.get("prize_pool") or None,
            "cash_frac": float(row.get("cash_frac") or 0.20),
            "entries": int(row.get("entries") or 1)}


# ── DK salary CSV import (the live slate's real prices) ──────────────────────

def parse_dk_salaries(file) -> pd.DataFrame:
    """Parse a DraftKings 'DKSalaries.csv' export -> [name, pos, salary, team, game].

    The bundle ships salaries from the pipeline, which lag the real slate (and
    are absent entirely preseason). Dropping in the actual DK file re-prices the
    pool, which is what makes the solve real for the week you are playing."""
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_name = pick("name", "nickname", "player")
    c_pos = pick("position", "roster position")
    c_sal = pick("salary")
    c_team = pick("teamabbrev", "team")
    c_game = pick("game info", "gameinfo")
    if not (c_name and c_sal):
        raise ValueError("not a DK salary file (need Name + Salary columns)")

    out = pd.DataFrame({
        "name": df[c_name].astype(str).str.strip(),
        "pos": (df[c_pos].astype(str).str.upper().str.strip() if c_pos else ""),
        "salary": pd.to_numeric(df[c_sal], errors="coerce"),
        "team": (df[c_team].astype(str).str.upper().str.strip() if c_team else ""),
        "game": (df[c_game].astype(str) if c_game else ""),
    })
    # a DK export lists showdown players twice (CPT + FLEX); keep the FLEX price
    out = out[out["pos"] != "CPT"]
    return out.dropna(subset=["salary"]).drop_duplicates(subset=["name", "pos"])


def apply_dk_salaries(players: list[dict], dk: pd.DataFrame) -> tuple[list[dict], dict]:
    """Overwrite bundle salaries with the real DK ones, matched on normalized
    name + position. Returns (players, report)."""
    def norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    lut = {(norm(r["name"]), str(r["pos"]).upper()): float(r["salary"])
           for _, r in dk.iterrows()}
    by_name = {}
    for _, r in dk.iterrows():
        by_name.setdefault(norm(r["name"]), float(r["salary"]))

    out, hit, miss = [], 0, 0
    for p in players:
        q = dict(p)
        k = (norm(p.get("name")), str(p.get("pos") or "").upper())
        sal = lut.get(k) or by_name.get(k[0])
        if sal:
            q["salary"] = sal
            hit += 1
        else:
            q["salary"] = None                 # not on the real slate -> unusable
            miss += 1
        out.append(q)
    return out, {"matched": hit, "unmatched": miss, "dk_rows": len(dk)}
