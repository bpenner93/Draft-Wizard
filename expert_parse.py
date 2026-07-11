"""
expert_parse.py -- ingest ANY respectable expert's weekly ranks/projections
(Boone, Clay, FantasyPros, a CSV export, a pasted numbered list...) and compare
them to our weekly numbers.

Design: parsing is fuzzy but MATCHING is strict -- every parsed line is resolved
against the app's known-player universe via the same full-name normalization the
draft board uses (draft_names.norm_name; never surnames -- the PBP lesson).
Unmatched lines are returned, not dropped silently.

Pure functions, no streamlit (headless-testable).
"""
from __future__ import annotations

import io
import json
import re
import datetime as _dt
from pathlib import Path

import pandas as pd

from draft_names import norm_name, pos_norm

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

# section headers that switch position context in pasted text
_SECTION = re.compile(
    r"^\s*(?:top\s+)?(quarterbacks?|running\s*backs?|wide\s*receivers?|tight\s*ends?|"
    r"kickers?|defenses?|dst|qbs?|rbs?|wrs?|tes?|ks?)\s*[:\-]?\s*$", re.I)
_SECTION_POS = {"quarterback": "QB", "qb": "QB", "running back": "RB", "rb": "RB",
                "wide receiver": "WR", "wr": "WR", "tight end": "TE", "te": "TE",
                "kicker": "K", "k": "K", "defense": "DST", "dst": "DST"}

_RANK_PREFIX = re.compile(r"^\s*(\d{1,3})\s*[\.\):\-]?\s+(.*)$")
_TRAILING_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*$")


def known_universe(bundle_players: list[dict], board_players: list[dict] | None = None) -> dict:
    """{name_key: {name,pos,team}} from the weekly bundle (primary) + the season
    draft board (fills anyone not in this week's pool)."""
    out = {}
    for p in (board_players or []):
        k = norm_name(p.get("name", ""))
        if k:
            out[k] = {"name": p["name"], "pos": pos_norm(p.get("pos")), "team": p.get("team")}
    for p in bundle_players or []:
        k = norm_name(p.get("name", ""))
        if k:
            out[k] = {"name": p["name"], "pos": p.get("pos"), "team": p.get("team")}
    return out


def _match_line(rest: str, known: dict) -> tuple[str | None, str]:
    """Longest known-player prefix match of the normalized line.
    Returns (name_key or None, the raw name text consumed)."""
    key = norm_name(re.split(r"[(\[,–—]| vs\.? | at | v\.? ", rest, maxsplit=1, flags=re.I)[0])
    if not key:
        return None, rest
    if key in known:
        return key, rest
    toks = key.split()
    # try longest prefix first (handles trailing team/notes: "josh allen buf home")
    for ln in range(min(len(toks), 5), 1, -1):
        cand = " ".join(toks[:ln])
        if cand in known:
            return cand, rest
    return None, rest


def parse_text(text: str, known: dict) -> pd.DataFrame:
    """Parse pasted rankings text -> rows [name, key, pos, team, rank, proj, matched].
    Rank = explicit number prefix if present, else running order. A trailing
    number with a decimal point is read as a points projection."""
    rows, order = [], 0
    ctx_pos = None
    for raw in (text or "").splitlines():
        line = raw.strip().strip("•*·|")
        if not line:
            continue
        m = _SECTION.match(line)
        if m:
            tok = re.sub(r"s$", "", m.group(1).lower().replace("  ", " "))
            for k, v in _SECTION_POS.items():
                if tok.startswith(k):
                    ctx_pos = v
                    break
            continue
        rank = None
        m = _RANK_PREFIX.match(line)
        rest = line
        if m:
            rank, rest = int(m.group(1)), m.group(2)
        proj = None
        t = _TRAILING_NUM.search(rest)
        if t and "." in t.group(1):                 # decimals only -- a bare int is jersey/team noise
            proj = float(t.group(1))
        key, _ = _match_line(rest, known)
        order += 1
        info = known.get(key, {})
        rows.append({
            "name": info.get("name") or rest[:40],
            "key": key, "pos": info.get("pos") or ctx_pos,
            "team": info.get("team"),
            "rank": rank if rank is not None else order,
            "proj": proj, "matched": key is not None,
        })
    return pd.DataFrame(rows)


def parse_csv(file_or_text, known: dict) -> pd.DataFrame:
    """Parse an uploaded CSV/TSV of ranks or projections. Column names are
    matched fuzzily (player/name, rank/rk/#, proj/fpts/pts, pos, team)."""
    if isinstance(file_or_text, str):
        file_or_text = io.StringIO(file_or_text)
    try:
        df = pd.read_csv(file_or_text)
    except Exception:
        file_or_text.seek(0)
        df = pd.read_csv(file_or_text, sep="\t")
    cols = {str(c).strip().lower(): c for c in df.columns}

    def find(*names):
        for n in names:
            for lc, c in cols.items():
                if lc == n or lc.startswith(n):
                    return c
        return None

    c_name = find("player", "name")
    c_rank = find("rank", "rk", "#", "ovr", "ecr")
    c_proj = find("proj", "fpts", "fantasy", "points", "pts", "fpts/g")
    c_pos  = find("pos")
    c_team = find("team", "tm")
    if c_name is None:
        return pd.DataFrame()
    rows = []
    for i, r in df.iterrows():
        nm = str(r[c_name])
        key = norm_name(re.sub(r"\(.*?\)", "", nm))
        if key not in known:
            key2, _ = _match_line(nm, known)
            key = key2
        info = known.get(key, {})
        pos = info.get("pos") or (pos_norm(str(r[c_pos])) if c_pos else None)
        rows.append({
            "name": info.get("name") or nm,
            "key": key, "pos": pos,
            "team": info.get("team") or (str(r[c_team]) if c_team else None),
            "rank": float(r[c_rank]) if c_rank and pd.notna(r[c_rank]) else i + 1,
            "proj": float(r[c_proj]) if c_proj and pd.notna(r[c_proj]) else None,
            "matched": key is not None,
        })
    return pd.DataFrame(rows)


# ── persistence (local runs write into the app repo's data/experts) ──────────

def experts_dir(base: Path | None = None) -> Path:
    return (base or Path(__file__).parent / "data") / "experts"


def save_expert(season: int, week: int, source: str, rows: pd.DataFrame,
                base: Path | None = None) -> Path:
    d = experts_dir(base)
    d.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_") or "expert"
    p = d / f"{season}_wk{week:02d}_{slug}.json"
    payload = {"source": source, "season": season, "week": week,
               "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
               "rows": rows.where(pd.notna(rows), None).to_dict("records")}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return p


def load_saved(season: int, week: int, base: Path | None = None) -> dict[str, pd.DataFrame]:
    d = experts_dir(base)
    out = {}
    if not d.exists():
        return out
    for p in sorted(d.glob(f"{season}_wk{week:02d}_*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            out[payload.get("source", p.stem)] = pd.DataFrame(payload.get("rows", []))
        except Exception:
            continue
    return out


# ── comparison ────────────────────────────────────────────────────────────────

def _pos_rank(df: pd.DataFrame, col: str, pos_col: str = "pos") -> pd.Series:
    """1-based rank within position, higher col value = better rank."""
    return df.groupby(pos_col)[col].rank(ascending=False, method="min")


def compare(players: pd.DataFrame, experts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per player: our positional rank (independent number), the market
    rank, each expert's positional rank, the expert consensus, and delta =
    ours - consensus (negative = WE are higher on him). Restricted to positions
    the experts actually submitted so deltas mean something."""
    df = players.copy()
    df["key"] = df["name"].map(norm_name)
    base_num = df["our"].where(df["our"].notna(), df["proj"])
    df["_base"] = pd.to_numeric(base_num, errors="coerce")
    df["our_rk"] = _pos_rank(df.assign(_v=df["_base"]), "_v")
    if "pub" in df.columns and df["pub"].notna().any():
        df["mkt_rk"] = _pos_rank(df.assign(_v=pd.to_numeric(df["pub"], errors="coerce")), "_v")
    else:
        df["mkt_rk"] = None

    src_cols = []
    for src, rows in experts.items():
        if rows is None or rows.empty:
            continue
        r = rows[rows["matched"] & rows["pos"].notna()].copy()
        if r.empty:
            continue
        # positional rank from their ordering (rank asc within pos)
        r["src_rk"] = r.groupby("pos")["rank"].rank(method="first")
        col = f"rk_{src}"
        df = df.merge(r[["key", "pos", "src_rk"]].rename(columns={"src_rk": col}),
                      on=["key", "pos"], how="left")
        src_cols.append(col)
    if not src_cols:
        return pd.DataFrame()

    df["experts"] = df[src_cols].mean(axis=1, skipna=True)
    df["n_experts"] = df[src_cols].notna().sum(axis=1)
    covered_pos = set()
    for src, rows in experts.items():
        if rows is not None and not rows.empty:
            covered_pos |= set(rows.loc[rows["matched"], "pos"].dropna().unique())
    out = df[df["pos"].isin(covered_pos) & (df["n_experts"] > 0)].copy()
    out["delta"] = out["our_rk"] - out["experts"]
    keep = ["name", "pos", "team", "opp", "salary", "our", "proj", "pub",
            "our_rk", "mkt_rk", "experts", "n_experts", "delta"] + src_cols
    out = out[[c for c in keep if c in out.columns]]
    return out.sort_values("delta", key=lambda s: s.abs(), ascending=False)


def missing_from_ours(players: pd.DataFrame, experts: dict[str, pd.DataFrame],
                      top_n: int = 30) -> pd.DataFrame:
    """Players an expert ranks inside their top_n at the position but who carry
    NO number in our pool (or aren't in it at all) -- exactly the availability /
    news class of miss we know we have. Returns [source, name, pos, their rank]."""
    have = {norm_name(n) for n in players["name"]}
    weak = {norm_name(r["name"]) for _, r in players.iterrows()
            if (pd.isna(r.get("our")) or (r.get("our") or 0) <= 0)}
    rows = []
    for src, r in experts.items():
        if r is None or r.empty:
            continue
        rr = r[r["pos"].notna()].copy()
        rr["src_rk"] = rr.groupby("pos")["rank"].rank(method="first")
        for _, x in rr[rr["src_rk"] <= top_n].iterrows():
            k = x["key"] or norm_name(x["name"])
            if (k not in have) or (k in weak):
                rows.append({"source": src, "name": x["name"], "pos": x["pos"],
                             "their_rank": int(x["src_rk"]),
                             "why": "not in our pool" if k not in have else "we project ~0"})
    return pd.DataFrame(rows).sort_values(["pos", "their_rank"]) if rows else pd.DataFrame()
