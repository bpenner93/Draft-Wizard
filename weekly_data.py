"""
weekly_data.py -- shared data layer for the weekly pages (Weekly DFS board,
Lineup Builder, Expert Compare, Vegas Watch).

Pure functions, no streamlit import (headless-testable). Pages wrap the loaders
in st.cache_data.

Data source: data/weekly_bundle.json, produced by the private pipeline's
`python export_weekly_bundle.py --season Y --week N` (same ship-a-compact-JSON
pattern as the draft board).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
BUNDLE   = DATA_DIR / "weekly_bundle.json"

SLATE_LABEL = {
    "main": "Main (Sun 1pm+4:25)", "sun_early": "Sun early (1pm)",
    "sun_late": "Sun late (4:25)", "sun_night": "SNF showdown",
    "mnf": "MNF showdown", "tnf": "TNF showdown", "all_day": "All games",
    "week_full": "Full week",
}


def load_bundle(path: Path | None = None) -> dict:
    p = Path(path) if path else BUNDLE
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def players_df(bundle: dict) -> pd.DataFrame:
    df = pd.DataFrame(bundle.get("players", []))
    if df.empty:
        return df
    for c in ("proj", "proj_engine", "pub", "our", "floor", "ceil", "own",
              "imp", "value", "def_mult", "salary"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def games_df(bundle: dict) -> pd.DataFrame:
    return pd.DataFrame(bundle.get("games", []))


def slate_options(bundle: dict) -> list[str]:
    """Slates that actually exist this week, ordered sensibly."""
    have = set()
    for g in bundle.get("games", []):
        have.update(g.get("slates", []))
    order = ["main", "all_day", "sun_early", "sun_late", "tnf", "sun_night", "mnf"]
    return [s for s in order if s in have] or ["all_day"]


def filter_slate(df: pd.DataFrame, slate: str) -> pd.DataFrame:
    if df.empty or "slates" not in df.columns:
        return df
    return df[df["slates"].apply(lambda lst: slate in (lst or []))].copy()


# ── live odds (ESPN scoreboard -- keyless, works from Streamlit Cloud) ─────────

# ESPN abbreviation -> ours (nflverse)
_ESPN2US = {"WSH": "WAS", "LAR": "LA"}


def _espn_abbr(a: str) -> str:
    return _ESPN2US.get(str(a).upper(), str(a).upper())


def fetch_espn_lines(year: int, week: int, timeout: int = 12) -> list[dict]:
    """Current lines for (year, week) from ESPN's public scoreboard API.
    Returns [{home, away, total, spread_home_book, home_imp, away_imp, state}].
    spread_home_book is BOOK convention (negative = home favored); the favorite
    gets the HIGHER implied half. Raises on network failure (caller shows it)."""
    import requests
    r = requests.get(
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
        # NB: the season year goes in `dates` -- a `year` param is silently
        # ignored and ESPN returns the CURRENT scoreboard (wrong week).
        params={"dates": year, "week": week, "seasontype": 2},
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (draft-wizard weekly)"},
    )
    r.raise_for_status()
    out = []
    for ev in r.json().get("events", []):
        for comp in ev.get("competitions", []):
            home = away = None
            for c in comp.get("competitors", []):
                ab = _espn_abbr(c.get("team", {}).get("abbreviation", ""))
                if c.get("homeAway") == "home":
                    home = ab
                else:
                    away = ab
            if not home or not away:
                continue
            total = spread_book = None
            odds = (comp.get("odds") or [{}])[0]
            ou = odds.get("overUnder")
            total = float(ou) if ou is not None else None
            det = str(odds.get("details") or "")            # e.g. "SEA -4.5" / "EVEN"
            m = re.match(r"^([A-Z]{2,4})\s*([+-]?\d+(?:\.\d+)?)$", det.strip())
            if m:
                fav, line = _espn_abbr(m.group(1)), abs(float(m.group(2)))
                if fav == home:
                    spread_book = -line                     # home favored
                elif fav == away:
                    spread_book = line                      # away favored
            elif det.strip().upper() == "EVEN":
                spread_book = 0.0
            home_imp = away_imp = None
            if total is not None and spread_book is not None:
                home_imp = round((total - spread_book) / 2, 2)   # favorite gets the higher half
                away_imp = round(total - home_imp, 2)
            out.append({
                "home": home, "away": away, "total": total,
                "spread_home_book": spread_book,
                "home_imp": home_imp, "away_imp": away_imp,
                "state": comp.get("status", {}).get("type", {}).get("state", ""),
            })
    return out


def line_moves(bundle: dict, live: list[dict]) -> pd.DataFrame:
    """Join bundle-baseline lines vs live lines by matchup -> movement table.
    One row per game: totals, spreads, per-team implied deltas."""
    base = {(g["home"], g["away"]): g for g in bundle.get("games", [])}
    rows = []
    for lv in live:
        g = base.get((lv["home"], lv["away"]))
        if g is None:                       # matchup not in the bundle week
            continue
        rows.append({
            "game": f"{lv['away']} @ {lv['home']}",
            "home": lv["home"], "away": lv["away"],
            "total_then": g.get("total"), "total_now": lv.get("total"),
            "spread_then": g.get("spread_home_book"), "spread_now": lv.get("spread_home_book"),
            "home_imp_then": g.get("home_imp"), "home_imp_now": lv.get("home_imp"),
            "away_imp_then": g.get("away_imp"), "away_imp_now": lv.get("away_imp"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for a, b, d in (("total_now", "total_then", "d_total"),
                    ("home_imp_now", "home_imp_then", "d_home_imp"),
                    ("away_imp_now", "away_imp_then", "d_away_imp")):
        df[d] = pd.to_numeric(df[a], errors="coerce") - pd.to_numeric(df[b], errors="coerce")
    df["max_abs_imp_move"] = df[["d_home_imp", "d_away_imp"]].abs().max(axis=1)
    return df.sort_values("max_abs_imp_move", ascending=False)


def team_imp_moves(moves: pd.DataFrame) -> dict[str, float]:
    """{team: implied-total delta} from a line_moves frame."""
    out = {}
    for _, r in moves.iterrows():
        if pd.notna(r.get("d_home_imp")):
            out[r["home"]] = float(r["d_home_imp"])
        if pd.notna(r.get("d_away_imp")):
            out[r["away"]] = float(r["d_away_imp"])
    return out
