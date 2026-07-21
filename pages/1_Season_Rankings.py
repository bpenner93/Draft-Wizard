"""
Season Rankings -- our season-long board as a multi-source comparison / testing
tool. Every reputable FREE projection we can get is a column: our number, Waldman,
WalterFootball, and Mike Clay (his full ESPN projection guide) -- each as sortable
projected POINTS and as a positional rank, next to the ADP market. Reads the same
data/draft_board.json the Draft Wizard uses.

Boone is rank-only where available (his point projections are paywalled) -- shown
in the Ranks view, not as a points column.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Season Rankings", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

BOARD = ROOT / "data" / "draft_board.json"

# source -> (points key, rank key, label)
SOURCES = [
    ("our_pts",  "our_rank", "Ours"),
    ("wald_pts", None,       "Waldman"),
    ("walt_pts", "walt",     "Walter"),
    ("clay_pts", "clay",     "Clay"),
]
PTS_KEYS = [s[0] for s in SOURCES]
RANK_KEYS = [s[1] for s in SOURCES if s[1]]


@st.cache_data(show_spinner=False)
def _board(mtime: float) -> tuple[list, dict]:
    with open(BOARD, encoding="utf-8") as f:
        b = json.load(f)
    return b.get("players", []), b.get("meta", {})


if not BOARD.exists():
    st.info("No board yet — run `python export_draft_board.py` in the pipeline and redeploy.")
    st.stop()

players, meta = _board(BOARD.stat().st_mtime)
st.title("📊 Season Rankings")
st.caption(f"Season {meta.get('season', '?')} · {meta.get('n_players', 0)} players · "
           f"built {str(meta.get('generated_at', ''))[:16]} · sources: ours + Waldman + "
           f"Walter + Clay (free ESPN guide) · market = FFC ADP")

df = pd.DataFrame(players)
for c in ["pts", "adp", "ecr", "our_rank", "clay", "walt", "floor", "ceil", "age",
          "rank", "pos_rank", "vor"] + PTS_KEYS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        df[c] = pd.NA
df = df[df["pts"].notna()].copy()

# Draft order = VALUE OVER REPLACEMENT (exported `rank`), NOT raw points -- raw points
# floods the top with QBs (they score most but draft rounds later).
if df["rank"].notna().any():
    df["ovr"] = df["rank"].fillna(df["pts"].rank(ascending=False, method="min")).astype(int)
else:
    df["ovr"] = df["pts"].rank(ascending=False, method="min").astype(int)
df["adp_edge"] = df["adp"] - df["ovr"]          # positive = market lets him slide past our value

# SOURCE DISAGREEMENT gauge (the testing lever): range of positional ranks across
# sources. Small = everyone agrees (likely chalk); large = sources split (investigate,
# and in DFS that spread tends to hold ownership DOWN -- a leverage tell).
rk = df[RANK_KEYS].apply(pd.to_numeric, errors="coerce")
df["n_src"] = rk.notna().sum(axis=1)
df["spread"] = (rk.max(axis=1) - rk.min(axis=1)).where(df["n_src"] >= 2)

c1, c2, c3, c4 = st.columns([1.5, 1.1, 1.2, 1.2])
pos = c1.radio("Position", ["All", "QB", "RB", "WR", "TE", "K", "DST"], horizontal=True)
view = c2.radio("Compare", ["Points", "Ranks"], horizontal=True,
                help="Points = each source's projected season points. Ranks = positional rank per source.")
sort_by = c3.selectbox("Sort by", ["Draft value (VOR)", "Consensus pts", "Ours", "Waldman",
                                    "Walter", "Clay", "ADP", "ECR", "Disagreement"])
c4a, c4b = c4.columns(2)
rookies = c4a.toggle("Rookies")
search = c4b.text_input("Search", placeholder="name…", label_visibility="collapsed")

v = df
if pos != "All":
    v = v[v["pos"] == pos]
if rookies:
    v = v[v.get("rookie", False) == True]        # noqa: E712
if search.strip():
    v = v[v["name"].str.contains(search.strip(), case=False, na=False)]

# sort: (column, ascending)
SORT = {
    "Draft value (VOR)": ("ovr", True), "Consensus pts": ("pts", False),
    "Ours": ("our_pts", False), "Waldman": ("wald_pts", False),
    "Walter": ("walt_pts", False), "Clay": ("clay_pts", False),
    "ADP": ("adp", True), "ECR": ("ecr", True), "Disagreement": ("spread", False),
}
skey, sasc = SORT[sort_by]
v = v.sort_values(skey, ascending=sasc, na_position="last")


def _flag(r):
    e = r.get("adp_edge")
    if pd.isna(e) or pd.isna(r.get("adp")):
        return ""
    return "💎" if e >= 12 else ("⚠️" if e <= -12 else "")


v = v.assign(flag=v.apply(_flag, axis=1))

if view == "Points":
    cols = ["ovr", "flag", "name", "pos", "team", "our_pts", "wald_pts", "walt_pts",
            "clay_pts", "pts", "spread", "adp", "adp_edge"]
    rename = {"ovr": "#", "flag": "", "name": "Player", "pos": "Pos", "team": "Tm",
              "our_pts": "Ours", "wald_pts": "Waldman", "walt_pts": "Walter",
              "clay_pts": "Clay", "pts": "Consensus", "spread": "Range",
              "adp": "ADP", "adp_edge": "ADP Δ"}
    ptscfg = {lbl: st.column_config.NumberColumn(format="%d", help=f"{lbl} projected season PPR points")
              for lbl in ["Ours", "Waldman", "Walter", "Clay"]}
    colcfg = {
        "#": st.column_config.NumberColumn(width="small", help="Overall draft rank (value over replacement, PPR/1QB)"),
        **ptscfg,
        "Consensus": st.column_config.NumberColumn(format="%d", help="Blended board points (ours + Waldman), market-guarded"),
        "Range": st.column_config.NumberColumn(width="small", format="%d",
                 help="Spread of positional ranks across sources — high = experts disagree (investigate; tends to suppress DFS ownership)"),
        "ADP": st.column_config.NumberColumn(format="%.0f"),
        "ADP Δ": st.column_config.NumberColumn(format="%.0f", help="ADP − our overall rank; positive = market lets him slide (💎 ≥ +12)"),
    }
else:  # Ranks
    cols = ["ovr", "flag", "name", "pos", "team", "our_rank", "clay", "walt", "ecr",
            "spread", "adp", "adp_edge"]
    rename = {"ovr": "#", "flag": "", "name": "Player", "pos": "Pos", "team": "Tm",
              "our_rank": "Ours", "clay": "Clay", "walt": "Walter", "ecr": "ECR",
              "spread": "Range", "adp": "ADP", "adp_edge": "ADP Δ"}
    colcfg = {
        "#": st.column_config.NumberColumn(width="small", help="Overall draft rank (VOR, PPR/1QB)"),
        "Ours": st.column_config.NumberColumn(width="small", help="Our positional rank"),
        "Clay": st.column_config.NumberColumn(width="small", help="Mike Clay positional rank (ESPN guide)"),
        "Walter": st.column_config.NumberColumn(width="small", help="WalterFootball positional rank"),
        "ECR": st.column_config.NumberColumn(width="small", help="Expert consensus rank (ours+Waldman+Clay+Boone+Walter)"),
        "Range": st.column_config.NumberColumn(width="small", format="%d",
                 help="Spread of positional ranks across sources — high = experts disagree"),
        "ADP": st.column_config.NumberColumn(format="%.0f"),
        "ADP Δ": st.column_config.NumberColumn(format="%.0f", help="ADP − our overall rank; positive = value (💎 ≥ +12)"),
    }

show = v[[c for c in cols if c in v.columns]].rename(columns=rename)
st.dataframe(show, hide_index=True, height=600, column_config=colcfg)

st.caption("Each source is its own published number — scales differ a little by scoring "
           "convention, so compare **within** a column (sortable) or use **Range** to find where "
           "the experts split. Clay = his full free ESPN projection guide (points + ranks). "
           "Boone's point projections are paywalled, so he's rank-only in the ECR. "
           "💎 value vs ADP · ⚠️ market reaches. This board is a season/testing gauge — weekly "
           "DFS numbers live on the Weekly DFS & Ownership pages.")
