"""
Season Rankings -- our season-long board (consensus points, our rank vs
Clay/ECR vs the ADP market) as a sortable, phone-friendly table. Reads the
same data/draft_board.json the Draft Wizard uses.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Season Rankings", page_icon="📊", layout="wide")

BOARD = ROOT / "data" / "draft_board.json"


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
           f"built {str(meta.get('generated_at', ''))[:16]} · consensus = ours + Waldman, "
           f"ECR = Clay/Boone/Walter, market = FFC ADP")

df = pd.DataFrame(players)
for c in ("pts", "adp", "ecr", "our_rank", "clay", "floor", "ceil", "age"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df["pts"].notna()].copy()
df["ovr"] = df["pts"].rank(ascending=False, method="min").astype(int)
df["adp_edge"] = df["adp"] - df["ovr"]          # positive = market lets him slide past our value

c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.4])
pos = c1.radio("Position", ["All", "QB", "RB", "WR", "TE", "K", "DST"], horizontal=True)
rookies = c2.toggle("Rookies only")
sort_by = c3.selectbox("Sort", ["Our points", "ADP", "ECR"])
search = c4.text_input("Search", placeholder="player name…")

v = df
if pos != "All":
    v = v[v["pos"] == pos]
if rookies:
    v = v[v.get("rookie", False) == True]        # noqa: E712
if search.strip():
    v = v[v["name"].str.contains(search.strip(), case=False, na=False)]
v = v.sort_values({"Our points": "ovr", "ADP": "adp", "ECR": "ecr"}[sort_by],
                  na_position="last")

def _flag(r):
    e = r.get("adp_edge")
    if pd.isna(e) or pd.isna(r.get("adp")):
        return ""
    if e >= 12:
        return "💎"           # big value vs market
    if e <= -12:
        return "⚠️"           # market takes him well before our number
    return ""

v = v.assign(flag=v.apply(_flag, axis=1))
for c in ("walt", "walt_pts"):                   # older boards predate the Walter columns
    if c not in v.columns:
        v[c] = pd.NA
show = v[["ovr", "flag", "name", "pos", "team", "pts", "our_rank", "clay", "walt", "ecr",
          "adp", "adp_edge", "walt_pts", "floor", "ceil", "age", "rookie"]].rename(columns={
    "ovr": "#", "flag": "", "name": "Player", "pos": "Pos", "team": "Tm",
    "pts": "Pts", "our_rank": "Our", "clay": "Clay", "walt": "Walt", "ecr": "ECR",
    "adp": "ADP", "adp_edge": "ADP Δ", "walt_pts": "Walt pts",
    "floor": "Wk floor", "ceil": "Wk ceil", "age": "Age", "rookie": "R",
})
st.dataframe(
    show, hide_index=True, height=620,
    column_config={
        "#": st.column_config.NumberColumn(width="small"),
        "Pts": st.column_config.NumberColumn(format="%.0f", help="Consensus season PPR points"),
        "Our": st.column_config.NumberColumn(width="small", help="Our positional rank"),
        "Clay": st.column_config.NumberColumn(width="small", help="Mike Clay positional rank"),
        "Walt": st.column_config.NumberColumn(width="small", help="WalterFootball positional rank"),
        "ECR": st.column_config.NumberColumn(width="small", help="Expert consensus positional rank (ours + Waldman + Clay + Boone + Walter)"),
        "Walt pts": st.column_config.NumberColumn(width="small", format="%d", help="WalterFootball projected season points"),
        "ADP": st.column_config.NumberColumn(format="%.0f"),
        "ADP Δ": st.column_config.NumberColumn(
            format="%.0f", help="ADP − our overall rank. Positive = the market drafts him LATER than our value (💎 ≥ +12)."),
        "Wk floor": st.column_config.NumberColumn(format="%.1f"),
        "Wk ceil": st.column_config.NumberColumn(format="%.1f"),
        "R": st.column_config.CheckboxColumn(width="small", help="Rookie"),
    },
)
st.caption("💎 = our number says he's a value at ADP (≥12 spots) · ⚠️ = market prices him "
           "well ahead of our number. In-season this board stays structural — weekly moves "
           "live on the Weekly DFS page.")
