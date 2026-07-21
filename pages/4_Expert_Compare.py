"""
Expert Compare -- paste/upload weekly ranks or projections from Justin Boone,
Mike Clay, FantasyPros, anyone respectable, and diff them against OUR weekly
number and the market. The weekly QA loop for the first weeks of the season:
big disagreements = check the news; players they rank that we zero = the
availability/news class of miss we know we have.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from weekly_data import BUNDLE, load_bundle, players_df       # noqa: E402
import expert_parse as EP                                     # noqa: E402

st.set_page_config(page_title="Expert Compare", page_icon="🔍", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def _bundle(mtime: float) -> dict:
    return load_bundle()


@st.cache_data(show_spinner=False)
def _board(path: str, mtime: float) -> list:
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("players", [])


if not BUNDLE.exists():
    st.info("No weekly bundle yet — run `python export_weekly_bundle.py --season 2026 --week N`.")
    st.stop()

b = _bundle(BUNDLE.stat().st_mtime)
meta = b.get("meta", {})
season, week = int(meta.get("season", 0)), int(meta.get("week", 0))
players = players_df(b)

board_path = ROOT / "data" / "draft_board.json"
board = _board(str(board_path), board_path.stat().st_mtime) if board_path.exists() else []
known = EP.known_universe(b.get("players", []), board)

st.title(f"🔍 Expert Compare — {season} Week {week}")
st.caption("Ingest any expert's weekly list, see where we disagree. Ranks are matched by "
           "full normalized name + position (never surnames).")

# session store of parsed sources; saved files auto-load for this week
ss = st.session_state
ss.setdefault("experts", {})
if not ss.get("_experts_loaded"):
    for src, rows in EP.load_saved(season, week).items():
        ss["experts"].setdefault(src, rows)
    ss["_experts_loaded"] = True

# ── ingest ────────────────────────────────────────────────────────────────────
with st.expander("➕ Add a source", expanded=not ss["experts"]):
    c1, c2 = st.columns([1, 2])
    source = c1.text_input("Source name", placeholder="Boone / Clay / FantasyPros…")
    up = c1.file_uploader("…or upload CSV", type=["csv", "tsv", "txt"])
    text = c2.text_area("Paste rankings text", height=220, placeholder=(
        "Quarterbacks\n1. Josh Allen, Bills\n2. Lamar Jackson at CLE\n…\n\n"
        "Running backs\n1. De'Von Achane vs BUF\n…\n\n"
        "Numbered lists, section headers, CSVs with Player/Rank/FPTS columns — all fine."))
    if st.button("Parse & add", type="primary"):
        rows = pd.DataFrame()
        if up is not None:
            rows = EP.parse_csv(up, known)
        elif text.strip():
            rows = EP.parse_text(text, known)
        if rows.empty:
            st.error("Nothing parsed — paste a list or upload a CSV first.")
        else:
            source = source.strip() or f"expert{len(ss['experts']) + 1}"
            ss["experts"][source] = rows
            n_ok = int(rows["matched"].sum())
            st.success(f"{source}: {n_ok}/{len(rows)} players matched.")
            um = rows[~rows["matched"]]
            if not um.empty:
                st.caption("Unmatched (review): " + ", ".join(um["name"].head(8)))
            st.rerun()

if not ss["experts"]:
    st.stop()

# ── source chips ──────────────────────────────────────────────────────────────
cols = st.columns(max(len(ss["experts"]), 1) + 1)
for i, (src, rows) in enumerate(list(ss["experts"].items())):
    with cols[i]:
        n_ok = int(rows["matched"].sum()) if not rows.empty else 0
        st.metric(src, f"{n_ok} matched")
        b1, b2 = st.columns(2)
        if b1.button("💾", key=f"save_{src}", help="Save into the app's data folder "
                     "(persists locally; commit to ship to the phone app)"):
            p = EP.save_expert(season, week, src, rows)
            st.toast(f"saved {p.name}")
        if b2.button("🗑", key=f"rm_{src}", help="Remove this source"):
            del ss["experts"][src]
            st.rerun()

# ── compare ───────────────────────────────────────────────────────────────────
cmp = EP.compare(players, ss["experts"])
if cmp.empty:
    st.info("No overlap between the sources and our pool yet.")
    st.stop()

st.subheader("Biggest disagreements")
c1, c2, c3 = st.columns([1.6, 1, 1])
pos = c1.radio("Position", ["All", "QB", "RB", "WR", "TE"], horizontal=True, key="cmp_pos")
min_gap = c2.slider("Min rank gap", 0, 20, 4)
only_relevant = c3.toggle("Startable only", value=True,
                          help="Experts' top-40 at the position (or our top-40)")

v = cmp.copy()
if pos != "All":
    v = v[v["pos"] == pos]
v = v[v["delta"].abs() >= min_gap]
if only_relevant:
    v = v[(v["experts"] <= 40) | (v["our_rk"] <= 40)]
v = v.assign(read=v["delta"].map(lambda d: "📈 we're higher" if d < 0 else "📉 they're higher"))

src_cols = [c for c in v.columns if c.startswith("rk_")]
show = v[["name", "pos", "team", "our_rk", "mkt_rk"] + src_cols + ["experts", "delta", "read"]].rename(
    columns={"name": "Player", "pos": "P", "team": "Tm", "our_rk": "Our #",
             "mkt_rk": "Mkt #", "experts": "Experts", "delta": "Δ", "read": ""} |
    {c: c[3:].title() for c in src_cols})
st.dataframe(show.head(80), hide_index=True, height=460, column_config={
    "Our #": st.column_config.NumberColumn(format="%.0f", help="Our positional rank (independent number)"),
    "Mkt #": st.column_config.NumberColumn(format="%.0f", help="Public projection positional rank"),
    "Experts": st.column_config.NumberColumn(format="%.1f", help="Mean expert positional rank"),
    "Δ": st.column_config.NumberColumn(format="%+.0f", help="Our rank − expert consensus (negative = we're higher)"),
} | {c[3:].title(): st.column_config.NumberColumn(format="%.0f") for c in src_cols})

st.download_button("⬇️ Download comparison CSV", cmp.to_csv(index=False),
                   file_name=f"expert_compare_{season}_wk{week:02d}.csv", mime="text/csv")

# ── the availability catch-list ───────────────────────────────────────────────
st.subheader("🚨 They rank him — we have (almost) nothing")
miss = EP.missing_from_ours(players, ss["experts"])
if miss.empty:
    st.caption("None — every expert-ranked player carries a number in our pool.")
else:
    st.dataframe(miss.rename(columns={"source": "Source", "name": "Player", "pos": "P",
                                      "their_rank": "Their #", "why": "Why flagged"}),
                 hide_index=True)
    st.caption("This is the exact class of miss the availability post-mortem found (who plays "
               "beats projection quality): usually a promoted starter, a return from injury, or "
               "a depth-chart change our feeds haven't caught. Fix the FACT in the pipeline "
               "(depth chart / availability), don't hand-tweak the number.")

st.divider()
st.caption("Workflow: Tue–Wed ingest experts as they publish → chase every big Δ to a REASON "
           "(news, role, injury) → fix facts in the pipeline → re-run + re-export the bundle. "
           "Our number is at parity when the facts are right — the gaps are usually facts.")
