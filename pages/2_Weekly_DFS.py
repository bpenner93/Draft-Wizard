"""
Weekly DFS -- the week's DFS board (projection, salary, value, projected
ownership, floor/ceiling, Vegas environment) with slate filtering, plus the
ours-vs-market divergence view (our sanity-flag list) and the games board.

Data: data/weekly_bundle.json  (python export_weekly_bundle.py --season Y --week N)
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from weekly_data import (BUNDLE, load_bundle, players_df, games_df,   # noqa: E402
                         slate_options, filter_slate, SLATE_LABEL)

st.set_page_config(page_title="Weekly DFS", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def _bundle(mtime: float) -> dict:
    return load_bundle()


if not BUNDLE.exists():
    st.info("No weekly bundle yet — run `python export_weekly_bundle.py --season 2026 --week N` "
            "in the pipeline, then redeploy (or refresh locally).")
    st.stop()

b = _bundle(BUNDLE.stat().st_mtime)
meta = b.get("meta", {})
season, week = meta.get("season"), meta.get("week")

st.title(f"🎯 Weekly DFS — {season} Week {week}")
st.caption(f"built {str(meta.get('generated_at', ''))[:16]} · odds as of "
           f"{str(meta.get('odds_asof') or 'schedule lines')[:16]}"
           + (" · ⚠ HISTORICAL bundle (backtest render)" if meta.get("historical") else ""))

df = players_df(b)
games = games_df(b)

tab_board, tab_vs, tab_games = st.tabs(["📋 Board", "🆚 Ours vs market", "🏟 Games"])

# ── Board ─────────────────────────────────────────────────────────────────────
with tab_board:
    c1, c2, c3, c4 = st.columns([1.3, 1.6, 1, 1])
    slates = slate_options(b)
    slate = c1.selectbox("Slate", slates, format_func=lambda s: SLATE_LABEL.get(s, s))
    pos = c2.radio("Pos", ["All", "QB", "RB", "WR", "TE", "DST"], horizontal=True)
    sort_by = c3.selectbox("Sort", ["Proj", "Value", "Ceiling", "Own%", "Salary"])
    hide_cheap = c4.toggle("Hide < 5 proj", value=True)

    v = filter_slate(df, slate)
    if pos != "All":
        v = v[v["pos"] == pos]
    if hide_cheap:
        v = v[v["proj"].fillna(0) >= 5]
    key = {"Proj": "proj", "Value": "value", "Ceiling": "ceil", "Own%": "own", "Salary": "salary"}[sort_by]
    v = v.sort_values(key, ascending=False, na_position="last")

    v = v.assign(Opp=v.apply(lambda r: ("" if r.get("home") else "@") + str(r.get("opp") or "?"), axis=1))
    show = v[["name", "pos", "team", "Opp", "salary", "proj", "our", "pub", "floor",
              "ceil", "own", "value", "imp", "def_mult"]].rename(columns={
        "name": "Player", "pos": "P", "team": "Tm", "salary": "Salary",
        "proj": "Proj", "our": "Ours", "pub": "Mkt", "floor": "Flr", "ceil": "Ceil",
        "own": "Own%", "value": "Val", "imp": "ImpTot", "def_mult": "DefX",
    })
    st.dataframe(
        show, hide_index=True, height=600,
        column_config={
            "Salary": st.column_config.NumberColumn(format="$%d"),
            "Proj": st.column_config.NumberColumn(format="%.1f",
                help="DFS selection number: public weekly market projection, our engine as fallback"),
            "Ours": st.column_config.NumberColumn(format="%.1f",
                help="Our INDEPENDENT weekly number (engine + in-season blend + injury overlay)"),
            "Mkt": st.column_config.NumberColumn(format="%.1f", help="Public weekly projection"),
            "Flr": st.column_config.NumberColumn(format="%.1f"),
            "Ceil": st.column_config.NumberColumn(format="%.1f"),
            "Own%": st.column_config.NumberColumn(format="%.1f",
                help="Projected DK %Drafted (calibrated appeal-softmax model)"),
            "Val": st.column_config.NumberColumn(format="%.2f", help="Proj pts per $1K salary"),
            "ImpTot": st.column_config.NumberColumn(format="%.1f", help="Team implied total"),
            "DefX": st.column_config.NumberColumn(format="%.2f", help="Matchup multiplier vs this defense"),
        },
    )
    n_nosal = int((filter_slate(df, slate)["salary"].isna()).sum())
    if n_nosal:
        st.caption(f"⚠ {n_nosal} players have no salary yet (pre-release fallbacks). Import the real "
                   f"DK CSV in the pipeline when the slate posts, re-export, and they'll price in.")

# ── Ours vs market ────────────────────────────────────────────────────────────
with tab_vs:
    st.caption("The market number WINS for DFS selection (validated 14/14 weeks) — this table is "
               "our **sanity-flag list**: a big gap = the model sees a different role/volume than "
               "the market. Check news (role change, injury, depth chart) before lock; if the market "
               "moved and we didn't, that's usually OUR miss to fix.")
    have_pub = df["pub"].notna().sum() if "pub" in df.columns else 0
    if have_pub == 0:
        st.info("No public weekly numbers in this bundle (preseason / salaries not posted). "
                "Showing our independent number vs the DFS selection number instead.")
        v = df[df["proj"].notna() & df["our"].notna()].copy()
        v["diff"] = v["our"] - v["proj"]
        a_col, b_col = "our", "proj"
        a_lab, b_lab = "Ours", "Selection"
    else:
        v = df[df["pub"].notna() & df["our"].notna()].copy()
        v["diff"] = v["our"] - v["pub"]
        a_col, b_col = "our", "pub"
        a_lab, b_lab = "Ours", "Market"
    c1, c2 = st.columns([1.6, 1])
    pos2 = c1.radio("Position", ["All", "QB", "RB", "WR", "TE"], horizontal=True, key="vs_pos")
    min_gap = c2.slider("Min gap (pts)", 0.0, 8.0, 2.0, 0.5)
    if pos2 != "All":
        v = v[v["pos"] == pos2]
    v = v[v["diff"].abs() >= min_gap].sort_values("diff", key=lambda s: s.abs(), ascending=False)
    v["flag"] = v["diff"].map(lambda d: "📈 we're higher" if d > 0 else "📉 we're lower")
    show = v[["name", "pos", "team", a_col, b_col, "diff", "flag", "salary", "own"]].rename(columns={
        "name": "Player", "pos": "P", "team": "Tm", a_col: a_lab, b_col: b_lab,
        "diff": "Δ", "flag": "", "salary": "Salary", "own": "Own%"})
    st.dataframe(show.head(60), hide_index=True, height=520, column_config={
        a_lab: st.column_config.NumberColumn(format="%.1f"),
        b_lab: st.column_config.NumberColumn(format="%.1f"),
        "Δ": st.column_config.NumberColumn(format="%+.1f"),
        "Salary": st.column_config.NumberColumn(format="$%d"),
        "Own%": st.column_config.NumberColumn(format="%.1f"),
    })
    st.caption("📈 rows: if news CONFIRMS the role we see, that's leverage the field misses. "
               "📉 rows: usually our lag — promoted starters, role changes we haven't caught "
               "(the known miss patterns: promoted QBs, ascending pass-catchers). "
               "Cross-check on the Expert Compare page.")

# ── Games ─────────────────────────────────────────────────────────────────────
with tab_games:
    if games.empty:
        st.info("No games in the bundle.")
    else:
        g = games.copy()
        g["Game"] = g["away"] + " @ " + g["home"]
        g["Fav (book)"] = g.apply(
            lambda r: ("EVEN" if (r.get("spread_home_book") or 0) == 0 else
                       (f"{r['home']} {r['spread_home_book']:+.1f}" if r["spread_home_book"] < 0
                        else f"{r['away']} {-r['spread_home_book']:+.1f}"))
            if pd.notna(r.get("spread_home_book")) else "—", axis=1)
        show = g[["Game", "day", "time", "total", "Fav (book)", "home_imp", "away_imp", "src"]].rename(
            columns={"day": "Day", "time": "ET", "total": "Total",
                     "home_imp": "Home imp", "away_imp": "Away imp", "src": "Lines"})
        st.dataframe(show, hide_index=True, height=560, column_config={
            "Total": st.column_config.NumberColumn(format="%.1f"),
            "Home imp": st.column_config.NumberColumn(format="%.1f"),
            "Away imp": st.column_config.NumberColumn(format="%.1f"),
        })
        st.caption("Stack targets = highest totals with a real passing game (the builder gates at "
                   "~44+). Live movement is on the 📈 Vegas Watch page.")
