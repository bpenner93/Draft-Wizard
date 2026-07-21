"""
DFS Ownership -- the weekly ownership / leverage tool. Projected DK %Drafted for
the slate (our calibrated appeal-softmax model, Spearman 0.93 vs real ownership),
turned into the GPP question that matters: who is CHALK (owned more than his
projection earns) and who is LEVERAGE (a strong projection the field is sleeping on).

Leverage score = projection percentile − ownership percentile, WITHIN position
(a WR at 15% and an RB at 15% aren't the same). Positive = under-owned for his
projection (contrarian edge); negative = over-owned (chalk trap).

Data: data/weekly_bundle.json  (python export_weekly_bundle.py --season Y --week N)
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from weekly_data import (BUNDLE, load_bundle, players_df,          # noqa: E402
                         slate_options, filter_slate, SLATE_LABEL)

st.set_page_config(page_title="DFS Ownership", page_icon="🫧", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def _bundle(mtime: float) -> dict:
    return load_bundle()


if not BUNDLE.exists():
    st.info("No weekly bundle yet — run `python export_weekly_bundle.py --season 2026 --week N`.")
    st.stop()

b = _bundle(BUNDLE.stat().st_mtime)
meta = b.get("meta", {})
df = players_df(b)
has_market = "pub" in df.columns and df["pub"].notna().any()
st.title(f"🫧 DFS Ownership — {meta.get('season')} Week {meta.get('week')}")
st.caption("Projected DK %Drafted (calibrated appeal-softmax model) → chalk vs leverage. "
           + ("" if has_market else
              "⚠ preseason bundle: ownership is projected off our engine number (no public market yet)."))

c1, c2, c3 = st.columns([1.4, 1.6, 1])
slate = c1.selectbox("Slate", slate_options(b), format_func=lambda s: SLATE_LABEL.get(s, s))
pos_f = c2.radio("Pos", ["All", "QB", "RB", "WR", "TE", "DST"], horizontal=True)
min_proj = c3.slider("Min proj", 0.0, 15.0, 6.0, 0.5)

pool = filter_slate(df, slate)
pool = pool[pool["proj"].fillna(0) >= min_proj].copy()
pool = pool[pool["own"].notna() & pool["salary"].notna()]   # DFS tool: only rosterable players
if pool.empty:
    st.warning("No players match — lower the min projection.")
    st.stop()

# ── leverage math: percentiles WITHIN position ───────────────────────────────
pool["proj_pct"] = pool.groupby("pos")["proj"].rank(pct=True) * 100
pool["own_pct"] = pool.groupby("pos")["own"].rank(pct=True) * 100
pool["lev"] = (pool["proj_pct"] - pool["own_pct"]).round(0)
pool["tag"] = pd.cut(pool["lev"], [-101, -25, 25, 101],
                     labels=["🔴 Chalk", "⚪ Neutral", "🟢 Leverage"])

view = pool if pos_f == "All" else pool[pool["pos"] == pos_f]

# ── summary metrics ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
chalk = view.loc[view["own"].idxmax()]
lev_pool = view[(view["proj_pct"] >= 60)]           # a real play, not a punt
best_lev = lev_pool.loc[lev_pool["lev"].idxmax()] if not lev_pool.empty else view.loc[view["lev"].idxmax()]
m1.metric("Players", len(view))
m2.metric("Avg ownership", f"{view['own'].mean():.1f}%")
m3.metric("Chalkiest", f"{chalk['name'].split()[-1]}", f"{chalk['own']:.0f}% owned")
m4.metric("Top leverage", f"{best_lev['name'].split()[-1]}", f"+{best_lev['lev']:.0f} lev")

tab_board, tab_split, tab_scatter = st.tabs(["🎯 Leverage board", "🔴 Chalk vs 🟢 Leverage", "📊 Own vs projection"])

# ── leverage board ───────────────────────────────────────────────────────────
with tab_board:
    sort_by = st.radio("Sort", ["Leverage", "Ownership", "Projection", "Value"],
                       horizontal=True)
    key = {"Leverage": "lev", "Ownership": "own", "Projection": "proj", "Value": "value"}[sort_by]
    vv = view.sort_values(key, ascending=False)
    vv = vv.assign(Opp=vv.apply(lambda r: ("" if r.get("home") else "@") + str(r.get("opp") or "?"), axis=1))
    show = vv[["tag", "name", "pos", "team", "Opp", "salary", "proj", "own", "value", "lev"]].rename(columns={
        "tag": "", "name": "Player", "pos": "P", "team": "Tm", "salary": "Salary",
        "proj": "Proj", "own": "Own%", "value": "Val", "lev": "Lev"})
    st.dataframe(show, hide_index=True, height=520, column_config={
        "Salary": st.column_config.NumberColumn(format="$%d"),
        "Proj": st.column_config.NumberColumn(format="%.1f"),
        "Own%": st.column_config.NumberColumn(format="%.1f", help="Projected DK %Drafted"),
        "Val": st.column_config.NumberColumn(format="%.2f", help="Proj points per $1K"),
        "Lev": st.column_config.NumberColumn(format="%+d",
               help="Projection percentile − ownership percentile (within position). + = under-owned for his projection"),
    })
    st.caption("🟢 Leverage = the field is under-rostering a strong projection (GPP edge). "
               "🔴 Chalk = owned more than the projection earns — fine in cash, a trap in large-field GPP "
               "unless he smashes. Cross-check the 🟢 plays against news on the Expert Compare page.")

# ── chalk vs leverage split ──────────────────────────────────────────────────
with tab_split:
    cc, lc = st.columns(2)
    with cc:
        st.subheader("🔴 Chalk (most owned)")
        top = view.nlargest(12, "own")[["name", "pos", "team", "own", "proj", "lev"]].rename(
            columns={"name": "Player", "pos": "P", "team": "Tm", "own": "Own%", "proj": "Proj", "lev": "Lev"})
        st.dataframe(top, hide_index=True, height=440, column_config={
            "Own%": st.column_config.NumberColumn(format="%.1f"),
            "Proj": st.column_config.NumberColumn(format="%.1f"),
            "Lev": st.column_config.NumberColumn(format="%+d")})
    with lc:
        st.subheader("🟢 Leverage (strong proj, low own)")
        pj = view[view["proj_pct"] >= 55]
        best = pj.nlargest(12, "lev")[["name", "pos", "team", "own", "proj", "lev"]].rename(
            columns={"name": "Player", "pos": "P", "team": "Tm", "own": "Own%", "proj": "Proj", "lev": "Lev"})
        st.dataframe(best, hide_index=True, height=440, column_config={
            "Own%": st.column_config.NumberColumn(format="%.1f"),
            "Proj": st.column_config.NumberColumn(format="%.1f"),
            "Lev": st.column_config.NumberColumn(format="%+d")})
    st.caption("Build GPP around the 🟢 column, pair with just enough chalk to stay correlated. "
               "The 🔴 column is what the field over-owns — being underweight there is where "
               "tournaments are won.")

# ── scatter: ownership vs projection ─────────────────────────────────────────
with tab_scatter:
    st.caption("Each dot is a player. The line of best fit is roughly what ownership *should* be "
               "for a projection — dots well BELOW the cloud at a given projection are leverage, "
               "dots ABOVE are chalk.")
    sc = view[["own", "proj", "pos", "name"]].copy()
    st.scatter_chart(sc, x="own", y="proj", color="pos", height=440,
                     x_label="Projected ownership %", y_label="Projection (DK pts)")
    st.caption("Ownership tracks the public **projection consensus** — the same multi-source read on the "
               "Season Rankings page. Where our number diverges from the field's (Weekly DFS → Ours vs "
               "market), ownership lags, and that gap is the leverage this page ranks.")
