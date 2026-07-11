"""
Vegas Watch -- follow the lines as they move. Baseline = the odds our current
projections were built on (bundled at export time); live = ESPN's public
scoreboard lines fetched on demand (keyless, works on Streamlit Cloud). Big
implied-total moves = the environment shifted since our last run -> re-run the
pipeline + re-export before lock.
"""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from weekly_data import (BUNDLE, load_bundle, players_df, games_df,   # noqa: E402
                         fetch_espn_lines, line_moves, team_imp_moves)

st.set_page_config(page_title="Vegas Watch", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def _bundle(mtime: float) -> dict:
    return load_bundle()


@st.cache_data(ttl=600, show_spinner="fetching live lines…")
def _live(year: int, week: int, nonce: int) -> list[dict]:
    return fetch_espn_lines(year, week)


if not BUNDLE.exists():
    st.info("No weekly bundle yet — run `python export_weekly_bundle.py --season 2026 --week N`.")
    st.stop()

b = _bundle(BUNDLE.stat().st_mtime)
meta = b.get("meta", {})
season, week = int(meta.get("season", 0)), int(meta.get("week", 0))

st.title(f"📈 Vegas Watch — {season} Week {week}")
st.caption(f"Baseline lines: what the current projections/bundle were built on "
           f"(as of {str(meta.get('odds_asof') or meta.get('generated_at', ''))[:16]}).")

games = games_df(b)
ss = st.session_state

c1, c2 = st.columns([1, 2])
if c1.button("🔄 Fetch live lines (ESPN)", type="primary", width="stretch"):
    ss["vw_nonce"] = ss.get("vw_nonce", 0) + 1
thresh = c2.slider("Alert threshold — team implied-total move (pts)", 0.5, 3.0, 1.0, 0.25)

live, err = None, None
if ss.get("vw_nonce"):
    try:
        live = _live(season, week, ss["vw_nonce"])
    except Exception as e:
        err = str(e)

if err:
    st.error(f"Live fetch failed ({err}) — showing baseline only.")

if live is None:
    # baseline-only view
    if games.empty:
        st.info("No games in the bundle.")
        st.stop()
    g = games.copy()
    g["Game"] = g["away"] + " @ " + g["home"]
    show = g[["Game", "day", "time", "total", "spread_home_book", "home_imp", "away_imp", "src"]].rename(
        columns={"day": "Day", "time": "ET", "total": "Total", "spread_home_book": "Home spread",
                 "home_imp": "Home imp", "away_imp": "Away imp", "src": "Lines"})
    st.dataframe(show, hide_index=True, height=560, column_config={
        "Total": st.column_config.NumberColumn(format="%.1f"),
        "Home spread": st.column_config.NumberColumn(format="%+.1f", help="Book convention: negative = home favored"),
        "Home imp": st.column_config.NumberColumn(format="%.1f"),
        "Away imp": st.column_config.NumberColumn(format="%.1f"),
    })
    st.caption("Tap **Fetch live lines** to compare against the market right now.")
    st.stop()

# ── movement view ─────────────────────────────────────────────────────────────
moves = line_moves(b, live)
n_lines = sum(1 for lv in live if lv.get("total") is not None)
st.caption(f"Live: {len(live)} games from ESPN, {n_lines} with posted lines · "
           f"fetched {dt.datetime.now().strftime('%H:%M')}")

if moves.empty:
    st.warning("No matchup overlap between the live feed and the bundle week — check that the "
               "bundle's week matches the current NFL week.")
    st.stop()

big = moves[moves["max_abs_imp_move"].fillna(0) >= thresh]
if big.empty:
    st.success(f"😴 No team's implied total has moved ≥ {thresh:g} pts since the baseline. "
               "Projections are still aligned with the market environment.")
else:
    st.error(f"🚨 {len(big)} game(s) moved ≥ {thresh:g} pts of implied total — the environment "
             "shifted since projections were built. Re-run + re-export before lock "
             "(`python export_weekly_bundle.py --fetch-odds …`), and check WHY it moved "
             "(injury news usually) on the Expert Compare page.")

mv = moves.copy()
mv["Δ total"] = mv["d_total"]
mv["alert"] = mv["max_abs_imp_move"].map(lambda x: "🚨" if pd.notna(x) and x >= thresh else "")
show = mv[["alert", "game", "total_then", "total_now", "Δ total",
           "home", "d_home_imp", "away", "d_away_imp"]].rename(columns={
    "alert": "", "game": "Game", "total_then": "Total (then)", "total_now": "Total (now)",
    "home": "Home", "d_home_imp": "Δ home imp", "away": "Away", "d_away_imp": "Δ away imp"})
st.dataframe(show, hide_index=True, height=480, column_config={
    "Total (then)": st.column_config.NumberColumn(format="%.1f"),
    "Total (now)": st.column_config.NumberColumn(format="%.1f"),
    "Δ total": st.column_config.NumberColumn(format="%+.1f"),
    "Δ home imp": st.column_config.NumberColumn(format="%+.1f"),
    "Δ away imp": st.column_config.NumberColumn(format="%+.1f"),
})

# ── affected players ──────────────────────────────────────────────────────────
imp_mv = team_imp_moves(moves)
hot = {t: d for t, d in imp_mv.items() if abs(d) >= thresh}
if hot:
    st.subheader("Players in moved games")
    df = players_df(b)
    aff = df[df["team"].isin(hot)].copy()
    aff["Δ imp"] = aff["team"].map(hot)
    aff = aff.sort_values(["Δ imp", "proj"], ascending=[False, False])
    show = aff[["name", "pos", "team", "Δ imp", "proj", "salary", "own"]].head(30).rename(
        columns={"name": "Player", "pos": "P", "team": "Tm", "proj": "Proj",
                 "salary": "Salary", "own": "Own%"})
    st.dataframe(show, hide_index=True, column_config={
        "Δ imp": st.column_config.NumberColumn(format="%+.1f", help="Team implied-total move"),
        "Proj": st.column_config.NumberColumn(format="%.1f"),
        "Salary": st.column_config.NumberColumn(format="$%d"),
        "Own%": st.column_config.NumberColumn(format="%.1f"),
    })
    st.caption("Totals moving UP = better environment (QB/WR/stacks gain; ownership will chase "
               "with a lag — early-week moves are where leverage hides). Totals CRASHING usually "
               "= injury news; check who's out before rostering anyone in that game.")
