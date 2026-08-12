"""
The Solver -- Monte-Carlo DFS lineup solver, in-app.

Simulates correlated player outcomes and a realistic opponent field, generates
hundreds of candidate lineups across construction archetypes, and ranks them by
simulated equity against a real contest's payout curve -- then lets you browse
EVERY candidate, not just the portfolio it picked.

Reading the numbers (this matters):
  * CASH ROI is a real number. The simulated cash arm independently reproduces
    the measured real-field result (+22% sim vs +23% measured, 63% vs 62% cash
    rate over three seasons).
  * GPP ROI is deliberately NOT shown. The simulated field is built from our own
    projections, so it can't disagree with us the way the real field does, and a
    top-heavy payout is convex in rank -- the sim reports thousands of percent
    where the measured reality is -12% to -63% by tier. The RANKING is still
    good (the bias is common to every candidate), so GPP lineups are ranked by
    simulated equity and the measured tier ROI is shown as the anchor.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from weekly_data import (BUNDLE, load_bundle, players_df, games_df,      # noqa: E402
                         slate_options, filter_slate, SLATE_LABEL)
import dfs_solver as S                                                   # noqa: E402
import contests as C                                                     # noqa: E402

st.set_page_config(page_title="The Solver", page_icon="🧮", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def _bundle(mtime: float) -> dict:
    return load_bundle()


@st.cache_data(show_spinner="fetching the DK lobby…", ttl=600)
def _lobby() -> pd.DataFrame:
    return C.fetch_dk_lobby("NFL")


if not BUNDLE.exists():
    st.info("No weekly bundle yet — run `python export_weekly_bundle.py --season 2026 --week N`.")
    st.stop()

b = _bundle(BUNDLE.stat().st_mtime)
meta = b.get("meta", {})
df = players_df(b)
games = games_df(b)

st.title(f"🧮 The Solver — {meta.get('season')} Week {meta.get('week')}")
st.caption(f"built {str(meta.get('generated_at',''))[:16]} · "
           f"odds {str(meta.get('odds_asof') or 'schedule lines')[:16]}")

# ── slate / format ────────────────────────────────────────────────────────────
SHOWDOWN_SLATES = {"sun_night", "mnf", "tnf"}
all_slates = slate_options(b)
c1, c2, c3 = st.columns([1.5, 1.2, 1])
fmt = c1.radio("Format", ["Classic", "Showdown"], horizontal=True,
               help="Classic = 9 roster spots across a multi-game slate. "
                    "Showdown = 1 CPT (1.5x points AND 1.5x salary) + 5 FLEX from ONE game.")
showdown = fmt == "Showdown"

if showdown:
    sd = [s for s in all_slates if s in SHOWDOWN_SLATES]
    slate = c2.selectbox("Game", sd or all_slates, format_func=lambda s: SLATE_LABEL.get(s, s))
else:
    cl = [s for s in all_slates if s not in SHOWDOWN_SLATES]
    slate = c2.selectbox("Slate", cl or all_slates, format_func=lambda s: SLATE_LABEL.get(s, s))

slate_recs = filter_slate(df, slate).to_dict("records")

# ── real DK salaries (optional but this is what makes it a live tool) ─────────
with st.expander("💲 Import the real DK salary file  (recommended for a live slate)"):
    st.caption("The bundle ships pipeline salaries, which lag the real slate and are "
               "absent preseason. Drop in DraftKings' **DKSalaries.csv** to re-price the "
               "pool — everything below then solves on the prices you'll actually pay.")
    up = st.file_uploader("DKSalaries.csv", type=["csv"], key="dk_sal")
    if up is not None:
        try:
            dk = C.parse_dk_salaries(up)
            slate_recs, rep = C.apply_dk_salaries(slate_recs, dk)
            st.success(f"Re-priced from DK: matched {rep['matched']}, "
                       f"unmatched {rep['unmatched']} (dropped), {rep['dk_rows']} DK rows.")
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")

rep = S.pool_report(slate_recs, showdown=showdown)
pool = S.prepare_pool(slate_recs, showdown=showdown)
if pool.empty or len(pool) < 12:
    st.warning(f"Only {len(pool)} usable players on this slate "
               f"(no salary: {rep['no_salary']}, no projection: {rep['no_proj']}). "
               "Import the DK salary file above, or pick another slate.")
    st.stop()

if showdown and pool["game_id"].nunique() > 1:
    gsel = c3.selectbox("Which game", sorted(pool["game_id"].unique()),
                        format_func=lambda g: g.replace("@", " vs "))
    pool = pool[pool["game_id"] == gsel].reset_index(drop=True)

if showdown:
    reach = S.sd_min_salary(pool["salary"].to_numpy()) / 0.90
    if reach < 0.92 * S.SALARY_CAP and st.session_state.get("dk_sal") is None:
        st.warning(
            f"**Showdown is running on CLASSIC salaries.** DK prices showdown slates on "
            f"their own scale so six players fill the cap; the best six here reach only "
            f"about **${reach:,.0f}** of $50,000, so the salary cap is not binding and the "
            f"solve reduces to picking the highest projections. Import the slate's real "
            f"**DKSalaries.csv** above to make showdown meaningful.")

# ── contest ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🎟️ Contest")
src = st.radio("Menu", ["Live DK lobby", "Tier presets"], horizontal=True,
               help="The live lobby is DraftKings' public contest feed — the actual "
                    "contests running, with their real field size, fee and prize pool.")
menu = _lobby() if src == "Live DK lobby" else C.preset_frame()
if src == "Live DK lobby" and menu.empty:
    st.info("DK lobby returned nothing (out of season, or the feed is unreachable) — "
            "using tier presets.")
    menu = C.preset_frame()

kind_f = st.radio("Type", ["Cash (double-up / 50-50)", "GPP (tournament)"], horizontal=True)
want = "cash" if kind_f.startswith("Cash") else "gpp"
sub = menu[menu["kind"] == want]
if sub.empty:
    st.warning(f"No {want} contests in this menu.")
    st.stop()
row = sub.iloc[st.selectbox("Contest", range(len(sub)),
                            format_func=lambda i: C.label(sub.iloc[i].to_dict()))]
contest = C.to_contest(row.to_dict())

# ── knobs ─────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
max_entries = max(1, int(contest.get("entries") or 1))
if max_entries <= 1:
    # SINGLE-ENTRY: no slider (Streamlit rejects min == max), and the cap is the
    # contest's, not a preference. The measured build rule for one entry is
    # max-projection, NOT the contrarian portfolio: -9% vs -54% ROI over 42
    # slates. A portfolio exists to cover outcomes across many entries; with one
    # entry you want the single highest-EV lineup.
    n_port = 1
    k1.metric("Lineups to build", "1")
    k1.caption("single-entry contest")
else:
    n_port = k1.slider("Lineups to build", 1, min(150, max_entries),
                       min(20, max_entries))
n_cand = k2.select_slider("Candidate lineups", [300, 600, 1000, 1500, 2500], value=600,
                          help="How many constructions to simulate and rank. More = a "
                               "richer board to browse, slower solve.")
n_sims = k3.select_slider("Simulations", [500, 1000, 2000, 4000], value=1000)
seed = k4.number_input("Seed", 0, 9999, 7)

ids = {p["id"]: f"{p['name']} ({p['position']} {p['team']} ${int(p['salary']):,})"
       for _, p in pool.sort_values("proj", ascending=False).iterrows()}
l1, l2 = st.columns(2)
locks = l1.multiselect("🔒 Lock into every lineup", list(ids), format_func=ids.get)
bans = l2.multiselect("🚫 Exclude", list(ids), format_func=ids.get)

with st.expander("⚙️ Advanced"):
    a1, a2, a3 = st.columns(3)
    max_expo = a1.slider("Max player exposure", 0.1, 1.0, 0.6, 0.05,
                         help="Cap any one player's share of the portfolio, so a single "
                              "bust can't sink every entry.")
    use_band = a2.toggle("Target the ownership band", value=not showdown,
                         help="Measured over 351 real contests / 31.8M entries, realized "
                              "GPP ROI by cumulative ownership: under 100 → −48.7%, "
                              "120-140 → −12.1%, 140-160 → +5.4%, 180+ → −1.3% (duplication "
                              "eats the top: 6.97 mean copies per lineup above 180 own in a "
                              "100k field, vs 1.05 below 100). The solver now TUNES a chalk "
                              "bonus toward this band rather than filtering candidates to "
                              "it — the old filter no-opped, because only 17 of 600 "
                              "candidates could reach it. It reports what it achieved.")
    band = a2.slider("Cumulative ownership band", 50, 250, S.OWN_BAND_DEFAULT,
                     disabled=not use_band)
    se_gpp = (max_entries <= 1 and contest["kind"] == "gpp")
    no_stack = a3.toggle("Max-proj build (no forced stack)", value=se_gpp,
                         help="Single-entry tournaments want MAX-PROJECTION, not the "
                              "contrarian stacked portfolio: -9% vs -54% ROI over 42 "
                              "slates. A portfolio exists to cover outcomes across many "
                              "entries; with one entry you want the single highest-EV "
                              "lineup, and a forced QB stack wrecks its floor.")
    corr = a3.selectbox("Correlation model", ["matrix", "factor", "tcopula"], index=0,
                        help="matrix (default) = the measured DK pair structure. "
                             "factor was the old default and is FALSIFIED: it puts the "
                             "same 0.607 correlation on every same-team pair, where the "
                             "realized values are QB-WR 0.33 and QB-TE 0.32 but WR-WR "
                             "0.01 and RB-RB −0.04 — team co-movement is only the "
                             "QB↔pass-catcher link. Against 311 real fields, matrix "
                             "reproduces the field's score dispersion (sd/mean 0.200 vs "
                             "0.190) while factor over-disperses by 23%. tcopula adds "
                             "GLOBAL tail dependence, which over-corrects every non-QB "
                             "pair; it also needs scipy, absent on Streamlit Cloud, where "
                             "it silently falls back to matrix.")

if st.button("⚡ Solve", type="primary", width="stretch"):
    with st.spinner(f"simulating {n_sims:,} outcomes across {n_cand:,} constructions…"):
        res = S.solve(pool, contest, n_port=int(n_port), n_sims=int(n_sims),
                      n_cand=int(n_cand), field_sim=1500, seed=int(seed), corr=corr,
                      locks=locks, bans=bans, max_exposure=float(max_expo),
                      own_band=tuple(band) if (use_band and not showdown and not no_stack) else None,
                      showdown=showdown, no_stack=no_stack)
    st.session_state["solved"] = res
    # escape $ -- Streamlit markdown reads $...$ as LaTeX, and DK contest names are
    # full of them ("$3.5M Millionaire [$1M to 1st]" renders as maths otherwise)
    st.session_state["solved_label"] = (
        f"{contest['label']} · {SLATE_LABEL.get(slate, slate)} · seed {seed}"
    ).replace("$", r"\$")

res = st.session_state.get("solved")
if not res:
    st.info("Set the contest and hit **Solve**.")
    st.stop()
if "error" in res:
    st.error(res["error"])
    st.stop()

# ── headline ──────────────────────────────────────────────────────────────────
st.divider()
met = S.display_metrics(res)
chosen = res["chosen"]
sel = met.iloc[chosen]
tier_roi, tier_note = S.tier_note(res)
cash_mode = S.roi_is_meaningful(res["contest"]["kind"], res.get("showdown", False))
anchor = (f"Measured real-field ROI for this tier: **{tier_roi:+d}%** ({tier_note})."
          if tier_roi is not None else
          f"**No measured baseline exists for this format** ({tier_note}) — treat every "
          f"number here as a relative ranking only.")

st.subheader(f"Portfolio — {st.session_state.get('solved_label','')}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Lineups", len(chosen))
m2.metric("Median score", f"{sel['med'].mean():.1f}")
m3.metric("Ceiling (90th)", f"{sel['p90'].mean():.1f}")
m4.metric("Cum. ownership", f"{sel['own'].mean():.0f}%")
if cash_mode:
    m5.metric("Sim ROI", f"{sel['roi'].mean():+.0f}%", f"cash {sel['p_cash'].mean():.0f}%")
else:
    m5.metric("Top-1% rate", f"{sel['p_top1'].mean():.2f}%")

if cash_mode:
    st.success(
        f"**Cash ROI is a real number here.** The simulated cash arm reproduces the measured "
        f"real-field result — sim **+22%** vs measured **+23%**, 63% vs 62% cash rate over "
        f"three seasons. {anchor}")
elif res.get("showdown"):
    st.warning(
        f"**Absolute ROI is withheld for showdown.** The cash validation was measured on "
        f"classic main slates; showdown has no backtest here, and with ~40 players in one game "
        f"the simulated field is far too self-similar to price (it will happily report a 100% "
        f"cash rate). Lineups are ranked by simulated equity — use the ordering, not the "
        f"levels. {anchor}")
else:
    st.warning(
        f"**Absolute GPP ROI is withheld on purpose.** The simulated field is built from our "
        f"own projections, so it can't disagree with us the way the real field does, and a "
        f"top-heavy payout is convex in rank — the sim reports ROI in the thousands of percent. "
        f"The **ranking** is still informative (every candidate carries the same bias), so "
        f"lineups are sorted by simulated equity. {anchor}")

note = res.get("band_note")
if note and note.startswith("NOT"):
    st.info(f"ℹ️ Ownership band **{note}** — so the portfolio was NOT steered toward the "
            f"winning ownership range. That is expected on a preseason bundle (ownership is "
            f"projected off our engine number with no public market yet); in-season, widen "
            f"the band or check the DFS Ownership page.")
st.caption(f"{res.get('n_cand_built', 0):,} candidate lineups simulated"
           + (f" · own band {note}" if note else ""))

tabs = st.tabs(["🏆 Portfolio", "🗂️ All lineups", "🏟️ Best games", "🔗 Best stacks", "📊 Exposure"])

MCOLS = {"equity": "Equity", "p_cash": "Cash%", "p_top1": "Top1%", "med": "Med",
         "p90": "Ceil", "own": "Own", "salary": "Salary", "dup_risk": "Dup"}
CFG = {
    "Equity": st.column_config.NumberColumn(format="%.0f", help="Percentile of simulated expected value among all candidates — the valid comparison"),
    "Cash%": st.column_config.NumberColumn(format="%.1f", help="Share of simulations finishing in the money"),
    "Top1%": st.column_config.NumberColumn(format="%.2f", help="Share of simulations finishing in the top 1% of the field"),
    "Med": st.column_config.NumberColumn(format="%.1f", help="Median simulated lineup score"),
    "Ceil": st.column_config.NumberColumn(format="%.1f", help="90th-percentile simulated score"),
    "Own": st.column_config.NumberColumn(format="%.0f", help="Cumulative projected ownership"),
    "Salary": st.column_config.NumberColumn(format="$%d"),
    "Dup": st.column_config.NumberColumn(format="%.0f", help="Duplication-risk percentile vs the other candidates (relative, not a count)"),
    "ROI%": st.column_config.NumberColumn(format="%+.0f"),
}


def _table(rows: list[int]) -> pd.DataFrame:
    t = met.iloc[rows].copy()
    t.insert(0, "Build", [S.stack_label(res, i) for i in rows])
    t.insert(0, "#", [int(i) for i in rows])
    keep = ["#", "Build"] + list(MCOLS)
    t = t[keep].rename(columns=MCOLS)
    if cash_mode:
        t["ROI%"] = met.iloc[rows]["roi"].to_numpy()
    return t


# ── portfolio ─────────────────────────────────────────────────────────────────
with tabs[0]:
    st.dataframe(_table(chosen), hide_index=True, column_config=CFG, height=min(420, 60 + 35 * len(chosen)))
    csv = S.to_dk_rows(res, chosen).to_csv(index=False)
    st.download_button("⬇️ Download DK upload CSV", csv,
                       file_name=f"dk_{meta.get('season')}_wk{meta.get('week'):02d}_"
                                 f"{'sd' if res['showdown'] else 'classic'}.csv",
                       mime="text/csv", width="stretch")
    st.caption("Open a lineup to see the roster, or switch to **All lineups** to browse "
               "every construction the solver simulated.")
    for n, ci in enumerate(chosen, 1):
        r = met.iloc[ci]
        with st.expander(f"#{n} · {S.stack_label(res, ci)} · ceil {r['p90']:.1f} · "
                         f"${r['salary']:,.0f} · own {r['own']:.0f}%", expanded=(n == 1)):
            t = S.lineup_rows(res, ci)
            t["Opp"] = np.where(t["home"], "", "@") + t["opp"].astype(str)
            show = t[["slot", "name", "position", "team", "Opp", "salary", "proj",
                      "est_ownership"]].rename(columns={
                "slot": "Slot", "name": "Player", "position": "P", "team": "Tm",
                "salary": "Salary", "proj": "Proj", "est_ownership": "Own%"})
            st.dataframe(show, hide_index=True, column_config={
                "Salary": st.column_config.NumberColumn(format="$%d"),
                "Proj": st.column_config.NumberColumn(format="%.1f"),
                "Own%": st.column_config.NumberColumn(format="%.1f")})

# ── every candidate ───────────────────────────────────────────────────────────
with tabs[1]:
    st.caption("Every construction the solver simulated, not just the ones it picked. "
               "Sort by whatever you're hunting — ceiling for a milly, cash% for a "
               "double-up, ownership for leverage.")
    f1, f2, f3 = st.columns([1.2, 1, 1])
    sort_by = f1.selectbox("Sort by", ["Equity", "Ceil", "Top1%", "Cash%", "Med",
                                       "Own (low→high)", "Dup (low→high)"])
    own_max = f2.slider("Max cumulative own", 0, 300, 300, 5)
    n_show = f3.slider("Show", 20, 500, 100, 20)

    view = met.copy()
    view = view[view["own"] <= own_max]
    key, asc = {"Equity": ("equity", False), "Ceil": ("p90", False), "Top1%": ("p_top1", False),
                "Cash%": ("p_cash", False), "Med": ("med", False),
                "Own (low→high)": ("own", True), "Dup (low→high)": ("dup_risk", True)}[sort_by]
    rows = list(view.sort_values(key, ascending=asc).head(int(n_show)).index)
    st.dataframe(_table(rows), hide_index=True, column_config=CFG, height=560)

    pick = st.selectbox("Inspect lineup #", rows, format_func=lambda i: f"#{i} · {S.stack_label(res, i)}")
    if pick is not None:
        t = S.lineup_rows(res, int(pick))
        t["Opp"] = np.where(t["home"], "", "@") + t["opp"].astype(str)
        st.dataframe(t[["slot", "name", "position", "team", "Opp", "salary", "proj",
                        "est_ownership"]].rename(columns={
            "slot": "Slot", "name": "Player", "position": "P", "team": "Tm",
            "salary": "Salary", "proj": "Proj", "est_ownership": "Own%"}),
            hide_index=True, column_config={
                "Salary": st.column_config.NumberColumn(format="$%d"),
                "Proj": st.column_config.NumberColumn(format="%.1f"),
                "Own%": st.column_config.NumberColumn(format="%.1f")})
        st.download_button("⬇️ Download these lineups as DK CSV",
                           S.to_dk_rows(res, rows).to_csv(index=False),
                           file_name="dk_candidates.csv", mime="text/csv")

# ── games ─────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.caption("Ceiling vs the field's attention. **ceil85** is the 85th-percentile "
               "simulated total of the game's best 5 DFS players — how much fantasy it "
               "can produce when it goes right. **field%** is the share of simulated "
               "opponent lineups stacking it. The game you want is high ceiling, low field%.")
    gb = S.game_board(res["pool"], res["scores"], res["field_bin"], showdown=res["showdown"])
    st.dataframe(gb.rename(columns={"game": "Game", "total": "Vegas", "imp": "Implied",
                                    "ceil85": "Ceil85", "median": "Median", "own": "Own",
                                    "field%": "Field%", "lev": "Lev", "n": "Players"}),
                 hide_index=True, height=440, column_config={
        "Vegas": st.column_config.NumberColumn(format="%.1f", help="Vegas game total"),
        "Ceil85": st.column_config.NumberColumn(format="%.1f"),
        "Median": st.column_config.NumberColumn(format="%.1f"),
        "Own": st.column_config.NumberColumn(format="%.0f", help="Summed projected ownership of the game's players"),
        "Field%": st.column_config.NumberColumn(format="%.1f"),
        "Lev": st.column_config.NumberColumn(format="%+d", help="Ceiling rank minus field-attention rank. Positive = an under-stacked ceiling game")})

# ── stacks ────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.caption("QB + 1 or 2 same-team pass-catchers, each with and without an opponent "
               "bring-back, scored on the same simulation draws as everything else. "
               "**Lev** is ceiling percentile minus ownership percentile — what the field "
               "is under-rostering relative to what it can actually produce.")
    only_bb = st.toggle("Bring-back only", value=False)
    sb = S.stack_board(res["pool"], res["scores"], top_n=200)
    if only_bb and not sb.empty:
        sb = sb[sb["bring"]]
    sort_stack = st.radio("Sort", ["Ceiling", "Leverage", "Projection"], horizontal=True)
    sb = sb.sort_values({"Ceiling": "ceil85", "Leverage": "lev",
                         "Projection": "proj"}[sort_stack], ascending=False)
    st.dataframe(sb[["stack", "n", "salary", "proj", "ceil85", "median", "own", "lev"]].head(80)
                 .rename(columns={"stack": "Stack", "n": "Size", "salary": "Salary",
                                  "proj": "Proj", "ceil85": "Ceil85", "median": "Median",
                                  "own": "Own", "lev": "Lev"}),
                 hide_index=True, height=520, column_config={
        "Salary": st.column_config.NumberColumn(format="$%d"),
        "Proj": st.column_config.NumberColumn(format="%.1f"),
        "Ceil85": st.column_config.NumberColumn(format="%.1f"),
        "Median": st.column_config.NumberColumn(format="%.1f"),
        "Own": st.column_config.NumberColumn(format="%.1f", help="Cumulative projected ownership of the stack"),
        "Lev": st.column_config.NumberColumn(format="%+d")})

# ── exposure ──────────────────────────────────────────────────────────────────
with tabs[4]:
    ex = S.exposure_table(res, chosen)
    st.dataframe(ex.rename(columns={"name": "Player", "position": "P", "team": "Tm",
                                    "salary": "Salary", "proj": "Proj",
                                    "est_ownership": "Field own%", "lineups": "Lineups",
                                    "expo%": "Expo%", "cpt": "CPT"}),
                 hide_index=True, height=520, column_config={
        "Salary": st.column_config.NumberColumn(format="$%d"),
        "Proj": st.column_config.NumberColumn(format="%.1f"),
        "Field own%": st.column_config.NumberColumn(format="%.1f"),
        "Expo%": st.column_config.NumberColumn(format="%.0f")})
    st.caption("Your exposure vs the field's projected ownership. Being heavier than the "
               "field on a high-ceiling player is the leverage; being heavier on chalk is "
               "just paying for the field's opinion.")
