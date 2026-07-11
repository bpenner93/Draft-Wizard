"""
Lineup Builder -- build DK classic lineups from the weekly bundle, in-app.

GPP mode mirrors the validated live construction (profile build: high-total-game
stacks, ~55% bring-back, max-per-game spread, band de-chalk); Cash mode is
max floor-weighted projection. The heavy engines (Monte-Carlo sim solver,
full profile portfolio) still run in the pipeline via optimizer.py -- this is
the interactive/phone tool for the same pool.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from weekly_data import (BUNDLE, load_bundle, players_df,          # noqa: E402
                         slate_options, filter_slate, SLATE_LABEL)
import lineup_opt as LO                                            # noqa: E402

st.set_page_config(page_title="Lineup Builder", page_icon="🧱", layout="wide")


@st.cache_data(show_spinner=False)
def _bundle(mtime: float) -> dict:
    return load_bundle()


if not BUNDLE.exists():
    st.info("No weekly bundle yet — run `python export_weekly_bundle.py --season 2026 --week N`.")
    st.stop()

b = _bundle(BUNDLE.stat().st_mtime)
meta = b.get("meta", {})
st.title(f"🧱 Lineup Builder — {meta.get('season')} Week {meta.get('week')}")

df = players_df(b)
SHOWDOWN = {"sun_night", "mnf", "tnf"}
slates = [s for s in slate_options(b) if s not in SHOWDOWN]

# ── controls ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.2, 1, 1])
slate = c1.selectbox("Slate", slates, format_func=lambda s: SLATE_LABEL.get(s, s))
mode = c2.radio("Contest", ["GPP", "Cash"], horizontal=True,
                help="GPP = ceiling + stacks + de-chalk (the validated profile build). "
                     "Cash = max floor-weighted projection (play big casual double-ups).")
n_lineups = c3.slider("Lineups", 1, 20, 10 if mode == "GPP" else 2)

pool_df = filter_slate(df, slate)
pool_all = pool_df.to_dict("records")
pool, n_dropped = LO.usable(pool_all)
opt_ids = {p["id"]: f"{p['name']} ({p['pos']} {p['team']} ${int(p['salary']):,})"
           for p in sorted(pool, key=lambda x: -(x.get("proj") or 0))}

c4, c5 = st.columns(2)
locks = c4.multiselect("🔒 Lock (in every lineup)", list(opt_ids), format_func=opt_ids.get)
bans = c5.multiselect("🚫 Exclude", list(opt_ids), format_func=opt_ids.get)

with st.expander("⚙️ Advanced knobs"):
    a1, a2, a3 = st.columns(3)
    max_expo = a1.slider("Max exposure (GPP)", 0.2, 1.0, 0.6, 0.05,
                         help="Cap on any player's share of lineups")
    min_diff = a2.slider("Min players different between lineups", 1, 5, 2)
    seed = a3.number_input("Seed (reproducible builds)", 0, 9999, 42)
    a4, a5, a6 = st.columns(3)
    frac_stack2 = a4.slider("QB+2 stack share", 0.0, 1.0, 0.40, 0.05,
                            help="Rest are QB+1 — top-50 winners average ~QB+1.4")
    frac_bring = a5.slider("Bring-back share", 0.0, 1.0, 0.55, 0.05)
    own_lambda = a6.slider("De-chalk strength", 0.0, 0.3, 0.10, 0.01,
                           help="Ceiling penalty per own% above 8% — 0.10 is the validated level; "
                                "overshooting tanked 2025 in testing")
    params = dict(frac_stack2=frac_stack2, frac_bring=frac_bring, own_lambda=own_lambda)

if n_dropped:
    st.caption(f"⚠ {n_dropped} slate players unusable (no salary or projection yet) — they'll "
               f"join once the real DK CSV is imported + re-exported.")

# ── build ─────────────────────────────────────────────────────────────────────
if st.button(f"⚡ Build {n_lineups} {mode} lineup{'s' if n_lineups > 1 else ''}",
             type="primary", width="stretch"):
    with st.spinner("building…"):
        port = LO.build_portfolio(
            pool, n=n_lineups, mode=mode.lower(), locks=locks, bans=bans,
            max_exposure=max_expo, min_diff=min_diff, params=params, seed=int(seed))
    st.session_state["lb_port"] = port
    st.session_state["lb_meta"] = f"{mode} · {SLATE_LABEL.get(slate, slate)} · seed {seed}"

port = st.session_state.get("lb_port", [])
if port:
    st.subheader(f"Lineups — {st.session_state.get('lb_meta', '')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Built", len(port))
    m2.metric("Avg proj", f"{sum(l['proj'] for l in port) / len(port):.1f}")
    m3.metric("Avg own", f"{sum(l['own'] for l in port) / len(port):.0f}%")
    m4.metric("Salary", f"${min(l['salary'] for l in port):,.0f}–{max(l['salary'] for l in port):,.0f}")

    rows = LO.to_dk_rows(port)
    csv = pd.DataFrame(rows, columns=LO.DK_COLUMNS).to_csv(index=False)
    st.download_button("⬇️ Download DK CSV", csv,
                       file_name=f"dk_{meta.get('season')}_wk{meta.get('week'):02d}_{mode.lower()}.csv",
                       mime="text/csv", width="stretch")

    for i, lu in enumerate(port, 1):
        with st.expander(f"#{i} · {lu['stack']} · proj {lu['proj']} · ${lu['salary']:,.0f} · "
                         f"own {lu['own']:.0f}%", expanded=(i <= 2)):
            t = pd.DataFrame([{
                "Slot": p.get("slot", p["pos"]), "Player": p["name"], "Tm": p["team"],
                "Opp": ("" if p.get("home") else "@") + str(p.get("opp") or "?"),
                "Salary": p["salary"], "Proj": p.get("proj"), "Own%": p.get("own"),
            } for p in lu["players"]])
            st.dataframe(t, hide_index=True, column_config={
                "Salary": st.column_config.NumberColumn(format="$%d"),
                "Proj": st.column_config.NumberColumn(format="%.1f"),
                "Own%": st.column_config.NumberColumn(format="%.1f"),
            })

    with st.expander("📊 Portfolio exposure"):
        cnt = Counter((p["name"], p["pos"], p["team"]) for lu in port for p in lu["players"])
        e = pd.DataFrame([{"Player": k[0], "Pos": k[1], "Tm": k[2],
                           "Lineups": v, "Expo%": round(100 * v / len(port))}
                          for k, v in cnt.most_common()])
        st.dataframe(e, hide_index=True, height=360)

    st.caption("Reminders from the backtests: play BIG-field casual GPPs (Millys) — our contrarian "
               "build fits big spread fields, not small sharky ones; cash edge = soft double-ups "
               "3–15k entries ≤13% rake. Late swap (multi-entry only) runs in the pipeline: "
               "`python late_swap.py --live`.")

# ── manual builder ────────────────────────────────────────────────────────────
st.divider()
st.subheader("✍️ Manual build / auto-complete")
sel = st.multiselect("Your picks (up to 9)", list(opt_ids), format_func=opt_ids.get,
                     max_selections=9, key="manual_sel")
if sel:
    chosen = [p for p in pool if p["id"] in sel]
    sal = sum(p["salary"] for p in chosen)
    proj = sum(p.get("proj") or 0 for p in chosen)
    errs = LO.validate_lineup(chosen, partial=len(chosen) < 9)
    c1, c2, c3 = st.columns(3)
    c1.metric("Picked", f"{len(chosen)}/9")
    c2.metric("Salary", f"${sal:,.0f}", delta=f"${LO.SALARY_CAP - sal:,.0f} left")
    c3.metric("Proj", f"{proj:.1f}")
    if errs:
        st.warning(" · ".join(errs))
    if len(chosen) < 9 and not errs:
        if st.button("🪄 Auto-complete this lineup", width="stretch"):
            lu = LO.build_lineup([p for p in pool if p["id"] not in set(bans)],
                                 mode=mode.lower(), locks=sel, params=params)
            if lu is None:
                st.error("Couldn't complete under the cap — free up salary or drop a pick.")
            else:
                lu = LO.sort_slots(lu)
                t = pd.DataFrame([{"Slot": p["slot"], "Player": p["name"], "Tm": p["team"],
                                   "Salary": p["salary"], "Proj": p.get("proj"), "Own%": p.get("own")}
                                  for p in lu])
                st.dataframe(t, hide_index=True, column_config={
                    "Salary": st.column_config.NumberColumn(format="$%d"),
                    "Proj": st.column_config.NumberColumn(format="%.1f"),
                    "Own%": st.column_config.NumberColumn(format="%.1f")})
                st.success(f"proj {sum(p.get('proj') or 0 for p in lu):.1f} · "
                           f"${sum(p['salary'] for p in lu):,.0f} · "
                           f"own {sum(p.get('own') or 0 for p in lu):.0f}%")
