"""
ZAP Prospects -- side-by-side prospect comparison cards, built on zap_engine.

Reads data/processed/zap_scores.parquet (produced by `python zap_engine.py`).

⚠ READ THE SCORE CORRECTLY. `zap` is draft capital PLUS whatever college adds on
top of it, and out of sample college only adds anything for RUNNING BACKS
(bootstrap 88%). For WR / TE / QB the lambda is at or near zero, which means the
ZAP column there is essentially the draft pick re-expressed as a percentile. It
is a browsing and comparison tool for those positions, not an independent
opinion. `zap_prod` IS the independent opinion -- it never sees the pick -- and
it is measurably WORSE than the pick at ranking outcomes. Both are shown on
purpose: the gap between them is the interesting part.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPO = ROOT.parent

st.set_page_config(page_title="ZAP Prospects", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

# ⚠ SHIPPED JSON FIRST. draft_wizard/ deploys to Streamlit Cloud as a standalone
# repo -- data/processed/zap_scores.parquet does NOT exist there. Read the baked
# file written by export_wizard_data.py, and fall back to the parquet only for
# local development inside the full repo.
SHIPPED = ROOT / "data" / "zap_scores.json"
PARQUET = REPO / "data" / "processed" / "zap_scores.parquet"


@st.cache_data(show_spinner=False)
def load(shipped: str, parquet: str, key: float) -> pd.DataFrame:
    if shipped and Path(shipped).exists():
        blob = json.loads(Path(shipped).read_text(encoding="utf-8"))
        p = blob["players"]                       # columnar: {"cols":[], "rows":[[]]}
        return pd.DataFrame(p["rows"], columns=p["cols"])
    return pd.read_parquet(parquet)


_src = SHIPPED if SHIPPED.exists() else PARQUET
if not _src.exists():
    st.info("No ZAP scores yet — run `python zap_engine.py` then "
            "`python export_wizard_data.py`.")
    st.stop()

df = load(str(SHIPPED) if SHIPPED.exists() else "", str(PARQUET), _src.stat().st_mtime)

# --------------------------------------------------------------------------
# panel definitions.  (column, label, format, note)
# "note" names the PFF metric a free column is standing in for, where it is
# standing in for one -- so the card never implies we have charting data.
# --------------------------------------------------------------------------
MODEL_PANEL = {
    "WR": [("age_at_draft", "Age at draft", "{:.1f}", ""),
           ("yrs_college", "Years in college", "{:.0f}",
            "3 or fewer = declared with eligibility left. Context only: it LOST the gate to age_at_draft, which carries it better"),
           ("breakout_age", "Breakout age", "{:.1f}", ""),
           ("height_in", "Height (in)", "{:.0f}", ""),
           ("weight", "Weight (lbs)", "{:.0f}", ""),
           ("bmi", "BMI", "{:.1f}", ""),
           ("forty", "40 time", "{:.2f}", ""),
           ("speed_score", "Speed score", "{:.1f}", ""),
           ("burst_score", "Burst score", "{:.1f}", ""),
           ("recruit_rating", "247 composite", "{:.4f}", "")],
    "RB": [("age_at_draft", "Age at draft", "{:.1f}", ""),
           ("yrs_college", "Years in college", "{:.0f}",
            "3 or fewer = declared with eligibility left. Context only: it LOST the gate to age_at_draft, which carries it better"),
           ("breakout_age", "Breakout age", "{:.1f}", ""),
           ("height_in", "Height (in)", "{:.0f}", ""),
           ("weight", "Weight (lbs)", "{:.0f}", ""),
           ("bmi", "BMI", "{:.1f}", ""),
           ("forty", "40 time", "{:.2f}", ""),
           ("speed_score", "Speed score", "{:.1f}", ""),
           ("agility_score", "Agility score", "{:.2f}", ""),
           ("recruit_rating", "247 composite", "{:.4f}", "")],
    "QB": [("age_at_draft", "Age at draft", "{:.1f}", ""),
           ("yrs_college", "Years in college", "{:.0f}",
            "3 or fewer = declared with eligibility left. Context only: it LOST the gate to age_at_draft, which carries it better"),
           ("height_in", "Height (in)", "{:.0f}", ""),
           ("weight", "Weight (lbs)", "{:.0f}", ""),
           ("recruit_rating", "247 composite", "{:.4f}", "")],
}
MODEL_PANEL["TE"] = MODEL_PANEL["WR"]

ADV_PANEL = {
    "WR": [("best_dominator", "Best dominator", "{:.3f}", ""),
           ("career_dominator", "Career dominator", "{:.3f}", ""),
           ("best_recyd_per_tmpa", "Best rec yds / team pass att", "{:.2f}",
            "the model actually uses THIS, not the PFF YPRR below -- see zap_engine"),
           ("best_rec_share", "Best reception share", "{:.3f}",
            "stand-in for target share — see note below"),
           ("final_recyd_share", "Final yds share", "{:.3f}", ""),
           ("best_rectd_share", "Best rec TD share", "{:.3f}", ""),
           ("final_reci_ypr", "Yards / reception", "{:.1f}", ""),
           # ADVANCED DATA, ordered to match the real card. `max` rows are the
           # best SEASON by rate (100-route floor); `career` rows are
           # route-weighted so a cameo cannot drag them.
           ("pff_max_yprr", "Max yards / route run", "{:.2f}", "PFF, best season"),
           ("pff_car_yprr", "Career yards / route run", "{:.2f}", "PFF, real routes"),
           ("pff_car_fdpr", "First downs / route run", "{:.3f}", "PFF"),
           ("pff_car_tprr", "Targets / route run", "{:.3f}", "PFF"),
           ("pff_car_slot", "Slot rate", "{:.1f}%", "PFF, share of pass plays"),
           ("pff_max_tgt_share", "Max target share", "{:.1f}%",
            "PFF; denominator is targets to route-runners"),
           ("pff_car_ctgt", "Contested targets", "{:.2f}", "PFF, share of targets"),
           ("pff_car_ccr", "Contested catch rate", "{:.1f}%", "PFF"),
           ("pff_car_yacpr", "YAC / reception", "{:.1f}", "PFF"),
           ("pff_car_avoided", "Avoided tackles / rec", "{:.2f}", "PFF"),
           ("pff_car_adot", "Career aDOT", "{:.1f}", "PFF"),
           ("pff_car_grade_route", "PFF route grade", "{:.1f}", "PFF"),
           ("tm_pass_att_final", "Team pass attempts", "{:.0f}",
            "context for every per-team-pass-att rate above -- a low-volume "
            "offence inflates them"),
           ("tmate_routes_peak", "Teammate routes (peak yr)", "{:.0f}",
            "how much competition for targets he faced -- a WR model feature"),
           ("tmate_best_grade_peak", "Best teammate route grade", "{:.1f}",
            "context only: explains a suppressed target share, but it LOST the "
            "gate as a feature"),
           ("tmate_n_good_peak", "Teammates grading 75+", "{:.0f}", "context only"),
           ("best_fd_per_tmpa", "1st downs / team pass att", "{:.3f}",
            "free fallback -- covers the ~15% PFF's FBS pull misses"),
           ("best_explosive_per_tmpa", "20+ yd catches / team pass att", "{:.3f}", ""),
           ("final_rz_rec_per_tmpa", "Red-zone catches / team pass att", "{:.3f}", ""),
           ("best_ppa_avg_all", "EPA per play", "{:.3f}", ""),
           ("tot_reci_yds", "Career rec yards", "{:.0f}", ""),
           ("tot_reci_td", "Career rec TD", "{:.0f}", "")],
    "RB": [("best_scrim_per_tmplay", "Scrimmage yds / team play", "{:.2f}",
            "= Adj Yards/Team Play"),
           ("best_ppr_per_tmgame", "Best PPR / team game", "{:.1f}", "= Max Season PPR"),
           ("best_rush_share", "Best carry share", "{:.3f}", ""),
           ("best_rushyd_share", "Best rush yds share", "{:.3f}", ""),
           ("best_rushtd_share", "Best rush TD share", "{:.3f}", ""),
           ("career_rush_ypc", "Career YPC", "{:.2f}", ""),
           ("best_rec_share", "Best reception share", "{:.3f}",
            "⭐ the one metric that beats the pick"),
           ("final_recyd_per_tmpa", "Rec yds / team pass att", "{:.2f}", ""),
           ("tm_pass_att_final", "Team pass attempts", "{:.0f}",
            "context for the rate above -- an option offence throws ~150 where "
            "the median is 399, which inflates it"),
           ("final_usage_third_down", "3rd-down usage", "{:.3f}", ""),
           ("best_ppa_avg_all", "EPA per play", "{:.3f}", ""),
           ("tot_rush_yds", "Career rush yards", "{:.0f}", ""),
           ("tot_reci_rec", "Career receptions", "{:.0f}", "")],
    "QB": [("best_qb_ppr_per_tmgame", "Best fantasy pts / team game", "{:.1f}", ""),
           ("best_qb_ypa", "Best Y/A", "{:.2f}", ""),
           ("career_qb_ypa", "Career Y/A", "{:.2f}", ""),
           ("best_qb_td_rate", "TD rate", "{:.3f}", ""),
           ("final_qb_int_rate", "INT rate", "{:.3f}", ""),
           ("career_pass_pct", "Completion %", "{:.1%}", ""),
           ("best_qb_rush_per_tmgame", "Best rush yds / team game", "{:.1f}",
            "⭐ strongest QB signal vs the pick"),
           ("career_qb_rush_per_tmgame", "Career rush yds / team game", "{:.1f}", ""),
           ("best_ppa_avg_all", "EPA per play", "{:.3f}", ""),
           ("tot_pass_yds", "Career pass yards", "{:.0f}", ""),
           ("tot_rush_td", "Career rush TD", "{:.0f}", "")],
}
ADV_PANEL["TE"] = ADV_PANEL["WR"]


def bar_html(pct: float) -> str:
    if pd.isna(pct):
        return '<div class="zbar"><div class="zfill" style="width:0%"></div></div>'
    p = float(np.clip(pct, 0, 100))
    color = "#2fa84f" if p >= 75 else "#7ab8ff" if p >= 50 else "#f0ad4e" if p >= 25 else "#d9534f"
    return (f'<div class="zbar"><div class="zfill" '
            f'style="width:{p:.0f}%;background:{color}"></div></div>')


CSS = """
<style>
.zcard{border:2px solid #262626;border-radius:6px;padding:14px 18px;margin-bottom:14px;
       box-shadow:4px 4px 0 #262626;}
.zname{font-size:30px;font-weight:800;letter-spacing:-.5px;line-height:1.1;}
.zmeta{font-size:13px;opacity:.75;margin-top:2px;}
.zscore{font-size:44px;font-weight:800;line-height:1;text-align:right;}
.zscorelab{font-size:10px;letter-spacing:2px;opacity:.6;text-align:right;}
.ztag{display:inline-block;border:1px solid #262626;border-radius:3px;padding:2px 8px;
      font-size:11px;letter-spacing:1px;margin:6px 6px 0 0;}
.zrow{display:flex;align-items:center;gap:10px;font-size:13px;padding:3px 0;}
.zlab{flex:0 0 46%;}
.zval{flex:0 0 16%;text-align:right;font-variant-numeric:tabular-nums;}
.zbar{flex:1;height:7px;background:rgba(128,128,128,.22);border-radius:4px;overflow:hidden;}
.zfill{height:100%;}
.zpct{flex:0 0 40px;text-align:right;font-size:11px;opacity:.7;}
.znote{font-size:10px;opacity:.55;margin:-2px 0 4px 0;}
.zsec{font-size:11px;letter-spacing:2px;font-weight:700;opacity:.7;
      border-bottom:1px solid rgba(128,128,128,.35);padding-bottom:4px;margin:14px 0 8px 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("🎓 ZAP Prospects")

# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------
c1, c2, c3 = st.columns([1, 1, 1])
with c3:
    classes = sorted(df["class"].dropna().unique().astype(int), reverse=True)
    cls_filter = st.multiselect("Draft classes", classes, default=classes[:3])
    pos_filter = st.multiselect("Positions", ["QB", "RB", "WR", "TE"],
                                default=["QB", "RB", "WR", "TE"])

pool = df[df["class"].isin(cls_filter) & df["position"].isin(pos_filter)].copy()
pool = pool.sort_values("zap", ascending=False)
labels = [f"{r.player_name} ({r.position}, {int(r['class'])}, pk {int(r.pick)})"
          for _, r in pool.iterrows()]
lut = dict(zip(labels, pool.index))

with c1:
    p1 = st.selectbox("Player 1", labels, index=0 if labels else None)
with c2:
    p2 = st.selectbox("Player 2", labels, index=min(1, len(labels) - 1) if labels else None)


def render(idx) -> str:
    r = df.loc[idx]
    pos = r["position"]
    zap = r.get("zap")
    dcd = r.get("dcd")
    lam = r.get("lambda_used")

    h = ['<div class="zcard">']
    h.append('<div style="display:flex;justify-content:space-between;align-items:flex-start">')
    h.append(f'<div><div class="zname">{r["player_name"].upper()}</div>'
             f'<div class="zmeta">{int(r["class"])} &nbsp;·&nbsp; '
             f'{r.get("final_team") or r.get("college") or "—"} &nbsp;·&nbsp; '
             f'Pick {int(r["pick"])} &nbsp;·&nbsp; '
             f'Age {r["age_at_draft"]:.1f}</div></div>' if pd.notna(r.get("age_at_draft"))
             else f'<div><div class="zname">{r["player_name"].upper()}</div></div>')
    h.append(f'<div><div class="zscorelab">ZAP SCORE</div>'
             f'<div class="zscore">{zap:.1f}</div></div>' if pd.notna(zap) else "<div></div>")
    h.append("</div>")

    h.append(f'<span class="ztag">{r.get("tag_production", "")}</span>'
             f'<span class="ztag">{r.get("tag_risk", "")}</span>')
    if pd.notna(dcd):
        h.append(f'<span class="ztag">DCD {dcd:+.1f}</span>')
    if pd.notna(lam) and lam == 0:
        h.append('<span class="ztag" title="college adds nothing at this position">'
                 'ZAP = DRAFT ORDER</span>')
    h.append(f'<div class="znote">capital-free opinion (zap_prod): '
             f'{r.get("zap_prod"):.1f}</div>' if pd.notna(r.get("zap_prod")) else "")

    for title, panel in (("MODEL DATA", MODEL_PANEL.get(pos, [])),
                         ("ADVANCED DATA", ADV_PANEL.get(pos, []))):
        h.append(f'<div class="zsec">{title} &nbsp; '
                 f'<span style="font-weight:400">PERCENTILES VS 2014+ CLASSES</span></div>')
        for col, label, fmt, note in panel:
            if col not in df.columns:
                continue
            val = r.get(col)
            pct = r.get(f"pct_{col}")
            vs = fmt.format(val) if pd.notna(val) else "—"
            h.append(f'<div class="zrow"><div class="zlab">{label}</div>'
                     f'<div class="zval">{vs}</div>{bar_html(pct)}'
                     f'<div class="zpct">{"" if pd.isna(pct) else f"{int(pct)}"}</div></div>')
            if note:
                h.append(f'<div class="znote">↳ {note}</div>')
    h.append("</div>")
    return "".join(h)


if labels:
    a, b = st.columns(2)
    with a:
        st.markdown(render(lut[p1]), unsafe_allow_html=True)
    with b:
        st.markdown(render(lut[p2]), unsafe_allow_html=True)

with st.expander("What this score is, and what it is not"):
    st.markdown("""
**`zap`** = draft capital **+ λ ×** whatever college production adds on top of it.
λ is chosen out-of-sample, per position, against actual NFL outcomes.

| pos | λ | holdout vs draft order | verdict |
|---|---|---|---|
| RB | 1.70 | +0.035 all / +0.014 played, 3/3 classes, **bootstrap 90%** | college genuinely helps |
| WR | 0.70 | +0.006 / +0.002, bootstrap 73% | not robust — treat as draft order |
| QB | 1.45 | −0.003 / +0.086, bootstrap 46% | coin flip |
| TE | 0.00 | identical by construction | college adds nothing |

**`zap_prod`** never sees the pick. It is the model's independent opinion and it is
*worse* than the pick at ranking outcomes at every position — shown because the
**gap** between `zap_prod` and the pick (that's `DCD`) is where disagreement lives.

**What's missing.** Yards per route run, slot rate, aDOT, contested-catch rate, YAC
and avoided tackles are PFF charting data and have no free source. College play-by-play
carries no depth or direction qualifier at all — 1.5% of pass plays contain the word
"deep" — so this is not a parsing gap that can be closed. Target share is also
unavailable: only 27.8% of incompletions name the intended receiver, so *reception*
share is used instead, which mildly rewards sure-handed receivers.

Those missing metrics are receiver-evaluation metrics, which is exactly where this
model is weakest. That is the argument for buying the data, and it is now a testable
one rather than a guess.
""")
