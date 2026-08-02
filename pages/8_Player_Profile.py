"""
Player Profile -- a current-player card in the same shape as the ZAP prospect
card, driven by data/player_history.db (see player_history.py).

Single or COMPARE mode (two players side by side).

Headline number is `skill_score`, the model's 0-100 position-normalised player
rating. Every bar is a PERCENTILE WITHIN POSITION: the current-season panel ranks
against the 2026 board, and each career row ranks against every player at that
position IN THAT SEASON -- so a 250-point WR season in 2016 and in 2025 are
scored against their own peers, not against each other.

That is also what makes CROSS-POSITION comparison meaningful: putting a WR next
to an RB compares two percentiles, each computed inside its own position, rather
than two raw point totals that were never on the same scale.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPO = ROOT.parent

st.set_page_config(page_title="Player Profile", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

DB = REPO / "data" / "player_history.db"
if not DB.exists():
    DB = ROOT / "data" / "player_history.db"

CURRENT_SEASON = 2026


# ⚠ SHIPPED JSON FIRST -- draft_wizard/ deploys standalone and has no SQLite DB.
# export_wizard_data.py bakes the same tables (plus the career arc) into
# data/player_history.json; the DB path is the local-dev fallback only.
SHIPPED = ROOT / "data" / "player_history.json"


def _cols(blob: dict, key: str) -> pd.DataFrame:
    d = blob.get(key) or {"cols": [], "rows": []}
    return pd.DataFrame(d["rows"], columns=d["cols"])


@st.cache_data(show_spinner=False)
def load(path: str, mtime: float) -> dict:
    if path.endswith(".json"):
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        players = _cols(blob, "players")
        seasons = _cols(blob, "seasons")
        weekly = _cols(blob, "weekly")
        arc = _cols(blob, "arc")
        for col in ("actual_ppr", "ppg_ppr", "proj_pts", "proj_pg", "skill_score"):
            if col in seasons.columns:
                seasons[f"pct_{col}"] = (
                    seasons.groupby(["season", "position"])[col].rank(pct=True) * 100).round(0)
        return {"players": players, "seasons": seasons, "weekly": weekly, "arc": arc}

    con = sqlite3.connect(path)
    players = pd.read_sql_query("SELECT * FROM players", con)
    seasons = pd.read_sql_query("SELECT * FROM season_scores", con)
    weekly = pd.read_sql_query(
        "SELECT player_id, season, week, points_ppr, season_to_date_ppr "
        "FROM weekly_snapshots", con)
    con.close()
    for col in ("actual_ppr", "ppg_ppr", "proj_pts", "proj_pg", "skill_score"):
        if col in seasons.columns:
            seasons[f"pct_{col}"] = (
                seasons.groupby(["season", "position"])[col].rank(pct=True) * 100).round(0)
    return {"players": players, "seasons": seasons, "weekly": weekly,
            "arc": pd.DataFrame()}


_src = SHIPPED if SHIPPED.exists() else DB
if not _src.exists():
    st.info("No player history yet — run `python player_history.py --backfill` "
            "then `python export_wizard_data.py`.")
    st.stop()

d = load(str(_src), _src.stat().st_mtime)
players, seasons, weekly = d["players"], d["seasons"], d["weekly"]

CSS = """
<style>
.pcard{border:2px solid #262626;border-radius:6px;padding:14px 18px;margin-bottom:14px;
       box-shadow:4px 4px 0 #262626;}
.pname{font-size:30px;font-weight:800;letter-spacing:-.5px;line-height:1.1;}
.pmeta{font-size:12px;opacity:.75;margin-top:3px;}
.pscore{font-size:44px;font-weight:800;line-height:1;text-align:right;}
.pscorelab{font-size:10px;letter-spacing:2px;opacity:.6;text-align:right;}
.ptag{display:inline-block;border:1px solid #262626;border-radius:3px;padding:2px 8px;
      font-size:11px;letter-spacing:1px;margin:6px 6px 0 0;}
.prow{display:flex;align-items:center;gap:9px;font-size:13px;padding:3px 0;}
.plab{flex:0 0 36%;}
.pval{flex:0 0 15%;text-align:right;font-variant-numeric:tabular-nums;}
.pwin{font-weight:800;}
.pbar{flex:1;height:7px;background:rgba(128,128,128,.22);border-radius:4px;overflow:hidden;}
.pfill{height:100%;}
.ppct{flex:0 0 38px;text-align:right;font-size:11px;opacity:.7;}
.psec{font-size:11px;letter-spacing:2px;font-weight:700;opacity:.7;
      border-bottom:1px solid rgba(128,128,128,.35);padding-bottom:4px;margin:15px 0 8px 0;}
.pnote{font-size:10px;opacity:.55;margin:2px 0 6px 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.title("🧬 Player Profile")


def bar(pct) -> str:
    if pd.isna(pct):
        return '<div class="pbar"><div class="pfill" style="width:0%"></div></div>'
    p = float(np.clip(pct, 0, 100))
    c = "#2fa84f" if p >= 75 else "#7ab8ff" if p >= 50 else "#f0ad4e" if p >= 25 else "#d9534f"
    return f'<div class="pbar"><div class="pfill" style="width:{p:.0f}%;background:{c}"></div></div>'


def row(label, value, pct, note="", other_pct=None) -> str:
    """One metric line. In compare mode `other_pct` is the same metric for the
    other player -- the higher PERCENTILE is bolded, not the higher raw value,
    because the two may be different positions on different scales."""
    # Compare the ROUNDED percentiles -- the same ones on screen. Comparing the
    # raw floats marked a winner on rows that both displayed "98", which reads as
    # the UI making it up. If the two numbers shown are equal, claim no winner.
    win = (other_pct is not None and pd.notna(pct) and pd.notna(other_pct)
           and round(float(pct)) > round(float(other_pct)))
    cls = "pval pwin" if win else "pval"
    mark = " ◂" if win else ""
    # round(), not int() -- int truncates, so 99.59 displayed as "99" while the
    # winner check (which rounds) saw 100, and the ◂ landed on a row showing
    # "99 vs 99". The number on screen and the number being compared must match.
    h = (f'<div class="prow"><div class="plab">{label}</div>'
         f'<div class="{cls}">{value}{mark}</div>{bar(pct)}'
         f'<div class="ppct">{"" if pd.isna(pct) else f"{round(float(pct))}"}</div></div>')
    if note:
        h += f'<div class="pnote">↳ {note}</div>'
    return h


def pct_of(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float((s < value).mean() * 100) if (len(s) and pd.notna(value)) else np.nan


# --------------------------------------------------------------------------
# per-player derived numbers
# --------------------------------------------------------------------------
def profile(pid: str) -> dict:
    P = players[players["player_id"] == pid].iloc[0]
    hist = seasons[seasons["player_id"] == pid].sort_values("season")
    now = hist[hist["season"] == CURRENT_SEASON]
    now = now.iloc[0] if len(now) else None
    past = hist[(hist["season"] < CURRENT_SEASON) & (hist["games"].fillna(0) > 0)]

    age = np.nan
    if pd.notna(P.get("birth_date")):
        bd = pd.to_datetime(P["birth_date"], errors="coerce")
        if pd.notna(bd):
            age = (pd.Timestamp(f"{CURRENT_SEASON}-09-01") - bd).days / 365.25

    career_best = past["actual_ppr"].max() if len(past) else np.nan
    career_best_yr = int(past.loc[past["actual_ppr"].idxmax(), "season"]) if len(past) else None
    last3 = past[past["season"] >= CURRENT_SEASON - 3]
    # best-of-LAST-3 and CAREER best need separate peer pools -- scoring a 2014
    # career year against 2023-25 peers compares a player to people he never
    # played with, which is what a single "best season" row used to do.
    best3 = last3["actual_ppr"].max() if len(last3) else np.nan
    avg3 = last3["ppg_ppr"].mean() if len(last3) else np.nan
    dur = (last3["games"].mean() / 17 * 100) if len(last3) else np.nan
    cons = (100 - min(last3["ppg_ppr"].std() / max(last3["ppg_ppr"].mean(), 1e-6) * 100, 100)
            if len(last3) >= 2 else np.nan)

    pos = P["position"]
    peer = seasons[(seasons["season"] >= CURRENT_SEASON - 3)
                   & (seasons["season"] < CURRENT_SEASON)
                   & (seasons["position"] == pos) & (seasons["games"].fillna(0) > 0)]
    pk = peer.groupby("player_id").agg(best=("actual_ppr", "max"),
                                       ppg=("ppg_ppr", "mean"),
                                       g=("games", "mean")).reset_index()
    career_peer = seasons[(seasons["position"] == pos) & (seasons["games"].fillna(0) > 0)
                          & (seasons["season"] < CURRENT_SEASON)]
    career_pk = career_peer.groupby("player_id")["actual_ppr"].max()

    n_pos = int((seasons[seasons["season"] == CURRENT_SEASON]["position"] == pos).sum())
    rk = int(now["position_rank"]) if (now is not None and pd.notna(now.get("position_rank"))) else None

    return dict(
        P=P, pos=pos, now=now, past=past, age=age, n_pos=n_pos, rk=rk,
        skill=(now["skill_score"] if now is not None else np.nan),
        pct_skill=(now.get("pct_skill_score") if now is not None else np.nan),
        pct_proj=(now.get("pct_proj_pts") if now is not None else np.nan),
        pct_projpg=(now.get("pct_proj_pg") if now is not None else np.nan),
        pct_rank=((100 - rk / max(n_pos, 1) * 100) if rk else np.nan),
        career_best=career_best, career_best_yr=career_best_yr,
        pct_career_best=pct_of(career_pk, career_best),
        best3=best3, pct_best3=pct_of(pk["best"], best3),
        avg3=avg3, pct_avg3=pct_of(pk["ppg"], avg3),
        dur=dur, pct_dur=pct_of(pk["g"] / 17 * 100, dur),
        cons=cons,
    )


def card_html(a: dict, b: dict | None = None) -> str:
    """`b` is the other player in compare mode -- used only to mark winners."""
    def o(key):
        return b[key] if b is not None else None

    P, now, past, pos = a["P"], a["now"], a["past"], a["pos"]
    h = ['<div class="pcard">']
    h.append('<div style="display:flex;justify-content:space-between;align-items:flex-start">')
    meta = " &nbsp;·&nbsp; ".join(filter(None, [
        pos,
        (now["team"] if now is not None and pd.notna(now.get("team")) else None),
        (f"Age {a['age']:.1f}" if pd.notna(a["age"]) else None),
        (f"{int(P['draft_year'])} pk {int(P['draft_pick'])}"
         if pd.notna(P.get("draft_year")) and pd.notna(P.get("draft_pick")) else None),
        (P["college"] if pd.notna(P.get("college")) else None),
    ]))
    h.append(f'<div><div class="pname">{P["display_name"].upper()}</div>'
             f'<div class="pmeta">{meta}</div></div>')
    if pd.notna(a["skill"]):
        h.append(f'<div><div class="pscorelab">SKILL SCORE</div>'
                 f'<div class="pscore">{a["skill"]:.1f}</div></div>')
    h.append("</div>")

    tags = []
    if a["rk"]:
        r = a["rk"]
        tags.append("ELITE" if r <= 6 else "STARTER" if r <= 24 else
                    "FLEX" if r <= 40 else "DEPTH")
    if pd.notna(a["dur"]):
        tags.append("DURABLE" if a["dur"] >= 88 else
                    "INJURY RISK" if a["dur"] < 65 else "NEUTRAL")
    if len(past) == 0:
        tags.append("NO NFL HISTORY")
    if pd.notna(P.get("zap")):
        tags.append(f"ZAP {P['zap']:.0f}")
    for t in tags:
        h.append(f'<span class="ptag">{t}</span>')

    h.append(f'<div class="psec">{CURRENT_SEASON} PROJECTION &nbsp; '
             f'<span style="font-weight:400">PERCENTILE VS {pos}s</span></div>')
    if now is not None:
        h.append(row("Projected points", f"{now['proj_pts']:.1f}", a["pct_proj"],
                     other_pct=o("pct_proj")))
        h.append(row("Projected / game", f"{now['proj_pg']:.1f}", a["pct_projpg"],
                     other_pct=o("pct_projpg")))
        h.append(row("Skill score", f"{a['skill']:.1f}" if pd.notna(a["skill"]) else "—",
                     a["pct_skill"], other_pct=o("pct_skill")))
        h.append(row("Positional rank", f"{pos}{a['rk']}" if a["rk"] else "—",
                     a["pct_rank"], f"of {a['n_pos']} ranked {pos}s",
                     other_pct=o("pct_rank")))
        if now.get("week_thru", 0) and now["week_thru"] > 0:
            h.append(row(f"Actual through wk {int(now['week_thru'])}",
                         f"{now['actual_ppr']:.1f}", now.get("pct_actual_ppr")))
    else:
        h.append('<div class="pnote">Not on the current board (unsigned, retired, '
                 'or filtered out).</div>')

    h.append('<div class="psec">CAREER PRODUCTION &nbsp; '
             '<span style="font-weight:400">EACH SEASON VS THAT SEASON\'S PEERS</span></div>')
    if len(past):
        for _, s in past.sort_values("season", ascending=False).iterrows():
            h.append(row(f"{int(s['season'])} &nbsp; {s['team'] or ''} &nbsp; "
                         f"<span style='opacity:.6'>{int(s['games'])}g</span>",
                         f"{s['actual_ppr']:.1f}", s.get("pct_actual_ppr"),
                         f"{s['ppg_ppr']:.1f} ppg"))
    else:
        h.append('<div class="pnote">No NFL regular-season history yet.</div>')

    h.append('<div class="psec">PROFILE &nbsp; '
             '<span style="font-weight:400">EACH ROW STATES ITS OWN WINDOW</span></div>')
    h.append(row("Career-best season",
                 f"{a['career_best']:.1f}" if pd.notna(a["career_best"]) else "—",
                 a["pct_career_best"],
                 f"in {a['career_best_yr']}, vs other {pos}s' career bests"
                 if a["career_best_yr"] else "", other_pct=o("pct_career_best")))
    h.append(row("Best of last 3", f"{a['best3']:.1f}" if pd.notna(a["best3"]) else "—",
                 a["pct_best3"], "vs the last 3 seasons only", other_pct=o("pct_best3")))
    h.append(row("Avg points / game", f"{a['avg3']:.1f}" if pd.notna(a["avg3"]) else "—",
                 a["pct_avg3"], "last 3 seasons", other_pct=o("pct_avg3")))
    h.append(row("Durability (games)", f"{a['dur']:.0f}%" if pd.notna(a["dur"]) else "—",
                 a["pct_dur"], "last 3 seasons, share of a 17-game season",
                 other_pct=o("pct_dur")))
    h.append(row("Consistency", f"{a['cons']:.0f}" if pd.notna(a["cons"]) else "—",
                 a["cons"], "last 3 seasons; 100 = identical ppg every season",
                 other_pct=o("cons")))
    h.append("</div>")
    return "".join(h)


def charts(a: dict, key: str) -> None:
    past = a["past"]
    if len(past):
        ch = past[["season", "actual_ppr", "ppg_ppr", "games"]].copy()
        ch["season"] = ch["season"].astype(int)
        st.bar_chart(ch.set_index("season")["actual_ppr"], height=190)
        st.dataframe(ch.rename(columns={"actual_ppr": "PPR", "ppg_ppr": "PPG",
                                        "games": "G"})
                     .sort_values("season", ascending=False),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No NFL history yet.")


# --------------------------------------------------------------------------
# selector
# --------------------------------------------------------------------------
cur = seasons[seasons["season"] == CURRENT_SEASON]
c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
with c3:
    pos_f = st.multiselect("Positions", ["QB", "RB", "WR", "TE", "K"],
                           default=["QB", "RB", "WR", "TE"])
with c4:
    only_active = st.checkbox(f"{CURRENT_SEASON} board only",
                              value=str(st.query_params.get("all", "0")) != "1")
    compare = st.checkbox("Compare two",
                          value=bool(st.query_params.get("player2")))

pool = players.merge(
    cur[["player_id", "team", "proj_pts", "position_rank"]], on="player_id",
    how="inner" if only_active else "left", suffixes=("", "_cur"))
pool = pool[pool["position"].isin(pos_f)]
pool = pool.sort_values("proj_pts", ascending=False, na_position="last")

labels, lut = [], {}
for _, r in pool.iterrows():
    rk = f", {r['position']}{int(r['position_rank'])}" if pd.notna(r.get("position_rank")) else ""
    labels.append(f"{r['display_name']} ({r['position']}{rk})")
    lut[labels[-1]] = r["player_id"]

if not labels:
    st.warning("No players match that filter.")
    st.stop()


def _index_for(param: str, fallback: int) -> int:
    q = st.query_params.get(param)
    if not q:
        return min(fallback, len(labels) - 1)
    q = str(q).lower()
    for i, lab in enumerate(labels):
        if q in lab.lower() or lut[lab].lower() == q:
            return i
    return min(fallback, len(labels) - 1)


with c1:
    pick1 = st.selectbox("Player", labels, index=_index_for("player", 0))
st.query_params["player"] = pick1.split(" (")[0]

pick2 = None
if compare:
    with c2:
        pick2 = st.selectbox("Compare with", labels, index=_index_for("player2", 1))
    st.query_params["player2"] = pick2.split(" (")[0]
elif "player2" in st.query_params:
    del st.query_params["player2"]

# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
if compare and pick2:
    A, B = profile(lut[pick1]), profile(lut[pick2])
    if A["pos"] != B["pos"]:
        st.caption(f"Comparing a {A['pos']} with a {B['pos']} — the **bars** are "
                   f"comparable (each is a percentile inside its own position), "
                   f"the **raw values** are not.")
    L, R = st.columns(2)
    with L:
        st.markdown(card_html(A, B), unsafe_allow_html=True)
    with R:
        st.markdown(card_html(B, A), unsafe_allow_html=True)
    L2, R2 = st.columns(2)
    with L2:
        st.markdown(f"**{A['P']['display_name']} — season history**")
        charts(A, "a")
    with R2:
        st.markdown(f"**{B['P']['display_name']} — season history**")
        charts(B, "b")
else:
    A = profile(lut[pick1])
    left, right = st.columns([3, 2])
    with left:
        st.markdown(card_html(A), unsafe_allow_html=True)
    with right:
        st.markdown("**Season scoring history**")
        charts(A, "a")
        past = A["past"]
        wk = (weekly[(weekly["player_id"] == lut[pick1])
                     & (weekly["season"] == past["season"].max())]
              if len(past) else pd.DataFrame())
        if len(wk):
            st.markdown(f"**Week-by-week, {int(wk['season'].iloc[0])}**")
            st.line_chart(wk.set_index("week")["points_ppr"], height=170)

# --------------------------------------------------------------------------
# career arc -- what did players at this exact stage do NEXT?
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _comps(pid: str, mtime: float) -> dict:
    """Shipped arc row first (phone), live career_comps second (local dev).

    The shipped row carries the summary + comp NAMES but not the full comp
    table, so the phone shows the distribution and who the comps were, without
    the per-comp before/after grid."""
    A = d.get("arc")
    if A is not None and len(A) and "player_id" in A.columns:
        row = A[A["player_id"] == pid]
        if len(row):
            r = row.iloc[0].to_dict()
            r["comps"] = []
            r["me"] = {"prev_ppg": r.get("prev_ppg")}
            r["median"] = r.get("comp_median"); r["p25"] = r.get("comp_p25")
            r["p75"] = r.get("comp_p75"); r["n_pool"] = r.get("n_comps")
            r["bust_rate"] = r.get("decline_rate"); r["leap_rate"] = r.get("leap_rate")
            r["career_yr"] = r.get("career_yr"); r["position"] = r.get("position")
            return r
    try:
        import player_history as ph
        c = ph.career_comps(pid, k=20)
        if c:
            c = dict(c); c["comps"] = c["comps"].to_dict("records")
        return c
    except Exception:
        return {}


def arc_block(pid: str, prof: dict, label: str = "") -> None:
    # ⚠ _src, not DB. On the phone there is no SQLite file, so DB.stat() raised
    # FileNotFoundError and killed the page BELOW the fold -- everything above
    # rendered fine, which is why the HTTP 200 check did not catch it.
    c = _comps(pid, _src.stat().st_mtime)
    if not c:
        st.caption(f"{label}No career arc — needs a draft year and at least 12 "
                   f"historical players at the same career stage.")
        return
    me, comps = c["me"], pd.DataFrame(c["comps"])
    proj = prof["now"]["proj_pg"] if prof["now"] is not None else np.nan

    st.markdown(f"**{label}Career year {c['career_yr']} — what {c['position']}s who "
                f"reached this same point did next** ({c['n_pool']} qualified, 20 closest used)")
    a, b, cc, dd = st.columns(4)
    a.metric("His last season", f"{me['prev_ppg']:.1f} ppg")
    b.metric("Comps' next year (median)", f"{c['median']:.1f} ppg",
             f"{c['median'] - me['prev_ppg']:+.1f} vs his last")
    cc.metric("Comp range (p25–p75)", f"{c['p25']:.1f}–{c['p75']:.1f}")
    # ⚠ `comps` is EMPTY on the phone -- the shipped arc row carries the summary
    # and the comp NAMES but not the per-comp before/after rows, so anything that
    # indexes comps["next_ppg"] must be guarded. Use the precomputed arc_pctile
    # when the detail is absent.
    if pd.notna(proj):
        if len(comps) and "next_ppg" in comps.columns:
            pctile = float((comps["next_ppg"] < proj).mean() * 100)
        else:
            pctile = c.get("arc_pctile")
        dd.metric("Our projection", f"{proj:.1f} ppg",
                  f"{pctile:.0f}th of comps" if pd.notna(pctile) else "—")
    st.caption(f"{c['bust_rate']:.0f}% of comps DECLINED >25% · "
               f"{c['leap_rate']:.0f}% LEAPT >25%. Comps are matched on draft pick and "
               f"production *up to* this point only, so the outcome year is never used "
               f"to pick them.")
    if len(comps) and "next_ppg" in comps.columns:
        show = comps.head(8)[["name", "comp_season", "draft_pick", "prev_ppg", "next_ppg"]]
        show.columns = ["comp", "their yr", "pick", "ppg before", "ppg after"]
        st.dataframe(show.round(1), hide_index=True, use_container_width=True)
    elif c.get("comp_names"):
        st.caption(f"Closest comps: {c['comp_names']}")


st.divider()
if compare and pick2:
    ca, cb = st.columns(2)
    with ca:
        arc_block(lut[pick1], A, f"{A['P']['display_name']} — ")
    with cb:
        arc_block(lut[pick2], B, f"{B['P']['display_name']} — ")
else:
    arc_block(lut[pick1], A)

with st.expander("What these numbers are"):
    st.markdown(f"""
**Skill score** is the model's 0-100, position-normalised player rating from `skill_engine`
— the same number that drives the within-room split in the projection. It is *not* a
projection; a backup on a good offence can out-project a better player stuck behind a stud.

**Every bar is a percentile within position.** The {CURRENT_SEASON} panel ranks against this
season's board. Each career row ranks against every player at that position **in that
season**, so a 250-point WR year in 2016 and in 2025 are each scored against their own peers.

**In compare mode the bolded value with a ◂ is the higher PERCENTILE, not the higher raw
number** — deliberately, because the two players may be different positions on completely
different scales. A WR's 280 points and an RB's 280 points are not the same achievement;
their percentiles are.

**Consistency** is `100 − (std/mean of ppg across the last 3 seasons)`, so 100 means an
identical per-game rate every year. It is a *stability* measure, not a quality one — a
reliably mediocre player scores high.

Deep links: `?player=Name`, `?player2=Name` (opens compare), `?all=1` (include players not
on the current board). Data from `data/player_history.db`, refreshed by
`player_history.py --update` in the weekly pipeline. Actuals are **regular season only**.
""")
