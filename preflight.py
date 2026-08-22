"""
preflight.py -- run this before a real draft.
=============================================
Drives the ACTUAL app script through streamlit.testing.v1.AppTest and asserts on
what it renders, then checks the data and the live league behind it.

Why AppTest and not "does it start": the failure mode that matters here is not a
crash, it is a panel that renders BLANK or a config that is quietly wrong. The
2026-08-21 audit found a league that had silently changed from Superflex to 1QB,
which produced no error anywhere and simply mis-valued every QB by ~110 VOR.

    python draft_wizard/preflight.py                # local repo
    python draft_wizard/preflight.py --league 1381162289226346496
    python draft_wizard/preflight.py --phone        # also simulate the standalone bundle

Exit code 0 = green.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FAILS: list[str] = []
WARNS: list[str] = []


def ok(msg):
    print(f"  ✓ {msg}")


def bad(msg):
    FAILS.append(msg)
    print(f"  ✗ {msg}")


def warn(msg):
    WARNS.append(msg)
    print(f"  ! {msg}")


def head(msg):
    print(f"\n=== {msg} ===")


# --------------------------------------------------------------------------- data
def check_board():
    head("BOARD DATA")
    p = HERE / "data" / "draft_board.json"
    if not p.exists():
        return bad("draft_board.json missing")
    raw = json.load(open(p, encoding="utf-8"))
    meta, players = raw["meta"], raw["players"]
    age_h = (time.time() - p.stat().st_mtime) / 3600
    print(f"  generated {meta.get('generated_at')}  ({age_h:.1f}h old)")
    if len(players) < 600:
        bad(f"only {len(players)} players on the board")
    else:
        ok(f"{len(players)} players")

    n_adp = sum(1 for x in players if x.get("adp"))
    (ok if n_adp >= 240 else bad)(f"{n_adp} players with ADP")
    if age_h > 96:
        warn(f"board is {age_h/24:.1f} days old — ADP moves daily in August; "
             f"re-run adp_ingest.py -> consensus.py -> export_draft_board.py")

    # every field the UI reads must exist on a real player, or the column is blank
    need = ["id", "name", "pos", "team", "pts", "adp", "adp_sd", "bye", "arc", "decl"]
    cov = Counter()
    for x in players:
        for k in need:
            if x.get(k) is not None:
                cov[k] += 1
    for k in need:
        if cov[k] == 0:
            bad(f"field `{k}` is empty on EVERY player — its column will render blank")
    ok("UI fields present: " + ", ".join(f"{k} {cov[k]}" for k in need))

    ids = [x["id"] for x in players]
    if len(set(ids)) != len(ids):
        dup = [i for i, c in Counter(ids).items() if c > 1][:5]
        bad(f"duplicate player ids (Streamlit key collisions): {dup}")
    else:
        ok("player ids unique")

    top = sorted([x for x in players if x.get("adp")], key=lambda x: x["adp"])[:24]
    if len(top) < 24:
        bad("fewer than 24 players carry an ADP")
    else:
        ok(f"market top-5: " + ", ".join(f"{x['name']}" for x in top[:5]))

    n_dst = sum(1 for x in players if x["pos"] == "DST" and x.get("adp"))
    (ok if n_dst >= 20 else warn)(f"{n_dst}/32 defenses carry an ADP")
    return players


# --------------------------------------------------------------------------- engine
def check_engine(players, cfgs):
    head("ENGINE")
    import numpy as np
    from draft_engine import (LeagueConfig, analyze, plan_draft, prep_valued,
                              mock_advance, grade_draft)
    for label, kw in cfgs:
        cfg = LeagueConfig(**kw)
        t0 = time.time()
        r = analyze(players, cfg, [])
        dt = (time.time() - t0) * 1000
        rec = r["recommendation"]
        if not rec:
            bad(f"{label}: no recommendation on an empty board")
            continue
        exp = cfg.teams * cfg.total_rounds()
        if r["total_picks"] != exp:
            bad(f"{label}: total_picks {r['total_picks']} != {exp}")
        # a 1QB league must not put a QB in the first handful of picks
        first_qb = next((i + 1 for i, x in enumerate(r["best_available"])
                         if x["pos"] == "QB"), None)
        note = f" firstQB@{first_qb}"
        if not cfg.superflex and first_qb is not None and first_qb <= 12:
            bad(f"{label}: a QB is board slot {first_qb} in a 1QB league — check superflex")
        ok(f"{label}: rec {rec['name']} ({rec['pos']}) · {r['total_picks']} picks ·"
           f"{note} · {dt:.0f}ms")
        if dt > 4000:
            warn(f"{label}: analyze() took {dt:.0f}ms — slow on a phone")
        if "ranked" not in r:
            bad(f"{label}: analyze() no longer returns `ranked` — the board would fall "
                f"back to raw VOR order")
        # ⭐ every column the UI renders must survive _slim()'s FIXED key set. A
        # field can be present on all 680 board players and still reach the screen
        # as None -- `bye` shipped that way and printed "None" on every row.
        _ba = r["best_available"][:40]
        for _col in ("name", "pos", "posrank", "team", "bye", "vor", "tier", "adp",
                     "survival", "rec_score", "cost_of_waiting", "arc", "decl"):
            if not any(x.get(_col) is not None for x in _ba):
                bad(f"{label}: `{_col}` is None on every row of best_available — "
                    f"that column renders blank. Check _slim() in draft_engine.")

    # a full mock has to terminate and produce legal rosters. Run it on a league
    # that STARTS a kicker when one is available -- the K/DST late-round logic is
    # inert in a no-kicker league, so cfgs[0] alone cannot exercise it.
    _kcfg = next((c for c in cfgs if c[1]["starters"].get("K")), cfgs[0])
    cfg = LeagueConfig(**_kcfg[1])
    print(f"  full-mock league: {_kcfg[0]}")
    rng = np.random.default_rng(11)
    valued = prep_valued(players, cfg)
    drafted, guard = [], 0
    total = cfg.teams * cfg.total_rounds()
    while len(drafted) < total and guard < total + 10:
        drafted = mock_advance(valued, cfg, drafted, rng)
        if len(drafted) >= total:
            break
        rr = analyze(players, cfg, drafted, n_sims=200).get("recommendation")
        if not rr:
            break
        drafted.append(rr["id"]); guard += 1
    if len(drafted) != total:
        bad(f"full mock ended at {len(drafted)}/{total} picks")
    else:
        ok(f"full mock completed {total}/{total} picks")
    by_id = {p["id"]: p for p in players}
    fin = analyze(players, cfg, drafted)
    if not fin["complete"]:
        bad("analyze() does not report complete at the last pick")
    elif fin["recommendation"] is not None:
        bad("analyze() still recommends a player after the final pick")
    else:
        ok("draft terminates cleanly (complete=True, recommendation=None)")

    shape = Counter()
    from draft_engine import team_on_clock
    for i, pid in enumerate(drafted):
        if team_on_clock(cfg, i + 1) == cfg.my_slot and pid in by_id:
            shape[by_id[pid]["pos"]] += 1
    ok("your mock roster: " + " ".join(f"{k}{v}" for k, v in sorted(shape.items())))
    for pos, want in cfg.starters.items():
        if pos in ("FLEX", "SUPERFLEX") or not want:
            continue
        if shape.get(pos, 0) < want:
            bad(f"mock roster is short at {pos}: {shape.get(pos,0)} < {want} starters")
    g = grade_draft(cfg, drafted, players)
    ok(f"grade_draft: {g['grade']} rank {g['my_rank']}/{g['teams']}")

    # ⭐ REGRESSION GUARD (found 2026-08-21): the board is shown in the DECISION
    # order, not raw VOR. Sorted by VOR, Brandon Aubrey reaches #2 OVERALL at pick
    # 73 of a 15-round league (VOR 39.4) while his decision score is -28.0 -- the
    # screen was offering a pick the engine would never make.
    #
    # ⚠ The criterion is the invariant, NOT the calendar. A first draft of this
    # check asserted "no K/DST in the top 5 before the last 3 rounds" and fired on
    # a defense at round 11 -- which is CORRECT behaviour: FFC has Seattle's D at
    # ADP 83 and eight defenses inside 140, i.e. rounds 7-12. The rule that cannot
    # be wrong is: while somebody startable is still worth taking (positive score),
    # a kicker or defense must not outrank him.
    viol = []
    for at in range(0, len(drafted), cfg.teams * 2):
        rr = analyze(players, cfg, drafted[:at], n_sims=150)
        top = (rr.get("ranked") or [])[:5]
        best_skill = max((x["rec_score"] for x in (rr.get("ranked") or [])
                          if x["pos"] not in ("K", "DST")), default=-999)
        for x in top:
            if x["pos"] in ("K", "DST") and best_skill > 0 and x["rec_score"] > best_skill:
                viol.append((at + 1, x["name"], round(x["rec_score"], 1),
                             round(best_skill, 1)))
    if viol:
        bad(f"a K/DST outranked a positively-scored skill player on the decision "
            f"board: {viol[:3]}")
    else:
        ok("no K/DST ever outranks a startable positive-score player")

    # and confirm the two orderings really are different, i.e. the fix is load-bearing
    _mid = analyze(players, cfg, drafted[: cfg.teams * 6], n_sims=150)
    _r5 = [x["name"] for x in _mid["ranked"][:5]]
    _v5 = [x["name"] for x in _mid["best_available"][:5]]
    if _r5 == _v5:
        warn("decision order and raw-VOR order agree in the top 5 at round 7 — the "
             "board selector is inert in this sample, not necessarily wrong")
    else:
        ok(f"decision vs raw-VOR order differ at round 7 (rec: {_r5[0]} · vor: {_v5[0]})")

    # ...and the mirror check: in a league that STARTS a kicker, one must actually
    # get drafted, or the late-round nudge has been broken the other way
    if cfg.starters.get("K"):
        kn = sum(1 for pid in drafted if by_id.get(pid, {}).get("pos") == "K")
        (ok if kn >= cfg.teams * 0.8 else bad)(
            f"{kn}/{cfg.teams} kickers drafted in a league that starts one")
    pl = plan_draft(players, cfg, drafted[: cfg.teams * 3], n_sims=40)
    (ok if pl.get("picks") else bad)("plan_draft returns a build path")
    return drafted


# --------------------------------------------------------------------------- board html
def check_visuals(players, cfg_kw, drafted):
    head("VISUAL COMPONENTS (pure HTML, rendered headlessly)")
    from draft_engine import LeagueConfig, analyze
    import draft_board as B
    cfg = LeagueConfig(**cfg_kw)
    by_id = {p["id"]: p for p in players}
    half = drafted[: max(cfg.teams * 4, len(drafted) // 3)]
    r = analyze(players, cfg, half)
    parts = {
        "status_bar_html": B.status_bar_html(cfg, None, r, "Team X"),
        "on_deck_html": B.on_deck_html(cfg, None, r["current_overall"], r["my_next_pick"],
                                       r["opponents"].get("seat_need")),
        "draft_board_html": B.draft_board_html(cfg, None, half, by_id),
        "draft_board_html(win)": B.draft_board_html(cfg, None, half, by_id, rounds_window=7),
        "run_strip_html": B.run_strip_html(cfg, half, by_id),
        "roster_matrix_html": B.roster_matrix_html(cfg, None, half, by_id, cfg.starters),
    }
    for name, html in parts.items():
        if not isinstance(html, str) or len(html) < 80:
            bad(f"{name} rendered {len(html) if html else 0} chars — effectively blank")
        elif "<script" in html.lower():
            bad(f"{name} contains a <script> tag — Streamlit strips it, so it will not run")
        else:
            ok(f"{name}: {len(html)} chars")

    # the grid must actually contain drafted players, not just empty cells
    grid = parts["draft_board_html"]
    shown = sum(1 for p in half if p in by_id and by_id[p]["name"].split()[-1][:8] in grid)
    if half and shown == 0:
        bad("draft board grid shows NO drafted players")
    else:
        ok(f"grid shows {shown}/{len(half)} drafted players")

    # snake pick labels: seat 1 owns 2.12, not 2.01
    if cfg.snake:
        s1_r2 = [o for o in range(cfg.teams + 1, 2 * cfg.teams + 1)
                 if B.pick_owner_slot(cfg, o) == 1]
        lbl = B.pick_label(cfg, s1_r2[0])
        (ok if lbl == f"2.{cfg.teams:02d}" else bad)(
            f"snake labels: seat 1's round-2 pick reads {lbl} (expect 2.{cfg.teams:02d})")


# --------------------------------------------------------------------------- app
def check_app(league_id: str | None = None):
    head("APP (streamlit AppTest driving the real script)")
    from streamlit.testing.v1 import AppTest
    script = str(HERE / "draft_app.py")

    for mode in ["Redraft", "Best Ball", "Rookie (dynasty)"]:
        at = AppTest.from_file(script, default_timeout=300)
        at.run()
        if at.exception:
            bad(f"{mode}: start gate raised {at.exception[0].value}")
            continue
        try:
            at.radio[0].set_value(mode).run()
        except Exception as e:
            bad(f"{mode}: could not select draft type ({e})")
            continue
        if at.exception:
            bad(f"{mode}: switching draft type raised {at.exception[0].value}")
            continue
        start = [b for b in at.button if "Start draft" in b.label]
        if not start:
            bad(f"{mode}: no Start draft button")
            continue
        start[0].click().run()
        if at.exception:
            bad(f"{mode}: main view raised {at.exception[0].value}")
            continue
        n_df = len(at.dataframe)
        if n_df == 0:
            bad(f"{mode}: main view rendered no tables")
        else:
            ok(f"{mode}: renders · {n_df} tables · {len(at.markdown)} markdown blocks")
        # make a pick through the primary control and confirm the state advances
        chips = [b for b in at.button if b.key and b.key.startswith("chip_")]
        if chips:
            before = len(at.session_state["drafted"])
            chips[0].click().run()
            after = len(at.session_state["drafted"])
            if at.exception:
                bad(f"{mode}: making a pick raised {at.exception[0].value}")
            elif after <= before:
                bad(f"{mode}: tapping a quick-pick chip did not record a pick")
            else:
                ok(f"{mode}: pick recorded ({before} -> {after})")
        else:
            warn(f"{mode}: no quick-pick chips rendered")
        # every board ordering + the phone column set must render
        try:
            rb = at.radio(key="rank_by")
            for opt in rb.options:
                at.radio(key="rank_by").set_value(opt).run()
                if at.exception:
                    bad(f"{mode}: ordering {opt!r} raised {at.exception[0].value}")
                    break
            else:
                ok(f"{mode}: all {len(rb.options)} board orderings render")
            at.toggle(key="compact_cols").set_value(True).run()
            (bad if at.exception else ok)(
                f"{mode}: compact (phone) column set renders"
                + (f" — {at.exception[0].value}" if at.exception else ""))
        except Exception as e:
            bad(f"{mode}: board controls missing ({e})")

    # a LIVE Sleeper session: the config override, auto-sync poller and the board's
    # real team names only exist on this path, so the other passes never touch them
    if league_id:
        at = AppTest.from_file(script, default_timeout=300)
        at.run()
        btns = [b for b in at.button if "Load this league" in b.label]
        sels = [x for x in at.selectbox if x.options and any("—" == o for o in x.options)]
        if sels and btns:
            tgt = next((o for o in sels[-1].options if o != "—"), None)
            # pick the league whose id matches, by name
            import json as _j
            pres = _j.load(open(HERE / "data" / "league_presets.json", encoding="utf-8"))["presets"]
            nm = next((p["name"] for p in pres if str(p.get("league_id")) == str(league_id)), None)
            if nm and nm in sels[-1].options:
                tgt = nm
            sels[-1].set_value(tgt).run()
            btns2 = [b for b in at.button if "Load this league" in b.label]
            btns2[0].click().run()
            if at.exception:
                bad(f"live sync: {at.exception[0].value}")
            else:
                try:                       # AppTest's SafeSessionState has no .get
                    cfgd = at.session_state["sleeper_cfg"] or {}
                except Exception:
                    cfgd = {}
                ok(f"live sync in-app: {cfgd.get('league_name')} · slot "
                   f"{cfgd.get('my_slot')} · {cfgd.get('rounds')} rds · "
                   f"sf={((cfgd.get('roster_cfg') or {}).get('superflex'))}")
                tg = [t for t in at.toggle if "Auto-sync" in (t.label or "")]
                (ok if tg else bad)("auto-sync toggle present on a live draft")
        else:
            warn("could not drive the in-app league loader")

    # practice mode: the auto-draft path is the one that used to run past the end
    at = AppTest.from_file(script, default_timeout=600)
    at.run()
    at.checkbox[0].set_value(True).run()          # practice
    [b for b in at.button if "Start draft" in b.label][0].click().run()
    if at.exception:
        bad(f"practice: {at.exception[0].value}")
    else:
        ok(f"practice: room advanced to pick {len(at.session_state['drafted'])+1}")
        auto = [b for b in at.button if "Auto-draft" in b.label]
        if auto:
            auto[0].click().run()
            if at.exception:
                bad(f"practice auto-draft: {at.exception[0].value}")
            else:
                n = len(at.session_state["drafted"])
                ok(f"practice auto-draft finished at {n} picks")


# --------------------------------------------------------------------------- matching
def check_matching(players):
    """Can we recognise the picks Sleeper will send us?

    An unmatched pick is NOT a crash: sleeper_sync parks a `__off_{n}` placeholder
    so pick numbers stay aligned. The damage is that the player stays on OUR board
    as available -- so you can be recommended someone who was taken 40 picks ago.

    ⭐ Two real misses found 2026-08-21:
      * DEFENSES -- Sleeper identifies them by TEAM CODE (player_id "SEA",
        first_name "Seattle", position "DEF"); our board says "SEA DST", so the
        name join matched 0/32. In a league that starts a DEF that is 12 picks.
      * ACCENTS -- Sleeper writes "Audric Estime", our board has "Audric Estimé",
        and norm_name does not decompose diacritics."""
    head("SLEEPER NAME MATCHING")
    import unicodedata
    import urllib.request
    from draft_names import norm_name, pos_norm
    try:
        req = urllib.request.Request("https://api.sleeper.app/v1/players/nfl",
                                     headers={"User-Agent": "Mozilla/5.0"})
        sp = json.load(urllib.request.urlopen(req, timeout=120))
    except Exception as e:
        return warn(f"could not fetch Sleeper's player DB ({e}) — matching unverified")

    def fold(x):
        return "".join(c for c in unicodedata.normalize("NFKD", norm_name(x))
                       if not unicodedata.combining(c))
    TA = {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "LVR": "LV",
          "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI", "AZ": "ARI", "GNB": "GB",
          "KAN": "KC", "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB"}

    def tk(t):
        t = str(t or "").upper().strip()
        return TA.get(t, t)

    km = {(norm_name(x["name"]), x["pos"]): x["id"] for x in players}
    fm = {}
    for x in players:
        fm.setdefault((fold(x["name"]), x["pos"]), x["id"])
    dm = {tk(x.get("team")): x["id"] for x in players if x["pos"] == "DST"}

    # every defense must resolve -- this is the one that was silently 0/32
    nfl32 = ["SEA", "DEN", "HOU", "LAR", "MIN", "NE", "DET", "PIT", "PHI", "LAC",
             "SF", "GB", "JAX", "BUF", "CLE", "KC", "BAL", "DAL", "CIN", "NYG",
             "ARI", "CAR", "IND", "LV", "MIA", "NYJ", "TB", "WAS", "ATL", "CHI",
             "NO", "TEN"]
    nod = [t for t in nfl32 if not dm.get(tk(t))]
    (bad if nod else ok)(f"defenses resolving by team code: {32-len(nod)}/32"
                         + (f"  MISSING {nod}" if nod else ""))

    tot = 0
    miss = []
    for pid, x in sp.items():
        if not x.get("team"):
            continue                       # retired/FA: Sleeper still ranks Brady
        pos = pos_norm(x.get("position"))
        if pos not in ("QB", "RB", "WR", "TE", "K", "DST"):
            continue
        sr = x.get("search_rank")
        if sr is None or sr > 320:
            continue
        nm = f"{x.get('first_name','')} {x.get('last_name','')}".strip()
        tot += 1
        hit = (dm.get(tk(x.get("team"))) if pos == "DST"
               else (km.get((norm_name(nm), pos)) or fm.get((fold(nm), pos))))
        if not hit:
            miss.append((sr, nm, pos, x.get("team")))
    rate = (tot - len(miss)) / max(tot, 1)
    (ok if rate >= 0.92 else bad)(
        f"draftable Sleeper players recognised: {tot-len(miss)}/{tot} ({rate*100:.0f}%)")
    if miss:
        print("      unmatched (become placeholders; pick numbers stay aligned): "
              + ", ".join(f"{n}" for _, n, _, _ in sorted(miss)[:8]))


# --------------------------------------------------------------------------- live league
def check_league(league_id: str):
    head(f"LIVE LEAGUE {league_id}")
    from sleeper_api import league_state
    try:
        s = league_state(league_id=league_id)
    except Exception as e:
        return bad(f"Sleeper unreachable: {e}")
    rc = s.get("roster_cfg") or {}
    print(f"  {s.get('league_name')} · {s['type']} · {s['teams']} teams · {s['rounds']} rounds"
          f" · status {s.get('status')}")
    print(f"  starters {rc.get('starters')}  bench {rc.get('bench')}  "
          f"superflex {rc.get('superflex')}  scoring {rc.get('scoring')}")
    if s.get("my_slot"):
        ok(f"your draft slot is {s['my_slot']} (roster {s.get('my_roster')})")
    else:
        warn("draft order is not set yet — your slot will be blank until the commissioner sets it")
    if not s.get("roster_team"):
        warn("no team names resolved")
    else:
        ok(f"{len(s['roster_team'])} team names resolved")
    if rc.get("idp_slots"):
        warn(f"{rc['idp_slots']} IDP slots — our board is offense-only, IDP picks will show "
             f"as placeholders (pick numbers stay aligned)")

    # the preset must agree with the live league, or the app silently mis-values
    pfile = HERE / "data" / "league_presets.json"
    if pfile.exists():
        pre = next((p for p in json.load(open(pfile, encoding="utf-8"))["presets"]
                    if str(p.get("league_id")) == str(league_id)), None)
        if not pre:
            warn("this league is not in league_presets.json — run import_leagues.py")
        else:
            diffs = []
            if bool(pre.get("superflex")) != bool(rc.get("superflex")):
                diffs.append(f"superflex preset={pre.get('superflex')} live={rc.get('superflex')}")
            if pre.get("rounds") != s.get("rounds"):
                diffs.append(f"rounds preset={pre.get('rounds')} live={s.get('rounds')}")
            if pre.get("starters") != rc.get("starters"):
                diffs.append(f"starters preset={pre.get('starters')} live={rc.get('starters')}")
            if bool(pre.get("snake", True)) != bool(s.get("snake")):
                diffs.append(f"snake preset={pre.get('snake')} live={s.get('snake')}")
            if diffs:
                bad("PRESET DISAGREES WITH THE LIVE LEAGUE: " + " | ".join(diffs)
                    + "  -> run `python import_leagues.py`")
            else:
                ok("saved preset matches the live league")
    return s


# --------------------------------------------------------------------------- phone
def check_phone():
    """The deployed bundle is its own git repo on Streamlit Cloud and cannot see the
    outer pipeline. The 2026-08-01 audit found columns that were silently EMPTY on
    the phone because they were written against the outer repo -- no crash, just
    blank, which is the worst failure mode mid-draft.

    So test it the only way that proves anything: copy the bundle to a scratch dir
    with the outer repo nowhere on the path, and drive the real app there."""
    head("PHONE CONDITIONS (isolated copy of the bundle)")
    import os
    import shutil
    import subprocess
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="dw_phone_"))
    dest = tmp / "draft_wizard"
    shutil.copytree(HERE, dest, ignore=shutil.ignore_patterns(
        "__pycache__", ".git", "*.pyc", ".devcontainer"))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)          # nothing from the outer repo on the path
    r = subprocess.run([sys.executable, "phone_probe.py"], capture_output=True,
                       text=True, cwd=str(dest), env=env)
    out = (r.stdout or "") + (r.stderr or "")
    vals = dict(line.split("=", 1) for line in out.splitlines() if "=" in line
                and line.split("=", 1)[0].isupper())
    if "PHONE_OK" not in out:
        bad(f"isolated bundle failed: ...{out.strip()[-600:]}")
    else:
        leaked = [x for x in vals.get("LEAKED", "").split(",") if x]
        if leaked:
            bad(f"the app can still import outer-repo modules {leaked} — those do NOT "
                f"exist on Streamlit Cloud")
        else:
            ok("outer-repo modules are unreachable (true standalone)")
        # VALUES, not just "it ran": a blank column is the phone failure mode
        for k in ("ARC", "DECL", "BYE", "ADP", "SURV", "TABLES"):
            v = vals.get(k)
            if v is None:
                bad(f"probe did not report {k}")
                continue
            n = int(str(v).split("/")[0])
            (ok if n else bad)(f"{k} on the isolated board: {v}"
                               + ("" if n else "   <-- renders BLANK on the phone"))
        for k, v in vals.items():
            if k.startswith("PANEL_"):
                n = int(v)
                (ok if n > 200 else bad)(f"{k[6:].lower()} panel: {n} chars"
                                         + ("" if n > 200 else "   <-- effectively blank"))
        ok("the real app script runs inside the isolated bundle")
    shutil.rmtree(tmp, ignore_errors=True)

    # none of it reaches the phone if it is not committed and pushed
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True,
                              cwd=str(HERE)).stdout.strip()

    files = set(git("ls-files").split())
    missing = [f for f in ("draft_app.py", "draft_engine.py", "draft_board.py", "bestball.py",
                           "sleeper_api.py", "draft_names.py", "requirements.txt",
                           "data/draft_board.json", "data/league_presets.json")
               if f not in files]
    if missing:
        bad(f"NOT tracked by git, so absent from the deployed app: {missing}")
    else:
        ok(f"all required bundle files tracked ({len(files)} total)")
    dirty = git("status", "--porcelain")
    ahead = git("log", "--oneline", "origin/main..HEAD")
    if dirty:
        warn("uncommitted changes — the LIVE phone app does not have them:\n      "
             + dirty.replace("\n", "\n      "))
    if ahead:
        warn("commits not pushed — the live app is behind:\n      "
             + ahead.replace("\n", "\n      "))
    if not dirty and not ahead:
        ok("working tree clean and pushed — the live app matches this code")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", action="append", default=[],
                    help="Sleeper league_id to validate against the saved preset")
    ap.add_argument("--phone", action="store_true")
    ap.add_argument("--quick", action="store_true", help="skip the full-mock + AppTest passes")
    a = ap.parse_args()

    players = check_board()
    if not players:
        print("\nFATAL: no board."); sys.exit(1)

    cfgs = [
        ("ReDrafters Rejoice 12T 1QB PPR 15rd",
         dict(teams=12, scoring="ppr", superflex=False, my_slot=10, snake=True, rounds=15,
              bench=6, starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2,
                                 "SUPERFLEX": 0, "K": 0, "DST": 1})),
        ("Weekend Warriors 10T 1QB PPR 16rd",
         dict(teams=10, scoring="ppr", superflex=False, my_slot=1, snake=True, rounds=16,
              bench=6, starters={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1,
                                 "SUPERFLEX": 0, "K": 1, "DST": 1})),
        ("The Stage Coach 12T STD 17rd",
         dict(teams=12, scoring="standard", superflex=False, my_slot=6, snake=True, rounds=17,
              bench=7, starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2,
                                 "SUPERFLEX": 0, "K": 1, "DST": 1})),
    ]
    drafted = check_engine(players, cfgs[:1] if a.quick else cfgs)
    check_visuals(players, cfgs[0][1], drafted)
    check_matching(players)
    for lid in a.league:
        check_league(lid)
    if not a.quick:
        # ⭐ the rendered-UI pass. Every 2026-08-21 defect was a DISPLAY bug that
        # the engine checks and AppTest both passed -- a chip labelled "Jr. · WR",
        # ADP printing "None", a constant column painted solid red. This walks a
        # full mock and asserts on the strings the UI would actually show.
        head("RENDERED UI (full mock, every pick)")
        import subprocess
        # ⚠ encoding="utf-8": the child writes utf-8, and text=True otherwise
        # decodes with the locale codepage, mangling every · and — it relays
        r = subprocess.run([sys.executable, str(HERE / "mock_render_check.py"),
                            "--seeds", "1"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           cwd=str(HERE), env={**__import__("os").environ,
                                               "PYTHONIOENCODING": "utf-8"})
        out = (r.stdout or "") + (r.stderr or "")
        for line in out.splitlines():
            if line.strip().startswith(("✗", "✓ all", "seed ", "suffix", "tables cont",
                                        "grids cont", "(no suffixed")):
                print("  " + line.strip())
        (ok if r.returncode == 0 else bad)(
            "rendered-UI mock" + ("" if r.returncode == 0 else
                                  f" FAILED — run `python mock_render_check.py`"))
        check_app(a.league[0] if a.league else None)
    if a.phone:
        check_phone()

    head("RESULT")
    for w in WARNS:
        print(f"  ! {w}")
    if FAILS:
        print(f"\n  ✗ {len(FAILS)} FAILURE(S):")
        for f in FAILS:
            print(f"      - {f}")
        sys.exit(1)
    print(f"\n  ✓ ALL GREEN ({len(WARNS)} warning(s))")


if __name__ == "__main__":
    main()
