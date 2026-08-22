"""
phone_probe.py -- run INSIDE an isolated copy of the bundle (see preflight.check_phone).
=======================================================================================
The deployed app is its own git repo on Streamlit Community Cloud and cannot see the
outer pipeline: no player_history, no data/processed/*.parquet, no season_rankings.

The failure mode that matters is NOT a crash. The 2026-08-01 audit found career-arc
columns written against the outer repo that on the phone silently returned EMPTY --
no error, just a blank column, which is the worst thing that can happen mid-draft.
So this probe asserts on VALUES, not on "it imported".

Prints machine-readable KEY=value lines for preflight.py to grade, then PHONE_OK.
"""
import importlib
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# anything here that imports is a module the phone will NOT have
leaked = []
for m in ("player_history", "season_rankings", "consensus", "export_draft_board",
          "draft_tune", "league_rankings", "adp_ingest"):
    try:
        importlib.import_module(m)
        leaked.append(m)
    except Exception:
        pass
print("LEAKED=" + ",".join(leaked))

import draft_engine as E          # noqa: E402
import draft_board, bestball, sleeper_api, draft_names   # noqa: E402,F401

from streamlit.testing.v1 import AppTest   # noqa: E402

at = AppTest.from_file("draft_app.py", default_timeout=300)
at.run()
assert not at.exception, at.exception[0].value
[b for b in at.button if "Start draft" in b.label][0].click().run()
assert not at.exception, at.exception[0].value
print("TABLES=%d" % len(at.dataframe))

board = E.load_board()
cfg = E.LeagueConfig(teams=12, scoring="ppr", superflex=False, my_slot=10,
                     snake=True, rounds=15, bench=6,
                     starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2,
                               "SUPERFLEX": 0, "K": 0, "DST": 1})
r = E.analyze(board, cfg, [])
ba = r["best_available"][:40]
# these are exactly the columns that were blank-on-phone before they were baked in
print("ARC=%d/%d" % (sum(1 for x in ba if x.get("arc") is not None), len(ba)))
print("DECL=%d/%d" % (sum(1 for x in ba if x.get("decl") is not None), len(ba)))
# ⚠ read these off `best_available`, NOT off `board`. _slim() copies a FIXED key
# set, so a column can be populated on every one of the 680 board players and
# still arrive at the UI as None -- which is exactly what `bye` did. Checking the
# board tests the input; the UI is the consumer.
print("BYE=%d/%d" % (sum(1 for x in ba if x.get("bye") is not None), len(ba)))
print("ADP=%d/%d" % (sum(1 for x in ba if x.get("adp") is not None), len(ba)))
print("SURV=%d/%d" % (sum(1 for x in ba if x.get("survival") is not None), len(ba)))

# the visual layer, rendered under phone conditions
by_id = {p["id"]: p for p in board}
import numpy as np      # noqa: E402
drafted = E.mock_advance(E.prep_valued(board, cfg), cfg, [], np.random.default_rng(3))
drafted = E.mock_advance(E.prep_valued(board, cfg), cfg, drafted + [r["best_available"][0]["id"]],
                         np.random.default_rng(4))
r2 = E.analyze(board, cfg, drafted)
panels = {
    "STATUS": draft_board.status_bar_html(cfg, None, r2, "Team X"),
    "ONDECK": draft_board.on_deck_html(cfg, None, r2["current_overall"], r2["my_next_pick"],
                                       r2["opponents"].get("seat_need")),
    "GRID": draft_board.draft_board_html(cfg, None, drafted, by_id),
    "RUNSTRIP": draft_board.run_strip_html(cfg, drafted, by_id),
    "MATRIX": draft_board.roster_matrix_html(cfg, None, drafted, by_id, cfg.starters),
}
for k, v in panels.items():
    print("PANEL_%s=%d" % (k, len(v)))

print("PHONE_OK")
