# Draft day

## The morning of

**1. Refresh the board.** ADP moves daily in August — a week-old board is a
week-old market. From the OUTER repo (`C:\dev\dfs_nfl`):

```bash
python adp_ingest.py && python walter_ingest.py && python consensus.py && python export_draft_board.py && python import_leagues.py
```

`import_leagues.py` last: it re-reads your leagues from Sleeper, and a league
that has been edited since the last run is exactly what this catches.

**2. Run the preflight.** From `C:\dev\dfs_nfl\draft_wizard`:

```bash
python preflight.py --league <your_league_id> --phone
```

Exit 0 and `ALL GREEN` or do not draft off it. It drives the real app in all
three modes, runs a full 180-pick mock, checks the live league against your saved
preset, verifies Sleeper's picks can be matched, then repeats the whole thing
inside an isolated copy of the bundle with the outer repo off the path — because
the phone failure mode is a **blank panel, not a crash**.

**3. Ship it to the phone.**

```bash
git -C draft_wizard add -A && git -C draft_wizard commit -m "Board refresh" && git -C draft_wizard push origin main
```

Streamlit Cloud redeploys on push. ⚠ `git push` under PowerShell 5.1 reports
`NativeCommandError` on a *successful* push (git writes progress to stderr) —
trust `git rev-parse HEAD` vs `origin/main`, not the wrapper's exit code.

---

## At the draft

1. Sidebar → **My Sleeper leagues** → pick the league → **📲 Load this league's
   draft**. Do this instead of setting things by hand: it reads your slot, the
   draft type, the true round count and the **live starting lineup** from Sleeper
   and overrides the widgets. If the saved preset disagreed, a banner says what
   it overrode.
2. Turn on **🔄 Auto-sync**. It polls every 12s and only reruns when the pick
   count actually moves. A dropped request leaves the last good board on screen.
3. On a phone, turn on **Compact** above the board.

### Reading the screen

* **Status bar** — the pick, who owns it *by name*, your next pick and how far.
* **⭐ card** — the pick, with why: survival %, tier cliff, roster hole, run risk.
* **🕐 On deck** — every seat between now and your turn, in order, with what each
  one needs. This is who is drafting where *before* it happens.
* **Best available** — defaults to **⭐ For me**: the wizard's own ranking, your
  roster and the tier cliffs priced in. **Raw value** is best-player-left ignoring
  your roster; **Market ADP** is the order the room will take them. `Score` is
  the number the pick is made on — a kicker can lead on VOR and be last on Score.
* **vsADP** — picks past his own ADP he has already fallen. `+20` is a bargain
  here; `−20` means taking him now is a reach.
* **🗺️ Draft board** tab — the grid, real team names, your seat starred, traded
  picks flagged.
* **👥 Team rosters** tab — every seat's shape and which starting slots they still
  owe. A team with no QB in round 12 is taking a QB.

### If something goes wrong

| Symptom | Do this |
|---|---|
| Picks stop updating | Toggle Auto-sync off/on, or hit **🔁 Sync** |
| A pick shows as a placeholder | Harmless — pick numbers stay aligned. ~5% of deep bench players aren't on our board |
| A drafted player still shows as available | **🔍 Mark a pick** tab → search him → Mark drafted |
| Wrong slot / round count | Re-run **📲 Load this league's draft**; Sleeper is authoritative |
| Sleeper is down | Everything still works manually — tap rows in Best available to mark picks |

---

## Known limits

* Our board is **offense + K/DST only**. IDP picks (none in ReDrafters Rejoice)
  show as placeholders.
* The **draft grade** ranks your roster's positive VOR against the other teams
  *on our own valuation*. It is not a projected finish.
* **Best Ball mode is not validated** to beat the redraft engine — see
  `BB_REPL_WEIGHT` in `draft_engine.py`. The forward planner is disabled there
  on purpose rather than shown with a caveat.
* Pages **7 (ZAP Prospects)** and **8 (Player Profile)** read local JSON exports;
  parts of them degrade on the phone. Nothing the draft page needs.
