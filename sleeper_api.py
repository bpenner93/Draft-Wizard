"""
sleeper_api.py
--------------
Thin Sleeper read-API client for the draft board: resolve a league's current
draft, pull picks, team names, and traded-pick ownership. Pure (no Streamlit),
so it can be tested headlessly.
"""

from collections import Counter

import requests

BASE = "https://api.sleeper.app/v1"
SLEEPER_USER_ID = "430840397841838080"   # PennerBoy


def _get(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def resolve_draft_id(league_id: str) -> str | None:
    """Newest draft for a league (latest season / start_time)."""
    drafts = _get(f"{BASE}/league/{league_id}/drafts")
    if not drafts:
        return None
    d = sorted(drafts, key=lambda x: (str(x.get("season") or ""), x.get("start_time") or 0))[-1]
    return d["draft_id"]


def _scoring_label(rec: float) -> str:
    return "ppr" if rec >= 1.0 else ("half" if rec >= 0.5 else "standard")


def roster_config(league: dict) -> dict:
    """The league's ACTUAL starting lineup, bench, superflex flag and scoring, read
    from `roster_positions` on the live league object.

    ⭐ This exists because a saved preset is a SNAPSHOT and leagues get edited.
    ReDrafters Rejoice was imported 2026-08-06 as 12T **Superflex** / 16 rounds;
    by 2026-08-21 the commissioner had removed the SUPER_FLEX slot and cut a
    round, and nothing told the app. Superflex moves QB replacement from QB12 to
    ~QB24, so Josh Allen read **VOR 149, board slot 6** under the stale preset vs
    **VOR 38, board slot 47** under the true 1QB settings -- i.e. the wizard would
    have recommended a QB in round 1 of a single-QB league, with no error anywhere.
    The live league is the only authority; presets are a convenience."""
    c = Counter(league.get("roster_positions") or [])
    sf = c.get("SUPER_FLEX", 0) + c.get("SUPERFLEX", 0)
    starters = {
        "QB": c.get("QB", 0), "RB": c.get("RB", 0), "WR": c.get("WR", 0),
        "TE": c.get("TE", 0),
        "FLEX": c.get("FLEX", 0) + c.get("WRRB_FLEX", 0) + c.get("REC_FLEX", 0),
        "SUPERFLEX": sf, "K": c.get("K", 0), "DST": c.get("DEF", 0) + c.get("DST", 0),
    }
    sc = league.get("scoring_settings") or {}
    return {
        "starters": starters,
        "superflex": sf > 0,
        "bench": int(c.get("BN", 0)),
        "scoring": _scoring_label(float(sc.get("rec", 0) or 0)),
        "te_premium": float(sc.get("bonus_rec_te", 0) or 0) or None,
        # IDP slots exist in the dynasty leagues and our board is offense-only;
        # surfaced so the UI can say so rather than look broken.
        "idp_slots": sum(v for k, v in c.items()
                         if k in ("DL", "LB", "DB", "IDP_FLEX", "DE", "DT", "CB", "S")),
    }


def league_state(league_id: str | None = None, draft_id: str | None = None) -> dict:
    """Everything the board + sync need: config, picks, per-slot team names, and a
    (round, original_roster) -> current_owner_roster map for traded picks."""
    if not draft_id:
        draft_id = resolve_draft_id(league_id)
    draft = _get(f"{BASE}/draft/{draft_id}")
    league_id = league_id or draft.get("league_id")

    league = _get(f"{BASE}/league/{league_id}") if league_id else {}
    users = _get(f"{BASE}/league/{league_id}/users") if league_id else []
    rosters = _get(f"{BASE}/league/{league_id}/rosters") if league_id else []
    traded = _get(f"{BASE}/league/{league_id}/traded_picks") if league_id else []
    picks = _get(f"{BASE}/draft/{draft_id}/picks")

    uname = {u["user_id"]: ((u.get("metadata") or {}).get("team_name") or u.get("display_name") or "Team")
             for u in users}
    roster_owner = {r["roster_id"]: r.get("owner_id") for r in rosters}
    roster_team = {rid: uname.get(oid, f"Team {rid}") for rid, oid in roster_owner.items()}
    slot_to_roster = {int(k): v for k, v in (draft.get("slot_to_roster_id") or {}).items()}

    settings = draft.get("settings") or {}
    order = draft.get("draft_order") or {}
    my_roster = next((r["roster_id"] for r in rosters if r.get("owner_id") == SLEEPER_USER_ID), None)

    season = str(draft.get("season") or "")
    tmap = {}
    for t in traded:
        if str(t.get("season")) == season:
            tmap[(t["round"], t["roster_id"])] = t["owner_id"]

    # the draft's own round count is authoritative; `settings.draft_rounds` on the
    # league is frequently absent or stale (here: 3, for a 15-round draft)
    rcfg = roster_config(league) if league else {}
    return {
        "draft_id": draft_id, "league_id": league_id,
        "league_name": league.get("name") if league else None,
        "type": draft.get("type"), "snake": draft.get("type") == "snake",
        "status": draft.get("status"), "start_time": draft.get("start_time"),
        "pick_timer": settings.get("pick_timer"),
        "roster_cfg": rcfg,
        "teams": settings.get("teams"), "rounds": settings.get("rounds"),
        "my_slot": order.get(SLEEPER_USER_ID), "my_roster": my_roster,
        "slot_to_roster": slot_to_roster, "roster_team": roster_team,
        # roster -> USER. Tendencies are keyed on user_id, never on seat or team
        # name: draft slots are redrawn every season and people rename their
        # teams, so anything else hands last year's habits to the wrong person.
        "roster_owner": roster_owner,
        "traded": tmap, "picks": sorted(picks, key=lambda x: x.get("pick_no", 0)),
    }


def pick_owner(state: dict, rnd: int, slot: int):
    """(current_team, was_traded, original_team) for the pick at (round, slot)."""
    orig = state["slot_to_roster"].get(slot)
    cur = state["traded"].get((rnd, orig), orig)
    return (state["roster_team"].get(cur, "?"), cur != orig, state["roster_team"].get(orig, "?"))
