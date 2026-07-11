"""
sleeper_api.py
--------------
Thin Sleeper read-API client for the draft board: resolve a league's current
draft, pull picks, team names, and traded-pick ownership. Pure (no Streamlit),
so it can be tested headlessly.
"""

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


def league_state(league_id: str | None = None, draft_id: str | None = None) -> dict:
    """Everything the board + sync need: config, picks, per-slot team names, and a
    (round, original_roster) -> current_owner_roster map for traded picks."""
    if not draft_id:
        draft_id = resolve_draft_id(league_id)
    draft = _get(f"{BASE}/draft/{draft_id}")
    league_id = league_id or draft.get("league_id")

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

    return {
        "draft_id": draft_id, "league_id": league_id,
        "type": draft.get("type"), "snake": draft.get("type") == "snake",
        "status": draft.get("status"),
        "teams": settings.get("teams"), "rounds": settings.get("rounds"),
        "my_slot": order.get(SLEEPER_USER_ID), "my_roster": my_roster,
        "slot_to_roster": slot_to_roster, "roster_team": roster_team,
        "traded": tmap, "picks": sorted(picks, key=lambda x: x.get("pick_no", 0)),
    }


def pick_owner(state: dict, rnd: int, slot: int):
    """(current_team, was_traded, original_team) for the pick at (round, slot)."""
    orig = state["slot_to_roster"].get(slot)
    cur = state["traded"].get((rnd, orig), orig)
    return (state["roster_team"].get(cur, "?"), cur != orig, state["roster_team"].get(orig, "?"))
