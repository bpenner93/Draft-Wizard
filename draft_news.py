"""
draft_news.py  --  Draft Wizard NFL beat feed
=============================================
Our own NFL news feed, built from **free, no-API-key** public sources and
tagged to the players on your draft board so the news that matters during a
draft (injuries, holdouts, depth-chart moves, camp buzz) shows up next to the
player it's about.

Why not a paid provider? Individual beat writers post to X/Twitter, whose API
is now paid and rate-limited. Instead we aggregate the public RSS/JSON feeds of
outlets whose reporting *is* beat-writer work (ESPN, CBS, Yahoo, ProFootballTalk,
Yardbarker's Rumor Mill) and player-tag every headline. It costs nothing, needs
no key, and runs fine on Streamlit Community Cloud.

Design notes
------------
- Pure Python: `requests` (already a dep) + stdlib xml/email/html. No feedparser.
- Every source is fetched independently and defensively; one dead feed never
  breaks the rest. `fetch_news()` returns whatever it got plus a list of errors.
- Player tagging reuses the board's own names (no external id mapping) and errs
  toward precision: full-name match > unique "F. Last" > globally-unique surname.
- The app layer owns caching (Streamlit `@st.cache_data(ttl=...)`); this module
  stays free of Streamlit so it's testable on its own.
"""
from __future__ import annotations

import html
import re
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests

# --------------------------------------------------------------------------- sources
# Free, public, no-key feeds. `kind` selects the parser. Add/remove freely;
# a broken or renamed feed is skipped, not fatal.
SOURCES = [
    {"name": "ESPN",            "kind": "espn_json",
     "url": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?limit=50"},
    {"name": "ESPN",            "kind": "rss", "url": "https://www.espn.com/espn/rss/nfl/news"},
    {"name": "ProFootballTalk", "kind": "rss", "url": "https://profootballtalk.nbcsports.com/feed/"},
    {"name": "CBS Sports",      "kind": "rss", "url": "https://www.cbssports.com/rss/headlines/nfl/"},
    {"name": "Yahoo Sports",    "kind": "rss", "url": "https://sports.yahoo.com/nfl/rss/"},
    {"name": "Yardbarker NFL",  "kind": "rss", "url": "https://www.yardbarker.com/rss/sport/2"},
]

_UA = ("Mozilla/5.0 (compatible; DraftWizard/1.0; +https://github.com/) "
       "news-aggregator")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

# --------------------------------------------------------------------------- team writers
# Board team abbr -> full name. Individual beat writers publish on X (paid API),
# but their *articles* are aggregated for free by Google News RSS per-team search,
# which surfaces The Athletic, local-paper beat reporters, PFF, team sites, etc.
TEAMS = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

# Terms that bias a team's Google-News search toward beat-writer roster reporting
# (the stuff that moves draft value) rather than gameday recaps.
_TEAM_QUERY_TERMS = ("injury OR practice OR starter OR \"depth chart\" OR snaps OR "
                     "questionable OR ruled OR backup OR return OR IR")


def _google_news_url(query: str, when_days: int = 3) -> str:
    q = quote_plus(f"{query} {_TEAM_QUERY_TERMS} when:{when_days}d")
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def team_sources(teams=None, when_days: int = 3) -> list[dict]:
    """A Google-News-RSS 'beat writer' feed per team. `teams` = iterable of board
    abbreviations (default: all 32). Each item is stamped with its team."""
    abbrs = list(teams) if teams is not None else list(TEAMS)
    out = []
    for ab in abbrs:
        full = TEAMS.get(ab)
        if not full:
            continue
        out.append({"name": f"Beat · {ab}", "kind": "rss", "team": ab,
                    "url": _google_news_url(f'"{full}"', when_days)})
    return out


# --------------------------------------------------------------------------- text utils
def _norm_text(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for name matching."""
    if not s:
        return ""
    s = html.unescape(str(s)).lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _strip_html(s: str, limit: int = 240) -> str:
    """Turn an RSS description/summary into a short plain-text blurb."""
    if not s:
        return ""
    s = _TAG_RE.sub("", str(s))
    s = html.unescape(s)
    s = _WS_RE.sub(" ", s).strip()
    return (s[: limit - 1] + "…") if len(s) > limit else s


def _epoch(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_rss_date(text: str | None) -> float | None:
    if not text:
        return None
    try:                                    # RFC-822 (RSS pubDate)
        return _epoch(parsedate_to_datetime(text))
    except (TypeError, ValueError):
        pass
    try:                                    # ISO-8601 (Atom / some RSS)
        return _epoch(datetime.fromisoformat(text.strip().replace("Z", "+00:00")))
    except ValueError:
        return None


def ago(published: float | None, now: float | None = None) -> str:
    """'3h ago' / '2d ago' relative label; '' when the timestamp is unknown."""
    if not published:
        return ""
    now = time.time() if now is None else now
    d = max(0, int(now - published))
    if d < 90:
        return "just now"
    if d < 3600:
        return f"{d // 60}m ago"
    if d < 86400:
        return f"{d // 3600}h ago"
    return f"{d // 86400}d ago"


# --------------------------------------------------------------------------- fetch + parse
def _get(url: str, timeout: float) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": _UA})
    r.raise_for_status()
    return r.text


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_rss(text: str, source: str) -> list[dict]:
    """Parse RSS 2.0 (<item>) and Atom (<entry>); namespace-agnostic."""
    root = ET.fromstring(text)
    items: list[dict] = []
    for el in root.iter():
        if _localname(el.tag) not in ("item", "entry"):
            continue
        f: dict[str, str] = {}
        link = ""
        byline = ""
        for child in el:
            name = _localname(child.tag)
            if name == "link":
                # Atom links carry the href on an attribute; RSS in the text.
                link = link or child.get("href") or (child.text or "").strip()
            elif name == "source":
                # Google News tags the originating publisher (the beat outlet) here.
                byline = (child.text or "").strip()
            elif name in ("title", "description", "summary", "pubDate",
                          "published", "updated", "content"):
                f.setdefault(name, (child.text or "").strip())
        title = _strip_html(f.get("title", ""), limit=200)
        if not title:
            continue
        # Google News formats titles "Headline - Publisher"; peel the byline off.
        if not byline and " - " in title:
            head, _, pub = title.rpartition(" - ")
            if head and len(pub) < 40:
                title, byline = head, pub
        published = _parse_rss_date(f.get("pubDate") or f.get("published") or f.get("updated"))
        summary = _strip_html(f.get("description") or f.get("summary") or f.get("content") or "")
        items.append({"title": title, "url": link, "source": source, "byline": byline,
                      "published": published, "summary": summary, "player_ids": []})
    return items


def _parse_espn_json(text: str, source: str) -> list[dict]:
    """ESPN's public news JSON. Also carries athlete tags we keep as a hint."""
    import json
    data = json.loads(text)
    items: list[dict] = []
    for a in data.get("articles", []):
        title = _strip_html(a.get("headline", ""), limit=200)
        if not title:
            continue
        link = (((a.get("links") or {}).get("web") or {}).get("href") or "").strip()
        published = _parse_rss_date(a.get("published") or a.get("lastModified"))
        athletes = [c.get("description", "") for c in (a.get("categories") or [])
                    if c.get("type") == "athlete" and c.get("description")]
        items.append({"title": title, "url": link, "source": source,
                      "published": published, "summary": _strip_html(a.get("description", "")),
                      "player_ids": [], "_athletes": athletes})
    return items


# --------------------------------------------------------------------------- player tagging
def build_player_index(board: list[dict]) -> dict:
    """Precompute name lookups from the board for fast, precise tagging."""
    players, last_counts, flast_counts = [], {}, {}
    for p in board:
        nn = _norm_text(p.get("name", ""))
        parts = nn.split()
        if len(parts) < 2:
            continue
        first, last, fi = parts[0], parts[-1], parts[0][:1]
        players.append({"id": p["id"], "name": p.get("name"), "pos": p.get("pos"),
                        "team": _norm_text(p.get("team") or ""),
                        "full": nn, "last": last, "fi": fi})
        last_counts[last] = last_counts.get(last, 0) + 1
        flast_counts[(fi, last)] = flast_counts.get((fi, last), 0) + 1
    return {"players": players, "last_counts": last_counts, "flast_counts": flast_counts}


def match_players(text: str, index: dict, item_team: str | None = None) -> dict[str, int]:
    """Return {player_id: confidence 1-4} for players named in `text`.

    3 = full name · 2 = unambiguous 'F. Last' · 1 = globally-unique surname;
    +1 when the player's team is mentioned in the text (disambiguates shared
    names). `item_team` (from a per-team beat feed) rescues an on-team surname
    even when it isn't globally unique — the feed already scoped us to that team.
    """
    it_team = _norm_text(item_team or "")
    t = " " + _norm_text(text) + " "
    words = set(t.split())
    hits: dict[str, int] = {}
    for p in index["players"]:
        if p["last"] not in words:          # cheap reject before substring work
            continue
        on_team = bool(it_team) and p["team"] == it_team
        conf = 0
        if (" " + p["full"] + " ") in t:
            conf = 3
        elif index["flast_counts"].get((p["fi"], p["last"]), 0) == 1 and (
                (" " + p["fi"] + " " + p["last"] + " ") in t
                or (" " + p["fi"] + p["last"] + " ") in t):
            conf = 2
        elif index["last_counts"].get(p["last"], 0) == 1:
            conf = 1
        elif on_team:                       # surname inside its own team's beat feed
            conf = 2
        if not conf:
            continue
        if p["team"] and ((" " + p["team"] + " ") in t or on_team):
            conf += 1
        hits[p["id"]] = max(hits.get(p["id"], 0), conf)
    return hits


def tag_items(items: list[dict], index: dict, by_name: dict | None = None) -> list[dict]:
    """Attach matched player ids to each item (in place) and return the list."""
    for it in items:
        text = f"{it.get('title','')} . {it.get('summary','')}"
        hits = match_players(text, index, item_team=it.get("team"))
        # Fold in ESPN's own athlete tags when we can resolve them by name.
        if by_name:
            for nm in it.pop("_athletes", []) or []:
                pid = by_name.get(_norm_text(nm))
                if pid:
                    hits[pid] = max(hits.get(pid, 0), 3)
        it.pop("_athletes", None)
        it["player_ids"] = sorted(hits, key=lambda pid: -hits[pid])
        it["player_conf"] = hits
    return items


# --------------------------------------------------------------------------- public API
def fetch_news(board: list[dict] | None = None, *, timeout: float = 8.0,
               per_source: int = 25, sources: list[dict] | None = None) -> dict:
    """Fetch, de-dupe, player-tag, and time-sort the NFL feed.

    Returns {"items": [...], "errors": [(source, msg)], "fetched_at": epoch}.
    Never raises for a single bad feed — failures land in "errors".
    """
    srcs = sources if sources is not None else SOURCES
    raw: list[dict] = []
    errors: list[tuple[str, str]] = []

    def _fetch_one(s: dict):
        text = _get(s["url"], timeout)
        parsed = (_parse_espn_json if s["kind"] == "espn_json" else _parse_rss)(text, s["name"])
        for it in parsed[:per_source]:
            it.setdefault("team", s.get("team"))        # beat feeds know their team
            it.setdefault("byline", "")
        return parsed[:per_source]

    # Feeds are independent and I/O-bound — fetch them concurrently so 38 team
    # beat feeds don't serialize into a 30s wait. One dead feed still can't
    # break the rest.
    if len(srcs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(12, len(srcs))) as ex:
            futures = {ex.submit(_fetch_one, s): s for s in srcs}
            for fut, s in ((f, futures[f]) for f in futures):
                try:
                    raw.extend(fut.result())
                except Exception as e:
                    errors.append((s["name"], f"{type(e).__name__}: {e}"))
    else:
        for s in srcs:
            try:
                raw.extend(_fetch_one(s))
            except Exception as e:                      # network, parse, HTTP — all non-fatal
                errors.append((s["name"], f"{type(e).__name__}: {e}"))

    # De-dupe by url (fallback: normalized title).
    seen, items = set(), []
    for it in raw:
        key = it.get("url") or _norm_text(it.get("title", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(it)

    if board:
        index = build_player_index(board)
        by_name = {p["full"]: p["id"] for p in index["players"]}
        tag_items(items, index, by_name)

    items.sort(key=lambda it: (it.get("published") or 0), reverse=True)
    return {"items": items, "errors": errors, "fetched_at": time.time()}


def items_for_player(news: dict, player_id: str, limit: int = 5) -> list[dict]:
    """Latest tagged items for one player (most recent first)."""
    out = [it for it in news.get("items", []) if player_id in it.get("player_ids", [])]
    return out[:limit]


def player_ids_in_news(news: dict) -> set[str]:
    ids: set[str] = set()
    for it in news.get("items", []):
        ids.update(it.get("player_ids", []))
    return ids


# --------------------------------------------------------------------------- signal extraction
# Turn a beat-writer headline into structured draft signals. Ordered most→least
# severe; the first match in each group wins. Keyword-based and deliberately
# conservative — a miss just means "no signal", not a wrong one.
_INJURY_RULES = [
    ("out",          ["ruled out", "will not play", "won't play", "is inactive",
                       "declared out", "will miss", "sidelined for"]),
    ("ir",           ["injured reserve", "placed on ir", "moved to ir", "to ir",
                       "reserve/pup", "pup list", "non-football injury", "carted off",
                       "season-ending", "out for the season", "torn "]),
    ("suspended",    ["suspended", "suspension"]),
    ("doubtful",     ["doubtful"]),
    ("questionable", ["questionable", "game-time decision", "gametime decision",
                      "game time decision"]),
    ("limited",      ["did not practice", "didn't practice", "dnp", "limited in practice",
                      "limited participant", "day-to-day", "day to day", "banged up",
                      "dealing with", "left the game", "exited"]),
    ("returning",    ["activated off", "activated from", "designated to return", "returns to practice",
                      "cleared to play", "full practice", "off the injury report",
                      "returns from", "back at practice", "expected to play"]),
]
_ROLE_RULES = [
    ("promoted", ["named the starter", "named starter", "will start", "gets the start",
                  "takes over as", "atop the depth chart", "first-team reps", "starting job",
                  "earns the starting", "promoted to", "the rb1", "the wr1", "the te1",
                  "lead back", "bell cow", "workhorse"]),
    ("benched",  ["benched", "demoted", "loses the starting", "lost the starting",
                  "lost his starting", "dropped to second", "no longer the starter",
                  "out of the starting"]),
    ("backup",   ["backup", "second-string", "second string", "in a committee",
                  "committee", "timeshare", "time share", "split carries",
                  "rotation", "behind the starter"]),
]

# How each signal scales a player's projection when "influence picks" is on.
# Multiplicative, then clamped — a single bad parse can't zero out a stud
# (except an explicit out/IR/suspension, which is the whole point).
INJURY_FACTOR = {"out": 0.12, "ir": 0.10, "suspended": 0.10, "doubtful": 0.50,
                 "questionable": 0.90, "limited": 0.95, "returning": 1.0}
ROLE_FACTOR = {"promoted": 1.06, "benched": 0.70, "backup": 0.72}
_SEVERITY = {lbl: i for i, (lbl, _) in enumerate(_INJURY_RULES)}


def classify_text(text: str) -> dict:
    """Best-effort {injury, role, phrase} from one item's text (or empty)."""
    t = _norm_text(text)
    out = {"injury": None, "role": None, "phrase": ""}
    for label, kws in _INJURY_RULES:
        hit = next((k for k in kws if k in t), None)
        if hit:
            out["injury"], out["phrase"] = label, hit.strip()
            break
    for label, kws in _ROLE_RULES:
        hit = next((k for k in kws if k in t), None)
        if hit:
            out["role"] = label
            out["phrase"] = out["phrase"] or hit.strip()
            break
    return out


def status_factor(injury: str | None, role: str | None) -> float:
    f = INJURY_FACTOR.get(injury or "", 1.0) * ROLE_FACTOR.get(role or "", 1.0)
    return round(max(0.10, min(1.10, f)), 3)


def status_label(st: dict) -> str:
    """Short human badge, e.g. '⛔ OUT' / '🔴 IR' / '🟡 questionable' / '↩︎ returning' / '↕︎ backup'."""
    inj, role = st.get("injury"), st.get("role")
    icon = {"out": "⛔ OUT", "ir": "🔴 IR", "suspended": "🚫 SUSP", "doubtful": "🟠 doubtful",
            "questionable": "🟡 quest.", "limited": "🟡 limited", "returning": "↩︎ returning"}
    parts = []
    if inj:
        parts.append(icon.get(inj, inj))
    if role:
        parts.append({"promoted": "⬆︎ starter", "benched": "⬇︎ benched",
                      "backup": "↕︎ backup"}.get(role, role))
    return " · ".join(parts)


def player_status(news: dict) -> dict:
    """Roll the tagged feed up into one current status per player.

    Attributes each item's signal to its single most-confident player (avoids
    spilling 'Player A ruled out' onto Player B mentioned in the same note).
    Items are already newest-first, so the first signal we see per player wins,
    and we keep the most *severe* injury seen in the recent window.
    Returns {player_id: {injury, role, factor, label, phrase, byline, source,
                         team, url, title, published}}.
    """
    status: dict[str, dict] = {}
    for it in news.get("items", []):
        ids = it.get("player_ids", [])
        if not ids:
            continue
        pid = ids[0]                                        # top-confidence player only
        sig = classify_text(f"{it.get('title','')} . {it.get('summary','')}")
        if not sig["injury"] and not sig["role"]:
            continue
        cur = status.get(pid)
        if cur is None:
            status[pid] = {
                "injury": sig["injury"], "role": sig["role"], "phrase": sig["phrase"],
                "byline": it.get("byline", ""), "source": it.get("source", ""),
                "team": it.get("team"), "url": it.get("url", ""),
                "title": it.get("title", ""), "published": it.get("published"),
            }
        else:
            # keep newest role/headline (already set), but escalate to a more
            # severe injury if an older-but-worse note exists in the window
            if sig["injury"] and (cur["injury"] is None or
                                  _SEVERITY.get(sig["injury"], 99) < _SEVERITY.get(cur["injury"], 99)):
                cur["injury"] = sig["injury"]
            if sig["role"] and cur["role"] is None:
                cur["role"] = sig["role"]
    for pid, st in status.items():
        st["factor"] = status_factor(st["injury"], st["role"])
        st["label"] = status_label(st)
    return status


def apply_status(board: list[dict], status: dict) -> tuple[list[dict], list[dict]]:
    """Return (adjusted_board, changes). Scales each flagged player's projection
    by its status factor so VOR/tiers/recommendations react. Non-destructive:
    operates on copies and leaves the input board untouched. `changes` lists the
    players whose value moved, for display."""
    changes = []
    out = []
    for p in board:
        st = status.get(p["id"])
        if not st or st["factor"] == 1.0:
            out.append(p)
            continue
        q = dict(p)
        f = st["factor"]
        for fld in ("pts", "rec", "dyn", "dp1", "dp2"):
            if isinstance(q.get(fld), (int, float)):
                q[fld] = q[fld] * f
        q["_status"] = st
        out.append(q)
        changes.append({"id": p["id"], "name": p.get("name"), "pos": p.get("pos"),
                        "team": p.get("team"), "factor": f, "label": st["label"]})
    changes.sort(key=lambda c: c["factor"])                 # biggest downgrades first
    return out, changes


# --------------------------------------------------------------------------- source health
def check_sources(sources: list[dict] | None = None, *, timeout: float = 8.0) -> list[dict]:
    """Ping each feed and report {name, kind, ok, count, ms, error, team} so the
    user can see, on their live deployment, which sources are actually up. This
    is the reliability check — run it from the app's 'Source health' panel or
    `python draft_news.py --check`."""
    srcs = sources if sources is not None else SOURCES
    rows = []
    for s in srcs:
        t0 = time.time()
        ok, count, err = False, 0, ""
        try:
            text = _get(s["url"], timeout)
            parsed = (_parse_espn_json if s["kind"] == "espn_json" else _parse_rss)(text, s["name"])
            ok, count = True, len(parsed)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        rows.append({"name": s["name"], "kind": s["kind"], "team": s.get("team"),
                     "ok": ok, "count": count, "ms": int((time.time() - t0) * 1000),
                     "error": err})
    return rows


# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":                                  # pragma: no cover
    import sys
    from pathlib import Path

    argv = sys.argv[1:]
    board_path = Path(__file__).parent / "data" / "draft_board.json"
    board = None
    if board_path.exists():
        import json
        board = json.load(open(board_path, encoding="utf-8")).get("players")

    if "--check" in argv:
        n = 4
        srcs = SOURCES + team_sources(list(TEAMS)[:n])
        print(f"Checking {len(srcs)} sources ({n} sample beat feeds)…\n")
        for r in check_sources(srcs, timeout=10):
            flag = "ok " if r["ok"] else "FAIL"
            tag = f" [{r['team']}]" if r["team"] else ""
            print(f"  [{flag}] {r['name']}{tag:8} {r['count']:>3} items  {r['ms']:>5}ms"
                  + (f"  {r['error']}" if r["error"] else ""))
    else:
        news = fetch_news(board, sources=SOURCES + team_sources(list(TEAMS)[:6]))
        byid = {p["id"]: p for p in (board or [])}
        print(f"{len(news['items'])} items, {len(news['errors'])} source errors\n")
        for it in news["items"][:25]:
            names = ", ".join(byid.get(pid, {}).get("name", pid) for pid in it["player_ids"][:3])
            when = ago(it["published"])
            print(f"  [{when:>8}] {it['title'][:70]}")
            if names:
                print(f"             ↳ {names}  ({it.get('byline') or it['source']})")
        st = player_status(news)
        if st:
            print("\nStatus signals:")
            for pid, s in list(st.items())[:15]:
                print(f"  {byid.get(pid,{}).get('name',pid):22} {s['label']:22} ×{s['factor']}  — {s['title'][:45]}")
