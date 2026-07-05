"""
draft_names.py (deploy-bundle copy)
-----------------------------------
Name normalization for the app's Sleeper-metadata matching and manual search.
Mirrors the root draft_names.py; the private pipeline already matched the board,
so this copy only needs to map live Sleeper names onto the exported board.
"""
from __future__ import annotations

import re

_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$", flags=re.IGNORECASE)
_PUNCT_RE  = re.compile(r"[.'’]")
_WS_RE     = re.compile(r"\s+")

ALIASES = {
    "hollywood brown": "marquise brown",
    "gabriel davis": "gabe davis",
    "cameron ward": "cam ward",
    "chigoziem okonkwo": "chig okonkwo",
    "joshua palmer": "josh palmer",
    "nathaniel dell": "tank dell",
    "damario douglas": "demario douglas",
    "gardner minshew ii": "gardner minshew",
    "scott miller": "scotty miller",
    "joshua kelley": "josh kelley",
    "chris brooks": "christopher brooks",
}


def norm_name(name, lastfirst: bool = False) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    if lastfirst and "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    s = s.lower()
    s = _PUNCT_RE.sub("", s)
    s = s.replace("-", " ")
    s = _WS_RE.sub(" ", s).strip()
    s = _SUFFIX_RE.sub("", s).strip()
    s = _WS_RE.sub(" ", s).strip()
    if s in ALIASES:
        s = ALIASES[s]
        s = _SUFFIX_RE.sub("", s).strip()
    return s


def pos_norm(pos) -> str:
    p = str(pos or "").upper().strip()
    return {"PK": "K", "DEF": "DST", "DST": "DST", "D/ST": "DST"}.get(p, p)
