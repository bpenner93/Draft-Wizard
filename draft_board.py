"""
draft_board.py
--------------
The visual layer of the draft room: a FantasyPros-style teams x rounds grid, the
on-deck queue ("who is drafting where between now and your turn"), a position-run
strip, and the roster-shape matrix for every seat.

Everything here is a PURE function returning an HTML string -- no Streamlit -- so
the whole visual layer can be rendered and asserted on headlessly. That matters:
these are the panels you read while a 2-minute pick clock is running, and the
only way to know they are right on draft day is to have checked them off the
clock.

⚠ No <script> anywhere. Streamlit's st.markdown(unsafe_allow_html=True) strips
scripts, so anything dynamic (scroll-to-current-round, timers) has to be done by
choosing what Python renders, not by JS. `rounds_window` exists for that reason.
"""

from draft_names import pos_norm

# position palette -- also used for the run strip and roster matrix so a colour
# means the same thing everywhere on screen
POS_BG = {"QB": "#7c3aed", "RB": "#059669", "WR": "#2563eb", "TE": "#d97706",
          "K": "#475569", "DST": "#475569"}
POS_DIM = {"QB": "#4c1d95", "RB": "#064e3b", "WR": "#1e3a8a", "TE": "#78350f",
           "K": "#1e293b", "DST": "#1e293b"}
LINE = "#334155"
INK = "#e2e8f0"
MUTE = "#94a3b8"
PANEL = "#0f172a"
FIELD = "#0b1220"
GOLD = "#f59e0b"
MINE = "#1d4ed8"


def _esc(s) -> str:
    return (str(s if s is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# a generational suffix is not part of the name you read on a draft board, and
# dropping it is what buys the room to show the rest ("M. Harrison Jr." vs
# "M. Harrison"). Also see surname() in draft_app.
_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def _short(name: str, n: int = 13) -> str:
    """`Amon-Ra St. Brown` -> `A. St. Brown`, then hard-truncate."""
    parts = [x for x in str(name or "").split() if x]
    # ⚠ a defense is named by its TEAM ("SEA DST"), so initialising the first
    # token turns it into "S. DST" -- which identifies nothing. Leave them whole.
    if parts and parts[-1].upper() in ("DST", "D/ST", "DEF"):
        return _esc(" ".join(parts))
    while len(parts) > 2 and parts[-1].lower().strip(".") in _SUFFIX:
        parts.pop()
    if len(parts) >= 2:
        name = f"{parts[0][:1]}. {' '.join(parts[1:])}"
    elif parts:
        name = parts[0]
    return _esc(name if len(name) <= n else name[: n - 1] + "…")


# --------------------------------------------------------------------------- seats
def slot_team(cfg, state, slot: int, short: bool = False) -> str:
    """Display name for a drafting slot. Real Sleeper team names when we have them.

    ⚠ Two different identifiers: `slot` is the seat in the draft order, `roster_id`
    is the franchise. Sleeper's slot_to_roster_id maps between them and they are
    NOT the same number -- your slot 10 is roster 6 in ReDrafters Rejoice."""
    nm = None
    if state:
        rid = (state.get("slot_to_roster") or {}).get(slot)
        nm = (state.get("roster_team") or {}).get(rid)
    nm = (nm or "").strip() or f"Team {slot}"
    return _short(nm, 11) if short else _esc(nm)


def pick_owner_slot(cfg, overall: int) -> int:
    """Seat that owns overall pick `overall` (1-based). Mirrors engine.team_on_clock
    so the board can be rendered without importing the engine."""
    rnd = (overall - 1) // cfg.teams
    idx = (overall - 1) % cfg.teams
    if cfg.snake and rnd % 2 == 1:
        return cfg.teams - idx
    return idx + 1


def pick_label(cfg, overall: int) -> str:
    """`4.03` -- round dot position-in-round. ⚠ NOT round dot seat: in a snake,
    seat 1 owns pick 2.12, not 2.01."""
    rnd = (overall - 1) // cfg.teams + 1
    inr = (overall - 1) % cfg.teams + 1
    return f"{rnd}.{inr:02d}"


def _traded_to(state, rnd: int, slot: int):
    """(current_owner_name, True) when this pick has changed hands, else (None, False)."""
    if not state:
        return (None, False)
    orig = (state.get("slot_to_roster") or {}).get(slot)
    cur = (state.get("traded") or {}).get((rnd, orig), orig)
    if cur == orig:
        return (None, False)
    return ((state.get("roster_team") or {}).get(cur, "?"), True)


def _drafted_at(overall: int, drafted, by_id, smeta):
    """(name, pos) for a completed pick, or None. Three sources, in order of trust:
    our matched board player, Sleeper's own pick metadata (covers IDP/UDFA our
    offense-only board cannot rank), then nothing."""
    if overall > len(drafted):
        return None
    pid = drafted[overall - 1]
    if pid and not str(pid).startswith("__off_"):
        p = by_id.get(pid)
        if p:
            return (p["name"], p["pos"])
    if overall in smeta:
        return smeta[overall]
    return None


def _sleeper_meta(state) -> dict:
    out = {}
    for pk in (state or {}).get("picks", []) or []:
        m = pk.get("metadata") or {}
        nm = f"{m.get('first_name', '')} {m.get('last_name', '')}".strip()
        if nm:
            out[pk.get("pick_no")] = (nm, pos_norm(m.get("position")))
    return out


# --------------------------------------------------------------------------- the grid
def draft_board_html(cfg, state, drafted, by_id, rounds_window: int | None = None) -> str:
    """Teams x rounds grid. Columns are seats in draft order, rows are rounds.

    rounds_window: show only this many rounds centred on the live pick. The full
    board is 12x15 = 180 cells, which on a phone means the round you care about is
    somewhere off-screen; passing e.g. 6 keeps the action in view. None = all.
    """
    teams, rounds = int(cfg.teams), int(cfg.total_rounds())
    made = len(drafted)
    cur = made + 1
    cur_round = min(rounds, (cur - 1) // teams + 1)
    smeta = _sleeper_meta(state)
    my_slot = int(cfg.my_slot)

    r_lo, r_hi = 1, rounds
    if rounds_window and rounds_window < rounds:
        r_lo = max(1, cur_round - 1)
        r_hi = min(rounds, r_lo + rounds_window - 1)
        r_lo = max(1, r_hi - rounds_window + 1)

    th = (f'padding:5px 4px;background:{PANEL};color:{MUTE};border:1px solid {LINE};'
          f'font-size:10px;font-weight:600;position:sticky;top:0;z-index:2')
    head = f'<th style="{th};left:0;z-index:3;min-width:26px">Rd</th>'
    for s in range(1, teams + 1):
        me = (s == my_slot)
        head += (f'<th style="{th};min-width:82px;'
                 f'background:{MINE if me else PANEL};color:{"#fff" if me else MUTE}">'
                 f'{"★ " if me else ""}{slot_team(cfg, state, s, short=True)}</th>')

    body = ""
    for r in range(r_lo, r_hi + 1):
        rowbg = PANEL if r != cur_round else "#1e293b"
        body += (f'<tr><td style="background:{rowbg};color:{MUTE};border:1px solid {LINE};'
                 f'text-align:center;font-size:11px;font-weight:600;position:sticky;left:0;'
                 f'z-index:1">{r}</td>')
        for s in range(1, teams + 1):
            p = (r - 1) * teams + (s if (not cfg.snake or r % 2 == 1) else teams - s + 1)
            hit = _drafted_at(p, drafted, by_id, smeta)
            if hit:
                nm, pos = hit
                bg = POS_BG.get(pos, "#475569")
                mark = " ★" if s == my_slot else ""
                body += (f'<td style="border:1px solid {LINE};background:{bg};color:#fff;'
                         f'padding:3px 4px;font-size:10px;line-height:1.2">'
                         f'<b>{_short(nm, 14)}</b><br>'
                         f'<span style="opacity:.85">{_esc(pos)} · {p}{mark}</span></td>')
                continue
            # --- not yet drafted
            newname, traded = _traded_to(state, r, s)
            mine = (s == my_slot) and not traded
            live = (p == cur)
            soon = cur < p <= cur + teams          # inside the next round of picks
            if live:
                brd, bg, col = f"2px solid {GOLD}", "#422006", GOLD
            elif mine:
                brd, bg, col = f"1px solid {MINE}", "#172554", "#93c5fd"
            elif soon:
                brd, bg, col = f"1px solid {LINE}", "#111c33", MUTE
            else:
                brd, bg, col = f"1px solid {LINE}", FIELD, "#475569"
            sub = ""
            if traded:
                sub = f'⇄ {_short(newname, 9)}'
            elif live:
                sub = "ON CLOCK"
            elif mine:
                sub = "YOU"
            body += (f'<td style="border:{brd};background:{bg};color:{col};padding:3px 4px;'
                     f'font-size:10px;line-height:1.2">{pick_label(cfg, p)}<br>'
                     f'<span style="font-size:9px;color:{GOLD if (live or traded) else col}">'
                     f'{sub}</span></td>')
        body += "</tr>"

    note = ""
    if (r_lo, r_hi) != (1, rounds):
        note = (f'<div style="color:{MUTE};font-size:11px;padding:4px 2px">'
                f'Rounds {r_lo}–{r_hi} of {rounds}</div>')
    return (f'{note}<div style="overflow:auto;max-width:100%;max-height:70vh;'
            f'border:1px solid {LINE};border-radius:8px">'
            f'<table style="border-collapse:separate;border-spacing:0;'
            f'font-family:system-ui,-apple-system,sans-serif;background:{FIELD}">'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


# --------------------------------------------------------------------------- on deck
def on_deck_html(cfg, state, current_overall: int, my_next: int | None,
                 seat_need: list[dict] | None, max_rows: int = 14,
                 complete: bool = False, tendencies: dict | None = None) -> str:
    """The queue between now and your next pick: every seat that picks before you,
    in order, with the position that seat most needs.

    This is the panel that answers "who is drafting where" -- the grid shows what
    has happened, this shows what is ABOUT to. Seats appear once per pick they
    own, so a team with back-to-back turns at the turn shows up twice."""
    need = {int(d["slot"]): d.get("top_need") for d in (seat_need or [])}
    total = int(cfg.teams) * int(cfg.total_rounds())
    # ⚠ `complete` must be passed in, not inferred. analyze() CLAMPS
    # current_overall to total_picks at the end, so the final pick looks
    # identical to a live one and this panel rendered "15.12 ON THE CLOCK" on a
    # finished draft. The app only avoided showing it by guarding the call --
    # a panel that lies unless its caller remembers to hide it is a trap.
    if complete or current_overall > total:
        return f'<div style="color:{MUTE};font-size:12px;padding:8px">Draft complete.</div>'
    end = my_next if my_next else min(total, current_overall + max_rows - 1)
    if end < current_overall:
        return f'<div style="color:{MUTE};font-size:12px;padding:8px">Draft complete.</div>'

    rows = ""
    n = 0
    for p in range(current_overall, min(end, current_overall + max_rows - 1) + 1):
        if p > total:
            break
        s = pick_owner_slot(cfg, p)
        rnd = (p - 1) // cfg.teams + 1
        owner, traded = _traded_to(state, rnd, s)
        is_me = (s == int(cfg.my_slot)) and not traded
        live = (p == current_overall)
        nm = owner if traded else slot_team(cfg, state, s)
        want = need.get(s)
        if is_me:
            bg, col, tag = "#172554", "#93c5fd", "★ YOUR PICK"
        elif live:
            bg, col, tag = "#422006", GOLD, "ON THE CLOCK"
        else:
            bg, col, tag = FIELD, INK, (f"needs {want}" if want else "")
        # ⚠ the TEAM NAME is the point of this panel, so it gets the flexible
        # space and everything else is fixed. `min-width:0` is required or the
        # flex item refuses to shrink below its content and pushes the tag off
        # the row instead of ellipsising -- at a 196px column that turned every
        # opponent into "F…". The overall pick number rides with the label
        # rather than taking a column of its own.
        # what this manager has actually done in past drafts, for THIS round only
        note = None
        if tendencies and not is_me:
            note = tendency_note(slot_manager(state, s, tendencies),
                                 tendencies.get("league_median_first_round"), rnd)
        name_cell = (
            f'<div style="flex:1 1 auto;min-width:0">'
            f'<div style="color:{col};font-weight:{600 if (live or is_me) else 400};'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap"'
            f' title="{_esc(nm)}">{_esc(nm)}{" ⇄" if traded else ""}</div>'
            + (f'<div style="color:#7c8aa5;font-size:9.5px;overflow:hidden;'
               f'text-overflow:ellipsis;white-space:nowrap">↳ {_esc(note)}</div>'
               if note else "")
            + '</div>')
        rows += (
            f'<div style="display:flex;align-items:center;gap:7px;padding:5px 8px;'
            f'background:{bg};border-left:3px solid {GOLD if live else (MINE if is_me else LINE)};'
            f'border-bottom:1px solid {LINE};font-size:12px">'
            f'<span style="color:{MUTE};flex:0 0 auto;font-variant-numeric:tabular-nums">'
            f'{pick_label(cfg, p)}'
            f'<span style="font-size:9px;opacity:.7"> #{p}</span></span>'
            + name_cell +
            f'<span style="color:{GOLD if (live or is_me) else MUTE};font-size:10px;'
            f'flex:0 0 auto;white-space:nowrap">{_esc(tag)}</span></div>')
        n += 1
    more = ""
    if my_next and my_next > current_overall + max_rows - 1:
        more = (f'<div style="padding:5px 8px;color:{MUTE};font-size:11px">'
                f'…{my_next - (current_overall + max_rows - 1)} more before your pick '
                f'at {pick_label(cfg, my_next)}</div>')
    return (f'<div style="border:1px solid {LINE};border-radius:8px;overflow:hidden;'
            f'background:{FIELD}">{rows}{more}</div>')


# --------------------------------------------------------------------------- run strip
def run_strip_html(cfg, drafted, by_id, state=None, n: int = 24) -> str:
    """Last `n` picks as position chips, oldest -> newest. A WR run is a visual
    block of blue; that is the read the numeric run_signal is quantifying, and
    seeing it is faster than reading it."""
    made = len(drafted)
    if not made:
        return f'<div style="color:{MUTE};font-size:12px">No picks yet.</div>'
    smeta = _sleeper_meta(state)
    lo = max(1, made - n + 1)
    chips = ""
    for p in range(lo, made + 1):
        hit = _drafted_at(p, drafted, by_id, smeta)
        pos = hit[1] if hit else None
        nm = hit[0] if hit else "—"
        bg = POS_BG.get(pos, "#334155")
        mine = pick_owner_slot(cfg, p) == int(cfg.my_slot)
        chips += (f'<span title="{_esc(nm)}" style="display:inline-block;background:{bg};'
                  f'color:#fff;font-size:9px;font-weight:700;padding:2px 0;width:26px;'
                  f'text-align:center;border-radius:3px;'
                  f'border:{"2px solid " + GOLD if mine else "2px solid transparent"}">'
                  f'{_esc(pos or "?")}</span> ')
    return (f'<div style="line-height:2.1">{chips}</div>'
            f'<div style="color:{MUTE};font-size:10px;margin-top:2px">'
            f'picks {lo}–{made}, oldest → newest · gold outline = yours</div>')


# --------------------------------------------------------------------------- roster matrix
def roster_matrix_html(cfg, state, drafted, by_id, starters: dict | None = None) -> str:
    """Every seat's roster SHAPE at a glance: counts by position, with the seats
    that still owe a starter flagged. This is how you predict the next few picks --
    a team with no QB in round 12 is taking a QB."""
    teams = int(cfg.teams)
    counts = {s: {} for s in range(1, teams + 1)}
    for i, pid in enumerate(drafted):
        p = by_id.get(pid)
        if not p:
            continue
        s = pick_owner_slot(cfg, i + 1)
        counts[s][p["pos"]] = counts[s].get(p["pos"], 0) + 1

    order = ["QB", "RB", "WR", "TE", "K", "DST"]
    need_min = dict(starters or {})
    head = (f'<th style="padding:4px 6px;text-align:left;color:{MUTE};font-size:11px;'
            f'border-bottom:1px solid {LINE}">Team</th>')
    for pos in order:
        head += (f'<th style="padding:4px 6px;color:{MUTE};font-size:11px;'
                 f'border-bottom:1px solid {LINE}">{pos}</th>')
    head += (f'<th style="padding:4px 6px;color:{MUTE};font-size:11px;'
             f'border-bottom:1px solid {LINE}">Still needs</th>')

    body = ""
    for s in range(1, teams + 1):
        me = s == int(cfg.my_slot)
        body += (f'<tr style="background:{"#172554" if me else "transparent"}">'
                 f'<td style="padding:4px 6px;color:{INK};font-size:12px;white-space:nowrap">'
                 f'{"★ " if me else ""}{slot_team(cfg, state, s, short=True)}</td>')
        gaps = []
        for pos in order:
            c = counts[s].get(pos, 0)
            want = int(need_min.get(pos, 0) or 0)
            short_of = want - c
            if short_of > 0:
                gaps.append(f"{pos}{'×' + str(short_of) if short_of > 1 else ''}")
            col = "#fff" if c else "#475569"
            bg = POS_DIM.get(pos, "#1e293b") if c else "transparent"
            body += (f'<td style="padding:3px 6px;text-align:center;font-size:12px;'
                     f'color:{col};background:{bg}">{c or "·"}</td>')
        body += (f'<td style="padding:4px 6px;color:{GOLD if gaps else MUTE};font-size:11px">'
                 f'{_esc(", ".join(gaps)) if gaps else "starters set"}</td></tr>')
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;'
            f'font-family:system-ui,-apple-system,sans-serif">'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


# --------------------------------------------------------------------------- header
def status_bar_html(cfg, state, res, on_clock_name: str) -> str:
    """The one strip you glance at while the clock runs: pick, who owns it, how far
    away you are."""
    done = res.get("complete")
    cur = res["current_overall"]
    mine = res.get("my_next_pick")
    away = res.get("picks_until_mine")
    is_me = res.get("on_clock") == int(cfg.my_slot)

    def cell(lbl, val, col=INK, big=True):
        return (f'<div style="flex:1;min-width:104px;padding:6px 10px">'
                f'<div style="color:{MUTE};font-size:10px;text-transform:uppercase;'
                f'letter-spacing:.06em">{lbl}</div>'
                f'<div style="color:{col};font-size:{"17px" if big else "13px"};'
                f'font-weight:700;line-height:1.25">{val}</div></div>')

    if done:
        mid = cell("Status", "🏁 Draft complete", GOLD)
    elif is_me:
        mid = cell("On the clock", "🟢 YOU", "#4ade80")
    else:
        mid = cell("On the clock", _esc(on_clock_name), INK)
    nxt = (f'{pick_label(cfg, mine)} <span style="color:{MUTE};font-weight:400;font-size:12px">'
           f'(#{mine} · {away} away)</span>') if mine else "—"
    return (f'<div style="display:flex;flex-wrap:wrap;align-items:stretch;gap:2px;'
            f'background:{PANEL};border:1px solid {LINE};border-radius:10px;'
            f'margin-bottom:8px">'
            + cell("Pick", f'{pick_label(cfg, cur)} <span style="color:{MUTE};font-weight:400;'
                           f'font-size:12px">#{cur}</span>')
            + mid
            + cell("Your next pick", nxt, GOLD if (away or 0) <= 2 else INK)
            + cell("Board", f'{res["board_size_remaining"]} <span style="color:{MUTE};'
                            f'font-weight:400;font-size:12px">left</span>', INK, big=False)
            + '</div>')


# --------------------------------------------------------------------------- everyone's team
POS_ORDER = ("QB", "RB", "WR", "TE", "K", "DST")


def team_rosters_html(cfg, state, drafted, by_id, starters: dict | None = None,
                      only_slot: int | None = None, tendencies: dict | None = None) -> str:
    """Every seat's ACTUAL ROSTER, side by side — the "let me see people's teams"
    view. The grid answers *when* a player went; this answers *who has what*.

    One card per seat, players grouped by position with the count against that
    league's starting requirement, so an unfilled slot reads as a gap rather than
    as an absence you have to notice. Cards flex-wrap, which is how this stays
    readable from a 12-across laptop down to one-per-row on a phone without any
    JS or a viewport measurement Streamlit cannot give us.

    Unmatched Sleeper picks (IDP, UDFA) come through `_drafted_at`, so a team's
    card is never silently short a player just because our offense-only board
    cannot rank him."""
    teams = int(cfg.teams)
    need = dict(starters or {})
    smeta = _sleeper_meta(state)
    my_slot = int(cfg.my_slot)

    # seat -> [(overall, name, pos)]
    rosters: dict[int, list] = {s: [] for s in range(1, teams + 1)}
    for i in range(len(drafted)):
        hit = _drafted_at(i + 1, drafted, by_id, smeta)
        if not hit:
            continue
        rosters[pick_owner_slot(cfg, i + 1)].append((i + 1, hit[0], hit[1]))

    cards = ""
    seats = [only_slot] if only_slot else range(1, teams + 1)
    for s in seats:
        me = (s == my_slot)
        _mgr = slot_manager(state, s, tendencies)
        picks = rosters.get(s, [])
        rows = ""
        for pos in POS_ORDER:
            got = [(o, n) for o, n, p in picks if p == pos]
            want = int(need.get(pos, 0) or 0)
            if not got and not want:
                continue                      # league doesn't use it and nobody took one
            short = want - len(got)
            cnt_col = GOLD if short > 0 else (MUTE if not want else "#4ade80")
            names = ("<br>".join(
                f'<span style="color:{INK}">{_short(n, 15)}</span>'
                f'<span style="color:{MUTE};font-size:9px"> {pick_label(cfg, o)}</span>'
                for o, n in got) or f'<span style="color:{GOLD}">— needs {pos}</span>')
            rows += (
                f'<tr>'
                f'<td style="vertical-align:top;padding:2px 6px 2px 0;white-space:nowrap">'
                f'<span style="display:inline-block;background:{POS_BG.get(pos, "#475569")};'
                f'color:#fff;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px">'
                f'{pos}</span> '
                f'<span style="color:{cnt_col};font-size:9px">'
                f'{len(got)}{"/" + str(want) if want else ""}</span></td>'
                f'<td style="padding:2px 0;font-size:11px;line-height:1.35">{names}</td>'
                f'</tr>')
        if not rows:
            rows = (f'<tr><td colspan="2" style="color:{MUTE};font-size:11px;padding:4px 0">'
                    f'No picks yet.</td></tr>')
        cards += (
            f'<div style="flex:1 1 210px;min-width:198px;max-width:340px;'
            f'border:1px solid {MINE if me else LINE};border-radius:9px;'
            f'background:{"#111c33" if me else FIELD};overflow:hidden">'
            f'<div style="background:{MINE if me else PANEL};color:#fff;padding:5px 8px;'
            f'font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis">{"★ " if me else ""}{slot_team(cfg, state, s)}'
            f'<span style="color:{"#bfdbfe" if me else MUTE};font-weight:400;font-size:10px">'
            f' · seat {s} · {len(picks)} picks</span></div>'
            # the manager's standing habits, from this league's own past drafts
            + (f'<div style="background:{PANEL};color:#7c8aa5;font-size:9.5px;'
               f'padding:2px 8px;border-bottom:1px solid {LINE}">'
               f'{_esc("; ".join((_mgr.get("notes") or [])[:2]))}</div>'
               if (_mgr and _mgr.get("notes")) else "")
            + f'<table style="width:100%;border-collapse:collapse;padding:4px">'
            f'{rows}</table></div>')

    return (f'<div style="display:flex;flex-wrap:wrap;gap:8px;'
            f'font-family:system-ui,-apple-system,sans-serif">{cards}</div>')


# --------------------------------------------------------------------------- tendencies
def tendency_note(mgr: dict, room_median: dict, rnd: int) -> str | None:
    """ONE short phrase about this manager, chosen for the round being drafted.

    A profile has several facts and almost all of them are irrelevant at any given
    moment -- "opens WR" tells you nothing in round 9, and "waits on TE until 13"
    tells you nothing in round 1. Showing the whole profile on every row would
    turn the on-deck panel into a wall you stop reading, which is worse than
    showing nothing. So: pick the one fact that bears on THIS round, or stay
    quiet.

    Only DEVIATIONS from the room count. "Takes a QB around round 8" is not a
    tendency, it is what everybody does."""
    if not mgr:
        return None
    rnd = int(rnd or 1)
    if rnd <= 2:
        pct = mgr.get("opens_with_pct") or 0
        if mgr.get("opens_with") and pct >= 60:
            return f"opens {mgr['opens_with']} {pct}%"
    best = None
    for pos, mine in (mgr.get("median_first_round") or {}).items():
        room = (room_median or {}).get(pos)
        if room is None or pos in ("K", "DST"):
            continue
        dev = float(mine) - float(room)
        if abs(dev) < 2:
            continue
        lo, hi = min(float(mine), float(room)), max(float(mine), float(room))
        if not (lo - 1 <= rnd <= hi + 1):        # not his window yet, or long past
            continue
        if best is None or abs(dev) > best[0]:
            best = (abs(dev), f"{'early' if dev < 0 else 'waits'} {pos} · rd{float(mine):g}")
    return best[1] if best else None


def slot_manager(state, slot: int, tend: dict | None) -> dict | None:
    """The tendency profile for whoever holds this SEAT, via roster -> user_id."""
    if not state or not tend:
        return None
    rid = (state.get("slot_to_roster") or {}).get(slot)
    uid = (state.get("roster_owner") or {}).get(rid)
    if not uid:
        return None
    for m in tend.get("managers", []):
        if str(m.get("user_id")) == str(uid):
            return m
    return None
