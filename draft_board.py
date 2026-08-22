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


def _short(name: str, n: int = 13) -> str:
    """`Amon-Ra St. Brown` -> `A. St. Brown`, then hard-truncate."""
    parts = str(name or "").split()
    if len(parts) >= 2:
        name = f"{parts[0][:1]}. {' '.join(parts[1:])}"
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
                 seat_need: list[dict] | None, max_rows: int = 14) -> str:
    """The queue between now and your next pick: every seat that picks before you,
    in order, with the position that seat most needs.

    This is the panel that answers "who is drafting where" -- the grid shows what
    has happened, this shows what is ABOUT to. Seats appear once per pick they
    own, so a team with back-to-back turns at the turn shows up twice."""
    need = {int(d["slot"]): d.get("top_need") for d in (seat_need or [])}
    total = int(cfg.teams) * int(cfg.total_rounds())
    end = my_next if my_next else min(total, current_overall + max_rows - 1)
    if end < current_overall:
        return (f'<div style="color:{MUTE};font-size:12px">Draft complete.</div>')

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
        rows += (
            f'<div style="display:flex;align-items:baseline;gap:7px;padding:5px 8px;'
            f'background:{bg};border-left:3px solid {GOLD if live else (MINE if is_me else LINE)};'
            f'border-bottom:1px solid {LINE};font-size:12px">'
            f'<span style="color:{MUTE};flex:0 0 auto;font-variant-numeric:tabular-nums">'
            f'{pick_label(cfg, p)}'
            f'<span style="font-size:9px;opacity:.7"> #{p}</span></span>'
            f'<span style="color:{col};flex:1 1 auto;min-width:0;'
            f'font-weight:{600 if (live or is_me) else 400};'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap"'
            f' title="{_esc(nm)}">{_esc(nm)}{" ⇄" if traded else ""}</span>'
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
