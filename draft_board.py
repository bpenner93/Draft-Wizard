"""
draft_board.py
--------------
FantasyPros-style draft-board grid (pure HTML string, no Streamlit) so it can be
tested headlessly. Teams are columns, rounds are rows; drafted players are colored
by position, the on-the-clock pick is outlined, your picks are highlighted, and
traded picks are flagged with the acquiring team.
"""

from draft_names import pos_norm

POS_BG = {"QB": "#7c3aed", "RB": "#059669", "WR": "#2563eb", "TE": "#d97706",
          "K": "#4b5563", "DST": "#4b5563"}
LINE = "#334155"


def draft_board_html(cfg, state, drafted, by_id) -> str:
    teams, rounds = int(cfg.teams), int(cfg.total_rounds())
    made = len(drafted)
    cur = made + 1

    smeta = {}
    if state:
        for pk in state.get("picks", []):
            m = pk.get("metadata") or {}
            nm = f"{(m.get('first_name') or '')[:1]}. {m.get('last_name') or ''}".strip()
            smeta[pk.get("pick_no")] = (nm, pos_norm(m.get("position")))
    my_roster = state.get("my_roster") if state else None

    def slot_team(s):
        if state:
            return state["roster_team"].get(state["slot_to_roster"].get(s), f"Team {s}")
        return f"Team {s}"

    def owner(r, s):
        if not state:
            return (None, False)
        orig = state["slot_to_roster"].get(s)
        cur_o = state["traded"].get((r, orig), orig)
        return (cur_o, cur_o != orig)

    head = f'<th style="padding:4px;background:#0f172a;color:#94a3b8;border:1px solid {LINE};font-size:10px">Rd</th>'
    for s in range(1, teams + 1):
        me = (s == cfg.my_slot)
        head += (f'<th style="padding:4px;min-width:74px;background:{"#1d4ed8" if me else "#0f172a"};'
                 f'color:#fff;border:1px solid {LINE};font-size:10px">{"★ " if me else ""}{slot_team(s)[:13]}</th>')

    body = ""
    for r in range(1, rounds + 1):
        body += (f'<tr><td style="background:#0f172a;color:#94a3b8;border:1px solid {LINE};'
                 f'text-align:center;font-size:11px">{r}</td>')
        for s in range(1, teams + 1):
            p = (r - 1) * teams + (s if (not cfg.snake or r % 2 == 1) else teams - s + 1)
            o_rid, traded = owner(r, s)
            mine = (o_rid == my_roster) if state else (s == cfg.my_slot)
            drafted_real = p <= made and drafted[p - 1] and not str(drafted[p - 1]).startswith("__off_")
            if p in smeta and p <= made:
                nm, pos = smeta[p]
                bg = POS_BG.get(pos, "#475569")
                body += (f'<td style="border:1px solid {LINE};background:{bg};color:#fff;padding:3px;'
                         f'font-size:10px;line-height:1.15"><b>{nm}</b><br>{pos} · {p}</td>')
            elif drafted_real:
                pl = by_id.get(drafted[p - 1])
                nm = pl["name"].split()[-1] if pl else "—"
                pos = pl["pos"] if pl else ""
                bg = POS_BG.get(pos, "#475569")
                body += (f'<td style="border:1px solid {LINE};background:{bg};color:#fff;padding:3px;'
                         f'font-size:10px;line-height:1.15"><b>{nm}</b><br>{pos} · {p}</td>')
            else:
                brd = "2px solid #f59e0b" if p == cur else f"1px solid {LINE}"
                bg = "#172554" if mine else "#0b1220"
                if traded:
                    sub = f'⇄ {state["roster_team"].get(o_rid, "")[:9]}'
                elif mine and not state:
                    sub = "YOU"
                else:
                    sub = ""
                body += (f'<td style="border:{brd};background:{bg};color:#64748b;padding:3px;'
                         f'font-size:10px;line-height:1.15">{r}.{s:02d}<br>'
                         f'<span style="font-size:9px;color:#f59e0b">{sub}</span></td>')
        body += "</tr>"

    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;'
            f'font-family:system-ui,sans-serif"><tr>{head}</tr>{body}</table></div>')
