"""Stage 2: verify. Exact checks in code, independent of the solver.

The generator already builds these rules in, so in a correct world every check
passes. They are re-checked from the output geometry anyway: a solver model is code
too, and an encoding bug would otherwise flow silently into the judges. Any failure
here is a bug report against generate.py, and should become a test.

Each check returns a signed margin (paper, Section 5): positive = passes with room
to spare, negative = fails by that much, in metres or m².
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .brief import Brief
from .generate import Plan, shared_wall


def raster(brief: Brief, plan: Plan) -> np.ndarray:
    """Room index per cell; -1 = uncovered, -2 = covered twice or outside the shell."""
    sh = brief.shell
    g = np.full((sh.H, sh.W), -1, dtype=int)
    for i, r in enumerate(plan.rooms):
        sub = g[r.y:r.y + r.h, r.x:r.x + r.w]
        sub[sub != -1] = -2
        sub[sub == -1] = i
    g[(~sh.mask) & (g != -1)] = -2
    return g


def check(brief: Brief, plan: Plan) -> dict[str, float]:
    """Margins per check. A plan is valid when every margin is >= 0."""
    sh, c = brief.shell, brief.shell.cell
    if not plan.ok or not plan.rooms:
        return {"constructed": -1.0}
    m: dict[str, float] = {}
    g = raster(brief, plan)
    bad = int(((g == -2) | ((g == -1) & sh.mask)).sum())
    m["tiling"] = -bad * c * c
    # sizes
    m["min_width"] = min((min(r.w, r.h) * c - brief.rooms[r.type].min_width) for r in plan.rooms)
    m["min_area"] = min((r.area * c * c - brief.rooms[r.type].min_area) for r in plan.rooms)
    m["aspect"] = min(brief.max_aspect - max(r.w, r.h) / min(r.w, r.h) for r in plan.rooms)
    # program: every required room present
    types = [r.type for r in plan.rooms]
    m["program"] = 0.0 if all(t in types for t, s in brief.rooms.items() if not s.optional) else -1.0
    # entry
    ent = sh.entry
    er = plan.rooms[plan.entry]
    on_line = (er.y + er.h if ent.side == "S" else er.y if ent.side == "N" else er.x + er.w if ent.side == "E" else er.x) == ent.fixed
    lo, hi = (er.x, er.x + er.w) if ent.side in ("N", "S") else (er.y, er.y + er.h)
    m["entry"] = (min(hi - ent.e, ent.s - lo) * c) if on_line and er.type in brief.entry_rooms else -1.0
    # daylight: longest contact with a window wall
    for i, r in enumerate(plan.rooms):
        if brief.rooms[r.type].daylight:
            m[f"daylight:{r.name}"] = window_contact(brief, r) * c - brief.window_min
    # doors: each joins two rooms that share that wall, follows the rules, and is wide enough
    dw = sh.cells(brief.door_width)
    door_margin = 0.0
    for d in plan.doors[1:]:
        a, b = plan.rooms[d.a], plan.rooms[d.b]
        sw = shared_wall(a, b)
        if not sw or sw[0] != d.orient or sw[1] != d.fixed or d.t0 < sw[2] or d.t1 > sw[3]:
            door_margin = min(door_margin, -1.0)
            continue
        if not brief.can_enter(b.type, a.type):
            door_margin = min(door_margin, -1.0)
        door_margin = min(door_margin, ((d.t1 - d.t0) - dw) * c if not d.open else 0.0)
    m["doors"] = door_margin
    # connected: every room reachable from the entry through doors
    adj = {i: [] for i in range(len(plan.rooms))}
    for d in plan.doors[1:]:
        adj[d.a].append(d.b)
        adj[d.b].append(d.a)
    seen, q = {plan.entry}, deque([plan.entry])
    while q:
        for j in adj[q.popleft()]:
            if j not in seen:
                seen.add(j)
                q.append(j)
    m["connected"] = 0.0 if len(seen) == len(plan.rooms) else -float(len(plan.rooms) - len(seen))
    return m


def window_contact(brief: Brief, r) -> int:
    """Longest run, in cells, of the room's outline that lies on a window wall."""
    best = 0
    for e in brief.shell.window_edges():
        if e.side == "N" and r.y == e.fixed or e.side == "S" and r.y + r.h == e.fixed:
            best = max(best, min(r.x + r.w, e.e) - max(r.x, e.s))
        elif e.side == "W" and r.x == e.fixed or e.side == "E" and r.x + r.w == e.fixed:
            best = max(best, min(r.y + r.h, e.e) - max(r.y, e.s))
    return best


def valid(margins: dict[str, float]) -> bool:
    return all(v >= -1e-9 for v in margins.values())
