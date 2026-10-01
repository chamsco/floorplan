"""Stage 1: generate. A CP-SAT model that is valid by construction.

Each room is an axis-aligned rectangle on the grid. The solver enforces the hard
rules that are global (and which a local method such as wave function collapse
cannot see): the rooms tile the shell exactly, minimum sizes, the front door opens
into an allowed room, daylight rooms touch a window wall, and every room hangs off
a door tree rooted at the entry, so the plan is connected and the door rules hold.

Variety comes from two levels of randomness (paper, Section 4): the knobs theta
(room shares, open-plan tendency, how far rooms are pulled to random positions)
and the seed (jitter on targets, preferred positions, solver search). The same
(theta, seed) always gives the same plan.

Known limit: rooms are rectangles, so L-shaped rooms are out of reach. An L-shaped
shell is fine; it is tiled by rectangles.
"""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field

from ortools.sat.python import cp_model

from .brief import Brief, Room
from .shell import Edge


@dataclass
class RoomRect:
    name: str
    type: str
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class Door:
    a: int          # parent room index (-1 = outside, for the front door)
    b: int          # child room index
    orient: str     # "v": door in a vertical wall (x = fixed); "h": horizontal wall (y = fixed)
    fixed: int      # grid line of the wall
    t0: int         # door span along the wall, cells [t0, t1)
    t1: int
    open: bool = False


@dataclass
class Plan:
    id: str
    seed: int
    theta: dict
    rooms: list[RoomRect] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    entry: int = -1
    ok: bool = False
    solve_ms: float = 0.0
    status: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Plan":
        p = Plan(id=d["id"], seed=d["seed"], theta=d["theta"])
        p.rooms = [RoomRect(**r) for r in d["rooms"]]
        p.doors = [Door(**x) for x in d["doors"]]
        p.entry, p.ok, p.solve_ms, p.status = d["entry"], d["ok"], d["solve_ms"], d["status"]
        return p


def _touch_edge(m: cp_model.CpModel, X, Y, XE, YE, edge: Edge, need: int):
    """A literal that is true only if the room lies along `edge` with at least `need` cells of contact."""
    b = m.NewBoolVar("")
    if edge.side in ("N", "S"):
        line = Y if edge.side == "N" else YE
        lo, hi = X, XE
    else:
        line = X if edge.side == "W" else XE
        lo, hi = Y, YE
    m.Add(line == edge.fixed).OnlyEnforceIf(b)
    m.Add(hi - edge.s >= need).OnlyEnforceIf(b)
    m.Add(edge.e - lo >= need).OnlyEnforceIf(b)
    return b


def _adjacent(m: cp_model.CpModel, v: dict, i: int, j: int, need: int) -> list:
    """Four literals, one per side; each true only if rooms i and j share at least `need` cells of wall there."""
    x, y, xe, ye = v["x"], v["y"], v["xe"], v["ye"]
    out = []
    for a, b, horiz in ((i, j, True), (j, i, True), (i, j, False), (j, i, False)):
        o = m.NewBoolVar("")
        if horiz:   # a is west of b; overlap along y
            m.Add(xe[a] == x[b]).OnlyEnforceIf(o)
            m.Add(ye[a] - y[b] >= need).OnlyEnforceIf(o)
            m.Add(ye[b] - y[a] >= need).OnlyEnforceIf(o)
        else:       # a is north of b; overlap along x
            m.Add(ye[a] == y[b]).OnlyEnforceIf(o)
            m.Add(xe[a] - x[b] >= need).OnlyEnforceIf(o)
            m.Add(xe[b] - x[a] >= need).OnlyEnforceIf(o)
        out.append(o)
    return out


def realise_theta(brief: Brief, theta: dict) -> tuple[list[Room], dict[str, float]]:
    """theta holds knobs in [0, 1]; map them to the program and target shares."""
    include = {t: bool(theta.get("include", {}).get(t, False)) for t, s in brief.rooms.items() if s.optional}
    program = brief.program(include)
    shares = {}
    for r in program:
        lo, hi = r.spec.share
        shares[r.name] = lo + (hi - lo) * float(theta.get("share", {}).get(r.type, 0.5))
    total = sum(shares.values())
    return program, {k: v / total for k, v in shares.items()}


def generate(brief: Brief, theta: dict, seed: int, time_limit: float = 0.03) -> Plan:
    sh = brief.shell
    rng = random.Random(seed)
    program, shares = realise_theta(brief, theta)
    n = len(program)
    plan = Plan(id=f"{seed:08x}", seed=seed, theta=theta)

    m = cp_model.CpModel()
    v = {k: [] for k in ("x", "y", "w", "h", "xe", "ye", "a")}
    xs, ys = [], []
    for r in program:
        mw = sh.cells(r.spec.min_width)
        x, y = m.NewIntVar(0, sh.W - mw, ""), m.NewIntVar(0, sh.H - mw, "")
        w, h = m.NewIntVar(mw, sh.W, ""), m.NewIntVar(mw, sh.H, "")
        xe, ye = m.NewIntVar(mw, sh.W, ""), m.NewIntVar(mw, sh.H, "")
        a = m.NewIntVar(0, sh.area_cells, "")
        m.AddMultiplicationEquality(a, [w, h])
        m.Add(a >= int(-(-r.spec.min_area // (sh.cell ** 2))))   # ceil
        k = int(round(brief.max_aspect * 10))
        m.Add(10 * w <= k * h)
        m.Add(10 * h <= k * w)
        xs.append(m.NewIntervalVar(x, w, xe, ""))
        ys.append(m.NewIntervalVar(y, h, ye, ""))
        for key, var in zip(("x", "y", "w", "h", "xe", "ye", "a"), (x, y, w, h, xe, ye, a)):
            v[key].append(var)
    for hx, hy, hw, hh in sh.holes:
        xs.append(m.NewFixedSizeIntervalVar(hx, hw, ""))
        ys.append(m.NewFixedSizeIntervalVar(hy, hh, ""))
    m.AddNoOverlap2D(xs, ys)
    m.Add(sum(v["a"]) == sh.area_cells)   # with no overlap and no holes covered: an exact tiling

    # Front door opens into exactly one allowed room.
    ent = sh.entry
    e_lit = []
    for i, r in enumerate(program):
        if r.type not in brief.entry_rooms:
            e_lit.append(None)
            continue
        b = _touch_edge(m, v["x"][i], v["y"][i], v["xe"][i], v["ye"][i], ent, ent.length)
        e_lit.append(b)
    if all(b is None for b in e_lit):
        raise ValueError("no room in the program may hold the front door")
    m.AddExactlyOne([b for b in e_lit if b is not None])

    # Daylight.
    need_win = sh.cells(brief.window_min)
    for i, r in enumerate(program):
        if r.spec.daylight:
            lits = [_touch_edge(m, v["x"][i], v["y"][i], v["xe"][i], v["ye"][i], e, need_win)
                    for e in sh.window_edges() if e.length >= need_win]
            if not lits:
                raise ValueError(f"{r.name} needs daylight but no window wall is long enough")
            m.AddBoolOr(lits)

    # Door tree rooted at the entry room: every other room has exactly one parent it may be
    # entered from, shares enough wall with it for a door, and sits deeper in the tree.
    dw = sh.cells(brief.door_width)
    rank = [m.NewIntVar(0, n - 1, "") for _ in range(n)]
    parent_lits: dict[tuple[int, int], cp_model.IntVar] = {}
    for i, ri in enumerate(program):
        cands = []
        for j, rj in enumerate(program):
            if i != j and brief.can_enter(ri.type, rj.type):
                p = m.NewBoolVar("")
                m.Add(rank[i] >= rank[j] + 1).OnlyEnforceIf(p)
                m.AddBoolOr(_adjacent(m, v, i, j, dw) + [p.Not()])
                parent_lits[(i, j)] = p
                cands.append(p)
        is_entry = e_lit[i]
        if is_entry is None:
            if not cands:
                raise ValueError(f"{ri.name} has no room it may be entered from")
            m.AddExactlyOne(cands)
        else:
            m.Add(sum(cands) == 0).OnlyEnforceIf(is_entry)
            m.Add(sum(cands) == 1).OnlyEnforceIf(is_entry.Not())
            m.Add(rank[i] == 0).OnlyEnforceIf(is_entry)

    # Objective: hit the jittered area targets, and drift toward random positions (seed-level variety).
    spread = float(theta.get("spread", 0.5))
    terms = []
    for i, r in enumerate(program):
        target = int(round(shares[r.name] * sh.area_cells * rng.lognormvariate(0, 0.12)))
        dev = m.NewIntVar(0, sh.area_cells, "")
        m.AddAbsEquality(dev, v["a"][i] - target)
        terms.append(4 * dev)
        px, py = rng.randint(0, 2 * sh.W), rng.randint(0, 2 * sh.H)
        dx, dy = m.NewIntVar(0, 2 * sh.W, ""), m.NewIntVar(0, 2 * sh.H, "")
        m.AddAbsEquality(dx, 2 * v["x"][i] + v["w"][i] - px)
        m.AddAbsEquality(dy, 2 * v["y"][i] + v["h"][i] - py)
        weight = int(round(3 * spread))
        if weight:
            terms.append(weight * (dx + dy))
    m.Minimize(sum(terms))

    solver = cp_model.CpSolver()
    # Deterministic time, not wall-clock: the same (theta, seed) must give the same plan on any machine.
    solver.parameters.max_deterministic_time = time_limit
    solver.parameters.max_time_in_seconds = 20 * time_limit + 1   # safety net only
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = seed % (2 ** 31)
    t0 = time.perf_counter()
    status = solver.Solve(m)
    plan.solve_ms = (time.perf_counter() - t0) * 1000
    plan.status = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return plan

    plan.rooms = [RoomRect(r.name, r.type, *(solver.Value(v[k][i]) for k in ("x", "y", "w", "h")))
                  for i, r in enumerate(program)]
    plan.entry = next(i for i, b in enumerate(e_lit) if b is not None and solver.Value(b))
    open_p = float(theta.get("open", 0.5))
    for (i, j), p in parent_lits.items():
        if solver.Value(p):
            pair = tuple(sorted((program[i].type, program[j].type)))
            is_open = any(pair == tuple(sorted(op)) for op in brief.open_pairs) and rng.random() < open_p
            plan.doors.append(_place_door(plan.rooms[j], plan.rooms[i], j, i, dw, rng, is_open))
    plan.doors.sort(key=lambda d: d.b)
    plan.doors.insert(0, Door(-1, plan.entry, "h" if ent.side in ("N", "S") else "v", ent.fixed, ent.s, ent.e))
    plan.ok = True
    return plan


def shared_wall(a: RoomRect, b: RoomRect):
    """(orient, fixed, s, e) of the wall the rooms share, or None."""
    if a.x + a.w == b.x or b.x + b.w == a.x:
        s, e = max(a.y, b.y), min(a.y + a.h, b.y + b.h)
        if e > s:
            return "v", (b.x if a.x + a.w == b.x else a.x), s, e
    if a.y + a.h == b.y or b.y + b.h == a.y:
        s, e = max(a.x, b.x), min(a.x + a.w, b.x + b.w)
        if e > s:
            return "h", (b.y if a.y + a.h == b.y else a.y), s, e
    return None


def _place_door(parent: RoomRect, child: RoomRect, pi: int, ci: int, dw: int, rng: random.Random, is_open: bool) -> Door:
    orient, fixed, s, e = shared_wall(parent, child)
    if is_open:
        return Door(pi, ci, orient, fixed, s, e, open=True)
    # near one end (leaves a usable wall) or centred, as people place doors
    slots = [s, e - dw, (s + e - dw) // 2]
    t0 = rng.choice(slots)
    return Door(pi, ci, orient, fixed, t0, t0 + dw)
