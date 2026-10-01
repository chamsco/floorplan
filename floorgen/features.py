"""Stage 3: describe and deduplicate, plus the target score u0 (paper, Sections 6 and 7).

Features describe what kind of plan it is, not whether it is good. Near-copies are
caught on the room-type raster: two plans whose rooms cover the same cells look
the same to a person whatever their seeds were.
"""
from __future__ import annotations

import math

import numpy as np

from .brief import Brief
from .checks import raster, window_contact
from .generate import Plan


def type_raster(brief: Brief, plan: Plan) -> np.ndarray:
    """uint8 grid of room-type codes (0 = outside)."""
    order = {t: k + 1 for k, t in enumerate(brief.rooms)}
    g = raster(brief, plan)
    out = np.zeros(g.shape, dtype=np.uint8)
    for i, r in enumerate(plan.rooms):
        out[g == i] = order[r.type]
    return out


def tree_depth(plan: Plan) -> dict[int, int]:
    depth = {plan.entry: 0}
    children: dict[int, list[int]] = {}
    for d in plan.doors[1:]:
        children.setdefault(d.a, []).append(d.b)
    stack = [plan.entry]
    while stack:
        a = stack.pop()
        for b in children.get(a, []):
            depth[b] = depth[a] + 1
            stack.append(b)
    return depth


def features(brief: Brief, plan: Plan) -> dict[str, float]:
    sh = brief.shell
    A = sh.area_cells
    f: dict[str, float] = {}
    for t in brief.rooms:
        f[f"share:{t}"] = sum(r.area for r in plan.rooms if r.type == t) / A
    aspects = [max(r.w, r.h) / min(r.w, r.h) for r in plan.rooms]
    f["max_aspect"] = max(aspects)
    f["mean_aspect"] = sum(aspects) / len(aspects)
    for t in ("living", "bedroom", "kitchen"):
        r = next((r for r in plan.rooms if r.type == t), None)
        f[f"cx:{t}"] = (r.x + r.w / 2) / sh.W if r else 0.5
        f[f"cy:{t}"] = (r.y + r.h / 2) / sh.H if r else 0.5
    f["open_kitchen"] = float(any(d.open for d in plan.doors))
    depth = tree_depth(plan)
    f["depth"] = float(max(depth.values()))
    by_parent: dict[str, list[str]] = {}
    for d in plan.doors[1:]:
        by_parent.setdefault(plan.rooms[d.a].type, []).append(plan.rooms[d.b].type)
    f["off_living"] = float(len([t for t in by_parent.get("living", []) if t not in ("kitchen",)]))
    f["bath_via_bedroom"] = float("bath" in by_parent.get("bedroom", []))
    f["entry_living"] = float(plan.rooms[plan.entry].type == "living")
    win_total = sum(e.length for e in sh.window_edges()) or 1
    lit = sum(window_contact(brief, r) for r in plan.rooms if r.type in ("living", "bedroom", "kitchen"))
    f["daylight_share"] = lit / win_total
    bed = next((r for r in plan.rooms if r.type == "bedroom"), None)
    ent = sh.entry
    if bed:
        ex = (ent.s + ent.e) / 2 if ent.side in ("N", "S") else ent.fixed
        ey = ent.fixed if ent.side in ("N", "S") else (ent.s + ent.e) / 2
        f["bed_from_door"] = math.hypot(bed.x + bed.w / 2 - ex, bed.y + bed.h / 2 - ey) * sh.cell
    else:
        f["bed_from_door"] = 0.0
    return f


def facts(brief: Brief, plan: Plan) -> str:
    """The plan as plain text for model judges: measured, never interpreted."""
    c = brief.shell.cell
    lines = []
    for r in plan.rooms:
        lines.append(f"{r.name}: {r.w * c:.1f} x {r.h * c:.1f} m = {r.area * c * c:.1f} m², "
                     f"window wall {window_contact(brief, r) * c:.1f} m")
    lines.append(f"front door opens into: {plan.rooms[plan.entry].name}")
    for d in plan.doors[1:]:
        kind = "open (no wall)" if d.open else "door"
        lines.append(f"{plan.rooms[d.b].name} is entered from {plan.rooms[d.a].name} ({kind})")
    return "\n".join(lines)


class Archive:
    """Novelty gate on the type raster: keep a plan only if it differs from every kept plan
    in at least `delta` of its cells."""

    def __init__(self, cells: int, delta: float = 0.06, capacity: int = 4096):
        self.delta = delta
        self.n = 0
        self.buf = np.zeros((capacity, cells), dtype=np.uint8)

    def offer(self, r: np.ndarray) -> bool:
        flat = r.ravel()
        if self.n:
            diff = (self.buf[: self.n] != flat).mean(axis=1)
            if diff.min() < self.delta:
                return False
        if self.n == len(self.buf):
            self.buf = np.concatenate([self.buf, np.zeros_like(self.buf)])
        self.buf[self.n] = flat
        self.n += 1
        return True


def standardise(rows: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    keys = sorted(rows[0])
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    return (X - X.mean(axis=0)) / sd, keys


def gate(y: float, goal: float, width: float, kind: str) -> float:
    if kind == "near":
        return math.exp(-((y - goal) ** 2) / (2 * width ** 2))
    if kind == "above":
        return 1 / (1 + math.exp(-(y - goal) / width))
    if kind == "below":
        return 1 / (1 + math.exp((y - goal) / width))
    raise ValueError(kind)


def target_score(brief: Brief, f: dict[str, float]) -> float:
    """u0: weighted geometric mean of the target gates (every target is needed)."""
    alias = {"living_share": "share:living", "hall_share": "share:hall"}
    targets = brief.targets
    if not targets:
        return 0.5
    logs = []
    for name, t in targets.items():
        y = f[alias.get(name, name)]
        logs.append(math.log(max(gate(y, t["goal"], t["width"], t["kind"]), 1e-9)))
    return math.exp(sum(logs) / len(logs))
