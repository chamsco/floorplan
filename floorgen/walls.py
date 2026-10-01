"""Exterior walls from a scan or a sketch: find geometry issues, propose fixes, and fill.

Input walls are straight segments [x0, y0, x1, y1] in metres, x east and y south.
Scans are messy: walls stop short of each other (gaps), run past a corner
(overshoots), sit slightly off the grid or off the axes. Each issue comes with a
short list of candidate fixes, each a complete replacement wall list, so a UI can
show the options side by side and let a person pick one ("Apply") or keep the
geometry as it is ("Mark as correct").
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-6
Wall = tuple[float, float, float, float]


@dataclass
class Fix:
    label: str
    detail: str
    walls: list[Wall]


@dataclass
class Issue:
    kind: str           # "off_axis" | "off_grid" | "gap" | "overshoot"
    message: str
    walls: list[int]    # indices of the walls involved
    fixes: list[Fix] = field(default_factory=list)


def _norm(w) -> Wall:
    x0, y0, x1, y1 = map(float, w)
    return (x0, y0, x1, y1)


def _horizontal(w: Wall) -> bool:
    return abs(w[1] - w[3]) < EPS


def _vertical(w: Wall) -> bool:
    return abs(w[0] - w[2]) < EPS


def _on_segment(px: float, py: float, w: Wall, eps: float = 1e-3) -> bool:
    x0, y0, x1, y1 = w
    if _horizontal(w):
        return abs(py - y0) < eps and min(x0, x1) - eps <= px <= max(x0, x1) + eps
    if _vertical(w):
        return abs(px - x0) < eps and min(y0, y1) - eps <= py <= max(y0, y1) + eps
    return False


def _ends(w: Wall) -> list[tuple[float, float]]:
    return [(w[0], w[1]), (w[2], w[3])]


def _replace(walls: list[Wall], i: int, new: Wall) -> list[Wall]:
    out = list(walls)
    out[i] = new
    return out


def _move_end(w: Wall, end: int, px: float, py: float) -> Wall:
    return (px, py, w[2], w[3]) if end == 0 else (w[0], w[1], px, py)


def _compass(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "eastward" if dx > 0 else "westward"
    return "southward" if dy > 0 else "northward"


def find_issues(walls_in, cell: float = 0.5, tol: float = 1.0) -> list[Issue]:
    """Issues are reported in a fixed order: axis, grid, then overshoots and gaps."""
    walls = [_norm(w) for w in walls_in]
    issues: list[Issue] = []

    for i, w in enumerate(walls):
        if not (_horizontal(w) or _vertical(w)):
            dx, dy = abs(w[2] - w[0]), abs(w[3] - w[1])
            snapped = (w[0], w[1], w[2], w[1]) if dx >= dy else (w[0], w[1], w[0], w[3])
            issues.append(Issue("off_axis", f"Wall {i + 1} is not straight along an axis", [i],
                                [Fix("Straighten the wall", f"Make wall {i + 1} {'horizontal' if dx >= dy else 'vertical'}.",
                                     _replace(walls, i, snapped))]))

    def snap(v: float) -> float:
        return round(v / cell) * cell

    off = [i for i, w in enumerate(walls) if any(abs(v - snap(v)) > 1e-3 for v in w)]
    if off:
        snapped = [tuple(snap(v) for v in w) for w in walls]
        issues.append(Issue("off_grid", f"{len(off)} wall(s) are off the {cell} m grid", off,
                            [Fix("Snap to grid", f"Round every wall end to the nearest {cell} m.", snapped)]))

    # Free ends: wall ends that do not touch another wall.
    free: list[tuple[int, int]] = []
    for i, w in enumerate(walls):
        for k, (px, py) in enumerate(_ends(w)):
            if not any(j != i and _on_segment(px, py, v) for j, v in enumerate(walls)):
                free.append((i, k))

    used: set[tuple[int, int]] = set()
    # Overshoot: a free end that runs past another wall it crosses, by at most tol.
    for i, k in free:
        w = walls[i]
        px, py = _ends(w)[k]
        ox, oy = _ends(w)[1 - k]
        for j, v in enumerate(walls):
            if j == i or _horizontal(w) == _horizontal(v):
                continue
            # crossing point of the two lines
            qx, qy = (v[0], w[1]) if _horizontal(w) else (w[0], v[1])
            if not (_on_segment(qx, qy, w) and _on_segment(qx, qy, v)):
                continue
            d = math.hypot(px - qx, py - qy)
            if not (EPS < d <= tol) or math.hypot(ox - qx, oy - qy) < d:
                continue
            fixes = [Fix("Trim the overshoot at the corner",
                         f"Shorten wall {i + 1} by {d:.2f} m so it ends at the junction with wall {j + 1}.",
                         _replace(walls, i, _move_end(w, k, qx, qy)))]
            # If the crossed wall ends at the junction, the scan may have put that wall in the wrong place.
            for kk, (rx, ry) in enumerate(_ends(v)):
                if math.hypot(rx - qx, ry - qy) < 1e-3:
                    sx, sy = (px - qx, 0.0) if _horizontal(w) else (0.0, py - qy)
                    moved = (v[0] + sx, v[1] + sy, v[2] + sx, v[3] + sy)
                    fixes.append(Fix("Move the wall to the scanned end",
                                     f"Shift wall {j + 1} by {d:.2f} m {_compass(sx, sy)} so the corner meets the overshoot.",
                                     _replace(walls, j, moved)))
            issues.append(Issue("overshoot", f"Wall {i + 1} runs {d:.2f} m past wall {j + 1}", [i, j], fixes))
            used.add((i, k))
            break

    # Gap: two free ends close together.
    rest = [f for f in free if f not in used]
    for a in range(len(rest)):
        for b in range(a + 1, len(rest)):
            (i, k), (j, kk) = rest[a], rest[b]
            if (i, k) in used or (j, kk) in used or i == j:
                continue
            wi, wj = walls[i], walls[j]
            px, py = _ends(wi)[k]
            rx, ry = _ends(wj)[kk]
            d = math.hypot(px - rx, py - ry)
            if d > tol:
                continue
            fixes: list[Fix] = []
            collinear = (_horizontal(wi) and _horizontal(wj) and abs(py - ry) < EPS) or \
                        (_vertical(wi) and _vertical(wj) and abs(px - rx) < EPS)
            if collinear:
                fixes.append(Fix(f"Extend wall {i + 1} to close the gap",
                                 f"Lengthen wall {i + 1} by {d:.2f} m {_compass(rx - px, ry - py)} to meet the opening.",
                                 _replace(walls, i, _move_end(wi, k, rx, ry))))
                fixes.append(Fix(f"Extend wall {j + 1} to close the gap",
                                 f"Lengthen wall {j + 1} by {d:.2f} m {_compass(px - rx, py - ry)} to meet the opening.",
                                 _replace(walls, j, _move_end(wj, kk, px, py))))
            elif _horizontal(wi) != _horizontal(wj):
                cx, cy = (rx, py) if _horizontal(wi) else (px, ry)
                both = _replace(_replace(walls, i, _move_end(wi, k, cx, cy)), j, _move_end(wj, kk, cx, cy))
                fixes.append(Fix("Extend both walls to the corner",
                                 f"Lengthen walls {i + 1} and {j + 1} until they meet.", both))
            if abs(px - rx) < EPS or abs(py - ry) < EPS:
                fixes.append(Fix("Add new wall segment",
                                 "Insert a segment of the same construction between the two walls.",
                                 walls + [(px, py, rx, ry)]))
            issues.append(Issue("gap", f"{d:.2f} m gap between wall {i + 1} and wall {j + 1}", [i, j], fixes))
            used.update({(i, k), (j, kk)})
    return issues


def mask_from_walls(walls_in, cell: float) -> np.ndarray:
    """Cells enclosed by the walls: flood the outside from the border, the rest is inside.

    Walls are snapped to the grid first; off-axis walls are ignored, so fix them first.
    An empty result means the walls do not close.
    """
    walls = [tuple(round(v / cell) for v in _norm(w)) for w in walls_in]
    walls = [w for w in walls if w[0] == w[2] or w[1] == w[3]]
    if not walls:
        return np.zeros((0, 0), dtype=bool)
    W = max(max(w[0], w[2]) for w in walls)
    H = max(max(w[1], w[3]) for w in walls)
    # blocked[v|h]: vertical wall on grid line x between rows y..y+1; horizontal on line y.
    vblock = np.zeros((H + 2, W + 3), dtype=bool)   # indexed [y+1, x+1]
    hblock = np.zeros((H + 3, W + 2), dtype=bool)
    for x0, y0, x1, y1 in walls:
        if x0 == x1:
            for y in range(min(y0, y1), max(y0, y1)):
                vblock[y + 1, x0 + 1] = True
        else:
            for x in range(min(x0, x1), max(x0, x1)):
                hblock[y0 + 1, x + 1] = True
    # padded grid of cells: index (y+1, x+1); padding ring is outside
    Hp, Wp = H + 2, W + 2
    seen = np.zeros((Hp, Wp), dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < Hp and 0 <= nx < Wp) or seen[ny, nx]:
                continue
            if dx == 1 and vblock[y, x + 1]:
                continue
            if dx == -1 and vblock[y, x]:
                continue
            if dy == 1 and hblock[y + 1, x]:
                continue
            if dy == -1 and hblock[y, x]:
                continue
            seen[ny, nx] = True
            q.append((ny, nx))
    return ~seen[1:-1, 1:-1]
