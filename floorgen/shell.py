"""The shell: the fixed outer walls a plan must fill.

Everything is on a square grid (default 0.5 m). Coordinates are in cells, x to the
east and y to the south, so north is the top of the drawing (y = 0).

A shell is built from either a rectilinear `outline` polygon or a list of exterior
wall segments (see walls.py), and carries the entrance and the sides that may have
windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIDES = ("N", "S", "W", "E")


@dataclass(frozen=True)
class Edge:
    """A run of exterior wall, facing `side`, at grid line `fixed`, spanning cells [s, e)."""
    side: str
    fixed: int
    s: int
    e: int

    @property
    def length(self) -> int:
        return self.e - self.s


@dataclass
class Shell:
    mask: np.ndarray            # bool [H, W]; True = inside the apartment
    cell: float                 # metres per cell
    entry: Edge                 # where the front door sits
    window_sides: tuple[str, ...]
    holes: list[tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h) outside cells in the bbox
    edges: list[Edge] = field(default_factory=list)                       # all exterior edge runs

    @property
    def H(self) -> int:
        return self.mask.shape[0]

    @property
    def W(self) -> int:
        return self.mask.shape[1]

    @property
    def area_cells(self) -> int:
        return int(self.mask.sum())

    @property
    def area_m2(self) -> float:
        return self.area_cells * self.cell ** 2

    def window_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.side in self.window_sides]

    def cells(self, metres: float) -> int:
        return int(round(metres / self.cell))

    def to_json(self) -> dict:
        return {
            "cell": self.cell,
            "mask": self.mask.astype(int).tolist(),
            "entry": vars(self.entry),
            "window_sides": list(self.window_sides),
        }


def _inside(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    """Even-odd point-in-polygon test."""
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > py) != (y1 > py):
            if px < x0 + (py - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


def mask_from_outline(outline_m: list[tuple[float, float]], cell: float) -> np.ndarray:
    xs = [p[0] for p in outline_m]
    ys = [p[1] for p in outline_m]
    if min(xs) < 0 or min(ys) < 0:
        raise ValueError("outline must start at (0, 0) or larger")
    W = int(round(max(xs) / cell))
    H = int(round(max(ys) / cell))
    mask = np.zeros((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            mask[y, x] = _inside((x + 0.5) * cell, (y + 0.5) * cell, outline_m)
    return mask


def exterior_edges(mask: np.ndarray) -> list[Edge]:
    """Maximal runs of cell sides where an inside cell meets an outside cell (or the bbox)."""
    H, W = mask.shape
    pad = np.pad(mask, 1)
    runs: list[Edge] = []
    # N/S faces run along x; W/E faces run along y.
    for side, dy, dx in (("N", -1, 0), ("S", 1, 0), ("W", 0, -1), ("E", 0, 1)):
        horizontal = side in ("N", "S")
        outer, inner = (H, W) if horizontal else (W, H)
        for a in range(outer):
            start = None
            for b in range(inner + 1):
                if b < inner:
                    y, x = (a, b) if horizontal else (b, a)
                    face = mask[y, x] and not pad[y + 1 + dy, x + 1 + dx]
                else:
                    face = False
                if face and start is None:
                    start = b
                elif not face and start is not None:
                    fixed = a + (1 if side in ("S", "E") else 0)
                    runs.append(Edge(side, fixed, start, b))
                    start = None
    return runs


def hole_rects(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover the outside cells of the bounding box with rectangles (row runs merged downwards)."""
    H, W = mask.shape
    open_rects: dict[tuple[int, int], list[int]] = {}  # (x0, x1) -> [y0, h]
    out: list[tuple[int, int, int, int]] = []
    for y in range(H + 1):
        runs = set()
        if y < H:
            x = 0
            while x < W:
                if not mask[y, x]:
                    x0 = x
                    while x < W and not mask[y, x]:
                        x += 1
                    runs.add((x0, x))
                else:
                    x += 1
        for key in list(open_rects):
            if key in runs:
                open_rects[key][1] += 1
                runs.discard(key)
            else:
                y0, h = open_rects.pop(key)
                out.append((key[0], y0, key[1] - key[0], h))
        for key in runs:
            open_rects[key] = [y, 1]
    return out


def entry_edge(mask: np.ndarray, cell: float, side: str, start_m: float, end_m: float) -> Edge:
    """The entrance as a span on an exterior face; it must lie on one exterior run."""
    s, e = int(round(start_m / cell)), int(round(end_m / cell))
    for edge in exterior_edges(mask):
        if edge.side == side and edge.s <= s and e <= edge.e:
            return Edge(side, edge.fixed, s, e)
    raise ValueError(f"entry {side} {start_m}-{end_m} m is not on a single exterior wall")


def build_shell(spec: dict) -> Shell:
    """spec: {"cell": 0.5, "outline": [[x, y], ...] | "walls": [...], "entry": {...}, "window_sides": [...]}"""
    cell = float(spec.get("cell", 0.5))
    if "outline" in spec:
        mask = mask_from_outline([tuple(p) for p in spec["outline"]], cell)
    elif "walls" in spec:
        from .walls import mask_from_walls
        mask = mask_from_walls(spec["walls"], cell)
    else:
        raise ValueError("shell needs an outline or walls")
    if not mask.any():
        raise ValueError("shell encloses no floor; run `floorgen walls` to find gaps")
    ent = spec["entry"]
    entry = entry_edge(mask, cell, ent["side"], ent["from"], ent["to"])
    return Shell(
        mask=mask,
        cell=cell,
        entry=entry,
        window_sides=tuple(spec.get("window_sides", SIDES)),
        holes=hole_rects(mask),
        edges=exterior_edges(mask),
    )
