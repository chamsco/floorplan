"""The brief, sorted into the paper's three layers.

- rules (never acceptable to break): room minimums, door rules, daylight, entry.
  The generator builds these in and checks.py verifies them again.
- targets (preferences with a measurable value): area shares, proportions. Scored by targets.py.
- taste (how it should feel): free text the model judges read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .shell import Shell, build_shell


@dataclass(frozen=True)
class RoomSpec:
    type: str
    min_width: float            # metres, shortest side
    min_area: float             # m²
    share: tuple[float, float]  # range of floor-area share the generator may target
    daylight: bool = False      # rule: needs a window wall of at least window_min
    optional: bool = False      # generator may leave it out
    count: int = 1


@dataclass
class Room:
    """One room instance in a plan's program, e.g. bedroom_2."""
    name: str
    spec: RoomSpec

    @property
    def type(self) -> str:
        return self.spec.type


@dataclass
class Brief:
    name: str
    shell: Shell
    rooms: dict[str, RoomSpec]
    entry_rooms: tuple[str, ...]        # room types the front door may open into
    doors: dict[str, tuple[str, ...]]   # child type -> types it may be entered from
    door_width: float = 1.0
    window_min: float = 1.0
    max_aspect: float = 3.0
    open_pairs: tuple[tuple[str, str], ...] = (("kitchen", "living"),)
    taste: str = ""
    targets: dict = field(default_factory=dict)

    def program(self, include_optional: dict[str, bool] | None = None) -> list[Room]:
        include_optional = include_optional or {}
        out: list[Room] = []
        for t, spec in self.rooms.items():
            if spec.optional and not include_optional.get(t, False):
                continue
            for k in range(spec.count):
                out.append(Room(t if k == 0 else f"{t}_{k + 1}", spec))
        return out

    def can_enter(self, child: str, parent: str) -> bool:
        return parent in self.doors.get(child, ())


def load_brief(path: str | Path) -> Brief:
    raw = json.loads(Path(path).read_text())
    return brief_from_dict(raw)


def brief_from_dict(raw: dict) -> Brief:
    rooms = {
        t: RoomSpec(
            type=t,
            min_width=float(r["min_width"]),
            min_area=float(r.get("min_area", 0)),
            share=tuple(r["share"]),
            daylight=bool(r.get("daylight", False)),
            optional=bool(r.get("optional", False)),
            count=int(r.get("count", 1)),
        )
        for t, r in raw["rooms"].items()
    }
    return Brief(
        name=raw.get("name", "brief"),
        shell=build_shell(raw["shell"]),
        rooms=rooms,
        entry_rooms=tuple(raw["entry_rooms"]),
        doors={k: tuple(v) for k, v in raw["doors"].items()},
        door_width=float(raw.get("door_width", 1.0)),
        window_min=float(raw.get("window_min", 1.0)),
        max_aspect=float(raw.get("max_aspect", 3.0)),
        open_pairs=tuple(tuple(p) for p in raw.get("open_pairs", [["kitchen", "living"]])),
        taste=raw.get("taste", ""),
        targets=raw.get("targets", {}),
    )
