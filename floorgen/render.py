"""Draw a plan: SVG for the review page, PNG for model judges (they do not read SVG)."""
from __future__ import annotations

import io

from .brief import Brief
from .generate import Plan

FILL = {
    "hall": "#ecebe7", "living": "#f3efe6", "kitchen": "#efe9df", "bedroom": "#e8ece6",
    "bath": "#e4eaec", "study": "#ece8e3", "storage": "#e6e4df",
}
INK = "#2b2b2b"
WINDOW = "#4a90c2"


def _geom(brief: Brief, plan: Plan, s: float):
    """Shared drawing primitives in pixel units: rooms, wall lines, windows, door gaps."""
    pad = 24
    sh = brief.shell
    px = lambda v: pad + v * s  # noqa: E731
    rooms = [(r, px(r.x), px(r.y), px(r.x + r.w), px(r.y + r.h)) for r in plan.rooms]
    windows = []
    for e in sh.window_edges():
        if e.side in ("N", "S"):
            windows.append((px(e.s), px(e.fixed), px(e.e), px(e.fixed)))
        else:
            windows.append((px(e.fixed), px(e.s), px(e.fixed), px(e.e)))
    doors = []
    for d in plan.doors:
        if d.orient == "h":
            doors.append((d, px(d.t0), px(d.fixed), px(d.t1), px(d.fixed)))
        else:
            doors.append((d, px(d.fixed), px(d.t0), px(d.fixed), px(d.t1)))
    size = (int(2 * pad + sh.W * s), int(2 * pad + sh.H * s))
    return rooms, windows, doors, size


def svg(brief: Brief, plan: Plan, scale: float = 24) -> str:
    rooms, windows, doors, (Wp, Hp) = _geom(brief, plan, scale)
    c = brief.shell.cell
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wp} {Hp}" font-family="Inter, system-ui, sans-serif">']
    for r, x0, y0, x1, y1 in rooms:
        out.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="{FILL.get(r.type, "#eee")}" '
                   f'stroke="{INK}" stroke-width="1.5"/>')
    sh = brief.shell
    for e in sh.edges:
        p = lambda v: 24 + v * scale  # noqa: E731
        if e.side in ("N", "S"):
            out.append(f'<line x1="{p(e.s)}" y1="{p(e.fixed)}" x2="{p(e.e)}" y2="{p(e.fixed)}" stroke="{INK}" stroke-width="5"/>')
        else:
            out.append(f'<line x1="{p(e.fixed)}" y1="{p(e.s)}" x2="{p(e.fixed)}" y2="{p(e.e)}" stroke="{INK}" stroke-width="5"/>')
    for x0, y0, x1, y1 in windows:
        out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{WINDOW}" stroke-width="2.5" stroke-dasharray="10 4"/>')
    for d, x0, y0, x1, y1 in doors:
        w = 7 if d.a < 0 else 4
        if d.open:
            out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#fafafa" stroke-width="{w}"/>')
            out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{INK}" stroke-width="1" stroke-dasharray="4 4"/>')
        else:
            out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#ffffff" stroke-width="{w}"/>')
            out.append(f'<circle cx="{x0}" cy="{y0}" r="2" fill="{INK}"/><circle cx="{x1}" cy="{y1}" r="2" fill="{INK}"/>')
    for r, x0, y0, x1, y1 in rooms:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        out.append(f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="11" fill="{INK}">{r.name.replace("_", " ").title()}</text>')
        out.append(f'<text x="{cx}" y="{cy + 11}" text-anchor="middle" font-size="9" fill="#777">{r.area * c * c:.1f} m²</text>')
    ent = [d for d in doors if d[0].a < 0]
    if ent:
        _, x0, y0, x1, y1 = ent[0]
        out.append(f'<text x="{(x0 + x1) / 2}" y="{max(y0, y1) + 16}" text-anchor="middle" font-size="8" '
                   f'letter-spacing="1" fill="#888">ENTRY</text>')
    out.append("</svg>")
    return "".join(out)


def png(brief: Brief, plan: Plan, scale: float = 40) -> bytes:
    from PIL import Image, ImageDraw

    rooms, windows, doors, size = _geom(brief, plan, scale)
    c = brief.shell.cell
    im = Image.new("RGB", (size[0], size[1] + 12), "white")
    dr = ImageDraw.Draw(im)
    for r, x0, y0, x1, y1 in rooms:
        dr.rectangle([x0, y0, x1, y1], fill=FILL.get(r.type, "#eeeeee"), outline=INK, width=2)
    for e in brief.shell.edges:
        p = lambda v: 24 + v * scale  # noqa: E731
        line = [p(e.s), p(e.fixed), p(e.e), p(e.fixed)] if e.side in ("N", "S") else [p(e.fixed), p(e.s), p(e.fixed), p(e.e)]
        dr.line(line, fill=INK, width=6)
    for x0, y0, x1, y1 in windows:
        dr.line([x0, y0, x1, y1], fill=WINDOW, width=3)
    for d, x0, y0, x1, y1 in doors:
        dr.line([x0, y0, x1, y1], fill="white", width=8 if d.a < 0 else 5)
        if d.open:
            n = int(max(abs(x1 - x0), abs(y1 - y0)) // 8)
            for k in range(0, n, 2):
                t0, t1 = k / n, (k + 1) / n
                dr.line([x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1], fill=INK, width=1)
    for r, x0, y0, x1, y1 in rooms:
        label = f"{r.name.replace('_', ' ').upper()}\n{r.area * c * c:.1f} m2"
        dr.multiline_text(((x0 + x1) / 2, (y0 + y1) / 2), label, fill=INK, anchor="mm", align="center")
    ent = [d for d in doors if d[0].a < 0]
    if ent:
        _, x0, y0, x1, y1 = ent[0]
        dr.text(((x0 + x1) / 2, max(y0, y1) + 14), "ENTRY", fill="#666666", anchor="mm")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()
