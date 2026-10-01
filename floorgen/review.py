"""The review page: one self-contained HTML file per round, for the person's picks.

Layout follows a wizard: the funnel's stages down the left, the shortlist in the
middle, the selected plan large on the right. Pick or reject each plan with an
optional one-line reason, then "Export labels" downloads labels.json for
`floorgen ingest`.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from .brief import Brief
from .render import svg


def summary(brief: Brief, r: dict) -> str:
    p, c = r["plan"], brief.shell.cell
    parts = [f"{x.name.replace('_', ' ')} {x.area * c * c:.0f} m²" for x in p.rooms if x.type != "hall"]
    f = r["features"]
    if f.get("open_kitchen"):
        parts.append("open kitchen")
    parts.append(f"front door into {p.rooms[p.entry].name}")
    if f.get("bath_via_bedroom"):
        parts.append("bath off bedroom")
    return ", ".join(parts)


STAGES = [("generated", "Generate"), ("valid", "Verify rules"), ("distinct", "Remove near-copies"),
          ("cheap_pass", "Cheap judge"), ("strong_judged", "Strong judge"), ("shortlist", "Shortlist")]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plan review</title>
<style>
:root{--bg:#e9e9eb;--panel:#fff;--ink:#1f1f1f;--mute:#77777c;--line:#e4e4e7;--soft:#f6f6f7;--ok:#22a06b;--pick:#1f1f1f;--rej:#c2410c;--focus:#3b82f6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111113;--panel:#1b1b1e;--ink:#ededef;--mute:#9a9aa1;--line:#2a2a2e;--soft:#232327;--pick:#ededef}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 Inter,system-ui,-apple-system,sans-serif}
.app{max-width:1280px;margin:24px auto;background:var(--panel);border-radius:14px;box-shadow:0 10px 40px rgba(0,0,0,.08);overflow:hidden}
header{display:flex;align-items:center;gap:8px;padding:14px 20px;border-bottom:1px solid var(--line)}
header .crumb{color:var(--mute)}header .sep{color:var(--mute)}header .grow{flex:1}
button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:7px 14px;cursor:pointer}
button.primary{background:var(--pick);color:var(--panel);border-color:var(--pick)}button:disabled{opacity:.4;cursor:default}
.cols{display:grid;grid-template-columns:220px 360px 1fr;height:calc(100vh - 110px);min-height:560px}.list,.view{min-height:0}
aside{border-right:1px solid var(--line);padding:16px}aside h4{margin:0 0 8px;font-size:10px;letter-spacing:.06em;color:var(--mute);font-weight:600}
.stage{display:flex;gap:8px;align-items:center;padding:7px 0;border-bottom:1px solid var(--soft)}
.dot{width:14px;height:14px;border-radius:50%;background:var(--ok);display:grid;place-items:center;color:#fff;font-size:9px;flex:none}
.stage .n{margin-left:auto;color:var(--mute);font-variant-numeric:tabular-nums}
.cost{margin-top:16px;color:var(--mute);font-size:12px}
.list{border-right:1px solid var(--line);display:flex;flex-direction:column}
.list .head{padding:16px 16px 8px}.list .head small{color:var(--mute)}.list .head h2{margin:2px 0 0;font-size:15px}
.nav{display:flex;align-items:center;gap:6px;float:right;color:var(--mute)}.nav button{padding:2px 8px;border:0}
.cards{overflow:auto;padding:0 12px 12px;flex:1}
.card{display:flex;gap:10px;padding:8px;border:1px solid var(--line);border-radius:10px;margin-bottom:8px;cursor:pointer;align-items:center}
.card.sel{border-color:var(--ink);box-shadow:0 0 0 1px var(--ink)}.card .th{width:84px;flex:none;background:var(--soft);border-radius:6px;padding:4px}
.card .th svg{width:100%;height:auto;display:block}.card b{display:block;font-size:12px}.card small{color:var(--mute);display:block;font-size:11px}
.badge{font-size:10px;padding:1px 6px;border-radius:9px;background:var(--soft);color:var(--mute);margin-left:4px}
.state{margin-left:auto;width:14px;height:14px;border-radius:50%;border:1.5px solid var(--line);flex:none}
.state.pick{background:var(--pick);border-color:var(--pick)}.state.reject{background:var(--rej);border-color:var(--rej)}
.view{background:var(--soft);display:flex;flex-direction:column}
.canvas{flex:1;display:flex;align-items:center;justify-content:center;padding:20px;min-height:0}.canvas svg{height:400px;width:auto;max-width:100%;background:#fafafa;border-radius:10px}
.detail{padding:12px 20px;background:var(--panel);border-top:1px solid var(--line)}
.crit{display:flex;gap:14px;flex-wrap:wrap;color:var(--mute);font-size:12px;margin:6px 0}.crit b{color:var(--ink)}
.facts{white-space:pre-line;color:var(--mute);font-size:11px;max-height:90px;overflow:auto}
.actions{display:flex;gap:8px;align-items:center;padding:12px 20px;border-top:1px solid var(--line);background:var(--panel)}
.actions input{flex:1;font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink)}
@media (max-width:900px){.app{margin:0;border-radius:0}.cols{grid-template-columns:1fr}aside,.list{border-right:0;border-bottom:1px solid var(--line)}.cards{max-height:280px}}
</style></head><body>
<div class="app">
<header><span class="crumb">Projects</span><span class="sep">›</span><b>__BRIEF__</b><span class="grow"></span>
<span class="crumb">Round __ROUND__</span><button id="export" class="primary">Export labels</button></header>
<div class="cols">
<aside><h4>FUNNEL</h4>__STAGES__<div class="cost">__COST__</div></aside>
<section class="list"><div class="head"><small>Shortlist · __K__ plans</small>
<span class="nav"><button id="prev">‹</button><span id="pos"></span><button id="next">›</button></span>
<h2>Pick what you would live in</h2></div><div class="cards" id="cards"></div></section>
<section class="view"><div class="canvas" id="canvas"></div>
<div class="detail"><div id="reason"></div><div class="crit" id="crit"></div><div class="facts" id="facts"></div></div>
<div class="actions"><input id="why" placeholder="Why? (optional, one line)"><button id="reject">Reject</button><button id="pick" class="primary">Pick →</button></div>
</section></div></div>
<script>
const DATA = __DATA__;
const KEY = 'floorgen-labels-' + DATA.round + '-' + DATA.brief;
let labels = {}; try { labels = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
let cur = 0;
const $ = id => document.getElementById(id);
function save(){ try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function cards(){
  $('cards').innerHTML = DATA.plans.map((p, i) => `<div class="card ${i === cur ? 'sel' : ''}" data-i="${i}">
    <div class="th">${p.svg}</div><div><b>Plan ${i + 1}${p.wildcard ? '<span class="badge">wildcard</span>' : ''}</b>
    <small>score ${(p.score * 100).toFixed(0)} · stands for ${p.group}</small><small>${p.reason}</small></div>
    <span class="state ${labels[p.id] ? labels[p.id].verdict : ''}"></span></div>`).join('');
  document.querySelectorAll('.card').forEach(c => c.onclick = () => show(+c.dataset.i));
}
function show(i){
  cur = Math.max(0, Math.min(DATA.plans.length - 1, i)); const p = DATA.plans[cur];
  $('canvas').innerHTML = p.svg; $('pos').textContent = `${cur + 1}/${DATA.plans.length}`;
  $('reason').innerHTML = p.reason;
  $('crit').innerHTML = Object.entries(p.criteria || {}).map(([k, v]) => `<span>${k} <b>${v.toFixed(1)}</b></span>`).join('');
  $('facts').textContent = p.facts; $('why').value = labels[p.id] ? labels[p.id].reason : '';
  cards();
}
function mark(verdict){
  const p = DATA.plans[cur];
  labels[p.id] = { verdict, reason: $('why').value.trim(), summary: p.summary, theta: p.theta, id: p.id };
  save(); show(cur + 1);
}
$('pick').onclick = () => mark('pick'); $('reject').onclick = () => mark('reject');
$('prev').onclick = () => show(cur - 1); $('next').onclick = () => show(cur + 1);
document.addEventListener('keydown', e => { if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowDown' || e.key === 'ArrowRight') show(cur + 1);
  if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') show(cur - 1); });
$('export').onclick = () => {
  const blob = new Blob([JSON.stringify({ round: DATA.round, labels: Object.values(labels) }, null, 2)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'labels.json'; a.click();
};
show(0);
</script></body></html>"""


def write_review(brief: Brief, short: list[dict], stats: dict, path: Path) -> None:
    from .features import facts

    plans = []
    # best first; wildcards last so they read as the exploratory picks they are
    short = sorted(short, key=lambda r: (r["shortlist"]["wildcard"], -r.get("strong", {}).get("u", 0)))
    for r in short:
        s = r.get("strong", {})
        plans.append({
            "id": r["plan"].id, "theta": r["plan"].theta, "svg": svg(brief, r["plan"]),
            "score": s.get("u", 0), "reason": html.escape(s.get("reason") or r.get("cheap", {}).get("reason", "")),
            "criteria": s.get("criteria"), "facts": facts(brief, r["plan"]), "summary": summary(brief, r),
            "wildcard": r["shortlist"]["wildcard"], "group": r["shortlist"]["group"],
        })
    f = stats["funnel"]
    stages = "".join(f'<div class="stage"><span class="dot">✓</span>{label}<span class="n">{f.get(key, 0):,}</span></div>'
                     for key, label in STAGES)
    cost = sum(stats.get("cost", {}).values())
    sim = stats.get("simulated_true_quality")
    note = f"Model cost ${cost:.2f}"
    if sim:
        note += (f"<br>Simulated client: valid plans average {sim['all_valid']:.2f}, "
                 f"shortlist {sim['shortlist']:.2f}, best possible {sim['best_possible_top_k']:.2f}")
    page = (PAGE.replace("__BRIEF__", html.escape(brief.name)).replace("__ROUND__", str(stats["round"]))
            .replace("__K__", str(len(plans))).replace("__STAGES__", stages).replace("__COST__", note)
            .replace("__DATA__", json.dumps({"round": stats["round"], "brief": brief.name, "plans": plans}).replace("</", "<\\/")))
    path.write_text(page)
