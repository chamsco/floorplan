"""One round of the funnel: generate -> verify -> dedupe -> cheap judge -> strong judge -> shortlist.

Every plan that survives dedupe is written to plans.jsonl with its outcome at each
stage, so any decision can be traced and any plan regenerated from (theta, seed).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .brief import Brief, load_brief
from .checks import check, valid
from .features import Archive, features, standardise, target_score, type_raster
from .generate import Plan, generate
from .judges import client_utility, make_judge
from .learn import Sampler
from .render import svg
from .review import write_review
from .select import shortlist

_BRIEF: Brief | None = None


def _init(brief_path: str) -> None:
    global _BRIEF
    _BRIEF = load_brief(brief_path)


def _gen(job: tuple[dict, int, float]) -> dict:
    theta, seed, tl = job
    return generate(_BRIEF, theta, seed, tl).to_json()


@dataclass
class RoundConfig:
    brief_path: str
    out: Path
    round: int = 1
    n: int = 2000
    workers: int = 4
    time_limit: float = 0.03       # CP-SAT deterministic time per plan (~0.15 s wall)
    delta: float = 0.06           # novelty: min share of cells that must differ
    cheap: str = "sim"            # "sim[:r]" | "jev" | claude model id
    strong: str = "sim"
    cheap_max: int = 4000         # cap on plans sent to the cheap judge (best u0 first)
    tau: float = 0.35             # cheap judge pass threshold: keep if u >= tau
    audit: float = 0.02           # share of cheap-judge rejects sent to the strong judge anyway
    strong_max: int = 400
    k: int = 12
    wildcards: int = 2
    seed: int = 0


def run_round(cfg: RoundConfig, log=print) -> dict:
    brief = load_brief(cfg.brief_path)
    root = cfg.out
    rdir = root / f"round-{cfg.round:02d}"
    (rdir / "svg").mkdir(parents=True, exist_ok=True)
    sampler_path = root / "sampler.json"
    sampler = Sampler.load(sampler_path) if sampler_path.exists() else Sampler.prior(brief)
    feedback_path = root / "feedback.json"
    feedback = json.loads(feedback_path.read_text()) if feedback_path.exists() else []
    fb_lines = [f"{'PICKED' if f['verdict'] == 'pick' else 'REJECTED'}: {f['summary']}"
                + (f" (client said: “{f['reason']}”)" if f.get("reason") else "") for f in feedback]
    stats: dict = {"round": cfg.round, "brief": brief.name, "funnel": {}, "cost": {}, "seconds": {}}
    rng = random.Random(f"{cfg.seed}:{cfg.round}")

    # 1 generate
    t = time.perf_counter()
    jobs = [(sampler.draw(rng), rng.getrandbits(31), cfg.time_limit) for _ in range(cfg.n)]
    with Pool(cfg.workers, initializer=_init, initargs=(cfg.brief_path,)) as pool:
        plans = [Plan.from_json(d) for d in pool.imap(_gen, jobs, chunksize=16)]
    built = [p for p in plans if p.ok]
    stats["funnel"]["generated"] = cfg.n
    stats["funnel"]["constructed"] = len(built)
    stats["seconds"]["generate"] = time.perf_counter() - t
    log(f"generate   {cfg.n} -> {len(built)} constructed  ({stats['seconds']['generate']:.0f}s)")

    # 2 verify
    rows = []
    failures: dict[str, int] = {}
    for p in built:
        m = check(brief, p)
        if valid(m):
            rows.append({"plan": p, "margins": m})
        else:
            for k, v in m.items():
                if v < 0:
                    failures[k] = failures.get(k, 0) + 1
    stats["funnel"]["valid"] = len(rows)
    stats["check_failures"] = failures
    log(f"verify     {len(built)} -> {len(rows)} valid" + (f"  failures: {failures}" if failures else ""))

    # 3 describe + dedupe; best u0 first so the kept copy of a near-duplicate group is the best one
    for r in rows:
        r["features"] = features(brief, r["plan"])
        r["u0"] = target_score(brief, r["features"])
    rows.sort(key=lambda r: -r["u0"])
    arch = Archive(brief.shell.H * brief.shell.W, cfg.delta)
    pool_rows = [r for r in rows if arch.offer(type_raster(brief, r["plan"]))]
    stats["funnel"]["distinct"] = len(pool_rows)
    log(f"dedupe     {len(rows)} -> {len(pool_rows)} distinct (delta {cfg.delta})")

    # 4 cheap judge
    t = time.perf_counter()
    cheap = make_judge(cfg.cheap, "filter", seed=cfg.seed * 7 + 1)
    to_cheap = pool_rows[: cfg.cheap_max]
    verdicts = cheap.judge_many(brief, [(r["plan"], r["features"]) for r in to_cheap], fb_lines)
    for r, v in zip(to_cheap, verdicts):
        r["cheap"] = {"u": v.u, "reason": v.reason, **v.detail}
    passed = [r for r in to_cheap if r["cheap"]["u"] >= cfg.tau]
    rejected = [r for r in to_cheap if r["cheap"]["u"] < cfg.tau]
    audits = rng.sample(rejected, int(round(cfg.audit * len(rejected))))
    stats["cost"]["cheap"] = sum(v.cost for v in verdicts)
    stats["seconds"]["cheap"] = time.perf_counter() - t
    stats["funnel"]["cheap_pass"] = len(passed)
    log(f"cheap      {len(to_cheap)} -> {len(passed)} pass  [{cheap.name}, tau {cfg.tau}]  (+{len(audits)} audits)")

    # 5 strong judge
    t = time.perf_counter()
    strong = make_judge(cfg.strong, "rank", seed=cfg.seed * 7 + 2)
    passed.sort(key=lambda r: -r["cheap"]["u"])
    to_strong = passed[: cfg.strong_max] + audits
    verdicts = strong.judge_many(brief, [(r["plan"], r["features"]) for r in to_strong], fb_lines)
    for r, v in zip(to_strong, verdicts):
        r["strong"] = {"u": v.u, "reason": v.reason, **{k: x for k, x in v.detail.items() if k != "runs"}}
    for r in audits:
        r["audit"] = True
    stats["cost"]["strong"] = sum(v.cost for v in verdicts)
    stats["seconds"]["strong"] = time.perf_counter() - t
    stats["funnel"]["strong_judged"] = len(to_strong)
    log(f"strong     {len(to_strong)} judged  [{strong.name}]")

    # audit: how many rejects would the strong judge have ranked in its top half?
    judged = [r for r in to_strong if not r.get("audit")]
    if audits and judged:
        median = float(np.median([r["strong"]["u"] for r in judged]))
        miss = sum(r["strong"]["u"] >= median for r in audits)
        stats["audit"] = {"n": len(audits), "rejects_above_strong_median": miss}

    # 6 select
    cands = [r for r in to_strong if not r.get("audit")]
    if cands:
        Z, _ = standardise([r["features"] for r in cands])
        picks, wild, groups = shortlist(np.array([r["strong"]["u"] for r in cands]), Z, cfg.k, cfg.wildcards)
    else:
        picks, wild, groups = [], [], []
    short = []
    for j, w, g in zip(picks, wild, groups):
        r = cands[j]
        r["shortlist"] = {"wildcard": w, "group": g}
        short.append(r)
    stats["funnel"]["shortlist"] = len(short)
    log(f"select     {len(cands)} -> {len(short)} shortlisted ({sum(wild)} wildcards)")

    # simulated mode: we know the hidden client's taste, so report what each stage bought
    if cfg.cheap.startswith("sim") and cfg.strong.startswith("sim") and short:
        true = lambda rs: float(np.mean([client_utility(r["features"]) for r in rs]))  # noqa: E731
        stats["simulated_true_quality"] = {
            "all_valid": true(rows), "shortlist": true(short),
            "best_possible_top_k": float(np.mean(sorted((client_utility(r["features"]) for r in pool_rows), reverse=True)[: cfg.k])),
        }

    # learning: the shortlist is the judges' elite; human picks are added by `floorgen ingest`
    sampler.update([(r["plan"].theta, 1.0) for r in short])
    sampler.save(sampler_path)

    # write
    with open(rdir / "plans.jsonl", "w") as fh:
        for r in pool_rows:
            fh.write(json.dumps({k: (v.to_json() if k == "plan" else v) for k, v in r.items()}) + "\n")
    for r in short:
        (rdir / "svg" / f"{r['plan'].id}.svg").write_text(svg(brief, r["plan"]))
    (rdir / "stats.json").write_text(json.dumps(stats, indent=2))
    write_review(brief, short, stats, rdir / "review.html")
    log(f"wrote      {rdir}/review.html")
    return stats


def ingest(brief_path: str, out: Path, labels_path: Path, pick_weight: float = 3.0) -> int:
    """Feed a person's picks and rejects back: they become judge feedback and the sampler's strongest elite."""
    labels = json.loads(Path(labels_path).read_text())
    feedback_path = out / "feedback.json"
    feedback = json.loads(feedback_path.read_text()) if feedback_path.exists() else []
    sampler_path = out / "sampler.json"
    sampler = Sampler.load(sampler_path) if sampler_path.exists() else Sampler.prior(load_brief(brief_path))
    elite = []
    for lab in labels["labels"]:
        feedback.append({"verdict": lab["verdict"], "summary": lab["summary"], "reason": lab.get("reason", ""),
                         "plan": lab["id"], "round": labels.get("round")})
        if lab["verdict"] == "pick":
            elite.append((lab["theta"], pick_weight))
    feedback_path.write_text(json.dumps(feedback, indent=2))
    if elite:
        sampler.update(elite)
        sampler.save(sampler_path)
    return len(labels["labels"])
