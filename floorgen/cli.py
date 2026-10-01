"""floorgen run | ingest | calibrate | walls"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def cmd_run(a) -> None:
    from .pipeline import RoundConfig, run_round

    out = Path(a.out)
    rounds = sorted(out.glob("round-*")) if out.exists() else []
    cfg = RoundConfig(brief_path=a.brief, out=out, round=a.round or len(rounds) + 1, n=a.n,
                      workers=a.workers, time_limit=a.time_limit, cheap=a.cheap, strong=a.strong,
                      tau=a.tau, strong_max=a.strong_max, k=a.k, seed=a.seed)
    stats = run_round(cfg)
    print(json.dumps(stats["funnel"]))
    if "simulated_true_quality" in stats:
        print("simulated client quality:", json.dumps(stats["simulated_true_quality"]))


def cmd_ingest(a) -> None:
    from .pipeline import ingest

    n = ingest(a.brief, Path(a.out), Path(a.labels))
    print(f"ingested {n} labels; feedback and sampler updated in {a.out}")


def cmd_calibrate(a) -> None:
    """How well does the strong judge agree with you? Pick = 1, reject = 0 against judge score."""
    labels = {lab["id"]: lab["verdict"] == "pick" for lab in json.loads(Path(a.labels).read_text())["labels"]}
    scores = {}
    for line in open(Path(a.round_dir) / "plans.jsonl"):
        r = json.loads(line)
        if r["plan"]["id"] in labels and "strong" in r:
            scores[r["plan"]["id"]] = r["strong"]["u"]
    pos = [scores[i] for i, y in labels.items() if y and i in scores]
    neg = [scores[i] for i, y in labels.items() if not y and i in scores]
    if not pos or not neg:
        print("need at least one pick and one reject with judge scores")
        return
    # AUC: chance the judge scores a random pick above a random reject (0.5 = no better than chance)
    auc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    print(f"{len(pos)} picks, {len(neg)} rejects; judge AUC vs you = {auc:.2f}")
    print("Shortlists are already judge-filtered, so this understates accuracy on the full pool (range restriction).")
    if len(pos) + len(neg) < 30:
        print("Fewer than 30 labels: treat this as a hint, not a measurement.")


def cmd_walls(a) -> None:
    from .walls import find_issues, mask_from_walls

    spec = json.loads(Path(a.file).read_text())
    walls, cell = spec["walls"], spec.get("cell", 0.5)
    issues = find_issues(walls, cell, a.tol)
    for n, iss in enumerate(issues, 1):
        print(f"[{n}/{len(issues)}] {iss.kind}: {iss.message}")
        for f in iss.fixes:
            print(f"    - {f.label}: {f.detail}")
    if not issues:
        m = mask_from_walls(walls, cell)
        print(f"no issues; encloses {m.sum() * cell * cell:.1f} m²" if m.any() else "no issues found, but the walls enclose nothing")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="floorgen", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one round of the funnel")
    r.add_argument("brief")
    r.add_argument("--out", default="runs/demo")
    r.add_argument("--round", type=int, default=0, help="default: next round in --out")
    r.add_argument("-n", type=int, default=2000)
    r.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    r.add_argument("--time-limit", type=float, default=0.03, help="CP-SAT deterministic time per plan; higher = closer to targets, slower")
    r.add_argument("--cheap", default="sim", help="sim[:r] | jev | claude model id")
    r.add_argument("--strong", default="sim", help="sim[:r] | claude-sonnet-5-5[:repeats]")
    r.add_argument("--tau", type=float, default=0.35)
    r.add_argument("--strong-max", type=int, default=400)
    r.add_argument("-k", type=int, default=12)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_run)

    i = sub.add_parser("ingest", help="feed labels.json from the review page back into the loop")
    i.add_argument("brief")
    i.add_argument("labels")
    i.add_argument("--out", default="runs/demo")
    i.set_defaults(fn=cmd_ingest)

    c = sub.add_parser("calibrate", help="agreement between the strong judge and your labels")
    c.add_argument("round_dir")
    c.add_argument("labels")
    c.set_defaults(fn=cmd_calibrate)

    w = sub.add_parser("walls", help="find gaps and overshoots in scanned walls, with fixes")
    w.add_argument("file")
    w.add_argument("--tol", type=float, default=1.0, help="largest gap or overshoot to flag, metres")
    w.set_defaults(fn=cmd_walls)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
