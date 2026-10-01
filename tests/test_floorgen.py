import json
import random
from pathlib import Path

import numpy as np
import pytest

from floorgen.brief import brief_from_dict, load_brief
from floorgen.checks import check, valid
from floorgen.features import Archive, features, standardise, type_raster
from floorgen.generate import generate
from floorgen.learn import Sampler
from floorgen.select import shortlist
from floorgen.shell import build_shell
from floorgen.walls import find_issues, mask_from_walls

ROOT = Path(__file__).resolve().parent.parent
RECT = ROOT / "briefs" / "one_bed_9x6.json"
LSHAPE = ROOT / "briefs" / "two_bed_L.json"


def plans(path, n, seed=0, tl=0.05):
    b = load_brief(path)
    s, rng = Sampler.prior(b), random.Random(seed)
    return b, [generate(b, s.draw(rng), seed * 1000 + i, tl) for i in range(n)]


# ---------------------------------------------------------------- shell + walls

def test_l_shape_shell_from_walls():
    b = load_brief(LSHAPE)
    assert b.shell.area_m2 == 68.0
    assert b.shell.holes == [(12, 0, 8, 6)]          # the missing NE corner
    assert {e.side for e in b.shell.window_edges()} == {"N", "E", "W"}


def test_entry_must_sit_on_one_exterior_wall():
    spec = json.loads(RECT.read_text())["shell"]
    spec["entry"] = {"side": "N", "from": 8.5, "to": 9.5}   # runs past the corner
    with pytest.raises(ValueError):
        build_shell(spec)


def test_wall_gap_and_overshoot_detected_and_fixes_close_the_shell():
    walls = [[0, 0, 6, 0], [6, 0, 6, 4.5], [6, 4, 0, 4], [0, 4, 0, 0.5]]  # overshoot + gap
    issues = {i.kind: i for i in find_issues(walls, 0.5)}
    assert set(issues) == {"overshoot", "gap"}
    assert issues["overshoot"].fixes[0].walls[1] == (6.0, 0.0, 6.0, 4.0)   # trimmed to the junction
    assert not mask_from_walls(walls, 0.5).any()                           # open shell encloses nothing
    for fix in issues["gap"].fixes:
        assert mask_from_walls(fix.walls, 0.5).sum() == 12 * 8


def test_collinear_gap_offers_three_fixes():
    walls = [[0, 0, 4, 0], [4.5, 0, 8, 0], [8, 0, 8, 4], [8, 4, 0, 4], [0, 4, 0, 0]]
    gap = next(i for i in find_issues(walls, 0.5) if i.kind == "gap")
    assert [f.label.split()[0] for f in gap.fixes] == ["Extend", "Extend", "Add"]
    for f in gap.fixes:
        assert mask_from_walls(f.walls, 0.5).sum() == 16 * 8


# ---------------------------------------------------------------- generate + verify

@pytest.mark.parametrize("path", [RECT, LSHAPE])
def test_generated_plans_pass_every_check(path):
    b, ps = plans(path, 12)
    built = [p for p in ps if p.ok]
    assert len(built) >= 10
    for p in built:
        m = check(b, p)
        assert valid(m), (p.id, {k: v for k, v in m.items() if v < 0})


def test_same_theta_and_seed_give_same_plan():
    b = load_brief(RECT)
    th = Sampler.prior(b).draw(random.Random(4))
    a, c = generate(b, th, 77, 0.05), generate(b, th, 77, 0.05)
    assert a.to_json()["rooms"] == c.to_json()["rooms"]


def test_checks_catch_a_broken_plan():
    b, ps = plans(RECT, 3)
    p = next(p for p in ps if p.ok)
    p.rooms[0].w += 1                        # now overlaps a neighbour or leaves the shell
    m = check(b, p)
    assert m["tiling"] < 0 and not valid(m)


def test_door_rules_are_enforced():
    raw = json.loads(RECT.read_text())
    raw["doors"]["bath"] = ["bedroom"]       # ensuite only
    b = brief_from_dict(raw)
    rng = random.Random(2)
    for i in range(6):
        p = generate(b, Sampler.prior(b).draw(rng), i, 0.05)
        if p.ok:
            bath = next(k for k, r in enumerate(p.rooms) if r.type == "bath")
            parent = next(d.a for d in p.doors if d.b == bath)
            assert p.rooms[parent].type == "bedroom"


# ---------------------------------------------------------------- dedupe, select, learn

def test_archive_rejects_copies():
    b, ps = plans(RECT, 4)
    p = next(p for p in ps if p.ok)
    arch = Archive(b.shell.H * b.shell.W, 0.06)
    r = type_raster(b, p)
    assert arch.offer(r) and not arch.offer(r.copy())


def test_shortlist_is_distinct_and_sized():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(50, 6))
    picks, wild, groups = shortlist(rng.random(50), Z, k=12, wildcards=2)
    assert len(picks) == len(set(picks)) == 12
    assert sum(wild) == 2 and sum(groups) == 50


def test_sampler_moves_toward_elite_within_leash():
    b = load_brief(RECT)
    s = Sampler.prior(b)
    th = s.draw(random.Random(0))
    th["share"]["living"] = 0.95
    s.update([(th, 1.0)] * 5, eta=0.3, kappa=3.0)
    assert s.mu["share:living"] > 0.5
    assert s.kl_to_prior() <= 3.0 + 1e-9


def test_features_are_finite():
    b, ps = plans(LSHAPE, 4)
    rows = [features(b, p) for p in ps if p.ok]
    Z, keys = standardise(rows)
    assert np.isfinite(Z).all() and "share:living" in keys


# ---------------------------------------------------------------- pipeline

def test_round_end_to_end(tmp_path):
    from floorgen.pipeline import RoundConfig, ingest, run_round

    cfg = RoundConfig(brief_path=str(RECT), out=tmp_path, n=60, workers=2, time_limit=0.03, k=6, strong_max=40)
    stats = run_round(cfg, log=lambda *_: None)
    assert stats["funnel"]["shortlist"] == 6
    rdir = tmp_path / "round-01"
    assert (rdir / "review.html").exists() and (rdir / "plans.jsonl").exists()
    short = [json.loads(line) for line in open(rdir / "plans.jsonl") if "shortlist" in json.loads(line)]
    labels = {"round": 1, "labels": [{"id": r["plan"]["id"], "verdict": "pick", "summary": "x", "reason": "bright",
                                      "theta": r["plan"]["theta"]} for r in short[:2]]}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    assert ingest(str(RECT), tmp_path, tmp_path / "labels.json") == 2
    assert len(json.loads((tmp_path / "feedback.json").read_text())) == 2
