"""Stages 4 and 5: the cheap judge and the strong judge.

- JevJudge: TypeSafe's Jev as the cheap, high-recall filter. It reads the measured
  facts as text and answers typed questions. The request shape mirrors the
  reference demo (moldandyeast/floorplan); check it against TypeSafe's docs.
- ClaudeJudge: Claude reads the drawing plus the facts and scores a rubric.
  As a filter (role="filter") it asks only "is there a clear reason to discard?".
- SimulatedJudge: a hidden client with a known taste, seen through noise of a chosen
  accuracy r. It costs nothing and lets you test the whole funnel offline, and see
  the paper's claim (quality grows with r) on your own generator. It says nothing
  about how well a real judge agrees with *you*; measure that with `floorgen calibrate`.
"""
from __future__ import annotations

import base64
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .brief import Brief
from .features import facts
from .generate import Plan
from .render import png


@dataclass
class Verdict:
    u: float                 # 0..1, higher = better / keep
    reason: str
    detail: dict = field(default_factory=dict)
    cost: float = 0.0


class Judge:
    name = "judge"

    def judge(self, brief: Brief, plan: Plan, feats: dict, feedback: list[str]) -> Verdict:
        raise NotImplementedError

    def judge_many(self, brief: Brief, items: list[tuple[Plan, dict]], feedback: list[str], workers: int = 8) -> list[Verdict]:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(lambda it: self.judge(brief, it[0], it[1], feedback), items))


# --------------------------------------------------------------------------- simulated

def client_utility(f: dict) -> float:
    """The simulated client's true taste, in [0, 1]. Mirrors the sample brief's taste text."""
    def near(y, g, w):
        return math.exp(-((y - g) ** 2) / (2 * w * w))
    s = (
        2.0 * near(f["share:living"], 0.38, 0.07)
        + 1.0 * near(f["share:hall"], 0.08, 0.04)
        + 1.0 * min(f["bed_from_door"] / 6.0, 1.0)
        + 1.0 * f["daylight_share"]
        + 0.6 * f["open_kitchen"]
        + 0.8 * (1.0 - min(f["off_living"], 2) / 2)
        + 0.8 * near(f["max_aspect"], 1.5, 0.6)
    )
    return s / 7.2


class SimulatedJudge(Judge):
    """u = r * z(true) + sqrt(1 - r²) * noise, squashed to [0, 1]. Deterministic per (plan, judge seed)."""

    def __init__(self, accuracy: float, seed: int = 0, name: str = "simulated"):
        self.r, self.seed, self.name = accuracy, seed, name

    def judge(self, brief, plan, feats, feedback):
        rng = random.Random(f"{self.seed}:{plan.id}")
        z = (client_utility(feats) - 0.55) / 0.12
        u = self.r * z + math.sqrt(max(0.0, 1 - self.r ** 2)) * rng.gauss(0, 1)
        return Verdict(1 / (1 + math.exp(-u)), "simulated", {"true": client_utility(feats)})


# --------------------------------------------------------------------------- TypeSafe Jev

WEAKNESS = {
    "small_living": "The living room is too small",
    "dark": "Too little daylight where the client spends time",
    "exposed": "The bedroom or bathroom door opens off the living room or kitchen",
    "through_room": "To reach a room you must walk through a room that is not the hall or living room",
    "wasted_hall": "Too much floor is lost to the hall",
    "cramped": "Rooms are narrow or awkwardly proportioned",
    "none": "No real weakness",
}


class JevJudge(Judge):
    name = "jev"
    url = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, model: str = "jev-latest", api_key: str | None = None, price_in: float = 0.042):
        self.model, self.price_in = model, price_in
        self.key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self.key:
            raise RuntimeError("set TYPESAFE_API_KEY to use Jev")

    def judge(self, brief, plan, feats, feedback):
        state = {"client_brief": brief.taste, "plan": facts(brief, plan)}
        if feedback:
            state["client_choices_so_far"] = feedback[-8:]
        questions = {
            "discard": {"type": "noul",
                        "instructions": "Is there a clear, serious reason this client would not want to see this `plan` at all?",
                        "criteria": {"true": "A serious problem for this client, something they would reject the plan for.",
                                     "false": "No serious problem; any weaknesses are minor or a matter of trade-off."}},
            "weakness": {"type": "choice", "instructions": "What is the `plan`'s main weakness for this client?",
                         "criteria": WEAKNESS},
        }
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        for attempt in range(5):
            req = urllib.request.Request(self.url, body, {"content-type": "application/json",
                                                          "authorization": f"Bearer {self.key}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as res:
                    j = json.loads(res.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(float(e.headers.get("retry-after") or 0.7 * 2 ** attempt))
                    continue
                raise RuntimeError(f"TypeSafe returned {e.code}: {e.read()[:300]!r}") from e
        a = j.get("answers", {})
        p_discard = float(a.get("discard", {}).get("noul", 0.5))
        wk = a.get("weakness", {}).get("choice")
        tokens = j.get("usage", {}).get("input_tokens", 0)
        return Verdict(1 - p_discard, WEAKNESS.get(wk, "") if wk else "", {"p_discard": p_discard, "weakness": wk},
                       cost=tokens * self.price_in / 1e6)


# --------------------------------------------------------------------------- Claude

CRITERIA = {
    "living": "Is the living room generous and well shaped for how this client lives?",
    "daylight": "Do the rooms where the client spends time get good daylight?",
    "circulation": "Can you move between rooms sensibly, without walking through rooms you should not?",
    "privacy": "Are the bedroom and bathroom quiet and private from the entry and the living room?",
    "proportions": "Are the rooms usefully proportioned (no corridors pretending to be rooms, no wasted floor)?",
}
CONTEXT = ("North is up. Exterior walls are thick black lines; blue dashed lines are walls that can have windows. "
           "The front door is marked ENTRY. Gaps with dots are doors; dashed lines inside mean open plan (no wall). "
           "Every plan already passes the hard rules in code: minimum room sizes, daylight to living room and bedroom, "
           "every room reachable through doors, and the door rules.")

PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-4-5": (1.0, 5.0)}


class ClaudeJudge(Judge):
    def __init__(self, model: str = "claude-sonnet-5-5", role: str = "rank", repeats: int = 1, effort: str = "medium"):
        import anthropic  # optional dependency: pip install 'floorgen[llm]'

        self.client = anthropic.Anthropic()
        self.model, self.role, self.repeats, self.effort = model, role, repeats, effort
        self.name = f"{model}:{role}"

    def _schema(self) -> dict:
        if self.role == "filter":
            props = {"keep": {"type": "number"}, "reason": {"type": "string"}}
        else:
            props = {k: {"type": "integer"} for k in CRITERIA} | {"reason": {"type": "string"}}
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    def _ask(self, brief: Brief, plan: Plan, feedback: list[str], variant: int) -> tuple[dict, float]:
        img = base64.standard_b64encode(png(brief, plan)).decode()
        fb = ("\n\nWhat the client picked and rejected so far (learn their taste from this):\n" + "\n".join(feedback[-12:])) if feedback else ""
        if self.role == "filter":
            system = ("You are the first, fast filter in a review of apartment floor plans for one client. Drop only plans "
                      "with a clear, serious problem for this client; when unsure, keep. A stronger reviewer sees what you keep.")
            ask = ('Return "keep": the probability (0 to 1) the client would want to see this plan, '
                   'and "reason": at most 12 words.')
        else:
            system = "You review apartment floor plans for one client and score them honestly against a rubric."
            ask = ("Score each criterion from 1 (poor) to 5 (excellent) for this client:\n"
                   + "\n".join(f"- {k}: {q}" for k, q in CRITERIA.items())
                   + '\nThen give "reason": one sentence, at most 25 words, on the plan\'s main strength or weakness.')
        # vary the wording order between repeats so averaging cancels some prompt noise (paper, Section 10)
        parts = [f"Client brief: {brief.taste}{fb}", f"Measured facts:\n{facts(brief, plan)}", CONTEXT]
        if variant % 2:
            parts = parts[1:] + parts[:1]
        msg = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": self._schema()}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
                {"type": "text", "text": "\n\n".join(parts) + "\n\n" + ask},
            ]}],
        )
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        cost = (msg.usage.input_tokens * pin + msg.usage.output_tokens * pout) / 1e6
        if msg.stop_reason == "refusal":
            return {}, cost
        text = next((b.text for b in msg.content if b.type == "text"), "{}")
        return json.loads(text), cost

    def judge(self, brief, plan, feats, feedback):
        runs, cost = [], 0.0
        for k in range(self.repeats):
            out, c = self._ask(brief, plan, feedback, k)
            cost += c
            if out:
                runs.append(out)
        if not runs:
            return Verdict(0.5, "no answer (refused)", cost=cost)
        if self.role == "filter":
            u = sum(min(max(float(r["keep"]), 0.0), 1.0) for r in runs) / len(runs)
            return Verdict(u, runs[0]["reason"], {"runs": runs}, cost)
        means = {k: sum(min(max(int(r[k]), 1), 5) for r in runs) / len(runs) for k in CRITERIA}
        u = (sum(means.values()) / len(means) - 1) / 4
        return Verdict(u, runs[0]["reason"], {"criteria": means, "runs": runs}, cost)


def make_judge(spec: str, role: str, seed: int = 0) -> Judge:
    """spec: "sim:0.6" | "jev" | a Claude model id, optionally with ":repeats", e.g. "claude-sonnet-5-5:3"."""
    if spec.startswith("sim"):
        r = float(spec.split(":")[1]) if ":" in spec else (0.4 if role == "filter" else 0.7)
        return SimulatedJudge(r, seed, name=f"sim r={r}")
    if spec == "jev":
        return JevJudge()
    model, _, rep = spec.partition(":")
    effort = "low" if role == "filter" else "medium"
    if model.startswith("claude-haiku"):
        raise ValueError("Haiku 4.5 does not take `effort`; use claude-sonnet-5-5 or add a Haiku code path")
    return ClaudeJudge(model, role=role, repeats=int(rep or 1), effort=effort)
