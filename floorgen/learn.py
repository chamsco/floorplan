"""The learning loop (paper, Section 12): move the generator's knobs toward what wins.

Each plan is drawn with its own theta. After a round, the thetas of the plans that
reached the shortlist (and, weighted higher, the ones the person picked) are the
elite; the sampler takes a smoothed step toward them. Two guards against drifting
into what the judges like rather than what you like:

- exploration floor: with probability epsilon, theta comes from the original prior.
  Those plans are also the control group for measuring whether learning helps.
- KL leash: the step is shrunk until the sampler stays within kappa of the prior.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from .brief import Brief

PRIOR_MU, PRIOR_SIGMA, MIN_SIGMA = 0.5, 0.3, 0.05


@dataclass
class Sampler:
    mu: dict[str, float]
    sigma: dict[str, float]
    p_include: dict[str, float]
    epsilon: float = 0.15
    history: list = field(default_factory=list)

    @staticmethod
    def prior(brief: Brief) -> "Sampler":
        keys = [f"share:{t}" for t in brief.rooms] + ["open", "spread"]
        return Sampler({k: PRIOR_MU for k in keys}, {k: PRIOR_SIGMA for k in keys},
                       {t: 0.3 for t, s in brief.rooms.items() if s.optional})

    def draw(self, rng: random.Random) -> dict:
        explore = rng.random() < self.epsilon
        src = Sampler.prior_like(self) if explore else self
        cont = {}
        for k in src.mu:
            for _ in range(8):
                x = rng.gauss(src.mu[k], src.sigma[k])
                if 0 <= x <= 1:
                    break
            cont[k] = min(max(x, 0.0), 1.0)
        return {
            "share": {k.split(":", 1)[1]: v for k, v in cont.items() if k.startswith("share:")},
            "open": cont["open"],
            "spread": cont["spread"],
            "include": {t: rng.random() < p for t, p in src.p_include.items()},
            "explore": explore,
        }

    @staticmethod
    def prior_like(s: "Sampler") -> "Sampler":
        return Sampler({k: PRIOR_MU for k in s.mu}, {k: PRIOR_SIGMA for k in s.mu}, {t: 0.3 for t in s.p_include}, 0)

    def kl_to_prior(self) -> float:
        kl = 0.0
        for k in self.mu:
            s1, s0 = self.sigma[k], PRIOR_SIGMA
            kl += math.log(s0 / s1) + (s1 ** 2 + (self.mu[k] - PRIOR_MU) ** 2) / (2 * s0 ** 2) - 0.5
        for t, p in self.p_include.items():
            p = min(max(p, 1e-6), 1 - 1e-6)
            kl += p * math.log(p / 0.3) + (1 - p) * math.log((1 - p) / 0.7)
        return kl

    def update(self, elite: list[tuple[dict, float]], eta: float = 0.2, kappa: float = 3.0) -> float:
        """elite: (theta, weight) pairs. Returns the step size actually used."""
        if not elite:
            return 0.0
        W = sum(w for _, w in elite)

        def flat(th):
            d = {f"share:{t}": v for t, v in th["share"].items()}
            d["open"], d["spread"] = th["open"], th["spread"]
            return d

        flats = [(flat(t), w) for t, w in elite]
        old = (dict(self.mu), dict(self.sigma), dict(self.p_include))
        step = eta
        for _ in range(6):
            self.mu, self.sigma, self.p_include = dict(old[0]), dict(old[1]), dict(old[2])
            for k in self.mu:
                vals = [(f[k], w) for f, w in flats if k in f]
                if not vals:
                    continue
                m = sum(v * w for v, w in vals) / sum(w for _, w in vals)
                var = sum(w * (v - m) ** 2 for v, w in vals) / sum(w for _, w in vals)
                self.mu[k] = (1 - step) * self.mu[k] + step * m
                self.sigma[k] = max(math.sqrt((1 - step) * self.sigma[k] ** 2 + step * var), MIN_SIGMA)
            for t in self.p_include:
                share = sum(w for th, w in elite if th["include"].get(t)) / W
                self.p_include[t] = (1 - step) * self.p_include[t] + step * share
            if self.kl_to_prior() <= kappa:
                break
            step /= 2
        self.history.append({"step": step, "kl": self.kl_to_prior(), "n_elite": len(elite)})
        return step

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))

    @staticmethod
    def load(path: Path) -> "Sampler":
        return Sampler(**json.loads(path.read_text()))
