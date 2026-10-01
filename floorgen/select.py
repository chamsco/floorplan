"""Stage 6: pick a short, varied list (paper, Section 11).

Greedy maximisation of F(S) = psi * mean score of S + (1 - psi) * coverage, where
coverage is how well every candidate is represented by its most similar pick.
F is monotone submodular, so greedy reaches at least 63% of the best set.
A few slots are kept for wildcards: the candidates least like anything picked.
"""
from __future__ import annotations

import numpy as np


def similarity(Z: np.ndarray) -> np.ndarray:
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    d = np.sqrt(d2)
    iu = np.triu_indices(len(Z), 1)
    delta = np.percentile(d[iu], 10) if len(iu[0]) else 1.0
    sigma = max(0.85 * delta, 1e-6)
    return np.exp(-d2 / (2 * sigma * sigma))


def shortlist(scores: np.ndarray, Z: np.ndarray, k: int = 12, wildcards: int = 2, psi: float = 0.6):
    """Returns (picked indices, wildcard flags, group size per pick)."""
    M = len(scores)
    if M == 0:
        return [], [], []
    k = min(k, M)
    S = similarity(Z)
    chosen: list[int] = []
    best_sim = np.zeros(M)
    for _ in range(k - min(wildcards, k - 1)):
        q = psi * scores / k
        cov = (1 - psi) * (np.maximum(best_sim[None, :], S).mean(axis=1) - best_sim.mean())
        gain = q + cov
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        best_sim = np.maximum(best_sim, S[j])
    wild = []
    while len(chosen) < k:
        rest = [i for i in range(M) if i not in chosen]
        j = min(rest, key=lambda i: S[i, chosen].max())
        chosen.append(j)
        wild.append(j)
    owner = np.argmax(S[:, chosen], axis=1)
    groups = [int((owner == n).sum()) for n in range(len(chosen))]
    return chosen, [c in wild for c in chosen], groups
