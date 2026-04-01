import math

import numpy as np


def r_epsilon_pukelsheim(d, eps):
    """Minimum sample threshold used with the Pukelsheim-style rounding."""
    return int(math.ceil((((d * (d + 1)) / 2.0) + 1.0) / eps))


def round_design_counts_eps_efficient(lam, N, K, max_iters=10_000):
    """
    Pukelsheim-style rounding with exact-count correction.

    1. Set N_i = ceil((N - s/2) * lambda_i) on the support.
    2. Move +/-1 counts until the total equals N.
    """
    w = np.array([lam.get(i, 0.0) for i in range(K)], dtype=float)
    w = np.maximum(w, 0)
    s = w.sum()
    if s <= 0:
        n = np.zeros(K, dtype=int)
        n[0] = N
        return {i: int(n[i]) for i in range(K)}
    w = w / s
    support = np.where(w > 0)[0]
    if support.size == 0:
        n = np.zeros(K, dtype=int)
        n[0] = N
        return {i: int(n[i]) for i in range(K)}

    ws = w[support]
    raw = (N - 0.5 * support.size) * ws
    ns = np.ceil(raw).astype(int)
    ns = np.maximum(ns, 1)
    total = int(ns.sum())

    it = 0
    while total > N and it < max_iters:
        valid = ns > 1
        if not np.any(valid):
            break
        ratios = (ns - 1) / ws
        ratios[~valid] = -np.inf
        j = int(np.argmax(ratios))
        ns[j] -= 1
        total -= 1
        it += 1

    while total < N and it < max_iters:
        ratios = ns / ws
        j = int(np.argmin(ratios))
        ns[j] += 1
        total += 1
        it += 1

    n = np.zeros(K, dtype=int)
    n[support] = ns
    return {i: int(n[i]) for i in range(K)}


def deterministic_schedule_from_counts(counts):
    counts = counts.copy()
    arms = sorted(counts.keys())
    seq = []
    more = True
    while more:
        more = False
        for a in arms:
            if counts[a] > 0:
                seq.append(a)
                counts[a] -= 1
                more = True
    return seq

