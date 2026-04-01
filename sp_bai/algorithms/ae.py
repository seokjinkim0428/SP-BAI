import math

import numpy as np

from .lucb import c_lil_sigma


def run_ae(
    num_arms,
    delta,
    sigma_sq_by_arm,
    rng,
    reward_fn,
    max_total_pulls=None,
    return_status=False,
):
    """
    Generic lil-style successive elimination baseline.

    reward_fn is called as reward_fn(arm_index, t, rng).
    """
    active = list(range(num_arms))
    counts = np.zeros(num_arms, dtype=int)
    means = np.zeros(num_arms)
    t = 0

    for i in active:
        if max_total_pulls is not None and t >= max_total_pulls:
            winner = int(np.argmax(means))
            return (winner, t, False) if return_status else (winner, t)
        t += 1
        reward = reward_fn(i, t, rng)
        means[i] = reward
        counts[i] = 1

    total_pulls = num_arms
    delta_per_arm = delta / num_arms

    while len(active) > 1:
        if max_total_pulls is not None and total_pulls >= max_total_pulls:
            winner = int(max(active, key=lambda i: means[i])) if active else int(np.argmax(means))
            return (winner, total_pulls, False) if return_status else (winner, total_pulls)

        for i in active:
            if max_total_pulls is not None and total_pulls >= max_total_pulls:
                winner = int(max(active, key=lambda i: means[i])) if active else int(np.argmax(means))
                return (winner, total_pulls, False) if return_status else (winner, total_pulls)
            t += 1
            reward = reward_fn(i, t, rng)
            means[i] = (means[i] * counts[i] + reward) / (counts[i] + 1)
            counts[i] += 1
            total_pulls += 1

        C = {}
        for i in active:
            sigma_i = math.sqrt(sigma_sq_by_arm[i])
            base_radius = c_lil_sigma(counts[i], delta_per_arm, sigma_i)
            C[i] = 2.0 * base_radius

        a = max(active, key=lambda i: means[i])
        active_new = [i for i in active if (means[a] - C[a]) < (means[i] + C[i])]
        if len(active_new) == 0:
            active_new = [a]
        active = active_new

    winner = int(active[0])
    return (winner, total_pulls, True) if return_status else (winner, total_pulls)
