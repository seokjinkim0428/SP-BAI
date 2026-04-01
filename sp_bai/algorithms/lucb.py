import math

import numpy as np


def u_lil_base(t, delta, eps=0.01):
    t = max(int(t), 1)
    delta = max(float(delta), 1e-15)
    inner = math.log(max(math.log((1.0 + eps) * t + 2.0) / delta, 1e-15))
    return (1.0 + math.sqrt(eps)) * math.sqrt((1.0 + eps) * inner / (2.0 * t))


def c_lil_sigma(t, delta, sigma, eps=0.01):
    return 2.0 * sigma * u_lil_base(t, delta, eps)


def run_lucb(
    num_arms,
    delta,
    sigma_sq_by_arm,
    rng,
    reward_fn,
    max_total_pulls=None,
):
    """
    Generic LUCB with lil-style confidence radii.

    reward_fn is called as reward_fn(arm_index, t, rng).
    """
    counts = np.zeros(num_arms, dtype=int)
    means = np.zeros(num_arms)
    t = 0

    for i in range(num_arms):
        if max_total_pulls is not None and t >= max_total_pulls:
            return int(np.argmax(means)), t
        t += 1
        reward = reward_fn(i, t, rng)
        means[i] = reward
        counts[i] = 1

    total_pulls = num_arms
    delta_per_arm = delta / num_arms

    while True:
        if max_total_pulls is not None and total_pulls >= max_total_pulls:
            return int(np.argmax(means)), total_pulls

        C = np.zeros(num_arms)
        for i in range(num_arms):
            sigma_i = math.sqrt(sigma_sq_by_arm[i])
            C[i] = c_lil_sigma(counts[i], delta_per_arm, sigma_i)

        ucbs = means + C
        lcbs = means - C
        j_t = int(np.argmax(means))
        l_t = -1
        max_other_ucb = -np.inf
        for i in range(num_arms):
            if i == j_t:
                continue
            if ucbs[i] > max_other_ucb:
                max_other_ucb = ucbs[i]
                l_t = i

        if lcbs[j_t] > ucbs[l_t]:
            break

        if max_total_pulls is not None and total_pulls + 1 > max_total_pulls:
            return int(np.argmax(means)), total_pulls
        t += 1
        reward_j = reward_fn(j_t, t, rng)
        means[j_t] = (means[j_t] * counts[j_t] + reward_j) / (counts[j_t] + 1)
        counts[j_t] += 1
        total_pulls += 1

        if max_total_pulls is not None and total_pulls + 1 > max_total_pulls:
            return int(np.argmax(means)), total_pulls
        t += 1
        reward_l = reward_fn(l_t, t, rng)
        means[l_t] = (means[l_t] * counts[l_t] + reward_l) / (counts[l_t] + 1)
        counts[l_t] += 1
        total_pulls += 1

    return j_t, total_pulls

