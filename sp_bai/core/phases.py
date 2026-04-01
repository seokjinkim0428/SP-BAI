import numpy as np

from .metrics import covariance_of_policy, draw_arm_from_policy, policy_to_array
from .utils import log_ridge, stable_inv


def ortho_phase_linear(X, delta, p, n_rounds, rng, reward_fn, t0=1):
    """
    Run a centered regression phase with an environment-specific reward callback.

    reward_fn is called as reward_fn(arm_index, arm_feature, t, rng).
    """
    K, d = X.shape
    p_vec = policy_to_array(p, list(range(K)), K)
    Sigma, xbar = covariance_of_policy(X, p_vec)

    B = np.zeros((d, d))
    b = np.zeros(d)
    count = 0
    for idx in range(n_rounds):
        t = t0 + idx
        a = draw_arm_from_policy(rng, p)
        x = X[a]
        x_tilde = x - xbar
        r = float(reward_fn(a, x, t, rng))
        B += np.outer(x_tilde, x_tilde)
        b += x_tilde * r
        count += 1

    beta = log_ridge(max(count, 2), delta)
    theta_hat = stable_inv(B + beta * np.eye(d)) @ b
    return theta_hat, count


def ols_phase_deterministic_linear(X, schedule, rng, reward_fn, t0=1):
    """
    Run an OLS phase on a deterministic arm schedule.

    reward_fn is called as reward_fn(arm_index, arm_feature, t, rng).
    """
    d = X.shape[1]
    A = np.zeros((d, d))
    b = np.zeros(d)
    for idx, a in enumerate(schedule):
        t = t0 + idx
        x = X[a]
        r = float(reward_fn(a, x, t, rng))
        A += np.outer(x, x)
        b += x * r
    theta_hat = stable_inv(A) @ b
    return theta_hat, len(schedule)

