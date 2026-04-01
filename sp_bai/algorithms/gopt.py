import math

import numpy as np


def run_g_opt(
    X,
    delta,
    R1,
    R2,
    rng,
    theta_star,
    phase_fn,
    policy_fn,
    verbose=False,
    sample_cap=None,
):
    """Run the one-shot G-Opt baseline."""
    _, d = X.shape
    vals_true = X @ theta_star
    order = np.argsort(vals_true)[::-1]
    a1, a2 = int(order[0]), int(order[1])
    Delta2 = float(vals_true[a1] - vals_true[a2])
    T = math.ceil(
        4 * R1 * (4 * d) / (Delta2 ** 2) * math.log((4 * d) / (Delta2 * delta))
        + 2 * R2 * (32 * (d ** 1.5)) / Delta2 * math.log(d / delta)
    )
    if sample_cap is not None and T > sample_cap:
        if verbose:
            print(f"[G-Opt] T ({T}) exceeds cap {sample_cap}. Using cap.")
        T = sample_cap
    if verbose:
        print(f"[G-Opt] T={T}, Δ2={Delta2:.4f}")

    p = policy_fn(X, set(range(X.shape[0])), rng)
    theta_hat, c = phase_fn(X, delta, p, int(T), rng, t0=1)
    hat_arm = int(np.argmax(X @ theta_hat))
    return c, hat_arm


def run_deo_one_shot(*args, **kwargs):
    """Backward-compatible alias for :func:`run_g_opt`."""
    return run_g_opt(*args, **kwargs)
