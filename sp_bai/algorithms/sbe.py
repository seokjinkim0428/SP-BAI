import math

import numpy as np


def run_sbe(
    X,
    delta,
    R1,
    R2,
    rng,
    phase_fn,
    policy_fn,
    max_phases=10,
    verbose=False,
    sample_cap=None,
):
    """Shared SBE elimination loop."""
    K, d = X.shape
    A = set(range(K))
    ell = 1
    t_total = 0

    while len(A) > 1 and ell <= max_phases:
        eps = 2.0 ** (-ell)
        delta_l = delta / (K * ell * (ell + 1))
        n_l = math.ceil(
            4 * R1 * (4 * d) / (eps ** 2) * math.log((4 * d) / (eps * delta_l))
            + 2 * R2 * (32 * (d ** 1.5)) / eps * math.log(d / delta_l)
        )
        if verbose:
            print(f"[SBE] phase {ell}: |A|={len(A)}, eps={eps:.4f}, n_l={n_l}")

        p = policy_fn(X, A, rng)
        if sample_cap is not None and n_l > sample_cap:
            if verbose:
                print(f"[SBE] n_l ({n_l}) exceeds cap {sample_cap}. Sampling cap and stopping.")
            theta_hat, c = phase_fn(X, delta, p, int(sample_cap), rng, t0=t_total + 1)
            t_total += c
            hat_arm = int(np.argmax(X @ theta_hat))
            return t_total, hat_arm

        theta_hat, c = phase_fn(X, delta, p, int(n_l), rng, t0=t_total + 1)
        t_total += c
        vals = X @ theta_hat
        A_sorted = sorted(list(A))
        i_star = A_sorted[int(np.argmax(vals[A_sorted]))]
        A = set([i for i in A if (vals[i_star] - vals[i]) < eps])
        ell += 1

    hat_arm = list(A)[0]
    return t_total, hat_arm

