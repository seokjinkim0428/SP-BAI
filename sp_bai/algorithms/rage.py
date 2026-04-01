import math

import numpy as np

from sp_bai.core.rounding import (
    deterministic_schedule_from_counts,
    r_epsilon_pukelsheim,
    round_design_counts_eps_efficient,
)


def run_rage(
    X,
    delta,
    rng,
    ols_phase_fn,
    xy_design_fn,
    epsilon_round=0.2,
    max_phases=10,
    c_rage=1.0,
    sample_scale=1.0,
    min_samples_fn=r_epsilon_pukelsheim,
    round_counts_fn=round_design_counts_eps_efficient,
    schedule_fn=deterministic_schedule_from_counts,
    verbose=False,
    sample_cap=None,
):
    """Shared RAGE elimination loop."""
    K, d = X.shape
    A = set(range(K))
    t = 1
    t_total = 0

    if verbose:
        print(f"[RAGE-OLS] start |A|={len(A)}, delta={delta}, eps-round={epsilon_round}")

    while len(A) > 1 and t <= max_phases:
        delta_t = delta / (t * t)
        p_t, rho_t = xy_design_fn(X, A)
        Nt_base = c_rage * sample_scale * int(
            math.ceil(
                8.0
                * (2 ** (t + 1)) ** 2
                * (1.0 + epsilon_round)
                * rho_t
                * math.log((K * K) / max(delta_t, 1e-12))
            )
        )
        Nt = max(Nt_base, min_samples_fn(d, epsilon_round))
        if sample_cap is not None and Nt > sample_cap:
            if verbose:
                print(f"[RAGE-OLS] N_t ({Nt}) exceeds cap {sample_cap}. Using cap.")
            Nt = sample_cap
        if verbose:
            print(
                f"[RAGE-OLS] phase {t}: |A|={len(A)}, rho={rho_t:.4f}, "
                f"N_t_base={Nt_base}, r(eps)={min_samples_fn(d, epsilon_round)}, N_t={Nt}"
            )

        counts = round_counts_fn(p_t, Nt, K)
        schedule = schedule_fn(counts)
        theta_hat, c = ols_phase_fn(X, schedule, rng, t0=t_total + 1)
        t_total += c

        vals = X @ theta_hat
        A_sorted = sorted(list(A))
        i_star = A_sorted[int(np.argmax(vals[A_sorted]))]
        eps_th = 2.0 ** (-(t + 2))
        A = set([i for i in A if (vals[i_star] - vals[i]) < eps_th])
        t += 1

    hat_arm = list(A)[0]
    return t_total, hat_arm

