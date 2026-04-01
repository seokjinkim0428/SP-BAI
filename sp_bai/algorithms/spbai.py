import math

import numpy as np

from sp_bai.core.metrics import vari_cov_XY
from sp_bai.core.utils import safe_log_pos


def run_sp_bai(
    X,
    delta,
    R1,
    R2,
    rng,
    p_G,
    phase_fn,
    xor_policy_fn,
    vari_fn=vari_cov_XY,
    max_phases=10,
    verbose=False,
    sample_cap=None,
):
    """Run the fixed-confidence SP-BAI phase-elimination loop."""
    K, d = X.shape
    A = set(range(K))
    ell = 1
    t_total = 0
    anchor = None

    while len(A) > 1 and ell <= max_phases:
        eps = 2.0 ** (-ell)
        active_count = len(A)
        delta_l = delta / ((active_count ** 2) * ell * (ell + 1))
        if verbose:
            print(f"[SP-BAI] phase {ell}: |A|={len(A)}, eps={eps:.4f}")

        p_xor = xor_policy_fn(X, A, rng, anchor=anchor)
        p = {}
        for i in range(K):
            a = p_xor.get(i, 0.0)
            b = p_G.get(i, 0.0)
            if (not np.isfinite(a)) or (a < 0):
                a = 0.0
            if (not np.isfinite(b)) or (b < 0):
                b = 0.0
            p[i] = 0.5 * a + 0.5 * b

        s = sum(p.values())
        if (not np.isfinite(s)) or s <= 0:
            p = {i: 1.0 / K for i in range(K)}
        else:
            p = {i: v / s for i, v in p.items() if v > 0.0}

        vari = vari_fn(X, A, p)
        if vari <= 0 or not np.isfinite(vari):
            vari = 1e-12

        n_l = math.ceil(
            R1 * vari / (eps ** 2) * safe_log_pos(vari / (eps * delta_l))
            + R2 * 32 * d * (vari ** 0.5) / eps * safe_log_pos(d / delta_l)
        )
        if sample_cap is not None and n_l > sample_cap:
            if verbose:
                print(f"[SP-BAI] n_l ({n_l}) exceeds cap {sample_cap}. Sampling cap and stopping.")
            theta_hat, c = phase_fn(X, delta, p, int(sample_cap), rng, t0=t_total + 1)
            t_total += c
            hat_arm = int(np.argmax(X @ theta_hat))
            return t_total, hat_arm

        theta_hat, c = phase_fn(X, delta, p, int(n_l), rng, t0=t_total + 1)
        t_total += c
        vals = X @ theta_hat
        A_sorted = sorted(list(A))
        i_star = A_sorted[int(np.argmax(vals[A_sorted]))]
        anchor = i_star
        A = set([i for i in A if (vals[i_star] - vals[i]) < eps])
        ell += 1

    hat_arm = list(A)[0]
    return t_total, hat_arm


def run_sp_bai_budgeted(
    X,
    delta,
    R1,
    R2,
    rng,
    p_G,
    phase_fn,
    xor_policy_fn,
    vari_fn=vari_cov_XY,
    max_phases=10,
    verbose=False,
    max_total_pulls=None,
):
    """
    Run a budget-capped SP-BAI loop.

    When a phase would exceed the remaining budget, it is truncated to the
    remaining number of pulls and the elimination step still proceeds.
    """
    K, d = X.shape
    A = set(range(K))
    ell = 1
    t_total = 0
    anchor = None
    current_best = 0

    while len(A) > 1 and ell <= max_phases:
        if max_total_pulls is not None and t_total >= max_total_pulls:
            break

        eps = 2.0 ** (-ell)
        active_count = len(A)
        delta_l = delta / ((active_count ** 2) * ell * (ell + 1))
        if verbose:
            print(f"[SP-BAI] phase {ell}: |A|={len(A)}, eps={eps:.4f}")

        p_xor = xor_policy_fn(X, A, rng, anchor=anchor)
        p = {}
        for i in range(K):
            a = p_xor.get(i, 0.0)
            b = p_G.get(i, 0.0)
            if (not np.isfinite(a)) or (a < 0):
                a = 0.0
            if (not np.isfinite(b)) or (b < 0):
                b = 0.0
            p[i] = 0.5 * a + 0.5 * b

        s = sum(p.values())
        if (not np.isfinite(s)) or s <= 0:
            p = {i: 1.0 / K for i in range(K)}
        else:
            p = {i: v / s for i, v in p.items() if v > 0.0}

        vari = vari_fn(X, A, p)
        if vari <= 0 or not np.isfinite(vari):
            vari = 1e-12

        n_l = math.ceil(
            R1 * vari / (eps ** 2) * safe_log_pos(vari / (eps * delta_l))
            + R2 * 32 * d * (vari ** 0.5) / eps * safe_log_pos(d / delta_l)
        )
        if max_total_pulls is not None:
            remaining = max_total_pulls - t_total
            if remaining <= 0:
                break
            n_l = min(n_l, remaining)

        theta_hat, c = phase_fn(X, delta, p, int(n_l), rng, t0=t_total + 1)
        t_total += c
        vals = X @ theta_hat
        A_sorted = sorted(list(A))
        i_star = A_sorted[int(np.argmax(vals[A_sorted]))]
        current_best = i_star
        anchor = i_star
        A = set([i for i in A if (vals[i_star] - vals[i]) < eps])
        ell += 1

    if len(A) == 1:
        return t_total, list(A)[0]
    return t_total, int(current_best)


def run_spbai(*args, **kwargs):
    """Backward-compatible alias for :func:`run_sp_bai`."""
    return run_sp_bai(*args, **kwargs)


def run_spbai_budgeted(*args, **kwargs):
    """Backward-compatible alias for :func:`run_sp_bai_budgeted`."""
    return run_sp_bai_budgeted(*args, **kwargs)
