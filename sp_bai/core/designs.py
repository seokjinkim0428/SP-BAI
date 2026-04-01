import itertools

import numpy as np
from scipy.optimize import minimize

from .utils import compress_to_span, proj_simplex, stable_inv


def leverage_scores_core(Xmat, p, reg=1e-6):
    p = np.asarray(p, dtype=float)
    if (p < 0).any() or not np.isfinite(p).all():
        p = np.clip(p, 0, None)
        s = p.sum()
        p = (p / s) if s > 0 else np.ones_like(p) / p.size
    P = np.diag(p)
    M = Xmat.T @ P @ Xmat + reg * np.eye(Xmat.shape[1])
    Minv = stable_inv(M, ridge=reg)
    return np.einsum("nd,dd,nd->n", Xmat, Minv, Xmat)


def g_optimal_design_slsqp_lowrank(Xmat, maxiter=500, reg=1e-6, verbose=False):
    K, _ = Xmat.shape

    def objective(p):
        return float(np.max(leverage_scores_core(Xmat, p, reg=reg)))

    constraints = (
        {"type": "eq", "fun": lambda p: np.sum(p) - 1.0},
        {"type": "ineq", "fun": lambda p: p},
    )
    bounds = [(0.0, 1.0)] * K
    p0 = np.ones(K) / K
    result = minimize(
        objective,
        p0,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-9, "disp": verbose},
    )
    if result.success and np.isfinite(result.fun):
        return proj_simplex(result.x)
    raise RuntimeError("SLSQP failed")


def g_optimal_design_kw_lowrank(Xmat, iters=1000, tol=1e-4, reg=1e-6):
    K, _ = Xmat.shape
    p = np.ones(K) / K
    last = None
    for t in range(1, iters + 1):
        scores = leverage_scores_core(Xmat, p, reg=reg)
        i_star = int(np.argmax(scores))
        gamma = 1.0 / (t + 1.0)
        e = np.zeros_like(p)
        e[i_star] = 1.0
        p = (1 - gamma) * p + gamma * e
        p = proj_simplex(p)
        if t % 25 == 0:
            cur = float(np.max(scores))
            if last is not None and abs(cur - last) < tol:
                break
            last = cur
    return p


def g_optimal_design_with_lowrank(
    X, prefer_slsqp=True, use_lowrank=True, lowrank_tol=1e-10
):
    if use_lowrank:
        Xr, _, _ = compress_to_span(X, tol=lowrank_tol)
    else:
        Xr = X
    if prefer_slsqp:
        try:
            return g_optimal_design_slsqp_lowrank(Xr)
        except Exception:
            pass
    return g_optimal_design_kw_lowrank(Xr)


def deo_policy_global(X, prefer_slsqp=True, use_lowrank=True, lowrank_tol=1e-10):
    K = X.shape[0]
    active = list(range(K))
    if len(active) == 1:
        return {active[0]: 1.0}
    anchor = active[0]
    others = active[1:]
    B = X[others] - X[anchor]
    p_tilde = g_optimal_design_with_lowrank(
        B, prefer_slsqp=prefer_slsqp, use_lowrank=use_lowrank, lowrank_tol=lowrank_tol
    )
    p = {anchor: 0.5}
    for idx, w in zip(others, p_tilde):
        p[idx] = 0.5 * w
    s = sum(p.values())
    if (not np.isfinite(s)) or s <= 0:
        return {i: 1.0 / K for i in active}
    return {k: v / s for k, v in p.items()}


def deo_policy(
    X,
    active_idx,
    rng=None,
    prefer_slsqp=True,
    use_lowrank=True,
    lowrank_tol=1e-10,
    random_anchor=False,
):
    active = sorted(list(active_idx))
    if len(active) == 1:
        return {active[0]: 1.0}
    if random_anchor:
        if rng is None:
            raise ValueError("rng is required when random_anchor=True")
        anchor = int(rng.choice(active))
        others = [i for i in active if i != anchor]
    else:
        anchor = active[0]
        others = active[1:]
    B = X[others] - X[anchor]
    p_tilde = g_optimal_design_with_lowrank(
        B, prefer_slsqp=prefer_slsqp, use_lowrank=use_lowrank, lowrank_tol=lowrank_tol
    )
    p = {anchor: 0.5}
    for idx, w in zip(others, p_tilde):
        p[idx] = 0.5 * w
    s = sum(p.values())
    if (not np.isfinite(s)) or s <= 0:
        return {i: 1.0 / len(active) for i in active}
    return {k: v / s for k, v in p.items()}


def xy_allocation_linear_lowrank(
    X,
    active_idx,
    rng,
    iters=800,
    step=0.2,
    tol=1e-4,
    anchor=None,
    use_lowrank=True,
    lowrank_tol=1e-10,
):
    active = sorted(list(active_idx))
    if anchor is None:
        anchor = int(rng.choice(active))
    else:
        anchor = int(anchor)
        if anchor not in active_idx:
            anchor = int(rng.choice(active))
    others = [i for i in active if i != anchor]
    if not others:
        return {anchor: 1.0}, anchor, 0.0

    B = X[others] - X[anchor]
    if use_lowrank:
        Br, basis, _ = compress_to_span(B, tol=lowrank_tol)
    else:
        Br = B
        basis = None

    q = g_optimal_design_with_lowrank(
        B, prefer_slsqp=True, use_lowrank=use_lowrank, lowrank_tol=lowrank_tol
    )
    q = q / q.sum()

    pairs = list(itertools.combinations(active, 2))
    U = np.stack([X[i] - X[j] for i, j in pairs], axis=0)
    Ur = U @ basis if use_lowrank else U

    def M_of(qv):
        return (Br.T * qv) @ Br

    def worst_pair(qv):
        M = M_of(qv)
        Minv = stable_inv(M)
        vals = np.einsum("pd,dd,pd->p", Ur, Minv, Ur)
        k = int(np.argmax(vals))
        return k, float(vals[k]), Minv

    last_vm = None
    vmax = 0.0
    for t in range(iters):
        k, vmax, Minv = worst_pair(q)
        u = Ur[k]
        Bu = Br @ (Minv @ u)
        grad = -(Bu ** 2)
        q = proj_simplex(q - step * grad)
        if t % 25 == 0:
            if last_vm is not None and abs(vmax - last_vm) < tol:
                break
            last_vm = vmax

    q = q / q.sum()
    return {others[i]: q[i] for i in range(len(others))}, anchor, vmax


def xor_policy(
    X,
    active_idx,
    rng,
    anchor=None,
    use_lowrank=True,
    lowrank_tol=1e-10,
):
    q, anchor, _ = xy_allocation_linear_lowrank(
        X,
        active_idx,
        rng,
        anchor=anchor,
        use_lowrank=use_lowrank,
        lowrank_tol=lowrank_tol,
    )
    p = {anchor: 0.5}
    for i, w in q.items():
        p[i] = 0.5 * w
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}


def xy_design_optimal_lowrank(
    X, active_idx, iters=500, use_lowrank=True, lowrank_tol=1e-10
):
    K, _ = X.shape
    active = sorted(list(active_idx))
    if use_lowrank:
        Xr, _, rank = compress_to_span(X, tol=lowrank_tol)
        Ar = [Xr[i] for i in active]
    else:
        Xr = X
        rank = X.shape[1]
        Ar = [X[i] for i in active]
    pairs = list(itertools.combinations(range(len(active)), 2))
    Yr = np.stack([Ar[i] - Ar[j] for (i, j) in pairs], axis=0)
    lam = np.ones(K) / K

    def Minv_of(lv):
        Mr = (Xr.T * lv) @ Xr
        return stable_inv(Mr + 1e-12 * np.eye(rank))

    def worst_y_and_grad(lv):
        Minv = Minv_of(lv)
        vals = np.einsum("pr,rr,pr->p", Yr, Minv, Yr)
        k = int(np.argmax(vals))
        y = Yr[k]
        XiMinvy = Xr @ (Minv @ y)
        g = -(XiMinvy ** 2)
        return float(vals[k]), g

    last = None
    for t in range(iters):
        rho, g = worst_y_and_grad(lam)
        i_fw = int(np.argmin(g))
        s = np.zeros(K)
        s[i_fw] = 1.0
        gamma = 2.0 / (t + 2.0)
        lam = (1 - gamma) * lam + gamma * s
        if t % 25 == 0:
            if last is not None and abs(rho - last) < 1e-4:
                break
            last = rho

    rho, _ = worst_y_and_grad(lam)
    return {i: lam[i] for i in range(K)}, rho

