import itertools

import numpy as np

from .utils import intrinsic_basis_from_differences, project_rows, stable_inv


def policy_to_array(p, A, K):
    vec = np.zeros(K, dtype=float)
    active = sorted(list(A))
    for i in active:
        w = p.get(i, 0.0)
        if (not np.isfinite(w)) or (w < 0):
            w = 0.0
        vec[i] = float(w)
    s = vec.sum()
    if (not np.isfinite(s)) or (s <= 0):
        if len(active) == 1:
            vec[active[0]] = 1.0
        elif active:
            for i in active:
                vec[i] = 1.0 / len(active)
        s = vec.sum()
    return vec / s


def covariance_of_policy(X, p_vec):
    xbar = X.T @ p_vec
    centered = X - xbar
    Sigma = (centered.T * p_vec) @ centered
    return Sigma, xbar


def covariance_of_policy_intrinsic(Xr_A, p_vec):
    xbar_r = Xr_A.T @ p_vec
    centered = Xr_A - xbar_r
    Sigma_r = (centered.T * p_vec) @ centered
    return Sigma_r, xbar_r


def vari_cov_XY(X, A, p, reg=1e-10):
    K = X.shape[0]
    active = sorted(list(A))
    if len(active) <= 1:
        return 0.0
    p_vec = policy_to_array(p, list(range(K)), K)
    Sigma, _ = covariance_of_policy(X, p_vec)
    Minv = stable_inv(Sigma, ridge=reg)
    maxv = 0.0
    for i, j in itertools.combinations(active, 2):
        u = X[i] - X[j]
        v = float(u.T @ Minv @ u)
        if np.isfinite(v) and v > maxv:
            maxv = v
    if (not np.isfinite(maxv)) or maxv < 0:
        return 0.0
    return maxv


def vari_cov_XY_intrinsic(X, A, p, tol=1e-10, reg=1e-10):
    K, _ = X.shape
    active = sorted(list(A))
    if len(active) <= 1:
        return 0.0
    basis, rank, _ = intrinsic_basis_from_differences(X, A, tol=tol)
    if rank == 0:
        return 0.0
    Xr_all = project_rows(X, basis)
    Xr_A = Xr_all[active]
    p_vec = policy_to_array(p, list(range(K)), K)
    xbar_r = Xr_all.T @ p_vec
    centered_all = Xr_all - xbar_r
    Sigma_r = (centered_all.T * p_vec) @ centered_all
    Minv_r = stable_inv(Sigma_r, ridge=reg)
    maxv = 0.0
    for a, b in itertools.combinations(range(len(active)), 2):
        u = Xr_A[a] - Xr_A[b]
        v = float(u.T @ Minv_r @ u)
        if np.isfinite(v) and v > maxv:
            maxv = v
    if (not np.isfinite(maxv)) or maxv < 0:
        return 0.0
    return maxv


def draw_arm_from_policy(rng, p):
    keys = list(p.keys())
    probs = np.array([p[i] for i in keys], dtype=float)
    probs = probs / probs.sum()
    return int(rng.choice(keys, p=probs))

