import math

import numpy as np
from numpy.linalg import inv


def stable_inv(matrix, ridge=1e-10):
    """Invert a PSD matrix with a small ridge fallback."""
    dim = matrix.shape[0]
    try:
        return inv(matrix)
    except np.linalg.LinAlgError:
        return inv(matrix + ridge * np.eye(dim))


def proj_simplex(y):
    """Project a vector onto the probability simplex."""
    y = np.asarray(y, dtype=float).copy()
    n = y.size
    if n == 0:
        return y
    if (not np.isfinite(y).all()) or np.all(y <= 0):
        return np.ones(n) / n
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    k = np.arange(1, n + 1)
    cond = u > (cssv - 1) / k
    idx = np.nonzero(cond)[0]
    if idx.size == 0:
        theta = (cssv[-1] - 1) / n
    else:
        r = idx[-1]
        theta = (cssv[r] - 1) / (r + 1)
    w = np.maximum(y - theta, 0.0)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones(n) / n
    return w / s


def log_ridge(n, delta):
    return max(1.0, math.log(max(n, 2) / max(delta, 1e-12)))


def safe_log_pos(x, eps=1e-18):
    if (not np.isfinite(x)) or (x <= 0):
        x = eps
    return math.log(x)


def fmt_sec(x):
    if x < 1:
        return f"{x * 1e3:.1f} ms"
    if x < 60:
        return f"{x:.2f} s"
    minutes = int(x // 60)
    seconds = x - 60 * minutes
    return f"{minutes}m {seconds:.1f}s"


def compress_to_span(X, tol=1e-10):
    """Compress rows of X into an intrinsic basis for their span."""
    _, svals, vt = np.linalg.svd(X, full_matrices=False)
    rank = int(np.sum(svals > tol))
    if rank == 0:
        basis = np.zeros((X.shape[1], 1))
        basis[0, 0] = 1.0
        return X @ basis, basis, 1
    basis = vt[:rank].T
    return X @ basis, basis, rank


def intrinsic_basis_from_differences(X, active_idx, tol=1e-10):
    """Return an orthonormal basis for the pairwise-difference span of active arms."""
    active = sorted(list(active_idx))
    if not active:
        basis = np.zeros((X.shape[1], 1))
        basis[0, 0] = 1.0
        return basis, 0, 0
    anchor = active[0]
    diffs = X[active] - X[anchor]
    nonzero = np.linalg.norm(diffs, axis=1) > tol
    effective = diffs[nonzero]
    if effective.shape[0] == 0:
        basis = np.zeros((X.shape[1], 1))
        basis[0, 0] = 1.0
        return basis, 0, anchor
    _, svals, vt = np.linalg.svd(effective, full_matrices=False)
    rank = int(np.sum(svals > tol))
    if rank == 0:
        basis = np.zeros((X.shape[1], 1))
        basis[0, 0] = 1.0
        return basis, 0, anchor
    basis = vt[:rank].T
    return basis, rank, anchor


def project_rows(X, basis):
    return X @ basis

