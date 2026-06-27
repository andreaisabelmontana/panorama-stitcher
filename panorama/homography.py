"""Homography estimation from scratch with NumPy.

A homography H is a 3x3 matrix that maps points in one image plane to another
(up to scale): for a source point x = (x, y, 1)^T and its match x' = (x', y', 1)^T,

    x' ~ H x        (~ means equal up to a non-zero scale factor)

We solve for H with the Direct Linear Transform (DLT). Writing H row-wise as
h = [h11 h12 h13 h21 h22 h23 h31 h32 h33]^T, the relation x' ~ H x can be turned
into two linear equations per correspondence by taking the cross product
x' x (H x) = 0:

    [ -x  -y  -1   0   0   0   x'x  x'y  x' ] h = 0
    [  0   0   0  -x  -y  -1   y'x  y'y  y' ] h = 0

Stacking these for n >= 4 correspondences gives A h = 0, a homogeneous system.
The least-squares solution (subject to ||h|| = 1) is the right singular vector of
A with the smallest singular value, i.e. the last column of V in A = U S V^T.

Raw pixel coordinates make A badly conditioned, so we first apply Hartley's
normalization: translate each point set to have zero mean and scale it so the
mean distance from the origin is sqrt(2). We solve for the normalized homography
and then denormalize.
"""

from __future__ import annotations

import numpy as np


def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    """Return a 3x3 similarity transform T that maps `pts` to zero mean and
    mean distance sqrt(2) from the origin (Hartley normalization)."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    # Guard against a degenerate (single-point / coincident) set.
    if mean_dist < 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(2.0) / mean_dist
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return T


def estimate_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Estimate the homography H mapping src_pts -> dst_pts via the normalized DLT.

    Parameters
    ----------
    src_pts, dst_pts : array_like, shape (n, 2), n >= 4
        Matched 2-D point coordinates.

    Returns
    -------
    H : ndarray, shape (3, 3)
        Homography normalized so that H[2, 2] == 1 when possible.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src_pts and dst_pts must both be (n, 2) arrays")
    n = src.shape[0]
    if n < 4:
        raise ValueError("at least 4 correspondences are required")

    # Hartley normalization.
    T_src = _normalization_matrix(src)
    T_dst = _normalization_matrix(dst)

    src_h = np.hstack([src, np.ones((n, 1))])
    dst_h = np.hstack([dst, np.ones((n, 1))])
    src_n = (T_src @ src_h.T).T
    dst_n = (T_dst @ dst_h.T).T

    # Build the 2n x 9 DLT matrix A.
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = src_n[i, 0], src_n[i, 1]
        xp, yp = dst_n[i, 0], dst_n[i, 1]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp]

    # h is the right singular vector with the smallest singular value.
    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)

    # Denormalize: H = T_dst^{-1} H_n T_src.
    H = np.linalg.inv(T_dst) @ H_n @ T_src

    # Normalize so the bottom-right entry is 1 (when not ~0).
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography H to an (n, 2) array of points, returning (n, 2)."""
    pts = np.asarray(pts, dtype=np.float64)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    mapped = (H @ pts_h.T).T
    w = mapped[:, 2:3]
    # Avoid division by zero for points at infinity.
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return mapped[:, :2] / w


def reprojection_error(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Per-correspondence Euclidean reprojection error ||H src - dst|| (n,)."""
    projected = apply_homography(H, src_pts)
    dst = np.asarray(dst_pts, dtype=np.float64)
    return np.sqrt(((projected - dst) ** 2).sum(axis=1))
