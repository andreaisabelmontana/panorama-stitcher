"""RANSAC robust homography estimation from scratch.

Feature matches are noisy: a sizable fraction are flat-out wrong (outliers).
A plain least-squares fit over all matches would be dragged off by those
outliers. RANSAC (RANdom SAmple Consensus) fits the model robustly:

    repeat for a number of iterations:
        1. sample a minimal set (4 correspondences for a homography) at random
        2. fit a candidate homography to that minimal set (DLT)
        3. count inliers: matches whose reprojection error is below `threshold`
        4. keep the candidate with the most inliers

    finally: refit the homography on ALL inliers of the best model.

The number of iterations needed to see at least one outlier-free sample with
probability `success_prob`, given inlier ratio w and sample size s = 4, is

    N = log(1 - success_prob) / log(1 - w^s)

We adapt N downward as better models (higher inlier ratios) are found.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .homography import estimate_homography, reprojection_error


class RansacResult(NamedTuple):
    H: np.ndarray             # refined 3x3 homography
    inlier_mask: np.ndarray   # bool array (n,), True where the match is an inlier
    n_inliers: int
    iterations: int           # iterations actually run


def _adaptive_iterations(inlier_ratio: float, sample_size: int, success_prob: float) -> int:
    if inlier_ratio <= 0.0:
        return 1 << 30  # effectively "keep going"
    denom = 1.0 - inlier_ratio ** sample_size
    if denom <= 0.0:
        return 1  # all points are inliers
    num = np.log(1.0 - success_prob)
    return int(np.ceil(num / np.log(denom)))


def ransac_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float = 3.0,
    max_iters: int = 2000,
    min_iters: int = 50,
    success_prob: float = 0.999,
    rng: np.random.Generator | None = None,
) -> RansacResult:
    """Robustly estimate the homography src_pts -> dst_pts with RANSAC.

    Parameters
    ----------
    src_pts, dst_pts : (n, 2) arrays of candidate correspondences (n >= 4).
    threshold : inlier reprojection-error threshold, in pixels.
    max_iters / min_iters : iteration bounds; the adaptive count is clamped here.
    success_prob : desired probability of drawing one clean minimal sample.
    rng : optional NumPy Generator for reproducibility.

    Returns
    -------
    RansacResult(H, inlier_mask, n_inliers, iterations)
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = src.shape[0]
    if n < 4:
        raise ValueError("at least 4 correspondences are required for RANSAC")
    if rng is None:
        rng = np.random.default_rng()

    sample_size = 4
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_count = 0

    iters_needed = max_iters
    i = 0
    while i < min(max_iters, max(min_iters, iters_needed)):
        i += 1
        # 1. random minimal sample of 4 distinct correspondences.
        idx = rng.choice(n, size=sample_size, replace=False)
        sample_src = src[idx]
        sample_dst = dst[idx]
        # Skip degenerate samples (collinear points break the DLT).
        if _is_degenerate(sample_src) or _is_degenerate(sample_dst):
            continue
        # 2. fit candidate homography.
        try:
            H = estimate_homography(sample_src, sample_dst)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(H)):
            continue
        # 3. count inliers over all matches.
        errs = reprojection_error(H, src, dst)
        inlier_mask = errs < threshold
        count = int(inlier_mask.sum())
        # 4. keep the best.
        if count > best_count:
            best_count = count
            best_inlier_mask = inlier_mask
            inlier_ratio = count / n
            iters_needed = _adaptive_iterations(inlier_ratio, sample_size, success_prob)

    if best_count < 4:
        # No consensus found; fall back to a fit over everything.
        H = estimate_homography(src, dst)
        mask = reprojection_error(H, src, dst) < threshold
        return RansacResult(H, mask, int(mask.sum()), i)

    # Final refit on ALL inliers of the best model.
    inlier_src = src[best_inlier_mask]
    inlier_dst = dst[best_inlier_mask]
    H_refined = estimate_homography(inlier_src, inlier_dst)
    # Recompute the inlier set with the refined model.
    final_mask = reprojection_error(H_refined, src, dst) < threshold
    if int(final_mask.sum()) < 4:
        final_mask = best_inlier_mask
    return RansacResult(H_refined, final_mask, int(final_mask.sum()), i)


def _is_degenerate(pts: np.ndarray, tol: float = 1e-6) -> bool:
    """True if the 4 points are (nearly) collinear or contain duplicates."""
    # Duplicate points.
    if len({tuple(np.round(p, 6)) for p in pts}) < len(pts):
        return True
    # Check no 3 of the points are collinear via the triangle-area test.
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            for c in range(b + 1, len(pts)):
                area = abs(
                    (pts[b, 0] - pts[a, 0]) * (pts[c, 1] - pts[a, 1])
                    - (pts[c, 0] - pts[a, 0]) * (pts[b, 1] - pts[a, 1])
                )
                if area < tol:
                    return True
    return False
