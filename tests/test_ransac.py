"""Tests for the from-scratch RANSAC homography estimator.

This is the proof the robust-fitting math is correct: synthesize correspondences
under a KNOWN homography, corrupt a chunk of them with gross outliers, and assert
RANSAC recovers the true homography AND flags the outliers.
"""

import numpy as np
import pytest

from panorama.homography import apply_homography, reprojection_error
from panorama.ransac import ransac_homography


def random_homography(rng):
    H = np.eye(3)
    H[:2, :2] += rng.uniform(-0.15, 0.15, (2, 2))
    H[:2, 2] = rng.uniform(-40, 40, 2)
    H[2, :2] = rng.uniform(-1e-4, 1e-4, 2)
    return H / H[2, 2]


def make_matches(rng, n_inliers=80, n_outliers=30, noise=0.2):
    H_true = random_homography(rng)
    src_in = rng.uniform(0, 500, (n_inliers, 2))
    dst_in = apply_homography(H_true, src_in) + rng.normal(0, noise, (n_inliers, 2))

    # Outliers: random src paired with random unrelated dst.
    src_out = rng.uniform(0, 500, (n_outliers, 2))
    dst_out = rng.uniform(0, 500, (n_outliers, 2))

    src = np.vstack([src_in, src_out])
    dst = np.vstack([dst_in, dst_out])
    is_inlier = np.array([True] * n_inliers + [False] * n_outliers)

    # Shuffle so inliers/outliers are interleaved.
    perm = rng.permutation(len(src))
    return H_true, src[perm], dst[perm], is_inlier[perm]


def test_ransac_recovers_known_homography_with_outliers():
    rng = np.random.default_rng(42)
    H_true, src, dst, is_inlier = make_matches(rng, n_inliers=80, n_outliers=30)

    result = ransac_homography(src, dst, threshold=3.0,
                               rng=np.random.default_rng(123))

    H_est = result.H / result.H[2, 2]

    # Compare geometrically: the estimated and true homographies must agree on
    # where they send a grid of test points (a few px tolerance ~ the noise).
    grid = np.array([[x, y] for x in (0, 250, 500) for y in (0, 200, 400)],
                    dtype=np.float64)
    err = reprojection_error(H_est, grid, apply_homography(H_true, grid))
    assert err.max() < 2.0, f"geometric disagreement up to {err.max():.3f}px"

    # The recovered model should reproject the TRUE inliers with sub-pixel error.
    errs = reprojection_error(H_est, src[is_inlier], dst[is_inlier])
    assert errs.mean() < 1.0


def test_ransac_flags_the_outliers():
    rng = np.random.default_rng(7)
    H_true, src, dst, is_inlier = make_matches(rng, n_inliers=70, n_outliers=40)

    result = ransac_homography(src, dst, threshold=3.0,
                               rng=np.random.default_rng(99))

    pred_inlier = result.inlier_mask
    # Confusion vs. ground truth.
    true_pos = int((pred_inlier & is_inlier).sum())
    false_pos = int((pred_inlier & ~is_inlier).sum())
    false_neg = int((~pred_inlier & is_inlier).sum())

    # Almost all true inliers caught, essentially no outliers admitted.
    assert true_pos >= int(0.9 * is_inlier.sum())
    assert false_pos == 0
    assert false_neg <= int(0.1 * is_inlier.sum())


def test_ransac_inlier_count_matches_majority():
    rng = np.random.default_rng(2024)
    H_true, src, dst, is_inlier = make_matches(rng, n_inliers=100, n_outliers=25)
    result = ransac_homography(src, dst, threshold=3.0,
                               rng=np.random.default_rng(5))
    # Found inliers should be at least the number of true inliers, minus a little.
    assert result.n_inliers >= int(0.9 * is_inlier.sum())
    assert result.n_inliers <= len(src)


def test_ransac_requires_four_points():
    with pytest.raises(ValueError):
        ransac_homography(np.zeros((3, 2)), np.zeros((3, 2)))


def test_ransac_deterministic_with_seed():
    rng = np.random.default_rng(11)
    _, src, dst, _ = make_matches(rng, n_inliers=60, n_outliers=20)
    r1 = ransac_homography(src, dst, rng=np.random.default_rng(1))
    r2 = ransac_homography(src, dst, rng=np.random.default_rng(1))
    np.testing.assert_allclose(r1.H, r2.H)
    assert r1.n_inliers == r2.n_inliers
