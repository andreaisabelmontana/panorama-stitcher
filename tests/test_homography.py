"""Tests for the from-scratch DLT homography estimator."""

import numpy as np
import pytest

from panorama.homography import (
    apply_homography,
    estimate_homography,
    reprojection_error,
)


def random_homography(rng):
    """A well-conditioned random homography close to identity + perspective."""
    H = np.eye(3)
    H[:2, :2] += rng.uniform(-0.15, 0.15, (2, 2))   # rotation/shear/scale
    H[:2, 2] = rng.uniform(-40, 40, 2)              # translation
    H[2, :2] = rng.uniform(-1e-4, 1e-4, 2)          # mild perspective
    return H / H[2, 2]


def test_four_points_mapped_exactly():
    """estimate_homography must map the 4 source points onto the dst points."""
    rng = np.random.default_rng(0)
    H_true = random_homography(rng)
    src = np.array([[10.0, 20.0], [200.0, 30.0], [180.0, 240.0], [25.0, 220.0]])
    dst = apply_homography(H_true, src)

    H_est = estimate_homography(src, dst)
    mapped = apply_homography(H_est, src)
    np.testing.assert_allclose(mapped, dst, atol=1e-6)


def test_recovers_known_homography_from_many_points():
    rng = np.random.default_rng(1)
    H_true = random_homography(rng)
    src = rng.uniform(0, 400, (50, 2))
    dst = apply_homography(H_true, src)

    H_est = estimate_homography(src, dst)
    # Compare up to scale by normalizing both.
    H_est = H_est / H_est[2, 2]
    np.testing.assert_allclose(H_est, H_true, atol=1e-6)
    assert reprojection_error(H_est, src, dst).max() < 1e-6


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        estimate_homography(np.zeros((3, 2)), np.zeros((3, 2)))


def test_robust_to_noisy_coordinates():
    """With small Gaussian noise on dst, the fit should still be close."""
    rng = np.random.default_rng(2)
    H_true = random_homography(rng)
    src = rng.uniform(0, 500, (80, 2))
    dst = apply_homography(H_true, src) + rng.normal(0, 0.3, (80, 2))

    H_est = estimate_homography(src, dst)
    # Mean reprojection error should be on the order of the noise, not huge.
    assert reprojection_error(H_est, src, dst).mean() < 1.0
