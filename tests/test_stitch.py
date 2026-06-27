"""End-to-end stitch test on a synthetic split-image pair.

We build a textured scene, split it into two overlapping crops, stitch them,
and assert the recovered geometry aligns the overlap. Because the two crops come
from the SAME source image, the true homography mapping the right crop into the
left crop's frame is a pure horizontal translation, which we check the recovered
homography against. We also check photometric agreement in the overlap region.
"""

import numpy as np
import pytest

from panorama import detect_and_match, ransac_homography, reprojection_error
from panorama.features import backend_name


def build_scene(width=900, height=400, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 60, (height, width, 3)).astype(np.uint8)
    # Add many distinct rectangles/blobs so there are corners to match.
    for _ in range(120):
        x = int(rng.integers(0, width - 40))
        y = int(rng.integers(0, height - 40))
        w = int(rng.integers(8, 38))
        h = int(rng.integers(8, 38))
        color = rng.integers(80, 256, size=3)
        img[y:y + h, x:x + w] = color
    return img


def split_overlapping(scene, overlap_frac=0.4):
    h, w = scene.shape[:2]
    cut_left = int(w * (0.5 + overlap_frac / 2))
    cut_right = int(w * (0.5 - overlap_frac / 2))
    left = scene[:, :cut_left].copy()
    right = scene[:, cut_right:].copy()
    # The right crop starts at column cut_right in the scene, the left crop at 0,
    # so right maps into left by translating +cut_right in x.
    true_shift_x = cut_right
    return left, right, true_shift_x


def test_split_pair_alignment():
    scene = build_scene()
    left, right, true_shift_x = split_overlapping(scene)

    src, dst = detect_and_match(left, right)
    assert len(src) >= 4, f"need >=4 matches, got {len(src)} ({backend_name()})"

    # Homography mapping the right crop into the left crop's frame.
    result = ransac_homography(dst, src, threshold=3.0,
                               rng=np.random.default_rng(0))
    assert result.n_inliers >= 8

    H = result.H / result.H[2, 2]
    # The mapping should be ~ a translation by +true_shift_x in x, 0 in y.
    # Check by mapping the right crop's origin.
    mapped_origin = (H @ np.array([0, 0, 1.0]))
    mapped_origin = mapped_origin[:2] / mapped_origin[2]
    assert abs(mapped_origin[0] - true_shift_x) < 3.0
    assert abs(mapped_origin[1] - 0.0) < 3.0

    # Inlier reprojection error should be sub-pixel-ish.
    errs = reprojection_error(H, dst[result.inlier_mask], src[result.inlier_mask])
    assert errs.mean() < 2.0


def test_overlap_photometric_agreement():
    """After recovering the shift, the overlap region of both crops should match."""
    scene = build_scene(seed=5)
    left, right, true_shift_x = split_overlapping(scene)

    src, dst = detect_and_match(left, right)
    assert len(src) >= 4
    result = ransac_homography(dst, src, threshold=3.0,
                               rng=np.random.default_rng(1))
    H = result.H / result.H[2, 2]
    shift = (H @ np.array([0, 0, 1.0]))
    shift_x = shift[0] / shift[2]

    # Compare left[:, shift_x: ] against right[:, :overlap_width].
    shift_x = int(round(shift_x))
    overlap_w = min(left.shape[1] - shift_x, right.shape[1])
    assert overlap_w > 20
    a = left[:, shift_x:shift_x + overlap_w].astype(np.float64)
    b = right[:, :overlap_w].astype(np.float64)
    mean_abs_diff = np.abs(a - b).mean()
    # Same pixels from the same scene — difference should be tiny.
    assert mean_abs_diff < 5.0, f"overlap MAD too high: {mean_abs_diff:.2f}"
