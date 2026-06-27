"""panorama — a from-scratch panorama stitcher.

Public surface:
    estimate_homography(src, dst)   normalized DLT homography (NumPy/SVD)
    ransac_homography(src, dst)     robust homography via RANSAC
    detect_and_match(img1, img2)    ORB (OpenCV) or Harris+NCC (NumPy) matches
    warp_and_blend(ref, mov, H)     warp into a canvas and feather-blend
"""

from .blend import warp_and_blend
from .features import backend_name, detect_and_match
from .homography import (
    apply_homography,
    estimate_homography,
    reprojection_error,
)
from .ransac import RansacResult, ransac_homography

__all__ = [
    "estimate_homography",
    "apply_homography",
    "reprojection_error",
    "ransac_homography",
    "RansacResult",
    "detect_and_match",
    "backend_name",
    "warp_and_blend",
]
