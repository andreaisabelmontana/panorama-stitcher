"""Stitch 2+ overlapping images into one panorama.

Pipeline per adjacent pair:
    detect + match features  ->  RANSAC homography  ->  warp into a common
    canvas  ->  feather-blend.

For 3+ images we anchor on the first, warp the second into it, then progressively
fold in each subsequent image (treating the running panorama as the reference).

Usage:
    python stitch.py left.png right.png -o output.png
    python stitch.py img1.jpg img2.jpg img3.jpg -o pano.png

With no arguments it runs the bundled demo (samples/left.png + samples/right.png),
generating the samples first if they are missing.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from panorama import (
    backend_name,
    detect_and_match,
    ransac_homography,
    reprojection_error,
    warp_and_blend,
)

try:
    import cv2

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


def _imread(path: str) -> np.ndarray:
    if HAVE_CV2:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return img
    from imageio import imread  # type: ignore

    return imread(path)[..., ::-1]  # RGB->BGR for consistency


def _imwrite(path: str, img: np.ndarray) -> None:
    if HAVE_CV2:
        cv2.imwrite(path, img)
    else:
        from imageio import imwrite  # type: ignore

        imwrite(path, img[..., ::-1])


def stitch_pair(img_ref: np.ndarray, img_mov: np.ndarray, threshold: float = 3.0,
                seed: int | None = 0, verbose: bool = True):
    """Stitch img_mov onto img_ref. Returns (panorama, stats dict)."""
    src, dst = detect_and_match(img_ref, img_mov)
    if len(src) < 4:
        raise RuntimeError(
            f"too few matches ({len(src)}) to estimate a homography"
        )
    rng = np.random.default_rng(seed) if seed is not None else None
    # We want the homography that maps the MOVING image into the REFERENCE frame,
    # so estimate dst(mov) -> src(ref).
    result = ransac_homography(dst, src, threshold=threshold, rng=rng)
    inlier_errs = reprojection_error(
        result.H, dst[result.inlier_mask], src[result.inlier_mask]
    )
    mean_err = float(inlier_errs.mean()) if len(inlier_errs) else float("nan")

    stats = {
        "matches": len(src),
        "inliers": result.n_inliers,
        "inlier_ratio": result.n_inliers / max(len(src), 1),
        "mean_reproj_error_px": mean_err,
        "ransac_iters": result.iterations,
    }
    if verbose:
        print(
            f"  matches={stats['matches']}  inliers={stats['inliers']}  "
            f"inlier_ratio={stats['inlier_ratio']:.2f}  "
            f"mean_reproj_err={stats['mean_reproj_error_px']:.3f}px  "
            f"iters={stats['ransac_iters']}"
        )
    pano = warp_and_blend(img_ref, img_mov, result.H)
    return pano, stats


def stitch(paths: list[str], out_path: str, threshold: float = 3.0,
           seed: int | None = 0) -> dict:
    print(f"backend: {backend_name()}")
    images = [_imread(p) for p in paths]
    print(f"loaded {len(images)} images: {[os.path.basename(p) for p in paths]}")

    pano = images[0]
    all_stats = []
    for i in range(1, len(images)):
        print(f"stitching image {i} onto running panorama:")
        pano, stats = stitch_pair(pano, images[i], threshold=threshold, seed=seed)
        all_stats.append(stats)

    _imwrite(out_path, pano)
    print(f"wrote {out_path}  ({pano.shape[1]}x{pano.shape[0]})")
    # Summary uses the last (or only) pair's numbers, plus totals.
    summary = {
        "output": out_path,
        "output_size": (pano.shape[1], pano.shape[0]),
        "pairs": all_stats,
    }
    return summary


def _ensure_samples():
    here = os.path.dirname(os.path.abspath(__file__))
    sdir = os.path.join(here, "samples")
    left = os.path.join(sdir, "left.png")
    right = os.path.join(sdir, "right.png")
    if not (os.path.exists(left) and os.path.exists(right)):
        print("samples missing — generating them...")
        sys.path.insert(0, sdir)
        import make_samples  # type: ignore

        make_samples.main()
    return [left, right]


def main(argv=None):
    parser = argparse.ArgumentParser(description="From-scratch panorama stitcher")
    parser.add_argument("images", nargs="*", help="2+ overlapping image paths (left-to-right)")
    parser.add_argument("-o", "--output", default="output.png", help="output panorama path")
    parser.add_argument("--threshold", type=float, default=3.0, help="RANSAC inlier threshold (px)")
    parser.add_argument("--seed", type=int, default=0, help="RANSAC RNG seed (use -1 for random)")
    args = parser.parse_args(argv)

    seed = None if args.seed == -1 else args.seed
    if not args.images:
        print("no images given — running bundled demo on samples/")
        images = _ensure_samples()
        out = args.output if args.output != "output.png" else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output.png"
        )
    else:
        if len(args.images) < 2:
            parser.error("need at least 2 images to stitch")
        images = args.images
        out = args.output

    summary = stitch(images, out, threshold=args.threshold, seed=seed)
    return summary


if __name__ == "__main__":
    main()
