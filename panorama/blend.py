"""Warping into a common canvas and feather (linear) blending.

Given a homography H that maps image A's pixels into image B's frame, we:
  1. compute the bounding box of both images in the output frame,
  2. shift everything by a translation so all coordinates are positive,
  3. warp each image into that canvas,
  4. blend the overlap with distance-based feather weights so the seam fades.

Warping uses OpenCV's `warpPerspective` when available; otherwise a NumPy
inverse-mapping bilinear sampler does the same job.
"""

from __future__ import annotations

import numpy as np

from .homography import apply_homography

try:
    import cv2  # type: ignore

    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False


def _corners(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)


def _bilinear_warp(img: np.ndarray, H: np.ndarray, out_w: int, out_h: int):
    """Inverse-map every output pixel through H^{-1} and bilinearly sample.

    Returns (warped, mask) where mask is 1.0 where the source covered the pixel.
    """
    Hinv = np.linalg.inv(H)
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    ones = np.ones_like(xs)
    dst = np.stack([xs.ravel(), ys.ravel(), ones.ravel()], axis=0).astype(np.float64)
    src = Hinv @ dst
    src /= src[2]
    sx = src[0].reshape(out_h, out_w)
    sy = src[1].reshape(out_h, out_w)

    h, w = img.shape[:2]
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)

    x0 = np.clip(np.floor(sx).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(sy).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = sx - x0
    wy = sy - y0

    if img.ndim == 2:
        img = img[..., None]
    chans = img.shape[2]
    out = np.zeros((out_h, out_w, chans), dtype=np.float64)
    for c in range(chans):
        ch = img[..., c]
        Ia = ch[y0, x0]
        Ib = ch[y0, x1]
        Ic = ch[y1, x0]
        Id = ch[y1, x1]
        top = Ia * (1 - wx) + Ib * wx
        bot = Ic * (1 - wx) + Id * wx
        out[..., c] = top * (1 - wy) + bot * wy
    out[~valid] = 0
    mask = valid.astype(np.float64)
    return out, mask


def _warp(img: np.ndarray, H: np.ndarray, out_w: int, out_h: int):
    if HAVE_CV2:
        warped = cv2.warpPerspective(img, H, (out_w, out_h))
        if warped.ndim == 2:
            warped = warped[..., None]
        # Mask of covered pixels via warping an all-ones image.
        ones = np.ones(img.shape[:2], dtype=np.float64)
        mask = cv2.warpPerspective(ones, H, (out_w, out_h))
        return warped.astype(np.float64), (mask > 0.5).astype(np.float64)
    return _bilinear_warp(img, H, out_w, out_h)


def _feather_weights(mask: np.ndarray) -> np.ndarray:
    """Distance-to-edge weights for a coverage mask (higher in the interior)."""
    if HAVE_CV2:
        dist = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    else:
        dist = _approx_distance_transform(mask > 0.5)
    return dist


def _approx_distance_transform(mask: np.ndarray, passes: int = 64) -> np.ndarray:
    """Cheap chamfer-style distance transform in NumPy (interior grows inward)."""
    d = np.where(mask, 0.0, np.inf)
    # Distance from the *background*; we actually want distance from edge inside
    # the mask, so seed background as 0 and propagate.
    d = np.where(mask, np.inf, 0.0)
    for _ in range(passes):
        shifted = [
            np.pad(d, ((1, 0), (0, 0)), constant_values=np.inf)[:-1] + 1,
            np.pad(d, ((0, 1), (0, 0)), constant_values=np.inf)[1:] + 1,
            np.pad(d, ((0, 0), (1, 0)), constant_values=np.inf)[:, :-1] + 1,
            np.pad(d, ((0, 0), (0, 1)), constant_values=np.inf)[:, 1:] + 1,
        ]
        new = np.minimum.reduce([d] + shifted)
        if np.array_equal(new, d):
            break
        d = new
    d[~mask] = 0.0
    d[np.isinf(d)] = 0.0
    return d


def warp_and_blend(img_ref: np.ndarray, img_mov: np.ndarray, H: np.ndarray):
    """Warp `img_mov` into `img_ref`'s frame via H and feather-blend the two.

    Parameters
    ----------
    img_ref : the reference image (stays in place, identity transform).
    img_mov : the moving image, mapped by H into the reference frame.
    H : 3x3 homography mapping img_mov coordinates -> img_ref coordinates.

    Returns
    -------
    panorama : uint8 blended image (H, W, C).
    """
    img_ref = np.asarray(img_ref)
    img_mov = np.asarray(img_mov)
    h_ref, w_ref = img_ref.shape[:2]
    h_mov, w_mov = img_mov.shape[:2]

    # Where do both images land in the output frame?
    ref_corners = _corners(w_ref, h_ref)
    mov_corners = apply_homography(H, _corners(w_mov, h_mov))
    all_corners = np.vstack([ref_corners, mov_corners])
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    out_w = int(max_xy[0] - min_xy[0])
    out_h = int(max_xy[1] - min_xy[1])

    # Translation so all coordinates are >= 0.
    tx, ty = -min_xy[0], -min_xy[1]
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    ref_warp, ref_mask = _warp(img_ref, T, out_w, out_h)
    mov_warp, mov_mask = _warp(img_mov, T @ H, out_w, out_h)

    # Feather weights.
    w_ref_f = _feather_weights(ref_mask)
    w_mov_f = _feather_weights(mov_mask)
    total = w_ref_f + w_mov_f
    total[total == 0] = 1.0
    w_ref_f = (w_ref_f / total)[..., None]
    w_mov_f = (w_mov_f / total)[..., None]

    blended = ref_warp * w_ref_f + mov_warp * w_mov_f
    # Where only one image covers, use it directly (weights already handle this,
    # but guard the no-coverage case).
    only_ref = (ref_mask > 0.5) & (mov_mask <= 0.5)
    only_mov = (mov_mask > 0.5) & (ref_mask <= 0.5)
    blended[only_ref] = ref_warp[only_ref]
    blended[only_mov] = mov_warp[only_mov]

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    if blended.shape[2] == 1:
        blended = blended[..., 0]
    return blended
