"""Feature detection and matching.

Primary path uses OpenCV ORB keypoints + binary (Hamming) brute-force matching,
which is fast and robust. If OpenCV is not importable, we fall back to a pure
NumPy implementation: Harris corner detection + normalized cross-correlation
(NCC) patch matching. Either path returns the same thing — a pair of matched
(n, 2) point arrays (src, dst) in (x, y) pixel coordinates.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore

    HAVE_CV2 = True
except Exception:  # pragma: no cover - exercised only when OpenCV is absent
    HAVE_CV2 = False


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert an image to a float64 grayscale array in [0, 255]."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        # Luma weights (BGR or RGB — weights are symmetric enough for matching).
        arr = arr[..., :3].astype(np.float64)
        gray = 0.114 * arr[..., 0] + 0.587 * arr[..., 1] + 0.299 * arr[..., 2]
    else:
        gray = arr.astype(np.float64)
    return gray


# --------------------------------------------------------------------------- #
# OpenCV ORB path
# --------------------------------------------------------------------------- #
def _match_orb(img1: np.ndarray, img2: np.ndarray, max_features: int, ratio: float):
    gray1 = _to_gray(img1).astype(np.uint8)
    gray2 = _to_gray(img2).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # k-NN + Lowe ratio test to drop ambiguous matches.
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 4:
        # Fall back to plain best-match if the ratio test was too strict.
        good = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]

    src = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float64)
    dst = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float64)
    return src, dst


# --------------------------------------------------------------------------- #
# NumPy Harris + NCC fallback
# --------------------------------------------------------------------------- #
def _sobel(gray: np.ndarray):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    return _convolve2d(gray, kx), _convolve2d(gray, ky)


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def _box_blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    return _convolve2d(img, kernel)


def harris_corners(gray: np.ndarray, max_corners: int = 400, k: float = 0.04,
                   min_distance: int = 8) -> np.ndarray:
    """Detect Harris corners, return (m, 2) array of (x, y) coordinates."""
    Ix, Iy = _sobel(gray)
    Sxx = _box_blur(Ix * Ix)
    Syy = _box_blur(Iy * Iy)
    Sxy = _box_blur(Ix * Iy)
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    response = det - k * trace * trace

    # Threshold and keep local maxima, sorted by response, with spacing.
    h, w = response.shape
    border = 12
    flat = []
    thresh = 0.01 * response.max()
    ys, xs = np.where(response > thresh)
    for y, x in zip(ys, xs):
        if y < border or y >= h - border or x < border or x >= w - border:
            continue
        flat.append((response[y, x], x, y))
    flat.sort(reverse=True)

    chosen: list[tuple[int, int]] = []
    md2 = min_distance * min_distance
    for _, x, y in flat:
        ok = True
        for cx, cy in chosen:
            if (cx - x) ** 2 + (cy - y) ** 2 < md2:
                ok = False
                break
        if ok:
            chosen.append((x, y))
        if len(chosen) >= max_corners:
            break
    return np.array(chosen, dtype=np.float64) if chosen else np.empty((0, 2))


def _patch(gray: np.ndarray, x: float, y: float, r: int) -> np.ndarray | None:
    xi, yi = int(round(x)), int(round(y))
    if xi - r < 0 or yi - r < 0 or xi + r + 1 > gray.shape[1] or yi + r + 1 > gray.shape[0]:
        return None
    p = gray[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(np.float64).ravel()
    p = p - p.mean()
    norm = np.linalg.norm(p)
    if norm < 1e-8:
        return None
    return p / norm


def _match_harris_ncc(img1: np.ndarray, img2: np.ndarray, max_features: int,
                      ncc_thresh: float = 0.8, patch_radius: int = 5):
    gray1 = _to_gray(img1)
    gray2 = _to_gray(img2)
    c1 = harris_corners(gray1, max_corners=max_features)
    c2 = harris_corners(gray2, max_corners=max_features)
    if len(c1) < 4 or len(c2) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    desc1 = [(_patch(gray1, x, y, patch_radius), (x, y)) for x, y in c1]
    desc2 = [(_patch(gray2, x, y, patch_radius), (x, y)) for x, y in c2]
    desc1 = [d for d in desc1 if d[0] is not None]
    desc2 = [d for d in desc2 if d[0] is not None]
    if len(desc1) < 4 or len(desc2) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    D2 = np.stack([d[0] for d in desc2])
    src, dst = [], []
    used = set()
    for p1, pt1 in desc1:
        scores = D2 @ p1  # NCC since patches are zero-mean, unit-norm.
        order = np.argsort(scores)[::-1]
        best, second = order[0], order[1] if len(order) > 1 else order[0]
        if scores[best] >= ncc_thresh and scores[best] > scores[second] + 0.05:
            if best in used:
                continue
            used.add(best)
            src.append(pt1)
            dst.append(desc2[best][1])
    return (np.array(src, dtype=np.float64) if src else np.empty((0, 2)),
            np.array(dst, dtype=np.float64) if dst else np.empty((0, 2)))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def detect_and_match(img1: np.ndarray, img2: np.ndarray, max_features: int = 2000,
                     ratio: float = 0.75):
    """Detect and match features between two images.

    Returns
    -------
    (src, dst) : two (n, 2) float arrays of matched (x, y) coordinates, where
        src points come from img1 and dst points from img2.
    """
    if HAVE_CV2:
        return _match_orb(img1, img2, max_features, ratio)
    return _match_harris_ncc(img1, img2, max_features)


def backend_name() -> str:
    return "opencv-orb" if HAVE_CV2 else "numpy-harris-ncc"
