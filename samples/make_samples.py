"""Generate a synthetic wide 'scene' and split it into two overlapping crops.

This gives the demo zero-manual-input sample inputs. We draw a textured skyline-
like scene (gradients, rectangles, circles, noise) so there are plenty of
corner features for ORB/Harris to latch onto, then save:
    samples/left.png   (left ~60% of the scene)
    samples/right.png  (right ~60% of the scene, overlapping the left in the middle)
"""

from __future__ import annotations

import os

import numpy as np

try:
    import cv2

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


def _save(path: str, img: np.ndarray) -> None:
    if HAVE_CV2:
        cv2.imwrite(path, img)
    else:  # minimal PNG writer via numpy + zlib through PIL-free path
        from imageio import imwrite  # type: ignore

        imwrite(path, img[..., ::-1])  # BGR->RGB


def build_scene(width: int = 1200, height: int = 500, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Sky gradient background.
    ys = np.linspace(0, 1, height)[:, None]
    sky = np.zeros((height, width, 3), dtype=np.float64)
    sky[..., 0] = 200 - 80 * ys           # B
    sky[..., 1] = 170 - 40 * ys           # G
    sky[..., 2] = 120 + 60 * ys           # R
    img = sky.copy()

    # A row of "buildings": rectangles of varied height/color along the bottom.
    x = 0
    while x < width:
        bw = int(rng.integers(40, 110))
        bh = int(rng.integers(120, 360))
        color = rng.integers(40, 200, size=3).astype(np.float64)
        x0, x1 = x, min(x + bw, width)
        y0 = height - bh
        img[y0:height, x0:x1] = color
        # windows
        for wy in range(y0 + 12, height - 10, 26):
            for wx in range(x0 + 8, x1 - 8, 20):
                if rng.random() > 0.35:
                    img[wy:wy + 12, wx:wx + 12] = np.array([60, 200, 255]) * rng.random()
        x += bw + int(rng.integers(4, 18))

    # Scatter some bright circles (sun/lights) for extra distinct features.
    for _ in range(30):
        cx, cy = rng.integers(0, width), rng.integers(0, height // 2)
        r = int(rng.integers(4, 16))
        yy, xx = np.ogrid[:height, :width]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[disk] = rng.integers(180, 256, size=3).astype(np.float64)

    # Light texture noise so flat regions still carry gradient.
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    scene = build_scene()
    h, w = scene.shape[:2]
    # Two overlapping crops: left [0, 0.62w], right [0.38w, w]. ~24% overlap.
    left = scene[:, : int(0.62 * w)].copy()
    right = scene[:, int(0.38 * w) :].copy()
    _save(os.path.join(here, "left.png"), left)
    _save(os.path.join(here, "right.png"), right)
    _save(os.path.join(here, "scene_truth.png"), scene)
    print(f"wrote left.png {left.shape}, right.png {right.shape}, scene_truth.png {scene.shape}")


if __name__ == "__main__":
    main()
