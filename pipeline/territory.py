"""Mask of the Bulgarian language territory on the reference page, taken from
vol. IV p. 23 (a map where the whole territory is one colour)."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from register import apply, estimate

ROOT = Path(__file__).resolve().parent.parent
# regions of the reference page that are not the main map
INSETS = [(100, 170, 260, 300), (890, 170, 1065, 350)]  # x0, y0, x1, y1
LEGEND_X1, LEGEND_Y0 = 450, 1250


def reference():
    return cv2.imread(str(ROOT / "data/raw/bda4/0143.png"))


def blank_frame(shape):
    m = np.ones(shape[:2], bool)
    for x0, y0, x1, y1 in INSETS:
        m[y0:y1, x0:x1] = False
    m[LEGEND_Y0:, :LEGEND_X1] = False
    m[:150, :] = False
    return m


@lru_cache(1)
def territory_mask():
    ref = reference()
    page = cv2.imread(str(ROOT / "data/raw/bda4/0023.png"))
    warp, _ = estimate(ref, page)
    al = apply(page, warp, ref.shape)
    hsv = cv2.cvtColor(al, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > 60) & blank_frame(ref.shape)
    # drop specks, close over thin black lines / text / dots
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = np.isin(lab, 1 + np.flatnonzero(sizes > 2000))
    # fill small holes (labels, dots) but keep lakes
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(holes)
    hs = ndimage.sum(holes, hl, range(1, hn + 1))
    m |= np.isin(hl, 1 + np.flatnonzero(hs < 1500))
    return m


INK_CACHE = ROOT / "data" / "work" / "base_ink.npy"


def ink(al):
    """Dark, unsaturated pixels (black/grey lines and text)."""
    hsv = cv2.cvtColor(al, cv2.COLOR_BGR2HSV)
    return (hsv[..., 2] < 120) & (hsv[..., 1] < 120)


def base_ink(pages=range(27, 172, 6)):
    """Ink present on (almost) every map: base-map lines, rivers, place names."""
    if INK_CACHE.exists():
        return np.load(INK_CACHE)
    ref = reference()
    acc = np.zeros(ref.shape[:2], np.float32)
    n = 0
    for p in pages:
        img = cv2.imread(str(ROOT / f"data/raw/bda4/{p:04d}.png"))
        warp, _ = estimate(ref, img)
        acc += cv2.dilate(ink(apply(img, warp, ref.shape)).astype(np.uint8), np.ones((3, 3), np.uint8))
        n += 1
    base = acc / n > 0.6
    base = cv2.dilate(base.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    INK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(INK_CACHE, base)
    return base
