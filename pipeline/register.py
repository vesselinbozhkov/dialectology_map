"""Align every map page of a volume to a reference page.

The base map (borders, rivers, city dots) is identical on every map, but the
scans are shifted/rotated slightly page to page. We align the dark base layer
of each page to the reference page: phase correlation for the coarse shift,
then ECC (affine) refinement.
"""
import cv2
import numpy as np


def dark_layer(img_bgr, blur=3):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # dark = black lines/text/dots; colour fills are much lighter
    d = (255 - g).astype(np.float32)
    d[g > 110] = 0
    return cv2.GaussianBlur(d, (0, 0), blur)


def estimate(ref_bgr, img_bgr):
    """Return 2x3 affine warp mapping img -> ref coordinates, and ECC score."""
    a, b = dark_layer(ref_bgr), dark_layer(img_bgr)
    h, w = a.shape
    hb, wb = b.shape
    if (hb, wb) != (h, w):
        b = cv2.copyMakeBorder(b, 0, max(0, h - hb), 0, max(0, w - wb), cv2.BORDER_CONSTANT)[:h, :w]
        img_bgr = cv2.copyMakeBorder(img_bgr, 0, max(0, h - hb), 0, max(0, w - wb),
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))[:h, :w]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a * win, b * win)
    warp = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        cc = float("nan")
    return warp, cc


def apply(img_bgr, warp, shape, nearest=False):
    h, w = shape[:2]
    flags = (cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP
    return cv2.warpAffine(img_bgr, warp, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
