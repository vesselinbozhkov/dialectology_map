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


def city_dots(img_bgr, thresh=70):
    """Centres of the small filled black dots that mark cities on the base map."""
    g = cv2.cvtColor(cv2.medianBlur(img_bgr, 3), cv2.COLOR_BGR2GRAY)
    b = (g < thresh).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(b)
    keep = [i for i in range(1, n)
            if 6 <= st[i][2] <= 13 and 6 <= st[i][3] <= 13
            and st[i][4] > 0.55 * st[i][2] * st[i][3] and abs(int(st[i][2]) - int(st[i][3])) <= 3]
    return cen[keep].astype(np.float32)


def dot_error(ref_dots, img_dots, warp):
    """Median distance (px) between reference dots and page dots mapped by warp (page->ref)."""
    if len(img_dots) == 0:
        return float("inf")
    mapped = img_dots @ warp[:, :2].T + warp[:, 2]
    d = np.linalg.norm(ref_dots[:, None] - mapped[None], axis=2).min(1)
    return float(np.median(d))


def dots_similarity(ref_dots, img_dots, tol=4.0, min_dist=150):
    """RANSAC over dot pairs: similarity transform page->ref with most inliers."""
    best, best_inl = None, 0
    if len(img_dots) < 3 or len(ref_dots) < 3:
        return None
    rd = np.linalg.norm(ref_dots[:, None] - ref_dots[None], axis=2)
    pd = np.linalg.norm(img_dots[:, None] - img_dots[None], axis=2)
    ri, rj = np.where(np.triu(rd > min_dist))
    pi, pj = np.where(np.triu(pd > min_dist))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pi), size=min(len(pi), 150), replace=False)
    for a in sel:
        p, q = img_dots[pi[a]], img_dots[pj[a]]
        ratio = rd[ri, rj] / pd[pi[a], pj[a]]
        for b in np.flatnonzero(np.abs(ratio - 1) < 0.12):
            for r0, r1 in ((ri[b], rj[b]), (rj[b], ri[b])):
                src = np.float32([p, q])
                dst = np.float32([ref_dots[r0], ref_dots[r1]])
                M, _ = cv2.estimateAffinePartial2D(src, dst)
                if M is None or abs(np.arctan2(M[1, 0], M[0, 0])) > 0.12:
                    continue
                mapped = img_dots @ M[:, :2].T + M[:, 2]
                inl = (np.linalg.norm(ref_dots[:, None] - mapped[None], axis=2).min(0) < tol).sum()
                if inl > best_inl:
                    best, best_inl = M, inl
    if best is None:
        return None
    # refit on inliers
    mapped = img_dots @ best[:, :2].T + best[:, 2]
    d = np.linalg.norm(ref_dots[:, None] - mapped[None], axis=2)
    m = d.min(0) < tol
    M, _ = cv2.estimateAffine2D(img_dots[m], ref_dots[d.argmin(0)[m]])
    return M if M is not None else best


def estimate_robust(ref_bgr, img_bgr, ref_dots=None):
    """Try ECC from phase correlation and from a dot-based similarity; keep the
    warp with the smallest median city-dot error. Returns (warp, dot_err, method)."""
    ref_dots = city_dots(ref_bgr) if ref_dots is None else ref_dots
    img_dots = city_dots(img_bgr)
    cands = []
    w, _ = estimate(ref_bgr, img_bgr)
    cands.append((dot_error(ref_dots, img_dots, cv2.invertAffineTransform(w)), w, "ecc"))
    if True:  # always try the dot-based fit; it is usually the most precise
        M = dots_hough(ref_dots, img_dots)
        if M is None:
            M = dots_similarity(ref_dots, img_dots)
        if M is not None:
            cands.append((dot_error(ref_dots, img_dots, M.astype(np.float32)), M.astype(np.float32), "dots"))
            a, b = dark_layer(ref_bgr), dark_layer(img_bgr)
            try:
                crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
                _, w2 = cv2.findTransformECC(a, b, cv2.invertAffineTransform(M).astype(np.float32),
                                             cv2.MOTION_AFFINE, crit, None, 5)
                cands.append((dot_error(ref_dots, img_dots, cv2.invertAffineTransform(w2)), w2, "dots+ecc"))
            except cv2.error:
                pass
    # ECC warps map ref->img (used with WARP_INVERSE_MAP); dot warps map img->ref
    best = min(cands, key=lambda c: c[0])
    err, w, method = best
    if method == "dots":
        w = cv2.invertAffineTransform(w).astype(np.float32)
    return w, err, method


def dots_hough(ref_dots, img_dots, bin_px=12, tol=(20, 8, 3)):
    """Fast page->ref fit: vote for the translation over all dot pairs, then
    iteratively refit an affine transform on the matched dots with a
    shrinking tolerance (handles a few degrees of rotation and ~10% scale)."""
    if len(img_dots) < 4 or len(ref_dots) < 4:
        return None
    off = (ref_dots[:, None, :] - img_dots[None, :, :]).reshape(-1, 2)
    keys = np.round(off / bin_px).astype(int)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    t = uniq[counts.argmax()] * bin_px
    M = np.float32([[1, 0, t[0]], [0, 1, t[1]]])
    for tl in tol:
        mapped = img_dots @ M[:, :2].T + M[:, 2]
        d = np.linalg.norm(ref_dots[:, None] - mapped[None], axis=2)
        m = d.min(0) < tl
        if m.sum() < 4:
            return None
        M2, _ = cv2.estimateAffine2D(img_dots[m], ref_dots[d.argmin(0)[m]])
        if M2 is None:
            return None
        M = M2.astype(np.float32)
    return M
