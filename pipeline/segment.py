"""Segment one aligned atlas map into areals by legend colour."""
import cv2
import numpy as np
from scipy import ndimage

from territory import LEGEND_Y0, territory_mask


def lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def find_swatches(al):
    """Coloured legend squares in the bottom-left legend block.

    Returns list of dicts: bbox, bgr colour, has_letter (a dark glyph inside).
    """
    # legend sits below/left of the territory; search everywhere off the map
    off_map = ~cv2.dilate(territory_mask().astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    region = al[LEGEND_Y0:, :].copy()
    region[~off_map[LEGEND_Y0:, :]] = 255
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    coloured = ((hsv[..., 1] > 50) | ((g < 200) & (g > 70))).astype(np.uint8)
    coloured = cv2.morphologyEx(coloured, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(coloured)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 16 <= w <= 34 and 16 <= h <= 34 and abs(w - h) <= 6 and area > 0.7 * w * h:
            box = region[y + 3:y + h - 3, x + 3:x + w - 3].reshape(-1, 3)
            gray = box.mean(1)
            fill = box[gray > 80]  # ignore a dark letter printed on the swatch
            out.append({
                "bbox": (int(x), int(y + LEGEND_Y0), int(w), int(h)),
                "bgr": tuple(int(v) for v in np.median(fill, 0)),
                "has_letter": bool((gray < 80).mean() > 0.03),
            })
    # legend reads bottom-to-top on the rotated page? keep page order: by x then y
    out.sort(key=lambda s: (s["bbox"][0] // 40, -s["bbox"][1]))
    return out


def colour_classes(swatches, tol=18.0):
    """Merge swatches of the same colour (e.g. red А / red Б) into classes."""
    classes = []
    for s in swatches:
        c = lab(np.uint8([[s["bgr"]]]))[0, 0]
        for k in classes:
            if np.linalg.norm(k["lab"] - c) < tol:
                k["swatches"].append(s)
                break
        else:
            classes.append({"lab": c, "bgr": s["bgr"], "swatches": [s]})
    return classes


def classify(al, territory, classes, max_dist=28.0, fill_radius=10):
    """Label raster: 0 = unassigned, k+1 = class k."""
    L = lab(al)
    centres = np.stack([k["lab"] for k in classes])
    d = np.linalg.norm(L[:, :, None, :] - centres[None, None], axis=-1)
    nearest = d.argmin(-1)
    ok = (d.min(-1) < max_dist) & territory
    labels = np.where(ok, nearest + 1, 0).astype(np.int32)
    # remove speckle (antialiasing, text fringes)
    for k in range(1, len(classes) + 1):
        m = labels == k
        opened = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        labels[m & ~opened] = 0
    # pixels under black lines / letters / dots: take nearest labelled pixel
    gaps = (labels == 0) & territory
    dist, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    fill = gaps & (dist <= fill_radius)
    labels[fill] = labels[iy[fill], ix[fill]]
    return labels


GREY_BGR = (231, 231, 231)  # base fill = territory where the mapped feature is absent


def add_grey_class(al, territory, classes, min_share=0.03):
    """Monochrome/distribution maps paint the rest of the territory light grey."""
    L = lab(al)[territory]
    g = lab(np.uint8([[GREY_BGR]]))[0, 0]
    share = (np.linalg.norm(L - g, axis=1) < 8).mean()
    if share > min_share and all(np.linalg.norm(k["lab"] - g) > 12 for k in classes):
        classes.append({"lab": g, "bgr": GREY_BGR, "swatches": [], "grey": True})
    return classes


def find_hatching(labels, territory, win=21, min_share=0.2, min_changes=0.12, min_area=150):
    """Striped zones (two alternating legend colours) -> list of (mask, class_a, class_b).

    Inside hatching the label changes every few pixels; at an ordinary border
    between areals it changes once. Measure the density of label changes in a
    window and the two dominant classes there.
    """
    lab_ = labels.astype(np.int32)
    ch = np.zeros(lab_.shape, np.float32)
    ch[:, 1:] += (lab_[:, 1:] != lab_[:, :-1]) & (lab_[:, 1:] > 0) & (lab_[:, :-1] > 0)
    ch[1:, :] += (lab_[1:, :] != lab_[:-1, :]) & (lab_[1:, :] > 0) & (lab_[:-1, :] > 0)
    density = cv2.boxFilter(ch, -1, (win, win))
    k = int(labels.max())
    shares = np.stack([cv2.boxFilter((labels == c).astype(np.float32), -1, (win, win))
                       for c in range(1, k + 1)], -1) if k else None
    if shares is None or k < 2:
        return []
    order = np.argsort(-shares, -1)
    top1 = np.take_along_axis(shares, order[..., :1], -1)[..., 0]
    top2 = np.take_along_axis(shares, order[..., 1:2], -1)[..., 0]
    cand = (density > min_changes) & (top2 > min_share) & (top1 > min_share) & territory
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, comp = cv2.connectedComponents(cand)
    out = []
    for i in range(1, n):
        m = comp == i
        if m.sum() < min_area:
            continue
        a, b = np.bincount(labels[m], minlength=k + 1)[1:].argsort()[::-1][:2] + 1
        out.append((m, int(a), int(b)))
    return out


def find_line_hatching(own_ink, territory, labels, win=25, min_density=0.16, max_ratio=0.45, min_area=300):
    """Zones hatched with thin parallel dark lines over one colour -> [(mask, class)].

    `own_ink` is the map's own ink (base-map lines and names removed). Hatching
    is many parallel lines: the window is crossed often along one axis and
    rarely along the other; letters and leftover river fragments are not.
    """
    ink_ = own_ink.astype(np.int8)
    hx = np.zeros(ink_.shape, np.float32)
    hy = np.zeros(ink_.shape, np.float32)
    hx[:, 1:] = np.abs(np.diff(ink_, axis=1))
    hy[1:, :] = np.abs(np.diff(ink_, axis=0))
    hx = cv2.boxFilter(hx, -1, (win, win))
    hy = cv2.boxFilter(hy, -1, (win, win))
    hi, lo = np.maximum(hx, hy), np.minimum(hx, hy)
    cand = (hi > min_density) & (lo < max_ratio * hi) & territory
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, comp = cv2.connectedComponents(cand)
    out = []
    for i in range(1, n):
        m = comp == i
        if m.sum() < min_area:
            continue
        c = np.bincount(labels[m], minlength=int(labels.max()) + 1)
        c[0] = 0
        out.append((m, int(c.argmax())))
    return out
