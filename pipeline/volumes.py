"""Per-volume settings: where the map pages are, how to align them to the
reference frame (vol. IV p. 143, which is georeferenced), the territory mask
and the base-map ink for each volume.

Both volumes use the same hand-drawn base map, so all maps end up in one
pixel frame and share georef.py.
"""
import csv
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

import territory as t4
from register import apply, apply_print, city_dots, estimate_robust, register_print

ROOT = Path(__file__).resolve().parent.parent
FILE_CODE = {"М": "M", "Ф": "F", "А": "A", "Л": "L"}  # ASCII for file names


def catalog_rows():
    return list(csv.DictReader(open(ROOT / "catalog/maps.csv")))


def clean_mask(m, min_area=2000, max_hole=1500):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = np.isin(lab, 1 + np.flatnonzero(sizes > min_area))
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(holes)
    hs = ndimage.sum(holes, hl, range(1, hn + 1))
    return m | np.isin(hl, 1 + np.flatnonzero(hs < max_hole))


class Volume:
    key = ""          # short name used in output paths
    book_id = ""
    parts = ""        # map-id letters in this volume
    title = ""
    subtitle = ""
    start_map = ""
    max_dist = 28.0   # Lab distance from a legend colour still counted as that class

    def __init__(self):
        self.ref = t4.reference()
        self.ref_dots = city_dots(self.ref)

    def maps(self):
        """[(map_id, file_page, printed_page, file_code)] in atlas order."""
        out = []
        for r in catalog_rows():
            if r["book_id"] == self.book_id and r["file_page"]:
                letter, n = r["id"].split()
                out.append((r["id"], int(r["file_page"]), int(r["page"]),
                            f"{FILE_CODE[letter]}{int(n):03d}", r["title_ocr"]))
        return out

    def page(self, file_page):
        return cv2.imread(str(ROOT / f"data/raw/{self.book_id}/{file_page:04d}.png"))

    # overridden below
    def align(self, img):
        raise NotImplementedError

    def territory(self):
        raise NotImplementedError

    def base_ink(self):
        raise NotImplementedError


class Vol4(Volume):
    key, book_id, parts = "vol4", "bda4", "М"
    title = "БДА · Морфология"
    subtitle = "Ареали, извлечени автоматично от Обобщаващ том IV (2016). Работна чернова за проверка."
    start_map = "М 117"
    page_title = "БДА Морфология · ареали"
    search_hint = "напр. 117, член, бъдеще"

    def align(self, img):
        warp, err, method = estimate_robust(self.ref, img, self.ref_dots)
        return apply(img, warp, self.ref.shape), err, method

    def territory(self):
        return t4.territory_mask()

    def base_ink(self):
        return t4.base_ink()


class Vol13(Volume):
    """Offset-printed scan: halftone dots, slightly different scale and curvature."""
    key, book_id, parts = "vol1-3", "bda", "ФАЛ"
    title = "БДА · Фонетика, акцентология, лексика"
    subtitle = "Ареали, извлечени автоматично от Обобщаващ том I–III (2001). Работна чернова за проверка."
    start_map = "Ф 34"
    page_title = "БДА Фонетика, акцентология, лексика"
    search_hint = "напр. Ф 34, Л 40, ят, котка"
    TERRITORY_PAGE = 55  # „Общи типологични особености“: whole territory in one colour
    INK_CACHE = ROOT / "data" / "work" / "base_ink_vol1-3.npy"
    max_dist = 45.0  # printed tints drift across the page

    def align(self, img, smooth=True):
        r = register_print(self.ref_dots, img)
        if r is None:
            raise RuntimeError("registration failed")
        coef, n, res = r
        al = apply_print(img, coef, self.ref.shape)
        if smooth:  # remove the halftone screen, keep edges and hatching
            al = cv2.pyrMeanShiftFiltering(al, 6, 20)
        return al, res, f"dots-poly2:{n}"

    @lru_cache(1)
    def territory(self):
        al, _, _ = self.align(self.page(self.TERRITORY_PAGE))
        hsv = cv2.cvtColor(al, cv2.COLOR_BGR2HSV)
        return clean_mask((hsv[..., 1] > 45) & t4.blank_frame(self.ref.shape))

    def base_ink(self):
        if self.INK_CACHE.exists():
            return np.load(self.INK_CACHE)
        files = [fp for _, fp, *_ in self.maps()][::8]
        acc = np.zeros(self.ref.shape[:2], np.float32)
        for fp in files:
            al, _, _ = self.align(self.page(fp))
            acc += cv2.dilate(t4.ink(al).astype(np.uint8), np.ones((3, 3), np.uint8))
        base = acc / len(files) > 0.5
        base = cv2.dilate(base.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        self.INK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.INK_CACHE, base)
        return base


VOLUMES = {"vol4": Vol4, "vol1-3": Vol13}
