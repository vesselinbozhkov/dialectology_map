"""Extract all maps of one BDA volume into a GeoPackage.

Usage: python pipeline/run_volume.py vol4|vol1-3 [first last]   (1-based positions in the volume)
Writes data/out/bda_<volume>.gpkg with layers
  maps          one row per map (id, page, registration score, n_classes …)
  legend        one row per legend swatch (colour, letter flag, OCR text)
  areals        polygons per colour class
  hatch_mixed   striped zones of two legend colours (both forms coexist)
  hatch_lines   zones hatched with dark lines over one colour
and QA images to data/out/qa/<volume>/.
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from pyproj import Transformer
from rasterio import features
from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from georef import pixel_to_laea  # noqa: E402
from legend import legend_crops, ocr  # noqa: E402
from segment import (add_grey_class, classify, colour_classes, find_hatching,  # noqa: E402
                     find_line_hatching, find_swatches)
from territory import ink  # noqa: E402
from volumes import VOLUMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"
TO_WGS = Transformer.from_crs(3035, 4326, always_xy=True)
SIMPLIFY_M = 400


def to_geo(geom):
    """Pixel-space shapely geometry -> WGS84."""
    g = shp_transform(lambda x, y, z=None: pixel_to_laea(x, y), geom)
    g = g.simplify(SIMPLIFY_M)
    return shp_transform(lambda x, y, z=None: TO_WGS.transform(x, y), g)


def polygons(mask):
    geoms = [shape(s) for s, v in features.shapes(mask.astype(np.uint8), mask=mask) if v == 1]
    return unary_union(geoms) if geoms else None


def hex_colour(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def process(vol, entry, T, B):
    map_id, file_page, page, code, title = entry
    al, dot_err, method = vol.align(vol.page(file_page))
    swatches = find_swatches(al, T)
    classes = add_grey_class(al, T, colour_classes(swatches))
    labels = classify(al, T, classes, max_dist=vol.max_dist) if classes else np.zeros(T.shape, np.int32)
    own = ink(al) & ~B & T
    mixed = find_hatching(labels, T)
    lines = find_line_hatching(own, T, labels, min_density=0.14, max_ratio=0.8, min_area=400)
    texts = [ocr(c) for c in legend_crops(al, swatches)] if swatches else []

    rows = {"maps": [], "legend": [], "areals": [], "hatch_mixed": [], "hatch_lines": []}
    unassigned = float(((labels == 0) & T).sum() / T.sum())
    rows["maps"].append({"map_id": map_id, "code": code, "page": page, "file_page": file_page,
                         "title_ocr": title,
                         "reg_method": method, "reg_dot_err_px": round(dot_err, 2), "n_swatches": len(swatches),
                         "n_classes": len(classes), "unassigned_share": round(unassigned, 4),
                         "n_hatch_mixed": len(mixed), "n_hatch_lines": len(lines), "geometry": None})
    for i, (s, t) in enumerate(zip(swatches, texts)):
        cls = next(k for k, c in enumerate(classes) if s in c["swatches"]) + 1
        rows["legend"].append({"map_id": map_id, "item": i + 1, "class": cls,
                               "colour": hex_colour(s["bgr"]), "has_letter": s["has_letter"],
                               "text_ocr": t, "geometry": None})
    for k, c in enumerate(classes, start=1):
        g = polygons(labels == k)
        if g is not None:
            rows["areals"].append({"map_id": map_id, "class": k, "colour": hex_colour(c["bgr"]),
                                   "special": "absent" if c.get("grey") else ("Н/Х/У" if c.get("beige") else ""), "geometry": to_geo(g)})
    for m, a, b in mixed:
        rows["hatch_mixed"].append({"map_id": map_id, "class_a": a, "class_b": b,
                                    "geometry": to_geo(polygons(m))})
    for m, c in lines:
        rows["hatch_lines"].append({"map_id": map_id, "class": c, "geometry": to_geo(polygons(m))})

    # QA image: aligned page | segmentation with hatch outlines
    pal = np.array([(255, 255, 255)] + [c["bgr"] for c in classes], np.uint8)
    seg = pal[labels].copy()
    seg[~T] = (235, 235, 235)
    for m, *_ in mixed:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg, cnts, -1, (0, 0, 0), 2)
    for m, _ in lines:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg, cnts, -1, (255, 0, 255), 2)
    qa = np.hstack([al, seg])
    qa_dir = OUT / "qa" / vol.key
    qa_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qa_dir / f"{code}.jpg"), cv2.resize(qa, None, fx=0.5, fy=0.5),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    return rows


_JOB = None


def main(key="vol4", first=1, last=None):
    vol = VOLUMES[key]()
    T, B = vol.territory(), vol.base_ink()
    allrows = {k: [] for k in ["maps", "legend", "areals", "hatch_mixed", "hatch_lines"]}
    global _JOB
    _JOB = (vol, T, B)
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:  # fork: workers inherit _JOB
        results = pool.imap(_run_one, vol.maps()[first - 1:last])
        for entry, rows in results:
            if isinstance(rows, str):  # keep going; report at the end
                print(f"{entry[0]}: ERROR {rows}", flush=True)
                continue
            _collect(allrows, rows)
    _write(vol, allrows)


def _run_one(entry):
    vol, T, B = _JOB
    try:
        return entry, process(vol, entry, T, B)
    except Exception as e:
        return entry, repr(e)


def _collect(allrows, rows):
    for k, v in rows.items():
        allrows[k].extend(v)
    m = rows["maps"][0]
    print(f"{m['map_id']}: classes={m['n_classes']} unassigned={m['unassigned_share']:.1%} "
          f"mixed={m['n_hatch_mixed']} lines={m['n_hatch_lines']} reg={m['reg_method']}:{m['reg_dot_err_px']}px", flush=True)


def _write(vol, allrows):
    OUT.mkdir(parents=True, exist_ok=True)
    gpkg = OUT / f"bda_{vol.key}.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    for layer, rows in allrows.items():
        if not rows:
            continue
        if layer in ("maps", "legend"):
            df = gpd.GeoDataFrame([{k: v for k, v in r.items() if k != "geometry"} for r in rows])
            df.to_file(gpkg, layer=layer, driver="GPKG")  # attribute-only table
        else:
            gpd.GeoDataFrame(rows, crs=4326).to_file(gpkg, layer=layer, driver="GPKG")
    print(f"wrote {gpkg}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "vol4", *map(int, args[1:3]))
