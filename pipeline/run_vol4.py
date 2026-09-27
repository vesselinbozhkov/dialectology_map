"""Extract all maps of BDA vol. IV (Морфология) into a GeoPackage.

Usage: python pipeline/run_vol4.py [first_map last_map]
Writes data/out/bda_vol4.gpkg with layers
  maps          one row per map (id, page, registration score, n_classes …)
  legend        one row per legend swatch (colour, letter flag, OCR text)
  areals        polygons per colour class
  hatch_mixed   striped zones of two legend colours (both forms coexist)
  hatch_lines   zones hatched with dark lines over one colour
and QA images to data/out/qa/.
"""
import csv
import sys
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
from register import apply, city_dots, estimate_robust  # noqa: E402
from segment import (add_grey_class, classify, colour_classes, find_hatching,  # noqa: E402
                     find_line_hatching, find_swatches)
from territory import base_ink, ink, reference, territory_mask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"
FIRST_PAGE = 26  # map М n is on page 26 + n
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


def process(n, ref, T, B, catalog):
    page = FIRST_PAGE + n
    img = cv2.imread(str(ROOT / f"data/raw/bda4/{page:04d}.png"))
    warp, dot_err, method = estimate_robust(ref, img, REF_DOTS)
    al = apply(img, warp, ref.shape)
    swatches = find_swatches(al)
    classes = add_grey_class(al, T, colour_classes(swatches))
    labels = classify(al, T, classes) if classes else np.zeros(T.shape, np.int32)
    own = ink(al) & ~B & T
    mixed = find_hatching(labels, T)
    lines = find_line_hatching(own, T, labels, min_density=0.14, max_ratio=0.8, min_area=400)
    map_id = f"М {n}"
    texts = [ocr(c) for c in legend_crops(al, swatches)] if swatches else []

    rows = {"maps": [], "legend": [], "areals": [], "hatch_mixed": [], "hatch_lines": []}
    unassigned = float(((labels == 0) & T).sum() / T.sum())
    rows["maps"].append({"map_id": map_id, "page": page, "title_ocr": catalog.get(map_id, ""),
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
    (OUT / "qa").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "qa" / f"M{n:03d}.jpg"), cv2.resize(qa, None, fx=0.5, fy=0.5),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    return rows


REF_DOTS = None


def main(first=1, last=145):
    global REF_DOTS
    ref, T, B = reference(), territory_mask(), base_ink()
    REF_DOTS = city_dots(ref)
    catalog = {r["id"]: r["title_ocr"] for r in csv.DictReader(open(ROOT / "catalog/maps.csv"))}
    allrows = {k: [] for k in ["maps", "legend", "areals", "hatch_mixed", "hatch_lines"]}
    for n in range(first, last + 1):
        try:
            rows = process(n, ref, T, B, catalog)
        except Exception as e:  # keep going; report at the end
            print(f"М {n}: ERROR {e!r}", flush=True)
            continue
        for k, v in rows.items():
            allrows[k].extend(v)
        m = rows["maps"][0]
        print(f"{m['map_id']}: classes={m['n_classes']} unassigned={m['unassigned_share']:.1%} "
              f"mixed={m['n_hatch_mixed']} lines={m['n_hatch_lines']} reg={m['reg_method']}:{m['reg_dot_err_px']}px", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    gpkg = OUT / "bda_vol4.gpkg"
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
    main(*map(int, sys.argv[1:3]))
