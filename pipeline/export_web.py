"""Export a volume's GeoPackage to per-map GeoJSON files + a web viewer.

Usage: python pipeline/export_web.py [vol4|vol1-3]
Writes data/out/web/ (vol4) or data/out/web_<volume>/ with index.html,
index.json, territory.geojson, cities.json, maps/<code>.geojson, qa/<code>.jpg
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from volumes import FILE_CODE, VOLUMES  # noqa: E402

OUT = ROOT / "data" / "out"


def web_dir(key):
    return OUT / "web" if key == "vol4" else OUT / f"web_{key}"


def code_of(map_id):
    letter, n = map_id.split()
    return f"{FILE_CODE[letter]}{int(n):03d}"


def main(key="vol4"):
    vol = VOLUMES[key]()
    gpkg = OUT / f"bda_{key}.gpkg"
    web = web_dir(key)
    maps = gpd.read_file(gpkg, layer="maps")
    legend = gpd.read_file(gpkg, layer="legend")
    areals = gpd.read_file(gpkg, layer="areals")
    mixed = gpd.read_file(gpkg, layer="hatch_mixed")
    lines = gpd.read_file(gpkg, layer="hatch_lines")
    (web / "maps").mkdir(parents=True, exist_ok=True)
    order = {mid: i for i, (mid, *_) in enumerate(vol.maps())}
    maps = maps.assign(_o=maps.map_id.map(order)).sort_values("_o")
    index = []
    for _, m in maps.iterrows():
        mid = m["map_id"]
        code = code_of(mid)
        a = areals[areals.map_id == mid].assign(kind="areal")
        h = mixed[mixed.map_id == mid].assign(kind="mixed")
        ln = lines[lines.map_id == mid].assign(kind="lines")
        feats = gpd.GeoDataFrame(pd.concat([a, h, ln], ignore_index=True), crs=4326)
        feats["geometry"] = feats.geometry.simplify(0.002)
        (web / "maps" / f"{code}.geojson").write_text(
            feats.drop(columns=["map_id"]).to_json(drop_id=True, na="drop"))
        leg = legend[legend.map_id == mid]
        classes = {int(r["class"]): r["colour"] for _, r in a.iterrows()}
        index.append({
            "id": mid, "file": code, "page": int(m["page"]), "title": m["title_ocr"],
            "reg_err": float(m["reg_dot_err_px"]),
            "classes": [{"class": int(c), "colour": col,
                         "special": next((r["special"] for _, r in a.iterrows() if int(r["class"]) == c), ""),
                         "legend": [r["text_ocr"] for _, r in leg.iterrows() if int(r["class"]) == c]}
                        for c, col in sorted(classes.items())],
        })
    (web / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    write_base(vol, web)
    write_page(vol, web)
    size = sum(f.stat().st_size for f in (web / "maps").glob("*.geojson"))
    print(f"{web}: {len(index)} maps, {size / 1e6:.1f} MB of GeoJSON")


def write_base(vol, web):
    """Territory outline and the reference cities, as a neutral base layer."""
    from run_volume import polygons, to_geo
    outline = to_geo(polygons(vol.territory())).simplify(0.003)
    gpd.GeoDataFrame(geometry=[outline], crs=4326).to_file(web / "territory.geojson", driver="GeoJSON")
    cities = [{"name": r["name_bg"], "lat": float(r["lat"]), "lon": float(r["lon"])}
              for r in csv.DictReader(open(ROOT / "pipeline/georef/cities.csv"))]
    (web / "cities.json").write_text(json.dumps(cities, ensure_ascii=False))


def write_page(vol, web):
    """Viewer page with Leaflet's CSS inlined, plus the QA images it links to."""
    tpl = (ROOT / "pipeline/web/index.html").read_text()
    css = (ROOT / "pipeline/web/leaflet-1.9.4.min.css").read_text()
    page = (tpl.replace("/*LEAFLET_CSS*/", css)
               .replace("{{TITLE}}", vol.title)
               .replace("{{SUBTITLE}}", vol.subtitle)
               .replace("{{START}}", vol.start_map)
               .replace("{{PAGE_TITLE}}", vol.page_title)
               .replace("{{SEARCH_HINT}}", vol.search_hint)
               .replace("{{KEY}}", vol.key))
    (web / "index.html").write_text(page)
    (web / "qa").mkdir(exist_ok=True)
    for src in (OUT / "qa" / vol.key, OUT / "qa"):  # older vol4 runs wrote to qa/ directly
        for f in src.glob("[MFAL][0-9][0-9][0-9].jpg"):
            if not (web / "qa" / f.name).exists() or src.name == vol.key:
                shutil.copy(f, web / "qa" / f.name)


if __name__ == "__main__":
    main(*sys.argv[1:2])
