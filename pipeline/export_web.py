"""Export the vol IV GeoPackage to per-map GeoJSON files + an index for a web viewer.

Usage: python pipeline/export_web.py
Writes data/out/web/index.json and data/out/web/maps/M###.geojson
"""
import csv
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
GPKG = ROOT / "data" / "out" / "bda_vol4.gpkg"
WEB = ROOT / "data" / "out" / "web"


def main():
    maps = gpd.read_file(GPKG, layer="maps")
    legend = gpd.read_file(GPKG, layer="legend")
    areals = gpd.read_file(GPKG, layer="areals")
    mixed = gpd.read_file(GPKG, layer="hatch_mixed")
    lines = gpd.read_file(GPKG, layer="hatch_lines")
    (WEB / "maps").mkdir(parents=True, exist_ok=True)
    index = []
    for _, m in maps.iterrows():
        mid = m["map_id"]
        n = int(mid.split()[1])
        a = areals[areals.map_id == mid].assign(kind="areal")
        h = mixed[mixed.map_id == mid].assign(kind="mixed")
        ln = lines[lines.map_id == mid].assign(kind="lines")
        feats = gpd.GeoDataFrame(pd.concat([a, h, ln], ignore_index=True), crs=4326)
        feats["geometry"] = feats.geometry.simplify(0.002)
        (WEB / "maps" / f"M{n:03d}.geojson").write_text(
            feats.drop(columns=["map_id"]).to_json(drop_id=True, na="drop"))
        leg = legend[legend.map_id == mid]
        classes = {int(r["class"]): r["colour"] for _, r in a.iterrows()}
        index.append({
            "id": mid, "n": n, "page": int(m["page"]), "title": m["title_ocr"],
            "reg_err": float(m["reg_dot_err_px"]),
            "classes": [{"class": int(c), "colour": col,
                         "special": next((r["special"] for _, r in a.iterrows() if int(r["class"]) == c), ""),
                         "legend": [r["text_ocr"] for _, r in leg.iterrows() if int(r["class"]) == c]}
                        for c, col in sorted(classes.items())],
        })
    (WEB / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    write_base()
    write_page()
    size = sum(f.stat().st_size for f in (WEB / "maps").glob("*.geojson"))
    print(f"{len(index)} maps, {size / 1e6:.1f} MB of GeoJSON")


def write_base():
    """Territory outline and the reference cities, as a neutral base layer."""
    from run_vol4 import polygons, to_geo
    from territory import territory_mask
    outline = to_geo(polygons(territory_mask())).simplify(0.003)
    gpd.GeoDataFrame(geometry=[outline], crs=4326).to_file(WEB / "territory.geojson", driver="GeoJSON")
    cities = [{"name": r["name_bg"], "lat": float(r["lat"]), "lon": float(r["lon"])}
              for r in csv.DictReader(open(ROOT / "pipeline/georef/cities.csv"))]
    (WEB / "cities.json").write_text(json.dumps(cities, ensure_ascii=False))


def write_page():
    """Viewer page with Leaflet's CSS inlined, plus the QA images it links to."""
    import shutil
    tpl = (ROOT / "pipeline/web/index.html").read_text()
    css = (ROOT / "pipeline/web/leaflet-1.9.4.min.css").read_text()
    (WEB / "index.html").write_text(tpl.replace("/*LEAFLET_CSS*/", css))
    (WEB / "qa").mkdir(exist_ok=True)
    for f in (ROOT / "data/out/qa").glob("M[0-9][0-9][0-9].jpg"):
        shutil.copy(f, WEB / "qa" / f.name)


if __name__ == "__main__":
    main()
