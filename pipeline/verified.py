"""Hand-checked titles and legends (catalog/legends/<volume>.jsonl), joined to
the automatically extracted legend swatches.

Usage: python pipeline/verified.py vol1-3|vol4
Adds `title` (maps), `text`, `grp`, `norm` (legend) and a `legend_extra` table
to data/out/bda_<volume>.gpkg and copies it to output/.
"""
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(key):
    """{map_id: record} or {} if the volume has not been transcribed yet."""
    path = ROOT / "catalog" / "legends" / f"{key}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def apply(key):
    gpkg = ROOT / "data" / "out" / f"bda_{key}.gpkg"
    ver = load(key)
    maps = gpd.read_file(gpkg, layer="maps")
    legend = gpd.read_file(gpkg, layer="legend")
    maps["title"] = [ver[m]["title"] if m in ver else t for m, t in zip(maps.map_id, maps.title_ocr)]
    maps["verified"] = maps.map_id.isin(ver.keys())
    items = {(mid, it["i"]): it for mid, r in ver.items() for it in r["items"]}
    legend["text"] = [items.get((m, i), {}).get("text") for m, i in zip(legend.map_id, legend["item"])]
    legend["grp"] = [items.get((m, i), {}).get("group") for m, i in zip(legend.map_id, legend["item"])]
    legend["norm"] = pd.array([items.get((m, i), {}).get("norm") for m, i in zip(legend.map_id, legend["item"])],
                              dtype="boolean")
    for name, df in (("maps", maps), ("legend", legend)):  # attribute-only tables
        gpd.GeoDataFrame(df).to_file(gpkg, layer=name, driver="GPKG")
    extra = pd.DataFrame([{"map_id": mid, "grp": e["group"], "text": e["text"], "norm": e["norm"]}
                          for mid, r in ver.items() for e in r["extra"]]
                         + [{"map_id": mid, "grp": "", "text": "! " + r["note"], "norm": None}
                            for mid, r in ver.items() if r["note"]])
    extra["norm"] = extra.norm.astype("boolean")
    gpd.GeoDataFrame(extra).to_file(gpkg, layer="legend_extra", driver="GPKG")
    missing = sorted(set(items) - set(zip(legend.map_id, legend["item"])))
    if missing:
        print("transcribed items without a detected swatch:", missing)
    shutil.copy(gpkg, ROOT / "output" / gpkg.name)
    print(f"{gpkg.name}: {maps.verified.sum()}/{len(maps)} maps verified, "
          f"{legend.text.notna().sum()}/{len(legend)} swatches with text")


if __name__ == "__main__":
    apply(sys.argv[1] if len(sys.argv) > 1 else "vol1-3")
