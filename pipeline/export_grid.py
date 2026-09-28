"""Export the generalised grid (pipeline/grid.py) to a web viewer.

Usage: python pipeline/export_grid.py
Writes data/out/web_grid/ with index.html, grid.json, territory.geojson and
cities.json.
"""
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"
WEB = OUT / "web_grid"
PART_NAMES = {"phon": "Фонетика", "acc": "Акцентология", "lex": "Лексика", "morph": "Морфология"}
KS = range(2, 13)
TOP = 8  # distinguishing features listed per cluster


def map_info():
    """Maps in atlas order with the proofread text and norm flag per class."""
    out = []
    for web in ("web_vol1-3", "web"):
        for m in json.loads((OUT / web / "index.json").read_text()):
            cls = {}
            for c in m["classes"]:
                texts = [i["text"] for i in c["items"] if i["text"]]
                norms = [i["norm"] for i in c["items"] if i["norm"] is not None]
                label = " / ".join(texts) or {"absent": "без явлението", "Н/Х/У": "Н / Х / У",
                                              "unlisted": "цвят извън легендата"}.get(c["special"], "")
                cls[str(c["class"])] = {"c": c["colour"], "t": label,
                                        "n": True if any(norms) else (False if norms else None)}
            out.append({"id": m["id"], "title": m["title"], "cls": cls})
    return out


def encode(s, cells, maps):
    """Per cell, two characters per map: class_a and class_b (chr(48 + class)), '..' if unsampled."""
    idx = {m["id"]: i for i, m in enumerate(maps)}
    buf = np.full((cells.cell.max() + 1, len(maps), 2), ord("."), np.uint8)
    buf[s.cell.values, s.map_id.map(idx).values, 0] = 48 + s.class_a.astype(int).values
    buf[s.cell.values, s.map_id.map(idx).values, 1] = 48 + s.class_b.astype(int).values
    return {c: buf[c].tobytes().decode("ascii") for c in cells.cell}


def distinctive(s, cells):
    """For each k and cluster: the (map, class) pairs most over-represented inside it."""
    long = pd.concat([s.assign(cls=s.class_a, w=0.5), s.assign(cls=s.class_b, w=0.5)])
    long["cls"] = long.cls.astype(int)
    share = long.pivot_table(index="cell", columns=["map_id", "cls"], values="w", aggfunc="sum", fill_value=0)
    out = {}
    for k in KS:
        lab = cells.set_index("cell")[f"k{k}"].dropna().astype(int)
        sh = share.loc[share.index.intersection(lab.index)]
        lab = lab.loc[sh.index]
        out[k] = {}
        for c in sorted(lab.unique()):
            inside = sh[lab == c].mean()
            outside = sh[lab != c].mean()
            score = (inside - outside).sort_values(ascending=False).head(TOP)
            out[k][int(c)] = [[mid, int(cl), round(float(inside[(mid, cl)]), 2), round(float(outside[(mid, cl)]), 2)]
                              for (mid, cl) in score.index]
    return out


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    cells = gpd.read_file(OUT / "grid.gpkg", layer="cells")
    s = pd.read_csv(OUT / "grid_samples.csv.gz")
    maps = map_info()
    codes = encode(s, cells, maps)
    dev_cols = [c for c in cells.columns if c.startswith("dev_")]
    feats = []
    for _, c in cells.iterrows():
        ring = [[round(x, 4), round(y, 4)] for x, y in list(c.geometry.exterior.coords)[:-1]]
        feats.append({
            "i": int(c.cell), "g": ring, "n": int(c.n_maps),
            "d": {k[4:]: (None if pd.isna(c[k]) else round(float(c[k]), 3)) for k in dev_cols},
            "k": None if pd.isna(c["k2"]) else [int(c[f"k{k}"]) for k in KS],
            "v": codes[c.cell],
        })
    places = pd.concat([pd.read_csv(ROOT / "pipeline/georef/cities.csv")[["name_bg", "lat", "lon"]],
                        pd.read_csv(ROOT / "pipeline/georef/places.csv")]).drop_duplicates("name_bg")
    data = {"maps": maps, "places": places.round(4).values.tolist(), "cells": feats, "parts": PART_NAMES, "ks": list(KS),
            "distinctive": distinctive(s, cells)}
    (WEB / "grid.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    for f in ("territory.geojson", "cities.json"):
        shutil.copy(OUT / "web" / f, WEB / f)
    tpl = (ROOT / "pipeline/web/grid.html").read_text()
    css = (ROOT / "pipeline/web/leaflet-1.9.4.min.css").read_text()
    (WEB / "index.html").write_text(tpl.replace("/*LEAFLET_CSS*/", css))
    print(f"{WEB}: {len(feats)} cells, {len(maps)} maps, grid.json {(WEB / 'grid.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
