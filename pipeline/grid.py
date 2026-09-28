"""Point × feature grid over the whole atlas and a first generalised map.

Samples every map (both volumes) at the centres of a hexagonal grid and
records which legend class covers each centre; two-colour hatching gives both
classes half weight. With the hand-checked `norm` flags each sample becomes a
distance from the literary language (0 = literary form, 1 = non-literary,
0.5 = both forms coexist).

Usage: python pipeline/grid.py [spacing_km]
Writes data/out/grid.gpkg (layer `cells`) and data/out/grid_samples.csv.gz
(cell, map_id, class_a, class_b, norm), and copies the GeoPackage to output/.
"""
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verified import load as load_verified  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"
VOLS = ("vol1-3", "vol4")
PARTS = {"Ф": "phon", "А": "acc", "Л": "lex", "М": "morph"}
KS = range(2, 13)  # cluster counts offered in the viewer


def hex_grid(area, spacing):
    """Hexagons (flat side up) of centre spacing `spacing` metres covering `area` (EPSG:3035)."""
    r = spacing / np.sqrt(3)
    x0, y0, x1, y1 = area.total_bounds
    cells = []
    for j, y in enumerate(np.arange(y0, y1 + spacing, spacing * np.sqrt(3) / 2)):
        for x in np.arange(x0 + (spacing / 2 if j % 2 else 0), x1 + spacing, spacing):
            cells.append(Polygon([(x + r * np.cos(a), y + r * np.sin(a))
                                  for a in np.radians(np.arange(30, 390, 60))]))
    g = gpd.GeoDataFrame(geometry=cells, crs=3035)
    g = g[g.centroid.within(area.union_all())].reset_index(drop=True)
    g["cell"] = np.arange(len(g))
    return g


def class_norms(key):
    """{(map_id, class): True/False/None} from the hand-checked legends.

    A class is literary if any of its legend items is, non-literary if all
    its rated items are not. The grey 'absent' class takes the rating of a
    '(сив фон)' legend note where there is one."""
    ver = load_verified(key)
    gpkg = OUT / f"bda_{key}.gpkg"
    leg = gpd.read_file(gpkg, layer="legend")
    areals = gpd.read_file(gpkg, layer="areals")
    out = {}
    for (mid, cls), rows in leg.groupby(["map_id", "class"]):
        vals = [v for v in rows.norm if pd.notna(v)]
        out[(mid, int(cls))] = True if any(vals) else (False if vals else None)
    for _, a in areals[areals.special == "absent"].iterrows():
        grey = [e for e in ver.get(a.map_id, {}).get("extra", []) if e["text"].startswith("(сив фон)")]
        out[(a.map_id, int(a["class"]))] = grey[0]["norm"] if grey else None
    for _, a in areals[areals.special.isin(["Н/Х/У", "unlisted"])].iterrows():
        out[(a.map_id, int(a["class"]))] = None
    return out


def samples(cells, key):
    gpkg = OUT / f"bda_{key}.gpkg"
    pts = gpd.GeoDataFrame(cells[["cell"]], geometry=cells.centroid, crs=3035).to_crs(4326)
    areals = gpd.read_file(gpkg, layer="areals")[["map_id", "class", "geometry"]]
    mixed = gpd.read_file(gpkg, layer="hatch_mixed")
    a = gpd.sjoin(pts, areals, predicate="within").drop_duplicates(["cell", "map_id"])
    m = gpd.sjoin(pts, mixed, predicate="within").drop_duplicates(["cell", "map_id"])
    s = a[["cell", "map_id", "class"]].rename(columns={"class": "class_a"})
    s["class_b"] = s["class_a"]
    s = s.set_index(["cell", "map_id"])
    mx = m.set_index(["cell", "map_id"])[["class_a", "class_b"]]
    s.update(mx)  # hatching overrides the plain fill under it
    s = pd.concat([s, mx[~mx.index.isin(s.index)]]).reset_index()
    norms = class_norms(key)
    na = s.apply(lambda r: norms.get((r.map_id, int(r.class_a))), axis=1)
    nb = s.apply(lambda r: norms.get((r.map_id, int(r.class_b))), axis=1)
    s["norm"] = [np.nanmean([np.nan if v is None else float(v) for v in (x, y)])
                 if (x is not None or y is not None) else np.nan for x, y in zip(na, nb)]
    return s


def main(spacing_km=12.0):
    terr = gpd.read_file(OUT / "web" / "territory.geojson").to_crs(3035)
    cells = hex_grid(terr, spacing_km * 1000)
    s = pd.concat([samples(cells, k) for k in VOLS], ignore_index=True)
    s["part"] = s.map_id.str[0].map(PARTS)
    s.to_csv(OUT / "grid_samples.csv.gz", index=False)
    dev = 1 - s.dropna(subset=["norm"]).pivot_table(index="cell", columns="part", values="norm", aggfunc="mean")
    dev["all"] = 1 - s.dropna(subset=["norm"]).groupby("cell").norm.mean()
    cells = cells.join(dev.add_prefix("dev_"), on="cell")
    cells["n_maps"] = cells.cell.map(s.groupby("cell").map_id.nunique()).fillna(0).astype(int)
    labels = clusters(s, cells)
    cells = cells.join(labels, on="cell")
    cells = cells.to_crs(4326)
    gpkg = OUT / "grid.gpkg"
    cells.to_file(gpkg, layer="cells", driver="GPKG")
    shutil.copy(gpkg, ROOT / "output" / gpkg.name)
    print(f"{len(cells)} cells, {s.map_id.nunique()} maps, {len(s)} samples; "
          f"median maps per cell {cells.n_maps.median():.0f}")


def clusters(s, cells, min_maps=250):
    """Ward clustering on one-hot class membership (hatched = half of each).

    Only features that actually vary are used, and cells with too few sampled
    maps (edges, insets) are left unclustered."""
    long = pd.concat([s.assign(f=s.map_id + ":" + s.class_a.astype(int).astype(str), w=0.5),
                      s.assign(f=s.map_id + ":" + s.class_b.astype(int).astype(str), w=0.5)])
    X = long.pivot_table(index="cell", columns="f", values="w", aggfunc="sum", fill_value=0)
    X = X[X.index.isin(cells.cell[cells.n_maps >= min_maps])]
    X = X.loc[:, (X > 0).mean().between(0.02, 0.98)]
    Z = linkage(X.values, method="ward")
    return pd.DataFrame({f"k{k}": fcluster(Z, k, criterion="maxclust") for k in KS}, index=X.index)


if __name__ == "__main__":
    main(*map(float, sys.argv[1:2]))
