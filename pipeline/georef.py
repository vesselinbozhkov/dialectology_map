"""Pixel (reference page) <-> geographic coordinates for the BDA base map.

Ground control points: city dots on the reference page (vol. IV p. 143) matched
to GeoNames coordinates, see georef/cities.csv. A 2nd-order polynomial in
EPSG:3035 (LAEA Europe) fits them with ~4 km RMS, about the accuracy of the
hand-drawn 1:3 000 000 base map itself.
"""
import csv
from pathlib import Path

import numpy as np
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
CRS = "EPSG:3035"
_to_laea = Transformer.from_crs(4326, 3035, always_xy=True)


def _design(x, y):
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)


def _fit():
    rows = list(csv.DictReader(open(HERE / "georef" / "cities.csv")))
    px = np.array([[float(r["px"]), float(r["py"])] for r in rows])
    geo = np.array([_to_laea.transform(float(r["lon"]), float(r["lat"])) for r in rows])
    coef, *_ = np.linalg.lstsq(_design(px[:, 0], px[:, 1]), geo, rcond=None)
    return coef


COEF = _fit()


def pixel_to_laea(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = _design(x, y) @ COEF
    return out[..., 0], out[..., 1]
