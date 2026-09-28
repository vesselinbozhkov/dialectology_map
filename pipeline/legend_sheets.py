"""Reading sheets for manual legend transcription.

For each map: the title strip and the legend block, rotated upright, with the
index of every detected legend swatch (same order as the `legend` layer,
`item` column) drawn next to it. Two maps per sheet.

Usage: python pipeline/legend_sheets.py vol1-3|vol4
Writes data/work/sheets/<volume>/<first-map-code>.png
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment import find_swatches  # noqa: E402
from volumes import VOLUMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCALE = 1.35


def upright(al):
    return cv2.rotate(al, cv2.ROTATE_90_CLOCKWISE)


def map_card(vol, entry, T):
    map_id, file_page, *_ = entry
    img = vol.page(file_page)
    if vol.key == "vol1-3":
        smooth, *_ = vol.align(img)
        plain, *_ = vol.align(img, smooth=False)
    else:
        smooth, *_ = vol.align(img)
        plain = smooth
    swatches = find_swatches(smooth, T)
    marked = plain.copy()
    for i, s in enumerate(swatches, start=1):
        x, y, w, h = s["bbox"]
        cv2.rectangle(marked, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 0, 255), 2)
        # number below the swatch (reads upright after rotation: to its left)
        cv2.putText(marked, str(i), (x + 2, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    H = marked.shape[0]
    rot = upright(marked)  # original (x, y) -> (H-1-y, x)
    # legend items stack along page x: extend past the last swatch for trailing
    # items without a swatch (У, Н, hatching keys)
    end = min(900, max([s["bbox"][0] + s["bbox"][2] for s in swatches] + [400]) + 130)
    legend = rot[50:end, H - 1 - 1625:H - 1 - 980]
    title = rot[80:230, H - 1 - 1150:H - 1 - 330]
    w = max(legend.shape[1], title.shape[1])
    pad = lambda a: cv2.copyMakeBorder(a, 0, 0, 0, w - a.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    label = np.full((34, w, 3), 255, np.uint8)
    cv2.putText(label, f"{map_id}   (swatches: {len(swatches)})", (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    card = np.vstack([label, pad(title), pad(legend)])
    return card


def main(key):
    out = ROOT / "data" / "work" / "sheets" / key
    out.mkdir(parents=True, exist_ok=True)
    maps = _vol(key).maps()
    todo = [i for i in range(0, len(maps), 2) if not (out / f"{maps[i][3]}.png").exists()]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for code in pool.imap(_sheet, [(key, maps[i:i + 2], str(out)) for i in todo]):
            print(code, flush=True)


def _sheet(args):
    key, pair, out = args
    vol = _vol(key)
    T = vol.territory()
    cards = [map_card(vol, e, T) for e in pair]
    w = max(c.shape[1] for c in cards)
    cards = [cv2.copyMakeBorder(c, 0, 8, 0, w - c.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for c in cards]
    sheet = cv2.resize(np.vstack(cards), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    code = pair[0][3]
    cv2.imwrite(str(Path(out) / f"{code}.png"), sheet)
    return code


_VOLS = {}


def _vol(key):
    if key not in _VOLS:
        _VOLS[key] = VOLUMES[key]()
    return _VOLS[key]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vol1-3")
