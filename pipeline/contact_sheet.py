"""Tile the per-map QA images (segmentation half only) into overview sheets."""
import sys
from pathlib import Path

import cv2
import numpy as np

QA = Path(__file__).resolve().parent.parent / "data" / "out" / "qa"


def main(per_sheet=12, cols=4):
    files = sorted(QA.glob("M[0-9][0-9][0-9].jpg"))
    for s in range(0, len(files), per_sheet):
        tiles = []
        for f in files[s:s + per_sheet]:
            im = cv2.imread(str(f))
            seg = im[:, im.shape[1] // 2:][60:720, 60:560]  # right half, map area
            seg = cv2.resize(seg, (330, 436))
            cv2.putText(seg, f.stem, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            tiles.append(seg)
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        out = QA / f"sheet_{s // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(out)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
