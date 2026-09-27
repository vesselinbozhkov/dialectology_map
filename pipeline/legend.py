"""OCR of legend texts next to the legend swatches (draft, needs proofreading)."""
import subprocess
import tempfile

import cv2
import numpy as np


def legend_crops(al, swatches, height=420):
    """Each legend item is a vertical strip: swatch at the bottom, text rotated
    90° running upwards from it. Return one horizontal (rotated back) crop per swatch."""
    xs = sorted({s["bbox"][0] for s in swatches})
    crops = []
    for s in swatches:
        x, y, w, h = s["bbox"]
        right = min([v for v in xs if v > x + 5] + [x + w + 70]) - 3
        crop = al[max(0, y - height):y - 2, max(0, x - 10):right]
        crops.append(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE))
    return crops


def ocr(img, psm=6):
    big = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, big)
        r = subprocess.run(["tesseract", f.name, "-", "-l", "bul", "--psm", str(psm)],
                           capture_output=True, text=True, env={"OMP_THREAD_LIMIT": "1"})
    return " ".join(r.stdout.split())
