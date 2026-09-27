"""Build searchable PDFs (with an OCR text layer) from the downloaded page images.

Usage: python scripts/build_pdfs.py [bookId ...]   (default: bda bda4)
Reads  data/raw/<bookId>/NNNN.png
Writes data/ocr/<bookId>/NNNN.{pdf,txt}  (per-page OCR, cached between runs)
       data/pdf/<bookId>.pdf             (merged, searchable)
       data/pdf/parts/<bookId>_pNNNN-NNNN.pdf  (same, split into <27 MB parts)
Needs: tesseract with the 'bul' language pack, Pillow, PyMuPDF.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf as fitz
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data"
DPI = 150  # the scans are 1200 px wide at 150 dpi


def ocr_page(png, ocr_dir):
    base = ocr_dir / png.stem
    if base.with_suffix(".pdf").exists() and base.with_suffix(".txt").exists():
        return
    jpg = base.with_suffix(".jpg")
    Image.open(png).convert("RGB").save(jpg, "JPEG", quality=85, dpi=(DPI, DPI))
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    subprocess.run(["tesseract", str(jpg), str(base), "-l", "bul", "--dpi", str(DPI), "pdf", "txt"],
                   check=True, capture_output=True, env=env)
    jpg.unlink()


def build(book_id):
    pages = sorted((ROOT / "raw" / book_id).glob("[0-9][0-9][0-9][0-9].png"))
    ocr_dir = ROOT / "ocr" / book_id
    ocr_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        list(pool.map(lambda p: ocr_page(p, ocr_dir), pages))

    pdf_dir = ROOT / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    page_pdfs = [ocr_dir / f"{p.stem}.pdf" for p in pages]
    write_pdf(page_pdfs, pdf_dir / f"{book_id}.pdf")

    # Also split into parts small enough to send/share (the full books are 70+ MB).
    part, size = [], 0
    for f in page_pdfs:
        if part and size + f.stat().st_size > PART_LIMIT:
            write_part(book_id, part, pdf_dir)
            part, size = [], 0
        part.append(f)
        size += f.stat().st_size
    write_part(book_id, part, pdf_dir)


PART_LIMIT = 27_000_000  # bytes


def write_part(book_id, files, pdf_dir):
    write_pdf(files, pdf_dir / "parts" / f"{book_id}_p{files[0].stem}-{files[-1].stem}.pdf")


def write_pdf(files, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = fitz.open()
    for f in files:
        with fitz.open(f) as page_pdf:
            merged.insert_pdf(page_pdf)
    merged.save(out, garbage=3, deflate=True)
    print(f"{out}: {len(files)} pages, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    for b in sys.argv[1:] or ["bda", "bda4"]:
        build(b)
