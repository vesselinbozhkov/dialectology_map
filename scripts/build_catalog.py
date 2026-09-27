"""Build a draft catalogue of all atlas maps from the OCR of the tables of contents.

Usage: python scripts/build_catalog.py
Reads  data/ocr/<bookId>/NNNN.txt (TOC pages)
Writes catalog/maps.csv  (id, volume, part, number, page, file_page, title)

`page` is the printed page number; `file_page` is the page index in the online
scan (data/raw/<bookId>/NNNN.png), empty when the scan lacks that page.

Titles come from OCR and need proofreading; page numbers are computed from
per-part offsets verified against the TOC (one map per page).
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARTS = {
    # letter: (bookId, volume, part name, number of maps, page of map 1 minus 1)
    "Ф": ("bda", "I-III", "Фонетика", 172, 58),
    "А": ("bda", "I-III", "Акцентология", 88, 284),
    "Л": ("bda", "I-III", "Лексика", 108, 394),
    "М": ("bda4", "IV", "Морфология", 145, 26),
}
TOC_PAGES = {"bda": range(5, 17), "bda4": range(5, 11)}
# printed pages absent from the online scan (verified from page headers);
# later files are shifted: file index = printed page - pages missing before it
SCAN_GAPS = {"bda": [72, 73], "bda4": []}


def file_page(book_id, printed):
    gaps = SCAN_GAPS[book_id]
    if printed in gaps:
        return ""
    return printed - sum(g < printed for g in gaps)


# OCR renders "№" as Хе/Хо/Ф/Ж/Ме... and sometimes Latin M
ENTRY = re.compile(r"Карта\s*\S{0,3}\s*([ФАЛМM])\s*(\d{1,3})\s+(.*)")
# OCR turns dot leaders into long letter runs ("нненнеенен..."); drop such tokens
JUNK_TOKEN = re.compile(r"^\S{22,}$|(?:нн|ене|еее|иии|ени){2}")
TRAIL = re.compile(r"[\s.…,:;+\-]*\d{0,3}\s*$")


def toc_text(book_id):
    pages = []
    for n in TOC_PAGES[book_id]:
        f = ROOT / "data" / "ocr" / book_id / f"{n:04d}.txt"
        if f.exists():
            pages.append(f.read_text())
    return "\n".join(pages)


def parse(book_id):
    titles, current = {}, None
    for line in toc_text(book_id).splitlines():
        line = line.strip()
        m = ENTRY.search(line)
        if m:
            letter = "М" if m.group(1) == "M" else m.group(1)
            current = (letter, int(m.group(2)))
            titles[current] = m.group(3)
        elif current and line and not line.startswith("БЪЛГАРСКИ"):
            titles[current] += " " + line
    return {k: clean(v) for k, v in titles.items()}


def clean(title):
    words = [w for w in title.split() if not JUNK_TOKEN.search(w)]
    return TRAIL.sub("", " ".join(words))


def main():
    titles = {}
    for book_id in TOC_PAGES:
        titles.update(parse(book_id))
    out = ROOT / "catalog" / "maps.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "book_id", "volume", "part", "number", "page", "file_page", "title_ocr"])
        missing = 0
        for letter, (book_id, vol, part, count, offset) in PARTS.items():
            for n in range(1, count + 1):
                title = titles.get((letter, n), "")
                missing += not title
                w.writerow([f"{letter} {n}", book_id, vol, part, n, offset + n,
                            file_page(book_id, offset + n), title])
    print(f"{out}: {sum(p[3] for p in PARTS.values())} maps, {missing} without OCR title")


if __name__ == "__main__":
    main()
