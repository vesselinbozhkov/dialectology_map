"""Convert hand-written legend notes into JSONL.

Notes format (catalog/legends/<volume>.notes), UTF-8:

    @Ф 1 | Застъпници на стб. ъ в коренна сричка в думите дъжд, сън < стб. дъждь, сънъ
    # I. Без първично изясняване на стб. ъ > о / 1. Гласна ъ        <- group heading
    1 | ъ – дъш, сън | +                                             <- swatch no. | text | norm
    2 | А: а – даш, сан | -
    x | щриховка: … | ?                                              <- item without swatch
    ! бележка                                                        <- note

norm: + (литературна форма), - (не), ? (неприложимо).
Usage: python scripts/legends_from_notes.py vol1-3
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NORM = {"+": True, "-": False, "?": None}


def parse(path):
    maps, cur, group = [], None, ""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):
            mid, _, title = line[1:].partition("|")
            cur = {"id": mid.strip(), "title": title.strip(), "items": [], "extra": [], "note": ""}
            maps.append(cur)
            group = ""
        elif line.startswith("#"):
            group = line[1:].strip()
        elif line.startswith("!"):
            cur["note"] = (cur["note"] + " " + line[1:].strip()).strip()
        else:
            parts = [p.strip() for p in line.split("|")]
            idx, text = parts[0], parts[1]
            norm = NORM[parts[2]] if len(parts) > 2 and parts[2] else None
            if idx == "x":
                cur["extra"].append({"group": group, "text": text, "norm": norm})
            else:
                cur["items"].append({"i": int(idx), "group": group, "text": text, "norm": norm})
    return maps


def main(key):
    src = ROOT / "catalog" / "legends" / f"{key}.notes"
    maps = parse(src)
    out = src.with_suffix(".jsonl")
    out.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in maps))
    print(f"{out}: {len(maps)} maps")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vol1-3")
