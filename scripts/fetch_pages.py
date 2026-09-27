"""Download all page images of the BDA volumes from the IBL e-library.

Usage: python scripts/fetch_pages.py [bookId ...]   (default: bda bda4)
Pages are saved to data/raw/<bookId>/<NNNN>.png; existing files are skipped.
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://ibl.bas.bg/lib/index.php"
ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"


def get(url, retries=8):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # network hiccups: back off and retry
            if attempt == retries - 1:
                raise
            print(f"  retry {url}: {e}", file=sys.stderr)
            time.sleep(min(2 ** (attempt + 1), 120))


def fetch_book(book_id):
    out = ROOT / book_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(get(f"{BASE}?action=metadata&bookId={book_id}"))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    pages = [p for spread in manifest["data"]["brOptions"]["data"] for p in spread]

    def one(p):
        dest = out / f"{int(p['leafNum']):04d}.png"
        if dest.exists() and dest.stat().st_size > 0:
            return
        tmp = dest.with_suffix(".part")
        try:
            tmp.write_bytes(get(f"https://ibl.bas.bg/lib/{p['uri']}"))
        except Exception as e:  # leave it missing; a rerun picks it up
            print(f"FAILED {book_id} {dest.name}: {e}", flush=True)
            return
        tmp.rename(dest)
        print(f"{book_id} {dest.name}", flush=True)

    # The server renders some books slowly (~25 s/page) and answers 504 when
    # overloaded, so keep concurrency low.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, pages))
    print(f"{book_id}: {len(pages)} pages done")


if __name__ == "__main__":
    for b in sys.argv[1:] or ["bda", "bda4"]:
        fetch_book(b)
