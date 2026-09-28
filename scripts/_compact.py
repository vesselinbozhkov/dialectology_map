"""Helper: append compact legend lines (id|title|item;item;...|norm indices|note) to the notes file.
Items are separated by a semicolon with no space after it; "; " inside an item is text.

Item prefixes: `# ` group heading, `x:` item without a swatch, `N=` explicit swatch number."""
import re
import sys
out = []
for line in sys.stdin.read().strip().splitlines():
    parts = line.split('|')
    mid, title, items, lit = parts[:4]
    note = parts[4] if len(parts) > 4 else ""
    lit = {int(x) for x in lit.split(',') if x}
    out.append(f"\n@{mid} | {title}")
    i = 0
    for t in re.split(r';(?! )', items):  # '; ' inside an item is text
        t = t.strip()
        if t.startswith('#'):  # group heading
            out.append(t)
            continue
        if t.startswith('x:'):
            out.append(f"x | {t[2:].strip()} | -")
            continue
        i += 1
        if '=' in t[:3] and t.split('=', 1)[0].isdigit():  # explicit swatch number
            i, t = int(t.split('=', 1)[0]), t.split('=', 1)[1]
        n = '?' if t.startswith('Н –') or t.startswith('У –') else ('+' if i in lit else '-')
        out.append(f"{i} | {t} | {n}")
    if note:
        out.append(f"! {note}")
open(sys.argv[1], 'a').write("\n".join(out) + "\n")
