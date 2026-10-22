"""Render a docx with LibreOffice and record the printed page number of every heading, figure and table.
Usage: python measure.py <report.docx> <pages.json>"""

import json
import os
import re
import subprocess
import sys

src, out = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
d = os.path.dirname(os.path.abspath(src))
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", d, src], check=True, capture_output=True, timeout=600)
pdf = os.path.splitext(os.path.abspath(src))[0] + ".pdf"
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
pages = []
for i in range(1, n + 1):
    t = subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), pdf, "-"], capture_output=True, text=True).stdout
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    label = lines[-1] if lines and re.fullmatch(r"[ivxlc]+|\d+", lines[-1]) else None
    norm = re.sub(r"\s+", " ", " ".join(lines)).upper()
    top = re.sub(r"\s+", " ", " ".join(lines[:3])).upper()
    pages.append((label, norm, top))

struct = json.load(open(os.path.join(here, "_structure.json")))
res, missing = {}, []


def find(needle, decimal, at_top=False):
    needle = re.sub(r"\s+", " ", needle).upper()
    for label, norm, top in pages:
        if label is None or label.isdigit() != decimal:
            continue
        if (top.startswith(needle) if at_top else needle in norm):
            return label
    return None


for h in ("CONTENTS", "LIST OF FIGURES", "LIST OF TABLES", "LIST OF SYMBOLS AND ABBREVIATIONS"):
    v = find(h, False, at_top=True)
    res["H|" + h] = v
for lvl, num, title in struct["heads"]:
    key = (num + " " + title).strip()
    v = find(key, True, at_top=(lvl == 1))
    if v is None and lvl > 1:
        v = find(num + " " + title[:40], True)
    if v is None:
        missing.append(key)
    res["H|" + key] = v
for num, cap in struct["figs"]:
    v = find(f"Figure {num}:", True)
    res["F|" + num] = v
    if v is None:
        missing.append("Figure " + num)
for num, cap in struct["tabs"]:
    v = find(f"Table {num}:", True)
    res["T|" + num] = v
    if v is None:
        missing.append("Table " + num)
json.dump(res, open(out, "w"), indent=0)
roman = sum(1 for l, _, _ in pages if l and not l.isdigit())
dec = [int(l) for l, _, _ in pages if l and l.isdigit()]
print(f"pages {n}: unnumbered {sum(1 for l, _, _ in pages if l is None)}, roman {roman}, arabic {len(dec)} (last {max(dec) if dec else None})")
print("missing:", missing)
