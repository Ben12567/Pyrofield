"""Gridded crop of a map for reading full-resolution pixel coordinates.

usage: python -I grid.py <image> <x0> <y0> <x1> <y1> <step> <out.png> [anchors.json]
anchors.json (optional): {"curves": {"name": [[x, y], ...]}, "snapped": {"name": [[x, y], ...]}} drawn on top.
"""
import json
import sys

from PIL import Image, ImageDraw

img, x0, y0, x1, y1, step, out = sys.argv[1], *map(int, sys.argv[2:7]), sys.argv[7]
im = Image.open(img).convert("RGB").crop((x0, y0, x1, y1))
d = ImageDraw.Draw(im)
for x in range((x0 // step + 1) * step, x1, step):
    d.line([(x - x0, 0), (x - x0, y1 - y0)], fill=(0, 170, 255), width=2)
    d.text((x - x0 + 3, 3), str(x), fill=(0, 0, 200))
for y in range((y0 // step + 1) * step, y1, step):
    d.line([(0, y - y0), (x1 - x0, y - y0)], fill=(0, 170, 255), width=2)
    d.text((3, y - y0 + 3), str(y), fill=(0, 0, 200))
if len(sys.argv) > 8:
    A = json.load(open(sys.argv[8], encoding="utf-8"))
    pal = [(0, 0, 0), (0, 90, 220), (0, 150, 60), (220, 120, 0), (160, 0, 160), (0, 170, 170)]
    for i, (name, pts) in enumerate(A.get("snapped", {}).items()):
        c = pal[i % len(pal)]
        for x, y in pts:
            d.ellipse((x - x0 - 3, y - y0 - 3, x - x0 + 3, y - y0 + 3), fill=c)
    for i, (name, pts) in enumerate(A.get("curves", {}).items()):
        c = pal[i % len(pal)]
        d.line([(x - x0, y - y0) for x, y in pts], fill=c, width=1)
        d.text((pts[0][0] - x0 + 6, pts[0][1] - y0 + 6), name, fill=c)
s = 1400 / max(im.size)
if s < 1:
    im = im.resize((int(im.size[0] * s), int(im.size[1] * s)))
im.save(out)
print("scale", round(min(1, s), 4))
