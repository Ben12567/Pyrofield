"""Snap hand-placed anchor polylines to the red isochrone lines of an Annaburroo isochrone map.

The polyline through the anchors is densified every 5 px; at each sample the red runs along the
local normal within +-search_px are found. The first sample takes the run nearest the anchor
polyline; later samples take the run nearest the previous snapped point if it lies within
jump_px (+5 px per preceding miss), so that the trace follows one drawn line where neighbouring
isochrones come close; after more than 6 misses the trace re-acquires the run nearest the polyline.

usage (stand-alone): python -I snap_isochrones.py <map.jpg> <anchors.json> <out.json> [search_px] [jump_px]
"""
import json
import sys

import numpy as np
from PIL import Image


def red_mask(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 140) & (r - g > 45) & (r - b > 30)


def snap_curve(red, pts, R=22.0, J=10.0):
    H, W = red.shape
    P = np.array(pts, float)
    seg = np.diff(P, axis=0)
    L = np.hypot(*seg.T)
    samples, normals = [], []
    for p, s, l in zip(P[:-1], seg, L):
        n = int(max(1, l // 5))
        t = s / l
        for i in range(n):
            samples.append(p + s * i / n)
            normals.append(np.array([-t[1], t[0]]))
    offs = np.arange(-R, R + 1, 1.0)
    snapped, prev, misses = [], None, 0
    for q, nv in zip(samples, normals):
        xs, ys = q[0] + offs * nv[0], q[1] + offs * nv[1]
        ok = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
        v = np.zeros_like(offs, dtype=bool)
        v[ok] = red[ys[ok].astype(int), xs[ok].astype(int)]
        runs, start = [], None
        for i, f in enumerate(np.append(v, False)):
            if f and start is None:
                start = i
            if not f and start is not None:
                runs.append((start, i - 1))
                start = None
        cands = [q + offs[(s0 + s1) // 2] * nv for s0, s1 in runs if s1 - s0 >= 2]
        if not cands:
            misses += 1
            continue
        if prev is None or misses > 6:
            best = min(cands, key=lambda c: np.hypot(*(c - q)))
        else:
            best = min(cands, key=lambda c: np.hypot(*(c - prev)))
            if np.hypot(*(best - prev)) > J + 5 * misses:
                misses += 1
                continue
        snapped.append([float(best[0]), float(best[1])])
        prev, misses = best, 0
    return snapped


if __name__ == "__main__":
    img, anc, out = sys.argv[1], sys.argv[2], sys.argv[3]
    R = float(sys.argv[4]) if len(sys.argv) > 4 else 22.0
    J = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    red = red_mask(img)
    A = json.load(open(anc, encoding="utf-8"))
    res = {name: snap_curve(red, pts, R, J) for name, pts in A["curves"].items()}
    for name, s in res.items():
        print(name, "anchors", len(A["curves"][name]), "snapped", len(s))
    json.dump({"curves": A["curves"], "snapped": res}, open(out, "w", encoding="utf-8"), indent=0)
