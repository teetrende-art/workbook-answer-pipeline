#!/usr/bin/env python3
"""Find every place a pupil is meant to write, in reading order.

Two kinds exist in real worksheets:
  * ruled lines under a question  (long, isolated)
  * gaps inside a sentence        (short rule with printed text on both sides)
Both are collected, columns are detected so a two-column page reads left-to-right,
and the result is a single ordered list the model can answer position by position.
"""
import numpy as np
from PIL import Image


def _runs(row):
    out, x = [], 0
    n = len(row)
    while x < n:
        if not row[x]:
            x += 1
            continue
        x0 = x
        while x < n and row[x]:
            x += 1
        out.append((x0, x))
    return out


def _ink(a, thr=232):
    return (a[:, :, 0] < thr) | (a[:, :, 1] < thr) | (a[:, :, 2] < thr)


def columns(ink, min_gutter_frac=0.02):
    """Split the page at a vertical white gutter.

    Measured on the question area only: headers and illustrations often run
    across the middle and hide the gutter.
    """
    H, W = ink.shape
    band = ink[int(H * 0.25):int(H * 0.97), :]
    colsum = band.sum(axis=0) / max(1, band.shape[0])
    gutters, run = [], None
    for x in range(int(W * 0.30), int(W * 0.70)):
        if colsum[x] < 0.005:
            run = x if run is None else run
        else:
            if run is not None and x - run > W * min_gutter_frac:
                gutters.append((run, x))
            run = None
    if run is not None and int(W * 0.75) - run > W * min_gutter_frac:
        gutters.append((run, int(W * 0.75)))
    if not gutters:
        return [0, W]
    g = max(gutters, key=lambda t: t[1] - t[0])
    return [0, (g[0] + g[1]) // 2, W]


def find_targets(path, min_line=300, max_line=1400, min_gap=55, max_gap=300, pad=11, thr=232):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    ink = _ink(a, thr)
    H, W = ink.shape
    long_c, gap_c = [], []

    for y in range(pad, H - pad):
        for x0, x1 in _runs(ink[y]):
            L = x1 - x0
            # bridge tiny breaks in a faint rule
            if x1 < W and L < min_line:
                j = x1
                while j < min(W, x1 + 6):
                    if ink[y, j]:
                        j += 1
                        continue
                    break
                if j > x1:
                    k = j
                    while k < W and ink[y, k]:
                        k += 1
                    x1, L = k, k - x0

            above = ink[y - pad:y - pad + 4, x0:x1].mean()
            below = ink[y + pad - 3:y + pad + 1, x0:x1].mean()
            if above > 0.06 or below > 0.06:
                continue

            if min_line <= L <= max_line:
                long_c.append({"y": y, "x0": x0, "x1": x1, "kind": "line", "len": L})
            elif min_gap <= L <= max_gap:
                # a gap sits between printed words on the same baseline
                left = ink[y - 6:y + 7, max(0, x0 - 45):max(0, x0 - 4)]
                right = ink[y - 6:y + 7, min(W, x1 + 4):min(W, x1 + 45)]
                if left.any() and right.any():
                    gap_c.append({"y": y, "x0": x0, "x1": x1, "kind": "gap", "len": L})

    def dedupe(items, tol=9):
        out = []
        for it in sorted(items, key=lambda d: (d["y"], d["x0"])):
            dup = False
            for o in out:
                ov = min(o["x1"], it["x1"]) - max(o["x0"], it["x0"])
                if abs(o["y"] - it["y"]) <= tol and ov > 0.4 * min(o["len"], it["len"]):
                    if it["len"] > o["len"]:
                        o.update(it)
                    dup = True
                    break
            if not dup:
                out.append(dict(it))
        return out

    long_c = dedupe(long_c)
    gap_c = dedupe(gap_c, tol=7)

    # drop gaps that sit on a long line (they are part of it)
    gap_c = [g for g in gap_c
             if not any(abs(g["y"] - l["y"]) <= 8 and g["x0"] >= l["x0"] - 5 and g["x1"] <= l["x1"] + 5
                        for l in long_c)]

    allc = long_c + gap_c
    cols = columns(ink)
    def colof(c):
        cx = (c["x0"] + c["x1"]) / 2
        for i in range(len(cols) - 1):
            if cols[i] <= cx < cols[i + 1]:
                return i
        return 0

    for c in allc:
        c["col"] = colof(c)
    allc.sort(key=lambda c: (c["col"], round(c["y"] / 12), c["x0"]))
    return allc


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        t = find_targets(p)
        kinds = {}
        for x in t:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        print(p, len(t), kinds, "cols", columns(_ink(np.asarray(Image.open(p).convert("RGB")).astype(int))))
        for x in t[:12]:
            print("   ", x["kind"], x["y"], x["x0"], x["x1"], "col", x["col"])
