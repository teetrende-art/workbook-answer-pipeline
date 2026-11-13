#!/usr/bin/env python3
"""Generic answer-line detector for photographed worksheets.

Works on coloured rules too (this book prints its answer lines in green), which
the earlier pale-blue detector missed. A line is accepted only if it is long and
*isolated*: little ink directly above or below it.
"""
import numpy as np
from PIL import Image


def find_lines(path, min_len=300, max_len=1200, thr=232, pad=19, max_neighbour=0.05):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    ink = (a[:, :, 0] < thr) | (a[:, :, 1] < thr) | (a[:, :, 2] < thr)
    H, W = ink.shape

    def runs(row):
        out, x = [], 0
        while x < W:
            if not row[x]:
                x += 1
                continue
            x0 = x
            while x < W and row[x]:
                x += 1
            out.append((x0, x))
        return out

    cands = []
    for y in range(pad, H - pad):
        for x0, x1 in runs(ink[y]):
            # bridge gaps up to 6 px (faint ink splits a rule)
            if x1 - x0 < min_len:
                continue
            gap_end = x1
            while True:
                nxt = ink[y, gap_end:gap_end + 7]
                if nxt.any():
                    idx = int(np.argmax(nxt))
                    if idx <= 6:
                        s2 = gap_end + idx
                        e2 = s2
                        while e2 < W and ink[y, e2]:
                            e2 += 1
                        gap_end = e2
                        continue
                break
            L = gap_end - x0
            if not (min_len <= L <= max_len):
                continue
            above = ink[y - pad:y - pad + 4, x0:gap_end]
            below = ink[y + pad - 3:y + pad + 1, x0:gap_end]
            if above.mean() <= max_neighbour and below.mean() <= max_neighbour:
                cands.append((y, x0, gap_end, L))

    cands.sort()
    out = []
    for y, x0, x1, L in cands:
        merged = False
        for o in out:
            ov = min(o["x1"], x1) - max(o["x0"], x0)
            if abs(o["y"] - y) <= 7 and (ov > 0.4 * min(o["len"], L) or abs(o["x0"] - x0) < 25):
                o["x0"] = min(o["x0"], x0)
                o["x1"] = max(o["x1"], x1)
                o["len"] = o["x1"] - o["x0"]
                merged = True
                break
        if not merged:
            out.append({"y": y, "x0": x0, "x1": x1, "len": L})
    # collapse near-duplicates at the same y
    dedup = []
    for l in sorted(out, key=lambda d: d["y"]):
        if dedup and abs(dedup[-1]["y"] - l["y"]) <= 10:
            if l["len"] > dedup[-1]["len"]:
                dedup[-1] = l
        else:
            dedup.append(l)

    # a writing block is a run of evenly spaced rules: drop anything with no
    # neighbour within 1.6x the median gap, and anything in the page header
    if len(dedup) >= 4:
        gaps = [dedup[i + 1]["y"] - dedup[i]["y"] for i in range(len(dedup) - 1)]
        med = float(np.median(gaps))
        limit = 1.6 * med
        keep = []
        for i, l in enumerate(dedup):
            prev_gap = dedup[i]["y"] - dedup[i - 1]["y"] if i > 0 else 1e9
            next_gap = dedup[i + 1]["y"] - dedup[i]["y"] if i < len(dedup) - 1 else 1e9
            if min(prev_gap, next_gap) <= limit:
                keep.append(l)
        dedup = keep

    out = [l for l in dedup if l["y"] > H * 0.06]
    out.sort(key=lambda d: (round(d["y"] / 45), d["x0"]))
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        ls = find_lines(p)
        print(p, len(ls), [l["y"] for l in ls])
