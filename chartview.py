#!/usr/bin/env python3
"""Give the model a zoomed view of the page's chart area.

The first live run read a bar 10 units low because the whole page was downscaled
to fit one image. Cropping the content area *above the answer lines* and
upscaling it gives the model a chart it can actually measure.
"""
import numpy as np
from PIL import Image

from lines import find_lines


def content_crop(path, out_path, lines=None, scale=2.0):
    """Crop the top content area (above the first answer line) and upscale it."""
    img = Image.open(path).convert("RGB")
    W, H = img.size
    if lines is None:
        lines = find_lines(path)
    bottom = min([l["y"] for l in lines], default=H * 0.75) - 10
    top = int(H * 0.04)
    left, right = int(W * 0.04), int(W * 0.96)
    if bottom - top < H * 0.12 or right - left < W * 0.5:
        return None
    crop = img.crop((left, top, right, max(top + 40, bottom)))
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    crop.save(out_path)
    return out_path


def has_chart(text, lines):
    import re
    if re.search(r"\b(graph|chart|diagram|table)\b", text or "", re.I):
        return True
    # many parallel rules in the content area is also a grid
    if lines:
        top_line = min(l["y"] for l in lines)
        above = [l for l in lines if l["y"] < top_line - 200]
        if len(above) >= 2:
            return True
    return False


if __name__ == "__main__":
    import sys, json, time
    sys.path.insert(0, "/root/answerkey")
    import worker
    p = sys.argv[1]
    t0 = time.time()
    text = worker.ocr(p)
    lines = find_lines(p)
    c = content_crop(p, p.replace(".png", "-chart.png"), lines)
    print("chart:", has_chart(text, lines), "crop:", c, f"{time.time()-t0:.1f}s")
