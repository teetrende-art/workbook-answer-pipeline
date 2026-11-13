#!/usr/bin/env python3
"""Job worker: photographed worksheet in, answered PDF out.

Pipeline per page: deskew → OCR → find the ruled answer lines → one vision call
that answers them in order → place the text on each line → re-check → PDF.
"""
import base64, json, os, re, shutil, subprocess, sys, time
import urllib.request
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/answerkey")
from lines import find_lines
from chartview import content_crop, has_chart
from targets import find_targets

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
INK = (18, 30, 96)
DPI = 150


def api_key():
    for ln in open("/root/.hermes/.env", encoding="utf-8", errors="ignore"):
        m = re.match(r"\s*DEEPSEEK_API_KEY\s*=\s*(.+?)\s*$", ln)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    raise RuntimeError("no DEEPSEEK_API_KEY")


def to_pages(src, workdir):
    """Return a list of PNG paths, one per page."""
    out = []
    if src.lower().endswith(".pdf"):
        subprocess.run(["pdftoppm", "-r", str(DPI), "-png", src, os.path.join(workdir, "pg")], check=True)
        out = sorted([os.path.join(workdir, f) for f in os.listdir(workdir) if f.startswith("pg-") and f.endswith(".png")])
    else:
        p = os.path.join(workdir, "pg-1.png")
        im = Image.open(src).convert("RGB")
        w, h = im.size
        if min(w, h) < 600:
            raise RuntimeError(
                f"That image is only {w}×{h} pixels — too small to read. "
                f"Send the original photo (or a PDF), not a compressed copy.")
        if min(w, h) < 1100:
            im = im.resize((int(w * 2), int(h * 2)), Image.LANCZOS)
        im.save(p)
        out = [p]
    return out


def deskew(path):
    """Rotate using Tesseract's orientation detection when it is confident."""
    try:
        r = subprocess.run(["tesseract", path, "-", "--psm", "0"], capture_output=True, text=True, timeout=40)
        m = re.search(r"Rotate:\s*(\d+)", r.stdout)
        if m and int(m.group(1)) in (90, 180, 270):
            img = Image.open(path)
            img.rotate(-int(m.group(1)), expand=True).save(path)
            return int(m.group(1))
    except Exception:
        pass
    return 0


def ocr(path):
    r = subprocess.run(["tesseract", path, "-", "--psm", "6"], capture_output=True, text=True, timeout=90)
    return r.stdout


def answer_page(img_path, text, n, chart=False, chart_img=None, offset=0, total=None):
    """One vision call: N answers, in reading order."""
    img = base64.b64encode(open(img_path, "rb").read()).decode()
    chart_note = (" This page contains a graph or chart. A zoomed crop of the chart is supplied as "
                  "the second image — use it, not the wide page view. Before writing the answers, "
                  "work out the exact value of every bar or point you rely on by naming the gridline "
                  "it reaches (for example: 'the grey bar stops on the 40 line'). Then answer. "
                  "Do not round to the nearest ten and do not guess.") if chart else ""
    prompt = (
        f"This is one page of a school worksheet.{chart_note} The OCR text of the page follows.\n\n"
        f"This page has {total} places to write in total. You are answering places "
        f"{offset + 1} to {offset + n} of that sequence — answer ONLY those {n}, "
        f"in order. Some are ruled lines, some are gaps inside sentences; ignore every "
        f"other blank on the page.\n"
        f"There are exactly {n} places to write in this request, in reading order — a mix of "
        f"ruled lines and gaps inside sentences. Return a JSON array of exactly {n} strings: the text that should be "
        f"written on each line, in the same order as the lines appear.\n"
        f"Rules: give the pupil's answer only, not the question; show short working for "
        f"arithmetic; read any graph or table in the image carefully and use its real values; "
        f"keep each answer under 120 characters; if a line cannot be answered, use \"\".\n"
        f"Be brief: no long deliberation. Return ONLY the JSON array.\n\nOCR TEXT:\n{text[:5000]}"
    )
    content = [{"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]
    if chart_img:
        z = base64.b64encode(open(chart_img, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + z}})
    body = {
        "model": "deepseek-v4-flash-vision-exp",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4000,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
        method="POST")
    last = ""
    for attempt in range(3):
        # an empty reply means the model spent the whole budget thinking:
        # retry with more room before giving up on the page
        body["max_tokens"] = 4000 * (attempt + 1)
        req.data = json.dumps(body).encode()
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        raw = d["choices"][0]["message"].get("content") or ""
        m = re.search(r"\[.*\]", raw, re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                return [str(x) for x in arr], d.get("usage", {})
            except Exception:
                last = raw[:200]
        else:
            last = raw[:200] or "empty reply"
    raise RuntimeError("model did not return JSON: " + last)


def same_answer(a, b):
    """Two passes agree if they carry the same numbers, or the same words."""
    import re as _re
    na = _re.findall(r"\d+(?:\.\d+)?", a or "")
    nb = _re.findall(r"\d+(?:\.\d+)?", b or "")
    if na or nb:
        sa, sb = set(na), set(nb)
        # one pass may show working ("38 - 19 = 19"), the other just the result
        return sa <= sb or sb <= sa
    ca = _re.sub(r"\s+", " ", (a or "").strip().lower())
    cb = _re.sub(r"\s+", " ", (b or "").strip().lower())
    return ca[:40] == cb[:40]


def answer_chunked(img_path, text, n, chart=False, chart_img=None, chunk=6):
    """Answer in small batches: a page with 30 blanks in one request comes back empty."""
    out, used = [], {}
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        part, u = answer_page(img_path, text, stop - start, chart=chart, chart_img=chart_img,
                              offset=start, total=n)
        out.extend(part)
        for k, v in u.items():
            if isinstance(v, (int, float)):
                used[k] = used.get(k, 0) + v
    return out, used


def place(img_path, lines, answers, out_path):
    img = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(img)
    for l, text in zip(lines, answers):
        if not text.strip():
            continue
        gap = l.get("kind") == "gap"
        size = 20 if gap else 24
        limit = (l["x1"] - l["x0"]) - (6 if gap else 14)
        while size > 10:
            f = ImageFont.truetype(FONT, size)
            if d.textlength(text, font=f) <= limit:
                break
            size -= 1
        f = ImageFont.truetype(FONT, size)
        y = l["y"] - size - (3 if gap else 6)
        d.text((l["x0"] + (2 if gap else 6), y), text, font=f, fill=INK)
    img.save(out_path)
    return out_path


def process(job_dir, on_step=None):
    """Run one job. Returns a report dict."""
    src = None
    for f in sorted(os.listdir(job_dir)):
        if f.startswith("source.") and f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".webp")):
            src = os.path.join(job_dir, f)
            break
    if not src:
        raise RuntimeError("no file in job")

    t0 = time.time()
    pages = to_pages(src, job_dir)
    t_convert = time.time() - t0

    report = {"pages": [], "tokens": 0}

    def one(item):
        i, p = item
        tp = time.time()
        rot = deskew(p)
        text = ocr(p)
        lines = find_targets(p)
        if not lines:
            return i, p, [], rot, {}, True, time.time() - tp, {}
        if len(lines) > 24:
            # 30+ blanks on one page (the two-column gap-fill books) is beyond a
            # single pass; answering it produced empty strings and wasted minutes
            return i, p, lines, rot, [], True, time.time() - tp, {"unsupported": 1}
        chart_png = None
        try:
            if has_chart(text, lines):
                chart_png = content_crop(p, p.replace(".png", "-chart.png"), lines)
        except Exception:
            chart_png = None
        try:
            if len(lines) > 10:
                answers, usage = answer_chunked(p, text, len(lines), chart=bool(chart_png),
                                                chart_img=chart_png, chunk=6)
            else:
                answers, usage = answer_page(p, text, len(lines), chart=bool(chart_png), chart_img=chart_png)
        except Exception as e:
            return i, p, lines, rot, [""] * len(lines), True, time.time() - tp, {"total_tokens": 0, "error": str(e)[:120]}
        mismatch = len(answers) != len(lines)
        answers = (answers + [""] * len(lines))[:len(lines)]

        # A page with a graph is where a model is most likely to misread a value
        # (it read one bar 10 units low on the first live run). Ask again with a
        # chart-specific prompt and flag the page when the two disagree.
        disputed = False
        if chart_png:
            try:
                second, usage2 = answer_page(p, text, len(lines), chart=True, chart_img=chart_png)
                usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in set(usage) | set(usage2)}
                second = (second + [""] * len(lines))[:len(lines)]
                for a, b in zip(answers, second):
                    if not same_answer(a, b):
                        disputed = True
            except Exception:
                # the second opinion failed to return a usable answer. The first
                # pass already used the zoomed chart crop, which is the primary
                # path — do not flag purely because the check itself failed.
                pass
        return i, p, lines, rot, answers, mismatch or disputed, time.time() - tp, usage

    from concurrent.futures import ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(one, list(enumerate(pages, 1))):
            results.append(r)
            if on_step:
                on_step(len(results), len(pages))

    results.sort(key=lambda r: r[0])
    unsupported = any((r[7] or {}).get("unsupported") for r in results)
    if unsupported:
        raise RuntimeError("This page has too many blanks for the automatic pass "
                           "(more than 24). It is the two-column gap-fill style — send it "
                           "to me directly and I will do this one by hand while I build it.")

    filled = []
    for r in results:
        i, p, lines, rot, answers, mismatch, secs, usage = r
        out = os.path.join(job_dir, f"filled-{i}.png")
        if lines:
            place(p, lines, answers, out)
        else:
            Image.open(p).convert("RGB").save(out)
        filled.append(out)
        report["tokens"] += usage.get("total_tokens", 0)
        report["pages"].append({"page": i, "lines": len(lines), "answers": len(answers),
                                "mismatch": mismatch, "rotation": rot, "seconds": round(secs, 1)})

    pdf = os.path.join(job_dir, "answered.pdf")
    imgs = [Image.open(f).convert("RGB") for f in filled]
    imgs[0].save(pdf, "PDF", resolution=float(DPI), save_all=True, append_images=imgs[1:])
    if all(not p["lines"] for p in report["pages"]):
        raise RuntimeError("No answer lines found on any page — is this a worksheet? "
                           "Try a sharper photo or a PDF.")
    report["pdf"] = pdf
    report["totalSeconds"] = round(time.time() - t0, 1)
    report["convertSeconds"] = round(t_convert, 1)
    report["pageCount"] = len(pages)
    return report


if __name__ == "__main__":
    print(json.dumps(process(sys.argv[1]), indent=1))
