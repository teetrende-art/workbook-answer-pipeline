# Worksheet → answered PDF

Turn a photograph of a workbook page into a PDF with the answers written on the printed lines.

Built for a real problem: a teacher photographs a page from a workbook, and the answered copy has
to come back looking like the page it came from — answers on the right lines, in the right columns,
at a size that fits.

## Pipeline

1. **Deskew and upscale.** A phone photo arrives rotated, cropped or small. Anything under
   ~900 px wide is refused rather than answered badly.
2. **Find the write targets** (`targets.py`) — three kinds:
   - ruled answer lines (long horizontal strokes, isolated from printed rules and chart gridlines)
   - inline sentence gaps (short runs of white space inside a line of text)
   - two-column layouts, split at the detected gutter so answers do not cross into the other column
3. **Read the page** with a vision model, once, and again on a zoomed crop of any chart on the page.
   Where a chart is involved the model must name the gridline a bar reaches before answering —
   that single instruction is the difference between 2/6 and 6/6 on bar-graph pages.
4. **Write the answers** onto the page, auto-fitting each one to the width of its blank and sitting
   on the printed baseline rather than floating above it.
5. **Re-check every page** — render it and read it back before the PDF is assembled. A page that
   cannot be read back is a failure, not a delivery.

## What it refuses

- Images too small to read (returns a plain explanation, not a blank PDF)
- Pages with no answer targets found
- Page types it cannot handle — the two-column inline-gap case is detected and flagged instead of
  returning empty answers

Refusing is deliberate. A blank PDF labelled "done" is worse than an error message.

## Numbers

- Single ruled page: **~9 s**
- Six workbook pages, end to end: **~3 min** (36.7 s/page on a chart-heavy set)
- Chart pages after the gridline instruction: **6/6 correct**

## Limitation, stated honestly

A page with 30+ inline gaps across two columns still needs block-by-block processing rather than
one pass over the whole page. It is detected and refused, not silently answered wrong.

## Files

| File | Role |
|---|---|
| `lines.py` | ruled-line detection and deduplication |
| `targets.py` | general write-target detection, incl. inline gaps and column split |
| `chartview.py` | cropped chart images for the second reading pass |
| `worker.py` | the job runner: OCR, answering, placement, verification |
