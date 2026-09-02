# make_condition_movie.py
# -*- coding: utf-8 -*-
"""
Build one supplementary movie per experiment set: a 2x2 grid of drug
conditions, each panel showing the whole vessel twice -

    top     GCaMP (green) + Lifeact-mCherry (magenta) merge, with a box round
            the elongated cell and the round cell, labelled
    bottom  the same GCaMP frame through Fiji's Green Fire Blue LUT

matching the layout of Fig. 5a-d and Supp. Figs 9g-j / 10.

Display scaling
---------------
By default every panel shares one range per channel, pooled over all frames of
all panels. That is only meaningful when the panels were acquired under the
same settings - check the file names first. In these representative sets they
were not (one set mixes GCamp7a_fli1lifeactmCh, GCamp7a_lifeactmCh and
GCamp_LAmCh), so --gcamp_per_panel / --lifeact_per_panel scale each panel on
its own, and --gcamp_range LABEL=lo:hi overrides individual panels by hand.

Whichever was used is printed and written to a sidecar txt, because the legend
has to say which: a shared range means panel brightness is comparable, a
per-panel range means it is not.

A second caveat on the pooled range: it is taken over the WHOLE field, most of
which is background and other tissue, while the quantification is made in the
two cell ROIs. Whole-field brightness is not a proxy for the measured effect -
in the ISO set E3 and ISO have the same whole-field median even though the ROI
quantification separates them clearly.

Photobleaching is NOT corrected: over 30 min the later frames genuinely dim,
and hiding that would misrepresent the raw data.

ROIs
----
Read from an ImageJ ROI set (.zip) saved on the SAME cropped TIFF, so the boxes
land where they do in the figure. Naming decides what is drawn:
    elongated* / flat*   -> elongated-cell box
    round*               -> round-cell box
    anything else (e.g. a VDA line) -> drawn as an unlabelled outline
The .roi parser is inline - no extra dependency.

Run:
    python make_condition_movie.py --in_dir "<folder with tif + _ROIs.zip>" \\
        --order "DMSO,Yoda1,E3,GsMTx4" --out movie_30hpf.mp4
    python make_condition_movie.py --manifest panels.csv --out ...
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# Config - edit here
# =============================================================================
FPS = 10
SCALEBAR_UM = 20.0
# Pooled percentiles for the shared range, computed on NON-ZERO pixels.
# The dynamic range here is narrow - camera offset ~105, background ~110,
# signal to ~215 - and registration leaves zero-filled borders, so a naive
# 0.5 percentile lands on 0 and washes the whole frame out.
PCT_LO, PCT_HI = 1.0, 99.9
DT_SECONDS = 30.0                 # falls back to the TIFF's finterval

CH_GCAMP, CH_LIFEACT = 0, 1       # channel order in the TIFF (TCYX)
# Fiji's Green Fire Blue LUT, so the movie and the figure colour bar match
# exactly. Set FIJI_LUT to the .lut file, or drop a copy next to this script.
# Without it the code falls back to a built-in approximation and says so.
LUT_CANDIDATES = [
    os.environ.get("FIJI_LUT", ""),
    "Green Fire Blue.lut",
    "C:/Program Files/Fiji.app/luts/Green Fire Blue.lut",
    "/Applications/Fiji.app/luts/Green Fire Blue.lut",
    str(Path.home() / "Fiji.app/luts/Green Fire Blue.lut"),
]

GAP = 6                           # px between the two rows of a panel
MARGIN = 10                       # px between panels in the 2x2 grid
BOX_LW = 2                        # scaled with the panel at render time
LINE_ALPHA = 128                  # traced lines (VDA) at 50% opacity
LABEL_ELONGATED = "elongated"
LABEL_ROUND = "round"
FONT_CANDIDATES = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]
FS_COND, FS_BOX, FS_TIME = 30, 24, 26     # font sizes, scaled by panel width

WHITE = (255, 255, 255)


# =============================================================================
# ImageJ .roi reader (inline; format per ij.io.RoiDecoder)
# =============================================================================
def _i16(b, o):
    return int.from_bytes(b[o:o + 2], "big", signed=True)


def _u16(b, o):
    return int.from_bytes(b[o:o + 2], "big", signed=False)


def _i32(b, o):
    return int.from_bytes(b[o:o + 4], "big", signed=True)


def parse_roi(buf: bytes, name: str) -> dict:
    """Bounds (and outline, when the ROI is not a rectangle) of one .roi."""
    if buf[:4] != b"Iout":
        raise ValueError("not an ImageJ ROI: %s" % name)
    version = _u16(buf, 4)
    rtype = buf[6]
    top, left, bottom, right = (_i16(buf, 8), _i16(buf, 10),
                                _i16(buf, 12), _i16(buf, 14))
    n = _u16(buf, 16)
    options = _u16(buf, 50)
    roi = dict(name=name, type=rtype, left=left, top=top,
               right=right, bottom=bottom, x=None, y=None)

    # polygon(0) line(3) freeline(4) polyline(5) freehand(7) traced(8)
    if rtype in (0, 3, 4, 5, 7, 8) and n > 0:
        base = 64
        sub_pixel = bool(options & 128) and version >= 222
        if sub_pixel:
            fb = base + 4 * n
            xs = np.frombuffer(buf[fb:fb + 4 * n], dtype=">f4").astype(float)
            ys = np.frombuffer(buf[fb + 4 * n:fb + 8 * n], dtype=">f4").astype(float)
        else:
            xs = np.frombuffer(buf[base:base + 2 * n], dtype=">i2").astype(float) + left
            ys = np.frombuffer(buf[base + 2 * n:base + 4 * n], dtype=">i2").astype(float) + top
        roi["x"], roi["y"] = xs, ys
    return roi


def read_roi_zip(path: Path) -> list:
    """Every ROI in a Fiji ROI set. The zip entry name is the ROI name."""
    out = []
    with zipfile.ZipFile(path) as zf:
        for entry in zf.namelist():
            if not entry.lower().endswith(".roi"):
                continue
            out.append(parse_roi(zf.read(entry), Path(entry).stem))
    return out


def classify(roi_name: str):
    """The label a ROI gets, or None to draw it unlabelled."""
    n = roi_name.lower()
    if n.startswith(("elong", "flat")):
        return LABEL_ELONGATED
    if n.startswith("round"):
        return LABEL_ROUND
    if n.startswith(("vda", "dda", "da_")):
        return roi_name.upper().replace("_", "")
    return None


# =============================================================================
# Rendering
# =============================================================================
def load_lut(path: Path) -> np.ndarray:
    """Fiji .lut -> (256, 3) uint8. Falls back to a blue-green-yellow ramp."""
    if path.exists():
        raw = path.read_bytes()
        if len(raw) == 768:
            return np.frombuffer(raw, dtype=np.uint8).reshape(3, 256).T.copy()
    print("[WARN] LUT not found at %s - using a built-in approximation" % path)
    i = np.linspace(0, 1, 256)
    r = np.clip(2 * i - 1, 0, 1)
    g = np.clip(2 * i, 0, 1)
    b = np.clip(1 - 2 * i, 0, 1) + np.clip(4 * i - 3, 0, 1)
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def stretch(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear to 0-255 with a FIXED range, so panels stay comparable."""
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return (np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def font(size: int):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_panel(gc, la, rois, cond, t_s, um_px, rng, fonts, scalebar_um, lw=BOX_LW):
    """One condition, one frame: merge on top, Green Fire Blue below."""
    (g_lo, g_hi), (l_lo, l_hi) = rng["gcamp"], rng["lifeact"]
    g8, l8 = stretch(gc, g_lo, g_hi), stretch(la, l_lo, l_hi)

    merge = np.zeros(gc.shape + (3,), np.uint8)
    merge[..., 0] = l8                       # magenta = Lifeact
    merge[..., 1] = g8                       # green   = GCaMP
    merge[..., 2] = l8
    fire = LUT[g8]

    h, w = gc.shape
    canvas = Image.new("RGB", (w, h * 2 + GAP), (0, 0, 0))
    # Traced lines (the VDA outline) go on a translucent layer so they mark the
    # vessel without hiding the signal underneath; boxes and all text stay
    # opaque so they are readable.
    is_line = lambda r: r["x"] is not None and r["type"] in (3, 4, 5)
    boxes = [r for r in rois if not is_line(r)]

    def clearance(px, py):
        """Distance from a point to the nearest box, 0 if inside one."""
        if not boxes:
            return 1e9
        return min(float(np.hypot(max(b["left"] - px, 0.0, px - b["right"]),
                                  max(b["top"] - py, 0.0, py - b["bottom"])))
                   for b in boxes)

    top = Image.fromarray(merge).convert("RGBA")
    lines = Image.new("RGBA", top.size, (0, 0, 0, 0))
    dl = ImageDraw.Draw(lines)
    for r in rois:
        if is_line(r):
            dl.line(list(zip(r["x"], r["y"])), fill=WHITE + (LINE_ALPHA,), width=lw)
    top = Image.alpha_composite(top, lines).convert("RGB")

    d = ImageDraw.Draw(top)
    for b in boxes:
        d.rectangle([b["left"], b["top"], b["right"], b["bottom"]],
                    outline=WHITE, width=lw)
        lab = classify(b["name"])
        if lab:
            tw = d.textlength(lab, font=fonts["box"])
            cx = 0.5 * (b["left"] + b["right"])
            ty = max(0, b["top"] - fonts["box"].size - 4)
            d.text((cx - tw / 2, ty), lab, fill=WHITE, font=fonts["box"])

    for r in rois:
        lab = classify(r["name"]) if is_line(r) else None
        if not lab:
            continue
        # Label whichever END of the line is further from any box - on the BDM
        # panel the round cell sits at the left edge and a fixed left-hand
        # label collided with it.
        tw = d.textlength(lab, font=fonts["box"])
        ends = []
        for i, anchor in ((int(np.argmin(r["x"])), "left"),
                          (int(np.argmax(r["x"])), "right")):
            px, py = float(r["x"][i]), float(r["y"][i])
            tx = px + 6 if anchor == "left" else px - 6 - tw
            ends.append((clearance(px, py), tx, py))
        _, tx, py = max(ends)
        tx = min(max(tx, 2.0), float(w) - tw - 2.0)
        d.text((tx, max(0.0, py - fonts["box"].size - 4)), lab,
               fill=WHITE, font=fonts["box"])

    d.text((10, 8), cond, fill=WHITE, font=fonts["cond"])
    canvas.paste(top, (0, 0))

    bot = Image.fromarray(fire)
    d2 = ImageDraw.Draw(bot)
    mm, ss = divmod(int(round(t_s)), 60)
    stamp = "%d:%02d" % (mm, ss)
    tw = d2.textlength(stamp, font=fonts["time"])
    d2.text((w - tw - 12, 8), stamp, fill=WHITE, font=fonts["time"])
    bar = max(2, int(round(scalebar_um / um_px)))
    bh = max(3, int(round(h * 0.012)))
    d2.rectangle([w - bar - 14, h - bh - 14, w - 14, h - 14], fill=WHITE)
    canvas.paste(bot, (0, h + GAP))
    return np.asarray(canvas)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="", help="folder of <cond>.tif + <cond>_ROIs.zip")
    ap.add_argument("--manifest", default="", help="csv: label,tif,roi")
    ap.add_argument("--order", default="",
                    help="comma list of panel labels in 2x2 reading order; "
                         "use '-' for a blank cell")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lut", default="",
                    help="path to Fiji's Green Fire Blue.lut "
                         "(or set FIJI_LUT)")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--scalebar_um", type=float, default=SCALEBAR_UM)
    ap.add_argument("--labels", default="",
                    help="comma list of DISPLAY names, same length as --order. "
                         "--order values are match keys and must follow the "
                         "file names; this is what is drawn on the panel, so "
                         "it can read 'piezo crispant + ISO' instead of "
                         "'piezo_ISO'.")
    ap.add_argument("--lifeact_per_panel", action="store_true",
                    help="scale the Lifeact channel per panel")
    ap.add_argument("--gcamp_per_panel", action="store_true",
                    help="scale GCaMP per panel too. Legitimate when the panels "
                         "are from different imaging sessions - check the file "
                         "names - but then say so in the legend, because "
                         "brightness is no longer comparable between panels")
    ap.add_argument("--gcamp_range", default="",
                    help="per-panel GCaMP overrides, LABEL=lo:hi,LABEL=lo:hi")
    ap.add_argument("--lifeact_range", default="", help="same for Lifeact")
    ap.add_argument("--range_gcamp", default="",
                    help="override the shared GCaMP range, e.g. 108,200")
    ap.add_argument("--range_lifeact", default="", help="same for Lifeact")
    ap.add_argument("--max_width", type=int, default=0,
                    help="downscale the finished grid to this width (0 = native)")
    args = ap.parse_args()

    global LUT
    lut = next((Path(c) for c in LUT_CANDIDATES if c and Path(c).exists()), None)
    LUT = load_lut(lut if lut else Path(args.lut or "Green Fire Blue.lut"))

    # ---- collect panels ----
    panels = []
    if args.manifest:
        import csv
        with open(args.manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                panels.append((row["label"], Path(row["tif"]),
                               Path(row["roi"]) if row.get("roi") else None))
    else:
        # Pair by CONDITION NAME, not by a filename convention: the TIFFs keep
        # their long acquisition names while the ROI sets are named after the
        # condition, so <stem>_ROIs.zip never matches. A label matches a file
        # when its alphanumeric core appears in the file's alphanumeric core;
        # a trailing digit is dropped too, so "Yoda1" finds "..._20uMYoda_..."
        # and "GsMTx4" finds "..._GsMTx_...". Ambiguity is an error, never a
        # silent guess.
        if not args.order:
            raise SystemExit("--in_dir needs --order to say which panel is which")
        d = Path(args.in_dir)
        tifs = sorted(d.glob("*.tif")) + sorted(d.glob("*.tiff"))
        zips = sorted(d.glob("*.zip"))

        def norm(x):
            return "".join(c for c in str(x).lower() if c.isalnum())

        def cond_token(stem):
            """The condition as it appears in '..._post_<COND>_...'.

            Matching on the whole filename is ambiguous: in the ISO set both
            files carry '_e3_' (the embryo number), so "E3" hits the ISO movie
            too. The token after '_post_' is the condition itself, and it
            resolves those cleanly.
            """
            m = re.search(r"_post_([^_]+)", stem, re.IGNORECASE)
            return m.group(1) if m else None

        def label_tokens(label):
            """A label as a list of parts, EVERY one of which must be present.

            "MIC_E3" has to mean MIC *and* E3: in the piezo set both movies
            carry '_post_E3_' or '_post_ISO_' and only the genotype in the
            middle of the name tells them apart, so a single substring cannot
            do it. Each part also gets a digit-stripped alternative, which is
            how "Yoda1" finds "_20uMYoda_" and "GsMTx4" finds "_GsMTx_"; the
            alternative is dropped when it would be too short to be specific
            (E3 -> "e" would match everything).
            """
            out = []
            for part in re.split(r"[^0-9A-Za-z]+", label):
                p = norm(part)
                if not p:
                    continue
                alts = [p]
                stripped = p.rstrip("0123456789")
                if stripped != p and len(stripped) >= 3:
                    alts.append(stripped)
                out.append(alts)
            return out

        def find(label, files, what):
            toks = label_tokens(label)
            if not toks:
                return None
            # the '_post_<COND>_' token first, then the whole name
            for scope in (cond_token, lambda st: st):
                hits = []
                for f in files:
                    hay = norm(scope(f.stem) or "")
                    if hay and all(any(a in hay for a in alts) for alts in toks):
                        hits.append(f)
                if len(hits) == 1:
                    return hits[0]
                if len(hits) > 1:
                    raise SystemExit(
                        "%r matches %d %s files: %s -- rename them, "
                        "or use --manifest"
                        % (label, len(hits), what, [h.name for h in hits]))
            return None

        panels = []
        for w in [x.strip() for x in args.order.split(",")]:
            if w == "-":
                panels.append(None)
                continue
            t = find(w, tifs, "tif")
            if t is None:
                raise SystemExit("no tif matching %r in %s" % (w, d))
            panels.append((w, t, find(w, zips, "roi zip")))

        if args.labels:
            shown = [x.strip() for x in args.labels.split(",")]
            if len(shown) != len(panels):
                raise SystemExit("--labels has %d entries, --order has %d"
                                 % (len(shown), len(panels)))
            panels = [None if p is None else (shown[i], p[1], p[2])
                      for i, p in enumerate(panels)]
    if not panels:
        raise SystemExit("no panels found")

    # ---- read stacks, work out the shared display range ----
    stacks, um_px, dt = [], None, DT_SECONDS
    for p in panels:
        if p is None:
            stacks.append(None)
            continue
        lab, tifp, roip = p
        with tiff.TiffFile(tifp) as tf:
            arr = tf.series[0].asarray()
            md = tf.imagej_metadata or {}
            if um_px is None:
                xr = tf.pages[0].tags["XResolution"].value
                um_px = xr[1] / xr[0]
            if md.get("finterval"):
                dt = float(md["finterval"])
        if arr.ndim != 4:
            raise SystemExit("%s: expected TCYX, got %s" % (tifp.name, arr.shape))
        rois = read_roi_zip(roip) if roip and roip.exists() else []
        if roip and not roip.exists():
            print("[WARN] no ROI set for %s" % lab)
        stacks.append(dict(label=lab, arr=arr, rois=rois, path=tifp))
        print("  %-10s %s  %d ROIs  <- %s"
              % (lab, arr.shape, len(rois), tifp.name))

    live = [s for s in stacks if s]
    sub = lambda a: a[::max(1, len(a) // 12)]

    def pooled(ch):
        v = np.concatenate([sub(s["arr"][:, ch]).ravel() for s in live])
        v = v[v > 0]                      # drop the registration fill
        return (float(np.percentile(v, PCT_LO)), float(np.percentile(v, PCT_HI)))

    rng = {"gcamp": pooled(CH_GCAMP), "lifeact": pooled(CH_LIFEACT)}
    if args.range_gcamp:
        rng["gcamp"] = tuple(float(v) for v in args.range_gcamp.split(","))
    if args.range_lifeact:
        rng["lifeact"] = tuple(float(v) for v in args.range_lifeact.split(","))

    # Per-panel display ranges.
    #
    # A shared GCaMP range only means something when the panels were acquired
    # under identical settings. Check the file names before assuming they were:
    # in these representative sets they are not (one set mixes
    # GCamp7a_fli1lifeactmCh, GCamp7a_lifeactmCh and GCamp_LAmCh, i.e. separate
    # sessions), so absolute brightness is not comparable between panels and a
    # shared range only makes the dimmer session look like less calcium.
    #
    # A second reason the pooled range tracks the biology poorly: it is taken
    # over the WHOLE field, most of which is background and other tissue, while
    # the measurement is made in the two cell ROIs. In the ISO set E3 and ISO
    # have the same whole-field median even though the ROI quantification
    # separates them clearly.
    #
    # So per-panel is allowed for either channel, and whatever is used is
    # written to the sidecar for the legend to quote.
    per_panel = {"gcamp": {}, "lifeact": {}}
    for ch_name, ch, flag in (("gcamp", CH_GCAMP, args.gcamp_per_panel),
                              ("lifeact", CH_LIFEACT, args.lifeact_per_panel)):
        if not flag:
            continue
        for st in live:
            v = st["arr"][:, ch].ravel()
            v = v[v > 0]
            per_panel[ch_name][st["label"]] = (
                float(np.percentile(v, PCT_LO)), float(np.percentile(v, PCT_HI)))

    for ch_name, spec in (("gcamp", args.gcamp_range),
                          ("lifeact", args.lifeact_range)):
        for item in [x for x in spec.split(",") if x.strip()]:
            lab, _, pair = item.partition("=")
            lo_hi = pair.split(":")
            if len(lo_hi) != 2:
                raise SystemExit("bad range spec %r; want LABEL=lo:hi" % item)
            known = [st["label"] for st in live]
            if lab.strip() not in known:
                raise SystemExit("range spec for unknown panel %r; have %s"
                                 % (lab.strip(), known))
            per_panel[ch_name][lab.strip()] = (float(lo_hi[0]), float(lo_hi[1]))

    def show(ch_name):
        if not per_panel[ch_name]:
            return "%.0f - %.0f (shared)" % rng[ch_name]
        return "; ".join(
            "%s %.0f-%.0f" % ((st["label"],)
                              + per_panel[ch_name].get(st["label"], rng[ch_name]))
            for st in live)

    print("\ndisplay ranges (%.1f-%.1f pct of non-zero pixels):" % (PCT_LO, PCT_HI))
    print("  GCaMP   %s" % show("gcamp"))
    print("  Lifeact %s" % show("lifeact"))
    print("pixel size %.4f um, frame interval %.0f s" % (um_px, dt))

    # ---- geometry: pad every panel to a common size ----
    ph = max(s["arr"].shape[2] for s in live)
    pw = max(s["arr"].shape[3] for s in live)
    n_frames = min(s["arr"].shape[0] for s in live)
    if len(set(s["arr"].shape[0] for s in live)) > 1:
        print("[WARN] frame counts differ %s - trimming all to %d"
              % ([s["arr"].shape[0] for s in live], n_frames))
    cell_w = pw
    ncol = 2
    nrow = int(np.ceil(len(stacks) / ncol))
    # Each ROW is only as tall as it needs to be, and a panel shorter than its
    # row is centred in it. Sizing every row to the tallest panel in the whole
    # grid left a black band under the shorter row.
    row_h = []
    for r in range(nrow):
        hs = [s["arr"].shape[2] * 2 + GAP
              for s in stacks[r * ncol:(r + 1) * ncol] if s]
        row_h.append(max(hs) if hs else 0)
    row_y = [MARGIN + sum(row_h[:r]) + r * MARGIN for r in range(nrow)]
    W = ncol * cell_w + (ncol + 1) * MARGIN
    H = sum(row_h) + (nrow + 1) * MARGIN
    scale = min(1.0, args.max_width / W) if args.max_width else 1.0
    # Render text large enough that it is still legible AFTER the grid is
    # downscaled to --max_width.
    fs = (pw / 2048.0) / max(scale, 1e-6)
    fonts = {k: font(max(10, int(round(v * fs))))
             for k, v in (("cond", FS_COND), ("box", FS_BOX), ("time", FS_TIME))}
    print("grid %dx%d, frame %dx%d%s, %d frames"
          % (nrow, ncol, W, H,
             "" if scale == 1 else " -> scaled x%.2f" % scale, n_frames))

    # ---- render ----
    import imageio.v2 as imageio
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ow = int(W * scale) // 2 * 2
    oh = int(H * scale) // 2 * 2
    wr = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=9,
                            macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for t in range(n_frames):
        canvas = np.zeros((H, W, 3), np.uint8)
        for k, s in enumerate(stacks):
            if s is None:
                continue
            r, c = divmod(k, ncol)
            x0 = MARGIN + c * (cell_w + MARGIN)
            prng = {k: per_panel[k].get(s["label"], rng[k])
                    for k in ("gcamp", "lifeact")}
            img = draw_panel(s["arr"][t, CH_GCAMP], s["arr"][t, CH_LIFEACT],
                             s["rois"], s["label"], t * dt, um_px, prng, fonts,
                             args.scalebar_um, lw=max(2, int(round(BOX_LW / scale))))
            y0 = row_y[r] + (row_h[r] - img.shape[0]) // 2      # centred in its row
            x0 += (cell_w - img.shape[1]) // 2                  # and in its column
            canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        if scale != 1:
            canvas = np.asarray(Image.fromarray(canvas).resize((ow, oh), Image.LANCZOS))
        wr.append_data(canvas)
        if t % 10 == 0:
            print("   frame %d/%d" % (t + 1, n_frames))
    wr.close()
    print("\n[OK] %s" % out)

    side = out.with_name(out.stem + "_display_settings.txt")
    side.write_text(
        "Movie: %s\n\nPanels (2x2 reading order):\n%s\n\n"
        "Display ranges, %.1f-%.1f percentile of non-zero pixels.\n"
        "'shared' means one range across the panels, so brightness IS\n"
        "comparable between them. A per-panel range means it is NOT, and the\n"
        "figure legend has to say so.\n"
        "  GCaMP    %s\n  Lifeact  %s\n\n"
        "GCaMP LUT: Fiji Green Fire Blue.\n"
        "Photobleaching is not corrected.\n"
        "Pixel size %.4f um; frame interval %.0f s; playback %g fps.\n"
        "Scale bar %.0f um.\n"
        % (out.name,
           "\n".join("  %d. %s" % (i + 1, s["label"] if s else "(blank)")
                     for i, s in enumerate(stacks)),
           PCT_LO, PCT_HI, show("gcamp"), show("lifeact"),
           um_px, dt, args.fps,
           args.scalebar_um),
        encoding="utf-8")
    print("[OK] %s" % side)


if __name__ == "__main__":
    main()
