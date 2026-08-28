# plot_main_foldchange.py
# -*- coding: utf-8 -*-
"""
Main-figure fold-change summary — combines multiple timepoints into ONE figure.

A 2x2 grid of log2 fold-change boxes:

        Amplitude            Events
  E  |  E-Amp        |   E-Events        (E = elongated = 'flat' cells)
  R  |  R-Amp        |   R-Events        (R = 'round' cells)

Each panel has 4 boxes = {Yoda, GsMTx} x {30hpf, 48hpf}, split by a dotted line
between the two timepoints. Each drug is normalised to ITS OWN vehicle
(Yoda / mean(DMSO), GsMTx / mean(E3)) within that dataset & cell class, per
embryo, then log2-transformed (reference line at 0 = no change).

Cross-dataset assembler: reads the built `Q2_embryo_summary_cells.csv` from each
dataset in DATASETS. Does NOT touch the per-dataset pipeline.

Events note: event rates hit 0 in many embryos, and log2(0) = -inf. So the
events panels add a per-metric pseudocount EPS (= 1/2 the smallest nonzero event
rate, pooled) to numerator and denominator before the ratio. Amplitude (> 0,
~1) uses no pseudocount.

Outputs (under OUT_DIR):
  * combined 2x2:  main_foldchange_2x2.{png,svg}
  * per panel:     panel_<E-Amp|E-Events|R-Amp|R-Events>.{svg,pdf,png}
                   (svg/pdf keep text editable for Illustrator)
  * main_foldchange.csv

Run:  python plot_main_foldchange.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
# keep text editable in vector exports (Illustrator)
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcium_config as cfg
from plot_Q2_cells_events import draw_prism_box, style_axes, _fmt_p, apply_dense_yticks
from scipy.stats import ttest_ind


# =============================================================================
# Config — edit here
# =============================================================================
WSUF = "20"   # analysis-window suffix used when the tables were built

# Data paths are machine-specific and live in paths_local.py (git-ignored).
from paths_local import DATASET_ROOTS, MAIN_FIG_DIRS

_SUMMARY = Path("_py_out_%smin" % WSUF) / "tables" / "Q2_embryo_summary_cells.csv"

# (label, path to that dataset's Q2_embryo_summary_cells.csv), plotted L->R
DATASETS = [
    ("30hpf", DATASET_ROOTS["30hpf"] / _SUMMARY),
    ("48hpf", DATASET_ROOTS["48hpf"] / _SUMMARY),
]

# (vehicle, drug) — drug plotted in this order within each timepoint group
PAIRS = [("DMSO", "Yoda"), ("E3", "GsMTx")]

# panels: rows = cell classes, cols = metrics
CELLCLASSES = [("E", "flat"), ("R", "round")]         # (label, cell_class token)
METRICS = [
    ("Amp",    "amplitude", f"embryo_mean_amplitude_post_{WSUF}min", False),
    ("Events", "events",    f"mean_events_per_cell_per_{WSUF}min_post", False),
]  # (label, key, column, needs_pseudocount)

# Per-metric effect transform for the fold-change grid:
#   amplitude -> log2(drug / vehicle mean)   (ratio; baseline ~1, stable)
#   events    -> drug - vehicle mean         (difference; zero-safe rate)
FC_MODE = {"amplitude": "log2", "events": "diff", "duration": "log2"}


def fc_ylab(mkey: str) -> str:
    if FC_MODE.get(mkey, "log2") == "diff":
        return r"$\Delta$ events / cell / %s min (drug $-$ vehicle)" % WSUF
    return "log2 fold change"


# ---------- Main-figure row layout (1 x 4) ----------
# The four panels of the main figure, side by side in one figure instead of four
# separately-exported panels pasted together (which left big margins between them
# and forced a scale-down that shrank the type).
#
# Drawn at final placement size: keep it at 100% in Illustrator. The point sizes
# below are what the reader sees; rescaling the artboard rescales the type too.
ROW_FIGSIZE   = (10.0, 3.3)                 # whole 4-panel row, inches
# title_pad clears the 30hpf/48hpf group labels, which sit just above the axes.
ROW_FONT      = dict(title=10.5, ylabel=9.5, tick=9, pval=8.5, group=9,
                     title_pad=15)
ROW_BOX_WIDTH = 0.45    # data units (per-dataset panels use STYLE 0.20).
                        # ~40% of the slot: readable without crowding.
ROW_X_WITHIN  = 1.15    # spacing between the two drugs inside one timepoint
ROW_X_GAP     = 0.55    # extra gap between timepoints
ROW_XMARGIN   = (0.26, 0.10)   # (left, right) axis pad; the wide left pad
                               # keeps the longest p-string off the y-ticks
ROW_JITTER    = 0.22    # dot jitter spread; keep < ROW_BOX_WIDTH
ROW_DOT_SIZE  = 20
ROW_LW_SCALE  = 1.6     # thicken box/median strokes to match the larger panels
ROW_YTICK_NBINS = 11    # fewer, larger-spaced ticks than the 18 used at 2x2 size
ROW_XTICK_ROT = 30      # 4 six-char labels do not fit horizontally at this width
ROW_YPAD_TOP  = 0.21    # headroom for the p-value text above the tallest box
ROW_YPAD_BOT  = 0.05


_NL = chr(10)   # literal newline for two-line axis labels

def fc_ylab_short(mkey: str) -> str:
    """Row-figure y-label, stacked on two lines.

    The normalisation belongs on the axis: a bare delta reads as an absolute
    change, not a change relative to the control. Two lines keep each line
    short enough for the panel height, which one long string was not.
    """
    if FC_MODE.get(mkey, "log2") == "diff":
        return (r"$\Delta$ events / cell / %s min" % WSUF) + _NL + "(drug $-$ vehicle)"
    return "log2 fold change" + _NL + "(drug / vehicle)"

LOG2 = True
OUT_DIR = MAIN_FIG_DIRS["Yoda_GsMTx"]

# Reader-facing panel titles (line 1 = metric, line 2 = cell class).
METRIC_PUB = {"amplitude": r"Ca$^{2+}$ Intensity",
              "events":    r"Ca$^{2+}$ Event Frequency",
              "duration":  r"Ca$^{2+}$ Event Duration"}


def panel_title(mkey: str, cclass: str) -> str:
    return f"{METRIC_PUB.get(mkey, mkey)}\n{cfg.cell_class_display(cclass)}"


CELL_CLASS_COL = "cell_class"
COND_COL = "condition"


def _color(cond: str) -> str:
    try:
        return cfg.color_for(cond)
    except Exception:
        return "tab:gray"


def _vals(df: pd.DataFrame, cond: str, cell_class: str, col: str) -> np.ndarray:
    s = df[(df[COND_COL].astype(str) == str(cond)) &
           (df[CELL_CLASS_COL].astype(str) == str(cell_class))][col]
    a = pd.to_numeric(s, errors="coerce").to_numpy(float)
    return a[np.isfinite(a)]


def build_panel(data, cclass, mcol, eps, mode="log2",
                x_within=1.0, x_gap=0.8):
    """Compute box data + stats for one panel (drug x timepoint).

    mode = "log2": each embryo -> log2( value / mean(own vehicle) ).  Suited to
                   amplitude (baseline bounded away from 0).
    mode = "diff": each embryo -> value - mean(own vehicle).  Suited to events
                   (a rate whose baseline can sit near 0, where any ratio is
                   unstable); zero-event embryos land at -mean(vehicle), finite
                   and meaningful, with no pseudocount.
    The drug-vs-vehicle Welch p is computed on RAW values either way."""
    box_data, colors, xpos, labels, vs_p = [], [], [], [], []
    group_spans, fc = [], []
    x = 1.0
    for (tlabel, _path) in DATASETS:
        df = data[tlabel]
        g_start = x
        any_box = False
        for veh, drug in PAIRS:
            dv = _vals(df, drug, cclass, mcol)
            vv = _vals(df, veh, cclass, mcol)
            if len(dv) == 0 or len(vv) == 0:
                continue
            vmean = float(np.mean(vv))
            if mode == "diff":
                val = dv - vmean
            else:
                fold = (dv + eps) / (vmean + eps)
                val = np.log2(fold) if LOG2 else fold
            val = val[np.isfinite(val)]
            if len(val) == 0:
                continue
            try:
                p = float(ttest_ind(dv, vv, equal_var=False,
                                    nan_policy="omit").pvalue)
            except Exception:
                p = float("nan")
            box_data.append(val); colors.append(_color(drug)); xpos.append(x)
            labels.append(cfg.pub_label(drug)); vs_p.append(p)
            fc.append(dict(cell_class=cclass, metric=mcol, timepoint=tlabel,
                           drug=str(drug), vehicle=str(veh), n_drug=int(len(dv)),
                           vehicle_mean=vmean, transform=mode,
                           log2=bool(LOG2 and mode == "log2"),
                           pseudocount=(eps if mode == "log2" else 0.0),
                           median=float(np.median(val)), mean=float(np.mean(val)),
                           p_vs_vehicle_welch=p))
            any_box = True
            x += x_within
        if any_box:
            group_spans.append((tlabel, g_start, x - x_within))
        x += x_gap   # gap between timepoints
    return dict(box_data=box_data, colors=colors, xpos=xpos, labels=labels,
                vs_p=vs_p, group_spans=group_spans, fc=fc)


def draw_panel(ax, P, ylim, ref, ylab, title, fonts=None, box_width=None,
               xmargin=0.10, jitter=None, dot_size=None, lw_scale=1.0, nbins=18,
               xtick_rot=0.0):
    """Draw one fold-change panel.

    The keyword arguments override typography/geometry for the main-figure row
    layout; with all of them left at their defaults this renders exactly as the
    2x2 grid and the individual panel exports always have."""
    F = fonts or {}
    f_title = F.get("title", cfg.FONT["title"])
    f_ylab = F.get("ylabel", cfg.FONT["label"])
    f_tick = F.get("tick", cfg.FONT["tick"])
    f_pval = F.get("pval", cfg.STATS.get("pval_fontsize", cfg.FONT["legend"]))
    f_grp = F.get("group", cfg.FONT["legend"])
    t_pad = F.get("title_pad", cfg.FONT["title_pad"])

    if not P["box_data"]:
        ax.set_title(f"{title} (no data)", fontsize=f_title)
        style_axes(ax)
        return
    draw_prism_box(ax, P["box_data"], P["xpos"], P["colors"],
                   width=box_width, jitter=jitter, dot_size=dot_size,
                   lw_scale=lw_scale)
    ax.axhline(ref, color="0.4", ls="--", lw=0.8 * lw_scale, zorder=0)
    gs = P["group_spans"]
    if len(gs) >= 2:
        ax.axvline((gs[0][2] + gs[1][1]) / 2.0, color="0.5", ls=":",
                   lw=1.0 * lw_scale, zorder=0)
    ax.set_xticks(P["xpos"])
    if xtick_rot:
        ax.set_xticklabels(P["labels"], fontsize=f_tick, rotation=xtick_rot,
                           ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(P["labels"], fontsize=f_tick)
    ax.set_ylabel(ylab, fontsize=f_ylab)
    ax.set_title(title, fontsize=f_title, pad=t_pad)
    ax.set_ylim(*ylim)
    style_axes(ax)
    ax.tick_params(axis="y", labelsize=f_tick)
    apply_dense_yticks(ax, nbins=nbins)
    yr = ylim[1] - ylim[0]
    dec = cfg.STATS.get("pval_decimals", 3)
    for xp, val, p in zip(P["xpos"], P["box_data"], P["vs_p"]):
        yt = float(np.nanmax(val)) + 0.03 * yr
        ax.text(xp, yt, f"p{_fmt_p(p, dec)}", ha="center", va="bottom",
                fontsize=f_pval)
    for tlabel, g0, g1 in P["group_spans"]:
        ax.text((g0 + g1) / 2.0, 1.0, tlabel, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=f_grp)
    # xmargin may be a scalar (symmetric) or (left, right). The row figure uses
    # an asymmetric pad: the leftmost box carries the longest p-string, which
    # would otherwise print over the y-tick labels.
    if isinstance(xmargin, (tuple, list)):
        xl, xr = float(xmargin[0]), float(xmargin[1])
        x0, x1 = float(min(P["xpos"])), float(max(P["xpos"]))
        sp = (x1 - x0) or 1.0
        ax.set_xlim(x0 - xl * sp, x1 + xr * sp)
    else:
        ax.margins(x=xmargin)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for label, path in DATASETS:
        if not path.exists():
            print(f"[ERROR] missing table for {label}: {path}")
            sys.exit(1)
        data[label] = pd.read_csv(path)
    print(f"Loaded {len(data)} datasets: {list(data)}")

    # ref = 0 works for both transforms: log2(1) = 0 and (vehicle - vehicle) = 0.
    ref = 0.0

    # uniform per-metric pseudocount (pooled across BOTH cell classes)
    all_conds = {v for pr in PAIRS for v in pr}
    cclasses = [cc for _, cc in CELLCLASSES]
    eps_by_metric = {}
    for (mlab, mkey, mcol, need_eps) in METRICS:
        if not need_eps:
            eps_by_metric[mkey] = 0.0
            continue
        pooled = np.concatenate([_vals(df, c, cc, mcol)
                                 for df in data.values()
                                 for c in all_conds for cc in cclasses])
        pos = pooled[pooled > 0]
        eps_by_metric[mkey] = 0.5 * float(pos.min()) if pos.size else 0.0
    print("pseudocount per metric:", eps_by_metric)

    # build all panels, then per-column shared ylim (with p-text headroom)
    panels = {}
    for clab, cclass in CELLCLASSES:
        for mlab, mkey, mcol, _ in METRICS:
            panels[(clab, mlab)] = build_panel(data, cclass, mcol,
                                               eps_by_metric[mkey],
                                               mode=FC_MODE.get(mkey, "log2"))
    col_ylim = {}
    for mlab, mkey, mcol, _ in METRICS:
        allv = np.concatenate([v for clab, _ in CELLCLASSES
                               for v in panels[(clab, mlab)]["box_data"]]
                              or [np.array([0.0])])
        lo, hi = float(np.min(allv)), float(np.max(allv))
        rng = (hi - lo) or 1.0
        col_ylim[mlab] = (lo - 0.06 * rng, hi + 0.18 * rng)

    # ---- combined 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 9.0))
    for i, (clab, cclass) in enumerate(CELLCLASSES):
        for j, (mlab, mkey, mcol, _) in enumerate(METRICS):
            draw_panel(axes[i][j], panels[(clab, mlab)], col_ylim[mlab], ref,
                       fc_ylab(mkey), panel_title(mkey, cclass))
    fig.suptitle("Effect vs own vehicle — 30hpf | 48hpf.  Intensity: log2 fold "
                 "change;  events: difference (drug - vehicle mean)",
                 fontsize=cfg.FONT["title"] + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_DIR / "main_foldchange_2x2.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "main_foldchange_2x2.svg", bbox_inches="tight")
    plt.close(fig)

    # ---- individual panels (svg + pdf editable text, + png) ----
    for clab, cclass in CELLCLASSES:
        for mlab, mkey, mcol, _ in METRICS:
            f1, a1 = plt.subplots(figsize=(3.6, 4.4))
            draw_panel(a1, panels[(clab, mlab)], col_ylim[mlab], ref,
                       fc_ylab(mkey), panel_title(mkey, cclass))
            f1.tight_layout()
            stem = OUT_DIR / f"panel_{clab}-{mlab}"
            for ext in ("svg", "pdf", "png"):
                f1.savefig(stem.with_suffix("." + ext),
                           dpi=200 if ext == "png" else None, bbox_inches="tight")
            plt.close(f1)

    # ---- MAIN FIGURE: 1x4 row (all four panels in one figure) ----
    # Rebuilt with tighter x-geometry so the boxes fill the panel instead of
    # sitting as thin slivers in a wide, mostly-empty axis.
    row_panels = {}
    for clab, cclass in CELLCLASSES:
        for mlab, mkey, mcol, _ in METRICS:
            row_panels[(clab, mlab)] = build_panel(
                data, cclass, mcol, eps_by_metric[mkey],
                mode=FC_MODE.get(mkey, "log2"),
                x_within=ROW_X_WITHIN, x_gap=ROW_X_GAP)

    # shared y-limits per metric column (both cell classes), tighter padding
    row_ylim = {}
    for mlab, mkey, mcol, _ in METRICS:
        allv = np.concatenate([v for clab, _ in CELLCLASSES
                               for v in row_panels[(clab, mlab)]["box_data"]]
                              or [np.array([0.0])])
        lo, hi = float(np.min(allv)), float(np.max(allv))
        rng_ = (hi - lo) or 1.0
        row_ylim[mlab] = (lo - ROW_YPAD_BOT * rng_, hi + ROW_YPAD_TOP * rng_)

    # left-to-right: E-Amp, E-Events, R-Amp, R-Events
    row_order = [(clab, cclass, mlab, mkey)
                 for clab, cclass in CELLCLASSES
                 for mlab, mkey, mcol, _ in METRICS]

    figr, axesr = plt.subplots(1, len(row_order), figsize=ROW_FIGSIZE)
    for axr, (clab, cclass, mlab, mkey) in zip(np.atleast_1d(axesr), row_order):
        draw_panel(axr, row_panels[(clab, mlab)], row_ylim[mlab], ref,
                   fc_ylab_short(mkey), panel_title(mkey, cclass),
                   fonts=ROW_FONT, box_width=ROW_BOX_WIDTH,
                   xmargin=ROW_XMARGIN, jitter=ROW_JITTER,
                   dot_size=ROW_DOT_SIZE, lw_scale=ROW_LW_SCALE,
                   nbins=ROW_YTICK_NBINS, xtick_rot=ROW_XTICK_ROT)
    figr.tight_layout(pad=0.5, w_pad=2.0)
    for ext in ("svg", "pdf", "png"):
        figr.savefig((OUT_DIR / "main_foldchange_row").with_suffix("." + ext),
                     dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(figr)

    rows_out = [r for P in panels.values() for r in P["fc"]]
    pd.DataFrame(rows_out).to_csv(OUT_DIR / "main_foldchange.csv", index=False)
    print(f"[OK] wrote 1x4 row + 2x2 + 4 individual panels + csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
