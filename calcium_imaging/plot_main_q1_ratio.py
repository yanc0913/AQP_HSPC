# plot_main_q1_ratio.py
# -*- coding: utf-8 -*-
"""
Main-figure: vDA/dDA ratio (Q1) in E3 control, 30hpf vs 48hpf.

Two boxes per figure, one per developmental stage, pooling the E3 control
embryos of every WT dataset recorded at that stage:

    30hpf  <- Analysis_30hpf_Yoda
    48hpf  <- Analysis_48hpf_Yoda  +  Analysis_48hpf_ISO

The MIC / piezo-crispant dataset is deliberately EXCLUDED: its E3 embryos are
injected backgrounds (MIC / Piezo genotypes), not WT, so they do not belong in a
wild-type developmental baseline. As a second guard the loader also filters on
genotype == WT, so a mixed dataset could never leak in.

Two figures are produced, one per phase:

  * pre     - the whole pre phase (~4.5 min, ~10 frames). The untouched
              baseline before the remount gap. Same quantity as the per-dataset
              `box_preonly` panel of plot_Q1_vDA_dDA_ratio.py.
  * post20  - the FIRST 20 MIN of the post phase (~40 frames), matching the
              cfg.POST_ANALYSIS_MIN window used everywhere else in the pipeline.

Both are recomputed here from the frame-level Q1_ratio_prepost.csv rather than
read from Q1_ratio_summary_prepost.csv, because that built summary medians over
the ENTIRE phase - for post that is ~30 min, not the 20-min analysis window.
(Recomputing pre this way reproduces the built summary to 4e-16, verified.)

Unit of analysis is the EMBRYO: each embryo contributes one number, the median
of its per-frame vDA/dDA ratio over the window. Stages are compared with a
Welch t-test on those per-embryo values. Each figure scales its own y-axis
(SHARE_Y=False): the post window contains one very high embryo, and a shared
axis squashes the pre plot into the bottom third.

Cross-dataset assembler: reads the built tables only. Does NOT touch the
per-dataset pipeline.

Outputs (under OUT_DIR):
  * main_q1_ratio_pre.{svg,pdf,png}
  * main_q1_ratio_post20.{svg,pdf,png}   (svg/pdf keep text editable)
  * main_q1_ratio.csv                    (per-embryo values, both phases)

Run:  python plot_main_q1_ratio.py
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
# Config - edit here
# =============================================================================
WSUF = "20"   # analysis-window suffix used when the tables were built

# Data paths are machine-specific and live in paths_local.py (git-ignored).
from paths_local import DATASET_ROOTS, MAIN_FIG_DIRS

_RATIO = Path("_py_out_%smin" % WSUF) / "tables" / "Q1_ratio_prepost.csv"

# (x-label, [dataset keys pooled into this box]) - plotted left to right.
# Piezo is absent on purpose: MIC / piezo-crispant are not wild type.
STAGES = [
    ("30hpf", ["30hpf"]),
    ("48hpf", ["48hpf", "ISO"]),
]

CONDITION = "E3"    # Q1 is a control-only read-out
GENOTYPE = "WT"     # guard: never pool injected backgrounds into a WT baseline
VALUE_COL = "ratio_mean_bgsub"   # per-frame ratio in the frame-level table

# One figure per entry. tmax = None means "use the whole phase".
PHASES = [
    dict(phase="pre",  tmax=None, stem="main_q1_ratio_pre",
         subtitle="E3 Control, pre"),
    dict(phase="post", tmax=float(WSUF), stem="main_q1_ratio_post20",
         subtitle="E3 Control, post 0-%s min" % WSUF),
]

# Reference line at ratio = 1 (ventral band equal to dorsal). Set to None to drop.
REF_LINE = 1.0

# Share one y-axis across the pre and post figures so they are comparable.
SHARE_Y = False   # post has a large outlier; a shared axis squashes the pre plot

OUT_DIR = MAIN_FIG_DIRS["Q1_ratio"]

# Typography / geometry - matched to the main-figure row layouts so all the
# main figures share one visual system. Drawn at final placement size.
FIGSIZE   = (2.9, 3.4)
FONT      = dict(title=10.5, ylabel=9.5, tick=9, pval=8.5, title_pad=6)
BOX_WIDTH = 0.42
X_STEP    = 1.0
XMARGIN   = (0.55, 0.55)   # single pair of boxes: keep them off the spines
JITTER    = 0.20
DOT_SIZE  = 20
LW_SCALE  = 1.6
YTICK_NBINS = 8

TITLE_MAIN = "vDA / dDA Ratio"
YLABEL = "vDA / dDA ratio"


def load_stage(keys: list[str], phase: str, tmax) -> pd.DataFrame:
    """Per-embryo median vDA/dDA ratio over the window, WT E3 embryos only."""
    frames = []
    for k in keys:
        path = DATASET_ROOTS[k] / _RATIO
        if not path.exists():
            print(f"[ERROR] missing Q1 table for {k}: {path}")
            sys.exit(1)
        df = pd.read_csv(path)
        need = {"condition", "genotype", "phase", "embryo_id", "time_min", VALUE_COL}
        missing = need - set(df.columns)
        if missing:
            print(f"[ERROR] {path} lacks columns: {sorted(missing)}")
            sys.exit(1)
        sub = df[(df["condition"].astype(str) == CONDITION) &
                 (df["genotype"].astype(str) == GENOTYPE) &
                 (df["phase"].astype(str) == phase)].copy()
        if tmax is not None:
            sub = sub[pd.to_numeric(sub["time_min"], errors="coerce") <= float(tmax)]
        sub[VALUE_COL] = pd.to_numeric(sub[VALUE_COL], errors="coerce")
        sub = sub.dropna(subset=[VALUE_COL])
        if sub.empty:
            print(f"    {k:6s}: no usable frames")
            continue
        g = (sub.groupby(["embryo_id", "batch_id"], as_index=False)
                .agg(value=(VALUE_COL, "median"), n_frames=(VALUE_COL, "size")))
        g["dataset"] = k
        frames.append(g)
        print(f"    {k:6s}: n={len(g)} embryos, "
              f"{int(g['n_frames'].min())}-{int(g['n_frames'].max())} frames each")
    if not frames:
        return pd.DataFrame(columns=["embryo_id", "batch_id", "value",
                                     "n_frames", "dataset"])
    return pd.concat(frames, ignore_index=True)


def draw_figure(labels, data_list, ylim, subtitle, stem):
    """One two-box figure (30hpf vs 48hpf) for a single phase."""
    xpos = [1.0 + i * X_STEP for i in range(len(labels))]
    colors = [cfg.color_for(CONDITION)] * len(labels)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_prism_box(ax, data_list, xpos, colors, width=BOX_WIDTH,
                   jitter=JITTER, dot_size=DOT_SIZE, lw_scale=LW_SCALE)
    if REF_LINE is not None:
        ax.axhline(REF_LINE, color="0.4", ls="--", lw=0.8 * LW_SCALE, zorder=0)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=FONT["tick"])
    ax.set_ylabel(YLABEL, fontsize=FONT["ylabel"])
    ax.set_title(TITLE_MAIN + chr(10) + subtitle,
                 fontsize=FONT["title"], pad=FONT["title_pad"])
    style_axes(ax)
    ax.tick_params(axis="y", labelsize=FONT["tick"])
    apply_dense_yticks(ax, nbins=YTICK_NBINS)
    ax.set_ylim(*ylim)

    # Welch t-test between the two stages (embryo unit), drawn as a bracket
    p = float("nan")
    if len(data_list) == 2 and all(len(v) >= 2 for v in data_list):
        try:
            p = float(ttest_ind(data_list[0], data_list[1],
                                equal_var=False, nan_policy="omit").pvalue)
        except Exception:
            p = float("nan")
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        hi = max(float(np.max(v)) for v in data_list if len(v))
        yb = hi + 0.05 * yr
        h = 0.022 * yr
        ax.plot([xpos[0], xpos[0], xpos[1], xpos[1]],
                [yb, yb + h, yb + h, yb], color="black",
                lw=cfg.STATS.get("bracket_lw", 1.0) * LW_SCALE)
        dec = cfg.STATS.get("pval_decimals", 3)
        ax.text((xpos[0] + xpos[1]) / 2.0, yb + h, f"p{_fmt_p(p, dec)}",
                ha="center", va="bottom", fontsize=FONT["pval"])

    xl, xr = XMARGIN
    sp = (max(xpos) - min(xpos)) or 1.0
    ax.set_xlim(min(xpos) - xl * sp, max(xpos) + xr * sp)

    fig.tight_layout(pad=0.5)
    for ext in ("svg", "pdf", "png"):
        fig.savefig((OUT_DIR / stem).with_suffix("." + ext),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [lab for lab, _ in STAGES]
    collected, rows_out = {}, []

    for spec in PHASES:
        ph, tmax = spec["phase"], spec["tmax"]
        win = "whole phase" if tmax is None else f"0-{tmax:g} min"
        print(f"\n=== phase={ph} ({win}) ===")
        per_stage = []
        for lab, keys in STAGES:
            print(f"  [{lab}] pooling {keys}")
            d = load_stage(keys, ph, tmax)
            per_stage.append(d["value"].to_numpy(float) if len(d) else np.array([]))
            for _, r in d.iterrows():
                rows_out.append(dict(phase=ph, window=win, stage=lab,
                                     dataset=r["dataset"], batch_id=r["batch_id"],
                                     embryo_id=r["embryo_id"],
                                     condition=CONDITION, genotype=GENOTYPE,
                                     n_frames=int(r["n_frames"]),
                                     ratio_median=float(r["value"])))
        collected[spec["stem"]] = (per_stage, spec["subtitle"])

    # shared y-limits across every figure so pre and post are comparable
    allv = np.concatenate([v for per_stage, _ in collected.values()
                           for v in per_stage if len(v)])
    if allv.size == 0:
        print("[ERROR] no data found; nothing to plot.")
        sys.exit(1)
    lo, hi = float(np.min(allv)), float(np.max(allv))
    if REF_LINE is not None:
        lo = min(lo, REF_LINE)
    rng = (hi - lo) or 1.0
    shared = (lo - 0.06 * rng, hi + 0.20 * rng)

    print()
    for stem, (per_stage, subtitle) in collected.items():
        if not SHARE_Y:
            v = np.concatenate([x for x in per_stage if len(x)])
            l2, h2 = float(np.min(v)), float(np.max(v))
            if REF_LINE is not None:
                l2 = min(l2, REF_LINE)
            r2 = (h2 - l2) or 1.0
            ylim = (l2 - 0.06 * r2, h2 + 0.20 * r2)
        else:
            ylim = shared
        p = draw_figure(labels, per_stage, ylim, subtitle, stem)
        parts = "  ".join(
            f"{lab}: n={len(v)} median={np.median(v):.3f}" if len(v) else f"{lab}: n=0"
            for lab, v in zip(labels, per_stage))
        print(f"  {stem:24s} {parts}   Welch p={p:.4g}")

    pd.DataFrame(rows_out).to_csv(OUT_DIR / "main_q1_ratio.csv", index=False)
    print(f"\n[OK] wrote {len(PHASES)} figures + csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
