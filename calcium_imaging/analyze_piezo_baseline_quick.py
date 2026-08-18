# analyze_piezo_baseline_quick.py
# -*- coding: utf-8 -*-
"""
Quick analysis for the piezo crispant BASELINE experiment (no drug, single 20-min segment)
==========================================================================================

Throwaway companion script to ``analyze_piezo_quick.py``. Where that one
handles the pre/post + Yoda data, this one handles the no-drug single-
segment data: 20 minutes of imaging on UIC vs piezo crispant, no remount.

Why a separate script: there is no pre phase, so there's no F0, no F/F0
amplitude, and no M_vDA correction. The only readouts we have here are:
  * event counts (using a self-referenced threshold on each trace)
  * trace morphology (line charts)
  * vDA band intensity (rough proxy for "overall calcium tone")

What it shares with the other script:
  * mean_bgsub as the cell signal (matches paper main figure choice)
  * same robust_sigma and detect_events_single_frame imported from
    calcium_build_tables (no re-implementation)
  * k_z = 2.0 (matches cfg.PEAK_DETECTION)

What is different:
  * threshold = median(WHOLE 20-min trace) + 2 * sigma_robust(trace)
    (no "pre" segment available; the whole trace is the trace).
    KNOWN LIMITATION: this means dim cells (smaller absolute sigma) get
    lower absolute thresholds, biasing detection toward more events in
    dim cells. The other conversation that analysed the same data found
    piezo events > UIC events; whether that is biology or this bias is
    not resolvable from event counts alone, so we also plot vDA band
    intensity and per-cell traces so PI can judge by eye.
  * no F/F0, no amplitude boxplot
  * file structure: flat ROOT containing *allChannels.csv (no exp/cond
    subfolders); genotype is parsed from the filename

Run::

    python analyze_piezo_baseline_quick.py /path/to/folder/of/csvs

Outputs (under <DATA_ROOT>/_quick_out_baseline/):
    boxplots/box_events_{flat,round}.png      # only events, no amplitude
    boxplots/box_band_intensity.png            # vDA band mean intensity per embryo
    traces/<embryo>_<roi>.png                  # one line chart per cell
    summary_cells.csv
    summary_embryo.csv
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).parent))
from calcium_build_tables import (
    robust_sigma,
    detect_events_single_frame,
)
import calcium_config as cfg


# =============================================================================
# Local config
# =============================================================================

SIGNAL_COL       = "mean_bgsub"
FRAME_INTERVAL_S = 30.0
K_Z              = cfg.PEAK_DETECTION.get("robust_z_k", 2.0) if hasattr(cfg, "PEAK_DETECTION") else 2.0
GROUP_COLOR      = {"UIC": "#4477AA", "Piezo": "#CC3322"}


# =============================================================================
# File parsing
# =============================================================================

def parse_file(path: Path) -> dict:
    """No folder structure; genotype is parsed from the filename.

    UIC files:    ..._20mins_UIC_e1_RunningBrightest_...
    Piezo files:  ..._20mins_piezo_crsp_e1_RunningBrightest_...
    """
    name = path.name
    if re.search(r"_UIC_e\d+", name):
        genotype = "UIC"
    elif re.search(r"piezo[_\-]?crsp", name, re.IGNORECASE):
        genotype = "Piezo"
    else:
        raise ValueError(f"Cannot determine UIC/Piezo from filename: {name}")

    m = re.search(r"_(e\d+)_", name)
    if not m:
        raise ValueError(f"Cannot find embryo id in {name}")
    embryo = m.group(1)

    return dict(genotype=genotype, embryo=embryo,
                embryo_id=f"{genotype}__{embryo}")


# =============================================================================
# Loading
# =============================================================================

def load_all(root: Path) -> pd.DataFrame:
    """Read every CSV in ROOT (flat layout) and concat with meta cols."""
    files = sorted(root.glob("*allChannels.csv"))
    if not files:
        files = sorted(root.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs found in {root}")
    rows = []
    for f in files:
        meta = parse_file(f)
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        for k, v in meta.items():
            df[k] = v
        df["src_file"] = f.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# Per-cell metrics
# =============================================================================

def per_cell_metrics(long: pd.DataFrame) -> pd.DataFrame:
    """For each (embryo_id, roi_name): event count using self-referenced
    threshold = median + 2*sigma_robust over the WHOLE trace."""
    g_g = long[long["channel"].astype(str).str.lower().str.contains("gcamp")]
    cells = g_g[g_g["roi_name"].astype(str).str.lower().str.startswith(("flat","round"))]

    out = []
    for (emb_id, roi), gg in cells.groupby(["embryo_id", "roi_name"]):
        gg = gg.sort_values("frame")
        meta = gg.iloc[0]
        trace = gg[SIGNAL_COL].dropna().values.astype(float)
        if len(trace) < 5:
            continue

        med = float(np.nanmedian(trace))
        sig = robust_sigma(trace)
        if not np.isfinite(sig) or sig <= 0:
            sig = float(np.nanstd(trace))
        thr = med + K_Z * sig

        ev = detect_events_single_frame(trace, external_threshold=thr)
        n_ev = int(ev.sum())

        out.append(dict(
            embryo_id=emb_id,
            genotype=meta["genotype"],
            embryo=meta["embryo"],
            roi_name=str(roi),
            cell_class="flat" if str(roi).startswith("flat") else "round",
            n_frames=len(trace),
            median=med,
            sigma_robust=sig,
            threshold=thr,
            n_events=n_ev,
            _trace=trace,
            _events=ev,
        ))
    return pd.DataFrame(out)


def per_embryo_band(long: pd.DataFrame) -> pd.DataFrame:
    """Median of GCaMP DA_band mean_bgsub per embryo — a rough 'overall
    calcium tone' read-out."""
    g = long[long["channel"].astype(str).str.lower().str.contains("gcamp")]
    band = g[g["roi_name"].astype(str) == "DA_band"]
    out = []
    for emb_id, gg in band.groupby("embryo_id"):
        meta = gg.iloc[0]
        v = gg[SIGNAL_COL].dropna().values.astype(float)
        if len(v) == 0:
            continue
        out.append(dict(
            embryo_id=emb_id,
            genotype=meta["genotype"],
            embryo=meta["embryo"],
            band_median=float(np.median(v)),
            band_mean=float(np.mean(v)),
        ))
    return pd.DataFrame(out)


# =============================================================================
# Plots
# =============================================================================

def plot_box(per_cell: pd.DataFrame, metric: str, cell_class: str | None,
             ylabel: str, title: str, out: Path):
    """Embryo-level boxplot. If cell_class is None, use the whole table."""
    sub = per_cell if cell_class is None else per_cell[per_cell["cell_class"] == cell_class]
    if len(sub) == 0:
        return
    embryo = sub.groupby(["embryo_id", "genotype"], as_index=False)[metric].mean()
    embryo = embryo.rename(columns={metric: "value"})

    fig, ax = plt.subplots(figsize=(4.2, 5.0))
    groups_order = ["UIC", "Piezo"]
    data = [embryo[embryo["genotype"] == g]["value"].values for g in groups_order]

    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False)
    for box, g in zip(bp["boxes"], groups_order):
        box.set_facecolor(GROUP_COLOR[g]); box.set_alpha(0.5)
        box.set_edgecolor("black"); box.set_linewidth(0.5)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(zip(groups_order, data)):
        if len(vals) == 0:
            continue
        jitter = (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), i+1) + jitter, vals,
                   s=40, facecolor=GROUP_COLOR[g], edgecolor="black",
                   linewidth=0.5, zorder=3)

    ax.set_xticks([1, 2]); ax.set_xticklabels(["UIC", "Piezo"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    a, b = data[0], data[1]
    if len(a) >= 2 and len(b) >= 2:
        _, p = ttest_ind(a, b, equal_var=False)
        ax.text(0.5, 0.97, f"Welch t  p={p:.3f}  (n={len(a)}/{len(b)})",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
    else:
        ax.text(0.5, 0.97, f"n={len(a)}/{len(b)}  (too few for p-value)",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color="darkred")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_cell_traces(per_cell: pd.DataFrame, out_dir: Path):
    """One PNG per cell: the raw mean_bgsub trace, median line, threshold line,
    and detected events as red triangles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, r in per_cell.iterrows():
        trace = r["_trace"]; ev = r["_events"]
        t = np.arange(len(trace)) * (FRAME_INTERVAL_S / 60.0)

        fig, ax = plt.subplots(figsize=(8, 3.3))
        ax.plot(t, trace, color=GROUP_COLOR[r["genotype"]], lw=1.2)
        ax.axhline(r["median"], color="0.5", ls=":", lw=0.7,
                   label=f"median = {r['median']:.1f}")
        ax.axhline(r["threshold"], color="red", ls="--", lw=0.8,
                   label=f"thr = median + {K_Z}σ = {r['threshold']:.1f}")
        ev_idx = np.where(ev)[0]
        for idx in ev_idx:
            ax.plot(t[idx], trace[idx], "rv", markersize=8)
        ax.set_xlabel("time (min)")
        ax.set_ylabel(SIGNAL_COL)
        ax.set_title(f"{r['genotype']} {r['embryo']} {r['roi_name']}  "
                     f"(events = {r['n_events']}, σ_robust = {r['sigma_robust']:.2f})")
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(out_dir / f"{r['embryo_id']}__{r['roi_name']}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    # Make stdout robust to non-UTF-8 consoles (e.g. Windows cp1252) so the
    # status emoji in the prints below don't raise UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path.cwd()
    if not root.exists():
        print(f"[ERROR] data root not found: {root}"); sys.exit(1)

    out_root = root / "_quick_out_baseline"
    out_root.mkdir(exist_ok=True)
    (out_root / "boxplots").mkdir(exist_ok=True)
    (out_root / "traces").mkdir(exist_ok=True)

    print(f"Reading data from: {root}")
    long = load_all(root)
    print(f"  rows: {len(long)}, embryos: {long['embryo_id'].nunique()}")
    for emb in sorted(long["embryo_id"].unique()):
        print(f"    {emb}")

    per_cell = per_cell_metrics(long)
    print(f"\n  cells processed: {len(per_cell)}")
    print(per_cell.groupby(["genotype","cell_class"]).size().to_string())

    band = per_embryo_band(long)

    csv_cols = [c for c in per_cell.columns if not c.startswith("_")]
    per_cell[csv_cols].to_csv(out_root / "summary_cells.csv", index=False)
    embryo_summary = per_cell.groupby(["embryo_id","genotype","embryo","cell_class"],
                                       as_index=False).agg(
        n_cells=("roi_name","count"),
        mean_events=("n_events","mean"),
        mean_median_signal=("median","mean"),
    )
    if len(band):
        embryo_summary = embryo_summary.merge(
            band[["embryo_id","band_median"]], on="embryo_id", how="left")
    embryo_summary.to_csv(out_root / "summary_embryo.csv", index=False)
    print(f"\n  Wrote: {out_root/'summary_cells.csv'}")
    print(f"  Wrote: {out_root/'summary_embryo.csv'}")

    for cls in ["flat", "round"]:
        plot_box(per_cell, "n_events", cls,
                 "events / cell / 20 min",
                 f"{cls} cells — event count (self-threshold)",
                 out_root / "boxplots" / f"box_events_{cls}.png")

    # band intensity boxplot (per embryo, no cell_class split)
    if len(band):
        # massage into the same shape expected by plot_box
        band_for_plot = band.rename(columns={"band_median": "value"}).copy()
        band_for_plot["cell_class"] = "band"
        # Use a small wrapper since plot_box expects per_cell shape
        fig, ax = plt.subplots(figsize=(4.2, 5.0))
        groups_order = ["UIC", "Piezo"]
        data = [band[band["genotype"] == g]["band_median"].values for g in groups_order]
        bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False)
        for box, g in zip(bp["boxes"], groups_order):
            box.set_facecolor(GROUP_COLOR[g]); box.set_alpha(0.5)
            box.set_edgecolor("black"); box.set_linewidth(0.5)
        for m in bp["medians"]:
            m.set_color("black"); m.set_linewidth(0.8)
        rng = np.random.default_rng(0)
        for i, (g, vals) in enumerate(zip(groups_order, data)):
            if len(vals) == 0: continue
            jitter = (rng.random(len(vals)) - 0.5) * 0.18
            ax.scatter(np.full(len(vals), i+1) + jitter, vals,
                       s=40, facecolor=GROUP_COLOR[g], edgecolor="black",
                       linewidth=0.5, zorder=3)
        ax.set_xticks([1,2]); ax.set_xticklabels(["UIC","Piezo"])
        ax.set_ylabel("vDA band median (mean_bgsub)")
        ax.set_title("vDA band intensity (overall calcium tone)", fontsize=11)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        a, b = data[0], data[1]
        if len(a) >= 2 and len(b) >= 2:
            _, p = ttest_ind(a, b, equal_var=False)
            ax.text(0.5, 0.97, f"Welch t  p={p:.3f}  (n={len(a)}/{len(b)})",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9)
        else:
            ax.text(0.5, 0.97, f"n={len(a)}/{len(b)}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9)
        plt.tight_layout()
        fig.savefig(out_root / "boxplots" / "box_band_intensity.png",
                    dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"  Wrote: {out_root/'boxplots'}/")

    plot_cell_traces(per_cell, out_root / "traces")
    print(f"  Wrote: {out_root/'traces'}/  ({len(per_cell)} files)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(embryo_summary.to_string(index=False))
    print()
    print("⚠️  KEY LIMITATION: events use a self-referenced threshold")
    print("    (median + 2σ on the whole trace). Dim cells get lower thresholds")
    print("    in absolute units, which can inflate events in dim cells. The")
    print("    side-by-side band intensity boxplot is provided so the PI can")
    print("    see WHY a difference in events might exist (dim signal vs more")
    print("    biological events).")


if __name__ == "__main__":
    main()
