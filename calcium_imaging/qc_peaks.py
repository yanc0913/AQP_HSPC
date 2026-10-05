# qc_peaks.py
# -*- coding: utf-8 -*-
"""
QC: show how every calcium event was selected.

Unlike `calcium_qc.py`, which is a sentinel on the RAW Fiji CSVs and is run
BEFORE the build, this reads the BUILT tables and so is run AFTER it:

    python run_all.py build
    python qc_peaks.py

It writes one contact sheet per condition (per genotype where a dataset has
more than one), one small panel per cell, elongated cells first then round.
Each panel carries everything the detector used to make its decision:

    blue trace      F/F0 over the analysis window
    red band        pre median +- k * sigma_robust(pre). Its UPPER EDGE is the
                    threshold, so the band IS the threshold's construction.
                    The pre median is 1 by construction, since F0 is the pre
                    median, so the band's width is the only cell-specific part.
    green dashes    the local bar, gate (ii) of the detector. That gate accepts
                    on EITHER of two tests, so the bar a candidate must really
                    clear is the lower of
                        (1 + rel_peak_frac) * max(neighbours)
                        (1 + prom_frac)     * P20(+- prom_win frames)
    red arrow       a detected event
    open circle     a candidate the detector looked at and rejected

Showing the rejected candidates is the point. A figure with only the accepted
peaks on it tells you what the detector did, not whether it was right.

NOTHING HERE DECIDES ANYTHING. The events come from
`detect_events_single_frame` imported from calcium_build_tables, and each
cell's count is asserted against `Q2_events_cells.csv` as it is drawn, so a
sheet cannot disagree with the tables. The two bars re-state the detector's own
conditions for drawing only, and the threshold is reconstructed and asserted
equal to the stored `pre_threshold`.

Outputs, under cfg.qc_dir() / "peaks":

    peaks_<condition>[_<genotype>].png   one contact sheet per group
    peak_qc_summary.csv                  one row per cell

The summary carries two flags worth watching:

    never_possible   the cell's whole analysis window sits BELOW its own
                     threshold, so it cannot register an event whatever its
                     shape. These cells are a floor at zero, not a measurement
                     of zero.
    always_above     the whole window sits above the threshold, so the global
                     gate does nothing and the local bar alone selects events.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcium_config as cfg
from calcium_build_tables import detect_events_single_frame, robust_sigma

PD = cfg.PEAK_DETECTION
REL = PD["rel_peak_frac"]
PW = PD["prom_win"]
PF = PD["prom_frac"]
KZ = PD["robust_z_k"]

C_TRACE, C_EVENT, C_THR, C_BAR = "#1a4f8a", "#c0392b", "#c0392b", "#2e8b57"
C_FLAT, C_ROUND = "#1a4f8a", "#8a4f1a"
NCOL = 6


# ----------------------------------------------------------------- drawing --
def local_bar(x: np.ndarray) -> np.ndarray:
    """Gate (ii)'s effective bar at every frame. For drawing only.

    Mirrors detect_events_single_frame, which accepts on neighbour OR
    prominence, so the bar a candidate must clear is the lower of the two.
    """
    n = x.size
    bar = np.full(n, np.nan)
    for t in range(1, n - 1):
        if not (np.isfinite(x[t - 1]) and np.isfinite(x[t]) and np.isfinite(x[t + 1])):
            continue
        neigh = max(x[t - 1], x[t + 1])
        b_neigh = (1.0 + REL) * neigh if (np.isfinite(neigh) and neigh > 0) else -np.inf
        a, b = max(0, t - PW), min(n, t + PW + 1)
        loc = x[a:b][np.isfinite(x[a:b])]
        if loc.size:
            base = np.percentile(loc, 20)
            b_prom = (1.0 + PF) * base if (np.isfinite(base) and base > 0) else -np.inf
            bar[t] = min(b_neigh, b_prom)
        else:
            bar[t] = b_neigh
    return bar


def split_rejected(x: np.ndarray, thr: float, bar: np.ndarray):
    """Shape-passing frames that were rejected, split by which bar stopped them."""
    by_local, by_thr = [], []
    for t in range(1, x.size - 1):
        if not (np.isfinite(x[t - 1]) and np.isfinite(x[t]) and np.isfinite(x[t + 1])):
            continue
        if not (x[t] > x[t - 1] and x[t] >= x[t + 1]):
            continue
        if np.isfinite(bar[t]) and x[t] < bar[t]:
            by_local.append(t)
        elif x[t] < thr:
            by_thr.append(t)
    return by_local, by_thr


def draw_cell(ax, x_pre, x_win, t_win, thr):
    """One panel. Returns (n_events, n_rej_local, n_rej_thr, pre_median, sigma)."""
    xp = x_pre[np.isfinite(x_pre)]
    med = float(np.median(xp)) if xp.size else np.nan
    sig = robust_sigma(xp) if xp.size else np.nan
    if (not np.isfinite(sig) or sig == 0) and xp.size:
        sig = float(np.nanstd(xp))

    if np.isfinite(med) and np.isfinite(sig):
        # The band is the threshold's construction, so it must agree with it.
        assert abs((med + KZ * sig) - thr) < 1e-9, "threshold reconstruction disagrees"
        ax.axhspan(med - KZ * sig, med + KZ * sig, color=C_THR, alpha=0.09, zorder=1)
        ax.axhline(med, lw=0.7, color=C_THR, alpha=0.5, zorder=2)

    ev = detect_events_single_frame(x_win, external_threshold=thr)
    bar = local_bar(x_win)
    rej_local, rej_thr = split_rejected(x_win, thr, bar)

    ax.axhline(thr, ls="--", lw=0.8, color=C_THR, zorder=2)
    ax.plot(t_win, bar, color=C_BAR, lw=0.8, ls=(0, (3, 2)), alpha=0.85, zorder=3)
    ax.plot(t_win, x_win, color=C_TRACE, lw=1.0, zorder=4)

    span = np.nanmax(x_win) - np.nanmin(x_win)
    if not np.isfinite(span) or span == 0:
        span = 1.0
    for t in np.where(ev)[0]:
        ax.annotate("", xy=(t_win[t], x_win[t] + span * 0.05),
                    xytext=(t_win[t], x_win[t] + span * 0.26),
                    arrowprops=dict(arrowstyle="-|>", color=C_EVENT, lw=1.0,
                                    mutation_scale=7), zorder=6)
    for t in rej_local + rej_thr:
        ax.plot(t_win[t], x_win[t], "o", ms=3.0, mfc="none", mec="0.45",
                mew=0.8, zorder=5)

    # headroom so the arrows never collide with the panel title
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, np.nanmax(x_win) + span * 0.42))
    return int(ev.sum()), len(rej_local), len(rej_thr), med, sig


# ------------------------------------------------------------------- sheet --
def window_of(post: pd.DataFrame, w: float):
    post = post.sort_values("time_min")
    x = post["cell_trace_main"].to_numpy(float)
    rel = post["time_min"].to_numpy(float) - float(post["time_min"].min())
    m = rel <= w
    return x[m], rel[m]


def build_sheet(cells, evt, condition, genotype, out_png, w):
    sub = evt[(evt.condition == condition) & (evt.genotype == genotype)].copy()
    sub["_order"] = (sub.cell_class != "flat").astype(int)
    sub = sub.sort_values(["_order", "embryo_id", "roi_name"])
    if sub.empty:
        return [], None

    nrow = int(np.ceil(len(sub) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(2.35 * NCOL, 1.32 * nrow + 0.9),
                             squeeze=False)
    rows, n_ev_tot, n_rej_tot = [], 0, 0
    for i, (_, r) in enumerate(sub.iterrows()):
        ax = axes[i // NCOL][i % NCOL]
        tr = cells[(cells.embryo_id == r.embryo_id) & (cells.roi_name == r.roi_name)]
        x_pre = (tr[tr.phase == "pre"].sort_values("time_min")["cell_trace_main"]
                 .to_numpy(float))
        x_win, t_win = window_of(tr[tr.phase == "post"], w)
        thr = float(r["pre_threshold"])
        if x_win.size < 3 or not np.isfinite(thr):
            ax.axis("off")
            continue

        n_ev, n_loc, n_thr, med, sig = draw_cell(ax, x_pre, x_win, t_win, thr)
        n_ev_tot += n_ev
        n_rej_tot += n_loc + n_thr

        # The sheet must never disagree with the table it documents.
        assert n_ev == int(r["n_events_post20min"]), (
            "qc_peaks disagrees with Q2_events_cells for %s %s: %d vs %d"
            % (r.embryo_id, r.roi_name, n_ev, int(r["n_events_post20min"])))

        finite = x_win[np.isfinite(x_win)]
        rows.append(dict(
            condition=condition, genotype=genotype, cell_class=r.cell_class,
            embryo_id=r.embryo_id, roi_name=r.roi_name,
            n_events=n_ev, n_rejected_local_bar=n_loc, n_rejected_threshold=n_thr,
            pre_median=med, pre_sigma_robust=sig, threshold=thr,
            post_min=float(finite.min()) if finite.size else np.nan,
            post_max=float(finite.max()) if finite.size else np.nan,
            never_possible=bool(finite.size and finite.max() < thr),
            always_above=bool(finite.size and finite.min() >= thr),
        ))
        ax.set_title("%s %s  n=%d  $\\sigma$=%.3f"
                     % (str(r.embryo_id).split("__")[-1], r.roi_name, n_ev, sig),
                     fontsize=6.2, pad=2,
                     color=C_FLAT if r.cell_class == "flat" else C_ROUND)
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.set_xticks([0, w / 2, w])

    for j in range(len(sub), nrow * NCOL):
        axes[j // NCOL][j % NCOL].axis("off")

    label = condition if genotype == "WT" else "%s (%s)" % (condition, genotype)
    fig.suptitle("%s  -  how every event was selected   |   %d cells, %d events, "
                 "%d rejected candidates" % (label, len(rows), n_ev_tot, n_rej_tot),
                 fontsize=10, weight="bold", y=0.998)
    fig.text(0.5, 0.978 if nrow > 4 else 0.962,
             "blue = F/F$_0$   red band = pre median $\\pm$ %g$\\sigma$, upper edge is "
             "the threshold   green = local bar   arrow = event   circle = rejected   "
             "title colour: blue elongated, brown round" % KZ,
             ha="center", fontsize=7, color="0.3")
    fig.tight_layout(rect=[0, 0, 1, 0.972 if nrow > 4 else 0.945])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return rows, (len(rows), n_ev_tot, n_rej_tot)


# -------------------------------------------------------------------- main --
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    tables = cfg.tables_dir()
    out_dir = cfg.qc_dir() / "peaks"
    w = cfg.post_analysis_min()

    f_cells = tables / "Q2_cells_vDA_prepost.csv"
    f_evt = tables / "Q2_events_cells.csv"
    for f in (f_cells, f_evt):
        if not f.exists():
            raise SystemExit(
                "[ERROR] %s not found.\n"
                "        qc_peaks.py reads the BUILT tables; run "
                "`python run_all.py build` first." % f)

    out_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(f_cells)
    evt = pd.read_csv(f_evt)

    print("Peak-selection QC")
    print("  tables : %s" % tables)
    print("  window : %g min   threshold = pre median + %g x robust SD" % (w, KZ))
    print("  out    : %s\n" % out_dir)

    all_rows = []
    for (cond, geno), _ in evt.groupby(["condition", "genotype"]):
        tag = cond if geno == "WT" else "%s_%s" % (cond, geno)
        rows, stats = build_sheet(cells, evt, cond, geno,
                                  out_dir / ("peaks_%s.png" % tag), w)
        all_rows += rows
        if stats:
            n, e, r = stats
            print("  %-16s %3d cells  %4d events  %4d rejected  -> peaks_%s.png"
                  % (tag, n, e, r, tag))

    if not all_rows:
        print("\n[WARN] no cells drawn.")
        return

    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_dir / "peak_qc_summary.csv", index=False)

    n = len(summary)
    never = int(summary.never_possible.sum())
    always = int(summary.always_above.sum())
    loc = int(summary.n_rejected_local_bar.sum())
    thr = int(summary.n_rejected_threshold.sum())
    print("\n  %d cells, %d events" % (n, int(summary.n_events.sum())))
    print("  rejected by the local bar : %5d" % loc)
    print("  rejected by the threshold : %5d" % thr)
    print("  cells whose window is entirely BELOW the threshold (cannot register")
    print("    an event at all, so their zero is a floor) : %d (%.0f%%)"
          % (never, 100.0 * never / n))
    print("  cells entirely ABOVE it (the local bar alone selects) : %d (%.0f%%)"
          % (always, 100.0 * always / n))
    print("\n  wrote %s" % (out_dir / "peak_qc_summary.csv"))

    worst = (summary.groupby(["condition", "genotype"])
             .never_possible.mean().mul(100).round(0).sort_values(ascending=False))
    if len(worst) and worst.iloc[0] >= 50:
        print("\n  [NOTE] groups where most cells cannot register an event:")
        for (c, g), v in worst[worst >= 50].items():
            print("     %-18s %3.0f%%" % (c if g == "WT" else "%s (%s)" % (c, g), v))


if __name__ == "__main__":
    main()
