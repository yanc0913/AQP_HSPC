#!/usr/bin/env python3
"""
qc_max_vs_mean.py — Standalone QC: is max_bgsub a clean cell signal, or is it
                    dominated by single-frame artefacts (vesicles/blood/hot
                    pixels/interpolation jitter)?
================================================================================

WHY THIS EXISTS
---------------
For small hand-drawn cell ROIs, `max_bgsub` (brightest pixel) is often used as
the cell signal on the theory that it captures transient amplitude. But if the
ROI sits on a vessel (flowing bright vesicles/blood), or the stack was SIFT-
aligned WITH interpolation, the brightest pixel can jump frame-to-frame for
reasons unrelated to calcium. This script quantifies whether that is happening,
by comparing the max-pixel trace against the spatial-mean trace per cell.

It is fully self-contained: it does NOT import the analysis pipeline. Drop it
anywhere, point it at a folder of Fiji `*_allChannels.csv` exports, and run.

WHAT IT CHECKS (per cell ROI, GCaMP channel)
--------------------------------------------
1. correlation(max_trace, mean_trace)
       clean cell  -> high (>~0.9). low (<~0.6) means max & mean disagree.
2. frame-to-frame volatility ratio  vol(max)/vol(mean)
       vol = median(|Δtrace|)/median(trace). max >> mean means max is jittery.
3. lag-1 autocorrelation of each trace
       real calcium transients last several frames -> high positive ACF1.
       single-frame spikes (vesicle/hot pixel/interp) -> low/near-zero ACF1.
       If ACF1(max) << ACF1(mean), max "events" are mostly single-frame = noise.
4. ROI area spread (max-pixel bias grows with ROI size; large spread => cells
   not comparable under a max statistic).

It also re-runs a simple event count on both traces (same detector as the
pipeline, inlined here for independence) so you can see if the max-vs-mean
choice flips your event-count result.

USAGE
-----
    python qc_max_vs_mean.py /path/to/folder/with/csvs
    python qc_max_vs_mean.py /path/to/folder --out qc_report.csv --plots qc.png

PRE/POST DATA
-------------
If the CSVs have a meaningful `phase` column (pre/post), pass --by-phase to run
the diagnostic SEPARATELY for pre and post. This matters because a paired
pre/post design partly cancels max artefacts WITHIN a cell, but if blood flow
differs between pre and post (e.g. after remounting) the cancellation is
incomplete — checking each phase separately reveals that.

INTERPRETATION CHEAT-SHEET
--------------------------
    corr high (>0.9), ACF1(max) high     -> max is fine, ROIs are clean
    corr low (<0.6),  ACF1(max) low      -> max is artefact-driven; use mean
    corr ok but ACF1(max) low            -> some real signal but spiky; prefer mean
"""

from __future__ import annotations
import argparse
import sys
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Config knobs (edit if your ROI naming differs)
# ----------------------------------------------------------------------------
GCAMP_CH_MATCH = "gcamp"      # case-insensitive channel substring
CELL_ROI_PREFIXES = ("flat", "round")   # what counts as a "cell" ROI
COL_MAX = "max_bgsub"
COL_MEAN = "mean_bgsub"
COL_AREA = "area"

# event detector defaults (match pipeline)
K_Z = 2.0
REL_PEAK_FRAC = 0.10
PROM_WIN = 2
PROM_FRAC = 0.10
REFRACTORY = 1


# ----------------------------------------------------------------------------
# Minimal, self-contained copies of the verified core functions
# (kept here so this QC tool has zero dependency on the pipeline)
# ----------------------------------------------------------------------------
def robust_sigma(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x); mad = np.median(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return float(np.std(x))
    return float(1.4826 * mad)


def detect_events(x, k_z=K_Z):
    x = np.asarray(x, float); n = x.size
    ev = np.zeros(n, bool)
    if n < 3:
        return ev
    med = np.nanmedian(x); sig = robust_sigma(x)
    if not np.isfinite(sig) or sig == 0:
        sig = np.nanstd(x)
    thr = med + k_z * sig if np.isfinite(sig) else np.inf
    cand = []
    for t in range(1, n - 1):
        if not (np.isfinite(x[t-1]) and np.isfinite(x[t]) and np.isfinite(x[t+1])):
            continue
        if not (x[t] > x[t-1] and x[t] >= x[t+1]):
            continue
        neigh = max(x[t-1], x[t+1])
        neighbor_pass = (x[t] >= (1+REL_PEAK_FRAC)*neigh) if (np.isfinite(neigh) and neigh > 0) else True
        a = max(0, t-PROM_WIN); b = min(n, t+PROM_WIN+1)
        loc = x[a:b]; loc = loc[np.isfinite(loc)]
        if loc.size == 0:
            prom_pass = False
        else:
            base = np.percentile(loc, 20)
            prom_pass = (x[t] >= (1+PROM_FRAC)*base) if (np.isfinite(base) and base > 0) else (np.isfinite(base) and x[t] >= base+1e-12)
        if not (neighbor_pass or prom_pass):
            continue
        if x[t] < thr:
            continue
        cand.append(t)
    if not cand:
        return ev
    cand = sorted(cand); kept = []; i = 0
    while i < len(cand):
        grp = [cand[i]]; j = i+1
        while j < len(cand) and cand[j]-cand[j-1] <= REFRACTORY:
            grp.append(cand[j]); j += 1
        kept.append(max(grp, key=lambda tt: x[tt])); i = j
    ev[kept] = True
    return ev


def acf1(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    x = x - np.mean(x)
    denom = np.sum(x*x)
    return float(np.sum(x[:-1]*x[1:])/denom) if denom > 0 else np.nan


def volatility(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(np.diff(x)))/med) if med > 0 else np.nan


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def read_csv_clean(path):
    text = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.strip() != ""]
    df = pd.read_csv(StringIO("\n".join(text)), skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def analyse_folder(folder, by_phase=False):
    folder = Path(folder)
    csvs = sorted(folder.glob("*allChannels.csv"))
    if not csvs:
        csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs found in {folder}")

    rows = []
    for p in csvs:
        df = read_csv_clean(p)
        need = {"channel", "roi_name", "frame", COL_MAX, COL_MEAN}
        if not need.issubset(df.columns):
            print(f"  [skip] {p.name}: missing {need - set(df.columns)}")
            continue
        df["frame"] = df["frame"].astype(int)
        g = df[df["channel"].astype(str).str.lower().str.contains(GCAMP_CH_MATCH)]
        cells = g[g["roi_name"].astype(str).str.lower().str.startswith(CELL_ROI_PREFIXES)]

        phases = (["__all__"] if not by_phase or "phase" not in df.columns
                  else sorted(cells["phase"].astype(str).unique()))
        for phase in phases:
            sub = cells if phase == "__all__" else cells[cells["phase"].astype(str) == phase]
            for roi, gg in sub.groupby("roi_name"):
                gg = gg.sort_values("frame")
                xmax = gg[COL_MAX].values.astype(float)
                xmean = gg[COL_MEAN].values.astype(float)
                if np.isfinite(xmax).sum() < 3:
                    continue
                area = float(gg[COL_AREA].iloc[0]) if COL_AREA in gg else np.nan
                corr = (np.corrcoef(xmax, xmean)[0, 1]
                        if np.std(xmax) > 0 and np.std(xmean) > 0 else np.nan)
                vmax, vmean = volatility(xmax), volatility(xmean)
                rows.append(dict(
                    file=p.name, phase=phase, roi=roi, area_px=area,
                    corr_max_mean=corr,
                    vol_max=vmax, vol_mean=vmean,
                    vol_ratio=(vmax/vmean if (vmean and np.isfinite(vmean) and vmean > 0) else np.nan),
                    acf1_max=acf1(xmax), acf1_mean=acf1(xmean),
                    n_ev_max=int(detect_events(xmax).sum()),
                    n_ev_mean=int(detect_events(xmean).sum()),
                ))
    return pd.DataFrame(rows)


def print_report(D):
    if D.empty:
        print("No cell ROIs analysed."); return
    print("\n" + "=" * 70)
    print(f"QC SUMMARY  ({len(D)} cell-ROI traces"
          + (f", split by phase" if D['phase'].nunique() > 1 else "") + ")")
    print("=" * 70)

    def block(sub, label):
        print(f"\n[{label}]  n={len(sub)}")
        print(f"  corr(max,mean)      : mean={np.nanmean(sub['corr_max_mean']):.3f}  "
              f"min={np.nanmin(sub['corr_max_mean']):.3f}")
        print(f"  volatility ratio    : mean={np.nanmean(sub['vol_ratio']):.2f}x "
              f"(max vs mean jitter)")
        print(f"  ACF1 max  / mean    : {np.nanmean(sub['acf1_max']):.3f} / "
              f"{np.nanmean(sub['acf1_mean']):.3f}")
        print(f"  events/cell max/mean: {sub['n_ev_max'].mean():.2f} / "
              f"{sub['n_ev_mean'].mean():.2f}")
        if COL_AREA:
            a = sub['area_px'].dropna()
            if len(a):
                print(f"  ROI area px         : median={a.median():.0f} "
                      f"range[{a.min():.0f},{a.max():.0f}]  (spread {a.max()/max(a.min(),1):.1f}x)")
        # verdict
        c = np.nanmean(sub['corr_max_mean']); am = np.nanmean(sub['acf1_max'])
        if np.isfinite(c) and c > 0.9 and am > 0.5:
            v = "max looks CLEAN — safe to use max"
        elif np.isfinite(c) and c < 0.6 and am < 0.35:
            v = "max is ARTEFACT-DRIVEN — use mean_bgsub instead"
        else:
            v = "MIXED — max has some signal but is spiky; mean_bgsub safer"
        print(f"  --> VERDICT: {v}")

    if D['phase'].nunique() > 1:
        for ph, sub in D.groupby('phase'):
            block(sub, f"phase={ph}")
    block(D, "ALL")


def make_plot(D, folder, out_png):
    """Scatter of corr vs ACF1(max), sized by volatility ratio."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot skipped: {e}]"); return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    sc = ax.scatter(D["corr_max_mean"], D["acf1_max"],
                    s=np.clip(D["vol_ratio"].fillna(1)*8, 10, 200),
                    c=D["acf1_mean"], cmap="viridis", alpha=0.8, edgecolor="k", lw=0.4)
    ax.axvline(0.9, color="green", ls=":", lw=1); ax.axvline(0.6, color="red", ls=":", lw=1)
    ax.axhline(0.5, color="green", ls=":", lw=1); ax.axhline(0.35, color="red", ls=":", lw=1)
    ax.set_xlabel("corr(max, mean)"); ax.set_ylabel("ACF1(max)")
    ax.set_title("Each dot = one cell ROI\nsize=volatility ratio, color=ACF1(mean)")
    plt.colorbar(sc, ax=ax, label="ACF1(mean)")
    ax = axes[1]
    ax.scatter(D["n_ev_mean"], D["n_ev_max"], alpha=0.7, edgecolor="k", lw=0.4)
    lim = max(D["n_ev_max"].max(), D["n_ev_mean"].max(), 1)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("events/cell (mean_bgsub)"); ax.set_ylabel("events/cell (max_bgsub)")
    ax.set_title("Event-count agreement\n(off-diagonal = choice changes result)")
    fig.suptitle(f"max-vs-mean QC — {Path(folder).name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"  saved plot: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="QC: max_bgsub vs mean_bgsub for cell ROIs")
    ap.add_argument("folder", help="folder containing Fiji *_allChannels.csv exports")
    ap.add_argument("--by-phase", action="store_true",
                    help="run separately per pre/post phase (for paired datasets)")
    ap.add_argument("--out", default=None, help="write per-cell CSV report here")
    ap.add_argument("--plots", default=None, help="write diagnostic PNG here")
    args = ap.parse_args()

    D = analyse_folder(args.folder, by_phase=args.by_phase)
    print_report(D)
    if args.out:
        D.to_csv(args.out, index=False); print(f"\n  wrote per-cell report: {args.out}")
    if args.plots:
        make_plot(D, args.folder, args.plots)


if __name__ == "__main__":
    main()
