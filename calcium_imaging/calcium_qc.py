# calcium_qc.py
# -*- coding: utf-8 -*-
"""
Quality-control sentinel for the calcium imaging pipeline
==========================================================

Runs **before** the main pipeline (``calcium_build_tables.py``). Reads the
raw Fiji ``*_allChannels.csv`` exports directly, computes a battery of
per-ROI quality metrics, and writes a report. **Never modifies data, never
auto-removes ROIs, never alters main pipeline behaviour.** The user inspects
the report, fixes problems in Fiji if needed, then re-runs QC and the main
pipeline.

Three severity levels (per ROI × phase):

    FAIL       — data integrity error; user MUST fix before main pipeline
                 (duplicate ROI in Fiji ROI Manager, channel/frame mismatch,
                 post too short)
    FLAG_HIGH  — quality concern; user reviews
                 (>10% post frames with mean_bgsub<0 ⇒ BG drift,
                 bg_pre_post_jump outside [0.85, 1.15], F0 below calibrated
                 threshold)
    FLAG_LOW   — observation only; usually accepted
                 (ROI area changed >2× between pre/post,
                 ACF1 < 0.3, corr(max,mean) < 0.5)

All thresholds live in ``cfg.QC``. Recalibrate by editing cfg, then re-run.

Outputs (under ``cfg.qc_dir()`` = ``ROOT/_qc/``):

    QC_cells.csv           — per (batch, embryo_id, phase, roi_name) row,
                             with every metric's raw value AND status
    QC_summary.txt         — human-readable summary; counts by status,
                             lists every FAIL / FLAG_HIGH item
    QC_plots/<batch>/      — three diagnostic figures per batch:
        overview.png       — cell trace overview (raw, normalised, vDA band,
                             event-density-proxy bar)
        max_vs_mean.png    — scatter: corr(max,mean) vs ACF1(max),
                             coloured by condition
        bg_diagnostic.png  — background ROI trace per embryo

Usage::

    python calcium_qc.py

Reads cfg.ROOT, walks ``<ROOT>/<batch>/<condition>/*allChannels.csv``,
writes everything under cfg.qc_dir().
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calcium_config as cfg


# =============================================================================
# Constants — column names assumed in Fiji exports (these are the column
# names produced by Calcium_ROI_selection_v4.ijm; if you change the macro
# you must change these).
# =============================================================================

COL_MEAN_BGSUB = "mean_bgsub"
COL_MAX_BGSUB  = "max_bgsub"
COL_AREA       = "area"
COL_BG_MEAN    = "bg_mean"
COL_FRAME      = "frame"
COL_ROI        = "roi_name"
COL_CHANNEL    = "channel"

GCAMP_MATCH   = "gcamp"     # case-insensitive channel substring
LIFEACT_MATCH = "lifeact"
CELL_PREFIXES = ("flat", "round")
BG_ROI_NAME   = "BG"


# =============================================================================
# Per-trace helpers (kept small; no dependency on pipeline internals)
# =============================================================================

def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x); mad = np.median(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return float(np.std(x))
    return float(1.4826 * mad)


def acf1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation (Pearson). Returns NaN for very short / flat traces."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    x = x - np.mean(x)
    denom = np.sum(x * x)
    if denom <= 0:
        return float("nan")
    return float(np.sum(x[:-1] * x[1:]) / denom)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def frac_negative(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float((x < 0).sum() / x.size)


def frac_above(x: np.ndarray, k_times_median: float) -> float:
    """Fraction of frames above k × median. Event-density proxy (algorithm-free)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    if med <= 0:
        return float("nan")
    return float((x > k_times_median * med).sum() / x.size)


# =============================================================================
# Discovery and loading
# =============================================================================

def parse_genotype(filename: str) -> str:
    """Genotype token parsed from the filename via cfg.GENOTYPE_ALIASES
    (e.g. 'MIC' / 'piezo_crsp' -> 'Piezo'); falls back to cfg.DEFAULT_GENOTYPE.

    Kept local (a tiny config-driven regex scan) so this QC sentinel stays
    independent of the pipeline internals, while still using the SAME alias
    table as the main build. Without this, two genotypes that share an embryo
    number within a drug (e.g. MIC_e2 and piezo_crsp_e2 under E3) would collide
    onto one embryo_id and corrupt the pre/post pairing."""
    import re
    for canon, patterns in getattr(cfg, "GENOTYPE_ALIASES", {}).items():
        for pat in patterns:
            if re.search(pat, filename):
                return canon
    return getattr(cfg, "DEFAULT_GENOTYPE", "WT")


@dataclass
class FileRecord:
    """One Fiji CSV = one (batch, condition, genotype, embryo, phase) record."""
    path: Path
    batch_id: str          # e.g. "exp1"
    condition: str         # e.g. "ISO" — from folder name, not aliased
    genotype: str          # e.g. "MIC" / "Piezo" / "WT" — from filename
    embryo: str            # e.g. "e3"
    phase: str             # "pre" or "post"
    df: pd.DataFrame = field(repr=False)

    @property
    def embryo_id(self) -> str:
        return f"{self.batch_id}__{self.condition}__{self.genotype}__{self.embryo}"


def discover_csvs(root: Path) -> list[FileRecord]:
    """Walk ROOT/<batch>/<condition>/*allChannels.csv and return parsed records."""
    import re

    files = sorted(root.glob("*/*/*allChannels.csv"))
    if not files:
        # try one level shallower (some users put everything in ROOT directly)
        files = sorted(root.glob("*allChannels.csv"))

    records: list[FileRecord] = []
    for f in files:
        parts = f.relative_to(root).parts
        if len(parts) >= 3:
            batch_id = parts[0]
            condition_dir = parts[1]
        else:
            batch_id = "unknown"
            condition_dir = "unknown"

        emb_match = re.search(r"_(e\d+)_", f.name)
        embryo = emb_match.group(1) if emb_match else "?"

        genotype = parse_genotype(f.name)

        if "_pre_" in f.name:
            phase = "pre"
        elif "_post_" in f.name:
            phase = "post"
        else:
            phase = "?"

        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:
            print(f"  [SKIP] {f.name}: read failed ({e})")
            continue

        records.append(FileRecord(
            path=f, batch_id=batch_id, condition=condition_dir,
            genotype=genotype, embryo=embryo, phase=phase, df=df,
        ))
    return records


# =============================================================================
# FAIL-level metrics — data integrity
# =============================================================================

def check_fails_one_file(rec: FileRecord) -> dict:
    """Return a dict of FAIL-level metric values for one file.

    Keys:
        n_frames_total           — total rows in the file
        duplicate_roi_count      — # of (channel, roi_name, frame) groups with >1 row
        channel_frame_mismatch   — bool: GCaMP frame count != Lifeact frame count
                                   for at least one ROI shared between channels
        post_n_frames            — number of frames in this file (post only)
        post_too_short           — bool: phase==post AND n_frames < cfg.QC['post_min_frames']
    """
    df = rec.df
    out = dict(
        n_frames_total=len(df),
        duplicate_roi_count=0,
        channel_frame_mismatch=False,
        post_n_frames=int(df[COL_FRAME].nunique()) if COL_FRAME in df.columns else 0,
        post_too_short=False,
    )

    if not {COL_CHANNEL, COL_ROI, COL_FRAME}.issubset(df.columns):
        out["duplicate_roi_count"] = -1   # sentinel: can't check
        return out

    # 1. duplicate (channel, roi_name, frame)
    grp = df.groupby([COL_CHANNEL, COL_ROI, COL_FRAME]).size()
    out["duplicate_roi_count"] = int((grp > 1).sum())

    # 2. channel frame count mismatch per ROI
    fc = df.groupby([COL_ROI, COL_CHANNEL])[COL_FRAME].nunique().unstack(fill_value=0)
    if fc.shape[1] >= 2:
        # for each ROI, max - min across channels
        mismatch_per_roi = (fc.max(axis=1) - fc.min(axis=1)) > 0
        out["channel_frame_mismatch"] = bool(mismatch_per_roi.any())

    # 3. post too short
    if rec.phase == "post":
        n_frames = int(df[COL_FRAME].nunique())
        out["post_n_frames"] = n_frames
        if n_frames < cfg.QC["post_min_frames"]:
            out["post_too_short"] = True

    return out


# =============================================================================
# Per-ROI metric collection — runs over GCaMP channel, both phases
# =============================================================================

def extract_cell_traces(rec: FileRecord) -> pd.DataFrame:
    """Pull GCaMP rows for cell ROIs (flat/round). Returns sorted-by-frame frame."""
    df = rec.df
    if COL_CHANNEL not in df.columns:
        return df.iloc[0:0]
    g = df[df[COL_CHANNEL].astype(str).str.lower().str.contains(GCAMP_MATCH)]
    cells = g[g[COL_ROI].astype(str).str.lower().str.startswith(CELL_PREFIXES)]
    return cells.sort_values(COL_FRAME).reset_index(drop=True)


def extract_bg_trace(rec: FileRecord, channel: str = "gcamp") -> pd.DataFrame:
    """Pull BG ROI rows for the given channel."""
    df = rec.df
    if COL_CHANNEL not in df.columns:
        return df.iloc[0:0]
    match = GCAMP_MATCH if channel == "gcamp" else LIFEACT_MATCH
    g = df[df[COL_CHANNEL].astype(str).str.lower().str.contains(match)]
    bg = g[g[COL_ROI].astype(str) == BG_ROI_NAME]
    return bg.sort_values(COL_FRAME).reset_index(drop=True)


def per_roi_metrics(rec: FileRecord) -> list[dict]:
    """One dict per ROI in this file (GCaMP cells only). All FLAG_HIGH and
    FLAG_LOW per-ROI metrics are computed here. Pre/post-paired metrics
    (area_ratio, bg_jump) are filled later in a separate pass."""
    cells = extract_cell_traces(rec)
    if len(cells) == 0:
        return []

    rows = []
    for roi, gg in cells.groupby(COL_ROI):
        xmean = gg[COL_MEAN_BGSUB].to_numpy(float) if COL_MEAN_BGSUB in gg else np.array([])
        xmax  = gg[COL_MAX_BGSUB].to_numpy(float)  if COL_MAX_BGSUB  in gg else np.array([])
        area  = float(gg[COL_AREA].iloc[0]) if COL_AREA in gg else float("nan")

        # depending on cfg.CELL_SIGNAL_STAT, pick the main signal for ACF1
        stat = getattr(cfg, "CELL_SIGNAL_STAT", "mean")
        x_main = xmean if stat == "mean" else xmax

        row = dict(
            batch_id=rec.batch_id, condition=rec.condition, genotype=rec.genotype,
            embryo=rec.embryo, embryo_id=rec.embryo_id, phase=rec.phase,
            roi_name=str(roi),
            cell_class=str(roi).split("_")[0],   # flat/round
            n_frames=len(gg),
            area=area,

            # F0 = median of pre signal (only meaningful for pre row;
            # post row gets NaN here, later we'll backfill onto the post
            # row from its paired pre)
            F0_mean = float(np.median(xmean)) if len(xmean) > 0 and rec.phase == "pre" else float("nan"),
            F0_max  = float(np.median(xmax))  if len(xmax)  > 0 and rec.phase == "pre" else float("nan"),

            # frac negative is post-relevant (BG drift symptom)
            frac_post_bgsub_negative = (
                frac_negative(xmean) if rec.phase == "post" and len(xmean) > 0 else float("nan")
            ),

            # signal continuity (uses the active CELL_SIGNAL_STAT trace)
            acf1_main = acf1(x_main),

            # sanity: do max and mean tell the same story for this ROI?
            max_mean_corr = pearson(xmax, xmean) if len(xmax) > 0 and len(xmean) > 0 else float("nan"),

            # event-density proxy (algorithm-free): fraction of frames above k×median
            # computed on whichever stat is active (we report this for both for the bar plot)
            frac_above_mean = (
                frac_above(xmean, cfg.QC["event_density_threshold_x_median"])
                if len(xmean) > 0 else float("nan")
            ),
            frac_above_max = (
                frac_above(xmax, cfg.QC["event_density_threshold_x_median"])
                if len(xmax) > 0 else float("nan")
            ),
        )
        rows.append(row)
    return rows


# =============================================================================
# Pre/post pairing — fills in metrics that need both phases
# =============================================================================

def pair_and_enrich(per_roi: pd.DataFrame, recs: list[FileRecord]) -> pd.DataFrame:
    """Add:
        F0_mean_pre, F0_max_pre — backfilled onto post rows from their paired pre
        roi_area_pre, roi_area_post, roi_area_ratio — per (batch, cond, emb, roi)
        bg_jump_gcamp — post bg_mean median / pre bg_mean median (gcamp channel)
                        (one value per embryo; broadcast onto every roi row of post phase)
    """
    out = per_roi.copy()

    # ---- F0 backfill (post rows inherit their pre's F0) ----
    # Keys include genotype so two genotypes sharing an embryo number within a
    # drug (e.g. MIC_e2 vs piezo_crsp_e2 under E3) pair pre/post independently.
    pre_F0 = (
        out[out["phase"] == "pre"]
        .set_index(["batch_id", "condition", "genotype", "embryo", "roi_name"])
        [["F0_mean", "F0_max"]]
        .rename(columns={"F0_mean": "F0_mean_pre", "F0_max": "F0_max_pre"})
    )
    out = out.join(pre_F0, on=["batch_id", "condition", "genotype", "embryo", "roi_name"])

    # ---- area ratio ----
    pre_area = (
        out[out["phase"] == "pre"]
        .set_index(["batch_id", "condition", "genotype", "embryo", "roi_name"])
        ["area"].rename("area_pre")
    )
    out = out.join(pre_area, on=["batch_id", "condition", "genotype", "embryo", "roi_name"])
    out["roi_area_ratio"] = out["area"] / out["area_pre"]
    # only meaningful on post rows; pre rows get ratio=1 by definition
    out.loc[out["phase"] == "pre", "roi_area_ratio"] = float("nan")

    # ---- BG jump (per embryo, GCaMP channel) ----
    bg_med = {}   # (batch, cond, geno, emb, phase) -> median of bg_mean across frames
    for rec in recs:
        bg = extract_bg_trace(rec, channel="gcamp")
        if len(bg) > 0 and COL_BG_MEAN in bg.columns:
            bg_med[(rec.batch_id, rec.condition, rec.genotype, rec.embryo, rec.phase)] = float(
                np.median(bg[COL_BG_MEAN].dropna())
            )
    bg_jump = {}
    for (b, c, g, e, ph), v in bg_med.items():
        if ph == "post":
            pre_key = (b, c, g, e, "pre")
            if pre_key in bg_med and bg_med[pre_key] > 0:
                bg_jump[(b, c, g, e)] = v / bg_med[pre_key]
    out["bg_jump_gcamp"] = out.apply(
        lambda r: bg_jump.get(
            (r["batch_id"], r["condition"], r["genotype"], r["embryo"]), float("nan")),
        axis=1,
    )
    # only meaningful on post rows
    out.loc[out["phase"] == "pre", "bg_jump_gcamp"] = float("nan")

    return out


# =============================================================================
# Status assignment — thresholds → PASS / FLAG_LOW / FLAG_HIGH / FAIL
# =============================================================================

# Severity order (higher wins)
_SEVERITY_RANK = {"PASS": 0, "FLAG_LOW": 1, "FLAG_HIGH": 2, "FAIL": 3}


def assign_status(table: pd.DataFrame, file_fail_table: pd.DataFrame) -> pd.DataFrame:
    """Add `status_<metric>` columns and an overall `status` column to the
    per-ROI table. Reasons are accumulated in `status_reasons`."""
    qc = cfg.QC
    out = table.copy()
    n = len(out)

    # Initialise per-metric status
    out["status_F0"] = "PASS"
    out["status_frac_bgsub_neg"] = "PASS"
    out["status_bg_jump"] = "PASS"
    out["status_area_ratio"] = "PASS"
    out["status_acf1"] = "PASS"
    out["status_max_mean_corr"] = "PASS"
    out["status_file_fail"] = "PASS"     # propagated from file_fail_table

    # ---- FLAG_HIGH: F0_low ----
    stat = getattr(cfg, "CELL_SIGNAL_STAT", "mean")
    if stat == "mean":
        thr_F0 = qc["F0_low_mean"]
        F0_col = "F0_mean_pre"
    else:
        thr_F0 = qc["F0_low_max"]
        F0_col = "F0_max_pre"
    out.loc[out[F0_col] < thr_F0, "status_F0"] = "FLAG_HIGH"

    # ---- FLAG_HIGH: frac_post_bgsub_negative ----
    out.loc[out["frac_post_bgsub_negative"] > qc["frac_post_bgsub_negative"],
            "status_frac_bgsub_neg"] = "FLAG_HIGH"

    # ---- FLAG_HIGH: bg_jump ----
    bg = out["bg_jump_gcamp"]
    out.loc[(bg > qc["bg_pre_post_jump_high"]) | (bg < qc["bg_pre_post_jump_low"]),
            "status_bg_jump"] = "FLAG_HIGH"

    # ---- FLAG_LOW: area_ratio ----
    a = out["roi_area_ratio"]
    out.loc[(a > qc["roi_area_ratio_high"]) | (a < qc["roi_area_ratio_low"]),
            "status_area_ratio"] = "FLAG_LOW"

    # ---- FLAG_LOW: acf1 ----
    out.loc[out["acf1_main"] < qc["acf1_low"], "status_acf1"] = "FLAG_LOW"

    # ---- FLAG_LOW: max_mean_corr ----
    out.loc[out["max_mean_corr"] < qc["max_mean_corr_low"],
            "status_max_mean_corr"] = "FLAG_LOW"

    # ---- FAIL: file-level integrity, broadcast onto all rows from that file ----
    # key on (batch, condition, genotype, embryo, phase) → file_fail row.
    # genotype is required so MIC_e2 and piezo_crsp_e2 (same drug) stay distinct;
    # without it the index is non-unique and ff.loc[key] returns >1 row.
    ff = file_fail_table.set_index(["batch_id", "condition", "genotype", "embryo", "phase"])
    def file_fail_status(r):
        key = (r["batch_id"], r["condition"], r["genotype"], r["embryo"], r["phase"])
        if key not in ff.index:
            return "PASS"
        row = ff.loc[key]
        if row["duplicate_roi_count"] > 0:
            return "FAIL"
        if row["channel_frame_mismatch"]:
            return "FAIL"
        if row["post_too_short"]:
            return "FAIL"
        return "PASS"
    out["status_file_fail"] = out.apply(file_fail_status, axis=1)

    # ---- Aggregate overall status (max severity across metrics) ----
    status_cols = [c for c in out.columns if c.startswith("status_") and c != "status_reasons"]
    def overall(row):
        worst = "PASS"
        reasons = []
        for c in status_cols:
            s = row[c]
            if _SEVERITY_RANK[s] > _SEVERITY_RANK[worst]:
                worst = s
            if s != "PASS":
                reasons.append(c.replace("status_", "") + "=" + s)
        return pd.Series({"status": worst, "status_reasons": ";".join(reasons)})
    out[["status", "status_reasons"]] = out.apply(overall, axis=1)

    return out


# =============================================================================
# Plots
# =============================================================================

def _color_for_condition(cond: str) -> str:
    """Use cfg.color_for if available, else a fallback."""
    try:
        return cfg.color_for(cond)
    except Exception:
        return "tab:gray"


def plot_max_vs_mean_scatter(per_roi: pd.DataFrame,
                              batch_id: Optional[str],
                              out_path: Path):
    """corr(max,mean) vs ACF1 scatter, coloured by condition.

    If ``batch_id`` is None, plot all batches combined into one figure.
    Otherwise plot just that batch.
    """
    if batch_id is None:
        sub = per_roi[per_roi["phase"] == "post"]
        title_batch = f"all batches: {', '.join(sorted(per_roi['batch_id'].unique()))}"
    else:
        sub = per_roi[(per_roi["batch_id"] == batch_id) & (per_roi["phase"] == "post")]
        title_batch = batch_id

    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cond in sub["condition"].unique():
        s = sub[sub["condition"] == cond]
        # size by volatility proxy: invert ACF1 (low ACF1 = jumpy)
        size = np.clip((1 - s["acf1_main"].fillna(0).clip(-1, 1)) * 60 + 20, 20, 180)
        ax.scatter(
            s["max_mean_corr"], s["acf1_main"],
            s=size, c=_color_for_condition(cond),
            alpha=0.65, edgecolor="k", lw=0.4,
            label=f"{cond} (n={len(s)})",
        )
    # thresholds
    ax.axvline(0.9, color="green", ls=":", lw=1)
    ax.axvline(cfg.QC["max_mean_corr_low"], color="red", ls=":", lw=1)
    ax.axhline(0.5, color="green", ls=":", lw=1)
    ax.axhline(cfg.QC["acf1_low"], color="red", ls=":", lw=1)
    ax.set_xlabel("corr(max, mean)")
    ax.set_ylabel("ACF1 (main signal trace)")
    ax.set_title(
        f"max-vs-mean QC — {title_batch} (post phase, n={len(sub)} ROIs)\n"
        f"dirty zone = lower-left of red lines"
    )
    ax.legend(title="condition", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_cell_overview(recs: list[FileRecord], per_roi: pd.DataFrame,
                       batch_id: str, out_path: Path, signal: str = "max"):
    """4-row overview, post phase, for one signal ('max' or 'mean'):
       row 1: raw <signal>_bgsub traces, one panel per condition
       row 2: trace / own_median (normalised, same signal)
       row 3: vDA band mean_bgsub (always mean for the band — it's not a cell)
       row 4: bar chart of `frac_above_<signal>` per (embryo, roi)
    """
    assert signal in ("max", "mean"), f"signal must be 'max' or 'mean', got {signal!r}"
    sig_col = COL_MAX_BGSUB if signal == "max" else COL_MEAN_BGSUB
    frac_col = "frac_above_max" if signal == "max" else "frac_above_mean"
    sig_label = f"{signal}_bgsub"

    post_recs = [r for r in recs if r.batch_id == batch_id and r.phase == "post"]
    if not post_recs:
        return
    conds = sorted({r.condition for r in post_recs})

    n_cond = len(conds)
    fig = plt.figure(figsize=(4.0 * n_cond, 13))
    gs = fig.add_gridspec(4, n_cond, hspace=0.45, wspace=0.30)

    # row 1+2: raw and normalised per condition (using selected signal)
    for j, cond in enumerate(conds):
        ax_raw = fig.add_subplot(gs[0, j])
        ax_norm = fig.add_subplot(gs[1, j])
        cond_recs = [r for r in post_recs if r.condition == cond]
        for r in cond_recs:
            cells = extract_cell_traces(r)
            for roi, gg in cells.groupby(COL_ROI):
                t = (gg[COL_FRAME].to_numpy() - 1) * 0.5  # 30s = 0.5 min
                x = gg[sig_col].to_numpy(float)
                med = np.median(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else np.nan
                ax_raw.plot(t, x, lw=0.7, alpha=0.7)
                if np.isfinite(med) and med > 0:
                    ax_norm.plot(t, x / med, lw=0.7, alpha=0.7)
        ax_raw.set_title(f"{cond} — raw cell {sig_label}", fontsize=10)
        ax_raw.set_xlabel("time (min)"); ax_raw.set_ylabel(sig_label)
        ax_norm.set_title(f"{cond} — cell / own median ({signal})", fontsize=10)
        ax_norm.set_xlabel("time (min)"); ax_norm.set_ylabel("trace / median")
        ax_norm.axhline(1, color="k", ls=":", lw=0.6)

    # row 3: vDA band mean_bgsub (always mean — the band is not a cell;
    # mean across the wide DA_band ROI is the standard band readout)
    ax_band = fig.add_subplot(gs[2, :])
    for r in post_recs:
        df = r.df
        g = df[df[COL_CHANNEL].astype(str).str.lower().str.contains(GCAMP_MATCH)]
        band = g[g[COL_ROI].astype(str) == "DA_band"]
        if len(band) > 0:
            t = (band[COL_FRAME].to_numpy() - 1) * 0.5
            ax_band.plot(t, band[COL_MEAN_BGSUB].to_numpy(float),
                         color=_color_for_condition(r.condition), lw=1.0, alpha=0.85,
                         label=f"{r.condition} {r.embryo}")
    ax_band.set_title("vDA band — GCaMP mean_bgsub (post phase, always mean)", fontsize=10)
    ax_band.set_xlabel("time (min)"); ax_band.set_ylabel("mean_bgsub")
    # dedupe legend by condition (one entry per condition, not per embryo)
    h, l = ax_band.get_legend_handles_labels()
    seen = set(); items = []
    for hh, ll in zip(h, l):
        c = ll.split()[0]
        if c not in seen:
            seen.add(c); items.append((hh, c))
    if items:
        ax_band.legend([i[0] for i in items], [i[1] for i in items],
                       fontsize=8, ncol=min(4, len(items)))

    # row 4: bar chart of frac_above_<signal> per ROI (event-density proxy)
    ax_bar = fig.add_subplot(gs[3, :])
    sub = per_roi[(per_roi["batch_id"] == batch_id) & (per_roi["phase"] == "post")].copy()
    sub = sub.sort_values(["condition", "embryo", "roi_name"])
    labels = [f"{r.condition[:4]} {r.embryo} {r.roi_name}" for r in sub.itertuples()]
    colors = [_color_for_condition(c) for c in sub["condition"]]
    ax_bar.bar(range(len(sub)), sub[frac_col].fillna(0).values, color=colors)
    ax_bar.set_xticks(range(len(sub))); ax_bar.set_xticklabels(labels, rotation=90, fontsize=7)
    ax_bar.axhline(cfg.QC["event_density_warn_frac"],
                   color="k", ls="--", lw=0.8,
                   label=f"warn = {cfg.QC['event_density_warn_frac']}")
    ax_bar.set_ylabel(f"frac frames > {cfg.QC['event_density_threshold_x_median']}×median ({signal})")
    ax_bar.set_title(f"event-density proxy per ROI (post 20min, {sig_label})", fontsize=10)
    ax_bar.legend(fontsize=8)

    fig.suptitle(f"QC overview ({sig_label}) — {batch_id} (post phase)",
                 fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_bg_diagnostic(recs: list[FileRecord], batch_id: str, out_path: Path):
    """Plot BG ROI traces (GCaMP) for every embryo in this batch, both phases.
    Highlights embryos where bg_pre_post_jump triggered."""
    batch_recs = [r for r in recs if r.batch_id == batch_id]
    if not batch_recs:
        return

    # group by condition+genotype+embryo (genotype keeps MIC_e2 and
    # piezo_crsp_e2 in separate panels instead of overlaying two embryos)
    embryos = sorted({(r.condition, r.genotype, r.embryo) for r in batch_recs})
    if not embryos:
        return

    n = len(embryos)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 2.8*rows), squeeze=False)

    for i, (cond, geno, emb) in enumerate(embryos):
        ax = axes[i // cols][i % cols]
        for ph, ls in [("pre", "--"), ("post", "-")]:
            recs_e = [r for r in batch_recs if r.condition == cond
                      and r.genotype == geno and r.embryo == emb and r.phase == ph]
            for r in recs_e:
                bg = extract_bg_trace(r, channel="gcamp")
                if len(bg) > 0:
                    t = (bg[COL_FRAME].to_numpy() - 1) * 0.5
                    ax.plot(t, bg[COL_BG_MEAN].to_numpy(float),
                            color=_color_for_condition(cond), ls=ls, lw=1.0,
                            label=ph)
        ax.set_title(f"{cond} {geno} {emb}", fontsize=9)
        ax.set_xlabel("min"); ax.set_ylabel("BG mean")
        ax.legend(fontsize=7)
    # hide unused
    for k in range(len(embryos), rows*cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"BG ROI traces — {batch_id} (-- pre, — post)", fontsize=12, y=1.0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Summary writer
# =============================================================================

def write_summary(table: pd.DataFrame, file_fail_table: pd.DataFrame, out_path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("QC SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total ROI-phase rows: {len(table)}")
    lines.append(f"Total files: {len(file_fail_table)}")
    lines.append(f"cfg.CELL_SIGNAL_STAT = {getattr(cfg, 'CELL_SIGNAL_STAT', 'mean')!r}")
    lines.append("")

    counts = table["status"].value_counts()
    for s in ["PASS", "FLAG_LOW", "FLAG_HIGH", "FAIL"]:
        lines.append(f"  {s:10s}: {int(counts.get(s, 0))}")
    lines.append("")

    # List FAILs
    fails = table[table["status"] == "FAIL"]
    if len(fails) > 0:
        lines.append("-" * 70)
        lines.append(f"FAIL ROWS ({len(fails)}) — MUST FIX IN FIJI BEFORE PIPELINE:")
        lines.append("-" * 70)
        for _, r in fails.iterrows():
            lines.append(f"  [{r['batch_id']:6s}] {r['condition']:6s} {r['genotype']:6s} "
                         f"{r['embryo']:4s} {r['phase']:4s} {r['roi_name']:10s}  "
                         f"reasons: {r['status_reasons']}")
        lines.append("")

    # List FLAG_HIGHs
    flags_h = table[table["status"] == "FLAG_HIGH"]
    if len(flags_h) > 0:
        lines.append("-" * 70)
        lines.append(f"FLAG_HIGH ROWS ({len(flags_h)}) — REVIEW RECOMMENDED:")
        lines.append("-" * 70)
        for _, r in flags_h.iterrows():
            lines.append(f"  [{r['batch_id']:6s}] {r['condition']:6s} {r['genotype']:6s} "
                         f"{r['embryo']:4s} {r['phase']:4s} {r['roi_name']:10s}  "
                         f"reasons: {r['status_reasons']}")
        lines.append("")

    flags_l = table[table["status"] == "FLAG_LOW"]
    if len(flags_l) > 0:
        lines.append("-" * 70)
        lines.append(f"FLAG_LOW ROWS ({len(flags_l)}) — observation only, listed for audit:")
        lines.append("-" * 70)
        for _, r in flags_l.iterrows():
            lines.append(f"  [{r['batch_id']:6s}] {r['condition']:6s} {r['genotype']:6s} "
                         f"{r['embryo']:4s} {r['phase']:4s} {r['roi_name']:10s}  "
                         f"reasons: {r['status_reasons']}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("Thresholds used (cfg.QC):")
    lines.append("=" * 70)
    for k, v in cfg.QC.items():
        lines.append(f"  {k:35s} = {v}")
    lines.append("")
    lines.append("Output files:")
    lines.append(f"  Per-row table:  QC_cells.csv")
    lines.append(f"  Plots (per batch): QC_plots/<batch>/{{overview,max_vs_mean,bg_diagnostic}}.png")

    out_path.write_text("\n".join(lines), encoding="utf-8")


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

    root = cfg.ROOT
    print(f"QC running on: {root}")
    print(f"  CELL_SIGNAL_STAT = {getattr(cfg, 'CELL_SIGNAL_STAT', 'mean')!r}")

    recs = discover_csvs(root)
    if not recs:
        print(f"[ERROR] No *allChannels.csv found under {root}")
        sys.exit(1)
    print(f"  Files discovered: {len(recs)}")

    # ---- File-level FAIL metrics ----
    file_fail_rows = []
    for rec in recs:
        m = check_fails_one_file(rec)
        m.update(dict(batch_id=rec.batch_id, condition=rec.condition,
                      genotype=rec.genotype, embryo=rec.embryo, phase=rec.phase,
                      path=str(rec.path.relative_to(root))))
        file_fail_rows.append(m)
    file_fail = pd.DataFrame(file_fail_rows)

    # ---- Per-ROI metric collection ----
    rows = []
    for rec in recs:
        rows.extend(per_roi_metrics(rec))
    per_roi = pd.DataFrame(rows)
    if len(per_roi) == 0:
        print("[ERROR] No GCaMP cell ROIs found.")
        sys.exit(1)
    print(f"  ROI-phase rows: {len(per_roi)}")

    # ---- Enrich with pre/post-paired metrics ----
    per_roi = pair_and_enrich(per_roi, recs)

    # ---- Assign status ----
    per_roi = assign_status(per_roi, file_fail)

    # ---- Write outputs ----
    qc_root = cfg.qc_dir()
    qc_root.mkdir(parents=True, exist_ok=True)
    (qc_root / "QC_plots").mkdir(exist_ok=True)

    # CSV: reorder columns for readability
    leading = ["batch_id", "condition", "genotype", "embryo", "embryo_id", "phase",
               "roi_name", "cell_class", "n_frames", "area", "status", "status_reasons"]
    trailing = [c for c in per_roi.columns if c not in leading]
    per_roi = per_roi[leading + trailing]
    per_roi.to_csv(qc_root / "QC_cells.csv", index=False)
    print(f"  Wrote: {qc_root / 'QC_cells.csv'}")

    file_fail.to_csv(qc_root / "QC_files.csv", index=False)
    print(f"  Wrote: {qc_root / 'QC_files.csv'}")

    write_summary(per_roi, file_fail, qc_root / "QC_summary.txt")
    print(f"  Wrote: {qc_root / 'QC_summary.txt'}")

    # ---- Plots per batch ----
    for batch_id in sorted(per_roi["batch_id"].unique()):
        bdir = qc_root / "QC_plots" / batch_id
        bdir.mkdir(parents=True, exist_ok=True)
        plot_max_vs_mean_scatter(per_roi, batch_id, bdir / "max_vs_mean.png")
        plot_cell_overview(recs, per_roi, batch_id, bdir / "overview_max.png", signal="max")
        plot_cell_overview(recs, per_roi, batch_id, bdir / "overview_mean.png", signal="mean")
        plot_bg_diagnostic(recs, batch_id, bdir / "bg_diagnostic.png")
        print(f"  Wrote plots: {bdir}/")

    # ---- All-batches combined max-vs-mean scatter ----
    if per_roi["batch_id"].nunique() > 1:
        all_dir = qc_root / "QC_plots" / "ALL_batches"
        all_dir.mkdir(parents=True, exist_ok=True)
        plot_max_vs_mean_scatter(per_roi, None, all_dir / "max_vs_mean.png")
        print(f"  Wrote all-batches scatter: {all_dir}/max_vs_mean.png")

    print()
    counts = per_roi["status"].value_counts()
    for s in ["PASS", "FLAG_LOW", "FLAG_HIGH", "FAIL"]:
        print(f"  {s:10s}: {int(counts.get(s, 0))}")

    if int(counts.get("FAIL", 0)) > 0:
        print("\n⚠️  FAIL rows present. Review QC_summary.txt and fix in Fiji before "
              "running the main pipeline.")
    else:
        print("\n✅ No FAIL rows. You may proceed with the main pipeline.")


if __name__ == "__main__":
    main()
