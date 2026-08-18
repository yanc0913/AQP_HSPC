# calcium_build_tables.py
# -*- coding: utf-8 -*-
"""
Calcium Imaging Table Builder and Event Detection
=================================================

Processes Fiji-exported time-series CSVs to build analysis-ready tables.

All tunable parameters now live in ``calcium_config.py``. Edit that file (not
this one) to change paths, drug aliases, peak-detection parameters, etc.

Processing overview
-------------------
1. Read all Fiji CSV exports under ``ROOT``, normalise condition names via the
   alias table, build a long-format table.
2. Optional post-phase frame trimming via per-file rules in
   ``cfg.POST_TRIM_RULES``.
3. Lifeact-based scaling correction: compute M per embryo from the Lifeact
   vDA band. Convention: ``M = median(pre) / median(post)``; applied as
   ``GCaMP_post_corrected = GCaMP_post × M``.
4. vDA/dDA ratio (E3 control only).
5. ΔF/F₀ normalisation: ``trace = signal_corrected / median(signal_corrected_pre)``.
   Because GCaMP values are background-subtracted upstream, F itself represents
   "ΔF relative to background"; we report F/F₀ (pre-median normalised to ~1).
6. Peak detection (single-frame, robust-z + local/prominence gate). Detection
   is run only on the first ``cfg.post_analysis_min()`` minutes of post phase
   so the median/MAD threshold is computed on a "clean" window.
7. Event-rate aggregation: per-cell rate over the analysis window, then
   averaged across cells to give an embryo-unit rate.

UK English spellings throughout (normalise, neighbour, etc.).
"""

from __future__ import annotations

import re
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import calcium_config as cfg


# =============================================================================
# Utility wrappers around cfg (so functions don't reach into cfg.* everywhere)
# =============================================================================

DT_SECONDS     = cfg.DT_SECONDS
ROI_VDA        = cfg.ROI_VDA
ROI_DDA        = cfg.ROI_DDA
ROI_BG         = cfg.ROI_BG
GCAMP_CH_MATCH = cfg.GCAMP_CH_MATCH
LIFE_CH_MATCH  = cfg.LIFE_CH_MATCH
COL_MEAN_BGSUB = cfg.COL_MEAN_BGSUB
COL_MAX_BGSUB  = cfg.COL_MAX_BGSUB

# Peak detection parameters (kept as module-level names for readability in
# the algorithm body; values come from config)
_PD                  = cfg.PEAK_DETECTION
REL_PEAK_FRAC        = _PD["rel_peak_frac"]
ROBUST_Z_K           = _PD["robust_z_k"]
REFRACTORY_FRAMES    = _PD["refractory_frames"]
PROM_WIN             = _PD["prom_win"]
PROM_FRAC            = _PD["prom_frac"]
USE_ROBUST_Z         = _PD["use_robust_z"]
USE_PROM_GATE        = _PD["use_prom_gate"]


def add_pbar(it, total=None, desc=""):
    """Wrap an iterable with tqdm if available; otherwise return as-is."""
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(it, total=total, desc=desc)
    except Exception:
        return it


# =============================================================================
# 1. CSV I/O and metadata parsing
# =============================================================================

def read_imagej_csv(path: Path) -> pd.DataFrame:
    """Read Fiji-exported CSV. Validates required columns are present."""
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    text = [ln for ln in text if ln.strip() != ""]
    df = pd.read_csv(StringIO("\n".join(text)), skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]

    required = ["image", "phase", "channel", "roi_name", "frame",
                "mean", "max", "area", "bg_mean", "mean_bgsub", "max_bgsub"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}\n"
            f"File: {path}\n"
            f"Expected columns from Fiji export: {required}\n"
            f"Found columns: {list(df.columns)}"
        )
    return df


def phase_from_filename(p: Path) -> str:
    """Fallback parser for 'pre' / 'post' when the CSV's phase column is absent."""
    s = p.stem.lower()
    if re.search(r"(^|[^a-z0-9])pre([^a-z0-9]|$)", s):
        return "pre"
    if re.search(r"(^|[^a-z0-9])post([^a-z0-9]|$)", s):
        return "post"
    return "unknown"


def parse_embryo_from_filename(p: Path) -> str:
    """Extract embryo ID (e.g. 'e1') from filename.

    Robustness fix: we first strip the ``_(pre|post)_<drug>_`` segment so the
    drug token cannot be misread as the embryo when it happens to look like
    one (e.g. drug='E3', embryo regex is `(e\\d+)`, IGNORECASE).

    With the typical filename ``..._e1_post_E3_...``, re.search would still
    pick the leftmost match (e1) — so behaviour is unchanged. The fix only
    matters if the filename ever has drug-before-embryo ordering, in which
    case the un-fixed regex would silently return the drug token as embryo.
    """
    stem = p.stem
    # Remove "_(pre|post)_<drug>_" to isolate the embryo token
    stem_clean = re.sub(
        r"_(pre|post)_[^_]+_",
        "_",
        stem,
        count=1,
        flags=re.IGNORECASE,
    )
    m = re.search(r"(^|_)(e\d+)(_|$)", stem_clean, flags=re.IGNORECASE)
    return m.group(2).lower() if m else "eNA"


def parse_drug_from_filename(p: Path) -> str:
    """Extract the raw drug token from the filename.

    Captures the token after ``_pre_`` or ``_post_`` and strips a leading
    concentration prefix like ``20uM``, ``1mM``. This is intentionally a
    *raw* token: it preserves variants like ``Yoda1`` vs ``Yoda`` so the
    alias step (``normalize_condition``) can do the canonicalisation in one
    place.
    """
    stem = p.stem
    m = re.search(r"_(pre|post)_([^_]+)_", stem, flags=re.IGNORECASE)
    token = m.group(2) if m else ""

    if token == "":
        # Fallbacks for unusual filenames
        if re.search(r"(^|_)E3($|_)", stem, flags=re.IGNORECASE):
            return "E3"
        if re.search(r"yoda", stem, flags=re.IGNORECASE):
            return "Yoda"
        return "Drug"

    # Strip leading concentration prefix
    token2 = re.sub(r"^\d+(\.\d+)?\s*(uM|um|nM|nm|mM|mm)", "", token, flags=re.IGNORECASE).strip()
    token3 = re.sub(r"[^A-Za-z0-9]+", "", token2)
    return token3 if token3 != "" else "Drug"


def normalize_condition(drug_raw: str) -> str:
    """Canonicalise a raw drug token using ``cfg.CONDITION_ALIASES``.

    Scans patterns in dict-insertion order; first regex match wins.
    Returns the raw token unchanged if no pattern matches (the unknown drug
    will be reported as a warning at the end of build_long_table).
    """
    for canonical, patterns in cfg.CONDITION_ALIASES.items():
        for pat in patterns:
            if re.match(pat, drug_raw):
                return canonical
    return drug_raw


def parse_genotype_from_filename(p: Path) -> str:
    """Canonical genotype from the filename via ``cfg.GENOTYPE_ALIASES``.

    Genotype is encoded only in the filename (no folder convention), e.g.
    ``..._piezo_crsp_e2_pre_E3_...`` -> "Piezo", ``..._UIC_e3_post_Yoda_...``
    -> "UIC". Patterns are *searched* anywhere in the stem (unlike the drug
    token, which is positional); first match in dict order wins. Files matching
    no alias get ``cfg.DEFAULT_GENOTYPE`` so single-background experiments
    collapse to one genotype and behave exactly as before.
    """
    stem = p.stem
    for canonical, patterns in cfg.GENOTYPE_ALIASES.items():
        for pat in patterns:
            if re.search(pat, stem):
                return canonical
    return cfg.DEFAULT_GENOTYPE


def get_batch_id(path: Path, root: Path, levels: int = 1) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) == 0:
        return "batchNA"
    take = min(levels, len(parts) - 1)
    take = max(take, 1)
    return "/".join(parts[:take])


def infer_pair_id_from_batch(batch_dir: Path) -> str:
    """Fallback when the pair folder is missing in the directory hierarchy."""
    try:
        for d in batch_dir.iterdir():
            if d.is_dir() and ("vs" in d.name):
                return d.name
    except Exception:
        pass

    conds = set()
    for p in batch_dir.rglob("*.csv"):
        drug_raw = parse_drug_from_filename(p)
        conds.add(normalize_condition(drug_raw))

    if any(c in conds for c in ["Yoda", "GsMTx"]):
        return "E3_vs_Yoda_vs_GsMTx"
    if any(c in conds for c in ["ISO", "BDM"]):
        return "E3_vs_ISO_vs_BDM"
    return "pairNA"


def get_pair_id(path: Path, root: Path) -> str:
    """Resolve the pair_id from ``ROOT/<batch>/<pair>/<cond>/file.csv``.

    Robust to a missing pair folder (recognised by parts[1] being a known
    condition canonical name).
    """
    rel = path.relative_to(root)
    parts = rel.parts

    if len(parts) >= 4:
        candidate = parts[1]
        if candidate in cfg.all_known_conditions():
            # pair folder is missing; parts[1] is actually a condition
            batch_dir = root / parts[0]
            return infer_pair_id_from_batch(batch_dir)
        return candidate

    if len(parts) >= 2:
        batch_dir = root / parts[0]
        return infer_pair_id_from_batch(batch_dir)

    return "pairNA"


def roi_class(roi_name: str) -> str:
    n = str(roi_name).strip()
    if n == ROI_VDA:
        return "band_vDA"
    if n == ROI_DDA:
        return "band_dDA"
    if n == ROI_BG:
        return "BG"
    s = n.lower()
    if s.startswith("flat"):
        return "flat"
    if s.startswith("round"):
        return "round"
    return "other"


# =============================================================================
# 2. Robust statistics primitives
# =============================================================================

def robust_sigma(x: np.ndarray) -> float:
    """Robust σ estimator: ``1.4826 × MAD``.

    The 1.4826 factor makes σ_robust equal to the sample standard deviation
    when x is drawn from a Gaussian. Falls back to plain std when MAD is zero
    or non-finite (e.g. constant trace).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0 or (not np.isfinite(mad)):
        return float(np.std(x))
    return float(1.4826 * mad)


def baseline_from_pre(pre_values: np.ndarray) -> float:
    """Baseline = median of pre-phase values (ignoring NaN)."""
    pre_values = np.asarray(pre_values, float)
    pre_values = pre_values[np.isfinite(pre_values)]
    if pre_values.size == 0:
        return np.nan
    return float(np.median(pre_values))


def safe_div(a, b):
    """Element-wise (or scalar) division with NaN where denominator is invalid.

    Denominator is considered invalid when it is non-finite (NaN/Inf), zero,
    or negative. The negative-denominator case arises in practice when a
    background ROI drifts over a brighter region than the signal (making
    ``mean_bgsub`` negative); the resulting F/F0 would be misleading rather
    than informative, so we return NaN and let downstream code skip it.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if b.ndim == 0:
        if (not np.isfinite(b)) or b <= 0:
            return np.full_like(a, np.nan, dtype=float)
        return a / float(b)
    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (b > 0)
    out[m] = a[m] / b[m]
    return out


# =============================================================================
# 3. Post-phase trimming
# =============================================================================

def trim_post_sync(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``cfg.POST_TRIM_RULES`` (file-substring → drop_n frames).

    Drops the first ``drop_n`` frames of post-phase recordings matching the
    rule, synchronously across all channels/ROIs of the same embryo. After
    trimming, post-phase frames are re-indexed from 1 and time_min resets to 0.
    """
    if not cfg.POST_TRIM_RULES:
        return df

    out = df.copy()

    drop_map: Dict[Tuple[str, str, str], int] = {}
    post_files = out[out["phase"] == "post"][
        ["batch_id", "pair_id", "embryo_id", "source_csv"]
    ].drop_duplicates()
    for _, r in post_files.iterrows():
        drop_n = 0
        fn = str(r["source_csv"])
        for key, n in cfg.POST_TRIM_RULES.items():
            if key in fn:
                drop_n = max(drop_n, int(n))
        if drop_n > 0:
            k = (str(r["batch_id"]), str(r["pair_id"]), str(r["embryo_id"]))
            drop_map[k] = max(drop_map.get(k, 0), drop_n)

    if not drop_map:
        return out

    keep_chunks = []
    grp_cols = ["batch_id", "pair_id", "embryo_id"]
    for gk, g in out.groupby(grp_cols, sort=False):
        drop_n = int(drop_map.get(tuple(map(str, gk)), 0))

        pre = g[g["phase"] == "pre"].copy()
        post = g[g["phase"] == "post"].copy()

        if drop_n > 0 and not post.empty:
            frames = np.sort(post["frame"].unique())
            cutoff = set(frames[:drop_n])
            post = post[~post["frame"].isin(cutoff)].copy()

        if not post.empty:
            frames2 = np.sort(post["frame"].unique())
            fmap = {int(f): i + 1 for i, f in enumerate(frames2)}
            post["frame"] = post["frame"].map(fmap).astype(int)
            post["time_s"] = (post["frame"] - 1) * DT_SECONDS
            post["time_min"] = post["time_s"] / 60.0

        keep_chunks.append(pre)
        keep_chunks.append(post)

    return pd.concat(keep_chunks, ignore_index=True)


# =============================================================================
# 4. Event detection (single-frame peaks)
# =============================================================================

def detect_events_single_frame(
    x: np.ndarray,
    rel_peak_frac: float = REL_PEAK_FRAC,
    use_robust_z: bool = USE_ROBUST_Z,
    k_z: float = ROBUST_Z_K,
    refractory_frames: int = REFRACTORY_FRAMES,
    use_prominence_gate: bool = USE_PROM_GATE,
    prom_win: int = PROM_WIN,
    prom_frac: float = PROM_FRAC,
    external_threshold: float | None = None,
) -> np.ndarray:
    """Detect peaks (calcium events) on a 1-D trace.

    Returns a boolean array of the same length as ``x``. A frame ``t`` is a
    peak when all three conditions hold:

        (i)   x[t-1] < x[t] >= x[t+1]                              -- shape
        (ii)  neighbour_pass OR prominence_pass                    -- locality
              where:
                 neighbour_pass    : x[t] >= (1+rel_peak_frac) * max(x[t-1], x[t+1])
                 prominence_pass   : x[t] >= (1+prom_frac) * P20(x[t-prom_win:t+prom_win+1])
        (iii) x[t] >= threshold                                    -- global

    The threshold (iii) can be provided externally via ``external_threshold``
    (recommended: ``pre_median + k * sigma_robust(pre)``). When
    ``external_threshold`` is None, the function computes a self-referenced
    threshold ``median(x) + k * robust_sigma(x)`` on the input array (legacy
    behaviour, used by the band-trace event detection in Q1).

    Surviving candidates within ``refractory_frames`` of each other are merged;
    the one with the largest x is kept.

    The OR in (ii) is intentional: the neighbour rule catches sharp single-frame
    spikes; the prominence rule catches broader transients where adjacent
    frames are also elevated. The global robust-z gate (iii) prevents
    high-frequency noise from being classified as events.
    """
    x = np.asarray(x, float)
    n = x.size
    ev = np.zeros(n, dtype=bool)
    if n < 3:
        return ev

    # Threshold (iii)
    if external_threshold is not None and np.isfinite(external_threshold):
        thr_z = float(external_threshold)
    else:
        med = np.nanmedian(x)
        sig = robust_sigma(x)
        if not np.isfinite(sig) or sig == 0:
            sig = np.nanstd(x)
        thr_z = med + k_z * sig if np.isfinite(sig) else np.inf

    cand = []
    for t in range(1, n - 1):
        if not (np.isfinite(x[t - 1]) and np.isfinite(x[t]) and np.isfinite(x[t + 1])):
            continue
        if not (x[t] > x[t - 1] and x[t] >= x[t + 1]):
            continue

        # neighbour ratio
        neigh = max(x[t - 1], x[t + 1])
        neighbor_pass = True
        if np.isfinite(neigh) and neigh > 0:
            neighbor_pass = (x[t] >= (1.0 + rel_peak_frac) * neigh)

        # prominence gate
        prom_pass = True
        if use_prominence_gate:
            a = max(0, t - prom_win)
            b = min(n, t + prom_win + 1)
            local = x[a:b]
            local = local[np.isfinite(local)]
            if local.size == 0:
                prom_pass = False
            else:
                base = np.percentile(local, 20)
                if np.isfinite(base) and base > 0:
                    prom_pass = (x[t] >= (1.0 + prom_frac) * base)
                else:
                    prom_pass = np.isfinite(base) and (x[t] >= base + 1e-12)

        if not (neighbor_pass or prom_pass):
            continue
        if use_robust_z and x[t] < thr_z:
            continue

        cand.append(t)

    if not cand:
        return ev

    cand = sorted(cand)
    kept = []
    i = 0
    while i < len(cand):
        group = [cand[i]]
        j = i + 1
        while j < len(cand) and cand[j] - cand[j - 1] <= refractory_frames:
            group.append(cand[j])
            j += 1
        best = max(group, key=lambda tt: x[tt])
        kept.append(best)
        i = j

    ev[kept] = True
    return ev


def per_peak_duration_local_valley(
    x: np.ndarray,
    peak_indices: np.ndarray,
    dt_min: float,
) -> List[float]:
    """For each detected peak, compute its duration as FWHM relative to local valleys.

    Algorithm (per peak):
      1. Find the lowest point to the LEFT of this peak (bounded on the far
         left by the previous peak, or the start of the trace).
      2. Find the lowest point to the RIGHT (bounded by the next peak, or end
         of the trace).
      3. local_baseline = mean of those two valley values.
      4. half = (peak_value + local_baseline) / 2
      5. Walk outward from the peak in each direction while trace is above
         ``half``; the resulting span is the FWHM.
      6. duration = (right - left + 1) * dt_min

    Rationale (vs. global-threshold time-above):
      * Each peak gets its OWN duration, independent of other peaks
      * When the whole trace is uniformly elevated (e.g. ISO), individual
        spikes still have measurable widths because their "valleys" are
        relative to nearby low points, not the global pre baseline.
      * Returns one duration per peak; aggregate as needed (mean, total).

    Returns: list of durations in minutes, same length as ``peak_indices``.
    """
    if len(peak_indices) == 0:
        return []
    peaks = np.asarray(peak_indices, int)
    n = len(x)
    durations: list[float] = []

    for i, pk in enumerate(peaks):
        # left valley: minimum between previous peak (or start) and current peak
        left_search_start = int(peaks[i - 1]) + 1 if i > 0 else 0
        if pk > left_search_start:
            left_min_idx = left_search_start + int(np.argmin(x[left_search_start:pk]))
        else:
            left_min_idx = left_search_start

        # right valley: minimum between current peak and next peak (or end)
        right_search_end = int(peaks[i + 1]) if i < len(peaks) - 1 else n
        if right_search_end > pk + 1:
            right_min_idx = pk + int(np.argmin(x[pk + 1:right_search_end])) + 1
        else:
            right_min_idx = pk

        local_baseline = (x[left_min_idx] + x[right_min_idx]) / 2.0
        half = (x[pk] + local_baseline) / 2.0

        # walk outward while above half
        left_cross = pk
        while left_cross > left_min_idx and x[left_cross - 1] > half:
            left_cross -= 1
        right_cross = pk
        while right_cross < right_min_idx and x[right_cross + 1] > half:
            right_cross += 1

        durations.append((right_cross - left_cross + 1) * dt_min)

    return durations


# =============================================================================
# 5. Discover + build long table
# =============================================================================

def discover_csvs(root: Path) -> List[Path]:
    """Find all CSV files under root, excluding our own output trees.

    Any directory whose name starts with an underscore is treated as a
    pipeline/tooling output (not source data) and skipped. This covers:
        _py_out_<W>min/    — main pipeline tables and plots
        _qc/               — calcium_qc.py outputs (QC_cells.csv etc.)
        _quick_out/        — throwaway analysis scripts (analyze_piezo_*.py)
        _quick_out_baseline/
    Any future tooling output should also live under an underscore-prefixed
    directory by convention.
    """
    csvs = []
    for p in root.rglob("*.csv"):
        if any(part.startswith("_") for part in p.parts):
            continue
        csvs.append(p)
    return sorted(csvs)


def build_long_table(csvs: List[Path]) -> pd.DataFrame:
    """Load all CSVs, attach metadata, concatenate to long format, then trim."""
    required = {"channel", "roi_name", "frame", COL_MEAN_BGSUB, COL_MAX_BGSUB}
    rows = []
    for f in add_pbar(csvs, total=len(csvs), desc="Reading CSVs"):
        df = read_imagej_csv(f)

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{f.name} missing columns: {missing}")

        df = df[df["roi_name"].notna()].copy()
        df["roi_name"] = df["roi_name"].astype(str).str.strip()
        df = df[df["roi_name"] != ""].copy()

        phase = str(df["phase"].iloc[0]).strip().lower() if "phase" in df.columns else "unknown"
        if phase not in ("pre", "post"):
            phase = phase_from_filename(f)

        drug_raw  = parse_drug_from_filename(f)
        condition = normalize_condition(drug_raw)
        embryo    = parse_embryo_from_filename(f)
        genotype  = parse_genotype_from_filename(f)

        batch_id = get_batch_id(f, cfg.ROOT, cfg.BATCH_LEVELS)
        pair_id  = get_pair_id(f, cfg.ROOT)

        # genotype is part of the embryo identity so that, e.g.,
        # UIC_e4_post_E3 and piezo_crsp_e4_post_E3 never collide.
        embryo_id = f"{batch_id}__{pair_id}__{genotype}__{drug_raw}__{embryo}"

        df["source_csv"] = f.name
        df["phase"]      = phase
        df["drug_raw"]   = drug_raw
        df["condition"]  = condition
        df["genotype"]   = genotype
        df["embryo"]     = embryo
        df["embryo_id"] = embryo_id
        df["batch_id"]   = batch_id
        df["pair_id"]    = pair_id

        df["roi_class"] = df["roi_name"].map(roi_class)
        df["frame"]     = df["frame"].astype(int)
        df["time_s"]    = (df["frame"] - 1) * DT_SECONDS
        df["time_min"]  = df["time_s"] / 60.0

        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out = trim_post_sync(out)

    # Report unknown drug names
    known = cfg.all_known_conditions()
    unknown = set(out["condition"].astype(str).unique()) - known
    if unknown:
        print(f"\n⚠️  WARNING: {len(unknown)} condition name(s) did not match any alias in cfg.CONDITION_ALIASES:")
        for u in sorted(unknown):
            n_rows = (out["condition"] == u).sum()
            print(f"     '{u}'  ({n_rows} rows)")
        if cfg.INCLUDE_UNKNOWN_IN_PLOTS:
            print(f"     → They will be plotted in {cfg.COLOR_FALLBACK} (INCLUDE_UNKNOWN_IN_PLOTS=True).")
        else:
            print(f"     → They will be DROPPED from plots (INCLUDE_UNKNOWN_IN_PLOTS=False).")
        print(f"     To suppress this warning, add an alias entry in calcium_config.py.\n")

    return out


# =============================================================================
# 6. Band tables (GCaMP, Lifeact)
# =============================================================================

def filter_channel(df: pd.DataFrame, match: str) -> pd.DataFrame:
    return df[df["channel"].astype(str).str.lower().str.contains(match)].copy()


def build_band_trace(df: pd.DataFrame, channel_label: str) -> pd.DataFrame:
    band = df[df["roi_name"].isin([ROI_VDA, ROI_DDA, ROI_BG])].copy()
    out = band[[
        "batch_id", "pair_id", "condition", "genotype", "drug_raw", "embryo", "embryo_id",
        "phase", "frame", "time_min", "roi_name", "roi_class",
        COL_MEAN_BGSUB, COL_MAX_BGSUB
    ]].copy()
    out = out.rename(columns={
        COL_MEAN_BGSUB: "mean_bgsub",
        COL_MAX_BGSUB:  "max_bgsub",
    })
    out["channel_label"] = channel_label
    return out


# =============================================================================
# 7. Lifeact-based scaling correction
# =============================================================================
# Convention (matches README and DOI):
#
#     M = median(Lifeact_vDA_pre) / median(Lifeact_vDA_post)
#     GCaMP_post_corrected = GCaMP_post × M
#
# Rationale: re-mounting between pre and post often shifts the overall
# fluorescence intensity. The Lifeact channel (structural) provides a per-
# embryo reference, and ``M`` rescales post intensity back to the pre regime.
# =============================================================================

def compute_M_vDA_lifeact(life_band: pd.DataFrame) -> pd.DataFrame:
    """Compute scaling factor M = pre / post (per embryo) on Lifeact vDA."""
    rows = []
    sub = life_band[life_band["roi_name"] == ROI_VDA].copy()
    for (batch_id, pair_id, condition, genotype, embryo_id), g in sub.groupby(
        ["batch_id", "pair_id", "condition", "genotype", "embryo_id"]
    ):
        pre  = g[g["phase"] == "pre"]["mean_bgsub"].values.astype(float)
        post = g[g["phase"] == "post"]["mean_bgsub"].values.astype(float)
        if pre.size == 0 or post.size == 0:
            continue
        pre_b  = baseline_from_pre(pre)
        post_m = float(np.nanmedian(post[np.isfinite(post)])) if np.isfinite(post).any() else np.nan

        # Convention: M = pre / post  (so that signal_corrected = signal × M)
        if np.isfinite(post_m) and post_m != 0:
            M = pre_b / post_m
        else:
            M = np.nan

        rows.append({
            "batch_id":                  batch_id,
            "pair_id":                   pair_id,
            "condition":                 condition,
            "genotype":                  genotype,
            "embryo_id":                 embryo_id,
            "lifeact_vDA_pre_baseline":  pre_b,
            "lifeact_vDA_post_median":   post_m,
            "M_vDA_pre_over_post":       M,
        })
    return pd.DataFrame(rows)


def apply_M_post_multiply(
    df: pd.DataFrame,
    Mtab: pd.DataFrame,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    """Apply Lifeact scaling to post phase only: ``post_corrected = post × M``.

    Pre phase is copied through unchanged.
    """
    out = df.merge(
        Mtab[["batch_id", "pair_id", "condition", "embryo_id", "M_vDA_pre_over_post"]],
        on=["batch_id", "pair_id", "condition", "embryo_id"], how="left"
    ).copy()
    out[out_col] = out[value_col].astype(float)

    is_post = (out["phase"] == "post").values
    M = out["M_vDA_pre_over_post"].values.astype(float)

    ok = is_post & np.isfinite(M) & (M != 0) & np.isfinite(out[out_col].values)
    out.loc[ok, out_col] = out.loc[ok, out_col].values * M[ok]
    return out


def add_pre_anchor_norm(df: pd.DataFrame, signal_col: str, out_col: str) -> pd.DataFrame:
    """Per (batch, pair, cond, embryo, roi), divide by median(pre signal).

    Yields trace_main with pre median ≈ 1, post is the multiplicative deviation.
    """
    out = df.copy()
    out[f"{out_col}_pre_base"] = np.nan
    for (batch_id, pair_id, condition, embryo_id, roi_name), g in out.groupby(
        ["batch_id", "pair_id", "condition", "embryo_id", "roi_name"]
    ):
        pre  = g[g["phase"] == "pre"][signal_col].values.astype(float)
        base = baseline_from_pre(pre)
        out.loc[g.index, f"{out_col}_pre_base"] = base
    out[out_col] = safe_div(out[signal_col].values, out[f"{out_col}_pre_base"].values)
    return out


# =============================================================================
# 8. Q1: vDA/dDA ratio
# =============================================================================

def build_Q1_ratio_and_events_post_only(gcamp_band: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy output: post-only ratio + per-band events for E3 control."""
    post = gcamp_band[
        (gcamp_band["phase"] == "post") &
        (gcamp_band["condition"] == cfg.CONTROL_CONDITION)
    ].copy()
    post = post[post["roi_name"].isin([ROI_VDA, ROI_DDA])].copy()

    v = post[post["roi_name"] == ROI_VDA][
        ["batch_id","pair_id","drug_raw","condition","genotype","embryo","embryo_id",
         "frame","time_min","mean_bgsub","max_bgsub"]
    ].copy()
    d = post[post["roi_name"] == ROI_DDA][
        ["batch_id","pair_id","embryo_id","frame","mean_bgsub","max_bgsub"]
    ].copy()
    d = d.rename(columns={"mean_bgsub": "dDA_mean_bgsub", "max_bgsub": "dDA_max_bgsub"})
    v = v.rename(columns={"mean_bgsub": "vDA_mean_bgsub", "max_bgsub": "vDA_max_bgsub"})
    ratio = v.merge(d, on=["batch_id","pair_id","embryo_id","frame"], how="left")

    denom = ratio["dDA_mean_bgsub"].values.astype(float)
    num   = ratio["vDA_mean_bgsub"].values.astype(float)
    ratio["ratio_mean_bgsub"] = np.where(
        np.isfinite(num) & np.isfinite(denom) & (denom > 0),
        num / denom, np.nan
    )

    rows = []
    for (batch_id, pair_id, genotype, embryo_id), g in post.groupby(
        ["batch_id","pair_id","genotype","embryo_id"]
    ):
        for region, roi in [("vDA", ROI_VDA), ("dDA", ROI_DDA)]:
            seg = g[g["roi_name"] == roi].sort_values("frame")
            x_mean = seg["mean_bgsub"].values.astype(float)
            x_max  = seg["max_bgsub"].values.astype(float)
            if x_mean.size == 0:
                continue

            ev_mean = detect_events_single_frame(x_mean)
            ev_max  = detect_events_single_frame(x_max)

            dur_min = (seg["time_min"].max() - seg["time_min"].min()) + (DT_SECONDS/60.0)
            rate30_mean = ev_mean.sum() / (dur_min/30.0) if dur_min > 0 else np.nan
            rate30_max  = ev_max.sum()  / (dur_min/30.0) if dur_min > 0 else np.nan

            rows.append({
                "batch_id":                   batch_id,
                "pair_id":                    pair_id,
                "condition":                  cfg.CONTROL_CONDITION,
                "genotype":                   genotype,
                "embryo_id":                  embryo_id,
                "region":                     region,
                "roi_name":                   roi,
                "n_events_post_mean":         int(ev_mean.sum()),
                "events_per_30min_post_mean": float(rate30_mean),
                "n_events_post_max":          int(ev_max.sum()),
                "events_per_30min_post_max":  float(rate30_max),
            })

    return ratio, pd.DataFrame(rows)


def build_Q1_ratio_prepost_control(gcamp_band: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pre + post vDA/dDA ratio for E3 control. Summary is per (embryo, phase)."""
    sub = gcamp_band[
        (gcamp_band["condition"] == cfg.CONTROL_CONDITION) &
        (gcamp_band["roi_name"].isin([ROI_VDA, ROI_DDA]))
    ].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    v = sub[sub["roi_name"] == ROI_VDA][
        ["batch_id","pair_id","drug_raw","condition","genotype","embryo","embryo_id",
         "phase","frame","time_min","mean_bgsub","max_bgsub"]
    ].copy()
    d = sub[sub["roi_name"] == ROI_DDA][
        ["batch_id","pair_id","embryo_id","phase","frame","mean_bgsub","max_bgsub"]
    ].copy()
    d = d.rename(columns={"mean_bgsub": "dDA_mean_bgsub", "max_bgsub": "dDA_max_bgsub"})
    v = v.rename(columns={"mean_bgsub": "vDA_mean_bgsub", "max_bgsub": "vDA_max_bgsub"})

    ratio = v.merge(d, on=["batch_id","pair_id","embryo_id","phase","frame"], how="left")

    denom = ratio["dDA_mean_bgsub"].values.astype(float)
    num   = ratio["vDA_mean_bgsub"].values.astype(float)
    ratio["ratio_mean_bgsub"] = np.where(
        np.isfinite(num) & np.isfinite(denom) & (denom > 0),
        num / denom, np.nan
    )

    summ = (
        ratio.groupby(
            ["batch_id","pair_id","condition","genotype","embryo_id","phase"],
            as_index=False
        )
        .agg(ratio_median=("ratio_mean_bgsub","median"))
    )
    return ratio, summ


# =============================================================================
# 9. Q2-cells: traces, events, embryo summary
# =============================================================================

def build_cell_table(gcamp_df: pd.DataFrame) -> pd.DataFrame:
    """Pick out individual cell ROIs (flat/round) and start a cell-only frame.

    The column driving downstream analysis (``cell_signal_raw``) is selected
    by ``cfg.CELL_SIGNAL_STAT``:

      "mean" (default, paper main figure):
          cell_signal_raw = mean_bgsub
          Robust to single-frame artefacts (vesicle flow, alignment jitter).
          See cfg section 4 for the rationale.

      "max" (legacy / internal sanity check):
          cell_signal_raw = max_bgsub
          Captures peak transient amplitude but vulnerable to single-frame
          artefacts when the ROI touches vasculature.

    Both raw columns (``mean_bgsub`` and ``max_bgsub``) are kept in the output
    table regardless, so downstream code or a sanity script can recompute the
    other stat without re-reading the Fiji CSVs.
    """
    cell = gcamp_df[gcamp_df["roi_class"].isin(["flat","round"])].copy()
    out = cell[[
        "batch_id","pair_id","condition","genotype","drug_raw","embryo","embryo_id",
        "phase","frame","time_min",
        "roi_name","roi_class","mean_bgsub","max_bgsub"
    ]].copy()

    stat = getattr(cfg, "CELL_SIGNAL_STAT", "mean")
    if stat == "mean":
        out["cell_signal_raw"] = out["mean_bgsub"].astype(float)
    elif stat == "max":
        out["cell_signal_raw"] = out["max_bgsub"].astype(float)
    else:
        raise ValueError(
            f"cfg.CELL_SIGNAL_STAT must be 'mean' or 'max', got {stat!r}"
        )
    out["cell_signal_stat"] = stat   # provenance: record which one was used
    return out


def compute_cell_events(
    cell_df: pd.DataFrame,
    trace_col: str = "cell_trace_main",
    post_window_min: float | None = None,
) -> pd.DataFrame:
    """Per-cell event metrics on the post analysis window.

    Algorithm
    ---------
    For each cell:

    1. Compute pre-segment threshold:
           threshold = median(pre_trace) + k * sigma_robust(pre_trace)
       where k = cfg.PEAK_DETECTION["robust_z_k"] (default 2.0).

       The threshold is anchored on the PRE phase ("quiet baseline") of the
       same cell. This makes detection sensitive to sustained-activation
       drugs: even when post trace is uniformly elevated, peaks rising above
       pre noise are found.

    2. Detect peaks in the first ``post_window_min`` minutes of post phase,
       using the externally supplied threshold (step 1).

    3. For each detected peak, compute its duration as the FWHM relative to
       local valleys (see ``per_peak_duration_local_valley``).

    4. Compute mean trace amplitude across the post window.

    Columns produced (W = window minutes, e.g. "10")
    ------------------------------------------------
        n_events_postWmin              -- peak count in post window
        events_per_cell_per_Wmin_post  -- = n_events_postWmin (rate per W min)
        mean_amplitude_post_Wmin       -- mean(trace) on post window
        mean_peak_duration_post_Wmin   -- mean of per-peak local-valley FWHM (NaN if no peaks)

    Also kept for compatibility:
        n_events_pre                   -- pre events (uses pre-window self-reference; legacy)
        n_events_post                  -- full-post events (uses post self-reference; legacy)
        events_per_cell_per_30min_post -- legacy rate
        active_post                    -- 1 if any peak in full post (legacy semantics)
    """
    if post_window_min is None:
        post_window_min = cfg.post_analysis_min()

    k_z              = cfg.PEAK_DETECTION["robust_z_k"]
    col_n_events     = cfg.n_events_col()              # n_events_postWmin
    col_per_cell     = cfg.events_col_per_cell()        # events_per_cell_per_Wmin_post
    wsuf             = cfg.window_suffix()
    col_mean_amp     = f"mean_amplitude_post_{wsuf}min"
    col_mean_dur     = f"mean_peak_duration_post_{wsuf}min"

    rows = []
    for (batch_id, pair_id, condition, genotype, embryo_id, roi_name), g in cell_df.groupby(
        ["batch_id","pair_id","condition","genotype","embryo_id","roi_name"]
    ):
        g    = g.sort_values(["phase","frame"])
        pre  = g[g["phase"] == "pre"].sort_values("frame")
        post = g[g["phase"] == "post"].sort_values("frame")

        x_pre  = pre[trace_col].values.astype(float)  if not pre.empty  else np.array([], float)
        x_post = post[trace_col].values.astype(float) if not post.empty else np.array([], float)

        # ----- pre: events on full pre phase (legacy: self-referenced) -----
        ev_pre  = detect_events_single_frame(x_pre)  if x_pre.size  else np.array([], dtype=bool)
        n_pre   = int(ev_pre.sum()) if ev_pre.size else 0

        # ----- full-post legacy (self-referenced threshold on full post) -----
        ev_post_full = detect_events_single_frame(x_post) if x_post.size else np.array([], dtype=bool)
        n_post_full  = int(ev_post_full.sum()) if ev_post_full.size else 0
        if post.empty:
            rate30 = np.nan
        else:
            dur_min_full = (post["time_min"].max() - post["time_min"].min()) + (DT_SECONDS/60.0)
            rate30 = n_post_full / (dur_min_full/30.0) if dur_min_full > 0 else np.nan

        # ----- New (pre-referenced) detection on the analysis window -----
        # Compute pre-segment threshold
        x_pre_finite = x_pre[np.isfinite(x_pre)] if x_pre.size else np.array([], float)
        if x_pre_finite.size >= 3:
            pre_med = float(np.median(x_pre_finite))
            pre_sig = robust_sigma(x_pre_finite)
            if not np.isfinite(pre_sig) or pre_sig == 0:
                pre_sig = float(np.nanstd(x_pre_finite)) if x_pre_finite.size else np.nan
            pre_threshold = pre_med + k_z * pre_sig if np.isfinite(pre_sig) else np.nan
        else:
            pre_threshold = np.nan

        # Slice post to analysis window
        if post.empty:
            n_post_w     = 0
            rate_w       = np.nan
            mean_amp     = np.nan
            mean_peak_dur = np.nan
        else:
            rel  = post["time_min"].astype(float).values - float(post["time_min"].min())
            mask = rel <= float(post_window_min)

            if not mask.any():
                n_post_w = 0; rate_w = np.nan
                mean_amp = np.nan; mean_peak_dur = np.nan
            else:
                x_post_w = x_post[mask]
                rel_w    = rel[mask]
                dur_w    = (rel_w.max() - rel_w.min()) + (DT_SECONDS / 60.0)

                # Mean amplitude (always computable when we have data)
                xw_finite = x_post_w[np.isfinite(x_post_w)]
                mean_amp = float(np.mean(xw_finite)) if xw_finite.size else np.nan

                # Detect with pre-referenced threshold
                if x_post_w.size and np.isfinite(pre_threshold):
                    ev_post_w = detect_events_single_frame(
                        x_post_w,
                        external_threshold=pre_threshold,
                    )
                    n_post_w  = int(ev_post_w.sum())
                    peak_idx  = np.where(ev_post_w)[0]
                    durs = per_peak_duration_local_valley(
                        x_post_w, peak_idx, dt_min=DT_SECONDS / 60.0,
                    )
                    mean_peak_dur = float(np.mean(durs)) if durs else np.nan
                else:
                    n_post_w = 0
                    mean_peak_dur = np.nan

                rate_w = n_post_w / (dur_w / float(post_window_min)) if dur_w > 0 else np.nan

        rows.append({
            "batch_id":   batch_id,
            "pair_id":    pair_id,
            "condition":  condition,
            "genotype":   genotype,
            "embryo_id":  embryo_id,
            "roi_name":   roi_name,
            "cell_class": g["roi_class"].iloc[0],

            # legacy / full-post (self-referenced threshold)
            "n_events_pre":                   n_pre,
            "n_events_post":                  n_post_full,
            "events_per_cell_per_30min_post": float(rate30),
            "active_post":                    int(n_post_full > 0),

            # New: pre-referenced detection on window
            col_n_events:  int(n_post_w),
            col_per_cell:  float(rate_w),
            col_mean_amp:  float(mean_amp) if np.isfinite(mean_amp) else np.nan,
            col_mean_dur:  float(mean_peak_dur) if np.isfinite(mean_peak_dur) else np.nan,
            "pre_threshold": float(pre_threshold) if np.isfinite(pre_threshold) else np.nan,
        })
    return pd.DataFrame(rows)


def embryo_summary_cells(events_cells: pd.DataFrame) -> pd.DataFrame:
    """Average per-cell event metrics within each embryo, separately for
    flat / round cell classes.

    Embryo-level aggregation: nanmean across cells within (batch, pair,
    condition, embryo, cell_class).
    """
    col_per_cell   = cfg.events_col_per_cell()
    col_per_embryo = cfg.events_col_per_embryo()
    wsuf = cfg.window_suffix()
    col_mean_amp_cell    = f"mean_amplitude_post_{wsuf}min"
    col_mean_dur_cell    = f"mean_peak_duration_post_{wsuf}min"
    col_mean_amp_embryo  = f"embryo_mean_amplitude_post_{wsuf}min"
    col_mean_dur_embryo  = f"embryo_mean_peak_duration_post_{wsuf}min"

    rows = []
    for (batch_id, pair_id, condition, genotype, embryo_id, cell_class), g in events_cells.groupby(
        ["batch_id","pair_id","condition","genotype","embryo_id","cell_class"]
    ):
        rows.append({
            "batch_id":   batch_id,
            "pair_id":    pair_id,
            "condition":  condition,
            "genotype":   genotype,
            "embryo_id":  embryo_id,
            "cell_class": cell_class,
            "n_cells":    int(g.shape[0]),

            "mean_events_per_cell_per_30min_post": float(
                np.nanmean(g["events_per_cell_per_30min_post"].values.astype(float))
            ),

            # Pre-referenced detection metrics
            col_per_embryo: float(
                np.nanmean(g[col_per_cell].values.astype(float))
            ),
            col_mean_amp_embryo: float(
                np.nanmean(g[col_mean_amp_cell].values.astype(float))
            ) if col_mean_amp_cell in g.columns else np.nan,
            col_mean_dur_embryo: float(
                np.nanmean(g[col_mean_dur_cell].values.astype(float))
            ) if col_mean_dur_cell in g.columns else np.nan,

            "fraction_active_post": float(
                np.nanmean(g["active_post"].values.astype(float))
            ),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 10. MAIN pipeline
# =============================================================================

def main():
    # Make stdout robust to non-UTF-8 consoles (e.g. Windows cp1252) so the
    # status emoji in the prints below don't raise UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"Using analysis window: {cfg.post_analysis_min()} min  (suffix: '_{cfg.window_suffix()}min')")
    print(f"Output root:           {cfg.out_root()}")

    out_tables = cfg.tables_dir()
    out_tables.mkdir(parents=True, exist_ok=True)

    csvs = discover_csvs(cfg.ROOT)
    print(f"Found CSVs: {len(csvs)}")
    if not csvs:
        raise FileNotFoundError(f"No CSV found under: {cfg.ROOT}")

    data = build_long_table(csvs)

    gcamp = filter_channel(data, GCAMP_CH_MATCH)
    life  = filter_channel(data, LIFE_CH_MATCH)

    gcamp_band = build_band_trace(gcamp, "GCaMP")
    life_band  = build_band_trace(life,  "Lifeact")

    gcamp_band.to_csv(out_tables / "band_traces_long_gcamp.csv", index=False)
    life_band.to_csv (out_tables / "band_traces_long_lifeact.csv", index=False)

    # --- Q1 legacy (post-only) ---
    q1_ratio_post, q1_events_post = build_Q1_ratio_and_events_post_only(gcamp_band)
    q1_ratio_post.to_csv (out_tables / "Q1_ratio_post_control.csv", index=False)
    q1_events_post.to_csv(out_tables / "Q1_band_events_post_control.csv", index=False)

    # --- Q1 new (pre+post ratio for E3) ---
    q1_ratio_prepost, q1_sum_prepost = build_Q1_ratio_prepost_control(gcamp_band)
    q1_ratio_prepost.to_csv(out_tables / "Q1_ratio_prepost.csv", index=False)
    q1_sum_prepost.to_csv  (out_tables / "Q1_ratio_summary_prepost.csv", index=False)

    # --- Q3 vDA band trace (all conditions) ---
    #   Was called "Q2-DA" in v4; reclassified as Q3 in the new naming scheme.
    #   The M_vDA table is shared by Q2 (cells) and Q3 (band), so it carries no
    #   Q-prefix in its filename.
    M_vDA = compute_M_vDA_lifeact(life_band)
    M_vDA.to_csv(out_tables / "M_vDA_from_Lifeact.csv", index=False)

    vda = gcamp_band[gcamp_band["roi_name"] == ROI_VDA].copy()
    vda = apply_M_post_multiply(vda, M_vDA,
                                value_col="mean_bgsub",
                                out_col="gcamp_vDA_mean_bgsub_corr")
    vda = add_pre_anchor_norm(vda,
                              signal_col="gcamp_vDA_mean_bgsub_corr",
                              out_col="gcamp_vDA_trace_main")
    vda.to_csv(out_tables / "Q3_vDA_trace_prepost.csv", index=False)

    # --- Q2 cells: trace + events + summary ---
    cell_raw = build_cell_table(gcamp)

    cell_raw = cell_raw.merge(
        M_vDA[["batch_id","pair_id","condition","embryo_id","M_vDA_pre_over_post"]],
        on=["batch_id","pair_id","condition","embryo_id"], how="left"
    )
    cell_raw["cell_signal_corr"] = cell_raw["cell_signal_raw"].astype(float)
    is_post = (cell_raw["phase"] == "post").values
    M_vals  = cell_raw["M_vDA_pre_over_post"].values.astype(float)
    ok = is_post & np.isfinite(M_vals) & (M_vals != 0) & np.isfinite(cell_raw["cell_signal_corr"].values)
    # Convention: post_corrected = post × M  (B1 fix)
    cell_raw.loc[ok, "cell_signal_corr"] = cell_raw.loc[ok, "cell_signal_corr"].values * M_vals[ok]

    cell_raw["cell_pre_base"] = np.nan
    for (batch_id, pair_id, condition, embryo_id, roi_name), g in cell_raw.groupby(
        ["batch_id","pair_id","condition","embryo_id","roi_name"]
    ):
        pre  = g[g["phase"] == "pre"]["cell_signal_corr"].values.astype(float)
        base = baseline_from_pre(pre)
        cell_raw.loc[g.index, "cell_pre_base"] = base

    cell_raw["cell_trace_main"] = safe_div(
        cell_raw["cell_signal_corr"].values,
        cell_raw["cell_pre_base"].values
    )
    cell_raw.to_csv(out_tables / "Q2_cells_vDA_prepost.csv", index=False)

    events_cells = compute_cell_events(cell_raw, trace_col="cell_trace_main")
    events_cells.to_csv(out_tables / "Q2_events_cells.csv", index=False)

    emb_sum_cells = embryo_summary_cells(events_cells)
    emb_sum_cells.to_csv(out_tables / "Q2_embryo_summary_cells.csv", index=False)

    # ----- Config snapshot -----
    # Save a copy of calcium_config.py next to the tables so you can always
    # tell what parameters this run used. Filename: _config_snapshot.py
    try:
        import shutil
        import datetime
        cfg_path = Path(cfg.__file__)
        snap = cfg.out_root() / "_config_snapshot.py"
        snap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg_path, snap)
        # Prepend a header noting when this snapshot was taken
        header = (
            f"# Snapshot of calcium_config.py at the time of build_tables run.\n"
            f"# Run datetime: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"# Source file:  {cfg_path}\n"
            f"# (This is a frozen copy. Edits here will NOT affect future runs;\n"
            f"#  edit calcium_config.py instead.)\n\n"
        )
        original = snap.read_text(encoding="utf-8")
        snap.write_text(header + original, encoding="utf-8")
        print(f"   Config snapshot: {snap}")
    except Exception as e:
        print(f"   ⚠️  Failed to write config snapshot: {e}")

    print("\n✅ Build complete.")
    print(f"   Tables: {out_tables}")
    print(f"   Event column (per cell):    {cfg.events_col_per_cell()}")
    print(f"   Event column (per embryo):  {cfg.events_col_per_embryo()}")


if __name__ == "__main__":
    main()
