import argparse
from pathlib import Path
import yaml
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import trackpy as tp

import matplotlib as mpl

mpl.rcParams.update({
    # -------- Font (Illustrator friendly) --------
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # -------- Figure aesthetics --------
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
})


# -----------------------------
# Utils / IO
# -----------------------------
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _boolish(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


# -----------------------------
# Feature extraction from labels
# -----------------------------
def extract_spots_from_labels(labels_3d: np.ndarray, px_um: float) -> pd.DataFrame:
    """
    Extract all objects from (T,Y,X) label stack.
    Output columns:
      frame,label,x_um,y_um,area_um2,eccentricity,solidity,orientation,major_um,minor_um
    """
    rows = []
    T = labels_3d.shape[0]
    for f in range(T):
        lab = labels_3d[f]
        props = regionprops_table(
            lab,
            properties=(
                "label",
                "area",
                "centroid",
                "eccentricity",
                "solidity",
                "orientation",
                "major_axis_length",
                "minor_axis_length",
            ),
        )
        if len(props["label"]) == 0:
            continue

        df = pd.DataFrame(props)
        df["frame"] = f
        df["x_um"] = df["centroid-1"] * px_um
        df["y_um"] = df["centroid-0"] * px_um
        df["area_um2"] = df["area"] * (px_um ** 2)
        df["major_um"] = df["major_axis_length"] * px_um
        df["minor_um"] = df["minor_axis_length"] * px_um

        df = df.drop(
            columns=[
                "centroid-0",
                "centroid-1",
                "area",
                "major_axis_length",
                "minor_axis_length",
            ]
        )
        rows.append(df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "frame",
                "label",
                "x_um",
                "y_um",
                "area_um2",
                "eccentricity",
                "solidity",
                "orientation",
                "major_um",
                "minor_um",
            ]
        )
    return pd.concat(rows, ignore_index=True)


# -----------------------------
# ROI (auto)
# -----------------------------
def estimate_roi_auto(spots: pd.DataFrame, roi_cfg: dict) -> Optional[Tuple[float, float, float, float]]:
    """
    Robust ROI bbox from early frames using x/y quantiles + expand + pads.
    Return (x0,x1,y0,y1) in um.
    """
    if spots.empty:
        return None

    early_frac = float(roi_cfg.get("early_frac", 0.3))
    min_spots = int(roi_cfg.get("min_spots", 20))

    x_expand_frac = float(roi_cfg.get("x_expand_frac", roi_cfg.get("expand_frac", 0.25)))
    y_expand_frac = float(roi_cfg.get("y_expand_frac", roi_cfg.get("expand_frac", 0.25)))
    x_pad_um = float(roi_cfg.get("x_pad_um", 0.0))
    y_pad_um = float(roi_cfg.get("y_pad_um", 0.0))

    T = int(spots["frame"].max()) + 1
    early_max_f = max(0, int(T * early_frac))
    early = spots[spots["frame"] <= early_max_f]
    if len(early) < min_spots:
        early = spots.copy()
    if len(early) < min_spots:
        return None

    xs = early["x_um"].to_numpy()
    ys = early["y_um"].to_numpy()

    x0 = float(np.quantile(xs, 0.02))
    x1 = float(np.quantile(xs, 0.98))
    y0 = float(np.quantile(ys, 0.02))
    y1 = float(np.quantile(ys, 0.98))

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    wx = (x1 - x0) * (1.0 + x_expand_frac) + 2 * x_pad_um
    wy = (y1 - y0) * (1.0 + y_expand_frac) + 2 * y_pad_um

    x0 = cx - wx / 2.0
    x1 = cx + wx / 2.0
    y0 = cy - wy / 2.0
    y1 = cy + wy / 2.0
    return (x0, x1, y0, y1)


def get_roi(spots: pd.DataFrame, roi_cfg: dict, px_um: float) -> Optional[Tuple[float, float, float, float]]:
    """
    Priority:
      1) roi.manual_um
      2) roi.manual_px -> convert
      3) auto
    """
    if not _boolish(roi_cfg.get("enable", True)):
        return None

    if roi_cfg.get("manual_um", None) is not None:
        x0, x1, y0, y1 = roi_cfg["manual_um"]
        return (float(x0), float(x1), float(y0), float(y1))

    if roi_cfg.get("manual_px", None) is not None:
        x0, x1, y0, y1 = roi_cfg["manual_px"]
        return (float(x0) * px_um, float(x1) * px_um, float(y0) * px_um, float(y1) * px_um)

    return estimate_roi_auto(spots, roi_cfg)


def apply_roi_filter(spots: pd.DataFrame, roi: Optional[Tuple[float, float, float, float]]) -> pd.DataFrame:
    if roi is None or spots.empty:
        return spots.copy()
    x0, x1, y0, y1 = roi
    m = (
        (spots["x_um"] >= x0) & (spots["x_um"] <= x1) &
        (spots["y_um"] >= y0) & (spots["y_um"] <= y1)
    )
    return spots[m].copy()


# -----------------------------
# Midline fit + band filtering
# -----------------------------
def fit_midline_tls(points_xy: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Total least squares line fit in 2D.
    Return (p0, v) where:
      p0 = centroid (2,)
      v  = unit direction vector along line (2,)
    """
    if points_xy is None or len(points_xy) < 2:
        return None
    p0 = points_xy.mean(axis=0)
    X = points_xy - p0
    C = X.T @ X
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    v = v / norm
    return p0, v


def point_line_distance_um(x: np.ndarray, y: np.ndarray, p0: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Distance from points to line in 2D.
    points = (x,y), line = p0 + t*v
    """
    P = np.stack([x, y], axis=1)
    W = P - p0.reshape(1, 2)
    proj = (W @ v.reshape(2, 1)).reshape(-1, 1) * v.reshape(1, 2)
    perp = W - proj
    return np.sqrt(np.sum(perp ** 2, axis=1))


def estimate_x_center_mode(spots: pd.DataFrame, fit_early_frac: float, min_points: int, n_bins: int = 64) -> Optional[float]:
    """
    Estimate midline x-center by histogram MODE (peak) in early frames.
    More robust than median(x) when left/right are imbalanced.
    """
    if spots is None or spots.empty:
        return None

    T = int(spots["frame"].max()) + 1
    fit_max = max(0, int(np.floor(T * fit_early_frac)))
    early = spots[spots["frame"] <= fit_max].copy()
    if len(early) < min_points:
        early = spots.copy()
    if len(early) < min_points:
        return None

    xs = early["x_um"].to_numpy(dtype=float)
    if not np.isfinite(xs).any():
        return None

    x_min = float(np.nanmin(xs))
    x_max = float(np.nanmax(xs))
    if x_max <= x_min + 1e-9:
        return float(np.nanmedian(xs))

    hist, edges = np.histogram(xs, bins=int(n_bins), range=(x_min, x_max))
    if hist.sum() <= 0:
        return float(np.nanmedian(xs))

    k = int(np.argmax(hist))
    x0 = edges[k]
    x1 = edges[k + 1]
    return float(0.5 * (x0 + x1))


def fit_midline_from_y_binned_x_median(
    df: pd.DataFrame,
    n_bins: int = 25,
    y_q_low: float = 0.05,
    y_q_high: float = 0.95,
    min_bin_points: int = 15
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Robust line fit:
    - bin points by y
    - take median x per bin as center
    - TLS fit through centers
    """
    if df is None or df.empty:
        return None

    x = df["x_um"].to_numpy(dtype=float)
    y = df["y_um"].to_numpy(dtype=float)
    if len(x) < 5:
        return None

    y0 = float(np.quantile(y, y_q_low))
    y1 = float(np.quantile(y, y_q_high))
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
        return None

    edges = np.linspace(y0, y1, int(n_bins) + 1)
    centers = []

    for i in range(int(n_bins)):
        ya = edges[i]
        yb = edges[i + 1]
        m = (y >= ya) & (y < yb)
        if np.sum(m) < int(min_bin_points):
            continue
        xm = float(np.median(x[m]))
        yc = 0.5 * (ya + yb)
        centers.append([xm, yc])

    if len(centers) < 5:
        return None

    pts = np.array(centers, dtype=float)
    return fit_midline_tls(pts)


def angle_deg_from_v(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def estimate_midline_tls_from_preband(
    spots: pd.DataFrame,
    fit_early_frac: float,
    min_points: int,
    preband_half_width_um: float,
    x_center_mode_bins: int = 64,
    y_mid_q0: float = 0.20,
    y_mid_q1: float = 0.80,
    ybin_fit_bins: int = 25,
    ybin_min_bin_points: int = 15,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, str]]:
    """
    Two-stage midline estimation:
      1) x_center by x-histogram MODE (early frames)
      2) select points within preband |x-x_center|<=preband_half_width_um (and mid-y range)
      3) robust fit by y-binned median-x centers, fallback to TLS on all cand points
    Returns (p0, v, x_center, fit_mode).
    """
    if spots is None or spots.empty:
        return None

    x_center = estimate_x_center_mode(spots, fit_early_frac=fit_early_frac, min_points=min_points, n_bins=x_center_mode_bins)
    if x_center is None:
        return None

    ys_all = spots["y_um"].to_numpy(dtype=float)
    y0 = float(np.quantile(ys_all, y_mid_q0))
    y1 = float(np.quantile(ys_all, y_mid_q1))

    T = int(spots["frame"].max()) + 1
    fit_max = max(0, int(np.floor(T * fit_early_frac)))
    early = spots[spots["frame"] <= fit_max].copy()
    if len(early) < min_points:
        early = spots.copy()

    cand = early[
        (early["y_um"] >= y0) & (early["y_um"] <= y1) &
        (np.abs(early["x_um"] - x_center) <= float(preband_half_width_um))
    ].copy()

    if len(cand) < min_points:
        cand = early[np.abs(early["x_um"] - x_center) <= float(preband_half_width_um)].copy()
    if len(cand) < min_points:
        return None

    # 1) robust fit via y-binned x-median centers
    fit = fit_midline_from_y_binned_x_median(
        cand,
        n_bins=int(ybin_fit_bins),
        y_q_low=0.05,
        y_q_high=0.95,
        min_bin_points=int(ybin_min_bin_points),
    )
    if fit is not None:
        p0, v = fit
        return p0, v, float(x_center), "y_binned_x_median_tls"

    # 2) fallback TLS on all candidate points
    pts = cand[["x_um", "y_um"]].dropna().to_numpy(dtype=float)
    if len(pts) < min_points:
        return None
    fit2 = fit_midline_tls(pts)
    if fit2 is None:
        return None
    p0, v = fit2
    return p0, v, float(x_center), "fallback_tls_on_candidates"


# -----------------------------
# Bilateral midline estimator (default)
# -----------------------------
# The LPM forms two bilateral bands of nuclei that converge on the midline, so
# the midline is the GAP BETWEEN the two dense bands. This estimator locates
# that gap directly: it builds the density from nucleus-sized objects, picks the
# pair of peaks that behaves like the two bands, and takes their midpoint.
#
# Selected with midline_tls.method: "bilateral" (the default). The simpler
# estimate_x_center_mode() above remains available as "mode_tls".
#
# Fits are checked against hand-drawn midline regions with midline_v3_check.py,
# which writes a per-movie diagnostic figure without touching the results tree.
_MID_STEP = 1.0     # um, density grid spacing
_MID_BW = 8.0       # um, density smoothing


def _density_1d(u, bw_um=_MID_BW, step=_MID_STEP):
    """Smoothed 1-D density of u on a regular grid."""
    u = np.asarray(u, dtype=float)
    if len(u) < 5:
        return None, None
    grid = np.arange(float(u.min()), float(u.max()) + step, step)
    if len(grid) < 5:
        return None, None
    c, _ = np.histogram(u, bins=len(grid), range=(grid[0], grid[-1] + step))
    return grid, gaussian_filter1d(c.astype(float), bw_um / step)


def _two_band_midpoint(u, min_sep_um):
    """Midpoint of the two bilateral bands in x, or None if there are not two.

    Every PAIR of peaks is scored, rather than simply taking the two tallest: a
    movie often has three peaks, the two bands plus the cells that have already
    converged on the midline, and the middle one can be as tall as a band.

    A pair is scored on what the two bilateral bands must satisfy:
      - both peaks tall and the valley between them deep   -> min(h) - valley
      - the two bands BRACKET the tissue: little mass left  -> 1 - frac_outside
        outside the pair
    The second term is what distinguishes (band, band) from (band, midline
    cells): the latter leaves a whole band outside the pair.
    """
    grid, dens = _density_1d(u)
    if grid is None:
        return None
    pk, _ = find_peaks(dens, distance=max(3, int(min_sep_um / _MID_STEP)))
    if len(pk) < 2:
        return None
    tot = float(dens.sum())
    if tot <= 0:
        return None

    best, best_s = None, -np.inf
    for a in range(len(pk)):
        for b in range(a + 1, len(pk)):
            i, j = int(pk[a]), int(pk[b])
            if (grid[j] - grid[i]) < min_sep_um:
                continue
            valley = float(dens[i:j + 1].min())
            depth = min(dens[i], dens[j]) - valley
            frac_out = float((dens.sum() - dens[i:j + 1].sum()) / tot)
            s = depth * (1.0 - frac_out)
            if s > best_s:
                best_s, best = s, (i, j)
    if best is None:
        return None
    i, j = best
    return dict(x=0.5 * (grid[i] + grid[j]), left=float(grid[i]),
                right=float(grid[j]), hL=float(dens[i]), hR=float(dens[j]), score=float(best_s))


def _gap_sharpness(u, x_cut, look_um=150.0):
    """Tall density on BOTH sides within look_um, low density at the cut.

    This is the angle objective. The cut is pinned to the global two-band
    anchor, so the sweep can only answer "which tilt" - it can never wander off
    to a wide empty region, which is how a free-position variant failed during
    development.
    """
    grid, dens = _density_1d(u)
    if grid is None or dens.max() <= 0:
        return float("-inf")
    i = int(np.clip(np.searchsorted(grid, x_cut), 1, len(grid) - 2))
    w = int(look_um / _MID_STEP)
    lo, hi = max(0, i - w), min(len(dens), i + w + 1)
    if i - lo < 5 or hi - i < 5:
        return float("-inf")
    return float(min(dens[lo:i].max(), dens[i + 1:hi].max()) - dens[i])


def estimate_midline_bilateral(spots, tcfg, mcfg):
    """Midline as the gap between the two bilateral nuclear bands.

    1. Density from NUCLEUS-SIZED objects only. The all-object profile is
       dominated by sub-nuclear debris, which is densest exactly in the midline
       region.
    2. Position = midpoint of the two dominant bands.
    3. Tilt from a rotation sweep scored by how sharply the gap separates the
       bands, then the position is RE-DERIVED in the rotated frame: the anchor
       was measured on a projection along a different axis, so a tilted line
       that kept it would inherit an offset belonging to the vertical one.
    4. Rotation pivot is the centre of the y RANGE, not median(y). The nuclei
       are skewed down the field; median(y) put the pivot at y~560 of a 60-650
       field on WT 20250616, so any tilt swung the whole line off the midline
       even when the angle itself was right.

    Returns None if two bands cannot be found, so the caller leaves the spots
    untouched rather than guessing.
    """
    if spots is None or spots.empty or "area_um2" not in spots.columns:
        return None

    amin = float(tcfg.get("min_area_um2", 20.0))
    amax = float(tcfg.get("max_area_um2", 60.0))
    early_frac = float(mcfg.get("fit_early_frac", 0.33))
    min_sep = float(mcfg.get("min_band_sep_um", 60.0))
    # Own key: max_abs_tilt_deg is the legacy path's force-vertical fuse (45 deg)
    # and means something different there. Do not reuse it.
    ang_max = float(mcfg.get("tilt_sweep_max_deg", 25.0))
    ang_step = float(mcfg.get("tilt_sweep_step_deg", 0.5))
    recentre_lim = float(mcfg.get("max_recenter_um", 60.0))
    gap_frac = float(mcfg.get("gap_frac", 0.30))
    allow_tilt = _boolish(mcfg.get("allow_tilt", True))

    T = int(spots["frame"].max()) + 1
    early = spots[spots["frame"] <= int(np.floor(T * early_frac))]
    nuc = early[(early["area_um2"] >= amin) & (early["area_um2"] <= amax)]
    if len(nuc) < int(mcfg.get("min_nuclei", 50)):
        return None

    x = nuc["x_um"].to_numpy(dtype=float)
    y = nuc["y_um"].to_numpy(dtype=float)
    ymid = float(0.5 * (np.percentile(y, 5) + np.percentile(y, 95)))
    yc = y - ymid

    anchor = _two_band_midpoint(x, min_sep)
    if anchor is None:
        return None

    def _place(ang):
        t = np.radians(ang)
        u = x * np.cos(t) - yc * np.sin(t)
        grid, dens = _density_1d(u)
        if grid is None:
            return None
        xc, anch, moved = anchor["x"], anchor, 0.0
        if abs(ang) > 1e-6:
            rot = _two_band_midpoint(u, min_sep)
            if rot is not None and abs(rot["x"] - anchor["x"]) <= recentre_lim:
                xc, anch, moved = rot["x"], rot, rot["x"] - anchor["x"]
        thr = gap_frac * max(anch["hL"], anch["hR"])
        i = int(np.clip(np.searchsorted(grid, xc), 1, len(grid) - 2))
        if dens[i] >= thr:
            gap = 0.0
        else:
            lo = hi = i
            while lo > 0 and dens[lo - 1] < thr:
                lo -= 1
            while hi < len(dens) - 1 and dens[hi + 1] < thr:
                hi += 1
            gap = float(grid[hi] - grid[lo])
        return {"ang": float(ang), "x_center": float(xc), "gap": gap,
                "recentre_um": float(moved), "sharp": _gap_sharpness(u, xc)}

    pick = _place(0.0)
    if pick is None:
        return None
    if allow_tilt:
        # Angle objective: how well separated the two bands are, scored with
        # the same criterion that chose them. Scoring the density AT the midline
        # instead is degenerate on the movies where converging cells sit there,
        # since lowering it favours smearing the projection.
        def _sep(a):
            t = np.radians(a)
            r = _two_band_midpoint(x * np.cos(t) - yc * np.sin(t), min_sep)
            return r["score"] if r is not None else np.nan

        angs = np.arange(-ang_max, ang_max + 1e-9, ang_step)
        sep = np.array([_sep(a) for a in angs], dtype=float)
        if np.isfinite(sep).any():
            free = _place(float(angs[int(np.nanargmax(sep))]))
            if free is not None:
                pick = free

    t = np.radians(pick["ang"])
    # A gap width of 0 means the midline is populated rather than empty, which
    # at 13-17 hpf is expected: the LPM is converging. It is reported as
    # cells_on_midline and never used to reject a fit.
    return {
        "p0": np.array([pick["x_center"] / np.cos(t), ymid], dtype=float),
        "v": np.array([np.sin(t), np.cos(t)], dtype=float),
        "x_center": float(pick["x_center"]),
        "fit_mode": "bilateral_band_midpoint_sweep",
        "tilt_deg": float(pick["ang"]),
        "gap_width_um": float(pick["gap"]),
        "recentre_um": float(pick["recentre_um"]),
        "n_nuclei_used": int(len(nuc)),
        "band_left_um": float(anchor["left"]),
        "band_right_um": float(anchor["right"]),
        "cells_on_midline": bool(pick["gap"] <= 0),
    }


def apply_midline_tls_filter_spots(
    spots: pd.DataFrame,
    cfg: dict,
    out_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit midline line using early frames (after ROI), then remove ALL spots in band
    across the ENTIRE time series (size-independent).

    This matches your design decision: remove all midline-associated objects (notochord)
    for BOTH WT and MUT, no time restriction.
    """
    mcfg = cfg.get("midline_tls", {}) or {}
    if not _boolish(mcfg.get("enable", False)):
        return spots.copy(), {"enabled": False, "reason": "disabled"}

    if spots.empty:
        return spots.copy(), {"enabled": True, "reason": "empty"}

    band_half = float(mcfg.get("band_half_width_um", 20.0))
    fit_early_frac = float(mcfg.get("fit_early_frac", 0.33))
    min_points = int(mcfg.get("min_points", 60))

    # preband
    preband_half = float(mcfg.get("preband_half_width_um", max(2.0 * band_half, 40.0)))
    x_mode_bins = int(mcfg.get("x_center_mode_bins", 64))
    y_mid_q0 = float(mcfg.get("y_mid_q0", 0.20))
    y_mid_q1 = float(mcfg.get("y_mid_q1", 0.80))

    # robust fit bins
    ybin_fit_bins = int(mcfg.get("fit_y_bins", 25))
    ybin_min_bin_points = int(mcfg.get("fit_min_bin_points", 15))

    # tilt constraint (optional safety)
    max_abs_tilt_deg = float(mcfg.get("max_abs_tilt_deg", 45.0))

    method = str(mcfg.get("method", "bilateral")).strip().lower()
    extra: Dict[str, Any] = {}

    if method == "bilateral":
        bl = estimate_midline_bilateral(spots, cfg.get("tracking", {}) or {}, mcfg)
        if bl is None:
            return spots.copy(), {"enabled": True, "reason": "bilateral_fit_failed"}
        p0, v, x_center, fit_mode = bl["p0"], bl["v"], bl["x_center"], bl["fit_mode"]
        extra = {k: bl[k] for k in ("tilt_deg", "gap_width_um", "recentre_um",
                                    "n_nuclei_used", "band_left_um",
                                    "band_right_um", "cells_on_midline")}
        # The bilateral sweep is already bounded by tilt_sweep_max_deg, so the
        # force-vertical fuse below can never fire for it. Left in place for
        # the legacy path, which needs it.
    else:
        fit = estimate_midline_tls_from_preband(
            spots,
            fit_early_frac=fit_early_frac,
            min_points=min_points,
            preband_half_width_um=preband_half,
            x_center_mode_bins=x_mode_bins,
            y_mid_q0=y_mid_q0,
            y_mid_q1=y_mid_q1,
            ybin_fit_bins=ybin_fit_bins,
            ybin_min_bin_points=ybin_min_bin_points,
        )

        if fit is None:
            info = {"enabled": True, "reason": "preband_fit_failed"}
            return spots.copy(), info

        p0, v, x_center, fit_mode = fit

    # constrain extreme tilt (prevents weird diagonal if something goes wrong)
    ang = angle_deg_from_v(v)
    tilt_from_vertical = min(abs(ang - 90.0), abs(ang + 90.0), abs(ang - 270.0))
    forced_vertical = False
    if method != "bilateral" and tilt_from_vertical > max_abs_tilt_deg:
        p0 = np.array([x_center, float(np.nanmedian(spots["y_um"].to_numpy(dtype=float)))], dtype=float)
        v = np.array([0.0, 1.0], dtype=float)
        ang = angle_deg_from_v(v)
        tilt_from_vertical = 0.0
        forced_vertical = True
        fit_mode = fit_mode + "_forced_vertical"

    # Apply band to ALL frames
    dist = point_line_distance_um(
        spots["x_um"].to_numpy(dtype=float),
        spots["y_um"].to_numpy(dtype=float),
        p0, v
    )
    remove = dist <= band_half
    out = spots.loc[~remove].copy()

    info = {
        "enabled": True,
        "mode": ("bilateral_all_frames" if method == "bilateral"
                 else "preband_tls_all_frames"),
        "method": method,
        "fit_mode": str(fit_mode),
        "band_half_width_um": float(band_half),
        "preband_half_width_um": float(preband_half),
        "x_center_um": float(x_center),
        "p0_x_um": float(p0[0]),
        "p0_y_um": float(p0[1]),
        "v_x": float(v[0]),
        "v_y": float(v[1]),
        "angle_deg": float(ang),
        "tilt_from_vertical_deg": float(tilt_from_vertical),
        "forced_vertical": bool(forced_vertical),
        "n_removed": int(np.sum(remove)),
        "n_remaining": int(len(out)),
    }
    info.update(extra)

    pd.DataFrame([info]).to_csv(out_dir / "midline_tls_spots_filter_info.csv", index=False)
    return out, info


def apply_midline_tls_filter_tracks(
    linked: pd.DataFrame,
    cfg: dict,
    out_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Track-level midline filter (STRICT, all frames):
    - Uses the fitted midline line parameters (p0, v) saved by spots-level step.
    - Drops an entire track if ANY point of that track is inside the midline band.
    """
    mcfg = cfg.get("midline_tls", {}) or {}
    if not _boolish(mcfg.get("enable", False)):
        return linked.copy(), {"enabled": False, "reason": "disabled"}

    if linked is None or linked.empty:
        return linked.copy(), {"enabled": True, "reason": "empty"}

    band_half = float(mcfg.get("band_half_width_um", 20.0))

    info_path = out_dir / "midline_tls_spots_filter_info.csv"
    if not info_path.exists():
        return linked.copy(), {"enabled": True, "reason": "no_midline_fit_info"}

    fit_info = pd.read_csv(info_path)
    if fit_info.empty:
        return linked.copy(), {"enabled": True, "reason": "empty_midline_fit_info"}

    p0 = np.array([float(fit_info.loc[0, "p0_x_um"]), float(fit_info.loc[0, "p0_y_um"])], dtype=float)
    v = np.array([float(fit_info.loc[0, "v_x"]), float(fit_info.loc[0, "v_y"])], dtype=float)
    vn = np.linalg.norm(v)
    if vn < 1e-12:
        return linked.copy(), {"enabled": True, "reason": "bad_v"}
    v = v / vn

    lk = linked.copy()
    dist = point_line_distance_um(
        lk["x_um"].to_numpy(dtype=float),
        lk["y_um"].to_numpy(dtype=float),
        p0, v
    )
    lk["_in_band"] = dist <= band_half

    any_in = lk.groupby("track_id")["_in_band"].any()
    to_drop = any_in.index[any_in.values].tolist()

    out = lk.loc[~lk["track_id"].isin(to_drop)].drop(columns=["_in_band"]).copy()

    info = {
        "enabled": True,
        "mode": "any_in_band_drop_track",
        "band_half_width_um": float(band_half),
        "p0_x_um": float(p0[0]),
        "p0_y_um": float(p0[1]),
        "v_x": float(v[0]),
        "v_y": float(v[1]),
        "n_tracks_before": int(linked["track_id"].nunique()),
        "n_tracks_dropped": int(len(to_drop)),
        "n_tracks_after": int(out["track_id"].nunique()) if not out.empty else 0,
    }

    pd.DataFrame([info]).to_csv(out_dir / "midline_tls_track_filter_info.csv", index=False)
    pd.DataFrame({"dropped_track_id": to_drop}).to_csv(out_dir / "midline_tls_dropped_tracks.csv", index=False)
    return out, info


# -----------------------------
# Density control
# -----------------------------
def limit_topN_by_area_per_frame(spots: pd.DataFrame, max_spots_per_frame: int) -> pd.DataFrame:
    if spots.empty or max_spots_per_frame is None or max_spots_per_frame <= 0:
        return spots
    s = spots.sort_values(["frame", "area_um2"], ascending=[True, False])
    s = s.groupby("frame", as_index=False, sort=False).head(max_spots_per_frame)
    return s.reset_index(drop=True)


def thin_by_min_separation(spots: pd.DataFrame, min_sep_um: float) -> pd.DataFrame:
    """
    Per-frame non-max suppression based on area: keep large objects first.
    """
    if spots.empty or min_sep_um is None or min_sep_um <= 0:
        return spots

    kept_rows = []
    r2 = float(min_sep_um) ** 2
    for _, df in spots.groupby("frame", sort=False):
        df = df.sort_values("area_um2", ascending=False).reset_index(drop=True)
        kept_xy = []
        for i in range(len(df)):
            x = float(df.loc[i, "x_um"])
            y = float(df.loc[i, "y_um"])
            ok = True
            for (kx, ky) in kept_xy:
                dx = x - kx
                dy = y - ky
                if dx * dx + dy * dy < r2:
                    ok = False
                    break
            if ok:
                kept_xy.append((x, y))
                kept_rows.append(df.loc[i])

    if not kept_rows:
        return spots.iloc[0:0].copy()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


# -----------------------------
# Burst detection + preburst traces
# -----------------------------
def detect_burst_events(linked: pd.DataFrame, spots_all: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Burst event requires BOTH:
      1) big nucleus area drop (area_now < area_prev * area_drop_ratio) [within-track]
         OR track ends at break_frame (track-break mode uses last point as "prev")
      2) nearby fragments appear within lookahead frames:
         fragment area <= area_prev * frag_area_ratio_max
         and within search_radius_um
         and count >= min_frags
    """
    bcfg = cfg.get("burst_detection", {}) or {}
    if not _boolish(bcfg.get("enable", False)):
        return pd.DataFrame()

    detect_track_break = _boolish(bcfg.get("detect_track_break", True))
    detect_within_track = _boolish(bcfg.get("detect_within_track", True))

    lookahead = int(bcfg.get("lookahead_frames", 2))
    radius = float(bcfg.get("search_radius_um", 8.0))
    big_area = float(bcfg.get("big_area_um2", 35.0))
    drop_ratio = float(bcfg.get("area_drop_ratio", 0.50))
    frag_ratio_max = float(bcfg.get("frag_area_ratio_max", 0.35))
    min_frags = int(bcfg.get("min_frags", 2))
    max_frags = int(bcfg.get("max_frags", 999999))

    if linked is None or linked.empty or spots_all is None or spots_all.empty:
        return pd.DataFrame()

    if not _ensure_cols(linked, ["track_id", "frame", "x_um", "y_um", "area_um2"]):
        return pd.DataFrame()

    lk = linked.sort_values(["track_id", "frame"]).copy()
    spots_by_frame = {int(f): df for f, df in spots_all.groupby("frame", sort=False)}

    r2 = radius * radius
    events = []
    event_key_set = set()

    def find_fragments_near(x0, y0, area_prev, break_frame):
        cand_frags = []
        a_thr = area_prev * frag_ratio_max
        for dt in range(1, lookahead + 1):
            ff = break_frame + dt
            if ff not in spots_by_frame:
                continue
            sf = spots_by_frame[ff]
            sf2 = sf[sf["area_um2"] <= a_thr]
            if sf2.empty:
                continue
            dx = (sf2["x_um"].to_numpy() - x0)
            dy = (sf2["y_um"].to_numpy() - y0)
            m = (dx * dx + dy * dy) <= r2
            if np.any(m):
                hit = sf2.loc[m].copy()
                hit["_ff"] = ff
                cand_frags.append(hit)
        if not cand_frags:
            return None
        cand = pd.concat(cand_frags, ignore_index=True)
        nfrag = int(len(cand))
        if nfrag < min_frags or nfrag > max_frags:
            return None
        return cand

    if detect_within_track:
        for tid, df in lk.groupby("track_id", sort=False):
            df = df.sort_values("frame").reset_index(drop=True)
            if len(df) < 2:
                continue
            for i in range(1, len(df)):
                f_prev = int(df.loc[i - 1, "frame"])
                f_now = int(df.loc[i, "frame"])
                if f_now != f_prev + 1:
                    continue
                area_prev = float(df.loc[i - 1, "area_um2"])
                area_now = float(df.loc[i, "area_um2"])
                if area_prev < big_area:
                    continue
                if area_now >= area_prev * drop_ratio:
                    continue

                x0 = float(df.loc[i - 1, "x_um"])
                y0 = float(df.loc[i - 1, "y_um"])

                cand = find_fragments_near(x0, y0, area_prev, f_prev)
                if cand is None:
                    continue

                key = (int(tid), int(f_prev), "within")
                if key in event_key_set:
                    continue
                event_key_set.add(key)

                events.append({
                    "mode": "within_track",
                    "track_id": int(tid),
                    "break_frame": int(f_prev),
                    "x_um": x0,
                    "y_um": y0,
                    "area_prev_um2": area_prev,
                    "area_next_um2": area_now,
                    "area_ratio_next_over_prev": area_now / (area_prev + 1e-9),
                    "n_frags": int(len(cand)),
                    "sum_frag_area_um2": float(cand["area_um2"].sum()),
                    "median_frag_area_um2": float(cand["area_um2"].median()),
                    "frag_frames_min": int(cand["_ff"].min()),
                    "frag_frames_max": int(cand["_ff"].max()),
                })

    if detect_track_break:
        last = lk.groupby("track_id", as_index=False).tail(1).copy()
        for _, r in last.iterrows():
            tid = int(r["track_id"])
            f_prev = int(r["frame"])
            area_prev = float(r["area_um2"])
            if area_prev < big_area:
                continue
            x0 = float(r["x_um"])
            y0 = float(r["y_um"])

            cand = find_fragments_near(x0, y0, area_prev, f_prev)
            if cand is None:
                continue

            key = (int(tid), int(f_prev), "break")
            if key in event_key_set:
                continue
            event_key_set.add(key)

            events.append({
                "mode": "track_break",
                "track_id": int(tid),
                "break_frame": int(f_prev),
                "x_um": x0,
                "y_um": y0,
                "area_prev_um2": area_prev,
                "area_next_um2": np.nan,
                "area_ratio_next_over_prev": np.nan,
                "n_frags": int(len(cand)),
                "sum_frag_area_um2": float(cand["area_um2"].sum()),
                "median_frag_area_um2": float(cand["area_um2"].median()),
                "frag_frames_min": int(cand["_ff"].min()),
                "frag_frames_max": int(cand["_ff"].max()),
            })

    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values(["break_frame", "track_id", "mode"]).reset_index(drop=True)


def extract_preburst_area_traces(linked: pd.DataFrame, burst_df: pd.DataFrame, n_pre_frames: int = 3) -> pd.DataFrame:
    """
    For each burst event, extract area trajectory in frames [-n_pre_frames .. -1] relative to burst frame.
    """
    if linked is None or linked.empty or burst_df is None or burst_df.empty:
        return pd.DataFrame()

    lk = linked.copy()
    lk["frame"] = lk["frame"].astype(int)

    rows = []
    for _, ev in burst_df.iterrows():
        tid = int(ev["track_id"])
        f0 = int(ev["break_frame"])

        tr = lk[lk["track_id"] == tid]
        for dt in range(-n_pre_frames, 0):
            f = f0 + dt
            r = tr[tr["frame"] == f]
            area = np.nan if r.empty else float(r.iloc[0]["area_um2"])
            rows.append({"track_id": tid, "burst_frame": f0, "dt": dt, "area_um2": area})

    return pd.DataFrame(rows)


def write_burst_outputs(burst_df: pd.DataFrame, n_frames: int, out_dir: Path):
    if burst_df is None or burst_df.empty:
        pd.DataFrame().to_csv(out_dir / "burst_events.csv", index=False)
        pd.DataFrame({"frame": np.arange(n_frames), "n_bursts": 0}).to_csv(out_dir / "burst_events_per_frame.csv", index=False)
        plt.figure()
        plt.plot(np.arange(n_frames), np.zeros(n_frames), marker="o")
        plt.xlabel("Frame"); plt.ylabel("Burst events"); plt.title("Burst events per frame")
        save_plot(out_dir / "burst_events_vs_time.png")
        return

    burst_df.to_csv(out_dir / "burst_events.csv", index=False)
    per = burst_df.groupby("break_frame").size().rename("n_bursts").reset_index().rename(columns={"break_frame": "frame"})
    full = pd.DataFrame({"frame": np.arange(n_frames)}).merge(per, on="frame", how="left").fillna(0)
    full["n_bursts"] = full["n_bursts"].astype(int)
    full.to_csv(out_dir / "burst_events_per_frame.csv", index=False)

    plt.figure()
    plt.plot(full["frame"], full["n_bursts"], marker="o")
    plt.xlabel("Frame"); plt.ylabel("Burst events"); plt.title("Burst events per frame")
    save_plot(out_dir / "burst_events_vs_time.png")


# -----------------------------
# LPM width
# -----------------------------
def compute_width_per_frame(spots_for_width: pd.DataFrame, T: int, q_low: float, q_high: float) -> pd.DataFrame:
    """
    Width proxy per frame: x_quantile(q_high) - x_quantile(q_low)
    Input is already ROI + midline-spots-filtered objects.
    """
    out = []
    for f in range(T):
        df = spots_for_width[spots_for_width["frame"] == f]
        if df.empty:
            out.append({"frame": f, "width_um": np.nan, "x_low_um": np.nan, "x_high_um": np.nan, "n": 0})
            continue
        xs = df["x_um"].to_numpy(dtype=float)
        xlow = float(np.quantile(xs, q_low))
        xhigh = float(np.quantile(xs, q_high))
        out.append({"frame": f, "width_um": xhigh - xlow, "x_low_um": xlow, "x_high_um": xhigh, "n": int(len(xs))})
    return pd.DataFrame(out)


# -----------------------------
# QC hyperstack (ImageJ TCYX)
# -----------------------------
def build_qc_hyperstack(labels_3d, tracks, roi, px_um, out_path: Path):
    """
    ImageJ hyperstack: axes="TCYX", uint8
      C0: labels grayscale
      C1: tracks overlay (dots + accumulated trails)
      C2: ROI box
      C3: midline overlay (center + band edges)
    """
    T, H, W = labels_3d.shape
    stack = np.zeros((T, 4, H, W), dtype=np.uint8)

    # C0 labels
    for t in range(T):
        lab = labels_3d[t].astype(np.float32)
        if lab.max() > 0:
            g = (lab / lab.max()) * 255.0
        else:
            g = lab
        stack[t, 0] = g.clip(0, 255).astype(np.uint8)

    # C2 ROI box (FIX: include right/top edges by using +1 slices)
    if roi is not None:
        x0, x1, y0, y1 = roi
        x0p = int(round(x0 / px_um)); x1p = int(round(x1 / px_um))
        y0p = int(round(y0 / px_um)); y1p = int(round(y1 / px_um))
        x0p = max(0, min(W - 1, x0p)); x1p = max(0, min(W - 1, x1p))
        y0p = max(0, min(H - 1, y0p)); y1p = max(0, min(H - 1, y1p))

        xs = slice(min(x0p, x1p), min(max(x0p, x1p) + 1, W))
        ys = slice(min(y0p, y1p), min(max(y0p, y1p) + 1, H))
        xL = min(x0p, x1p)
        xR = max(x0p, x1p)
        yT = min(y0p, y1p)
        yB = max(y0p, y1p)

        for t in range(T):
            stack[t, 2, yT, xs] = 255
            stack[t, 2, yB, xs] = 255
            stack[t, 2, ys, xL] = 255
            stack[t, 2, ys, xR] = 255

    # C3 midline overlay from midline_tls_spots_filter_info.csv (center + band edges)
    info_path = out_path.parent / "midline_tls_spots_filter_info.csv"
    if info_path.exists():
        try:
            info = pd.read_csv(info_path)
            if not info.empty and {"p0_x_um", "p0_y_um", "v_x", "v_y"}.issubset(set(info.columns)):
                p0 = np.array([float(info.loc[0, "p0_x_um"]), float(info.loc[0, "p0_y_um"])], dtype=float)
                v = np.array([float(info.loc[0, "v_x"]), float(info.loc[0, "v_y"])], dtype=float)
                vn = float(np.linalg.norm(v))
                if vn > 1e-12:
                    v = v / vn
                    n = np.array([-v[1], v[0]], dtype=float)
                    band_half = float(info.loc[0, "band_half_width_um"]) if "band_half_width_um" in info.columns else 20.0

                    def _draw_param_line(p_um: np.ndarray, v_unit: np.ndarray, val: int = 255):
                        tmax = max(H, W) * px_um * 2.0
                        ts = np.linspace(-tmax, tmax, 5000)
                        xs = (p_um[0] + ts * v_unit[0]) / px_um
                        ys = (p_um[1] + ts * v_unit[1]) / px_um
                        xi = np.rint(xs).astype(int)
                        yi = np.rint(ys).astype(int)
                        m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                        xi = xi[m]; yi = yi[m]
                        if len(xi) == 0:
                            return
                        stack[:, 3, yi, xi] = val  # draw in all frames

                    # center line + band edges
                    _draw_param_line(p0, v, 255)
                    _draw_param_line(p0 + band_half * n, v, 255)
                    _draw_param_line(p0 - band_half * n, v, 255)
        except Exception:
            pass

    dbg_path = out_path.parent / "qc_tracks_debug.txt"
    dbg_path.write_text("", encoding="utf-8")

    def _dbg(s):
        with open(dbg_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    _dbg(f"labels_3d shape={labels_3d.shape}, px_um={px_um}")
    _dbg(f"tracks empty? {tracks is None or tracks.empty}")

    if tracks is None or tracks.empty:
        tiff.imwrite(str(out_path), stack, metadata={"axes": "TCYX"})
        _dbg("No tracks -> wrote only labels/roi/midline.")
        return

    need = {"track_id", "frame", "x_um", "y_um"}
    missing = need - set(tracks.columns)
    if missing:
        _dbg(f"ERROR missing columns: {sorted(list(missing))}")
        tiff.imwrite(str(out_path), stack, metadata={"axes": "TCYX"})
        return

    tr = tracks[["track_id", "frame", "x_um", "y_um"]].copy().dropna()
    tr["frame"] = tr["frame"].astype(int)
    tr["x_px"] = tr["x_um"] / px_um
    tr["y_px"] = tr["y_um"] / px_um
    tr = tr.sort_values(["track_id", "frame"]).reset_index(drop=True)

    _dbg(f"n_rows={len(tr)}, n_tracks={tr['track_id'].nunique()}, frame_min={tr['frame'].min()}, frame_max={tr['frame'].max()}")

    def draw_dot(ti, x, y, val=255):
        xi = int(round(x)); yi = int(round(y))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            return 0
        wrote = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                xx = xi + dx; yy = yi + dy
                if 0 <= xx < W and 0 <= yy < H:
                    stack[ti, 1, yy, xx] = val
                    wrote += 1
        return wrote

    def draw_line(ti, x0, y0, x1, y1, val=255):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        if n <= 1:
            return draw_dot(ti, x0, y0, val)
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        wrote = 0
        for x, y in zip(xs, ys):
            wrote += draw_dot(ti, x, y, val)
        return wrote

    n_seg, n_dot, n_pix = 0, 0, 0
    for tid, df in tr.groupby("track_id", sort=False):
        xs = df["x_px"].to_numpy()
        ys = df["y_px"].to_numpy()
        fs = df["frame"].to_numpy().astype(int)

        for i in range(1, len(df)):
            f1 = fs[i]
            for ti in range(min(f1, T - 1), T):
                n_pix += draw_line(ti, xs[i - 1], ys[i - 1], xs[i], ys[i], 255)
            n_seg += 1

        for i in range(len(df)):
            f0 = fs[i]
            for ti in range(min(f0, T - 1), T):
                n_pix += draw_dot(ti, xs[i], ys[i], 255)
            n_dot += 1

    _dbg(f"drawn segments={n_seg}, dots={n_dot}, pixels_written={n_pix}")
    _dbg(f"C1 max={int(stack[:,1].max())}, nonzero={int(np.count_nonzero(stack[:,1]))}")

    tiff.imwrite(str(out_path), stack, metadata={"axes": "TCYX"})
    _dbg(f"Wrote: {out_path.name}")


# -----------------------------
# Main analysis per labels.tif
# -----------------------------
def analyze_one_labels_tif(labels_path: Path, out_dir: Path, cfg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_cfg = cfg.get("roi", {}) or {}
    tcfg = cfg.get("tracking", {}) or {}
    cmp_cfg = cfg.get("compare", {}) or {}

    px_um = float(tcfg["px_um"])
    dt_min = float(tcfg["dt_min"])

    min_area_um2 = float(tcfg["min_area_um2"])
    max_area_um2 = float(tcfg["max_area_um2"])
    fragment_area_um2 = float(tcfg["fragment_area_um2"])

    search_range_um = float(tcfg.get("search_range_um", 8.5))
    memory = int(tcfg.get("memory", 2))
    min_track_len = int(tcfg.get("min_track_len", 3))
    max_spots_per_frame = int(tcfg.get("max_spots_per_frame", 0))
    min_separation_um = float(tcfg.get("min_separation_um", 0.0))

    adaptive_stop_um = tcfg.get("adaptive_stop_um", None)
    adaptive_step = tcfg.get("adaptive_step", None)
    adaptive_stop_um = float(adaptive_stop_um) if adaptive_stop_um is not None else None
    adaptive_step = float(adaptive_step) if adaptive_step is not None else None

    width_q_low = float(cmp_cfg.get("width_q_low", 0.10))
    width_q_high = float(cmp_cfg.get("width_q_high", 0.90))

    labels_3d = tiff.imread(str(labels_path))
    if labels_3d.ndim != 3:
        raise ValueError(f"Expected (T,Y,X) labels, got shape {labels_3d.shape} in {labels_path.name}")
    T = int(labels_3d.shape[0])

    # ---- Extract all objects
    spots_raw_all = extract_spots_from_labels(labels_3d, px_um)
    spots_raw_all.to_csv(out_dir / "spots_raw_all_objects.csv", index=False)

    # ---- ROI
    roi = get_roi(spots_raw_all, roi_cfg, px_um)
    if roi is not None:
        pd.DataFrame([{"x0_um": roi[0], "x1_um": roi[1], "y0_um": roi[2], "y1_um": roi[3]}]).to_csv(
            out_dir / "safety_roi_um.csv", index=False
        )
    spots_roi = apply_roi_filter(spots_raw_all, roi)
    spots_roi.to_csv(out_dir / "spots_all_objects_roi.csv", index=False)

    # ---- Midline TLS spots-level filter (STRICT, size-independent)
    spots_all, mid_spot_info = apply_midline_tls_filter_spots(spots_roi, cfg, out_dir)
    spots_all.to_csv(out_dir / "spots_all_objects_roi_midline_spotsfiltered.csv", index=False)

    # ---- Fragments count (from spots_all AFTER ROI+midline spots-level filter)
    frags = spots_all[spots_all["area_um2"] < fragment_area_um2].copy()
    frags_pf = frags.groupby("frame").size().rename("n_fragments").reset_index()
    frags_pf.to_csv(out_dir / "fragments_per_frame.csv", index=False)

    # ---- Width per frame (use nuclear-sized objects only; avoids huge artifacts dominating)
    width_spots = spots_all[(spots_all["area_um2"] >= min_area_um2) & (spots_all["area_um2"] <= max_area_um2)].copy()
    width_pf = compute_width_per_frame(width_spots, T, width_q_low, width_q_high)
    width_pf.to_csv(out_dir / "width_per_frame.csv", index=False)
    plt.figure()
    plt.plot(width_pf["frame"], width_pf["width_um"], marker="o")
    plt.xlabel("Frame"); plt.ylabel("Width (µm)")
    plt.title(f"LPM width proxy vs time (x q{width_q_low:.2f}..q{width_q_high:.2f}, nuclear-size filtered)")
    save_plot(out_dir / "width_vs_time.png")

    # ---- Spots for tracking (from spots_all)
    spots_track = spots_all[
        (spots_all["area_um2"] >= min_area_um2) &
        (spots_all["area_um2"] <= max_area_um2)
    ].copy()
    spots_track.to_csv(out_dir / "spots_for_tracking_raw.csv", index=False)

    if spots_track.empty:
        (out_dir / "link_error.txt").write_text("No spots for tracking after filters.\n", encoding="utf-8")
        pd.DataFrame().to_csv(out_dir / "tracks_linked.csv", index=False)
        write_burst_outputs(pd.DataFrame(), T, out_dir)
        build_qc_hyperstack(labels_3d, pd.DataFrame(), roi, px_um, out_dir / "qc_tracks_hyperstack.tif")
        print(f"[WARN] No spots for tracking after filters: {labels_path.name}")
        return

    # ---- Density control
    spots_track = limit_topN_by_area_per_frame(spots_track, max_spots_per_frame)
    spots_track = thin_by_min_separation(spots_track, min_separation_um)
    spots_track.to_csv(out_dir / "spots_for_tracking_final.csv", index=False)

    # ---- Link
    tp_df = spots_track.rename(columns={"x_um": "x", "y_um": "y"}).sort_values(["frame"]).reset_index(drop=True)

    link_kwargs = dict(search_range=search_range_um, memory=memory)
    if adaptive_stop_um is not None:
        link_kwargs["adaptive_stop"] = adaptive_stop_um
    if adaptive_step is not None:
        link_kwargs["adaptive_step"] = adaptive_step

    try:
        linked = tp.link_df(tp_df, **link_kwargs)
    except Exception as e:
        msg = f"FIRST link_df failed with:\n{repr(e)}\n\n"
        msg += f"n_tp_df={len(tp_df)}; search_range_um={search_range_um}; memory={memory}; "
        msg += f"adaptive_stop={link_kwargs.get('adaptive_stop', None)}; adaptive_step={link_kwargs.get('adaptive_step', None)}\n"
        (out_dir / "link_error.txt").write_text(msg, encoding="utf-8")

        # second attempt: stronger thinning
        try:
            s2 = limit_topN_by_area_per_frame(spots_track, max(80, min(160, max_spots_per_frame or 160)))
            s2 = thin_by_min_separation(s2, max(3.5, min_separation_um))
            tp_df2 = s2.rename(columns={"x_um": "x", "y_um": "y"}).sort_values(["frame"]).reset_index(drop=True)
            linked = tp.link_df(tp_df2, **link_kwargs)
            spots_track = s2
            tp_df = tp_df2
            (out_dir / "link_error.txt").write_text(msg + "\nSECOND attempt succeeded after stronger thinning.\n", encoding="utf-8")
        except Exception as e2:
            (out_dir / "link_error.txt").write_text(msg + f"\nSECOND link_df failed with:\n{repr(e2)}\n", encoding="utf-8")
            pd.DataFrame().to_csv(out_dir / "tracks_linked.csv", index=False)
            write_burst_outputs(pd.DataFrame(), T, out_dir)
            build_qc_hyperstack(labels_3d, pd.DataFrame(), roi, px_um, out_dir / "qc_tracks_hyperstack.tif")
            print(f"[ERROR] link_df failed. See: {out_dir / 'link_error.txt'}")
            return

    if min_track_len > 1:
        linked = tp.filter_stubs(linked, threshold=min_track_len)

    linked = linked.rename(columns={"x": "x_um", "y": "y_um", "particle": "track_id"}).reset_index(drop=True)

    # ---- Track-level midline filter (STRICT)
    linked_f, mid_track_info = apply_midline_tls_filter_tracks(linked, cfg, out_dir)

    linked_f.to_csv(out_dir / "tracks_linked.csv", index=False)

    # ---- QC stack uses filtered linked
    build_qc_hyperstack(labels_3d, linked_f, roi, px_um, out_dir / "qc_tracks_hyperstack.tif")

    # ---- Summary per frame (tracked)
    all_frames = pd.DataFrame({"frame": np.arange(T)})

    if linked_f.empty:
        summary = all_frames.copy()
        summary["n_tracked"] = 0
        summary["median_area_um2"] = np.nan
        summary["mean_area_um2"] = np.nan
    else:
        summary = linked_f.groupby("frame", as_index=False).agg(
            n_tracked=("track_id", "nunique"),
            median_area_um2=("area_um2", "median"),
            mean_area_um2=("area_um2", "mean"),
        )
        summary = all_frames.merge(summary, on="frame", how="left")

    summary = summary.merge(frags_pf, on="frame", how="left")
    summary["n_tracked"] = summary["n_tracked"].fillna(0).astype(int)
    summary["n_fragments"] = summary["n_fragments"].fillna(0).astype(int)
    summary.to_csv(out_dir / "summary_per_frame.csv", index=False)

    # ---- Speed per frame
    speed_pf = all_frames.copy()
    speed_pf["median_speed_um_per_min"] = np.nan
    speed_pf["mean_speed_um_per_min"] = np.nan

    if not linked_f.empty:
        lk2 = linked_f.sort_values(["track_id", "frame"]).copy()
        lk2["dx_um"] = lk2.groupby("track_id")["x_um"].diff()
        lk2["dy_um"] = lk2.groupby("track_id")["y_um"].diff()
        lk2["step_um"] = np.sqrt(lk2["dx_um"] ** 2 + lk2["dy_um"] ** 2)
        lk2["speed_um_per_min"] = lk2["step_um"] / float(dt_min)

        tmp = lk2.groupby("frame", as_index=False).agg(
            median_speed_um_per_min=("speed_um_per_min", "median"),
            mean_speed_um_per_min=("speed_um_per_min", "mean"),
        )
        speed_pf = speed_pf.merge(tmp, on="frame", how="left", suffixes=("", "_tmp"))
        speed_pf["median_speed_um_per_min"] = speed_pf["median_speed_um_per_min_tmp"]
        speed_pf["mean_speed_um_per_min"] = speed_pf["mean_speed_um_per_min_tmp"]
        speed_pf = speed_pf.drop(columns=["median_speed_um_per_min_tmp", "mean_speed_um_per_min_tmp"])

        if _boolish((cfg.get("compare", {}) or {}).get("speed_skip_first_frame", True)):
            speed_pf.loc[speed_pf["frame"] == 0, ["median_speed_um_per_min", "mean_speed_um_per_min"]] = np.nan

    speed_pf.to_csv(out_dir / "speed_per_frame.csv", index=False)

    # ---- Burst detection uses filtered tracks + spots_all fragments
    burst_df = detect_burst_events(linked_f, spots_all, cfg)
    write_burst_outputs(burst_df, T, out_dir)

    preburst_df = extract_preburst_area_traces(linked_f, burst_df, n_pre_frames=3)
    preburst_df.to_csv(out_dir / "preburst_area_traces.csv", index=False)

    # ---- Burst/track ratio (per movie)
    n_tracks = int(linked_f["track_id"].nunique()) if not linked_f.empty else 0
    n_bursts = int(len(burst_df)) if burst_df is not None and not burst_df.empty else 0
    burst_per_track_ratio = (n_bursts / n_tracks) if n_tracks > 0 else np.nan

    # ---- PLOTS (per movie)
    plt.figure()
    plt.plot(summary["frame"], summary["n_tracked"], marker="o")
    plt.xlabel("Frame"); plt.ylabel("Tracked nuclei count")
    plt.title("Tracked nuclei per frame")
    save_plot(out_dir / "tracked_nuclei_count_vs_time.png")

    plt.figure()
    plt.plot(summary["frame"], summary["median_area_um2"], marker="o")
    plt.xlabel("Frame"); plt.ylabel("Median area (µm²)")
    plt.title("Median nucleus area per frame (tracked)")
    save_plot(out_dir / "median_area_vs_time.png")

    plt.figure()
    plt.plot(summary["frame"], summary["n_fragments"], marker="o")
    plt.xlabel("Frame"); plt.ylabel(f"N objects with area < {fragment_area_um2} µm²")
    plt.title("Small fragments per frame (ROI + midline spots-filtered all objects)")
    save_plot(out_dir / "fragments_vs_time.png")

    plt.figure()
    plt.plot(speed_pf["frame"], speed_pf["median_speed_um_per_min"], marker="o")
    plt.xlabel("Frame"); plt.ylabel("Median speed (µm/min)")
    plt.title("Median migration speed per frame (tracked; frame0 skipped)")
    save_plot(out_dir / "median_speed_vs_time.png")

    plt.figure()
    plt.hist(spots_all["area_um2"].values, bins=80)
    plt.xlabel("Area (µm²)"); plt.ylabel("Count")
    plt.title("All frames: area distribution (ROI + midline spots-filtered objects)")
    save_plot(out_dir / "area_hist_all_objects.png")

    if not linked_f.empty:
        plt.figure()
        plt.hist(linked_f["area_um2"].values, bins=60)
        plt.xlabel("Area (µm²)"); plt.ylabel("Count")
        plt.title("Tracked spots: area distribution")
        save_plot(out_dir / "tracked_area_hist.png")

    last_f = T - 1
    last = spots_all[spots_all["frame"] == last_f]
    plt.figure()
    plt.hist(last["area_um2"].values, bins=60)
    plt.xlabel("Area (µm²)"); plt.ylabel("Count")
    plt.title(f"Last frame (frame={last_f}): area distribution (ROI + midline spots-filtered)")
    save_plot(out_dir / "area_hist_last_frame.png")

    if not linked_f.empty:
        track_len = linked_f.groupby("track_id").size().values
        plt.figure()
        plt.hist(track_len, bins=30)
        plt.xlabel("Track length (frames)"); plt.ylabel("Count")
        plt.title("Track length distribution")
        save_plot(out_dir / "track_length_hist.png")

    # ---- Movie summary (for compare)
    mean_median_area_um2 = float(np.nanmean(summary["median_area_um2"].to_numpy(dtype=float)))
    mean_median_speed_um_per_min = float(np.nanmean(speed_pf["median_speed_um_per_min"].to_numpy(dtype=float)))

    ms = {
        "movie": labels_path.stem,
        "n_frames": int(T),
        "px_um": float(px_um),
        "dt_min": float(dt_min),
        "roi_used": bool(roi is not None),

        "search_range_um": float(search_range_um),
        "memory": int(memory),
        "min_track_len": int(min_track_len),
        "max_spots_per_frame": int(max_spots_per_frame),
        "min_separation_um": float(min_separation_um),
        "adaptive_stop_um": float(adaptive_stop_um) if adaptive_stop_um is not None else np.nan,
        "adaptive_step": float(adaptive_step) if adaptive_step is not None else np.nan,

        "midline_tls_enabled": bool(_boolish((cfg.get("midline_tls", {}) or {}).get("enable", False))),
        "midline_band_half_width_um": float((cfg.get("midline_tls", {}) or {}).get("band_half_width_um", np.nan)),
        "midline_spots_removed": int(mid_spot_info.get("n_removed", 0)) if isinstance(mid_spot_info, dict) else 0,
        "midline_tracks_dropped": int(mid_track_info.get("n_tracks_dropped", 0)) if isinstance(mid_track_info, dict) else 0,

        "mean_tracked_nuclei": float(np.nanmean(summary["n_tracked"].to_numpy(dtype=float))),
        "median_tracked_nuclei": float(np.nanmedian(summary["n_tracked"].to_numpy(dtype=float))),
        "max_tracked_nuclei": int(np.nanmax(summary["n_tracked"].to_numpy(dtype=float))) if len(summary) else 0,

        "mean_fragments_per_frame": float(np.nanmean(summary["n_fragments"].to_numpy(dtype=float))),
        "max_fragments_per_frame": int(np.nanmax(summary["n_fragments"].to_numpy(dtype=float))) if len(summary) else 0,

        "mean_median_area_um2": float(mean_median_area_um2),
        "mean_median_speed_um_per_min": float(mean_median_speed_um_per_min),

        "n_tracks": int(n_tracks),
        "n_burst_events": int(n_bursts),
        "burst_per_track_ratio": float(burst_per_track_ratio) if burst_per_track_ratio == burst_per_track_ratio else np.nan,
    }
    pd.DataFrame([ms]).to_csv(out_dir / "movie_summary.csv", index=False)

    print(f"[OK] Outputs written to: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels_path", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    analyze_one_labels_tif(Path(args.labels_path), Path(args.out_dir), cfg)


if __name__ == "__main__":
    main()
