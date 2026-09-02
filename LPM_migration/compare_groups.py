import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import matplotlib as mpl


# -------- Type sizes --------
# Edit here. The panels are ~4.6 x 3.2 in, so these are the sizes as they land
# on the page before any scaling in Illustrator.
FONT = dict(
    base=12,      # fallback for anything not named below
    label=13,     # axis labels
    tick=12,      # tick labels  <- this is the one that was too small
    title=10,     # panel title; usually cropped when the figure is assembled
    legend=11,
    pval=11,      # the "p = ..." annotation on the boxplots
)

mpl.rcParams.update({
    # -------- Font (Illustrator friendly) --------
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,   # keep TrueType for Illustrator
    "ps.fonttype": 42,
    "svg.fonttype": "none",   # real <text> in the svg, editable in Illustrator

    # -------- Figure aesthetics --------
    "font.size": FONT["base"],
    "axes.labelsize": FONT["label"],
    "axes.titlesize": FONT["title"],
    "legend.fontsize": FONT["legend"],
    "xtick.labelsize": FONT["tick"],
    "ytick.labelsize": FONT["tick"],

    "axes.linewidth": 0.9,
    "lines.linewidth": 1.4,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
})

from scipy.stats import ttest_ind


# -----------------------------
# Publication style constants
# -----------------------------
LABEL_WT = "wild type"
LABEL_MUT = "aqp1a.1-/-"

# RGB given by user
COLOR_WT = (107/255, 124/255, 147/255)   # light grey-blue
COLOR_MUT = (204/255, 0/255, 102/255)    # magenta

# Transparency (apply everywhere)
ALPHA_MEAN_LINE = 0.95
ALPHA_FILL = 0.22
ALPHA_MOVIE_LINE = 0.28   # individual movie lines in preburst/time-series (if enabled)

# Taller figure default (Nature-ish)
# Time axis: fix the tick step and rotation instead of letting matplotlib
# choose. With larger tick type it silently dropped to whole-hour ticks,
# losing the half-hour grid.
XTICK_STEP_HPF = 0.5
XTICK_ROT = 45

# Enlarged from (3.2, 4.2) / (4.6, 3.2). At the bigger tick and label type the
# rotated x labels ate the axes height and the long y label ("Median tracked
# nuclear area (um2)") ran past the top of the canvas.
FIGSIZE_TALL = (3.8, 5.0)   # inches; adjust if you want slimmer/wider
FIGSIZE_WIDE = (5.4, 4.0)


# -----------------------------
# Stats helpers
# -----------------------------
def ttest_welch(a, b):
    """
    Two-tailed Welch's t-test.
    Returns p-value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = ttest_ind(a, b, equal_var=False)
    return p


def format_p_value(p: float) -> str:
    """Numeric p-value text (no stars)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p = nan"
    if p < 1e-4:
        return "p < 1e-4"
    return f"p = {p:.3f}"


def robust_ylim(vals_list, pad_frac=0.18, pad_min=0.6):
    """
    vals_list: list of 1D arrays
    Return (ymin, ymax) with padding so boxes don't stick to bottom/top.
    """
    allv = []
    for v in vals_list:
        if v is None:
            continue
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            allv.append(v)
    if not allv:
        return None

    vv = np.concatenate(allv)
    y0 = float(np.min(vv))
    y1 = float(np.max(vv))
    if np.isclose(y0, y1):
        y0 -= 1.0
        y1 += 1.0

    rng = y1 - y0
    pad = max(pad_min, rng * pad_frac)
    return (y0 - pad, y1 + pad)



# -----------------------------
# IO
# -----------------------------
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_plot(path: Path, also_png: bool = True, dpi_png: int = 300):
    """
    Save publication-quality SVG for Illustrator (+ optional PNG preview).
    The input 'path' can be any suffix; we will save as .svg and (optionally) .png.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".svg"), bbox_inches="tight")

    if also_png:
        png_path = path.with_suffix(".png")
        plt.savefig(png_path, dpi=dpi_png, bbox_inches="tight")

    plt.close()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Time axis helpers
# -----------------------------
def frame_to_hpf_end_aligned(frame: int, n_frames: int, end_hpf: float, dt_min: float) -> float:
    """
    Map a frame index to an absolute hpf axis, end-aligned at end_hpf.
    - last frame (n_frames-1) => end_hpf
    - earlier frames move backwards by dt_min
    """
    return float(end_hpf - (n_frames - 1 - int(frame)) * (dt_min / 60.0))

def frame_to_hpf_window(frame: int, n_frames: int, start_hpf: float, end_hpf: float) -> float:
    """
    Map frame index to absolute hpf using per-movie window.
    Assumes frames are uniformly spaced across [start_hpf, end_hpf] with n_frames.
    Example:
      n_frames=19, start=14, end=17 -> dt = (17-14)/(19-1) = 10 min
    """
    if n_frames <= 1:
        return float(start_hpf)
    step = (end_hpf - start_hpf) / float(n_frames - 1)
    return float(start_hpf + int(frame) * step)

def build_common_hpf_grid(start_hpf: float, end_hpf: float, dt_min: float) -> np.ndarray:
    step = dt_min / 60.0
    n = int(round((end_hpf - start_hpf) / step)) + 1
    grid = start_hpf + np.arange(n) * step
    grid = np.round(grid, 6)
    return grid


def parse_hpf_window_from_name(name: str) -> tuple[float | None, float | None]:
    """
    Parse patterns like '13-17hpf' or '14-17hpf' or '13-16hpf' from filename/movie name.
    Returns (start_hpf, end_hpf) as floats, or (None, None) if not found.
    """
    m = re.search(r'(\d+(?:\.\d+)?)[-_](\d+(?:\.\d+)?)\s*hpf', name, flags=re.IGNORECASE)
    if not m:
        return (None, None)
    a = float(m.group(1))
    b = float(m.group(2))
    # ensure start <= end
    if a > b:
        a, b = b, a
    return (a, b)


# -----------------------------
# Summaries
# -----------------------------
def collect_movies(cfg: dict) -> list[dict]:
    results_root = Path(cfg["project"]["results_root"])
    groups = cfg["dataset"]["groups"]

    rows = []
    for group, gcfg in groups.items():
        group_dir = results_root / group
        if not group_dir.exists():
            continue
        for movie_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
            track_dir = movie_dir / "trackpy"
            ms_path = track_dir / "movie_summary.csv"
            ms = safe_read_csv(ms_path)
            if ms.empty:
                continue
            d = ms.iloc[0].to_dict()
            d["group"] = group
            d["movie"] = movie_dir.name
            d["movie_dir"] = str(movie_dir)
            d["track_dir"] = str(track_dir)
            rows.append(d)
    return rows


def pick_group_order(cfg: dict, df: pd.DataFrame) -> list[str]:
    """
    Enforce WT left, MUT right whenever both exist.
    Falls back to config order, but still fixes WT/MUT ordering if present.
    """
    present = set(df["group"].astype(str).unique().tolist())

    # strong default: always WT then MUT if both present
    if "WT" in present and "MUT" in present:
        return ["WT", "MUT"]

    cmp_cfg = (cfg.get("compare", {}) or {})
    order = cmp_cfg.get("group_order", None)
    if isinstance(order, list) and len(order) >= 2:
        cleaned = [str(x) for x in order if str(x) in present]
        # If WT/MUT present but not both, keep cleaned
        if cleaned:
            return cleaned

    return sorted(list(present))


# -----------------------------
# Per-movie time series -> aligned common grid
# -----------------------------
def load_per_movie_time_series(track_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Expect these files from track_from_labels.py:
      - summary_per_frame.csv (frame, n_tracked, median_area_um2, mean_area_um2, n_fragments,...)
      - speed_per_frame.csv (frame, median_speed_um_per_min, mean_speed_um_per_min)
      - burst_events_per_frame.csv (frame, n_bursts)
      - width_per_frame.csv (frame, width_um)
    """
    out = {}
    out["summary"] = safe_read_csv(track_dir / "summary_per_frame.csv")
    out["speed"]   = safe_read_csv(track_dir / "speed_per_frame.csv")
    out["burst"]   = safe_read_csv(track_dir / "burst_events_per_frame.csv")
    out["width"]   = safe_read_csv(track_dir / "width_per_frame.csv")
    return out


def load_preburst_traces(track_dir: Path) -> pd.DataFrame:
    """
    Expect from track_from_labels.py:
      preburst_area_traces.csv with columns:
        track_id, burst_frame, dt, area_um2
    """
    p = track_dir / "preburst_area_traces.csv"
    df = safe_read_csv(p)
    if df.empty:
        return df
    need = {"dt", "area_um2"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["dt"] = df["dt"].astype(int)
    df["area_um2"] = pd.to_numeric(df["area_um2"], errors="coerce")
    return df


def summarize_preburst_per_movie(pre_df: pd.DataFrame, n_pre_frames: int,
                                 agg: str = "mean") -> pd.DataFrame:
    """
    Convert event-level traces -> per-movie summary at each dt.
    Returns df with columns: dt, value, n_events
    dt range: [-n_pre_frames .. -1]
    agg: "mean" or "median" across events at each dt
    """
    if pre_df is None or pre_df.empty:
        dts = list(range(-n_pre_frames, 0))
        return pd.DataFrame({"dt": dts, "value": np.nan, "n_events": 0})

    dts = list(range(-n_pre_frames, 0))
    out = []
    for dt in dts:
        v = pre_df.loc[pre_df["dt"] == dt, "area_um2"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out.append({"dt": dt, "value": np.nan, "n_events": 0})
            continue
        if agg == "median":
            val = float(np.nanmedian(v))
        else:
            val = float(np.nanmean(v))
        out.append({"dt": dt, "value": val, "n_events": int(v.size)})

    return pd.DataFrame(out)


def align_one_movie(ts, common_hpf, speed_skip_first_frame,
                    movie_start_hpf, movie_end_hpf,
                    default_end_hpf, default_dt_min):
    """
    Produce aligned dataframe indexed by common_hpf with columns:
      n_tracked, median_area_um2, mean_area_um2,
      median_speed_um_per_min, mean_speed_um_per_min,
      n_bursts, width_um

    If movie_start_hpf/movie_end_hpf parsed from filename -> use per-movie window mapping.
    Else fallback to old behavior: end-align to default_end_hpf with default_dt_min.
    """
    # Determine n_frames from summary (prefer) otherwise from others
    n_frames = None
    if ts.get("summary", pd.DataFrame()).empty is False and "frame" in ts["summary"].columns:
        n_frames = int(ts["summary"]["frame"].max()) + 1
    elif ts.get("speed", pd.DataFrame()).empty is False and "frame" in ts["speed"].columns:
        n_frames = int(ts["speed"]["frame"].max()) + 1
    elif ts.get("burst", pd.DataFrame()).empty is False and "frame" in ts["burst"].columns:
        n_frames = int(ts["burst"]["frame"].max()) + 1
    elif ts.get("width", pd.DataFrame()).empty is False and "frame" in ts["width"].columns:
        n_frames = int(ts["width"]["frame"].max()) + 1
    else:
        return pd.DataFrame({"hpf": common_hpf})

    frames = np.arange(n_frames, dtype=int)

    # Build movie hpf vector
    if (movie_start_hpf is not None) and (movie_end_hpf is not None):
        hpfs = np.array([frame_to_hpf_window(f, n_frames, float(movie_start_hpf), float(movie_end_hpf))
                         for f in frames], dtype=float)
    else:
        # fallback: end-aligned to default_end_hpf
        hpfs = np.array([frame_to_hpf_end_aligned(f, n_frames, float(default_end_hpf), float(default_dt_min))
                         for f in frames], dtype=float)

    hpfs = np.round(hpfs, 6)

    base = pd.DataFrame({"frame": frames, "hpf": hpfs})
    df = base.copy()

    # merge sources
    if ts.get("summary", pd.DataFrame()).empty is False:
        df = df.merge(ts["summary"], on="frame", how="left")

    if ts.get("speed", pd.DataFrame()).empty is False:
        df = df.merge(ts["speed"], on="frame", how="left")

    if ts.get("burst", pd.DataFrame()).empty is False:
        df = df.merge(ts["burst"], on="frame", how="left")

    if ts.get("width", pd.DataFrame()).empty is False:
        df = df.merge(ts["width"], on="frame", how="left")

    # optional: remove first frame speed (avoid 0->1 jump)
    if speed_skip_first_frame:
        for c in ["median_speed_um_per_min", "mean_speed_um_per_min"]:
            if c in df.columns:
                df.loc[df["frame"] == 0, c] = np.nan

    # reduce to aligned grid
    aligned = pd.DataFrame({"hpf": common_hpf})
    aligned = aligned.merge(df.drop(columns=["frame"]), on="hpf", how="left")

    keep = ["hpf",
            "n_tracked", "median_area_um2", "mean_area_um2",
            "median_speed_um_per_min", "mean_speed_um_per_min",
            "n_bursts", "width_um"]
    for k in keep:
        if k not in aligned.columns:
            aligned[k] = np.nan

    return aligned[keep]


def mean_sd_over_movies(aligned_list: list[pd.DataFrame], value_col: str) -> pd.DataFrame:
    """
    aligned_list: each has columns [hpf, value_col]
    return: hpf, mean, sd, n
    """
    if not aligned_list:
        return pd.DataFrame(columns=["hpf", "mean", "sd", "n"])

    hpfs = aligned_list[0]["hpf"].to_numpy()
    mat = []
    for a in aligned_list:
        mat.append(a[value_col].to_numpy(dtype=float))
    M = np.vstack(mat)  # (n_movies, n_time)

    n = np.sum(~np.isnan(M), axis=0)

    # mean: only where n>=1
    mean = np.full(M.shape[1], np.nan, dtype=float)
    ok_mean = n >= 1
    mean[ok_mean] = np.nanmean(M[:, ok_mean], axis=0)

    # sd: only where n>=2; otherwise sd=0 for stable shading
    sd = np.zeros(M.shape[1], dtype=float)
    ok_sd = n >= 2
    sd[ok_sd] = np.nanstd(M[:, ok_sd], axis=0, ddof=1)

    return pd.DataFrame({"hpf": hpfs, "mean": mean, "sd": sd, "n": n})


# -----------------------------
# Plotting
# -----------------------------
def plot_mean_sd(df_wt: pd.DataFrame, df_mut: pd.DataFrame, ylab: str, title: str, out_path: Path,
                 xlim=(13.0, 17.0), xlabel: str = "Time (hpf)"):
    plt.figure(figsize=FIGSIZE_WIDE)
    ax = plt.gca()

    # WT
    ax.plot(df_wt["hpf"], df_wt["mean"], marker="o", label=LABEL_WT, color=COLOR_WT)
    ax.fill_between(df_wt["hpf"], df_wt["mean"] - df_wt["sd"], df_wt["mean"] + df_wt["sd"],
                    alpha=ALPHA_FILL, color=COLOR_WT, linewidth=0)

    # MUT
    ax.plot(df_mut["hpf"], df_mut["mean"], marker="o", label=LABEL_MUT, color=COLOR_MUT)
    ax.fill_between(df_mut["hpf"], df_mut["mean"] - df_mut["sd"], df_mut["mean"] + df_mut["sd"],
                    alpha=ALPHA_FILL, color=COLOR_MUT, linewidth=0)

    # NB: movies are aligned by the hpf window parsed from their filename when
    # present (frame_to_hpf_window); end-alignment is only the fallback for
    # names without a window. So the axis is plain absolute hpf, not
    # necessarily "end-aligned".
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    if XTICK_STEP_HPF and np.isfinite(xlim[0]) and np.isfinite(xlim[1]):
        ticks = np.arange(xlim[0], xlim[1] + 1e-9, XTICK_STEP_HPF)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["%.1f" % t for t in ticks],
                           rotation=XTICK_ROT, ha="right")
    ax.legend(frameon=True)

    # y padding so it doesn't stick to bottom/top
    ymin = np.nanmin([df_wt["mean"] - df_wt["sd"], df_mut["mean"] - df_mut["sd"]])
    ymax = np.nanmax([df_wt["mean"] + df_wt["sd"], df_mut["mean"] + df_mut["sd"]])
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = 0.12 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    save_plot(out_path)



def plot_preburst_group_compare(
    wt_movies: list[pd.DataFrame],
    mut_movies: list[pd.DataFrame],
    title: str,
    out_path: Path,
    show_individual_movies: bool = True,
):
    plt.figure(figsize=FIGSIZE_WIDE)
    ax = plt.gca()

    def _stack(movies: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
        if not movies:
            return np.array([]), np.array([[]])
        dt = movies[0]["dt"].to_numpy(dtype=int)
        M = []
        for m in movies:
            M.append(m["value"].to_numpy(dtype=float))
        return dt, np.vstack(M)

    def _plot_group(dt, M, color, label):
        if M.size == 0:
            return
        mean = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] >= 2 else np.zeros_like(mean)

        if show_individual_movies:
            for i in range(M.shape[0]):
                ax.plot(dt, M[i], linewidth=0.9, alpha=ALPHA_MOVIE_LINE, color=color)

        ax.plot(dt, mean, marker="o", label=label, color=color, alpha=ALPHA_MEAN_LINE)
        ax.fill_between(dt, mean - sd, mean + sd, alpha=ALPHA_FILL, color=color, linewidth=0)

    dt_wt, M_wt = _stack(wt_movies)
    dt_mut, M_mut = _stack(mut_movies)

    _plot_group(dt_wt, M_wt, COLOR_WT, LABEL_WT)
    _plot_group(dt_mut, M_mut, COLOR_MUT, LABEL_MUT)

    ax.set_xlabel("Time to burst (frames; Δt=10 min)")
    ax.set_ylabel("Nuclear area (µm²)")
    ax.set_title(title)
    ax.legend(frameon=True)

    # x ticks -3 -2 -1
    if dt_wt.size or dt_mut.size:
        ax.set_xticks(sorted(set(list(dt_wt) + list(dt_mut))))
    else:
        ax.set_xticks([-3, -2, -1])

    # y padding
    ymins = []
    ymaxs = []
    for M in [M_wt, M_mut]:
        if M.size:
            ymins.append(np.nanmin(M))
            ymaxs.append(np.nanmax(M))
    if ymins and ymaxs:
        y0 = float(np.min(ymins)); y1 = float(np.max(ymaxs))
        if np.isfinite(y0) and np.isfinite(y1) and (y1 > y0):
            pad = 0.15 * (y1 - y0)
            ax.set_ylim(y0 - pad, y1 + pad)

    save_plot(out_path)



def _compute_ylim_with_padding(all_values: list[np.ndarray], pad_frac: float = 0.12):
    """
    Determine y-limits with a little headroom for p-value text.
    """
    vals = np.concatenate([v[np.isfinite(v)] for v in all_values if v is not None and len(v) > 0], axis=0) \
        if any(v is not None and len(v) > 0 for v in all_values) else np.array([])

    if vals.size == 0:
        return None

    y_min = float(np.min(vals))
    y_max = float(np.max(vals))

    if np.isclose(y_min, y_max):
        # avoid zero-range
        y_min -= 1.0
        y_max += 1.0

    rng = y_max - y_min
    y_max_padded = y_max + rng * pad_frac
    return y_min, y_max_padded


def _add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], linewidth=1.0, color="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=FONT["pval"])

def boxplot_two_groups(df: pd.DataFrame, value_col: str, group_order: list[str],
                       title: str, ylab: str, out_path: Path):
    # Expect exactly two groups: WT then MUT (your design constraint)
    plt.figure(figsize=FIGSIZE_TALL)
    ax = plt.gca()

    # Map group key -> style
    style = {
        "WT":  {"label": LABEL_WT,  "color": COLOR_WT},
        "MUT": {"label": LABEL_MUT, "color": COLOR_MUT},
    }

    g1 = group_order[0]
    g2 = group_order[1]

    v1 = df.loc[df["group"] == g1, value_col].astype(float).to_numpy()
    v2 = df.loc[df["group"] == g2, value_col].astype(float).to_numpy()
    v1 = v1[np.isfinite(v1)]
    v2 = v2[np.isfinite(v2)]

    # tighter spacing + narrower boxes
    x1, x2 = 1.00, 1.55
    positions = [x1, x2]
    widths = 0.28

    bp = ax.boxplot(
        [v1, v2],
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=1.2, color="black", zorder=2),
        boxprops=dict(linewidth=1.0, color="black", zorder=1),
        whiskerprops=dict(linewidth=1.0, color="black", zorder=1),
        capprops=dict(linewidth=1.0, color="black", zorder=1),
    )

    # Fill colors (solid)
    fill_colors = [style.get(g1, {}).get("color", COLOR_WT),
                   style.get(g2, {}).get("color", COLOR_MUT)]
    for patch, c in zip(bp["boxes"], fill_colors):
        patch.set_facecolor(c)
        patch.set_alpha(1.0)
        patch.set_zorder(1)

    # Overlay per-movie points (solid)
    rng = np.random.default_rng(0)
    jitter = 0.06
    ax.scatter(np.full(v1.shape, x1) + rng.uniform(-jitter, jitter, size=v1.shape),
               v1, s=18, color="black", alpha=0.9, linewidths=0, zorder=5)
    ax.scatter(np.full(v2.shape, x2) + rng.uniform(-jitter, jitter, size=v2.shape),
               v2, s=18, color="black", alpha=0.9, linewidths=0, zorder=5)

    # Labels
    ax.set_xticks(positions)
    ax.set_xticklabels([style.get(g1, {}).get("label", g1),
                        style.get(g2, {}).get("label", g2)])

    ax.set_ylabel(ylab)
    ax.set_title(title)

    # y padding
    allv = np.concatenate([v1, v2]) if (v1.size + v2.size) else np.array([np.nan])
    ymin = np.nanmin(allv)
    ymax = np.nanmax(allv)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = 0.18 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

        # bracket + p-value
        p = ttest_welch(v1, v2)
        if np.isfinite(p):
            y = ymax + 0.06 * (ymax - ymin)
            h = 0.03 * (ymax - ymin)
            _add_sig_bracket(ax, x1, x2, y, h, f"p = {p:.3g}")

    save_plot(out_path)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    cmp_cfg = (cfg.get("compare", {}) or {})
    if not bool(cmp_cfg.get("enable", True)):
        print("[INFO] compare.enable=false; skip.")
        return

    start_hpf = float(cmp_cfg.get("start_hpf", 13.0))
    end_hpf = float(cmp_cfg.get("end_hpf", 17.0))
    dt_min = float(cmp_cfg.get("dt_min", cfg.get("tracking", {}).get("dt_min", 10.0)))
    speed_skip_first_frame = bool(cmp_cfg.get("speed_skip_first_frame", True))

    common_hpf = build_common_hpf_grid(start_hpf, end_hpf, dt_min)

    results_root = Path(cfg["project"]["results_root"])
    out_dir = results_root / "_group_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- collect per-movie summary
    rows = collect_movies(cfg)
    if not rows:
        print("[WARN] No movie_summary.csv found. Did you run tracking?")
        return

    all_df = pd.DataFrame(rows)
    all_df.to_csv(out_dir / "all_movies_summary.csv", index=False)

    group_order = pick_group_order(cfg, all_df)

    # ---- boxplots (WT left, MUT right)
    def pick_col(cands: list[str]) -> str:
        for c in cands:
            if c in all_df.columns:
                return c
        return ""

    col_area = pick_col(["mean_median_area_um2", "mean_area_um2", "median_area_um2"])
    col_speed = pick_col(["mean_median_speed_um_per_min", "mean_median_speed", "median_median_speed_um_per_min"])
    col_bratio = pick_col(["burst_per_track_ratio", "mean_burst_per_track_ratio"])

    if col_area:
        boxplot_two_groups(
            all_df, col_area, group_order,
            title="Per-movie nuclear area",
            ylab="Area (µm²)",
            out_path=out_dir / "boxplot_area",      
        )

    if col_speed:
        boxplot_two_groups(
            all_df, col_speed, group_order,
            title="Per-movie speed",
            ylab="Speed (µm/min)",
            out_path=out_dir / "boxplot_speed",
        )

    if col_bratio:
        boxplot_two_groups(
            all_df, col_bratio, group_order,
            title="Burst / track ratio per movie",
            ylab="Burst events / tracked nuclei",
            out_path=out_dir / "boxplot_burst_track_ratio",
        )

    # ---- time-series compare (mean±SD of per-movie per-frame metrics)
    per_movie_records = []
    grouped = {g: [] for g in group_order}

    # ---- time window QC table (one row per movie)
    time_rows = []

    for _, r in all_df.iterrows():
        group = str(r["group"])
        movie = str(r["movie"])
        track_dir = Path(r["track_dir"])

        ts = load_per_movie_time_series(track_dir)

        # parse start/end hpf from movie name (e.g., 13-16hpf, 14-17hpf, 13-17hpf)
        s_hpf, e_hpf = parse_hpf_window_from_name(movie)

        # align to common grid using per-movie window when available
        aligned = align_one_movie(
            ts=ts,
            common_hpf=common_hpf,
            speed_skip_first_frame=speed_skip_first_frame,
            movie_start_hpf=s_hpf,
            movie_end_hpf=e_hpf,
            default_end_hpf=end_hpf,
            default_dt_min=dt_min,
        )
        aligned["group"] = group
        aligned["movie"] = movie
        per_movie_records.append(aligned)

        if group in grouped:
            grouped[group].append(aligned)

        # --- write QC row: what window got used + what hpf range exists in this aligned df
        # Determine effective mapping range for this movie (based on parsed window or fallback end-aligned)
        if (s_hpf is not None) and (e_hpf is not None):
            used_start = float(s_hpf); used_end = float(e_hpf)
            used_mode = "parsed_from_name"
        else:
            used_start = float(frame_to_hpf_end_aligned(0, int(r["n_frames"]), float(end_hpf), float(dt_min)))
            used_end = float(end_hpf)
            used_mode = "fallback_end_aligned_to_default_end"

        # what timepoints are present (non-NaN) for a representative metric
        # pick n_tracked if available, else median_area
        rep = aligned["n_tracked"].to_numpy(dtype=float)
        if np.all(np.isnan(rep)):
            rep = aligned["median_area_um2"].to_numpy(dtype=float)

        has = ~np.isnan(rep)
        present_min = float(aligned.loc[has, "hpf"].min()) if np.any(has) else np.nan
        present_max = float(aligned.loc[has, "hpf"].max()) if np.any(has) else np.nan

        time_rows.append({
            "group": group,
            "movie": movie,
            "n_frames": int(r.get("n_frames", np.nan)) if "n_frames" in r else np.nan,
            "parsed_start_hpf": s_hpf,
            "parsed_end_hpf": e_hpf,
            "used_start_hpf": used_start,
            "used_end_hpf": used_end,
            "used_mode": used_mode,
            "present_hpf_min": present_min,
            "present_hpf_max": present_max,
        })

    # write window QC
    pd.DataFrame(time_rows).to_csv(out_dir / "movie_time_windows.csv", index=False)


    if per_movie_records:
        per_movie_df = pd.concat(per_movie_records, ignore_index=True)
        per_movie_df.to_csv(out_dir / "aligned_time_series_per_movie.csv", index=False)

    # Require WT and MUT exist for time series compare
    if "WT" not in grouped or "MUT" not in grouped or (len(grouped["WT"]) == 0) or (len(grouped["MUT"]) == 0):
        print("[WARN] Need both WT and MUT movies for time-series compare.")
        return


    # ---- PRE-BURST area dynamics (burst-aligned)
    pre_n = int((cmp_cfg.get("preburst_n_pre_frames", 3)))
    pre_agg = str((cmp_cfg.get("preburst_agg", "mean"))).strip().lower()
    show_indiv = bool(cmp_cfg.get("preburst_show_individual_movies", True))

    wt_pre_movies = []
    mut_pre_movies = []

    # We'll reuse all_df rows to locate track_dir for each movie
    for _, r in all_df.iterrows():
        g = r["group"]
        track_dir = Path(r["track_dir"])
        pre_df = load_preburst_traces(track_dir)

        # If no bursts in this movie, skip it (cannot define pre-burst trace)
        if pre_df is None or pre_df.empty:
            continue

        per_movie = summarize_preburst_per_movie(pre_df, n_pre_frames=pre_n, agg=pre_agg)
        per_movie["group"] = g
        per_movie["movie"] = r["movie"]

        if g == "WT":
            wt_pre_movies.append(per_movie)
        elif g == "MUT":
            mut_pre_movies.append(per_movie)

    # Save a tidy table for downstream QC / stats
    if wt_pre_movies or mut_pre_movies:
        pre_all = pd.concat(wt_pre_movies + mut_pre_movies, ignore_index=True)
        pre_all.to_csv(out_dir / "preburst_area_per_movie.csv", index=False)

    # Plot if both groups have at least 1 movie with bursts
    if wt_pre_movies and mut_pre_movies:
        plot_preburst_group_compare(
            wt_pre_movies,
            mut_pre_movies,
            title=f"Pre-burst nuclear area dynamics (per-movie {pre_agg}; mean±SD across movies)",
            out_path=out_dir / "preburst_area_vs_time.png",
            show_individual_movies=show_indiv,
        )
    else:
        print("[WARN] Pre-burst plot skipped: need at least one bursting movie in BOTH WT and MUT.")

    
    # ---- Pre-burst boxplot at dt = -1 (WT vs MUT)
    # We use per-movie summary value at dt=-1 (consistent with movie-level philosophy)
    def _dt_value(movies: list[pd.DataFrame], dt_target: int) -> np.ndarray:
        vals = []
        for m in movies:
            hit = m.loc[m["dt"] == dt_target, "value"]
            if len(hit) == 1 and np.isfinite(float(hit.iloc[0])):
                vals.append(float(hit.iloc[0]))
        return np.array(vals, dtype=float)

    wt_dt1 = _dt_value(wt_pre_movies, -1)
    mut_dt1 = _dt_value(mut_pre_movies, -1)

    if wt_dt1.size >= 2 and mut_dt1.size >= 2:
        # Build a tiny df to reuse your boxplot function (expects df with group/value_col)
        tmp = pd.DataFrame({
            "group": (["WT"] * wt_dt1.size) + (["MUT"] * mut_dt1.size),
            "preburst_area_dt_minus1": np.concatenate([wt_dt1, mut_dt1]),
        })
        boxplot_two_groups(
            tmp,
            value_col="preburst_area_dt_minus1",
            group_order=["WT", "MUT"],
            title="Pre-burst nuclear area at -1 frame",
            ylab="Nuclear area (µm²)",
            out_path=out_dir / "boxplot_preburst_area_dt_minus1.png",
        )
    else:
        print("[WARN] Pre-burst dt=-1 boxplot skipped: need >=2 movies per group with valid dt=-1 value.")


    # area: per-frame median_area_um2
    wt_area = mean_sd_over_movies(grouped["WT"], "median_area_um2")
    mut_area = mean_sd_over_movies(grouped["MUT"], "median_area_um2")
    plot_mean_sd(
        wt_area, mut_area,
        ylab="Median tracked nuclear area (µm²)",
        title="Median tracked nuclear area vs time (mean±SD across movies)",
        out_path=out_dir / "area_vs_time_mean_sd",
        xlim=(start_hpf, end_hpf),
    )

    # speed: per-frame median_speed_um_per_min
    wt_spd = mean_sd_over_movies(grouped["WT"], "median_speed_um_per_min")
    mut_spd = mean_sd_over_movies(grouped["MUT"], "median_speed_um_per_min")
    plot_mean_sd(
        wt_spd, mut_spd,
        ylab="Median tracked speed (µm/min)",
        title="Median tracked speed vs time (mean±SD across movies)",
        out_path=out_dir / "speed_vs_time_mean_sd",
        xlim=(start_hpf, end_hpf),
    )

    # bursts: n_bursts
    wt_bst = mean_sd_over_movies(grouped["WT"], "n_bursts")
    mut_bst = mean_sd_over_movies(grouped["MUT"], "n_bursts")
    plot_mean_sd(
        wt_bst, mut_bst,
        ylab="Burst events (# / frame)",
        title="Burst events vs time (mean±SD across movies)",
        out_path=out_dir / "bursts_vs_time_mean_sd",
        xlim=(start_hpf, end_hpf),
    )

    # width: width_um
    wt_w = mean_sd_over_movies(grouped["WT"], "width_um")
    mut_w = mean_sd_over_movies(grouped["MUT"], "width_um")
    plot_mean_sd(
        wt_w, mut_w,
        ylab="LPM left-right width (µm)",
        title="LPM width vs time (mean±SD across movies)",
        out_path=out_dir / "width_vs_time_mean_sd",
        xlim=(start_hpf, end_hpf),
    )

    # n_tracked
    wt_n = mean_sd_over_movies(grouped["WT"], "n_tracked")
    mut_n = mean_sd_over_movies(grouped["MUT"], "n_tracked")
    plot_mean_sd(
        wt_n, mut_n,
        ylab="Tracked nuclei (# / frame)",
        title="Tracked nuclei vs time (mean±SD across movies)",
        out_path=out_dir / "n_tracked_vs_time_mean_sd",
        xlim=(start_hpf, end_hpf),
    )

    print(f"[OK] Compare outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
