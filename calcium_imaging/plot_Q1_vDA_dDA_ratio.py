# plot_Q1_vDA_dDA_ratio.py
# -*- coding: utf-8 -*-
"""
Q1: vDA/dDA ratio (E3 control only)
===================================

Reads ``Q1_ratio_prepost.csv`` and ``Q1_ratio_summary_prepost.csv`` (produced
by ``calcium_build_tables.py``) and produces, per pair_id:

  * lines     : vDA/dDA ratio over time (per-embryo traces + pooled mean ± SEM),
                pre phase in black, post phase in the control colour, separated
                by the visual remount gap.
  * box_preonly  : a single box of per-embryo pre-phase median ratio.
  * box_prepost  : pre vs post per-embryo median ratio side by side.

Q1 is computed for the control condition only (cfg.CONTROL_CONDITION) — the
vDA/dDA ratio is a within-control read-out, not a drug comparison.

All tunable parameters live in ``calcium_config.py``. Was
``plot_Q1_ratio_prepost_v5.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import calcium_config as cfg
from plot_style import STYLE


# =============================================================================
# User-facing knobs
# =============================================================================

ONLY: Optional[str] = None   # None / "lines" / "box_preonly" / "box_prepost"
PAIR_ID: Optional[str] = None

CONDITION     = cfg.CONTROL_CONDITION    # Q1 is control-only
PRE_SHOW_MIN  = cfg.PRE_SHOW_MIN
POST_SHOW_MIN = cfg.POST_SHOW_MIN
GAP_W         = cfg.GAP_W

plt.rcParams.update({
    "font.family":      cfg.FONT["family"],
    "font.sans-serif":  cfg.FONT["sans"],
    "axes.linewidth":   STYLE["lw"]["axis"],
    # cfg.FONT is authoritative for per-dataset figures: axis labels and tick
    # labels used to fall through to matplotlib's 10 pt default, ignoring the
    # configured values entirely.
    "font.size":        cfg.FONT["tick"],
    "axes.titlesize":   cfg.FONT["title"],
    "axes.labelsize":   cfg.FONT["label"],
    "xtick.labelsize":  cfg.FONT["tick"],
    "ytick.labelsize":  cfg.FONT["tick"],
    "legend.fontsize":  cfg.FONT["legend"],
})


# =============================================================================
# Style and I/O helpers
# =============================================================================

def style_axes(ax):
    if STYLE["axes"]["hide_top_right"]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out",
                   length=STYLE["axes"]["tick_len"],
                   width=STYLE["axes"]["tick_w"])
    ax.grid(False)


def save_both(fig, rel: Path):
    """Write PNG and SVG under cfg.plots_*_dir() / cfg.Q1_PLOT_SUBDIR / rel.*"""
    png_dir = cfg.plots_png_dir() / cfg.Q1_PLOT_SUBDIR
    svg_dir = cfg.plots_svg_dir() / cfg.Q1_PLOT_SUBDIR
    (png_dir / rel.parent).mkdir(parents=True, exist_ok=True)
    (svg_dir / rel.parent).mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / rel.with_suffix(".png"), dpi=STYLE["dpi"], bbox_inches="tight")
    fig.savefig(svg_dir / rel.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def pick_first(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing {what} column. Tried: {candidates}. Found: {list(df.columns)}")


def build_pre_axis(tt: np.ndarray) -> np.ndarray:
    tt = np.asarray(tt, float)
    finite = tt[np.isfinite(tt)]
    if finite.size == 0:
        return tt
    return tt - np.nanmax(finite)


def build_post_axis(tt: np.ndarray) -> np.ndarray:
    tt = np.asarray(tt, float)
    finite = tt[np.isfinite(tt)]
    if finite.size == 0:
        return tt
    return tt - np.nanmin(finite)


def draw_gap(ax):
    ax.axvspan(0, GAP_W, color="0.92", zorder=0)
    ax.axvline(0, color="0.75", lw=1.2)
    ax.axvline(GAP_W, color="0.75", lw=1.2)
    ax.set_xlim(-PRE_SHOW_MIN, GAP_W + POST_SHOW_MIN)


def xlabel_gap_axis() -> str:
    return f"Time (min)  pre(-{int(PRE_SHOW_MIN)}..0)  gap  post(0..{int(POST_SHOW_MIN)})"


def nanmean_sem(A: np.ndarray):
    m = np.nanmean(A, axis=0)
    n = np.sum(np.isfinite(A), axis=0)
    s = np.nanstd(A, axis=0, ddof=1)
    sem = np.where(n > 0, s / np.sqrt(np.maximum(n, 1)), np.nan)
    return m, sem


# =============================================================================
# lines: vDA/dDA ratio over time
# =============================================================================

def plot_lines_ratio(ratio: pd.DataFrame, pair_id: str):
    ratio_col = pick_first(ratio, ["ratio_mean_bgsub", "ratio_post", "ratio"], "ratio")
    phase_col = pick_first(ratio, ["phase"], "phase")
    time_col  = pick_first(ratio, ["time_min"], "time_min")
    emb_col   = pick_first(ratio, ["embryo_id"], "embryo_id")
    cond_col  = pick_first(ratio, ["condition"], "condition")

    pub = bool(getattr(cfg, "PUBLICATION_MODE", False))

    df = ratio[(ratio[cond_col] == CONDITION) & (ratio["pair_id"].astype(str) == str(pair_id))].copy()
    if df.empty:
        return

    pre  = df[df[phase_col] == "pre"].copy()
    post = df[df[phase_col] == "post"].copy()

    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["l3"])
    col = cfg.color_for(CONDITION)

    # ---------- pre: per-embryo thin traces + pooled (black) ----------
    if not pre.empty:
        grid = np.linspace(-PRE_SHOW_MIN, 0, 201)
        A = []
        for emb, g in pre.groupby(emb_col):
            t = build_pre_axis(g[time_col].to_numpy(float))
            y = pd.to_numeric(g[ratio_col], errors="coerce").to_numpy(float)
            m = (t >= -PRE_SHOW_MIN) & (t <= 0)
            t, y = t[m], y[m]
            if len(t) >= 2:
                order = np.argsort(t)
                if not pub:
                    ax.plot(t[order], y[order], color="black", alpha=0.15, lw=1.0)
                A.append(np.interp(grid, t[order], y[order], left=np.nan, right=np.nan))
        if A:
            A = np.vstack(A)
            mpre, spre = nanmean_sem(A)
            ax.fill_between(grid, mpre - spre, mpre + spre, color="black",
                            alpha=STYLE["alpha"]["pooled_fill"], linewidth=0)
            ax.plot(grid, mpre, color="black", lw=STYLE["lw"]["pooled"],
                    alpha=STYLE["alpha"]["pooled_line"],
                    label=f"pre pooled (n_emb={A.shape[0]})")

    # ---------- post: per-embryo thin traces + pooled (control colour) ----------
    if not post.empty:
        grid = np.linspace(0, POST_SHOW_MIN, 201)
        A = []
        for emb, g in post.groupby(emb_col):
            t = build_post_axis(g[time_col].to_numpy(float))
            y = pd.to_numeric(g[ratio_col], errors="coerce").to_numpy(float)
            m = (t >= 0) & (t <= POST_SHOW_MIN)
            t, y = t[m], y[m]
            if len(t) >= 2:
                order = np.argsort(t)
                if not pub:
                    ax.plot(t[order] + GAP_W, y[order], color=col, alpha=0.25, lw=1.2)
                A.append(np.interp(grid, t[order], y[order], left=np.nan, right=np.nan))
        if A:
            A = np.vstack(A)
            mpost, spost = nanmean_sem(A)
            ax.fill_between(grid + GAP_W, mpost - spost, mpost + spost, color=col,
                            alpha=STYLE["alpha"]["pooled_fill"], linewidth=0)
            ax.plot(grid + GAP_W, mpost, color=col, lw=STYLE["lw"]["pooled"],
                    alpha=STYLE["alpha"]["pooled_line"],
                    label=f"post pooled (n_emb={A.shape[0]})")

    draw_gap(ax)

    if pub:
        ax.set_title(cfg.display_name(CONDITION),
                     pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
    else:
        ax.set_title(f"Q1 vDA/dDA ratio | {pair_id} | {cfg.display_name(CONDITION)}",
                     pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
    ax.set_xlabel(xlabel_gap_axis())
    ax.set_ylabel("vDA/dDA ratio" if pub else f"ratio ({ratio_col})")
    style_axes(ax)
    if not pub:
        ax.legend(frameon=False, fontsize=cfg.FONT["legend"])

    save_both(fig, Path(str(pair_id)) / "lines"
              / f"Q1_ratio_prepost_gapaxis_0to{int(POST_SHOW_MIN)}min")


# =============================================================================
# box helpers (Prism-style: dots with black edge, jitter 0.18)
# =============================================================================

DOT_JITTER  = 0.18
DOT_EDGE_LW = 0.5


def _style_box(bp, color):
    for b in bp["boxes"]:
        b.set_facecolor(color)
        b.set_alpha(STYLE["box"]["box_alpha"])
        b.set_edgecolor("black")
        b.set_linewidth(STYLE["box"]["box_edge_lw"])
    for m in bp["medians"]:
        m.set_color("black")
        m.set_linewidth(STYLE["box"]["median_lw"])


def _scatter_dots(ax, x_center, y, color, rng):
    if len(y) == 0:
        return
    jitter = (rng.random(len(y)) - 0.5) * DOT_JITTER
    ax.scatter(np.full(len(y), x_center) + jitter, y,
               s=STYLE["box"]["dot_size"],
               facecolor=color, edgecolor="black",
               linewidth=DOT_EDGE_LW, zorder=3,
               alpha=STYLE["box"].get("dot_alpha", 1.0))


def plot_box_preonly(summary: pd.DataFrame, pair_id: str):
    phase_col = pick_first(summary, ["phase"], "phase")
    val_col   = pick_first(summary, ["ratio_median"], "ratio_median")
    pub = bool(getattr(cfg, "PUBLICATION_MODE", False))

    df = summary[(summary["pair_id"].astype(str) == str(pair_id)) &
                 (summary["condition"] == CONDITION) &
                 (summary[phase_col] == "pre")].copy()
    y = pd.to_numeric(df[val_col], errors="coerce").to_numpy(float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return

    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["box"])
    bp = ax.boxplot([y], widths=STYLE["box"]["width"],
                    showfliers=False, patch_artist=True)
    col = cfg.color_for(CONDITION)
    _style_box(bp, col)

    rng = np.random.default_rng(0)
    _scatter_dots(ax, 1, y, col, rng)

    ax.set_xticks([1])
    ax.set_xticklabels(["pre"])
    if pub:
        ax.set_title(cfg.display_name(CONDITION),
                     pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
    else:
        ax.set_title(f"Q1 pre-only ratio (embryo unit)\n{pair_id} | {cfg.display_name(CONDITION)}",
                     pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
    ax.set_ylabel("median(vDA/dDA)" if pub else "median(vDA/dDA ratio)")
    style_axes(ax)
    save_both(fig, Path(str(pair_id)) / "box" / "pre_only" / "Q1_ratio_pre_only")


def plot_box_prepost(summary: pd.DataFrame, pair_id: str):
    phase_col = pick_first(summary, ["phase"], "phase")
    val_col   = pick_first(summary, ["ratio_median"], "ratio_median")
    pub = bool(getattr(cfg, "PUBLICATION_MODE", False))

    df = summary[(summary["pair_id"].astype(str) == str(pair_id)) &
                 (summary["condition"] == CONDITION)].copy()
    pre  = pd.to_numeric(df[df[phase_col] == "pre"][val_col], errors="coerce").to_numpy(float)
    post = pd.to_numeric(df[df[phase_col] == "post"][val_col], errors="coerce").to_numpy(float)
    pre  = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    if len(pre) == 0 and len(post) == 0:
        return

    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["box"])
    data = [pre, post]
    bp = ax.boxplot(data, widths=STYLE["box"]["width"],
                    showfliers=False, patch_artist=True)
    col = cfg.color_for(CONDITION)
    _style_box(bp, col)

    rng = np.random.default_rng(0)
    for i, yy in enumerate(data, start=1):
        _scatter_dots(ax, i, yy, col, rng)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["pre", "post"])
    if pub:
        ax.set_title(cfg.display_name(CONDITION),
                     pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
    else:
        ax.set_title(f"Q1 ratio median (embryo unit)\n{pair_id} | {cfg.display_name(CONDITION)}",
                     pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
    ax.set_ylabel("median(vDA/dDA)" if pub else "median(vDA/dDA ratio)")
    style_axes(ax)
    save_both(fig, Path(str(pair_id)) / "box" / "pre_vs_post" / "Q1_ratio_pre_vs_post")


# =============================================================================
# Main
# =============================================================================

def main():
    ratio_path = cfg.tables_dir() / "Q1_ratio_prepost.csv"
    summ_path  = cfg.tables_dir() / "Q1_ratio_summary_prepost.csv"
    if not ratio_path.exists():
        raise FileNotFoundError(f"Missing table: {ratio_path}")
    if not summ_path.exists():
        raise FileNotFoundError(f"Missing table: {summ_path}")

    ratio = pd.read_csv(ratio_path)
    summ  = pd.read_csv(summ_path)

    print(f"Input analysis window: {cfg.window_suffix()} min")
    print(f"Control condition: {CONDITION}")

    pair_ids = [PAIR_ID] if PAIR_ID else sorted(ratio["pair_id"].astype(str).unique())

    for pid in pair_ids:
        if ONLY in (None, "lines"):
            plot_lines_ratio(ratio, pid)
        if ONLY in (None, "box_preonly"):
            plot_box_preonly(summ, pid)
        if ONLY in (None, "box_prepost"):
            plot_box_prepost(summ, pid)

    print("[OK] Q1_vDA_dDA_ratio done.")


if __name__ == "__main__":
    main()
