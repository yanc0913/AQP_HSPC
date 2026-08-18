# plot_Q3_vDA_trace.py
# -*- coding: utf-8 -*-
"""
Q3: vDA band GCaMP trace (all conditions)
=========================================

Reads ``Q3_vDA_trace_prepost.csv`` (produced by ``calcium_build_tables.py``)
and produces, per pair_id:

  * One trace panel per condition: per-batch (repeat) mean ± SEM in faint
    lines, plus the pooled mean ± SEM in a thick line. Pre phase in black,
    post phase in the condition colour, separated by the visual remount gap.

  * One "overlay" panel per pair_id: the pooled post-phase mean of every
    condition drawn together (each in its own colour) for direct comparison.

All tunable parameters live in ``calcium_config.py``. Was ``plot_Q2_DA_v5.py``;
renamed Q3 in the new scheme (single-cell events stay Q2; the vDA band trace
is Q3).
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

ONLY: Optional[str] = None   # None / "lines" / "overlay"
PAIR_ID: Optional[str] = None

PRE_SHOW_MIN  = cfg.PRE_SHOW_MIN
POST_SHOW_MIN = cfg.POST_SHOW_MIN
GAP_W         = cfg.GAP_W

plt.rcParams.update({
    "font.family":     cfg.FONT["family"],
    "font.sans-serif": cfg.FONT["sans"],
    "axes.linewidth":  STYLE["lw"]["axis"],
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
    """Write the figure to PNG and SVG under cfg.plots_*_dir() / cfg.Q3_VDA_PLOT_SUBDIR / rel.*"""
    png_dir = cfg.plots_png_dir() / cfg.Q3_VDA_PLOT_SUBDIR
    svg_dir = cfg.plots_svg_dir() / cfg.Q3_VDA_PLOT_SUBDIR
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


def ylabel_for(trace_col: str, pub: bool) -> str:
    """Y-axis label: concise mathtext in publication mode, raw column name in
    monitoring mode (so you can tell which trace variant was plotted)."""
    if pub:
        # mathtext so the subscript 0 renders on any system font (Arial lacks
        # a glyph for the unicode subscript-zero, which would show as a box).
        return r"vDA $F/F_0$"
    return trace_col


# =============================================================================
# Condition resolution (uses Q3 order + unknown-drug handling)
# =============================================================================

def resolve_conditions(present_in_data: set[str]) -> list[str]:
    """Return ordered conditions to plot for Q3.

    Begins with cfg.Q3_CONDITIONS_ORDER (filtered to those present), then
    appends unknown drugs (if cfg.INCLUDE_UNKNOWN_IN_PLOTS) with a warning.
    """
    base = [c for c in cfg.Q3_CONDITIONS_ORDER if c in present_in_data]

    # Conditions present in data but missing from the configured order
    known = cfg.all_known_conditions()
    missing_known = sorted(
        (present_in_data & known) - set(cfg.Q3_CONDITIONS_ORDER)
    )
    if missing_known:
        print(f"  ⚠️  Conditions present in data but not in Q3_CONDITIONS_ORDER: "
              f"{missing_known}  → appended to the end.")
        base = base + missing_known

    if cfg.INCLUDE_UNKNOWN_IN_PLOTS:
        unknown = sorted(present_in_data - known - set(base))
        if unknown:
            print(f"  ⚠️  Unknown drugs (no alias): {unknown}  → plotted in "
                  f"{cfg.COLOR_FALLBACK}.")
        return base + unknown
    return base


# =============================================================================
# Per-condition trace builder (returns grids + pooled mean/sem for reuse)
# =============================================================================

def _collect_pre(d: pd.DataFrame, time_col, trace_col, emb_col, batch_col, grid):
    """Return (per-batch list of (batch_id, A), pooled-A-or-None) for pre phase."""
    per_batch = []
    pooled_blocks = []
    for batch_id, gb in d.groupby(batch_col):
        traces = []
        for emb, ge in gb.groupby(emb_col):
            t = build_pre_axis(ge[time_col].to_numpy(float))
            y = pd.to_numeric(ge[trace_col], errors="coerce").to_numpy(float)
            m = (t >= -PRE_SHOW_MIN) & (t <= 0)
            t, y = t[m], y[m]
            if len(t) >= 2:
                order = np.argsort(t)
                traces.append(np.interp(grid, t[order], y[order], left=np.nan, right=np.nan))
        if traces:
            A = np.vstack(traces)
            per_batch.append((str(batch_id), A))
            pooled_blocks.append(A)
    pooled = np.vstack(pooled_blocks) if pooled_blocks else None
    return per_batch, pooled


def _collect_post(d: pd.DataFrame, time_col, trace_col, emb_col, batch_col, grid):
    per_batch = []
    pooled_blocks = []
    for batch_id, gb in d.groupby(batch_col):
        traces = []
        for emb, ge in gb.groupby(emb_col):
            t = build_post_axis(ge[time_col].to_numpy(float))
            y = pd.to_numeric(ge[trace_col], errors="coerce").to_numpy(float)
            m = (t >= 0) & (t <= POST_SHOW_MIN)
            t, y = t[m], y[m]
            if len(t) >= 2:
                order = np.argsort(t)
                traces.append(np.interp(grid, t[order], y[order], left=np.nan, right=np.nan))
        if traces:
            A = np.vstack(traces)
            per_batch.append((str(batch_id), A))
            pooled_blocks.append(A)
    pooled = np.vstack(pooled_blocks) if pooled_blocks else None
    return per_batch, pooled


# =============================================================================
# Plot: one panel per condition
# =============================================================================

def plot_pair_per_condition(df: pd.DataFrame, pair_id: str, cond_order: list[str]):
    phase_col = pick_first(df, ["phase"], "phase")
    time_col  = pick_first(df, ["time_min"], "time_min")
    emb_col   = pick_first(df, ["embryo_id"], "embryo_id")
    batch_col = pick_first(df, ["batch_id"], "batch_id")
    cond_col  = pick_first(df, ["condition"], "condition")
    trace_col = pick_first(
        df,
        ["gcamp_vDA_trace_main", "gcamp_vDA_trace_main_pre_base", "gcamp_vDA_trace_main_norm"],
        "trace",
    )

    pub = bool(getattr(cfg, "PUBLICATION_MODE", False))
    sub = df[df["pair_id"].astype(str) == str(pair_id)].copy()
    if sub.empty:
        return

    x_pre_grid  = np.linspace(-PRE_SHOW_MIN, 0, 201)
    x_post_grid = np.linspace(0, POST_SHOW_MIN, 201)

    for cond in cond_order:
        d = sub[sub[cond_col].astype(str) == str(cond)].copy()
        if d.empty:
            continue

        col = cfg.color_for(cond)
        fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["l3"])

        # PRE: per-batch + pooled, in black (keeps pre visually neutral)
        pre = d[d[phase_col] == "pre"]
        if not pre.empty:
            per_batch, pooled = _collect_pre(pre, time_col, trace_col, emb_col, batch_col, x_pre_grid)
            for batch_id, A in per_batch:
                mrep, srep = nanmean_sem(A)
                ax.fill_between(x_pre_grid, mrep - srep, mrep + srep, color="black",
                                alpha=STYLE["alpha"]["repeat_fill"], linewidth=0)
                ax.plot(x_pre_grid, mrep, color="black", lw=STYLE["lw"]["repeat"],
                        alpha=STYLE["alpha"]["repeat_line"],
                        label=f"{batch_id} pre (n_emb={A.shape[0]})")
            if pooled is not None:
                mpre, spre = nanmean_sem(pooled)
                ax.fill_between(x_pre_grid, mpre - spre, mpre + spre, color="black",
                                alpha=STYLE["alpha"]["pooled_fill"], linewidth=0)
                ax.plot(x_pre_grid, mpre, color="black", lw=STYLE["lw"]["pooled"],
                        alpha=STYLE["alpha"]["pooled_line"],
                        label=f"pooled pre (n_emb={pooled.shape[0]})")

        # POST: per-batch + pooled, in condition colour
        post = d[d[phase_col] == "post"]
        if not post.empty:
            per_batch, pooled = _collect_post(post, time_col, trace_col, emb_col, batch_col, x_post_grid)
            for batch_id, A in per_batch:
                mrep, srep = nanmean_sem(A)
                ax.fill_between(x_post_grid + GAP_W, mrep - srep, mrep + srep, color=col,
                                alpha=STYLE["alpha"]["repeat_fill"], linewidth=0)
                ax.plot(x_post_grid + GAP_W, mrep, color=col, lw=STYLE["lw"]["repeat"],
                        alpha=STYLE["alpha"]["repeat_line"],
                        label=f"{batch_id} post (n_emb={A.shape[0]})")
            if pooled is not None:
                mpost, spost = nanmean_sem(pooled)
                ax.fill_between(x_post_grid + GAP_W, mpost - spost, mpost + spost, color=col,
                                alpha=STYLE["alpha"]["pooled_fill"], linewidth=0)
                ax.plot(x_post_grid + GAP_W, mpost, color=col, lw=STYLE["lw"]["pooled"],
                        alpha=STYLE["alpha"]["pooled_line"],
                        label=f"pooled post (n_emb={pooled.shape[0]})")

        draw_gap(ax)

        if pub:
            ax.set_title(cfg.display_name(cond),
                         pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
        else:
            ax.set_title(
                f"Q3 vDA | {pair_id} | {cfg.display_name(cond)}\n"
                f"pre(-{int(PRE_SHOW_MIN)}..0) + post(0..{int(POST_SHOW_MIN)}) gap-axis",
                pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
            )
        ax.set_xlabel(xlabel_gap_axis())
        ax.set_ylabel(ylabel_for(trace_col, pub))
        style_axes(ax)

        # Legend: hidden in publication mode (use figure caption instead)
        if not pub:
            ax.legend(frameon=False, fontsize=cfg.FONT["legend"])

        save_both(fig, Path(str(pair_id)) / cond
                  / f"Q3_vDA_{cond}_gapaxis_0to{int(POST_SHOW_MIN)}min")


# =============================================================================
# Plot: one overlay panel per pair (all conditions, pooled mean only)
# =============================================================================

def plot_pair_overlay(df: pd.DataFrame, pair_id: str, cond_order: list[str]):
    phase_col = pick_first(df, ["phase"], "phase")
    time_col  = pick_first(df, ["time_min"], "time_min")
    emb_col   = pick_first(df, ["embryo_id"], "embryo_id")
    batch_col = pick_first(df, ["batch_id"], "batch_id")
    cond_col  = pick_first(df, ["condition"], "condition")
    trace_col = pick_first(
        df,
        ["gcamp_vDA_trace_main", "gcamp_vDA_trace_main_pre_base", "gcamp_vDA_trace_main_norm"],
        "trace",
    )

    pub = bool(getattr(cfg, "PUBLICATION_MODE", False))
    sub = df[df["pair_id"].astype(str) == str(pair_id)].copy()
    if sub.empty:
        return

    x_pre_grid  = np.linspace(-PRE_SHOW_MIN, 0, 201)
    x_post_grid = np.linspace(0, POST_SHOW_MIN, 201)

    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["l3"])
    any_plotted = False

    for cond in cond_order:
        d = sub[sub[cond_col].astype(str) == str(cond)].copy()
        if d.empty:
            continue
        col = cfg.color_for(cond)

        # PRE pooled mean (thin, condition colour, to show common baseline)
        pre = d[d[phase_col] == "pre"]
        if not pre.empty:
            _, pooled = _collect_pre(pre, time_col, trace_col, emb_col, batch_col, x_pre_grid)
            if pooled is not None:
                mpre, _ = nanmean_sem(pooled)
                ax.plot(x_pre_grid, mpre, color=col, lw=STYLE["lw"]["repeat"],
                        alpha=0.5)

        # POST pooled mean ± SEM (thick, condition colour)
        post = d[d[phase_col] == "post"]
        if not post.empty:
            _, pooled = _collect_post(post, time_col, trace_col, emb_col, batch_col, x_post_grid)
            if pooled is not None:
                mpost, spost = nanmean_sem(pooled)
                ax.fill_between(x_post_grid + GAP_W, mpost - spost, mpost + spost,
                                color=col, alpha=STYLE["alpha"]["pooled_fill"], linewidth=0)
                ax.plot(x_post_grid + GAP_W, mpost, color=col, lw=STYLE["lw"]["pooled"],
                        alpha=STYLE["alpha"]["pooled_line"],
                        label=f"{cfg.display_name(cond)} (n_emb={pooled.shape[0]})")
                any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    draw_gap(ax)

    if pub:
        ax.set_title(str(pair_id), pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
    else:
        ax.set_title(
            f"Q3 vDA overlay | {pair_id}\n"
            f"pooled post mean ± SEM, all conditions",
            pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
        )
    ax.set_xlabel(xlabel_gap_axis())
    ax.set_ylabel(ylabel_for(trace_col, pub))
    style_axes(ax)
    # Overlay keeps its legend even in publication mode — it's the only way
    # to tell the conditions apart (each is a coloured line, not a separate panel).
    ax.legend(frameon=False, fontsize=cfg.FONT["legend"])

    save_both(fig, Path(str(pair_id)) / f"Q3_vDA_overlay_0to{int(POST_SHOW_MIN)}min")


# =============================================================================
# Main
# =============================================================================

def main():
    path = cfg.tables_dir() / "Q3_vDA_trace_prepost.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")

    df = pd.read_csv(path)

    print(f"Input analysis window: {cfg.window_suffix()} min")
    print(f"Conditions in data: {sorted(df['condition'].astype(str).unique())}")

    pair_ids = [PAIR_ID] if PAIR_ID else sorted(df["pair_id"].astype(str).unique())

    for pid in pair_ids:
        present = set(df[df["pair_id"].astype(str) == str(pid)]["condition"].astype(str).unique())
        cond_order = resolve_conditions(present)
        if not cond_order:
            continue

        if ONLY in (None, "lines"):
            plot_pair_per_condition(df, pid, cond_order)
        if ONLY in (None, "overlay"):
            plot_pair_overlay(df, pid, cond_order)

    print("[OK] Q3_vDA_trace done.")


if __name__ == "__main__":
    main()
