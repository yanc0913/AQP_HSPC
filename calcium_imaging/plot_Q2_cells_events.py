# plot_Q2_cells_events.py
# -*- coding: utf-8 -*-
"""
Q2: Single-cell calcium event plots and boxplots
================================================

Reads tables produced by ``calcium_build_tables.py`` and produces:

  L1: per-embryo, all individual cell traces (pre + post, gap-axis)
  L2: per-embryo mean trace across cells (pre + post)
  L3: per-batch (repeat) mean ± SEM, plus pooled mean ± SEM; with overlays
  BOX: embryo-unit event-rate boxplot with statistics

All tunable parameters live in ``calcium_config.py``.

Statistics
----------
The omnibus test (corner annotation) is chosen automatically:
  * 2 groups → Welch's t-test
  * ≥3 groups → one-way ANOVA
This can be overridden via ``cfg.STATS["omnibus_test"]``.

Pairwise post-hoc tests are always Welch's t-test with Holm-Bonferroni
multiple-comparison correction (proper monotonic step-down implementation).

Renamed from plot_Q2_cells_levels_v6.py.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import f_oneway, ttest_ind

import calcium_config as cfg
from plot_style import STYLE

# statsmodels powers the two-way (genotype x drug) ANOVA + Tukey HSD path used
# when more than one genotype is present. Imported lazily-guarded so single-
# genotype (WT) datasets still work if statsmodels is ever absent.
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False


# =============================================================================
# User-facing knobs (rarely changed; bulk of config in calcium_config.py)
# =============================================================================

ONLY: Optional[str] = None   # None / "L1" / "L2" / "L3" / "box"
PAIR_ID: Optional[str] = None

# Pull frequently-used values from config
PRE_SHOW_MIN  = cfg.PRE_SHOW_MIN
POST_SHOW_MIN = cfg.POST_SHOW_MIN
GAP_W         = cfg.GAP_W

# L3 overlay options
L3_EXTRA_OVERLAYS    = True
L3_OVERLAY_DRAW_PRE  = True
L3_OVERLAY_MAX_CELL  = 9999


plt.rcParams.update({
    "font.family":      cfg.FONT["family"],
    "font.sans-serif":  cfg.FONT["sans"],
    "axes.linewidth":   STYLE["lw"]["axis"],
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
    """Write the figure to PNG and SVG under cfg.plots_*_dir() / cfg.Q2_CELLS_PLOT_SUBDIR / rel.*"""
    png_dir = cfg.plots_png_dir() / cfg.Q2_CELLS_PLOT_SUBDIR
    svg_dir = cfg.plots_svg_dir() / cfg.Q2_CELLS_PLOT_SUBDIR
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


def ensure_time_min(df: pd.DataFrame) -> pd.DataFrame:
    if "time_min" in df.columns:
        return df
    if "frame" in df.columns:
        df = df.copy()
        df["time_min"] = df["frame"].astype(float) * (cfg.DT_SECONDS / 60.0)
        return df
    raise KeyError("No time_min/frame column to build a time axis from.")


def build_pre_axis(tt: np.ndarray) -> np.ndarray:
    """Shift pre-phase times so end = 0; negative values = minutes before remount."""
    tt = np.asarray(tt, float)
    finite = tt[np.isfinite(tt)]
    if finite.size == 0:
        return tt
    return tt - np.nanmax(finite)


def build_post_axis(tt: np.ndarray) -> np.ndarray:
    """Shift post-phase times so start = 0."""
    tt = np.asarray(tt, float)
    finite = tt[np.isfinite(tt)]
    if finite.size == 0:
        return tt
    return tt - np.nanmin(finite)


def draw_gap(ax):
    ax.axvspan(0, GAP_W, color="0.92", zorder=0)
    ax.axvline(0,     color="0.75", lw=1.2)
    ax.axvline(GAP_W, color="0.75", lw=1.2)
    ax.set_xlim(-PRE_SHOW_MIN, GAP_W + POST_SHOW_MIN)


def xlabel_gap_axis() -> str:
    """Dynamic x-axis label that mirrors PRE/POST_SHOW_MIN."""
    return f"Time (min)  pre(-{int(PRE_SHOW_MIN)}..0)  gap  post(0..{int(POST_SHOW_MIN)})"


def nanmean_sem(A: np.ndarray):
    m = np.nanmean(A, axis=0)
    n = np.sum(np.isfinite(A), axis=0)
    s = np.nanstd(A, axis=0, ddof=1)
    sem = np.where(n > 0, s / np.sqrt(np.maximum(n, 1)), np.nan)
    return m, sem


# =============================================================================
# Statistics: proper Holm, omnibus dispatch, p-value annotation
# =============================================================================

def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction with monotonicity enforcement.

    Standard procedure:
      1. Sort p-values ascending.
      2. Multiply the k-th smallest by (m - k + 1).
      3. Enforce monotonicity (each value ≥ the previous one in sorted order).
      4. Cap at 1.
      5. Return in original order.

    The v6 implementation was missing step 3, which is the actual Holm
    correction (without it the result is closer to a step-up Hochberg variant
    and gives anti-conservative results on tied or near-tied p-values).
    """
    pvals = np.asarray(pvals, float)
    m = pvals.size
    if m == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj = sorted_p * (m - np.arange(m))           # step-down weights: m, m-1, ..., 1
    adj = np.maximum.accumulate(adj)              # enforce monotonicity (cumulative max)
    adj = np.minimum(adj, 1.0)                    # cap at 1
    out = np.empty_like(pvals)
    out[order] = adj
    return out


def _fmt_p(p: float, sig: int = 3) -> str:
    """Format the operator+value of a p-value annotation as plain decimal
    digits (never scientific notation); callers prepend the literal 'p'.

    Returns e.g. '=0.532' or '=0.0000137'. When p is below the display floor
    cfg.STATS['pval_min_display'], returns '<floor' (e.g. '<0.0001') instead of
    a long run of leading zeros. `sig` = significant figures for the exact case."""
    if p is None or not np.isfinite(p):
        return "=n/a"
    p = float(p)
    floor = cfg.STATS.get("pval_min_display", 0)
    if floor and 0.0 <= p < floor:
        return "<" + np.format_float_positional(float(floor), unique=True,
                                                fractional=True, trim="-")
    return "=" + np.format_float_positional(p, precision=sig, unique=False,
                                            fractional=False, trim="-")


def _omnibus_pvalue(groups: list[np.ndarray]) -> tuple[Optional[float], str]:
    """Choose and run the omnibus test according to cfg.STATS["omnibus_test"].

    Returns (p-value-or-None, label-shown-in-corner). Label e.g. "ANOVA p=…",
    "Welch t p=…", or "" if test should be hidden / not applicable.
    """
    mode = cfg.STATS["omnibus_test"]
    if mode == "none" or not cfg.STATS.get("show_omnibus", True):
        return None, ""

    valid = [g for g in groups if g is not None and len(g) >= 2]
    if len(valid) < 2:
        return None, ""

    decimals = cfg.STATS.get("pval_decimals", 3)

    # Choose test
    if mode == "auto":
        chosen = "ttest" if len(valid) == 2 else "anova"
    elif mode in ("anova", "ttest"):
        chosen = mode
    else:
        raise ValueError(f"Unknown cfg.STATS['omnibus_test'] = {mode!r}")

    if chosen == "ttest":
        if len(valid) != 2:
            # Forced t-test but >2 groups: skip omnibus (pairwise will still run)
            return None, ""
        try:
            p = ttest_ind(valid[0], valid[1], equal_var=False, nan_policy="omit").pvalue
        except Exception:
            return None, ""
        return float(p), f"Welch t p{_fmt_p(p, decimals)}"

    # ANOVA path
    try:
        p = f_oneway(*valid).pvalue
    except Exception:
        return None, ""
    return float(p), f"ANOVA p{_fmt_p(p, decimals)}"


# Remembers (planned_pairs, conditions) combos already warned about, so the
# "planned_pairs not present -> falling back" message prints once, not per panel.
_PLANNED_FALLBACK_WARNED: set = set()


def add_pvals(ax, data_list: list[np.ndarray], x_positions: Sequence[float],
              labels: Optional[Sequence[str]] = None):
    """Draw omnibus + pairwise p-values inside the plot area (compact).

    Bracket placement:
      Starts just above the highest data point across all groups (not at a
      fixed axis fraction). The axis ylim is extended upward to make room
      for the brackets without overlapping the boxes.

    Planned-comparisons mode:
      When ``cfg.STATS["planned_pairs"]`` is non-empty and ``labels`` (the
      condition token per box, aligned with ``data_list``) is provided, only
      those pairs are drawn, with RAW Welch-t p-values (no Holm) and no
      omnibus label. This is for a few pre-planned comparisons rather than the
      all-pairwise fishing expedition. Default (empty) keeps the old behaviour.
    """
    planned = cfg.STATS.get("planned_pairs") or []
    planned_set = {frozenset((str(a), str(b))) for a, b in planned}
    use_planned = bool(planned_set) and labels is not None
    if use_planned:
        _label_set = {str(x) for x in labels}
        if not any(p.issubset(_label_set) for p in planned_set):
            # None of the planned pairs exist in this dataset's conditions
            # (e.g. Yoda-panel planned_pairs left set on the E3/ISO/BDM data).
            # Fall back to the default omnibus + all-pairwise instead of
            # silently drawing nothing. Warn once per (planned, conditions).
            _key = (tuple(sorted(planned)), tuple(sorted(_label_set)))
            if _key not in _PLANNED_FALLBACK_WARNED:
                _PLANNED_FALLBACK_WARNED.add(_key)
                print(f"  [WARN] planned_pairs {planned} not present in "
                      f"conditions {sorted(_label_set)}; showing omnibus + "
                      f"all-pairwise instead.")
            use_planned = False

    # Omnibus in top-left corner (placed AFTER ylim extension below, so we
    # don't compute its y position yet). Suppressed in planned mode.
    omnibus_p, omnibus_label = _omnibus_pvalue(data_list)
    if use_planned:
        omnibus_label = ""

    def _lbl(idx):
        return str(labels[idx]) if labels is not None else str(idx)

    # Collected for the stats workbook (every p-value the caller may want to log).
    result = {
        "test": "welch",
        "correction": "none" if use_planned else "holm",
        "omnibus_p": None if use_planned else omnibus_p,
        "pairwise": [],   # list of (group1, group2, p_shown, significant)
    }

    if not cfg.STATS.get("show_pairwise", True):
        if omnibus_label:
            ax.text(0.02, 0.98, omnibus_label,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=cfg.STATS.get("pval_fontsize", cfg.FONT["legend"]))
        return result

    # Pairwise Welch. In planned mode, keep only the requested pairs.
    pairs, pvals = [], []
    for i in range(len(data_list)):
        for j in range(i + 1, len(data_list)):
            if use_planned and frozenset(
                (str(labels[i]), str(labels[j]))) not in planned_set:
                continue
            a, b = data_list[i], data_list[j]
            if a is None or b is None or len(a) < 2 or len(b) < 2:
                continue
            p = ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue
            pairs.append((i, j))
            pvals.append(p)

    if not pvals:
        # No pairwise to draw; just place the omnibus and return
        if omnibus_label:
            ax.text(0.02, 0.98, omnibus_label,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=cfg.STATS.get("pval_fontsize", cfg.FONT["legend"]))
        return result

    # Raw p in planned mode; Holm-corrected otherwise.
    p_corr = np.asarray(pvals, float) if use_planned else holm(np.asarray(pvals, float))

    # Find the data ceiling: highest finite value across all groups
    all_vals = []
    for g in data_list:
        if g is None or len(g) == 0: continue
        finite = g[np.isfinite(g)]
        if finite.size: all_vals.append(finite.max())
    if not all_vals:
        return result
    data_max = max(all_vals)

    # Current axis range
    y0_cur, y1_cur = ax.get_ylim()
    yr_cur = y1_cur - y0_cur

    # Bracket geometry: starts a bit above the data, each new bracket steps up
    step = 0.06 * yr_cur
    h    = 0.018 * yr_cur
    gap  = 0.04 * yr_cur            # gap between top of data and first bracket

    y_first = data_max + gap
    y_top   = y_first + step * len(pairs) + h

    # Extend ylim if needed so brackets + omnibus label fit comfortably
    needed_top = y_top + 0.04 * yr_cur
    if needed_top > y1_cur:
        ax.set_ylim(y0_cur, needed_top)

    decimals   = cfg.STATS.get("pval_decimals", 3)
    bracket_lw = cfg.STATS.get("bracket_lw", 1.2)
    pval_fs    = cfg.STATS.get("pval_fontsize", cfg.FONT["legend"])

    y = y_first
    for (i, j), p in zip(pairs, p_corr):
        x1, x2 = x_positions[i], x_positions[j]
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=bracket_lw)
        ax.text((x1 + x2) / 2, y + h, f"p{_fmt_p(p, decimals)}",
                ha="center", va="bottom", fontsize=pval_fs)
        result["pairwise"].append((_lbl(i), _lbl(j), float(p), bool(p < 0.05)))
        y += step

    # Omnibus label LAST so its y-coordinate uses the extended ylim
    if omnibus_label:
        ax.text(0.02, 0.98, omnibus_label,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=cfg.STATS.get("pval_fontsize", cfg.FONT["legend"]))

    return result


# =============================================================================
# Two-way (genotype x drug) statistics + shared box drawing
# =============================================================================

def resolve_genotypes(present: set[str]) -> list[str]:
    """Genotypes to plot, in cfg.GENOTYPE_ORDER then any extras alphabetically."""
    base = [g for g in getattr(cfg, "GENOTYPE_ORDER", []) if g in present]
    return base + sorted(present - set(base))


def genotype_box_color(g: str) -> str:
    """Box fill per genotype (distinct from drug colours, which encode condition).
    Palette lives in calcium_config.GENOTYPE_COLORS (single source of truth)."""
    return cfg.genotype_color(g)


def twoway_anova_label(stat_df: pd.DataFrame, value: str = "value", decimals: int = 3):
    """Type-II two-way ANOVA  value ~ C(genotype)*C(condition)  with interaction.

    stat_df is one row per embryo with columns value/genotype/condition.
    Returns (label, stats, ok). `label` is a SHORT corner label for the figure
    ("2-way ANOVA (II)" when estimable, else a short reason). `stats` is a dict
    of the numbers (geno_p/geno_eta2p/drug_p/drug_eta2p/gxd_p/gxd_eta2p) when
    estimable, else None — the full numbers go to the stats workbook, not the
    figure (keeps the corner from colliding with the boxes/brackets). Partial
    eta^2 is reported (more honest than p at small n). `decimals` is kept for
    signature compatibility with callers.
    """
    sub = stat_df[[value, "genotype", "condition"]].copy()
    sub[value] = pd.to_numeric(sub[value], errors="coerce")
    sub = sub.dropna()
    ng, nc = sub["genotype"].nunique(), sub["condition"].nunique()
    if ng < 2 or nc < 2:
        return ("", None, False)
    n_cells = ng * nc
    if sub.groupby(["genotype", "condition"]).ngroups < n_cells:
        return ("2-way ANOVA: empty cell(s) — not estimable", None, False)
    if len(sub) - n_cells < 1:
        return ("2-way ANOVA: n too small (no residual df)", None, False)
    if not _HAVE_SM:
        return ("(statsmodels unavailable)", None, False)
    try:
        model = ols("yval ~ C(genotype)*C(condition)",
                    data=sub.rename(columns={value: "yval"})).fit()
        aov = sm.stats.anova_lm(model, typ=2)
    except Exception as e:
        return (f"2-way ANOVA failed: {type(e).__name__}", None, False)
    ssr = float(aov.loc["Residual", "sum_sq"])
    def eta2p(term):
        ss = float(aov.loc[term, "sum_sq"])
        return ss / (ss + ssr) if (ss + ssr) > 0 else float("nan")
    stats = dict(
        geno_p=float(aov.loc["C(genotype)", "PR(>F)"]),
        geno_eta2p=eta2p("C(genotype)"),
        drug_p=float(aov.loc["C(condition)", "PR(>F)"]),
        drug_eta2p=eta2p("C(condition)"),
        gxd_p=float(aov.loc["C(genotype):C(condition)", "PR(>F)"]),
        gxd_eta2p=eta2p("C(genotype):C(condition)"),
    )
    # Figure shows only the short header; the numbers live in the workbook.
    return ("2-way ANOVA (II)", stats, True)


def tukey_significant_pairs(stat_df: pd.DataFrame, value: str, label_col: str):
    """Tukey HSD across genotype x drug cells (cells with >=2 obs).

    Returns list of (labelA, labelB, p_adj, reject)."""
    if not _HAVE_SM:
        return []
    sub = stat_df[[value, label_col]].copy()
    sub[value] = pd.to_numeric(sub[value], errors="coerce")
    sub = sub.dropna()
    counts = sub[label_col].value_counts()
    sub = sub[sub[label_col].isin(counts[counts >= 2].index)]
    if sub[label_col].nunique() < 2:
        return []
    res = pairwise_tukeyhsd(sub[value].to_numpy(float),
                            sub[label_col].astype(str).to_numpy())
    # Read documented attributes rather than parsing summary().data by column
    # position (which is fragile across statsmodels versions). Pair order
    # follows itertools.combinations over the sorted unique groups — the same
    # enumeration MultiComparison uses, so it aligns with pvalues/reject.
    groups = [str(g) for g in res.groupsunique]
    return [
        (groups[i], groups[j], float(res.pvalues[k]), bool(res.reject[k]))
        for k, (i, j) in enumerate(itertools.combinations(range(len(groups)), 2))
    ]


def draw_sig_brackets(ax, sig_pairs, xpos_by_label, data_by_label, decimals=3):
    """Draw brackets above the data for significant Tukey pairs only."""
    pairs = [(a, b, p) for (a, b, p, rej) in sig_pairs
             if rej and a in xpos_by_label and b in xpos_by_label]
    if not pairs:
        return
    allv = [v for ys in data_by_label.values() for v in ys if np.isfinite(v)]
    if not allv:
        return
    data_max = max(allv)
    y0, y1 = ax.get_ylim(); yr = y1 - y0
    step = 0.07 * yr; h = 0.018 * yr; gap = 0.04 * yr
    # draw shorter-span brackets first so they nest underneath
    pairs.sort(key=lambda t: abs(xpos_by_label[t[0]] - xpos_by_label[t[1]]))
    y = data_max + gap
    needed = y + step * len(pairs) + h + 0.04 * yr
    if needed > y1:
        ax.set_ylim(y0, needed)
    lw = cfg.STATS.get("bracket_lw", 1.2)
    fs = cfg.STATS.get("pval_fontsize", cfg.FONT["legend"])
    for (a, b, p) in pairs:
        x1, x2 = xpos_by_label[a], xpos_by_label[b]
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=lw)
        ax.text((x1 + x2) / 2, y + h, f"p{_fmt_p(p, decimals)}",
                ha="center", va="bottom", fontsize=fs)
        y += step


def draw_brackets_for_pairs(ax, wanted, tukey_all, xpos_by_label, data_by_label,
                            decimals=3):
    """Draw brackets for a chosen set of label pairs, significant or not.

    `wanted`    : list of (labelA, labelB) to annotate (order-insensitive).
    `tukey_all` : full list of (a, b, p_adj, reject) from tukey_significant_pairs
                  (which returns *every* pair); p is looked up from it.
    Pairs whose p is not found are skipped."""
    pmap = {}
    for a, b, p, _rej in tukey_all:
        pmap[frozenset((a, b))] = float(p)
    pairs = []
    for a, b in wanted:
        if a in xpos_by_label and b in xpos_by_label:
            p = pmap.get(frozenset((a, b)))
            if p is not None:
                pairs.append((a, b, p))
    if not pairs:
        return
    allv = [v for ys in data_by_label.values() for v in ys if np.isfinite(v)]
    if not allv:
        return
    data_max = max(allv)
    y0, y1 = ax.get_ylim(); yr = y1 - y0
    step = 0.09 * yr; h = 0.018 * yr; gap = 0.04 * yr
    # draw shorter-span brackets first so wider ones stack above them
    pairs.sort(key=lambda t: abs(xpos_by_label[t[0]] - xpos_by_label[t[1]]))
    y = data_max + gap
    needed = y + step * len(pairs) + h + 0.04 * yr
    if needed > y1:
        ax.set_ylim(y0, needed)
    lw = cfg.STATS.get("bracket_lw", 1.2)
    fs = cfg.STATS.get("pval_fontsize", cfg.FONT["legend"])
    for (a, b, p) in pairs:
        x1, x2 = xpos_by_label[a], xpos_by_label[b]
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=lw)
        ax.text((x1 + x2) / 2, y + h, f"p{_fmt_p(p, decimals)}",
                ha="center", va="bottom", fontsize=fs)
        y += step


def apply_dense_yticks(ax, nbins=None):
    """Increase the number of y-axis major ticks (smaller steps) while still
    snapping to 'nice' values. Leaves the data limits unchanged. Pass an
    explicit `nbins` to override cfg.BOX_YTICK_NBINS (e.g. for wider-range
    figures that need more ticks to reach the same step size)."""
    n = int(nbins if nbins is not None else getattr(cfg, "BOX_YTICK_NBINS", 8))
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]))


def draw_prism_box(ax, data_list, xpos, fill_colors):
    """Shared Prism-style box drawing: 10-90 whisker boxes + jittered embryo dots.

    Matches the original single-genotype rendering exactly (same widths, whiskers,
    alpha, jitter seed) so existing figures are unchanged."""
    bp = ax.boxplot(data_list, positions=xpos, widths=STYLE["box"]["width"],
                    showfliers=False, whis=(10, 90), patch_artist=True)
    for patch, c in zip(bp["boxes"], fill_colors):
        patch.set_facecolor(c)
        patch.set_alpha(STYLE["box"]["box_alpha"])
        patch.set_edgecolor("black")
        patch.set_linewidth(STYLE["box"]["box_edge_lw"])
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(STYLE["box"]["median_lw"])
    rng = np.random.default_rng(0)
    for x, y, c in zip(xpos, data_list, fill_colors):
        if len(y) == 0:
            continue
        jitter = (rng.random(len(y)) - 0.5) * 0.18
        ax.scatter(x + jitter, y, s=STYLE["box"]["dot_size"],
                   facecolor=c, edgecolor="black", linewidth=0.5, zorder=3,
                   alpha=STYLE["box"].get("dot_alpha", 1.0))
    return bp


# =============================================================================
# Fold-change plot: drug / vehicle-mean, per planned (vehicle, drug) pair
# =============================================================================

def _annotate_foldchange(ax, fold_data, xpos, labels, vs_veh_p,
                         gvy_rows, pid, cell_class, metric_key):
    """Annotate the fold-change box: the two-group Welch p (drug vs its own
    vehicle, computed on RAW values — the valid test) above each box, plus a
    G-vs-Y Welch p bracket on the folds if >=2 drugs. Appends the G-vs-Y row
    to gvy_rows for the workbook."""
    dec = cfg.STATS.get("pval_decimals", 3)
    fs = cfg.STATS.get("pval_fontsize", cfg.FONT["legend"])
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    tops = []
    for k, fold in enumerate(fold_data):
        fmax = float(np.nanmax(fold)) if len(fold) else y1
        yt = fmax + 0.03 * yr
        ax.text(xpos[k], yt, f"p{_fmt_p(vs_veh_p[k], dec)}",
                ha="center", va="bottom", fontsize=fs)
        tops.append(yt)
    if len(fold_data) >= 2:
        try:
            p_gvy = float(ttest_ind(fold_data[0], fold_data[1],
                                    equal_var=False, nan_policy="omit").pvalue)
        except Exception:
            p_gvy = float("nan")
        yb = max(tops) + 0.09 * yr
        h = 0.02 * yr
        x1, x2 = xpos[0], xpos[1]
        ax.plot([x1, x1, x2, x2], [yb, yb + h, yb + h, yb],
                color="black", lw=cfg.STATS.get("bracket_lw", 1.0))
        ax.text((x1 + x2) / 2, yb + h, f"p{_fmt_p(p_gvy, dec)}",
                ha="center", va="bottom", fontsize=fs)
        need = yb + h + 0.05 * yr
        if need > y1:
            ax.set_ylim(y0, need)
        gvy_rows.append(dict(
            pair_id=pid, cell_class=cell_class, metric=metric_key,
            test="foldchange_G_vs_Y", group1=labels[0], group2=labels[1],
            p_value=p_gvy, significant=bool(np.isfinite(p_gvy) and p_gvy < 0.05)))


def plot_foldchange_for_pid(dfp, pid, box_metrics, fc_pairs, use_log2, pub,
                            wsuf, cell_class_col):
    """Draw fold-change boxes for one single-genotype (drug-panel) pid.

    For each planned (vehicle, drug) pair, fold = each drug embryo divided by
    the MEAN of its vehicle (same cell_class); the boxes for the drugs are shown
    together on one 'fold change' axis (reference line at 1, or 0 for log2), so
    the vehicles collapse to the baseline. 'Drug vs vehicle' significance uses
    the two-group Welch t on the raw values (dividing by the vehicle mean throws
    away the control's spread, so a one-sample-vs-1 test would be
    anti-conservative). Returns (fc_rows, gvy_rows) for the workbook.
    """
    fc_rows, gvy_rows = [], []
    ref = 0.0 if use_log2 else 1.0
    ylab = "log2 fold change" if use_log2 else r"fold change ($\div$ vehicle mean)"

    for metric in box_metrics:
        col = metric["col"]
        panels = {}   # cell_class -> (fig, ax)
        for cell_class in ["flat", "round"]:
            fold_data, labels, colors, vs_veh_p = [], [], [], []
            for veh, drug in fc_pairs:
                dv = pd.to_numeric(dfp[(dfp["condition"].astype(str) == str(drug)) &
                    (dfp[cell_class_col].astype(str) == cell_class)][col],
                    errors="coerce").to_numpy(float)
                vv = pd.to_numeric(dfp[(dfp["condition"].astype(str) == str(veh)) &
                    (dfp[cell_class_col].astype(str) == cell_class)][col],
                    errors="coerce").to_numpy(float)
                dv = dv[np.isfinite(dv)]
                vv = vv[np.isfinite(vv)]
                if len(dv) == 0 or len(vv) == 0:
                    continue
                vmean = float(np.mean(vv))
                if not np.isfinite(vmean) or vmean == 0:
                    continue
                fold = dv / vmean
                if use_log2:
                    fold = np.log2(fold[fold > 0])
                if len(fold) == 0:
                    continue
                try:
                    p_vs = float(ttest_ind(dv, vv, equal_var=False,
                                           nan_policy="omit").pvalue)
                except Exception:
                    p_vs = float("nan")
                fold_data.append(fold)
                labels.append(cfg.pub_label(drug))
                colors.append(cfg.color_for(drug))
                vs_veh_p.append(p_vs)
                fc_rows.append(dict(
                    pair_id=pid, cell_class=cell_class, metric=metric["key"],
                    drug=str(drug), vehicle=str(veh), n_drug_embryos=int(len(dv)),
                    vehicle_mean=vmean, fold_mean=float(np.mean(fold)),
                    fold_median=float(np.median(fold)),
                    fold_sd=(float(np.std(fold, ddof=1)) if len(fold) > 1
                             else float("nan")),
                    p_vs_vehicle_welch=p_vs, log2=bool(use_log2)))
            if not fold_data:
                continue
            fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["box"])
            xpos = np.linspace(0.9, 0.9 + 0.5 * (len(fold_data) - 1), len(fold_data))
            draw_prism_box(ax, fold_data, xpos, colors)
            ax.axhline(ref, color="0.4", ls="--", lw=0.8, zorder=0)
            ax.set_xticks(xpos)
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylab)
            ax.set_title(
                (f"{metric['title_pub']} fold change\n{cfg.cell_class_display(cell_class)}" if pub else
                 f"Q2 {metric['title_metric']} fold change | {pid} | {cell_class}\n"
                 f"(drug / vehicle mean; embryo unit)"),
                pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
            style_axes(ax)
            apply_dense_yticks(ax)
            _annotate_foldchange(ax, fold_data, xpos, labels, vs_veh_p,
                                 gvy_rows, pid, cell_class, metric["key"])
            panels[cell_class] = (fig, ax)
        # share the y-axis across flat/round, then save
        if getattr(cfg, "SHARE_CELLCLASS_YLIM", True) and len(panels) > 1:
            los, his = zip(*(a.get_ylim() for _, a in panels.values()))
            for _, a in panels.values():
                a.set_ylim(min(los), max(his))
        for cc, (fig, ax) in panels.items():
            save_both(fig, Path("boxplot_foldchange") / cc / str(pid)
                      / f"box_foldchange_{metric['key']}_{cc}_post{wsuf}min")
    return fc_rows, gvy_rows


def _annotate_foldchange_genotype(ax, groups, cmp_rows, pid, cell_class, metric_key):
    """Per drug group: vs-vehicle Welch p (within genotype) above each box, and a
    between-genotype Welch p bracket on the folds (the 'does the drug response
    differ by genotype' question). Appends between-genotype rows to cmp_rows."""
    dec = cfg.STATS.get("pval_decimals", 3)
    fs = cfg.STATS.get("pval_fontsize", cfg.FONT["legend"])
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    top_used = y1
    for grp in groups:
        boxes = grp["boxes"]   # list of (geno, xpos, fold, p_vs_vehicle)
        tops = []
        for geno, xp, fold, p_vs in boxes:
            fmax = float(np.nanmax(fold)) if len(fold) else y1
            yt = fmax + 0.03 * yr
            ax.text(xp, yt, f"p{_fmt_p(p_vs, dec)}", ha="center", va="bottom",
                    fontsize=fs)
            tops.append(yt)
        if len(boxes) >= 2:
            (gA, xA, fA, _), (gB, xB, fB, _) = boxes[0], boxes[1]
            try:
                p_bt = float(ttest_ind(fA, fB, equal_var=False,
                                       nan_policy="omit").pvalue)
            except Exception:
                p_bt = float("nan")
            yb = max(tops) + 0.09 * yr
            h = 0.02 * yr
            ax.plot([xA, xA, xB, xB], [yb, yb + h, yb + h, yb],
                    color="black", lw=cfg.STATS.get("bracket_lw", 1.0))
            ax.text((xA + xB) / 2, yb + h, f"p{_fmt_p(p_bt, dec)}",
                    ha="center", va="bottom", fontsize=fs)
            top_used = max(top_used, yb + h + 0.05 * yr)
            cmp_rows.append(dict(
                pair_id=pid, cell_class=cell_class, metric=metric_key,
                test="foldchange_between_geno",
                group1=f"{gA}|{grp['drug']}", group2=f"{gB}|{grp['drug']}",
                p_value=p_bt,
                significant=bool(np.isfinite(p_bt) and p_bt < 0.05)))
        elif tops:
            top_used = max(top_used, max(tops) + 0.05 * yr)
    if top_used > y1:
        ax.set_ylim(y0, top_used)


def plot_foldchange_genotype_for_pid(dfp, pid, box_metrics, fc_pairs, use_log2,
                                     pub, wsuf, cell_class_col):
    """Genotype x drug fold change: each embryo divided by the mean of ITS OWN
    genotype's vehicle (so MIC-ISO / mean(MIC-E3) and Piezo-ISO / mean(Piezo-E3)
    separately). One box per (genotype, drug), coloured by genotype; the
    between-genotype Welch on the folds asks whether the drug response differs by
    genotype (the key interaction question, read straight off the plot). Returns
    (fc_rows, cmp_rows)."""
    fc_rows, cmp_rows = [], []
    ref = 0.0 if use_log2 else 1.0
    ylab = ("log2 fold change" if use_log2
            else r"fold change ($\div$ genotype vehicle mean)")
    genos = resolve_genotypes(set(dfp["genotype"].astype(str).unique()))

    for metric in box_metrics:
        col = metric["col"]

        def _vals(geno, cond, cell_class, _col=col):
            s = dfp[(dfp["genotype"].astype(str) == str(geno)) &
                    (dfp["condition"].astype(str) == str(cond)) &
                    (dfp[cell_class_col].astype(str) == cell_class)][_col]
            a = pd.to_numeric(s, errors="coerce").to_numpy(float)
            return a[np.isfinite(a)]

        panels = {}
        for cell_class in ["flat", "round"]:
            data, fill_colors, xpos = [], [], []
            box_geno, box_drug, groups = [], [], []
            x, within, group_gap = 0.6, 0.42, 0.5
            for veh, drug in fc_pairs:
                if not ((dfp["condition"].astype(str) == str(drug)).any() and
                        (dfp["condition"].astype(str) == str(veh)).any()):
                    continue
                cond_xs, grp_boxes = [], []
                for geno in genos:
                    vv = _vals(geno, veh, cell_class)
                    dv = _vals(geno, drug, cell_class)
                    if len(vv) == 0 or len(dv) == 0:
                        continue
                    vmean = float(np.mean(vv))
                    if not np.isfinite(vmean) or vmean == 0:
                        continue
                    fold = dv / vmean
                    if use_log2:
                        fold = np.log2(fold[fold > 0])
                    if len(fold) == 0:
                        continue
                    try:
                        p_vs = float(ttest_ind(dv, vv, equal_var=False,
                                               nan_policy="omit").pvalue)
                    except Exception:
                        p_vs = float("nan")
                    data.append(fold)
                    fill_colors.append(genotype_box_color(geno))
                    xpos.append(x)
                    box_geno.append(str(geno))
                    box_drug.append(str(drug))
                    cond_xs.append(x)
                    grp_boxes.append((str(geno), x, fold, p_vs))
                    fc_rows.append(dict(
                        pair_id=pid, cell_class=cell_class, metric=metric["key"],
                        genotype=str(geno), drug=str(drug), vehicle=str(veh),
                        n_drug_embryos=int(len(dv)), vehicle_mean=vmean,
                        fold_mean=float(np.mean(fold)),
                        fold_median=float(np.median(fold)),
                        fold_sd=(float(np.std(fold, ddof=1)) if len(fold) > 1
                                 else float("nan")),
                        p_vs_vehicle_welch=p_vs, log2=bool(use_log2)))
                    x += within
                if cond_xs:
                    groups.append(dict(drug=str(drug), boxes=grp_boxes))
                x += group_gap
            if not data:
                continue
            fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["box"])
            draw_prism_box(ax, data, xpos, fill_colors)
            ax.axhline(ref, color="0.4", ls="--", lw=0.8, zorder=0)
            # Label each box by its genotype directly (no legend). One drug ->
            # genotype alone is unambiguous; multiple drugs -> two-line
            # "geno\ndrug" ticks so it stays readable without a legend.
            drugs_here = list(dict.fromkeys(box_drug))
            single_drug = len(drugs_here) == 1
            ax.set_xticks(xpos)
            ax.set_xticklabels(box_geno if single_drug
                               else [f"{g}\n{cfg.pub_label(d)}" for g, d in zip(box_geno, box_drug)])
            ax.set_ylabel(ylab)
            _dsuffix = f" ({drugs_here[0]})" if single_drug else ""
            ax.set_title(
                (f"{metric['title_pub']} fold change{_dsuffix}\n{cfg.cell_class_display(cell_class)}" if pub else
                 f"Q2 {metric['title_metric']} fold change{_dsuffix} | {pid} | {cell_class}\n"
                 f"(drug / genotype vehicle mean; embryo unit)"),
                pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"])
            style_axes(ax)
            apply_dense_yticks(ax)
            _annotate_foldchange_genotype(ax, groups, cmp_rows, pid, cell_class,
                                          metric["key"])
            panels[cell_class] = (fig, ax)
        if getattr(cfg, "SHARE_CELLCLASS_YLIM", True) and len(panels) > 1:
            los, his = zip(*(a.get_ylim() for _, a in panels.values()))
            for _, a in panels.values():
                a.set_ylim(min(los), max(his))
        for cc, (fig, ax) in panels.items():
            save_both(fig, Path("boxplot_foldchange") / cc / str(pid)
                      / f"box_foldchange_{metric['key']}_{cc}_genoXdrug_post{wsuf}min")
    return fc_rows, cmp_rows


# =============================================================================
# L3 trace plotting (overlay helper for cells/embryos)
# =============================================================================

def plot_overlay_with_pooled(
    *, cells_df,
    ax, level, TIME_COL, TRACE_COL, PHASE_COL, EMB_COL, ROI_COL,
    color, PRE_SHOW_MIN, POST_SHOW_MIN, GAP_W,
    draw_pre=True, max_cell=9999,
):
    """Draw pooled mean+SEM (thick, on top) over thin individual traces.

    level:
      'cell'   — pooled + every cell ROI as a thin trace
      'embryo' — pooled + per-embryo mean trace
    """

    def pooled_curve(df_phase, phase):
        if df_phase.empty:
            return None
        if phase == "pre":
            grid = np.linspace(-PRE_SHOW_MIN, 0, 201)
            blocks = []
            for emb_id, ge in df_phase.groupby(EMB_COL):
                ge = ge.copy()
                ge["t"] = build_pre_axis(ge[TIME_COL].to_numpy(float))
                ge = ge[(ge["t"] >= -PRE_SHOW_MIN) & (ge["t"] <= 0)]
                if ge.empty:
                    continue
                y = ge.groupby("t")[TRACE_COL].mean()
                t = y.index.to_numpy(float)
                yy = pd.to_numeric(y.values, errors="coerce").astype(float)
                if len(t) >= 2:
                    blocks.append(np.interp(grid, t, yy, left=np.nan, right=np.nan))
            if not blocks:
                return None
            A = np.vstack(blocks)
            m, s = nanmean_sem(A)
            return grid, m, s

        grid = np.linspace(0, POST_SHOW_MIN, 201)
        blocks = []
        for emb_id, ge in df_phase.groupby(EMB_COL):
            ge = ge.copy()
            ge["t"] = build_post_axis(ge[TIME_COL].to_numpy(float))
            ge = ge[(ge["t"] >= 0) & (ge["t"] <= POST_SHOW_MIN)]
            if ge.empty:
                continue
            y = ge.groupby("t")[TRACE_COL].mean()
            t = y.index.to_numpy(float)
            yy = pd.to_numeric(y.values, errors="coerce").astype(float)
            if len(t) >= 2:
                blocks.append(np.interp(grid, t, yy, left=np.nan, right=np.nan))
        if not blocks:
            return None
        A = np.vstack(blocks)
        m, s = nanmean_sem(A)
        return grid, m, s

    # ---------- thin traces underneath ----------
    if level == "cell":
        keys = [(e, r) for (e, r), _ in cells_df.groupby([EMB_COL, ROI_COL])]
        if len(keys) > max_cell:
            keys = keys[:max_cell]
        cmap = plt.get_cmap("tab20")
        for k, (emb_id, roi) in enumerate(keys):
            g = cells_df[(cells_df[EMB_COL] == emb_id) & (cells_df[ROI_COL] == roi)]
            if draw_pre:
                pre = g[g[PHASE_COL] == "pre"]
                if not pre.empty:
                    t = build_pre_axis(pre[TIME_COL].to_numpy(float))
                    y = pd.to_numeric(pre[TRACE_COL], errors="coerce").to_numpy(float)
                    m = (t >= -PRE_SHOW_MIN) & (t <= 0)
                    t, y = t[m], y[m]
                    if len(t) >= 2:
                        o = np.argsort(t)
                        ax.plot(t[o], y[o], color=cmap(k % 20), alpha=0.25, lw=0.9, zorder=1)
            post = g[g[PHASE_COL] == "post"]
            if not post.empty:
                t = build_post_axis(post[TIME_COL].to_numpy(float))
                y = pd.to_numeric(post[TRACE_COL], errors="coerce").to_numpy(float)
                m = (t >= 0) & (t <= POST_SHOW_MIN)
                t, y = t[m], y[m]
                if len(t) >= 2:
                    o = np.argsort(t)
                    ax.plot(t[o] + GAP_W, y[o], color=cmap(k % 20), alpha=0.25, lw=0.9, zorder=1)

    elif level == "embryo":
        for emb_id, g in cells_df.groupby(EMB_COL):
            if draw_pre:
                pre = g[g[PHASE_COL] == "pre"].copy()
                if not pre.empty:
                    pre["t"] = build_pre_axis(pre[TIME_COL].to_numpy(float))
                    pre = pre[(pre["t"] >= -PRE_SHOW_MIN) & (pre["t"] <= 0)]
                    if not pre.empty:
                        y = pre.groupby("t")[TRACE_COL].mean()
                        t = y.index.to_numpy(float)
                        yy = pd.to_numeric(y.values, errors="coerce").astype(float)
                        if len(t) >= 2:
                            o = np.argsort(t)
                            ax.plot(t[o], yy[o], color=color, alpha=0.20, lw=1.0, zorder=1)
            post = g[g[PHASE_COL] == "post"].copy()
            if not post.empty:
                post["t"] = build_post_axis(post[TIME_COL].to_numpy(float))
                post = post[(post["t"] >= 0) & (post["t"] <= POST_SHOW_MIN)]
                if not post.empty:
                    y = post.groupby("t")[TRACE_COL].mean()
                    t = y.index.to_numpy(float)
                    yy = pd.to_numeric(y.values, errors="coerce").astype(float)
                    if len(t) >= 2:
                        o = np.argsort(t)
                        ax.plot(t[o] + GAP_W, yy[o], color=color, alpha=0.20, lw=1.0, zorder=1)
    else:
        raise ValueError("level must be 'cell' or 'embryo'")

    # ---------- pooled thick trace on top ----------
    if draw_pre:
        out = pooled_curve(cells_df[cells_df[PHASE_COL] == "pre"], "pre")
        if out is not None:
            x, m, s = out
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(x, m, color=color, lw=3.2, alpha=1.0, zorder=3)

    out = pooled_curve(cells_df[cells_df[PHASE_COL] == "post"], "post")
    if out is not None:
        x, m, s = out
        ax.fill_between(x + GAP_W, m - s, m + s, color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x + GAP_W, m, color=color, lw=3.2, alpha=1.0, zorder=3)

    draw_gap(ax)


# =============================================================================
# Conditions: order, colours, unknown-drug handling
# =============================================================================

def resolve_conditions(present_in_data: set[str]) -> list[str]:
    """Return the ordered list of conditions to plot.

    Begins with cfg.Q2_CELLS_CONDITIONS_ORDER (filtered to those actually
    present in the data), then optionally appends unknown drugs (those not in
    cfg.CONDITION_ALIASES) if cfg.INCLUDE_UNKNOWN_IN_PLOTS is True.
    """
    base = [c for c in cfg.Q2_CELLS_CONDITIONS_ORDER if c in present_in_data]
    if cfg.INCLUDE_UNKNOWN_IN_PLOTS:
        known = cfg.all_known_conditions()
        unknown = sorted(present_in_data - known - set(base))
        return base + unknown
    return base


# =============================================================================
# Main
# =============================================================================

def main():
    cells_path = cfg.tables_dir() / "Q2_cells_vDA_prepost.csv"
    emb_path   = cfg.tables_dir() / "Q2_embryo_summary_cells.csv"
    if not cells_path.exists():
        raise FileNotFoundError(f"Missing table: {cells_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing table: {emb_path}")

    cells = pd.read_csv(cells_path)
    emb   = pd.read_csv(emb_path)
    cells = ensure_time_min(cells)

    print(f"Input analysis window: {cfg.window_suffix()} min")
    print(f"Cells conditions in data: {sorted(cells['condition'].astype(str).unique())}")
    print(f"Embryo summary conditions: {sorted(emb['condition'].astype(str).unique())}")

    # Column resolution (robust to old/new naming)
    CELL_CLASS_COL = pick_first(cells, ["cell_class", "roi_class", "cell_type"], "cell class (cells)")
    TRACE_COL      = pick_first(cells, ["cell_trace_main", "cell_only_norm_LA_B2", "cell_trace"], "trace (cells)")
    PHASE_COL      = pick_first(cells, ["phase"], "phase")
    TIME_COL       = pick_first(cells, ["time_min"], "time_min")
    EMB_COL        = pick_first(cells, ["embryo_id"], "embryo_id")
    ROI_COL        = pick_first(cells, ["roi_name"], "roi_name")

    EMB_CELL_CLASS_COL = pick_first(emb, ["cell_class", "roi_class", "cell_type"], "cell class (emb summary)")

    # The events column name is window-dependent (events_per_..._<W>min_post).
    # Prefer the config-derived name; fall back to legacy names if reading an
    # older table.
    expected_events_col = cfg.events_col_per_embryo()
    EVENTS_COL = pick_first(
        emb,
        [
            expected_events_col,
            "mean_events_per_cell_per_10min_post",
            "events_per_cell_per_10min_post",
        ],
        f"events column ({expected_events_col})",
    )
    print(f"Using events column: {EVENTS_COL}")

    # Optional pair_id filter
    if PAIR_ID is not None:
        cells = cells[cells["pair_id"] == PAIR_ID].copy()
        emb   = emb  [emb  ["pair_id"] == PAIR_ID].copy()

    pair_ids = sorted(cells["pair_id"].astype(str).unique())

    # ---------- L1: per embryo, all individual cell traces ----------
    if ONLY in (None, "L1"):
        for pid in pair_ids:
            subp = cells[cells["pair_id"] == pid]
            for cell_class in ["flat", "round"]:
                sub = subp[subp[CELL_CLASS_COL] == cell_class]
                for (cond, emb_id), g in sub.groupby(["condition", EMB_COL]):
                    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["l1"])
                    cmap = plt.get_cmap("tab20")
                    rois = sorted(g[ROI_COL].astype(str).unique())

                    for k, roi in enumerate(rois):
                        gg = g[g[ROI_COL].astype(str) == roi]

                        pre = gg[gg[PHASE_COL] == "pre"]
                        if not pre.empty:
                            tpre = build_pre_axis(pre[TIME_COL].to_numpy(float))
                            ypre = pd.to_numeric(pre[TRACE_COL], errors="coerce").to_numpy(float)
                            m = (tpre >= -PRE_SHOW_MIN) & (tpre <= 0)
                            ax.plot(tpre[m], ypre[m],
                                    color=cmap(k % 20),
                                    alpha=STYLE["alpha"]["cell"],
                                    lw=STYLE["lw"]["cell"])

                        post = gg[gg[PHASE_COL] == "post"]
                        if not post.empty:
                            tpost = build_post_axis(post[TIME_COL].to_numpy(float))
                            ypost = pd.to_numeric(post[TRACE_COL], errors="coerce").to_numpy(float)
                            m = (tpost >= 0) & (tpost <= POST_SHOW_MIN)
                            ax.plot((tpost[m] + GAP_W), ypost[m],
                                    color=cmap(k % 20),
                                    alpha=STYLE["alpha"]["cell"],
                                    lw=STYLE["lw"]["cell"])

                    draw_gap(ax)
                    ax.set_title(
                        f"Q2 L1 | {pid} | {cfg.display_name(cond)} | {emb_id} | {cell_class}\n"
                        f"(all cells; pre+post gap-axis)",
                        pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
                    )
                    ax.set_xlabel(xlabel_gap_axis())
                    ax.set_ylabel(TRACE_COL)
                    style_axes(ax)
                    save_both(fig, Path("L1_cells") / cell_class / pid / cond / f"{emb_id}")

    # ---------- L2: per embryo mean trace across cells ----------
    if ONLY in (None, "L2"):
        for pid in pair_ids:
            subp = cells[cells["pair_id"] == pid]
            for cell_class in ["flat", "round"]:
                sub = subp[subp[CELL_CLASS_COL] == cell_class]
                for (cond, emb_id), g in sub.groupby(["condition", EMB_COL]):
                    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["l2"])
                    col = cfg.color_for(cond)

                    pre = g[g[PHASE_COL] == "pre"].copy()
                    if not pre.empty:
                        pre["tpre"] = build_pre_axis(pre[TIME_COL].to_numpy(float))
                        pre = pre[(pre["tpre"] >= -PRE_SHOW_MIN) & (pre["tpre"] <= 0)]
                        y = pre.groupby("tpre")[TRACE_COL].mean()
                        t  = y.index.to_numpy(float)
                        yy = pd.to_numeric(y.values, errors="coerce").astype(float)
                        ax.plot(t, yy, color=col,
                                alpha=STYLE["alpha"]["embryo"], lw=STYLE["lw"]["embryo"])

                    post = g[g[PHASE_COL] == "post"].copy()
                    if not post.empty:
                        post["tpost"] = build_post_axis(post[TIME_COL].to_numpy(float))
                        post = post[(post["tpost"] >= 0) & (post["tpost"] <= POST_SHOW_MIN)]
                        y = post.groupby("tpost")[TRACE_COL].mean()
                        t  = y.index.to_numpy(float)
                        yy = pd.to_numeric(y.values, errors="coerce").astype(float)
                        ax.plot(t + GAP_W, yy, color=col,
                                alpha=STYLE["alpha"]["embryo"], lw=STYLE["lw"]["embryo"])

                    draw_gap(ax)
                    ax.set_title(
                        f"Q2 L2 | {pid} | {cfg.display_name(cond)} | {emb_id} | {cell_class}\n"
                        f"(embryo mean; pre+post gap-axis)",
                        pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
                    )
                    ax.set_xlabel(xlabel_gap_axis())
                    ax.set_ylabel(TRACE_COL)
                    style_axes(ax)
                    save_both(fig, Path("L2_embryo") / cell_class / pid / cond / f"{emb_id}")

    # ---------- L3: per-batch repeat mean±SEM + pooled, with overlays ----------
    if ONLY in (None, "L3"):

        def _interp_embryo_mean_to_grid(df_one_phase: pd.DataFrame, grid: np.ndarray, phase: str):
            if df_one_phase.empty:
                return None
            if phase == "pre":
                t = build_pre_axis(df_one_phase[TIME_COL].to_numpy(float))
                msk = (t >= -PRE_SHOW_MIN) & (t <= 0)
            else:
                t = build_post_axis(df_one_phase[TIME_COL].to_numpy(float))
                msk = (t >= 0) & (t <= POST_SHOW_MIN)
            t = t[msk]
            y = pd.to_numeric(df_one_phase[TRACE_COL], errors="coerce").to_numpy(float)[msk]
            ok = np.isfinite(t) & np.isfinite(y)
            t, y = t[ok], y[ok]
            if t.size < 2:
                return None
            tmp = pd.DataFrame({"t": t, "y": y}).groupby("t")["y"].mean()
            tt = tmp.index.to_numpy(float)
            yy = tmp.values.astype(float)
            if tt.size < 2:
                return None
            order = np.argsort(tt)
            tt, yy = tt[order], yy[order]
            return np.interp(grid, tt, yy, left=np.nan, right=np.nan)

        def _collect_A_by_batch(df_phase: pd.DataFrame, grid: np.ndarray, phase: str):
            out = {}
            if df_phase.empty:
                return out
            for batch_id, gb in df_phase.groupby("batch_id"):
                rows = []
                for emb_id, ge in gb.groupby(EMB_COL):
                    v = _interp_embryo_mean_to_grid(ge, grid, phase)
                    if v is not None:
                        rows.append(v)
                if rows:
                    out[str(batch_id)] = np.vstack(rows)
            return out

        x_pre_grid  = np.linspace(-PRE_SHOW_MIN, 0, 201)
        x_post_grid = np.linspace(0, POST_SHOW_MIN, 201)

        for pid in pair_ids:
            subp = cells[cells["pair_id"].astype(str) == str(pid)]

            for cell_class in ["flat", "round"]:
                subc = subp[subp[CELL_CLASS_COL].astype(str) == str(cell_class)]

                present = set(subc["condition"].astype(str).unique())
                conds_here = resolve_conditions(present)
                # Split pooled traces by genotype so backgrounds are never
                # averaged together (one figure per genotype x drug).
                genos_here = (
                    resolve_genotypes(set(subc["genotype"].astype(str).unique()))
                    if "genotype" in subc.columns else [None]
                )

                for cond, geno in [(c, g) for c in conds_here for g in genos_here]:
                    d = subc[subc["condition"].astype(str) == str(cond)].copy()
                    if geno is not None:
                        d = d[d["genotype"].astype(str) == str(geno)]
                    if d.empty:
                        continue
                    gpfx = "" if geno is None else f"{geno} "   # title prefix
                    gtag = "" if geno is None else f"{geno}_"   # filename tag

                    col = cfg.color_for(cond)
                    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["l3"])

                    # PRE
                    pre = d[d[PHASE_COL].astype(str) == "pre"].copy()
                    pre_batches = _collect_A_by_batch(pre, x_pre_grid, "pre")
                    pooled_pre_blocks = []
                    for batch_id, A in pre_batches.items():
                        m, s = nanmean_sem(A)
                        ax.fill_between(x_pre_grid, m - s, m + s,
                                        color=col, alpha=STYLE["alpha"]["repeat_fill"], linewidth=0)
                        ax.plot(x_pre_grid, m,
                                color=col, lw=STYLE["lw"]["repeat"],
                                alpha=STYLE["alpha"]["repeat_line"],
                                label=f"{batch_id} pre (n_emb={A.shape[0]})")
                        pooled_pre_blocks.append(A)
                    if pooled_pre_blocks:
                        Aall = np.vstack(pooled_pre_blocks)
                        m, s = nanmean_sem(Aall)
                        ax.fill_between(x_pre_grid, m - s, m + s,
                                        color=col, alpha=STYLE["alpha"]["pooled_fill"], linewidth=0)
                        ax.plot(x_pre_grid, m,
                                color=col, lw=STYLE["lw"]["pooled"],
                                alpha=STYLE["alpha"]["pooled_line"],
                                label=f"pooled pre (n_emb={Aall.shape[0]})")

                    # POST
                    post = d[d[PHASE_COL].astype(str) == "post"].copy()
                    post_batches = _collect_A_by_batch(post, x_post_grid, "post")
                    pooled_post_blocks = []
                    for batch_id, A in post_batches.items():
                        m, s = nanmean_sem(A)
                        ax.fill_between(x_post_grid + GAP_W, m - s, m + s,
                                        color=col, alpha=STYLE["alpha"]["repeat_fill"], linewidth=0)
                        ax.plot(x_post_grid + GAP_W, m,
                                color=col, lw=STYLE["lw"]["repeat"],
                                alpha=STYLE["alpha"]["repeat_line"],
                                label=f"{batch_id} post (n_emb={A.shape[0]})")
                        pooled_post_blocks.append(A)
                    if pooled_post_blocks:
                        Aall = np.vstack(pooled_post_blocks)
                        m, s = nanmean_sem(Aall)
                        ax.fill_between(x_post_grid + GAP_W, m - s, m + s,
                                        color=col, alpha=STYLE["alpha"]["pooled_fill"], linewidth=0)
                        ax.plot(x_post_grid + GAP_W, m,
                                color=col, lw=STYLE["lw"]["pooled"],
                                alpha=STYLE["alpha"]["pooled_line"],
                                label=f"pooled post (n_emb={Aall.shape[0]})")

                    draw_gap(ax)
                    ax.set_title(
                        f"Q2 L3 | {pid} | {gpfx}{cfg.display_name(cond)} | {cell_class}\n"
                        f"repeat mean±SEM + pooled | pre+post gap-axis",
                        pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
                    )
                    ax.set_xlabel(xlabel_gap_axis())
                    ax.set_ylabel(f"{TRACE_COL} (embryo-mean of cells)")
                    style_axes(ax)
                    ax.legend(frameon=False, fontsize=cfg.FONT["legend"])
                    save_both(fig,
                              Path("L3_repeat_pooled") / cell_class / str(pid)
                              / f"L3_{gtag}{cond}_{cell_class}_gapaxis_0to{int(POST_SHOW_MIN)}min")

                    # ---- overlays ----
                    if L3_EXTRA_OVERLAYS:
                        # (A) pooled + every cell trace
                        fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["l3"])
                        plot_overlay_with_pooled(
                            cells_df=d, ax=ax, level="cell",
                            TIME_COL=TIME_COL, TRACE_COL=TRACE_COL, PHASE_COL=PHASE_COL,
                            EMB_COL=EMB_COL, ROI_COL=ROI_COL,
                            color=col, PRE_SHOW_MIN=PRE_SHOW_MIN, POST_SHOW_MIN=POST_SHOW_MIN,
                            GAP_W=GAP_W, draw_pre=L3_OVERLAY_DRAW_PRE, max_cell=L3_OVERLAY_MAX_CELL
                        )
                        ax.set_title(
                            f"Q2 L3 overlay (cells) | {pid} | {gpfx}{cfg.display_name(cond)} | {cell_class}",
                            pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
                        )
                        ax.set_xlabel(xlabel_gap_axis())
                        ax.set_ylabel(TRACE_COL)
                        style_axes(ax)
                        # NB: the v6 hard-coded ylim=(0,6) override for cond=="Yoda" has been removed.
                        save_both(fig,
                                  Path("L3_overlay_cells") / cell_class / str(pid)
                                  / f"overlay_cells_{gtag}{cond}_{cell_class}_0to{int(POST_SHOW_MIN)}min")

                        # (B) pooled + per-embryo mean
                        fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["l3"])
                        plot_overlay_with_pooled(
                            cells_df=d, ax=ax, level="embryo",
                            TIME_COL=TIME_COL, TRACE_COL=TRACE_COL, PHASE_COL=PHASE_COL,
                            EMB_COL=EMB_COL, ROI_COL=ROI_COL,
                            color=col, PRE_SHOW_MIN=PRE_SHOW_MIN, POST_SHOW_MIN=POST_SHOW_MIN,
                            GAP_W=GAP_W, draw_pre=L3_OVERLAY_DRAW_PRE, max_cell=L3_OVERLAY_MAX_CELL
                        )
                        ax.set_title(
                            f"Q2 L3 overlay (embryos) | {pid} | {gpfx}{cfg.display_name(cond)} | {cell_class}",
                            pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
                        )
                        ax.set_xlabel(xlabel_gap_axis())
                        ax.set_ylabel(TRACE_COL)
                        style_axes(ax)
                        save_both(fig,
                                  Path("L3_overlay_embryos") / cell_class / str(pid)
                                  / f"overlay_embryos_{gtag}{cond}_{cell_class}_0to{int(POST_SHOW_MIN)}min")

    # ---------- BOX: 3 panels per (pair, cell_class) ----------
    # 1. events: events / cell / W min
    # 2. mean_amplitude: post W-min trace mean
    # 3. mean_peak_duration: per-peak FWHM averaged across cells in embryo
    if ONLY in (None, "box"):
        wsuf = cfg.window_suffix()
        pub = bool(getattr(cfg, "PUBLICATION_MODE", False))

        # The three metrics to plot. Each entry has:
        #   key       = subdirectory / filename token
        #   col       = column name in Q2_embryo_summary_cells.csv
        #   ylabel_long   = label used in monitoring mode (verbose)
        #   ylabel_short  = label used in publication mode (concise)
        #   title_metric  = metric name used in titles
        box_metrics = [
            dict(
                key="events",
                col=EVENTS_COL,
                ylabel_long  = f"events / cell / {wsuf} min (post)  [embryo unit]",
                ylabel_short = f"events / cell / {wsuf} min",
                title_metric = "events count",
                title_pub    = r"Ca$^{2+}$ Event Frequency",
            ),
            dict(
                key="amplitude",
                col=f"embryo_mean_amplitude_post_{wsuf}min",
                ylabel_long  = f"mean trace amplitude (post {wsuf} min)  [embryo unit]",
                ylabel_short = r"mean amplitude ($F/F_0$)",
                title_metric = "mean amplitude",
                title_pub    = r"Ca$^{2+}$ Amplitude",
            ),
            dict(
                key="duration",
                col=f"embryo_mean_peak_duration_post_{wsuf}min",
                ylabel_long  = f"mean peak duration (min, post {wsuf} min)  [embryo unit]",
                ylabel_short = "peak duration (min)",
                title_metric = "mean peak duration",
                title_pub    = r"Ca$^{2+}$ Event Duration",
            ),
        ]

        # Verify all metric columns exist (with friendly warning if not)
        missing = [m for m in box_metrics if m["col"] not in emb.columns]
        if missing:
            for m in missing:
                print(f"  ⚠️  Column '{m['col']}' not in embryo_summary table; "
                      f"skipping '{m['key']}' boxplot.")
            box_metrics = [m for m in box_metrics if m["col"] in emb.columns]

        # Every p-value drawn on any boxplot is also logged here and written to
        # an Excel workbook next to the tables, so the figures can stay clean.
        anova_rows: list[dict] = []      # one row per genotype x drug figure
        pairwise_rows: list[dict] = []   # Tukey + single-genotype Welch pairs + omnibus
        fc_rows: list[dict] = []         # per-drug fold-change summary (drug / vehicle mean)

        for pid in sorted(emb["pair_id"].astype(str).unique()):
            dfp = emb[emb["pair_id"].astype(str) == str(pid)].copy()

            present = set(dfp["condition"].astype(str).unique())
            cond_order = resolve_conditions(present)
            if not cond_order:
                continue

            genos_present = (
                resolve_genotypes(set(dfp["genotype"].astype(str).unique()))
                if "genotype" in dfp.columns else []
            )
            multi_geno = len(genos_present) >= 2

            # Collect each drawn panel per metric so we can share the y-axis
            # across the flat/round panels before saving (unify pass below).
            deferred: dict = {}

            for cell_class in ["flat", "round"]:
                for metric in box_metrics:
                    col = metric["col"]
                    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["box"])

                    if multi_geno:
                        # ---- genotype x drug layout: two-way ANOVA + Tukey HSD ----
                        # Boxes coloured by genotype, grouped along x by drug.
                        data_list, fill_colors, xpos = [], [], []
                        label_of, stat_rows = {}, []
                        tick_pos, tick_lab = [], []
                        x, within, group_gap = 0.6, 0.42, 0.5
                        for cond in cond_order:
                            cond_xs = []
                            for geno in genos_present:
                                y = dfp[
                                    (dfp["condition"].astype(str) == str(cond)) &
                                    (dfp["genotype"].astype(str) == str(geno)) &
                                    (dfp[EMB_CELL_CLASS_COL].astype(str) == str(cell_class))
                                ][col]
                                y = pd.to_numeric(y, errors="coerce").to_numpy(float)
                                y = y[np.isfinite(y)]
                                idx = len(data_list)
                                data_list.append(y)
                                fill_colors.append(genotype_box_color(geno))
                                xpos.append(x)
                                label_of[idx] = f"{geno}|{cond}"
                                cond_xs.append(x)
                                for v in y:
                                    stat_rows.append(
                                        {"value": float(v), "genotype": geno, "condition": cond}
                                    )
                                x += within
                            if cond_xs:
                                tick_pos.append(float(np.mean(cond_xs)))
                                tick_lab.append(cfg.pub_label(cond) if pub else cfg.display_name(cond))
                            x += group_gap

                        if all(len(y) == 0 for y in data_list):
                            plt.close(fig)
                            continue

                        draw_prism_box(ax, data_list, xpos, fill_colors)
                        ax.set_xticks(tick_pos)
                        ax.set_xticklabels(tick_lab)
                        ax.set_ylabel(metric["ylabel_short" if pub else "ylabel_long"])
                        ax.legend(
                            handles=[Patch(facecolor=genotype_box_color(g), edgecolor="black",
                                           alpha=STYLE["box"]["box_alpha"], label=g)
                                     for g in genos_present],
                            frameon=False, fontsize=cfg.FONT["legend"], loc="upper right",
                        )

                        # Two-way ANOVA (corner) + Tukey HSD brackets (significant pairs)
                        stat_df = pd.DataFrame(stat_rows)
                        dec = cfg.STATS.get("pval_decimals", 3)
                        anova_label, anova_stats, _ = twoway_anova_label(
                            stat_df, "value", decimals=dec)
                        if anova_label:
                            ax.text(0.02, 0.98, anova_label, transform=ax.transAxes,
                                    ha="left", va="top",
                                    fontsize=cfg.STATS.get("pval_fontsize", cfg.FONT["legend"]))
                        if anova_stats is not None:
                            anova_rows.append(dict(
                                pair_id=pid, cell_class=cell_class,
                                metric=metric["key"], **anova_stats))
                        if not stat_df.empty:
                            stat_df["cell"] = stat_df["genotype"] + "|" + stat_df["condition"]
                            tukey_all = tukey_significant_pairs(stat_df, "value", "cell")
                            for a, b, p, reject in tukey_all:
                                pairwise_rows.append(dict(
                                    pair_id=pid, cell_class=cell_class, metric=metric["key"],
                                    test="tukey_hsd", group1=a, group2=b,
                                    p_value=float(p), significant=bool(reject)))
                            xpos_by_label = {label_of[i]: xpos[i] for i in label_of}
                            data_by_label = {label_of[i]: list(data_list[i]) for i in label_of}
                            # Always annotate the planned comparisons (whether or
                            # not significant): genotype-vs-genotype within each
                            # condition, and condition-vs-condition within each
                            # genotype.
                            wanted = []
                            for cond in cond_order:
                                for gi in range(len(genos_present)):
                                    for gj in range(gi + 1, len(genos_present)):
                                        wanted.append((f"{genos_present[gi]}|{cond}",
                                                       f"{genos_present[gj]}|{cond}"))
                            for geno in genos_present:
                                for ci in range(len(cond_order)):
                                    for cj in range(ci + 1, len(cond_order)):
                                        wanted.append((f"{geno}|{cond_order[ci]}",
                                                       f"{geno}|{cond_order[cj]}"))
                            draw_brackets_for_pairs(ax, wanted, tukey_all,
                                                    xpos_by_label, data_by_label, decimals=dec)

                        ax.set_title(
                            f"{metric['title_pub']}\n{cfg.cell_class_display(cell_class)}" if pub else
                            f"Q2 {metric['title_metric']} | {pid} | {cell_class}\n"
                            f"(genotype × drug; embryo unit)",
                            pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
                        )
                        style_axes(ax)
                        apply_dense_yticks(ax)
                        deferred.setdefault(metric["key"], []).append((
                            fig, ax,
                            Path(f"boxplot_{metric['key']}") / cell_class / str(pid)
                            / f"box_{metric['key']}_{cell_class}_genoXdrug_post{wsuf}min"
                        ))

                    else:
                        # ---- single-genotype: original per-drug layout (unchanged) ----
                        xpos = np.linspace(0.9, 0.9 + 0.5 * (len(cond_order) - 1), len(cond_order))
                        data_list, labels_long, labels_short, colors = [], [], [], []
                        for cond in cond_order:
                            y = dfp[
                                (dfp["condition"].astype(str) == str(cond)) &
                                (dfp[EMB_CELL_CLASS_COL].astype(str) == str(cell_class))
                            ][col]
                            y = pd.to_numeric(y, errors="coerce").to_numpy(float)
                            y = y[np.isfinite(y)]
                            data_list.append(y)
                            labels_long.append(cfg.display_name(cond))   # e.g. "Isoprenaline"
                            labels_short.append(cfg.pub_label(cond))      # "ISO", "Yoda1", "GsMTx4"
                            colors.append(cfg.color_for(cond))

                        if all(len(y) == 0 for y in data_list):
                            plt.close(fig)
                            continue

                        draw_prism_box(ax, data_list, xpos, colors)

                        ax.set_xticks(xpos)
                        ax.set_xticklabels(labels_short if pub else labels_long)

                        # y-axis override only applies to the events box (back-compat).
                        # For amplitude / duration, always auto-scale (different units).
                        used_fixed_yticks = False
                        if metric["key"] == "events":
                            ylim_override, yticks_override = cfg.boxplot_ylim_for(pid)
                            if ylim_override is not None:
                                ax.set_ylim(*ylim_override)
                                if yticks_override is not None:
                                    ax.set_yticks(yticks_override)
                                    used_fixed_yticks = True

                        ax.set_ylabel(metric["ylabel_short" if pub else "ylabel_long"])

                        if pub:
                            ax.set_title(
                                f"{metric['title_pub']}\n{cfg.cell_class_display(cell_class)}",
                                pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
                            )
                        else:
                            ax.set_title(
                                f"Q2 cells {metric['title_metric']} | {pid} | {cell_class}\n"
                                f"(embryo unit; pooled across batches)",
                                pad=cfg.FONT["title_pad"], fontsize=cfg.FONT["title"]
                            )
                        style_axes(ax)
                        if not used_fixed_yticks:
                            apply_dense_yticks(ax)

                        # p-values (uses cfg.STATS); positions itself above the data.
                        # cond_order is aligned with data_list, so planned-pairs
                        # mode can match comparisons by condition token.
                        stat_res = add_pvals(ax, data_list, xpos, labels=cond_order)
                        if stat_res is not None:
                            if stat_res.get("omnibus_p") is not None:
                                pairwise_rows.append(dict(
                                    pair_id=pid, cell_class=cell_class, metric=metric["key"],
                                    test="omnibus", group1="(all)", group2="",
                                    p_value=float(stat_res["omnibus_p"]),
                                    significant=bool(stat_res["omnibus_p"] < 0.05)))
                            for g1, g2, p, sig in stat_res.get("pairwise", []):
                                pairwise_rows.append(dict(
                                    pair_id=pid, cell_class=cell_class, metric=metric["key"],
                                    test=f"welch_{stat_res['correction']}",
                                    group1=g1, group2=g2,
                                    p_value=float(p), significant=bool(sig)))

                        deferred.setdefault(metric["key"], []).append((
                            fig, ax,
                            Path(f"boxplot_{metric['key']}") / cell_class / str(pid)
                            / f"box_{metric['key']}_{cell_class}_post{wsuf}min"
                        ))

            # ---- share the y-axis across the flat/round panels, then save ----
            # Each panel's finalized ylim already includes its bracket headroom,
            # so the union of the two is the tightest range that fits both. With
            # a shared scale a box only looks wider when its spread genuinely is.
            share_yl = getattr(cfg, "SHARE_CELLCLASS_YLIM", True)
            for _mkey, _panels in deferred.items():
                if share_yl and len(_panels) > 1:
                    _los, _his = zip(*(a.get_ylim() for _, a, _ in _panels))
                    _lo, _hi = min(_los), max(_his)
                    for _f, _a, _r in _panels:
                        _a.set_ylim(_lo, _hi)
                for _f, _a, _r in _panels:
                    save_both(_f, _r)

            # ---- fold-change plot: drug / vehicle-mean (drug panel only) ----
            # Uses planned_pairs as (vehicle, drug). Skipped for genotype x drug
            # pids (fold change there needs genotype-specific normalisation).
            _fc_pairs = cfg.STATS.get("planned_pairs") or []
            _use_log2 = bool(getattr(cfg, "FOLDCHANGE_LOG2", False))
            if _fc_pairs and not multi_geno:
                _fc, _gvy = plot_foldchange_for_pid(
                    dfp, pid, box_metrics, _fc_pairs, _use_log2, pub, wsuf,
                    EMB_CELL_CLASS_COL)
                fc_rows.extend(_fc)
                pairwise_rows.extend(_gvy)
            elif _fc_pairs and multi_geno:
                # genotype x drug: normalise within each genotype's own vehicle
                _fc, _cmp = plot_foldchange_genotype_for_pid(
                    dfp, pid, box_metrics, _fc_pairs, _use_log2, pub, wsuf,
                    EMB_CELL_CLASS_COL)
                fc_rows.extend(_fc)
                pairwise_rows.extend(_cmp)

        # ---- cell-count summary per genotype x drug (the 2x2 for MIC/Piezo x
        # E3/ISO; WT drug datasets give one row per drug). Split by cell_class,
        # so the sample size behind every figure travels with the workbook.
        # Uses the SAME n_cells the boxplots roll up.
        ncell_df = pd.DataFrame()
        _keys = ["genotype", "condition"]
        if not emb.empty and set(_keys + ["cell_class", "n_cells",
                                          "embryo_id"]).issubset(emb.columns):
            _piv = emb.pivot_table(index=_keys, columns="cell_class",
                                   values="n_cells", aggfunc="sum", fill_value=0)
            _piv.columns = [f"n_cells_{c}" for c in _piv.columns]
            _agg = emb.groupby(_keys).agg(
                n_embryos=("embryo_id", "nunique"),
                n_cells_total=("n_cells", "sum"))
            ncell_df = _agg.join(_piv)
            ncell_df["cells_per_embryo_mean"] = (
                ncell_df["n_cells_total"] / ncell_df["n_embryos"]).round(2)
            ncell_df = ncell_df.reset_index()

        # ---- write the stats + counts to an Excel workbook next to the tables ----
        if anova_rows or pairwise_rows or fc_rows or not ncell_df.empty:
            xlsx = cfg.tables_dir() / "Q2_stats_pvalues.xlsx"
            xlsx.parent.mkdir(parents=True, exist_ok=True)
            try:
                with pd.ExcelWriter(xlsx, engine="openpyxl") as _xw:
                    if not ncell_df.empty:
                        ncell_df.to_excel(_xw, sheet_name="n_cells", index=False)
                    if fc_rows:
                        pd.DataFrame(fc_rows).to_excel(
                            _xw, sheet_name="foldchange", index=False)
                    if anova_rows:
                        pd.DataFrame(anova_rows).to_excel(
                            _xw, sheet_name="twoway_anova", index=False)
                    if pairwise_rows:
                        pd.DataFrame(pairwise_rows).to_excel(
                            _xw, sheet_name="pairwise", index=False)
                print(f"[OK] wrote stats workbook: {xlsx}")
            except Exception as e:
                # Fall back to CSVs if the Excel engine is unavailable.
                if not ncell_df.empty:
                    ncell_df.to_csv(xlsx.with_name("Q2_stats_n_cells.csv"), index=False)
                if fc_rows:
                    pd.DataFrame(fc_rows).to_csv(
                        xlsx.with_name("Q2_stats_foldchange.csv"), index=False)
                if anova_rows:
                    pd.DataFrame(anova_rows).to_csv(
                        xlsx.with_name("Q2_stats_twoway_anova.csv"), index=False)
                if pairwise_rows:
                    pd.DataFrame(pairwise_rows).to_csv(
                        xlsx.with_name("Q2_stats_pairwise.csv"), index=False)
                print(f"[WARN] Excel write failed ({type(e).__name__}); wrote CSVs instead.")

    print("[OK] Q2_cells_events done.")


if __name__ == "__main__":
    main()
