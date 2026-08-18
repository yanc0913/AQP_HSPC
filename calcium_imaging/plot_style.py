"""
Stylistic preset for Calcium Imaging Plots
===========================================

This module owns the *purely visual* styling that is shared across all plot
scripts: transparency hierarchy, line widths, boxplot cosmetics, p-value
bracket geometry, axes appearance, output resolution, and vector-export
settings.

Configuration that is *analysis-relevant* (font sizes, figure sizes, drug
colours, condition order, statistics, peak detection, paths) lives in
``calcium_config.py`` — that file is the single source of truth for anything
you might tune between runs.

Division of responsibility
--------------------------
    calcium_config.py          plot_style.py
    -----------------          -------------
    FONT (family, sizes)       dpi
    FIG_SIZE (box, l1..l3)     alpha (transparency hierarchy)
    COLORS (drug palette)      lw    (line widths)
    STATS (tests, brackets)    box   (boxplot cosmetics)
    PEAK_DETECTION             pval  (legacy bracket geometry)
    PATHS, CONDITIONS          axes  (spines, ticks)
                               vector (PDF/SVG export)

Historical note: ``font``, ``fig`` and ``colors`` previously lived here too
(duplicated into config during a migration). They have been removed; use
``cfg.FONT``, ``cfg.FIG_SIZE``, ``cfg.COLORS`` instead.

Usage:
    import calcium_config as cfg
    from plot_style import STYLE
    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE["box"], dpi=STYLE["dpi"])
    ax.plot(..., color=cfg.color_for("E3"), alpha=STYLE["alpha"]["pooled_line"])
"""

from __future__ import annotations

MM_TO_IN = 1 / 25.4  # Millimetres to inches conversion

STYLE = dict(
    # ============ Output resolution ============
    dpi=300,  # 300 DPI for publication-quality figures

    # ============ Transparency hierarchy (alpha values) ============
    # Visual depth ordering: individual cells faintest, pooled means opaque.
    alpha=dict(
        cell=0.45,         # Individual cell traces (faint)
        embryo=0.85,       # Embryo-level traces (medium)
        repeat_line=0.45,  # Technical repeat (per-batch) lines
        repeat_fill=0.05,  # Technical repeat SEM fill (very faint)
        pooled_line=1.00,  # Pooled mean line (opaque)
        pooled_fill=0.14,  # Pooled SEM fill (subtle)
    ),

    # ============ Line widths (pt) ============
    lw=dict(
        cell=0.8,          # Individual cell trace
        embryo=0.5,        # Embryo trace
        repeat=0.7,        # Technical repeat (per-batch) mean
        pooled=1.2,        # Pooled mean (thick for emphasis)
        axis=0.6,          # Axis spine width
    ),

    # ============ Boxplot styling ============
    # Note: dot jitter and dot edge are set directly in the plot scripts
    # (Prism-style: jitter 0.18, black edge). The keys below cover the box
    # body, median, and dot size/alpha.
    box=dict(
        width=0.2,         # Box width relative to x-axis spacing
        box_alpha=0.5,     # Box fill transparency
        box_edge_lw=0.3,   # Box edge line width
        median_lw=0.8,     # Median line width (emphasised)
        whisker_lw=0.6,    # Whisker line width
        cap_lw=0.6,        # Whisker cap line width
        showfliers=False,  # Hide outliers (shown as scatter overlay)
        dot_size=12,       # Scatter dot size (points^2)
        dot_jitter=0.08,   # Legacy default; scripts override with 0.18
        dot_alpha=1.0,     # Scatter dot fill alpha
    ),

    # ============ p-value annotation brackets (legacy geometry) ============
    # The current add_pvals() in plot_Q2_cells_events.py places brackets
    # above the data automatically and reads most settings from cfg.STATS.
    # These keys are retained for backward compatibility / fallback.
    pval=dict(
        bracket_lw=0.6,    # Bracket line width
        fontsize=6,        # p-value text font size
        step_frac=0.10,    # Vertical step between nested brackets (fraction of y-range)
        y_pad_frac=0.22,   # Vertical padding above data (fraction of y-range)
        show_anova=True,   # Display omnibus result in corner
        anova_pos=(0.02, 0.98),  # Omnibus text position (axes coordinates)
    ),

    # ============ Axes styling ============
    axes=dict(
        hide_top_right=True,  # Hide top and right spines (cleaner look)
        tick_len=2.5,         # Tick mark length (pt)
        tick_w=0.6,           # Tick mark width (pt)
    ),

    # ============ Vector export friendliness (Illustrator compatibility) ============
    vector=dict(
        pdf_fonttype=42,   # TrueType fonts in PDF (editable in Illustrator)
        ps_fonttype=42,    # TrueType fonts in PostScript
        svg_fonttype="none",  # No font conversion in SVG
    ),
)
