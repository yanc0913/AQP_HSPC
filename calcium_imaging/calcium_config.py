# calcium_config.py
# -*- coding: utf-8 -*-
"""
Single source of truth for the calcium-imaging analysis pipeline.

Edit this file (and ONLY this file) when:
  * data lives somewhere else                       → PATHS
  * you add a new drug / condition                  → CONDITIONS section
  * you want to tweak which plots show which group  → PLOTTING.conditions_order
  * you want a different time window or gap-axis    → PLOTTING.time_window
  * a particular pair_id needs a custom y-axis      → PLOTTING.boxplot_ylim
  * you want to (rarely) retune peak detection      → PEAK_DETECTION

Items most-frequently changed are placed at the TOP; rarely-touched items at the BOTTOM.

This is a plain Python module: import it with `from calcium_config import CFG`
or `import calcium_config as cfg`. No YAML, no CLI; just edit and rerun.
"""

from __future__ import annotations
from pathlib import Path

# =============================================================================
# 1. PATHS
# =============================================================================
# All output goes under ROOT / "_py_out". Change ROOT when you move machines
# or analyse a different dataset.

# Data root is machine-specific and lives in paths_local.py (git-ignored).
# Copy paths_local.example.py -> paths_local.py and set DATA_BASE there.
from paths_local import ROOT

# Output base. Sub-paths (tables/plots) are computed by helpers below so they can
# include the analysis window in their names (e.g. "_py_out_10min", "_py_out_20min")
# — this lets you run different POST_ANALYSIS_MIN settings side-by-side without
# overwriting each other.
OUT_ROOT_BASE = ROOT / "_py_out"

# Per-script plot subdirs. Each plot script writes its outputs into
#   plots_png_dir() / <subdir> / ...   and   plots_svg_dir() / <subdir> / ...
# Subdirs are auto-created on the fly. Changing these strings only changes
# where new plots are written; no script reads back from them.
# Convention: subdir name matches the plot script base name.
Q1_PLOT_SUBDIR        = "Q1_vDA_dDA_ratio"
Q2_CELLS_PLOT_SUBDIR  = "Q2_cells_events"
Q3_VDA_PLOT_SUBDIR    = "Q3_vDA_trace"


# =============================================================================
# 2. PLOTTING — frequently tweaked (time window, which conditions, x/y scale)
# =============================================================================

# ----- Time axis (shared by all line plots) -----
PRE_SHOW_MIN  = 5.0   # how many minutes of pre-phase to display (negative side)
POST_SHOW_MIN = 20.0  # how many minutes of post-phase to display in line plots
GAP_W         = 1.2   # width of the visual "remount gap" box between pre and post


# ----- Analysis window for event detection & counting -----
# This is the window on which:
#   (a) event detection runs (median/MAD threshold is computed on this slice)
#   (b) events are counted for boxplots ("events / cell / N min")
#   (c) output table column names get the suffix `_<N>min`
#   (d) output directory gets the suffix `_<N>min`
#
# Set to None to use POST_SHOW_MIN automatically (99% of the time what you want).
# Set to a specific number to decouple analysis window from display window
# (e.g. plot 30 min but count events only in first 10 min).
POST_ANALYSIS_MIN = None    # None → use POST_SHOW_MIN


# ----- Which conditions appear on which plot, in which order -----
# Q1 (vDA/dDA ratio): always control-only; do not list other drugs here.
Q1_CONDITIONS_ORDER       = ["E3"]
# Q2 (single-cell events boxplots & traces): typically a focused subset
Q2_CELLS_CONDITIONS_ORDER = ["DMSO", "Yoda", "E3", "GsMTx", "ISO", "BDM"]   # superset across datasets; filtered to those present (vehicle first)
# Q3 (vDA band trace, all conditions present): the "drug comparison" panel
Q3_CONDITIONS_ORDER       = ["E3", "DMSO", "GsMTx", "Yoda", "ISO", "BDM"]
# ["DMSO", "Yoda", "E3", "GsMTx"]
# ["E3", "ISO", "BDM"]


# ----- Font sizes (pt). Set any value to None to use matplotlib's default. -----
# These now drive matplotlib rcParams in every per-dataset plot script, so the
# values here are authoritative for Q1/Q2/Q3 output (they used to be declared
# but ignored for `label`/`tick`, which silently fell back to matplotlib's
# 10 pt default). Sized to match the main-figure row layouts, so a per-dataset
# panel dropped into a supplementary figure carries the same typography.
# The main-figure scripts pass their own explicit sizes and are unaffected.
FONT = dict(
    family       = "sans-serif",
    sans         = ["Arial", "Helvetica", "DejaVu Sans"],
    title        = 10.5,
    label        = 9.5,
    tick         = 9,
    legend       = 9,
    title_pad    = 6,
)


# ----- Figure sizes (inches). Set any entry to None to use matplotlib's default. -----
# These map to STYLE['fig'] in the old plot_style.py.
# Example: FIG_SIZE['box'] = None  →  plt.subplots() called without figsize argument.
FIG_SIZE = dict(
    box = (3.2, 4.2),    # boxplot panels (Prism-like: narrow + tall)
    l1  = (5.3, 3.2),    # per-embryo/all-cells trace panel
    l2  = (5.0, 3.0),    # per-embryo mean trace
    l3  = (6.2, 3.4),    # repeat+pooled trace (the main one)
)


# ----- Y-axis ranges for boxplots -----
# Default: auto-scale (matplotlib decides).
# To override for a specific pair_id, set the override here AND flip its use_override flag.
#
# Two-level lookup: BOXPLOT_YLIM[pair_id] -> dict with keys
#     use_override : bool   — if False, the ylim/yticks below are ignored (auto-scale)
#     ylim         : tuple  — (ymin, ymax)
#     yticks       : list   — tick positions (or None to let mpl decide ticks)
#
# Pair_ids not listed here always auto-scale.
BOXPLOT_YLIM = {
    "E3_vs_Yoda_vs_GsMTx": dict(
        use_override = False,                  # set True to apply the override below
        ylim         = (-0.2, 7.0),
        yticks       = list(range(0, 8)),
    ),
    # Add more pair_ids as needed:
    # "E3_vs_ISO_vs_BDM": dict(
    #     use_override = False,
    #     ylim         = (-0.2, 5.0),
    #     yticks       = [0, 1, 2, 3, 4, 5],
    # ),
}

# Share the y-axis between the flat and round panels of each boxplot metric
# (events, amplitude, duration) so their spreads are visually comparable: a
# wider-looking box then means a genuinely wider spread, not just a tighter
# auto-scale. Both panels are set to the UNION of their individual ranges.
# A per-pair_id override in BOXPLOT_YLIM still wins (both panels use it, so
# they already match). Set False to auto-scale each panel independently.
SHARE_CELLCLASS_YLIM = True


# ----- Statistics: which tests, when to show -----
# Omnibus test (the "single overall p-value" shown in the corner of a boxplot):
#   "auto"   → 2 groups: Welch t-test;  ≥3 groups: one-way ANOVA  (recommended)
#   "anova"  → always one-way ANOVA (note: with 2 groups this equals t² but is less standard)
#   "ttest"  → always Welch t-test (with ≥3 groups, omnibus is skipped — only pairwise shown)
#   "none"   → never show an omnibus result; only pairwise brackets
#
# Pairwise post-hoc test (between every pair of groups) is always Welch t-test
# with Holm-Bonferroni multiple-comparison correction.
STATS = dict(
    omnibus_test         = "auto",           # "auto" / "anova" / "ttest" / "none"
    pairwise_func        = "welch_ttest",    # Welch t-test (equal_var=False)
    multipletest_method  = "holm",           # Holm-Bonferroni (proper monotonic version)
    show_omnibus         = True,             # print omnibus p in top-left corner
    show_pairwise        = True,             # draw brackets with corrected p-values
    bracket_lw           = 0.6,
    pval_fontsize        = 8.5,
    bracket_top_band     = 0.78,             # brackets start at 78% of axis height
    bracket_max_band     = 0.98,             # don't draw above 98%
    pval_decimals        = 3,                # display p to 3 sig figs
    # Below this, show "p<floor" instead of a long string of leading zeros
    # (e.g. p<0.0001 rather than p=0.0000137). Set to 0/None to always show
    # the exact digits. Never uses scientific notation either way.
    pval_min_display     = 1e-4,
    # Targeted comparisons: list of (vehicle, drug) condition-token pairs.
    # This is a SUPERSET of every experiment's pairs — each run automatically
    # uses only the pairs whose BOTH conditions are present in the loaded
    # dataset, so you never edit this when switching datasets:
    #   E3/GsMTx/DMSO/Yoda data -> uses (E3,GsMTx) + (DMSO,Yoda)
    #   E3/ISO/BDM data         -> uses (E3,ISO)   + (E3,BDM)
    # For the matched pairs the single-genotype boxplots draw ONLY those pairwise
    # Welch-t brackets (raw p, no Holm) and hide the omnibus; the same pairs
    # drive the fold-change plot (drug / vehicle mean), so order matters —
    # first = vehicle, second = drug. If NONE of the pairs are present, the
    # boxplots fall back to omnibus + all-pairwise Holm (fold-change is skipped).
    # Empty [] -> always omnibus + all-pairwise Holm, no fold-change.
    # Does NOT touch the genotype x drug path (two-way ANOVA + Tukey), which
    # fires when >=2 genotypes share a pair_id (the MIC/Piezo x {E3, ISO} expt).
    planned_pairs        = [("E3", "GsMTx"), ("DMSO", "Yoda"),
                            ("E3", "ISO"), ("E3", "BDM")],
)


# ----- Publication mode -----
# Two presentation styles for boxplots, controlled by a single switch:
#
#   False (default) — "monitoring" / data exploration mode
#       Verbose titles ("Q2 cells events count | E3_vs_ISO_vs_BDM | round")
#       Verbose y-labels ("events / cell / 20 min (post) [embryo unit]")
#       X-tick labels use full DISPLAY_NAMES ("Isoprenaline")
#       Useful when scanning many plots to spot what's what.
#
#   True — "publication" / paper-figure mode
#       Short titles (just metric name, or empty)
#       Concise y-labels (no [embryo unit] tag)
#       X-tick labels use short canonical names ("ISO", "BDM")
#       Clean Prism-style output suitable for direct figure use.
#
# Only affects boxplot panels (BOX section in plot_Q2_cells_events.py for now).
# Trace plots (L1/L2/L3) are unchanged by this switch.
PUBLICATION_MODE = True    # set True for paper figures


# =============================================================================
# 3. CONDITIONS — drug name normalisation, display labels, and colour mapping
# =============================================================================
# When you add a new drug:
#   1. Add an entry to CONDITION_ALIASES (canonical name -> list of regex patterns)
#   2. Add the same canonical name to COLORS with a colour
#   3. Add it to DISPLAY_NAMES with the label you want shown on plots
#   4. (optional) Add it to the relevant *_CONDITIONS_ORDER above
#   5. Re-run the pipeline (python run_all.py)
#
# Unknown drugs (not matching any alias) keep their drug_raw value, get a warning
# at build time, and (if INCLUDE_UNKNOWN_IN_PLOTS = True) are plotted in COLOR_FALLBACK.

CONTROL_CONDITION = "E3"

# Canonical name -> list of regex patterns (case-insensitive prefix matches recommended).
# First match wins, scanning in dict insertion order.
CONDITION_ALIASES = {
    "E3":     [r"(?i)^E3"],         # E3, E3i, E3g, ...
    "Yoda":   [r"(?i)^Yoda"],       # Yoda, Yoda1, Yoda-1, ...
    "GsMTx":  [r"(?i)^GsMTx"],      # GsMTx, GsMTx4, ...
    "DMSO":   [r"(?i)^DMSO"],
    "ISO":    [r"(?i)^ISO"],
    "BDM":    [r"(?i)^BDM"],
}

# Canonical name -> the label shown to readers on plot titles, legends, x-tick labels.
# Internal grouping, filenames, and column names still use the canonical name (short).
# Missing keys fall back to the canonical name itself (see display_name() helper).
DISPLAY_NAMES = dict(
    E3    = "E3",
    Yoda  = "Yoda1",
    GsMTx = "GsMTx4",
    ISO   = "Isoprenaline",
    BDM   = "BDM",
    DMSO  = "DMSO",
)

# Drug condition colours used in all plots.
# Must include every canonical name from CONDITION_ALIASES.
# Unknown drugs (warning at build time) get fallback "tab:gray".
# Palette rules: controls are grey; no orange, no magenta.
COLORS = dict(
    E3    = "#4D4D4D",  # control (dark grey)
    DMSO  = "#808080",  # vehicle control (grey)
    Yoda  = "#6699CC",  # Piezo1 agonist (Picton blue) — activation
    GsMTx = "#CC3333",  # mechanosensitive channel blocker — inhibition
    ISO   = "#117733",  # β-adrenergic agonist
    BDM   = "#795548",  # myosin inhibitor
)
COLOR_FALLBACK = "tab:gray"   # used when an unknown drug is plotted

# Handling of drugs that don't match any alias above:
#   True  → still plot them (using COLOR_FALLBACK); a warning is printed at build time
#   False → drop them from plots entirely (still appear in raw tables, just not plotted)
INCLUDE_UNKNOWN_IN_PLOTS = True


# =============================================================================
# 3b. GENOTYPE — genetic-background comparisons (e.g. piezo1 crispant vs control)
# =============================================================================
# For experiments that compare genetic backgrounds, the genotype is encoded
# ONLY in the filename (there is no folder convention for it), e.g.
#     ..._30sInt_5mins_piezo_crsp_e2_pre_E3_RunningBrightest_...   -> Piezo
#     ..._30sInt_30mins_UIC_e3_post_Yoda_RunningBrightest_...      -> UIC
# The drug condition is parsed exactly as for WT data; genotype is an extra,
# orthogonal dimension. Files matching no alias get DEFAULT_GENOTYPE, so
# single-background experiments (ISO, Yoda) collapse to one genotype and behave
# exactly as before (genotype drops out of every grouping).
#
# Add a new genotype:
#   1. Add canonical -> [regex patterns] to GENOTYPE_ALIASES (searched anywhere
#      in the filename stem; first match in dict order wins).
#   2. (optional) Add it to GENOTYPE_ORDER for a fixed plot/facet order.

GENOTYPE_ALIASES = {
    "MIC":   [r"(?i)(^|_)MIC(_|$)"],     # mock-injected control (real experiment)
    "UIC":   [r"(?i)(^|_)UIC(_|$)"],     # uninjected control (proof-of-concept)
    "Piezo": [r"(?i)piezo[_\-]?cr(sp)?(?=_|$)"],  # piezo1 crispant: piezo_crsp, piezo-crsp, piezo_Cr
}
DEFAULT_GENOTYPE = "WT"   # used when no alias matches (non-genotype experiments)

# Facet/plot order; genotypes not listed are appended alphabetically.
GENOTYPE_ORDER = ["WT", "MIC", "UIC", "Piezo"]

# Genotype box/bar colours (same palette rules: controls grey; no orange/magenta).
# Used by the genotype x drug boxplots and the ISO main-figure bars.
GENOTYPE_COLORS = {
    "WT":    "#4D4D4D",  # uninjected control (dark grey)
    "MIC":   "#808080",  # mock-injected control (grey)
    "UIC":   "#B3B3B3",  # uninjected control, proof-of-concept (light grey)
    "Piezo": "#CC3333",  # piezo1 crispant — experimental (red)
}
GENOTYPE_COLOR_FALLBACK = "tab:gray"


def genotype_color(genotype: str) -> str:
    """Box/bar colour for a genotype (controls grey; crispant coloured)."""
    return GENOTYPE_COLORS.get(str(genotype), GENOTYPE_COLOR_FALLBACK)


# =============================================================================
# 4. CELL SIGNAL STATISTIC — which Fiji column drives downstream traces
# =============================================================================
# Cell-level downstream analysis (M_vDA correction, F/F0, events, mean
# amplitude, peak duration) is computed from a single per-frame pixel
# statistic per ROI. The Fiji exports include both:
#
#   "max_bgsub"   = max pixel inside the ROI, background subtracted
#   "mean_bgsub"  = spatial mean of pixels inside the ROI, background subtracted
#
# Both columns are read in and kept in the cell table. The one chosen here
# becomes `cell_signal_raw` and feeds every cell-level metric downstream.
#
# Default: "mean" (paper main figure).
# Rationale: max is dominated by single-frame artefacts on ROIs that touch
# vasculature (flowing bright vesicles; subpixel registration jitter from
# SIFT alignment). QC on the ISO and Yoda datasets showed corr(max,mean)
# in the 0.5-0.7 range and ACF1(max) < 0.35 for many post ROIs, with
# event counts inflated 2-5x by these single-frame spikes. Switching to
# mean yields the same direction of effect (ISO > E3 > BDM; Yoda > DMSO)
# with comparable significance, and is robust to ROI-area drift between
# pre/post phases.
#
# max remains computed as an internal sanity check (see briefing A §2):
# if a main-figure effect is significant under mean but completely
# absent under max, that warrants investigation (possibly a real local
# vs whole-cell biology signal, not an artefact).
#
# To switch, set CELL_SIGNAL_STAT below. Output goes to a window-suffixed
# directory; if you want to keep results from both stats side by side,
# also append the stat to the output suffix manually, or run twice into
# separate ROOT folders.

CELL_SIGNAL_STAT = "mean"   # "mean" (default, paper main) or "max" (legacy / sanity)


# =============================================================================
# 5. PEAK DETECTION (event-finding algorithm) — rarely tweaked
# =============================================================================
# These together define what counts as a calcium "event" in a single-cell trace.
# See README §"Peak detection" for the rationale.
#
# A frame t is flagged as a peak when ALL hold:
#   shape: x[t-1] < x[t] >= x[t+1]
#   (neighbour_ratio  OR  prominence_gate) is True
#   x[t] >= pre_median + robust_z_k * sigma_robust(pre_segment)
#
# Threshold is computed on the PRE phase of the same cell (the "quiet baseline"),
# then applied to detect peaks in the POST phase analysis window. This makes the
# detector sensitive to sustained-activation drugs (e.g. ISO): even when post
# trace is uniformly elevated, peaks rising above pre noise floor are still found.
#
# Adjacent surviving candidates (within refractory_frames) are merged; the
# highest is kept.
#
# Note: The analysis window (how many minutes of post phase to use) is set in
# the PLOTTING section above (POST_ANALYSIS_MIN), since it's the value you'd
# normally adjust together with POST_SHOW_MIN.

PEAK_DETECTION = dict(
    # candidate shape & local conditions
    rel_peak_frac   = 0.10,   # neighbour ratio: x[t] >= 1.10 * max(x[t-1], x[t+1])
    prom_win        = 2,      # ±2-frame window for local prominence
    prom_frac       = 0.10,   # x[t] >= 1.10 * P20(window)
    use_prom_gate   = True,
    # global threshold (computed on PRE segment of each cell)
    use_robust_z    = True,
    robust_z_k      = 2.0,    # threshold = pre_median + k * sigma_robust(pre)
    # merging
    refractory_frames = 1,    # adjacent candidates within this many frames are merged
)


# =============================================================================
# 6. ACQUISITION & ROI NAMING — almost never changes
# =============================================================================
# These follow from the Fiji macro (Calcium_ROI_selection_v4.ijm) and the imaging
# protocol. Change only if the upstream macro or microscopy setup changes.

DT_SECONDS = 30.0     # imaging interval (one frame every 30 s)

ROI_VDA = "DA_band"           # ventral DA band (the ROI of interest)
ROI_DDA = "DA_band_dorsal"    # dorsal DA band (reference for ratio)
ROI_BG  = "BG"                # background ROI

GCAMP_CH_MATCH = "gcamp"      # substring used to identify GCaMP channel
LIFE_CH_MATCH  = "life"       # substring used to identify Lifeact channel

COL_MEAN_BGSUB = "mean_bgsub" # column name from Fiji export
COL_MAX_BGSUB  = "max_bgsub"

BATCH_LEVELS = 1              # ROOT/<batch>/<pair>/<cond>/file.csv → batch_id = parts[0]


# =============================================================================
# 7. POST-PHASE FRAME TRIMMING — per-file overrides
# =============================================================================
# When specific post-phase recordings have known bad initial frames (e.g. focus
# settling, bleaching artefacts at the start), list a filename substring and
# the number of leading frames to drop.
#
# IMPORTANT: the integer value is a FRAME COUNT, not minutes.
#   With DT_SECONDS = 30, "10" means "drop the first 10 frames" = first 5 minutes.
#   If you later change DT_SECONDS, re-do the arithmetic.
#
# After trimming, remaining frames are re-indexed from 1 and time_min resets to 0.
# Only post-phase frames are affected; pre-phase is untouched.

POST_TRIM_RULES = {
    "30hpf_GCamp7a_fli1lifeactmCh_30x_1um_30sInt_5mins_e5_post_BDM_RunningBrightest_roi_timeseries_allChannels":   10,   # drop first 10 frames = 5 min
    "30hpf_GCamp7a_fli1lifeactmCh_30x_1um_30sInt_30mins_e1_post_GsMTx_RunningBrightest_roi_timeseries_allChannels": 10,  # drop first 10 frames = 5 min
}


# =============================================================================
# 8. CONVENIENCE HELPERS (no behaviour, just lookups)
# =============================================================================

def post_analysis_min() -> float:
    """The effective event-analysis window in minutes.

    Falls back to POST_SHOW_MIN when POST_ANALYSIS_MIN is left at None.
    All other helpers (window_suffix, output paths, column names) should
    derive from this function so the whole pipeline stays in sync.
    """
    return float(POST_ANALYSIS_MIN if POST_ANALYSIS_MIN is not None else POST_SHOW_MIN)


def window_suffix() -> str:
    """String form of the analysis window for use in filenames and column names.

    Examples:  10.0 -> "10",   20.0 -> "20",   12.5 -> "12p5"
    Integer-valued windows are shown without decimal; non-integer values
    have the dot replaced by 'p' so the suffix remains filename-safe.
    """
    w = post_analysis_min()
    if w == int(w):
        return str(int(w))
    return str(w).replace(".", "p")


# --- Output paths: derived from OUT_ROOT_BASE + window_suffix() ---
# These are FUNCTIONS, not constants, because they depend on POST_ANALYSIS_MIN.
# Call them at runtime; do not cache the result if you might change config later.

def out_root() -> "Path":
    """Output root for the current window setting (e.g. ROOT/_py_out_10min)."""
    return OUT_ROOT_BASE.with_name(OUT_ROOT_BASE.name + "_" + window_suffix() + "min")


def tables_dir() -> "Path":
    """Directory for built CSV tables."""
    return out_root() / "tables"


def plots_png_dir() -> "Path":
    """Directory for PNG plots."""
    return out_root() / "plots_png"


def plots_svg_dir() -> "Path":
    """Directory for SVG plots."""
    return out_root() / "plots_svg"


# --- Column-name generators: keep CSV column names in sync with the window ---

def events_col_per_cell() -> str:
    """Per-cell event-rate column in Q2_events_cells.csv (window-aware)."""
    return f"events_per_cell_per_{window_suffix()}min_post"


def events_col_per_embryo() -> str:
    """Embryo-mean event-rate column in Q2_embryo_summary_cells.csv."""
    return f"mean_events_per_cell_per_{window_suffix()}min_post"


def n_events_col() -> str:
    """Per-cell event-count column."""
    return f"n_events_post{window_suffix()}min"


# =============================================================================
# 9. QC THRESHOLDS — quality-control sentinel module (calcium_qc.py)
# =============================================================================
# calcium_qc.py reads the raw Fiji *_allChannels.csv files and produces
# QC_cells.csv (per-ROI status + numerical values), QC_summary.txt, and a
# QC_plots/ directory with per-batch diagnostic figures. It is run BEFORE
# the main pipeline (calcium_build_tables.py) as an independent sentinel —
# it never modifies data, never alters main pipeline behaviour, and never
# auto-removes ROIs.
#
# Three severity levels:
#   FAIL       = data integrity error (likely Fiji workflow mistake)
#                → user MUST review and fix in Fiji before main pipeline
#   FLAG_HIGH  = data quality concern (likely needs attention)
#                → user reviews; may decide to fix or accept
#   FLAG_LOW   = mild observation (often physiology, not error)
#                → audit trail only
#
# Output location: cfg.ROOT / "_qc" / { QC_cells.csv, QC_summary.txt,
#                                       QC_plots/<batch>/*.png }

QC = dict(
    # ---- FAIL: data integrity ----
    post_min_frames           = 26,    # post must have ≥ 26 frames (≈13 min @ 30s int)
                                       # below this, M_vDA and amp are unreliable

    # ---- FLAG_HIGH: data quality ----
    frac_post_bgsub_negative  = 0.10,  # > 10% post frames with mean_bgsub<0 ⇒ BG drift
    bg_pre_post_jump_high     = 1.15,  # post bg_mean / pre bg_mean > 1.15 ⇒ BG moved
    bg_pre_post_jump_low      = 0.85,  # ... or < 0.85 ⇒ BG moved
    F0_low_mean               = 15.0,  # pre median of mean_bgsub < 15 ⇒ signal near baseline
                                       # (calibrated from ISO+Yoda pooled, P05=14.8)
    F0_low_max                = 40.0,  # pre median of max_bgsub < 40 ⇒ same, max-mode
                                       # (calibrated from ISO+Yoda pooled, P05=47.6 → buffer)

    # ---- FLAG_LOW: observation only ----
    roi_area_ratio_high       = 2.0,   # area_post / area_pre > 2.0 ⇒ ROI re-drawn larger
    roi_area_ratio_low        = 0.5,   # ... or < 0.5 ⇒ ROI re-drawn smaller
    acf1_low                  = 0.30,  # ACF1(cell_signal_raw) < 0.3 ⇒ signal lacks
                                       # frame-to-frame continuity (artefact or fast biology)
    max_mean_corr_low         = 0.50,  # corr(max_bgsub, mean_bgsub) < 0.5 ⇒ max could be
                                       # artefact-driven (sanity check, even in mean mode)

    # ---- Plot ranges (event-density proxy in overview figure) ----
    event_density_threshold_x_median = 1.3,   # frac frames > 1.3×median = event density proxy
    event_density_warn_frac          = 0.40,  # > 40% frames above 1.3×median = noisy ROI
)


def qc_dir():
    """Output dir for QC artefacts. Lives next to _py_out_<W>min/ but is
    independent of the analysis window (QC is on raw Fiji CSVs)."""
    return ROOT / "_qc"


# --- Other lookups ---

def all_known_conditions() -> set[str]:
    """All canonical drug names known to the config."""
    return set(CONDITION_ALIASES.keys())


def all_known_genotypes() -> set[str]:
    """All canonical genotypes known to the config, incl. the default."""
    return set(GENOTYPE_ALIASES.keys()) | {DEFAULT_GENOTYPE}


def genotype_sort_key(genotype: str):
    """Sort key honouring GENOTYPE_ORDER; unlisted genotypes sort after, by name."""
    order = {g: i for i, g in enumerate(GENOTYPE_ORDER)}
    return (order.get(genotype, len(order)), genotype)


def color_for(condition: str) -> str:
    """Return the plot colour for a condition; falls back if unknown."""
    return COLORS.get(condition, COLOR_FALLBACK)


def display_name(condition: str) -> str:
    """
    Return the human-readable label for a condition (used on plot titles,
    legends, x-tick labels). Falls back to the canonical name when not listed.

    Internal grouping, filenames, and CSV column names continue to use the
    canonical short name — only reader-facing text uses display names.
    """
    return DISPLAY_NAMES.get(condition, condition)


# Human-readable cell-class names for figure titles. Internal tokens stay
# "flat"/"round"; only reader-facing title text uses these.
CELL_CLASS_DISPLAY = {"flat": "Elongated Cell", "round": "Round Cell"}


def cell_class_display(cell_class: str) -> str:
    """Reader-facing name for a cell class ('flat' -> 'Elongated Cell')."""
    return CELL_CLASS_DISPLAY.get(str(cell_class), str(cell_class))


# Short x-tick labels for PUBLICATION figures. Falls back to the canonical
# token, so only compounds whose token is a lossy abbreviation need an entry
# (e.g. "Yoda" -> "Yoda1", "GsMTx" -> "GsMTx4"). Kept separate from
# DISPLAY_NAMES so short tokens like ISO stay short on axes (DISPLAY_NAMES
# gives the verbose "Isoprenaline" used in monitoring titles).
PUB_SHORT_NAMES = {"Yoda": "Yoda1", "GsMTx": "GsMTx4"}


def pub_label(cond: str) -> str:
    """Corrected short label for publication x-ticks."""
    return PUB_SHORT_NAMES.get(str(cond), str(cond))


# Target number of y-axis major ticks on box plots (denser = smaller steps).
# Passed to matplotlib's MaxNLocator; the locator still snaps to "nice" values.
BOX_YTICK_NBINS = 18


def boxplot_ylim_for(pair_id: str):
    """
    Return (ylim, yticks) for a pair_id if override is active; else (None, None).

    Plot code should do:
        ylim, yticks = boxplot_ylim_for(pid)
        if ylim is not None:
            ax.set_ylim(*ylim)
            if yticks is not None:
                ax.set_yticks(yticks)
    """
    entry = BOXPLOT_YLIM.get(pair_id)
    if entry is None or not entry.get("use_override", False):
        return None, None
    return entry.get("ylim"), entry.get("yticks")
