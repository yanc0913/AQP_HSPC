# Zebrafish GCaMP calcium-imaging analysis pipeline

Turns Fiji-exported ROI time-series into analysis tables, QC reports,
publication figures, supplementary movies and a Source Data workbook.

This code produces the calcium-imaging analysis and figures in Kondrychyn
*et al.*, *"Osmo-hydraulic volume regulation ensures robust
endothelial-to-haematopoietic transition"* (Figure 5, Supplementary Figures 9
and 10).

> **Code only.** Raw imaging data and analysis outputs are not in this
> repository. Copy `paths_local.example.py` → `paths_local.py` (git-ignored)
> and point it at your own data.

## Contents

- [1. Pipeline overview](#1-pipeline-overview)
- [2. Setup](#2-setup)
- [3. Input data](#3-input-data) — the Fiji step, ROI names, CSV format, layout
- [4. Running](#4-running)
- [5. Methods](#5-methods) — corrections, F/F₀, event detection, statistics
- [6. Outputs](#6-outputs) — directory tree, every table, every plot
- [7. Parameters and Methods alignment](#7-parameters-and-methods-alignment)
- [8. Load-bearing decisions](#8-load-bearing-decisions)
- [9. Gotchas and tests](#9-gotchas-and-tests)

---

## 1. Pipeline overview

```
Fiji, upstream and separate:
    1. register the time-series (SIFT / linear stack alignment)  ──► "Aligned_" prefix
    2. Calcium_ROI_selection_v4.ijm: draw and measure ROIs
                                     ──► <movie>_roi_timeseries_allChannels.csv
                                     ──► <movie>_ROIs.zip
       │
       ▼
calcium_qc.py            raw CSVs ──► per-ROI quality metrics   (run FIRST; read-only)
       ▼
calcium_build_tables.py  raw CSVs ──► long table ──► band traces ──► Lifeact M
                         correction ──► F/F0 ──► event detection ──► per-cell and
                         per-embryo tables
       ▼
plot_Q1 / plot_Q2 / plot_Q3     tables ──► figures (PNG + SVG) + stats workbook
       ▼
plot_main_*  /  collect_supp_figures  /  build_source_data  /  make_condition_movie
```

| Script | Role |
|---|---|
| `calcium_config.py` | **Every** tunable parameter. Nothing else should be edited to change the analysis. |
| `calcium_qc.py` | Independent QC sentinel; reads raw CSVs, never writes into the analysis |
| `calcium_build_tables.py` | The data core: CSVs → all analysis tables |
| `plot_Q1_vDA_dDA_ratio.py` | vDA/dDA ratio figures |
| `plot_Q2_cells_events.py` | Single-cell traces, boxplots, **and all the statistics** |
| `plot_Q3_vDA_trace.py` | vDA band trace figures |
| `run_all.py` | Orchestrator; each step in its own subprocess |
| `plot_main_foldchange.py` | Cross-dataset fold change, Yoda1 / GsMTx4 (Fig 5e–h) |
| `plot_main_foldchange_iso.py` | Cross-dataset, ISO / MIC / piezo crispant (Fig 5i–l) |
| `plot_main_q1_ratio.py` | vDA/dDA ratio, 30 vs 48 hpf (Supp Fig 9b) |
| `make_subset_figures.py` | Re-run one dataset restricted to a subset of conditions |
| `collect_supp_figures.py` | Copy the supplementary panels into one folder |
| `build_source_data.py` | Source Data workbook, one sheet per figure |
| `make_condition_movie.py` | 2×2 condition movies |
| `plot_style.py` | Shared matplotlib style |
| `qc/qc_max_vs_mean.py` | Standalone check behind the choice of `mean_bgsub` over `max_bgsub` as the cell signal (§5.2) |

---

## 2. Setup

Tested on **Python 3.10.19** (conda). Nothing requires 3.11.

```bash
pip install -r requirements.txt
cp paths_local.example.py paths_local.py     # then edit DATA_BASE
```

`openpyxl` is needed for the stats workbook and the Source Data builder;
`plot_Q2` falls back to CSVs without it.

For the movie builder only: `tifffile`, `imageio`, `Pillow`, and Fiji's
`Green Fire Blue.lut` so movie colours match the figure colour bar — point
`FIJI_LUT` at it or pass `--lut`. Without it a built-in approximation is used
and the script says so.

---

## 3. Input data

### 3.1 The Fiji step, upstream

Two **separate** operations run in Fiji before any Python:

1. **Registration.** SIFT / linear stack alignment with interpolation. This is
   where the `Aligned_` filename prefix comes from. Do this if there is drift.
2. **ROI selection and measurement.** The included macro
   `Calcium_ROI_selection_v4.ijm` walks through drawing the ROIs, measures mean,
   max and area for every ROI in every frame and every channel, subtracts the
   background, and exports. **It does no registration.**

### 3.2 ROI names — the contract

The Python side keys off these names, so they must be exact:

| ROI name | Shape | Meaning |
|---|---|---|
| `DA_band` | segmented line, thick | **ventral** dorsal-aorta band ("vDA") — the region of interest |
| `DA_band_dorsal` | segmented line, thick | **dorsal** band ("dDA") — the ratio reference |
| `BG` | rectangle | background; subtracted from every other ROI |
| `flat_01`, `flat_02`, … | rectangle | single **elongated** cells |
| `round_01`, `round_02`, … | rectangle | single **round** cells |

Channels must be named `GCaMP` and `Lifeact`.

### 3.3 The exported CSV

One CSV per movie, named `*_roi_timeseries_allChannels.csv`:

```csv
image,phase,channel,roi_name,frame,mean,max,area,bg_mean,mean_bgsub,max_bgsub
```

| Column | Meaning |
|---|---|
| `image` | source filename |
| `phase` | `pre` (before drug) or `post` (after) |
| `channel` | `GCaMP` or `Lifeact` |
| `roi_name` | one of the names above |
| `frame` | 0-based time index |
| `mean`, `max` | raw intensity |
| `area` | ROI area in pixels |
| `bg_mean` | mean of the `BG` ROI in that frame and channel |
| `mean_bgsub`, `max_bgsub` | background-subtracted |

**Only the background-subtracted columns are used.** A missing column raises a
descriptive error rather than silently producing wrong numbers.

### 3.4 Condition and genotype come from the FILENAME

Folders are organisational only. The tokens are matched by
`CONDITION_ALIASES` and `GENOTYPE_ALIASES` in `calcium_config.py`. A movie with
no genotype token becomes `DEFAULT_GENOTYPE` (`WT`), so experiments without
genotypes behave exactly as before.

### 3.5 Layout

```
DATA_BASE/
  Analysis_<experiment>/           <- one dataset; cfg.ROOT points here
    <batch>/<condition>/
        Aligned_..._e1_pre_DMSO_..._roi_timeseries_allChannels.csv
        Aligned_..._e1_post_DMSO_..._roi_timeseries_allChannels.csv
        ...
```

**`discover_csvs` skips any directory whose name starts with `_`.** Every
output, temp and QC tree is underscore-prefixed for exactly that reason. A new
working directory that is *not* underscore-prefixed will be ingested as source
data.

---

## 4. Running

### Per dataset

```bash
python calcium_qc.py         # QC sentinel - run BEFORE the build
python run_all.py            # build tables, then Q1 + Q2 + Q3 plots
python run_all.py build      # tables only
python run_all.py plots      # q1 + q2 + q3, assuming tables exist
python run_all.py q2         # one step: build / q1 / q2 / q3
```

To run on a subset without touching your real outputs, redirect the config in a
script — and do **not** start the directory name with `_`:

```python
import calcium_config as cfg
cfg.ROOT = Path("tmp_root")
cfg.OUT_ROOT_BASE = cfg.ROOT / "_py_out"
main()
```

### Cross-dataset figures

These read the built `Q2_embryo_summary_cells.csv` from several datasets at
once; set `DATASET_ROOTS` and `MAIN_FIG_DIRS` in `paths_local.py` first.

**Order matters here.** `make_subset_figures.py` builds the E3-vs-ISO-only tree
`_py_out_<W>min_E3_ISO`, and both `plot_main_foldchange_iso.py` and
`build_source_data.py` read it — the main ISO figure contrasts only those two
groups, so its p-values come from a two-group run while the three-group
E3/ISO/BDM version stays intact for the supplementary figure. Run the subset
step first or those two scripts fail on a missing tree.

```bash
python make_subset_figures.py        # E3-vs-ISO-only tree; must come first
python plot_main_foldchange.py       # Fig 5e-h
python plot_main_foldchange_iso.py   # Fig 5i-l
python plot_main_q1_ratio.py         # Supp Fig 9b
```

### Collecting and deriving

```bash
python collect_supp_figures.py    # copy panels into _SuppFigures/
python build_source_data.py       # assemble Source_Data.xlsx
```

### Supplementary movies

```bash
export MOVIE_ROOT="/path/to/RepresentativeMovies/Movies"
export MOVIE_PY="/path/to/python"
export FIJI_LUT="/path/to/Green Fire Blue.lut"
bash build_supp_movies.sh
```

Each panel needs a cropped 2-channel TIFF (`TCYX`) and an ImageJ ROI set saved
on that **same cropped** image, containing ROIs named `elongated`, `round` and
optionally `VDA`. `build_supp_movies.sh` records the exact per-panel display
ranges used for the published movies.

---

## 5. Methods

### 5.1 Two experimental tracks

- **Track A — pre/post** (the main pipeline). Each recording has a `pre` phase
  and a `post` phase, separated by a remount gap. All normalisation and all
  thresholds are referenced to the pre phase.
- **Track B — single segment** (`analyze_piezo_baseline_quick.py`). One 20-min
  segment, self-referenced threshold. Used for piezo-crispant baseline data.

`DT_SECONDS` = 30, so one frame = 0.5 min. The pre phase is 5 min (10 frames);
the analysis window is `POST_SHOW_MIN` = **20 min** post-drug (40 frames).

### 5.2 The cell signal

`CELL_SIGNAL_STAT = "mean"` — the per-cell signal is `mean_bgsub`, the
background-subtracted mean over the ROI. `max_bgsub` is retained in every table
but is not what is analysed: a single bright vesicle or a registration artefact
moves the max but not the mean.

### 5.3 Lifeact M correction (post phase only)

Remounting between the pre and post recordings shifts the absolute intensity.
The Lifeact channel is structural, so its vDA signal gives the shift:

```
M = median( Lifeact_vDA, pre ) / median( Lifeact_vDA, post )
GCaMP_post_corrected(t) = GCaMP_post(t) × M
```

Applied to the post phase only. Written to `M_vDA_from_Lifeact.csv`, one row
per embryo, so the factor applied to each recording is auditable.

### 5.4 F/F₀

```
F0   = median( GCaMP_corrected(t) ,  t over the ENTIRE pre phase )
F/F0 = GCaMP_corrected(t) / F0
```

Computed per cell. **F₀ uses the whole pre phase, not the first N post
frames** — so the pre-phase median of F/F₀ is ≈1 by construction, and no post
frame can enter its own baseline.

This is a **ratio, not ΔF/F₀**: vehicle baselines sit at 0.72–1.20, away from
zero, which keeps the fold changes in Figure 5 stable.

### 5.5 Event detection

An event is a **single frame** on the F/F₀ trace satisfying all three:

| | Condition |
|---|---|
| **(i) shape** | `x[t-1] < x[t] >= x[t+1]` |
| **(ii) locality** | `x[t] >= (1 + rel_peak_frac)·max(x[t-1], x[t+1])` **OR** `x[t] >= (1 + prom_frac)·P20(x[t-2 … t+2])` |
| **(iii) global** | `x[t] >= pre_median + k · σ_robust(pre)` |

with `rel_peak_frac` = `prom_frac` = 0.10, `prom_win` = 2, `k` = 2.0, and

```
σ_robust = 1.4826 × MAD        (falls back to the standard deviation when MAD = 0)
```

Candidates within `refractory_frames` (1) of each other are merged, keeping the
largest.

The **OR** in (ii) is deliberate: the neighbour rule catches sharp single-frame
spikes, the prominence rule catches broader transients where the adjacent
frames are also elevated. The threshold in (iii) is **referenced to the pre
phase**, so a drug that raises the whole trace raises the event count rather
than moving the threshold with it. Legacy self-referenced columns are still
written but are not what is plotted.

Event rate is reported as **events per cell per 20 min**.

### 5.6 vDA/dDA ratio

```
ratio(t) = GCaMP_vDA(t) / GCaMP_dDA(t)
```

summarised per embryo as the median over frames. A ratio above 1 means the
ventral wall is brighter — activity polarised to the side where haematopoietic
cells emerge. Computed for control embryos.

### 5.7 Statistics

**The embryo is the unit of analysis.** Cells are averaged to an embryo value
before any test. Every dot on every boxplot is one embryo.

**Planned comparisons.** `STATS["planned_pairs"]` is a *superset* of
`(vehicle, drug)` pairs across all experiments —
`(E3, GsMTx)`, `(DMSO, Yoda)`, `(E3, ISO)`, `(E3, BDM)`. Each run
automatically uses only the pairs whose both conditions are present, so it is
never edited per dataset. For those pairs the boxplots draw only the
drug-vs-vehicle Welch-t brackets, **raw p, no correction**: DMSO/Yoda1 and
E3/GsMTx4 have *different vehicles* and are therefore separate families.
Multiplicity is corrected **within** a family sharing a control, never across.

If none of the planned pairs are present, the code falls back to omnibus
one-way ANOVA + all-pairwise Holm, with a one-time warning.

**Genotype × drug.** When two genotypes share a `pair_id`, the layout switches
to a Type-II two-way ANOVA (main effects, interaction, partial η²) with Tukey
HSD post-hoc.

**Fold change** (main figures). Each drug embryo is divided by the **mean of
its own vehicle**, so vehicles collapse to the baseline. The drug-vs-vehicle
p-value is the two-group Welch on **raw** values — dividing by the vehicle mean
discards the control's spread, so a one-sample test against 1 would be
anti-conservative.

**Event rate is reported as a difference, not a ratio.** Control event rates
reach zero, and a ratio diverges there.

---

## 6. Outputs

```
Analysis_<experiment>/
├── _qc/                                  written by calcium_qc.py
│   ├── QC_files.csv                      one row per CSV: FAIL / FLAG / OK
│   ├── QC_cells.csv                      one row per cell ROI
│   ├── QC_summary.txt
│   └── QC_plots/
└── _py_out_20min/                        <- the "20" is POST_SHOW_MIN
    ├── _config_snapshot.py               the parameters this run used
    ├── tables/
    └── plots_png/ , plots_svg/           identical trees, two formats
```

### `tables/`

| File | One row per | Contents |
|---|---|---|
| `band_traces_long_gcamp.csv` | frame × ROI | vDA / dDA / BG GCaMP traces, long format |
| `band_traces_long_lifeact.csv` | frame × ROI | the same for Lifeact |
| `M_vDA_from_Lifeact.csv` | embryo | the M correction factor applied |
| `Q1_ratio_prepost.csv` | frame | vDA/dDA ratio over time |
| `Q1_ratio_summary_prepost.csv` | embryo × phase | median ratio |
| `Q1_ratio_post_control.csv` | embryo | control-only post ratio |
| `Q1_band_events_post_control.csv` | embryo | band-level event counts |
| `Q2_cells_vDA_prepost.csv` | frame × cell | the F/F₀ traces — the largest table |
| `Q2_events_cells.csv` | **cell** | events pre and post, rate, mean amplitude, mean peak duration, `pre_threshold` |
| `Q2_embryo_summary_cells.csv` | **embryo × cell class** | `n_cells`, `mean_events_per_cell_per_20min_post`, `embryo_mean_amplitude_post_20min`, `embryo_mean_peak_duration_post_20min`, `fraction_active_post` — **the dots on every boxplot** |
| `Q3_vDA_trace_prepost.csv` | frame | vDA band trace per condition |
| `Q2_stats_pvalues.xlsx` | — | see below |

**`Q2_stats_pvalues.xlsx`** carries every p-value, including the exact value
where a figure floors it to `p<0.0001`:

| Sheet | Contents |
|---|---|
| `pairwise` | every pairwise comparison: `pair_id`, `cell_class`, `metric`, `test`, `group1`, `group2`, `p_value` |
| `twoway_anova` | the genotype × drug ANOVA table |
| `n_cells` | genotype × drug sample sizes: n embryos and n cells, split flat/round |
| `foldchange` | the fold-change numbers behind the main figures, with the vehicle mean used |

### `plots_svg/` (and the identical `plots_png/`)

```
Q1_vDA_dDA_ratio/<pair_id>/                    vDA/dDA ratio, per experiment
Q3_vDA_trace/<pair_id>/                        vDA band traces
Q2_cells_events/
    L1_cells/<cell_class>/<pair_id>/<condition>/    one panel per CELL
    L2_embryo/<cell_class>/<pair_id>/               embryo means
    L3_overlay_cells/  L3_overlay_embryos/  L3_repeat_pooled/
                                                    overlays and pooled repeats
    boxplot_amplitude/<cell_class>/<pair_id>/       Ca2+ intensity (F/F0)
    boxplot_events/<cell_class>/<pair_id>/          events per cell per 20 min
    boxplot_duration/<cell_class>/<pair_id>/        mean peak duration
    boxplot_foldchange/<cell_class>/<pair_id>/      drug / its own vehicle
```

`<cell_class>` is `flat` or `round`; `<pair_id>` identifies the experiment,
e.g. `E3_vs_Yoda_vs_GsMTx`.

SVGs carry real text (`svg.fonttype = "none"`), so every label stays editable
in Illustrator.

**Note on the metric name.** Figures are labelled "Ca²⁺ Intensity", but the
files and columns keep the original `amplitude` key so tables, figures and the
stats workbook cross-reference. That is why the intensity panels live under
`boxplot_amplitude/`.

### Cross-dataset outputs

```
DATA_BASE/
├── _MainFigures/
│   ├── Yoda_GsMTx/            main_foldchange.csv + row figure     (Fig 5e-h)
│   ├── ISO_MIC_Piezo_.../     main_iso_foldchange.xlsx + figures   (Fig 5i-l)
│   └── Q1_vDA_dDA_ratio/      main_q1_ratio.csv + figures          (Supp 9b)
├── _SuppFigures/<dataset>/    collected intensity and event panels
└── _SourceData/Source_Data.xlsx
```

---

## 7. Parameters and Methods alignment

| Methods statement | Value | Config key |
|---|---|---|
| Frame interval | 30 s (1 frame = 0.5 min) | `DT_SECONDS` |
| Pre phase shown | 5 min | `PRE_SHOW_MIN` |
| Post analysis window | 20 min | `POST_SHOW_MIN` |
| Cell signal | background-subtracted mean | `CELL_SIGNAL_STAT = "mean"` |
| vDA / dDA / background ROI | `DA_band` / `DA_band_dorsal` / `BG` | `ROI_VDA`, `ROI_DDA`, `ROI_BG` |
| Lifeact correction | median(pre) / median(post), post only | `calcium_build_tables.py` |
| Baseline F₀ | median over the entire pre phase | `calcium_build_tables.py` |
| Normalisation | F/F₀ (ratio) | `calcium_build_tables.py` |
| Event locality gate | 10% over neighbour **or** over the 20th percentile of ±2 frames | `PEAK_DETECTION.rel_peak_frac`, `.prom_frac`, `.prom_win` |
| Event threshold | pre-median + 2 × robust σ | `PEAK_DETECTION.robust_z_k`, `.use_robust_z` |
| Robust σ | 1.4826 × MAD | `robust_sigma()` |
| Refractory period | 1 frame | `PEAK_DETECTION.refractory_frames` |
| Planned comparisons | drug vs its own vehicle | `STATS["planned_pairs"]` |
| Multiplicity | Holm **within** a shared-control family | `plot_Q2_cells_events.py` |
| Genotype × drug | Type-II two-way ANOVA + Tukey HSD | `plot_Q2_cells_events.py` |
| Unit of analysis | the embryo | throughout |

---

## 8. Load-bearing decisions

Each of these was investigated and chosen deliberately.

- **`mean_bgsub` is the primary cell signal.** `max_bgsub` is edge-dirty —
  single-frame vesicle and registration artefacts — and is kept only as an
  internal sanity check. Both columns are always written.
- **F₀ = median of the *entire* pre phase**, not the first N post frames.
- **The event threshold is pre-referenced**, so a drug that raises the whole
  trace raises the event count instead of moving its own threshold.
- **Lifeact M is applied to the post phase only.** Lifeact as a
  frame-by-frame bleaching reference was tried and abandoned for Track A.
- **`embryo_id = batch__pair__genotype__drug__embryo`** — genotype is in the id
  so two genotypes with the same embryo number never collide. Nothing parses it
  positionally; keep it opaque.
- **The analysis window stays config-adjustable.** Output paths and column
  suffixes are *functions* of it (`out_root()`, `window_suffix()`,
  `events_col_per_embryo()`), so changing the window keeps the whole pipeline
  in sync. Do not hardcode it.
- **Multiple-comparison families are defined by the shared control.** Two
  vehicle–drug pairs with different vehicles are two families, not one.

---

## 9. Gotchas and tests

- **`print()` with emoji crashes on a Windows cp1252 console.** Each `main()`
  reconfigures stdout to UTF-8 (guarded) at its start — preserve that, and keep
  new console and figure text ASCII-safe where practical.
- A temp or output directory **must** start with `_`, or `discover_csvs` will
  ingest it as source data.
- The stats workbook needs `openpyxl`; without it `plot_Q2` silently writes
  CSVs instead.

The load-bearing pure functions have regression tests — `holm`,
`robust_sigma`, `safe_div`, event detection, peak duration, two-way ANOVA +
Tukey:

```bash
python tests/test_stats.py     # plain runner, no pytest needed
python -m pytest tests/        # same tests, if pytest is installed
```

There is no lint config. Otherwise "testing" a change means running the
relevant step on real or synthetic data and checking the outputs.

---

## License

MIT — see [LICENSE](../LICENSE) at the repository root.
