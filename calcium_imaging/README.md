# Zebrafish GCaMP calcium-imaging analysis pipeline

Turns Fiji-exported ROI time-series into analysis tables, QC reports,
publication figures, supplementary movies and a Source Data workbook.

This code produces the calcium-imaging analysis and figures in Kondrychyn
*et al.*, *"Cellular hydraulics ensures robust endothelial-to-haematopoietic
transition"* (Figure 5, Supplementary Figures 9 and 10).

> **Code only.** Raw imaging data and analysis outputs are not in this
> repository. Point the pipeline at your own data by copying
> `paths_local.example.py` → `paths_local.py` (git-ignored).

---

## 1. What happens upstream, in Fiji

Two **separate** Fiji steps run before any Python:

1. **Registration** — SIFT / linear stack alignment with interpolation. This is
   where the `Aligned_` filename prefix comes from.
2. **ROI selection and measurement** — the included macro
   `Calcium_ROI_selection_v4.ijm`. It draws and measures ROIs and exports one
   `*_allChannels.csv` per movie. **It does no registration.**

ROIs are named by what they are, and the Python side depends on those names:

| ROI name | Meaning |
|---|---|
| `DA_band` | ventral dorsal-aorta band ("vDA") — the region of interest |
| `DA_band_dorsal` | dorsal band ("dDA") — the ratio reference |
| `BG` | background, subtracted from every other ROI |
| `flat_01`, `flat_02`, … | single elongated cells |
| `round_01`, `round_02`, … | single round cells |

**Condition and genotype are parsed from the FILENAME, not the folder.** Folders
are organisational only. The token lists live in `calcium_config.py`
(`CONDITION_ALIASES`, `GENOTYPE_ALIASES`); a movie with no genotype token
becomes `WT`, so experiments without genotypes behave exactly as before.

---

## 2. Data layout

```
DATA_BASE/
  Analysis_<experiment>/          <- cfg.ROOT points at one of these
      <batch>/<drug>/*_allChannels.csv
      _py_out_20min/              <- created by the pipeline
          tables/  plots_png/  plots_svg/
      _qc/                        <- created by calcium_qc.py
  _MainFigures/                   <- cross-dataset figures
  _SuppFigures/                   <- collected supplementary panels
  _SourceData/                    <- Source_Data.xlsx
```

**`discover_csvs` skips any directory whose name starts with `_`.** Every
output, temp and QC tree is underscore-prefixed for exactly that reason. A new
working directory that is *not* underscore-prefixed will be ingested as source
data.

---

## 3. Setup

Tested on **Python 3.10.19** (conda). Nothing requires 3.11.

```bash
pip install -r requirements.txt
cp paths_local.example.py paths_local.py     # then edit DATA_BASE
```

`openpyxl` is needed for the stats workbook and the Source Data builder;
`plot_Q2` falls back to CSVs without it.

Two optional extras, only for the movie builder:

- `tifffile`, `imageio`, `Pillow`
- Fiji's `Green Fire Blue.lut`, so movie colours match the figure colour bar.
  Point `FIJI_LUT` at it, or pass `--lut`. Without it a built-in approximation
  is used and the script says so.

---

## 4. Running

### The per-dataset pipeline

```bash
python calcium_qc.py         # independent QC sentinel - run BEFORE the build
python run_all.py            # build tables, then Q1 + Q2 + Q3 plots
python run_all.py build      # tables only
python run_all.py plots      # q1 + q2 + q3, assuming tables exist
python run_all.py q2         # one step: build / q1 / q2 / q3
```

`calcium_qc.py` reads the raw Fiji CSVs directly, computes per-ROI quality
metrics (FAIL / FLAG_HIGH / FLAG_LOW) and writes `_qc/`. It **never modifies
data** and never changes pipeline behaviour.

To run on a subset without touching your real outputs, point the config at a
throwaway tree in a script — and do **not** start its name with `_`:

```python
import calcium_config as cfg
cfg.ROOT = Path("tmp_root")
cfg.OUT_ROOT_BASE = cfg.ROOT / "_py_out"
main()
```

### Cross-dataset figures

These read the built `Q2_embryo_summary_cells.csv` from several datasets at
once. Set `DATASET_ROOTS` and `MAIN_FIG_DIRS` in `paths_local.py` first.

```bash
python plot_main_foldchange.py       # Yoda1 / GsMTx4, 30 vs 48 hpf   (Fig 5e-h)
python plot_main_foldchange_iso.py   # ISO / MIC / piezo crispant     (Fig 5i-l)
python plot_main_q1_ratio.py         # vDA/dDA ratio, 30 vs 48 hpf    (Supp 9b)
```

### Collecting and deriving

```bash
python make_subset_figures.py     # rebuild one dataset restricted to a subset
                                  # of conditions, into a sibling output tree
python collect_supp_figures.py    # copy intensity + event panels into _SuppFigures/
python build_source_data.py       # assemble Source_Data.xlsx
```

`make_subset_figures.py` redirects `cfg.out_root()` so tables, figures **and**
the stats workbook move together — that is what produced the E3-vs-ISO
two-group run used by the main figure.

### Supplementary movies

```bash
export MOVIE_ROOT="/path/to/RepresentativeMovies/Movies"
export MOVIE_PY="/path/to/python"           # needs tifffile, imageio, Pillow
export FIJI_LUT="/path/to/Green Fire Blue.lut"
bash build_supp_movies.sh
```

`build_supp_movies.sh` records the exact per-panel display ranges used for the
published movies, so re-running reproduces them. For one movie by hand:

```bash
python make_condition_movie.py \
    --in_dir "<folder of cropped TIFFs + ROI sets>" \
    --order "DMSO,Yoda1,E3,GsMTx4" \
    --labels "DMSO,Yoda1,E3 buffer,GsMTx4" \
    --out movie.mp4 --lifeact_per_panel
```

Each panel needs a cropped 2-channel TIFF (`TCYX`) and an ImageJ ROI set saved
on that **same cropped** image, containing ROIs named `elongated`, `round` and
optionally `VDA`. Panels are matched to files by the `--order` labels; `-`
leaves a cell of the grid blank.

---

## 5. Architecture

Single source of truth → core builder → plot consumers. **Edit parameters only
in the config.**

| File | Role |
|---|---|
| `calcium_config.py` | Every tunable parameter. Output paths and column suffixes are *functions* derived from the analysis window, so changing `W` keeps the whole pipeline in sync. Frequently-edited items are at the top. |
| `calcium_build_tables.py` | The data core: Fiji CSVs → long table → band traces (vDA/dDA/BG) → Lifeact `M` correction → F/F₀ → event detection → per-cell and per-embryo tables. Writes `Q1_*`, `Q2_*`, `Q3_*` and a `_config_snapshot.py` of the parameters used. |
| `plot_Q1_vDA_dDA_ratio.py` | vDA/dDA ratio figures |
| `plot_Q2_cells_events.py` | The largest consumer: trace panels, boxplots, and **all the statistics** |
| `plot_Q3_vDA_trace.py` | vDA trace figures |
| `calcium_qc.py` | Independent QC sentinel, reads raw CSVs, never writes into the analysis |
| `run_all.py` | Orchestrator; runs each step in a subprocess with the active interpreter |
| `plot_style.py` | Shared matplotlib style |
| `make_condition_movie.py` | 2×2 condition movies: merge over Green Fire Blue, ROI boxes, scale bar, timestamp |
| `build_source_data.py` | Source Data workbook, one sheet per figure |

`analyze_piezo_*quick.py` are deliberately throwaway single-experiment scripts.
They **import** the verified primitives from `calcium_build_tables` rather than
re-implementing them.

### Two analysis tracks

- **Track A — pre/post** (the main pipeline). Each recording has a `pre` phase
  and a `post` phase separated by a remount gap. F/F₀ and event thresholds are
  referenced to the pre phase.
- **Track B — single segment** (`analyze_piezo_baseline_quick.py`). One 20-min
  segment, self-referenced threshold. Used for piezo-crispant baseline data.

---

## 6. Statistics

All tests treat **the embryo as the unit of analysis**: cells are rolled up to
embryo means first. Running tests on individual cells would be
pseudoreplication.

**Planned comparisons.** `cfg.STATS["planned_pairs"]` is a *superset* of
`(vehicle, drug)` pairs across all experiments; each run automatically uses only
the pairs whose both conditions are present, so it never needs editing per
dataset. For matched pairs the boxplots draw only those Welch-t brackets — raw
p, no Holm, no omnibus — because the two vehicle-drug pairs have different
vehicles and are therefore separate families. If none of the pairs are present,
it falls back to omnibus + all-pairwise Holm with a one-time warning.

**Genotype × drug.** When two genotypes share a `pair_id`, the layout switches
to genotype × drug with a Type-II two-way ANOVA (main effects, interaction,
partial η²) and Tukey HSD post-hoc. This is independent of `planned_pairs`.

**Stats workbook.** To keep figures uncluttered the genotype × drug panel prints
only a `2-way ANOVA (II)` header. *Every* p-value goes to
`tables/Q2_stats_pvalues.xlsx`:

| Sheet | Contents |
|---|---|
| `twoway_anova` | two-way ANOVA table |
| `pairwise` | all pairwise comparisons, with the **exact** p even where the figure floors it to `p<0.0001` |
| `n_cells` | genotype × drug sample sizes: n embryos and n cells, split flat/round |
| `foldchange` | the fold-change numbers behind the main figures |

**Fold change.** Each drug embryo is normalised by the **mean of its own
vehicle**, so vehicles collapse to the baseline. Drug-vs-vehicle p is the
two-group Welch on **raw** values — dividing by the vehicle mean discards the
control's spread, so a one-sample test against 1 would be anti-conservative.
Fold change is unstable when the vehicle baseline is near zero, which is why
**event rate is reported as a difference, not a ratio**: control rates reach 0.

---

## 7. Load-bearing decisions — do not silently undo

Each of these was investigated and chosen deliberately.

- **`mean_bgsub` is the primary cell signal** (`cfg.CELL_SIGNAL_STAT = "mean"`).
  `max_bgsub` is edge-dirty — single-frame vesicle and registration artefacts —
  and is kept only as an internal sanity check. Both columns are always written.
- **F₀ = median of the *entire* pre phase**, not the first N post frames. The
  F/F₀ pre-median is ≈1 by construction and no post frame enters the baseline.
- **Event threshold is pre-referenced**: `pre_median + k·σ_robust(pre)` with
  `k = 2.0` and `σ_robust = 1.4826·MAD` (falling back to std when MAD = 0).
  Legacy self-referenced columns are still written but are not what is plotted.
- **Lifeact `M` correction**: `M = median(Lifeact_vDA_pre) / median(post)`,
  applied to post only as `post × M`. Lifeact-as-bleaching-reference was tried
  and abandoned for Track A.
- **`embryo_id = batch__pair__genotype__drug__embryo`** — genotype is in the id
  so two genotypes with the same embryo number never collide. Nothing parses it
  positionally; keep it opaque.
- **The analysis window `W` stays config-adjustable** (`POST_ANALYSIS_MIN` /
  `POST_SHOW_MIN`, default 20 min). Do not hardcode it. `DT_SECONDS = 30`, so
  one frame = 0.5 min.

---

## 8. Gotchas

- **`print()` with emoji crashes on a Windows cp1252 console.** Each `main()`
  reconfigures stdout to UTF-8 (guarded) at its start — preserve that, and keep
  new console and figure text ASCII-safe where practical.
- A temp or output directory **must** start with `_`, or `discover_csvs` will
  ingest it as source data.
- The stats workbook needs `openpyxl`; without it `plot_Q2` silently writes CSVs
  instead.

---

## 9. Tests

The load-bearing pure functions have regression tests: `holm`, `robust_sigma`,
`safe_div`, event detection, peak duration, two-way ANOVA + Tukey.

```bash
python tests/test_stats.py     # plain runner, no pytest needed
python -m pytest tests/        # same tests, if pytest is installed
```

There is no lint config. Otherwise "testing" a change means running the relevant
step on real or synthetic data and checking the outputs.

---

## License

MIT — see [LICENSE](../LICENSE) at the repository root.
