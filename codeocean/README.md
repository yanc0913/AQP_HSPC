# Code Ocean capsules

Scaffolding for sharing this code as executable [Code Ocean](https://codeocean.com)
capsules during peer review.

Nothing here changes the pipelines. Each capsule reuses the repository as-is
and supplies only what a capsule needs: an entry script, an environment build
step, and paths pointing at `/data` and `/results`.

## Two capsules, not one

A capsule has a single environment, and the two pipelines do not share a
dependency stack — `LPM_migration/` needs TensorFlow and StarDist, which pin
numpy to 1.x, while `calcium_imaging/` runs on numpy 2.x. Forcing them
together would mean shipping an environment neither was validated on.

| | Capsule A — calcium | Capsule B — LPM migration |
|---|---|---|
| Reproduces | Fig 5, Supp Figs 9–10, `Source_Data.xlsx` | Fig 2c–h, midline QC |
| Run command | `bash /code/codeocean/calcium/run.sh` | `bash /code/codeocean/lpm/run.sh` |
| Environment | Python 3.10, `codeocean/calcium/postInstall` | Python 3.11 + TensorFlow 2.12, `codeocean/lpm/postInstall` |
| Data asset | ~21 MB | ~2.4 GB (default) or ~4.7 GB (with raw movies) |
| GPU | no | only for `LPM_MODE=full` |

Both capsules import this whole repository as `/code`. Each one simply ignores
the other pipeline's directory.

---

## Capsule A — calcium

### Data asset layout

```
calcium-data/
├── Analysis_30hpf_Yoda/          Fiji-exported *_allChannels.csv
│   └── <batch>/<condition>/*.csv     e.g. exp1/Yoda/Aligned_..._allChannels.csv
├── Analysis_48hpf_Yoda/
├── Analysis_48hpf_ISO/
├── Analysis_48hpf_Piezo/
└── lpm_group_compare/
    ├── all_movies_summary.csv
    └── aligned_time_series_per_movie.csv
```

217 CSV files, ~21 MB total.

Copy only the raw CSVs. Do **not** include any directory whose name begins
with `_` — those are output trees from previous runs (`_py_out_20min/`,
`_qc/`, `_MainFigures/`), and the capsule regenerates all of them. The
pipeline ignores underscore-prefixed directories when it discovers input
files, so including them would not corrupt a run, only inflate the asset.

`lpm_group_compare/` holds two small tables from Capsule B. Figure 2 comes
from the LPM pipeline, but its numbers belong in the same Source Data
workbook, so those two files (~50 KB) travel with this capsule to let
`build_source_data.py` produce the complete workbook. They are outputs of
Capsule B and can be regenerated there.

### Why the run copies its input

The pipeline writes each dataset's output tree beside that dataset's inputs
(`Analysis_*/\_py_out_20min/`), and a data asset mounts read-only. `run.sh`
therefore copies the 21 MB of CSVs into `/results` first and works there, so
every output lands next to the input it came from.

### What the run does

1. `tests/test_stats.py` — the regression tests for the load-bearing pure
   functions, so a broken environment fails immediately.
2. For each of the four datasets: `calcium_qc.py` (an independent sentinel
   that reads the raw CSVs and never modifies anything), then `run_all.py`
   (build tables → Q1, Q2, Q3 figures).
3. `make_subset_figures.py` — the E3-vs-ISO-only tree. The main ISO figure
   contrasts just those two groups, so it is drawn from a two-group run while
   the three-group E3/ISO/BDM version stays intact for the supplementary
   figure.
4. The cross-dataset assemblers, then `build_source_data.py`.

---

## Capsule B — LPM migration

### Data asset layout

```
lpm-data/
├── labels/                       StarDist label stacks (~2.4 GB)
│   ├── WT/<movie>/seg/*.tif
│   └── MUT/<movie>/seg/*.tif
└── movies/                       raw timelapses (~2.3 GB, optional)
    ├── wildtype_posterior/*.tif
    └── aqp1a1_mutant_posterior/*.tif
```

13 movies: 7 wild type, 6 mutant.

The `labels/` tree must mirror the pipeline's own results layout —
`<group>/<movie>/seg/` — because that is what `--labels_from` reads.

### Two modes

`run.sh` reads `LPM_MODE`:

- **`labels`** (default) — starts from the stored label stacks. Reproduces the
  midline fit, tracking, burst detection and the group comparison: everything
  Figure 2 is made of. Minutes, CPU only. Only `labels/` is needed.
- **`full`** — re-runs StarDist segmentation from the raw movies first. Hours,
  and wants a GPU. Needs `movies/`.

Ship `movies/` as well if you want reviewers to be able to verify the
segmentation; drop it to halve the asset if not.

### Output paths

`results_root` is `/results/tracking`, one level below `/results`, because
`midline_v3_check.py` writes its report to `<results_root>/../_midline_v3_check`
and that has to land inside the results directory.

`run.sh` builds the runtime config from the committed `*.example.yaml` by
substituting only the two paths, and writes the result to
`/results/config_used.yaml`. Every analysis parameter therefore comes from the
repository, and the exact configuration that ran is preserved with the output.

---

## What these capsules do not cover

The calcium pipeline starts **after** Fiji. Two upstream steps are interactive
and cannot run in a capsule:

1. registration — SIFT / linear stack alignment, which produces the `Aligned_`
   filename prefix;
2. ROI selection and measurement export, via the macro
   `calcium_imaging/Calcium_ROI_selection_v4.ijm`, which performs no
   registration itself.

The macro is in the repository so the ROI and measurement logic can be
inspected. The capsule reproduces everything downstream of its
`*_allChannels.csv` output.

Figure 2j (endothelial cell number in the dorsal aorta) is not from either
pipeline and is not part of these capsules.
