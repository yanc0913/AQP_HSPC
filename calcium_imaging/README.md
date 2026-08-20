# Zebrafish GCaMP calcium-imaging analysis pipeline

Python pipeline for analysing zebrafish GCaMP/Lifeact calcium-imaging data:
it turns Fiji-exported ROI time-series into analysis tables, QC reports, and
publication figures.

This code produces the calcium-imaging analysis and figures in Kondrychyn *et al.*,
*"Cellular hydraulics ensures robust endothelial-to-haematopoietic transition"*.

> **Note.** This repository contains code only. The raw imaging data and all
> analysis outputs are not included. Point the pipeline at your own data by
> editing `paths_local.py` (see Setup).

## Pipeline overview

```
Fiji ROI extraction (Calcium_ROI_selection_v4.ijm, upstream)
        │   exports *_allChannels.csv per movie
        ▼
calcium_build_tables.py   raw CSVs ──► analysis tables
        ▼
plot_Q1 / plot_Q2 / plot_Q3   tables ──► figures (PNG + SVG) and stats
```

ROIs are extracted upstream in Fiji using the included ImageJ macro
(`Calcium_ROI_selection_v4.ijm`, SIFT registration with interpolation). Drug /
condition and genotype are parsed from **filenames**, not folders.

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

Then create your local paths file (this file is git-ignored, so your machine
paths never get committed):

```bash
cp paths_local.example.py paths_local.py
```

Edit `DATA_BASE` in `paths_local.py` to point at the folder that holds your
per-dataset analysis directories. Everything else is derived from it.

## Running

```bash
python run_all.py            # full pipeline: build tables, then Q1 + Q2 + Q3 plots
python run_all.py build      # rebuild tables only
python run_all.py plots      # q1 + q2 + q3 (assumes tables already built)
python run_all.py q2         # a single step (build / q1 / q2 / q3)
python calcium_qc.py         # independent QC sentinel (run BEFORE build)
```

Cross-dataset main figures (assemble fold-change panels across several datasets):

```bash
python plot_main_foldchange.py       # Yoda1 / GsMTx4, 30 vs 48 hpf
python plot_main_foldchange_iso.py   # ISO / MIC / Piezo crispant
```

## Configuration

- **`paths_local.py`** — machine-specific data paths (not committed).
- **`calcium_config.py`** — all analysis parameters: condition/genotype aliases,
  peak-detection settings, analysis window, plotting and stats options, QC
  thresholds. Edit parameters here; the plot scripts consume it.

## Tests

The load-bearing pure functions (multiple-comparison correction, robust sigma,
event detection, peak duration, two-way ANOVA + Tukey) have regression tests:

```bash
python tests/test_stats.py     # plain runner, no pytest needed
python -m pytest tests/        # same tests, if pytest is installed
```

## License

MIT — see [LICENSE](../LICENSE) at the repository root.
