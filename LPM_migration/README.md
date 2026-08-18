# LPM migration tracking pipeline

Segmentation → tracking → group-comparison pipeline for zebrafish **lateral plate
mesoderm (LPM)** nuclei in timelapse movies. It compares wild-type against
`aqp1a.1-/-` mutants over the 13–17 hpf window.

> **Note.** This directory contains code only. Raw movies and analysis outputs are
> not included — point the pipeline at your own data via `config.yaml` (see Setup).

## Pipeline overview

```
raw timelapse .tif
       │
       ▼
segment_stardist.py     StarDist2D nuclear segmentation ──► <movie>_labels.tif
       ▼
track_from_labels.py    label stack ──► spots ──► ROI + midline filter ──►
                        trackpy linking ──► per-frame metrics, burst events
       ▼
compare_groups.py       per-movie summaries ──► WT vs mutant figures + stats
```

`run_pipeline.py` orchestrates segmentation + tracking for every movie in every
group, then runs the group comparison.

### What each script does

| Script | Role |
|---|---|
| `run_pipeline.py` | Orchestrator: loops groups/movies, runs each stage, writes `pipeline_status.csv` |
| `segment_stardist.py` | StarDist2D segmentation of each frame → `(T,Y,X)` uint16 label stack |
| `track_from_labels.py` | Core: region properties → ROI + midline (notochord) removal → area/density filters → trackpy linking → speed, width, fragment and burst metrics |
| `compare_groups.py` | Aggregates per-movie summaries → WT vs mutant boxplots and time-series (mean±SD), Welch's t-test |
| `make_qc_stack_from_labels.py` | Builds a 5-channel ImageJ QC hyperstack (labels / spots / tracked / ROI / trajectories) |
| `visualize_tracks_on_labels.py` | Renders a track-overlay movie for visual inspection |

## Setup

Requires Python 3.9+. StarDist needs TensorFlow — install a build matching your
platform (and GPU/CUDA, if used) first.

```bash
pip install -r requirements.txt
```

Then create your local config (git-ignored, so your paths are never committed):

```bash
cp config.example.yaml config.yaml
```

Edit `project.data_root` and `project.results_root` in `config.yaml` to point at
your own data. Input movies are expected under `data_root/<group input_dir>/`.

## Running

```bash
python run_pipeline.py --config config.yaml
```

Useful flags:

```bash
python run_pipeline.py --config config.yaml --only_group WT
python run_pipeline.py --config config.yaml --only_movie 20250616-wt_13-17hpf
python run_pipeline.py --config config.yaml --skip_compare
```

Individual stages can also be run directly (see each script's `--help`).

## Configuration

All parameters live in `config.yaml`: segmentation thresholds, ROI estimation,
midline (TLS) fit and band width, tracking (pixel size, frame interval, area
limits, search range, memory), gap closing, burst detection, and the comparison
window. Movie filenames of the form `..._13-17hpf...` are parsed to align each
movie on an absolute hpf axis.

## Analysis notes

- **The movie is the unit of analysis** — per-movie summary values are compared
  between groups (Welch's t-test), not individual nuclei.
- **Midline removal** excludes notochord-associated objects within a fitted band,
  applied identically to both genotypes across all frames.
- Speed skips the first frame by default (no preceding frame to difference).
