# AQP_HSPC

Image-analysis code supporting Kondrychyn *et al.*, *"Osmo-hydraulic volume
regulation ensures robust endothelial-to-haematopoietic transition"*.

Two independent pipelines. They share no code, have different dependency
stacks, and are set up separately — pick the one you need.

| Directory | Pipeline | Produces |
|---|---|---|
| [`calcium_imaging/`](calcium_imaging/) | Zebrafish GCaMP calcium imaging | Figure 5, Supplementary Figures 9 and 10, the supplementary movies, and the Source Data workbook |
| [`LPM_migration/`](LPM_migration/) | LPM nuclear migration tracking | Figure 2c–h |

Each directory has its own `README.md`, `requirements.txt` and configuration —
**start there.** This page is only orientation.

Both pipeline READMEs are structured the same way: what goes in, what the
method actually does (with the definitions and formulae), what lands on disk
file by file, and a **Parameters and Methods alignment** table mapping every
number in the paper's Methods section to the configuration key that sets it.

---

## What each pipeline actually does

**`calcium_imaging/`** starts *after* Fiji. Upstream, each movie is registered
(SIFT / linear stack alignment) and then ROIs are drawn and measured with the
included macro `Calcium_ROI_selection_v4.ijm`, which exports one
`*_allChannels.csv` per movie. The Python side reads those CSVs and produces
F/F₀ traces, single-frame event detection, per-cell and per-embryo tables, the
statistics, and the figures.

**`LPM_migration/`** starts from raw timelapse TIFFs: StarDist2D segmentation,
then tracking with trackpy, then a wild-type vs mutant comparison of migration
speed, nuclear area, nuclear-fragmentation events and cell number.

Both treat **the embryo as the unit of analysis** — cells or nuclei are rolled
up to an embryo value before any statistical test. That is the single most
important thing to preserve if you adapt this code: testing individual cells
would be pseudoreplication.

---

## Getting started

```bash
git clone https://github.com/yanc0913/AQP_HSPC.git
cd AQP_HSPC
```

Then pick a pipeline. In both cases the first step is the same shape: copy the
example configuration, edit the paths, run.

```bash
# calcium
cd calcium_imaging
pip install -r requirements.txt
cp paths_local.example.py paths_local.py     # edit DATA_BASE
python run_all.py

# LPM migration
cd LPM_migration
pip install -r requirements.txt              # TensorFlow first, for StarDist
cp config.example.yaml config.yaml           # edit data_root and results_root
python run_pipeline.py --config config.yaml
```

---

## Environments

Install the two into **separate environments** — they do not overlap:

| | Needs |
|---|---|
| `calcium_imaging/` | numpy, pandas, scipy, statsmodels, matplotlib, openpyxl. Tested on Python 3.10.19. |
| `LPM_migration/` | additionally StarDist + csbdeep (which require TensorFlow), trackpy, scikit-image, tifffile |

The supplementary-movie builder in `calcium_imaging/` also wants `tifffile`,
`imageio` and `Pillow`, and — to match the figure colour bar exactly — Fiji's
`Green Fire Blue.lut`. Without the LUT it falls back to an approximation and
says so.

Exact versions that produced the published results are in
[`versions_actual.txt`](versions_actual.txt).

---

## Data and paths

This repository is **code only**. Raw imaging data and analysis outputs are not
included, and no machine-specific path is committed: both pipelines read their
paths from a local, git-ignored configuration file.

- `calcium_imaging/` — copy `paths_local.example.py` → `paths_local.py`
- `LPM_migration/` — copy `config.example.yaml` → `config.yaml`
  (and `config_midline_v3.example.yaml` if you keep a second results tree)

Nothing else needs editing to point either pipeline at your own data.

---

## If you are adapting rather than reproducing

A few decisions are load-bearing. Each README has a section spelling them out;
the short version:

- **The embryo, not the cell, is the unit of analysis.**
- **Calcium:** F₀ is the median of the *entire* pre-drug phase, and the event
  threshold is referenced to that same pre phase. Event rate is reported as a
  *difference* from vehicle rather than a fold change, because control rates
  reach zero and a ratio diverges.
- **LPM:** the midline is located as the gap between the two bilateral nuclear
  bands, and the exclusion band sits there. Its width is an analysis choice, not
  a display setting — the midline is populated by converging cells, so widening
  it removes part of the population being measured. Check the fit with
  `midline_v3_check.py` before trusting a run.
- **LPM:** movies are placed on an axis of absolute developmental time (hpf),
  not stretched to a common frame count.
- Multiple-comparison families are defined by the **shared control**: two
  vehicle–drug pairs with different vehicles are two families, not one, and are
  not cross-corrected.

---

## License

MIT — see [LICENSE](LICENSE).
