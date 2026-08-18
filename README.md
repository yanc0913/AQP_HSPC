# AQP_HSPC

Image-analysis code supporting Kondrychyn *et al.*, *"Cellular hydraulics ensures
robust endothelial-to-haematopoietic transition"*.

This repository collects the two independent analysis pipelines used in the study.
They share no code and are set up separately — pick the one you need:

| Directory | Pipeline | What it does |
|---|---|---|
| [`calcium_imaging/`](calcium_imaging/) | Zebrafish GCaMP calcium imaging | Fiji-exported ROI time-series → analysis tables, QC reports and publication figures (F/F₀ traces, event detection, per-condition and cross-dataset comparisons) |
| [`LPM_migration/`](LPM_migration/) | LPM nuclear migration tracking | Timelapse movies → StarDist segmentation → trackpy tracking → wild-type vs `aqp1a.1-/-` migration, width and nuclear-burst comparisons |

Each directory has its own `README.md`, `requirements.txt` and configuration —
start there.

> **Data availability.** This repository contains **code only**. Raw imaging data
> and analysis outputs are not included. Both pipelines read their input paths
> from a local, git-ignored configuration file, so no machine-specific paths are
> committed:
>
> - `calcium_imaging/` — copy `paths_local.example.py` → `paths_local.py`
> - `LPM_migration/` — copy `config.example.yaml` → `config.yaml`

## Environments

The two pipelines have different dependency stacks and are best installed into
**separate environments**:

- **`calcium_imaging/`** — numpy / pandas / scipy / statsmodels / matplotlib (pinned versions).
- **`LPM_migration/`** — additionally StarDist + csbdeep (require TensorFlow), trackpy and scikit-image.

## License

MIT — see [LICENSE](LICENSE).
