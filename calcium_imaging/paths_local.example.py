# paths_local.example.py
# -*- coding: utf-8 -*-
"""
Template for machine-specific data paths.

SETUP:
  1. Copy this file to  paths_local.py
  2. Edit DATA_BASE below to point at the folder that holds your per-dataset
     analysis directories (each containing a "_py_out_<N>min" output tree).

paths_local.py is git-ignored, so your real paths never get committed.
Raw imaging data and analysis outputs are NOT part of this repository.

Expected layout under DATA_BASE:
    <DATA_BASE>/
        Analysis_30hpf_Yoda/
            _py_out_20min/tables/Q2_embryo_summary_cells.csv
        Analysis_48hpf_Yoda/
        Analysis_48hpf_ISO/
        Analysis_48hpf_Piezo/
        _MainFigures/
            Yoda_GsMTx/
            ISO_MIC_Piezo/
"""

from pathlib import Path

# EDIT THIS to your own data directory.
DATA_BASE = Path(r"/path/to/your/calcium-imaging-data")

# Primary dataset root used by the single-dataset pipeline (calcium_config.ROOT).
ROOT = DATA_BASE / "Analysis_30hpf_Yoda"

# Per-dataset analysis roots used by the cross-dataset main-figure scripts.
DATASET_ROOTS = {
    "30hpf": DATA_BASE / "Analysis_30hpf_Yoda",
    "48hpf": DATA_BASE / "Analysis_48hpf_Yoda",
    "ISO":   DATA_BASE / "Analysis_48hpf_ISO",
    "Piezo": DATA_BASE / "Analysis_48hpf_Piezo",
}

# Output directories for the assembled main figures.
MAIN_FIG_DIRS = {
    "Yoda_GsMTx":    DATA_BASE / "_MainFigures" / "Yoda_GsMTx",
    "ISO_MIC_Piezo": DATA_BASE / "_MainFigures" / "ISO_MIC_Piezo",
}
