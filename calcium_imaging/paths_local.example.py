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
    "Q1_ratio":      DATA_BASE / "_MainFigures" / "Q1_vDA_dDA_ratio",

    # --- The ISO figure exists in TWO versions. They differ ONLY in the WT
    # --- bar's p-value; the folder names say which is which.
    # MAIN: the version that goes in the paper. WT bar = plain Welch t-test,
    #       read from the E3-vs-ISO-only run (that panel asks only about ISO).
    "ISO_MIC_Piezo_MAIN": DATA_BASE / "_MainFigures" / "ISO_MIC_Piezo_MAINFIG_ttest",
    # ALT: reference only, NOT for the paper. WT bar = Holm-corrected within the
    #      E3 family of the three-group E3/ISO/BDM run.
    "ISO_MIC_Piezo_ALT":  DATA_BASE / "_MainFigures" / "ISO_MIC_Piezo_ALT_holm_reference_only",
}

# Root for assembled supplementary figures (collect_supp_figures.py gathers the
# per-dataset panels here, mirroring the _MainFigures layout).
SUPP_FIG_ROOT = DATA_BASE / "_SuppFigures"

# Group-comparison folder of the LPM cell-migration pipeline (a separate
# repo). build_source_data.py reads Fig 2 from here.
LPM_GROUP_COMPARE = Path("/path/to/LPM/Analysis/results") / "_group_compare"
