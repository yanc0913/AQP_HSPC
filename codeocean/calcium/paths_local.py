# paths_local.py  --  Code Ocean capsule variant
# -*- coding: utf-8 -*-
"""
Capsule copy of paths_local.py. run.sh puts this in place of the template.

Two differences from paths_local.example.py:

  * paths come from the environment, so run.sh sets them once and every script
    in the pipeline agrees;
  * ROOT - the single dataset the per-dataset pipeline works on - is selected
    by CALCIUM_DATASET, so run.sh can loop over all four datasets without
    editing anything.

DATA_BASE points at a WRITABLE directory. The pipeline writes its output tree
beside its inputs (ROOT/_py_out_20min/...), and a Code Ocean data asset mounts
read-only, so run.sh copies the ~21 MB of input CSVs into /results first and
points DATA_BASE there. Everything the run produces then lands under /results
next to the inputs it came from.
"""

import os
from pathlib import Path

DATA_BASE = Path(os.environ.get("CALCIUM_DATA_BASE", "/results"))

DATASET_ROOTS = {
    "30hpf": DATA_BASE / "Analysis_30hpf_Yoda",
    "48hpf": DATA_BASE / "Analysis_48hpf_Yoda",
    "ISO":   DATA_BASE / "Analysis_48hpf_ISO",
    "Piezo": DATA_BASE / "Analysis_48hpf_Piezo",
}

# Which dataset the per-dataset scripts (calcium_build_tables, plot_Q1..Q3,
# calcium_qc) operate on. run.sh sets this in a loop.
ROOT = DATASET_ROOTS[os.environ.get("CALCIUM_DATASET", "30hpf")]

MAIN_FIG_DIRS = {
    "Yoda_GsMTx": DATA_BASE / "_MainFigures" / "Yoda_GsMTx",
    "Q1_ratio":   DATA_BASE / "_MainFigures" / "Q1_vDA_dDA_ratio",

    # The ISO figure exists in two versions differing ONLY in the WT bar's
    # p-value. MAIN is the one in the paper (plain Welch, from the
    # E3-vs-ISO-only run); ALT is Holm-corrected within the three-group family
    # and is reference only.
    "ISO_MIC_Piezo_MAIN": DATA_BASE / "_MainFigures" / "ISO_MIC_Piezo_MAINFIG_ttest",
    "ISO_MIC_Piezo_ALT":  DATA_BASE / "_MainFigures" / "ISO_MIC_Piezo_ALT_holm_reference_only",
}

SUPP_FIG_ROOT = DATA_BASE / "_SuppFigures"

# Figure 2 of the paper comes from the LPM pipeline, which is a separate
# capsule. build_source_data.py needs only its two group-comparison tables
# (~50 KB), shipped in this capsule's data asset so the Source Data workbook
# builds complete. See codeocean/README.md.
LPM_GROUP_COMPARE = Path(
    os.environ.get("CALCIUM_LPM_GROUP_COMPARE", "/data/lpm_group_compare"))
