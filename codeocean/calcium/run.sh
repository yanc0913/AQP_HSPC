#!/usr/bin/env bash
# Code Ocean capsule entry point -- calcium imaging pipeline.
#
# Reproduces Figure 5, Supplementary Figures 9 and 10, and the Source Data
# workbook, from the Fiji-exported CSVs.
#
# Set the capsule's Reproducible Run command to:
#     bash /code/codeocean/calcium/run.sh
set -euo pipefail

CODE=/code/calcium_imaging
DATA=${CO_DATA:-/data}
RESULTS=${CO_RESULTS:-/results}

echo "=============================================================="
echo " Kondrychyn et al. -- calcium imaging"
echo " code    : $CODE"
echo " data    : $DATA   (read-only)"
echo " results : $RESULTS"
echo "=============================================================="

# The pipeline writes its output tree beside its inputs (ROOT/_py_out_20min),
# and the data asset is mounted read-only, so work from a copy. ~21 MB.
echo
echo "--- copying input CSVs into the results tree ---"
cp -r "$DATA"/Analysis_* "$RESULTS"/
du -sh "$RESULTS"/Analysis_* | sed 's/^/    /'

export CALCIUM_DATA_BASE="$RESULTS"
export CALCIUM_LPM_GROUP_COMPARE="$DATA/lpm_group_compare"

# The capsule variant of paths_local.py reads those two variables.
cp /code/codeocean/calcium/paths_local.py "$CODE/paths_local.py"
cd "$CODE"

# The load-bearing pure functions (Holm, robust sigma, event detection, peak
# duration, two-way ANOVA + Tukey). Fast, and it fails loudly if the
# environment is wrong, before anything expensive runs.
echo
echo "--- regression tests ---"
python tests/test_stats.py

# ---------------------------------------------------------------- per dataset
# calcium_qc.py is an independent sentinel: it reads the raw CSVs directly,
# never modifies data, and never changes pipeline behaviour. Run it first.
for DS in 30hpf 48hpf ISO Piezo; do
    echo
    echo "=============================================================="
    echo " dataset: $DS"
    echo "=============================================================="
    export CALCIUM_DATASET="$DS"
    python calcium_qc.py
    python run_all.py
done
unset CALCIUM_DATASET

# ------------------------------------------------------------ cross-dataset
# The E3-vs-ISO-only subset tree. The main ISO figure contrasts only those two
# groups, so it is drawn from a two-group run; the three-group E3/ISO/BDM
# version stays untouched for the supplementary figure. Must precede the two
# scripts below, which read _py_out_20min_E3_ISO.
echo
echo "--- E3-vs-ISO subset figures ---"
python make_subset_figures.py

echo
echo "--- main figures ---"
python plot_main_q1_ratio.py         # Supp Fig 9b
python plot_main_foldchange.py       # Fig 5e-h
python plot_main_foldchange_iso.py   # Fig 5i-l

echo
echo "--- supplementary figure panels ---"
python collect_supp_figures.py

echo
echo "--- Source Data workbook ---"
python build_source_data.py

echo
echo "=============================================================="
echo " done. Key outputs:"
echo "   $RESULTS/_SourceData/Source_Data.xlsx"
echo "   $RESULTS/_MainFigures/          Fig 5"
echo "   $RESULTS/_SuppFigures/          Supp Figs 9 and 10"
echo "   $RESULTS/Analysis_*/_py_out_20min/tables/    analysis tables"
echo "   $RESULTS/Analysis_*/_qc/                     QC reports"
echo "=============================================================="
