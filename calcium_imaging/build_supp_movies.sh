#!/usr/bin/env bash
# build_supp_movies.sh
#
# The four supplementary movies, with the exact per-panel display ranges that
# were chosen by eye. Re-running this reproduces the delivered files.
#
# Display scaling
# ---------------
# The base range is SHARED across the panels of a set (pooled 1-99.9 percentile
# of non-zero pixels). Four panels were then overridden by hand because they
# read too dark or too bright: lowering the high end brightens a panel, raising
# it dims one. The low end (~110) is the camera offset and is never touched.
#
# These sets are NOT single imaging sessions - each mixes GCamp7a_fli1lifeactmCh,
# GCamp7a_lifeactmCh and GCamp_LAmCh - so absolute brightness was never
# comparable between panels and per-panel scaling is legitimate. The figure does
# the same: its colour bars read low -> high with no numbers. The legend must
# say so, e.g. "display ranges were set per panel for visibility; the colour
# scale is not calibrated between panels". Each movie's _display_settings.txt
# records the numbers actually used.
#
# Lifeact is scaled per panel throughout: it is the structural channel and no
# claim rides on how bright the membrane is.
set -euo pipefail

# Set these for your machine, or export them before running:
#   MOVIE_ROOT  folder holding <set>_group/ subfolders of cropped TIFFs + ROI sets
#   MOVIE_PY    python with tifffile, imageio and Pillow
#   FIJI_LUT    Fiji's "Green Fire Blue.lut" (optional; without it the LUT is
#               approximated and make_condition_movie.py says so)
PY="${MOVIE_PY:-python}"
B="${MOVIE_ROOT:?set MOVIE_ROOT to the folder holding the <set>_group folders}"
OUT="${MOVIE_OUT:-$B/_out}"
COMMON="--lifeact_per_panel"

"$PY" make_condition_movie.py --in_dir "$B/30hpf_Yoda_group" \
    --order "DMSO,Yoda1,E3,GsMTx4" \
    --out "$OUT/Movie_30hpf_Yoda_lifeactPerPanel.mp4" $COMMON \
    --gcamp_range "DMSO=110:165"

"$PY" make_condition_movie.py --in_dir "$B/48hpf_Yoda_group" \
    --order "DMSO,Yoda1,E3,GsMTx4" \
    --out "$OUT/Movie_48hpf_Yoda_lifeactPerPanel.mp4" $COMMON \
    --gcamp_range "GsMTx4=110:170,Yoda1=110:180"

"$PY" make_condition_movie.py --in_dir "$B/48hpf_ISO_group" \
    --order "E3,ISO,BDM,-" \
    --out "$OUT/Movie_48hpf_ISO_lifeactPerPanel.mp4" $COMMON \
    --gcamp_range "E3=109:235"

# No overrides here. Note piezo_E3 has a handful of very bright pixels (99.5
# percentile 171, 99.9 percentile 320); under the shared range that is harmless,
# but do not switch this set to --gcamp_per_panel without pinning it, or the
# outlier sets its high end and the panel goes dark.
# --order values must follow the FILE NAMES; --labels is what is drawn on the
# panel, which is why the piezo conditions are spelled out here.
"$PY" make_condition_movie.py --in_dir "$B/48hpf_piezo_Yoda_group" \
    --order "MIC_E3,MIC_ISO,piezo_E3,piezo_ISO" \
    --labels "MIC E3,MIC ISO,piezo crispant E3,piezo crispant ISO" \
    --out "$OUT/Movie_48hpf_MIC_Piezo_lifeactPerPanel.mp4" $COMMON

echo
echo "done -> $OUT"
