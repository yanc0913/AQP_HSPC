#!/usr/bin/env bash
# Code Ocean capsule entry point -- LPM nuclear migration pipeline.
#
# Reproduces Figure 2c-h, plus the midline-fit QC figures.
#
# Set the capsule's Reproducible Run command to:
#     bash /code/codeocean/lpm/run.sh
#
# Two modes, selected by LPM_MODE:
#
#   labels  (default)  Start from the stored StarDist label stacks in the data
#                      asset. Reproduces the midline fit, tracking, burst
#                      detection and the group comparison -- everything the
#                      figure is made of. Minutes, CPU only.
#
#   full               Re-run StarDist segmentation from the raw movies first,
#                      then everything above. Hours, and wants a GPU. Use this
#                      to verify the segmentation itself.
set -euo pipefail

CODE=/code/LPM_migration
DATA=${CO_DATA:-/data}
RESULTS=${CO_RESULTS:-/results}
MODE=${LPM_MODE:-labels}

# results_root sits one level down: midline_v3_check.py writes its report to
# <results_root>/../_midline_v3_check, which must land inside /results.
TRACKING="$RESULTS/tracking"

echo "=============================================================="
echo " Kondrychyn et al. -- LPM nuclear migration"
echo " mode    : $MODE"
echo " code    : $CODE"
echo " data    : $DATA   (read-only)"
echo " results : $RESULTS"
echo "=============================================================="

cd "$CODE"
mkdir -p "$TRACKING"

# Build the runtime config from the committed example, changing only the two
# paths. Every analysis parameter therefore comes from the repository, and the
# exact config that ran is written into the results for the record.
CFG="$RESULTS/config_used.yaml"
TEMPLATE=$([ "$MODE" = full ] && echo config.example.yaml \
                              || echo config_midline_v3.example.yaml)

python - "$TEMPLATE" "$CFG" "$DATA/movies" "$TRACKING" <<'PY'
import sys, yaml
template, out, data_root, results_root = sys.argv[1:5]
cfg = yaml.safe_load(open(template))
cfg["project"]["data_root"] = data_root
cfg["project"]["results_root"] = results_root
with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print("wrote %s (from %s)" % (out, template))
PY

echo
if [ "$MODE" = full ]; then
    echo "--- full pipeline: StarDist segmentation, then tracking ---"
    python run_pipeline.py --config "$CFG"
else
    echo "--- tracking from stored labels ---"
    python run_midline_v3.py --config "$CFG" --labels_from "$DATA/labels"
fi

# Read-only diagnostic. Three panels per movie: the fitted midline over the
# nuclei, the density across it, and the angle sweep. This is the check that
# the exclusion band sits in the gap between the two bilateral nuclear bands
# rather than on one of them.
echo
echo "--- midline fit report ---"
python midline_v3_check.py --config "$CFG"

echo
echo "=============================================================="
echo " done. Key outputs:"
echo "   $TRACKING/_group_compare/           Fig 2c-h"
echo "     boxplot_speed.*                     2d"
echo "     boxplot_burst_track_ratio.*         2g"
echo "     speed_vs_time_mean_sd.*             2c"
echo "     area_vs_time_mean_sd.*              2e"
echo "     bursts_vs_time_mean_sd.*            2f"
echo "     n_tracked_vs_time_mean_sd.*         2h"
echo "     all_movies_summary.csv              per-embryo values"
echo "     aligned_time_series_per_movie.csv   per-embryo time courses"
echo "   $RESULTS/_midline_v3_check/         midline fit, one figure per movie"
echo "   $RESULTS/config_used.yaml           parameters this run used"
echo "=============================================================="
