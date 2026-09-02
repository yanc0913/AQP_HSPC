# run_midline_v3.py
# -*- coding: utf-8 -*-
"""
Re-run tracking with the bilateral midline estimator, into a NEW results tree.

Why not run_pipeline.py
-----------------------
run_pipeline.py always re-runs StarDist, and the labels for these 13 movies are
~7 GB that took hours to produce. Nothing about segmentation changed - only the
midline estimator did - so this driver points track_from_labels.py at the
EXISTING labels of another results tree and writes everything else into the
new one.

The source tree is opened read-only - only its label stacks are read - so an
existing set of results is never modified.

Run:  python run_midline_v3.py --config config_midline_v3.yaml \
          --labels_from "<results root that already holds seg/ label stacks>" \
          --python "<python with stardist and trackpy installed>"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("[%s] >>> %s\n" % (datetime.now().isoformat(timespec="seconds"),
                                   " ".join(cmd)))
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=f).returncode
        f.write("[RETURN CODE] %d\n" % rc)
    return rc


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_midline_v3.yaml")
    ap.add_argument("--labels_from", required=True,
                    help="results root holding the existing seg/ label stacks")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only_group", default="")
    ap.add_argument("--only_movie", default="")
    ap.add_argument("--skip_compare", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_root = Path(cfg["project"]["data_root"])
    results_root = Path(cfg["project"]["results_root"])
    labels_root = Path(args.labels_from)
    if results_root.resolve() == labels_root.resolve():
        raise SystemExit("refusing to run: results_root == labels_from, that "
                         "would overwrite the existing results tree")
    results_root.mkdir(parents=True, exist_ok=True)
    suffix = cfg.get("segmentation", {}).get("labels_suffix", "_labels.tif")

    rows = []
    for group, gcfg in cfg["dataset"]["groups"].items():
        if args.only_group and group != args.only_group:
            continue
        files = sorted((data_root / gcfg["input_dir"]).glob(gcfg.get("glob", "*.tif")))
        if args.only_movie:
            files = [p for p in files if p.stem == args.only_movie]
        if not files:
            print("[WARN] no movies for group %s" % group)
            continue

        print("\n=== GROUP %s: %d movies ===" % (group, len(files)))
        for raw in files:
            mid = raw.stem
            labels = labels_root / group / mid / "seg" / (mid + suffix)
            out_dir = results_root / group / mid
            out_dir.mkdir(parents=True, exist_ok=True)
            log = out_dir / "run_log.txt"
            log.write_text("[MIDLINE V3 RERUN] group=%s movie=%s\nlabels=%s\n"
                           "config=%s\n" % (group, mid, labels,
                                            Path(args.config).resolve()),
                           encoding="utf-8")

            if not labels.exists():
                print("  [FAIL] %-44s missing labels" % mid)
                rows.append(dict(group=group, movie=mid, ok=False,
                                 reason="labels not found: %s" % labels))
                continue

            print("  %-44s tracking..." % mid, end="", flush=True)
            rc = run_logged([args.python, "track_from_labels.py",
                             "--config", args.config,
                             "--labels_path", str(labels),
                             "--out_dir", str(out_dir / "trackpy")], log)
            ok = (rc == 0) and (out_dir / "trackpy" / "movie_summary.csv").exists()
            print(" %s" % ("OK" if ok else "FAILED (rc=%d)" % rc))
            rows.append(dict(group=group, movie=mid, ok=ok,
                             reason="" if ok else "track_from_labels rc=%d" % rc))

    st = pd.DataFrame(rows)
    st.to_csv(results_root / "pipeline_status.csv", index=False)
    print("\n%d/%d movies OK" % (int(st.ok.sum()), len(st)))

    if not args.skip_compare and st.ok.all() and len(st):
        print("\n=== group comparison ===")
        rc = run_logged([args.python, "compare_groups.py", "--config", args.config],
                        results_root / "_group_compare_run_log.txt")
        print("compare_groups.py rc=%d -> %s" % (rc, results_root / "_group_compare"))


if __name__ == "__main__":
    main()
