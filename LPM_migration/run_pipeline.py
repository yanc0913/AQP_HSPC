import argparse
import subprocess
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] >>> " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=f)
        f.write(f"[RETURN CODE] {p.returncode}\n")
        f.flush()
        return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--python", default="python", help="Python executable (python or full path)")
    ap.add_argument("--only_group", default="", help="Run only one group name, e.g. WT")
    ap.add_argument("--only_movie", default="", help="Run only one movie stem, e.g. 20250616-wt_13-17hpf")
    ap.add_argument("--skip_compare", action="store_true", help="Skip compare_groups.py")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    data_root = Path(cfg["project"]["data_root"])
    results_root = Path(cfg["project"]["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)

    labels_suffix = cfg.get("segmentation", {}).get("labels_suffix", "_labels.tif")
    groups = cfg["dataset"]["groups"]

    status_rows = []

    for group, gcfg in groups.items():
        if args.only_group and group != args.only_group:
            continue

        in_dir = data_root / gcfg["input_dir"]
        glob_pat = gcfg.get("glob", "*.tif")
        files = sorted(in_dir.glob(glob_pat))

        if args.only_movie:
            files = [p for p in files if p.stem == args.only_movie]

        if not files:
            print(f"[WARN] No files found for group={group} in {in_dir} (glob={glob_pat})")
            continue

        print(f"\n=== GROUP {group}: {len(files)} movies ===")

        for raw in files:
            movie_id = raw.stem
            out_dir = results_root / group / movie_id
            out_dir.mkdir(parents=True, exist_ok=True)

            log_file = out_dir / "run_log.txt"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"[PIPELINE LOG] group={group}, movie={movie_id}\n")
                f.write(f"raw_file={raw}\n")
                f.write(f"config={Path(args.config).resolve()}\n")

            print(f"\n--- [{group}] {movie_id} ---")
            ok = True
            fail_step = ""
            fail_reason = ""

            # Step 1: segmentation
            seg_cmd = [
                args.python, "segment_stardist.py",
                "--config", str(Path(args.config)),
                "--input", str(raw),
                "--out_dir", str(out_dir),
            ]
            rc = run_logged(seg_cmd, log_file)
            if rc != 0:
                ok = False
                fail_step = "segmentation"
                fail_reason = f"segment_stardist.py return code {rc}"

            labels_path = out_dir / "seg" / f"{movie_id}{labels_suffix}"
            if ok and (not labels_path.exists()):
                ok = False
                fail_step = "segmentation"
                fail_reason = f"labels file not found: {labels_path}"

            # Step 2: tracking
            if ok:
                track_cmd = [
                    args.python, "track_from_labels.py",
                    "--config", str(Path(args.config)),
                    "--labels_path", str(labels_path),
                    "--out_dir", str(out_dir / "trackpy"),
                ]
                rc = run_logged(track_cmd, log_file)
                if rc != 0:
                    ok = False
                    fail_step = "tracking"
                    fail_reason = f"track_from_labels.py return code {rc}"

            ms_path = out_dir / "trackpy" / "movie_summary.csv"
            if ok and (not ms_path.exists()):
                ok = False
                fail_step = "tracking"
                fail_reason = f"movie_summary.csv not found: {ms_path}"

            status_rows.append({
                "group": group,
                "movie": movie_id,
                "raw_path": str(raw),
                "out_dir": str(out_dir),
                "status": "OK" if ok else "FAILED",
                "failed_step": fail_step,
                "reason": fail_reason,
                "log_file": str(log_file),
            })

            if ok:
                print(f"[OK] {group}/{movie_id}")
            else:
                print(f"[ERROR] {group}/{movie_id} failed at {fail_step}. See log: {log_file}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("\n" + "!" * 80 + "\n")
                    f.write(f"[PIPELINE ERROR] step={fail_step}\n")
                    f.write(f"[REASON] {fail_reason}\n")
                    f.write("!" * 80 + "\n")

    # Step 3: compare groups
    cmp_cfg = cfg.get("compare", {}) or {}
    compare_enabled = _boolish(cmp_cfg.get("enable", True))
    if (not args.skip_compare) and compare_enabled:
        compare_log = results_root / "_group_compare_run_log.txt"
        compare_cmd = [args.python, "compare_groups.py", "--config", str(Path(args.config))]
        rc = run_logged(compare_cmd, compare_log)
        if rc != 0:
            print(f"[WARN] compare_groups.py failed (rc={rc}). See log: {compare_log}")
        else:
            print(f"[OK] Group comparison finished. See {results_root / '_group_compare'}")
    else:
        print("[INFO] compare skipped (either --skip_compare or compare.enable=false).")

    status_df = pd.DataFrame(status_rows)
    status_path = results_root / "pipeline_status.csv"
    status_df.to_csv(status_path, index=False)
    print(f"\n[OK] Wrote pipeline status: {status_path}")


def _boolish(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


if __name__ == "__main__":
    main()
