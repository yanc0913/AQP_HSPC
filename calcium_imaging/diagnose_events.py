# diagnose_events.py
# -*- coding: utf-8 -*-
"""
诊断脚本：找出为什么 ISO_e4 的 flat_01/flat_02 只检测到 1/2 个 events。

跑法（在你项目目录下，python 环境激活后）：
    python diagnose_events.py

输出：每个候选点的详细过滤情况，让我们看清"哪一条阈值挡掉了哪些峰"。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import calcium_config as cfg

# === 选要诊断的 cell ===
TARGET_EMBRYO_ID = "exp1__E3_vs_ISO_vs_BDM__ISO__e4"
TARGET_ROIS      = ["flat_01", "flat_02"]
PHASE            = "post"
POST_WINDOW_MIN  = cfg.post_analysis_min()   # 默认 10 / 20 / 等


# === 复现 detect_events_single_frame（与 build_tables 完全一致）===

REL_PEAK_FRAC     = cfg.PEAK_DETECTION["rel_peak_frac"]
ROBUST_Z_K        = cfg.PEAK_DETECTION["robust_z_k"]
REFRACTORY_FRAMES = cfg.PEAK_DETECTION["refractory_frames"]
PROM_WIN          = cfg.PEAK_DETECTION["prom_win"]
PROM_FRAC         = cfg.PEAK_DETECTION["prom_frac"]

def robust_sigma(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x); mad = np.median(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad): return float(np.std(x))
    return float(1.4826 * mad)


def diagnose_one_trace(x: np.ndarray, t_min: np.ndarray, label: str):
    print(f"  Full trace: {[round(v, 3) for v in x]}")
    print(f"\n{'='*78}")
    print(f"DIAGNOSIS: {label}")
    print('='*78)
    print(f"  Trace length: {len(x)} frames ({t_min[0]:.1f} – {t_min[-1]:.1f} min)")
    print(f"  min={x.min():.3f}  max={x.max():.3f}  median={np.median(x):.3f}")
    print(f"  MAD={np.median(np.abs(x - np.median(x))):.4f}")
    sig = robust_sigma(x)
    print(f"  sigma_robust = 1.4826 * MAD = {sig:.4f}")
    thr = np.median(x) + ROBUST_Z_K * sig
    print(f"  threshold = median + {ROBUST_Z_K}*sigma_robust = {thr:.4f}")
    print()

    # Top-10 highest frames
    top_idx = np.argsort(x)[-10:][::-1]
    print(f"  Top-10 highest frames:")
    print(f"    {'frame':>6} {'t(min)':>8} {'value':>8} {'is_local_peak':>15} {'>thr':>6}")
    for t in top_idx:
        is_peak = (1 <= t < len(x)-1 and x[t] > x[t-1] and x[t] >= x[t+1])
        print(f"    {t:>6} {t_min[t]:>8.2f} {x[t]:>8.3f} {str(is_peak):>15} "
              f"{str(bool(x[t] >= thr)):>6}")
    print()

    # All shape-pass candidates
    print(f"  All shape-pass candidates (x[t-1] < x[t] >= x[t+1]):")
    print(f"    {'frame':>6} {'t(min)':>8} {'value':>8} {'neigh_ratio':>12} "
          f"{'prom_ratio':>11} {'>thr':>6} {'kept?':>7}")

    cand = []
    for t in range(1, len(x)-1):
        if not (np.isfinite(x[t-1]) and np.isfinite(x[t]) and np.isfinite(x[t+1])): continue
        if not (x[t] > x[t-1] and x[t] >= x[t+1]): continue
        neigh = max(x[t-1], x[t+1])
        n_pass = (x[t] >= (1 + REL_PEAK_FRAC) * neigh) if (neigh > 0) else True
        n_ratio = x[t] / neigh if neigh > 0 else np.nan

        a, b = max(0, t-PROM_WIN), min(len(x), t+PROM_WIN+1)
        local = x[a:b]; local = local[np.isfinite(local)]
        if local.size > 0:
            base = np.percentile(local, 20)
            p_pass = (x[t] >= (1 + PROM_FRAC) * base) if (base > 0) else False
            p_ratio = x[t] / base if base > 0 else np.nan
        else:
            p_pass, p_ratio = False, np.nan

        z_pass = x[t] >= thr

        local_pass = n_pass or p_pass
        all_pass = local_pass and z_pass
        if all_pass:
            cand.append(t)

        # Mark each failure mode
        mark_n = "✓" if n_pass else "✗"
        mark_p = "✓" if p_pass else "✗"
        mark_z = "✓" if z_pass else "✗"
        print(f"    {t:>6} {t_min[t]:>8.2f} {x[t]:>8.3f} "
              f"{n_ratio:>9.3f}({mark_n}) {p_ratio:>8.3f}({mark_p}) "
              f"{mark_z:>6} {'YES' if all_pass else '':>7}")

    # Apply refractory merge
    print(f"\n  Surviving candidates (before refractory merge): {cand}")
    if cand:
        kept = []; i = 0
        while i < len(cand):
            group = [cand[i]]; j = i + 1
            while j < len(cand) and cand[j] - cand[j-1] <= REFRACTORY_FRAMES:
                group.append(cand[j]); j += 1
            best = max(group, key=lambda tt: x[tt])
            kept.append(best); i = j
        print(f"  After refractory merge (refractory={REFRACTORY_FRAMES}): {kept}")
        print(f"  → final event count = {len(kept)}")
    else:
        print(f"  → no events detected")


def main():
    # 找到 cells 表
    cells_path = cfg.tables_dir() / "Q2_cells_vDA_prepost.csv"
    print(f"Reading: {cells_path}")
    df = pd.read_csv(cells_path)

    # 过滤目标
    sub = df[df["embryo_id"] == TARGET_EMBRYO_ID].copy()
    if sub.empty:
        print(f"\nERROR: embryo_id {TARGET_EMBRYO_ID!r} not found.")
        print(f"Available embryo_ids:")
        for e in sorted(df["embryo_id"].unique()):
            print(f"  {e}")
        sys.exit(1)

    for roi in TARGET_ROIS:
        g = sub[(sub["roi_name"] == roi) & (sub["phase"] == PHASE)].sort_values("frame")
        if g.empty:
            print(f"\nWARN: no rows for ROI={roi}, phase={PHASE}")
            continue

        # Apply the same window cut as F1-strict
        t_min_all = g["time_min"].to_numpy(float)
        x_all     = g["cell_trace_main"].to_numpy(float)

        # Window-limited slice (F1-strict)
        t0 = t_min_all.min()
        rel = t_min_all - t0
        mask = rel <= POST_WINDOW_MIN
        t_min_w = rel[mask]
        x_w     = x_all[mask]

        diagnose_one_trace(
            x_w, t_min_w,
            f"{roi}, post, first {POST_WINDOW_MIN} min (F1-strict window)"
        )

        # Also show full post for reference
        diagnose_one_trace(
            x_all, t_min_all - t0,
            f"{roi}, post, ENTIRE recording ({len(x_all)} frames)"
        )

if __name__ == "__main__":
    main()
