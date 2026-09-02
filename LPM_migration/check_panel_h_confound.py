"""Nucleus counts with and without tracking.

The per-frame nuclei count normally comes from linked tracks, so it depends
on tracking quality as well as on how many nuclei are present. This counts
nucleus-sized SEGMENTED objects instead - straight from the ROI and
midline-filtered spots, no linking involved - and reports both, plus the
tracked/segmented fraction per group.
"""
import io, re, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.stats import ttest_ind

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def _results_root(cfg_path):
    """The results tree a config points at.

    Never hardcode it: these scripts are run against whichever tree is of
    interest, and result folders get renamed."""
    import yaml
    with io.open(cfg_path, encoding="utf-8") as f:
        return Path(yaml.safe_load(f)["project"]["results_root"])


def _args(*extra):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_midline_v3.yaml",
                    help="config whose results_root holds the run to analyse")
    for name, default, helptext in extra:
        ap.add_argument(name, default=default, help=helptext)
    return ap.parse_args()

_ARGS = _args()
NEW = _results_root(_ARGS.config)
cfg = yaml.safe_load(io.open(_ARGS.config, encoding="utf-8"))
amin = float(cfg["tracking"]["min_area_um2"])
amax = float(cfg["tracking"]["max_area_um2"])

rows = []
for grp in ("WT", "MUT"):
    for mdir in sorted((NEW / grp).iterdir()):
        if not mdir.is_dir():
            continue
        f = mdir / "trackpy" / "spots_all_objects_roi_midline_spotsfiltered.csv"
        if not f.exists():
            f = mdir / "trackpy" / "spots_all_objects_roi.csv"
        s = pd.read_csv(f)
        nf = int(pd.read_csv(mdir / "trackpy" / "movie_summary.csv").iloc[0]["n_frames"])
        s0, s1 = [float(v) for v in re.search(
            r"(\d+(?:\.\d+)?)[-_](\d+(?:\.\d+)?)\s*hpf", mdir.name, re.I).groups()]
        s = s[(s.area_um2 >= amin) & (s.area_um2 <= amax)]
        per = s.groupby("frame").size()
        hpf = s0 + per.index.to_numpy(float) * (s1 - s0) / max(1, nf - 1)
        for h, n in zip(hpf, per.to_numpy()):
            rows.append(dict(group=grp, movie=mdir.name, hpf=round(float(h), 4),
                             n_seg=int(n)))
d = pd.DataFrame(rows)

a = pd.read_csv(NEW / "_group_compare" / "aligned_time_series_per_movie.csv")

print("nucleus-sized SEGMENTED objects per frame - no tracking involved")
for lo, hi in ((13, 14), (14, 15), (15, 16), (16, 17), (14, 17)):
    e = (d[(d.hpf >= lo) & (d.hpf <= hi)].groupby(["group", "movie"])
         .n_seg.mean().reset_index())
    w = e.loc[e.group == "WT", "n_seg"]
    u = e.loc[e.group == "MUT", "n_seg"]
    print("  %.0f-%.0f hpf : WT %5.1f  MUT %5.1f  Welch p = %.4g"
          % (lo, hi, w.mean(), u.mean(), ttest_ind(w, u, equal_var=False).pvalue))

print("\nfor comparison, TRACKED nuclei (what panel h plots)")
for lo, hi in ((13, 14), (14, 15), (15, 16), (16, 17), (14, 17)):
    e = (a[(a.hpf >= lo) & (a.hpf <= hi)].groupby(["group", "movie"])
         .n_tracked.mean().reset_index())
    w = e.loc[e.group == "WT", "n_tracked"].dropna()
    u = e.loc[e.group == "MUT", "n_tracked"].dropna()
    print("  %.0f-%.0f hpf : WT %5.1f  MUT %5.1f  Welch p = %.4g"
          % (lo, hi, w.mean(), u.mean(), ttest_ind(w, u, equal_var=False).pvalue))

print("\nratio tracked / segmented (is the mutant simply harder to track?)")
e1 = (d[(d.hpf >= 14) & (d.hpf <= 17)].groupby(["group", "movie"])
      .n_seg.mean().reset_index())
e2 = (a[(a.hpf >= 14) & (a.hpf <= 17)].groupby(["group", "movie"])
      .n_tracked.mean().reset_index())
m = e1.merge(e2, on=["group", "movie"])
m["frac"] = m.n_tracked / m.n_seg
for g in ("WT", "MUT"):
    s = m.loc[m.group == g, "frac"]
    print("  %-4s %.2f  (%s)" % (g, s.mean(), np.round(s.values, 2)))
print("  Welch p = %.4g" % ttest_ind(m.loc[m.group == "WT", "frac"],
                                     m.loc[m.group == "MUT", "frac"],
                                     equal_var=False).pvalue)
d.to_csv(NEW / "_group_compare" / "_vs_submitted" / "segmented_nuclei_per_frame.csv",
         index=False)
