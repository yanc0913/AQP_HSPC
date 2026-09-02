# midline_v3_check.py
# -*- coding: utf-8 -*-
"""
Midline diagnostics. READ-ONLY: writes only into a new directory.

Fits the midline for every movie with the same estimator the pipeline uses, and
writes one figure per movie so the fit can be checked by eye before a run is
trusted:

    left    nuclei and other objects, with the fitted midline and the
            exclusion band drawn on
    middle  the density across the midline at the chosen angle
    right   the angle sweep, so it is visible whether the tilt is
            well-determined or flat

The estimator, in three parts:

1. Build the density from NUCLEUS-SIZED objects only
   (tracking.min_area_um2 .. max_area_um2). The all-object profile is dominated
   by sub-nuclear debris, which is densest in the midline region.
2. Take the midpoint of the two bilateral bands. Every PAIR of peaks is scored,
   because a movie often has three - the two bands plus cells that have already
   converged on the midline - and the middle peak can be as tall as a band.
3. Get the tilt from a rotation sweep scored on the same band-separation
   criterion, pivoted on the centre of the y range.

Output (a NEW directory; the results tree is untouched)
    <results_root>/../_midline_v3_check/
        midline_v3_comparison.csv    per movie: centre, tilt, gap width, flags
        <group>__<movie>.png         the three-panel diagnostic
        _summary_v3.png              centre and tilt across all movies

A midline_manual.csv in that directory, with columns
group,movie,x_center_um,tilt_deg, pins individual movies; the hand-set line is
drawn in purple beside the automatic one.

Run:  python midline_v3_check.py --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

STEP = 1.0          # um, density grid spacing
BW_UM = 8.0         # um, density smoothing
ANG_MAX = 25.0      # deg, sweep range either side of vertical
ANG_STEP = 0.5


def density(u, bw_um=BW_UM, step=STEP):
    """Smoothed 1-D density of u on a regular grid."""
    grid = np.arange(float(u.min()), float(u.max()) + step, step)
    if len(grid) < 5:
        return None, None
    c, _ = np.histogram(u, bins=len(grid), range=(grid[0], grid[-1] + step))
    return grid, gaussian_filter1d(c.astype(float), bw_um / step)


def two_band_midpoint(x, min_sep_um, bw_um=BW_UM):
    """Midpoint of the two bilateral bands in x, or None if there are not two.

    Every PAIR of peaks is scored, rather than simply taking the two tallest:
    a movie often has three peaks, the two bands plus the cells that have
    already converged on the midline, and the middle one can be as tall as a
    band.

    A pair is scored on what the two bilateral bands must satisfy:
      - both peaks tall and the valley between them deep   -> min(h) - valley
      - the two bands BRACKET the tissue: little mass left  -> 1 - frac_outside
        outside the pair
    The second term is what rejects (left band, midline cells): that pair
    leaves the whole right band outside it.
    """
    grid, dens = density(x, bw_um=bw_um)
    if grid is None:
        return None
    pk, _ = find_peaks(dens, distance=max(3, int(min_sep_um / STEP)))
    if len(pk) < 2:
        return None
    tot = float(dens.sum())
    if tot <= 0:
        return None

    best, best_s = None, -np.inf
    for a in range(len(pk)):
        for b in range(a + 1, len(pk)):
            i, j = int(pk[a]), int(pk[b])
            if (grid[j] - grid[i]) < min_sep_um:
                continue
            valley = float(dens[i:j + 1].min())
            depth = min(dens[i], dens[j]) - valley
            frac_out = float((dens.sum() - dens[i:j + 1].sum()) / tot)
            s = depth * (1.0 - frac_out)
            if s > best_s:
                best_s, best = s, (i, j)
    if best is None:
        return None
    i, j = best
    return dict(x=0.5 * (grid[i] + grid[j]), left=float(grid[i]),
                right=float(grid[j]), hL=float(dens[i]), hR=float(dens[j]), score=float(best_s))


def gap_sharpness(u, x_cut, look_um=150.0):
    """How cleanly a gap at x_cut separates the two bands.

    Tall density on BOTH sides within `look_um`, low density at the cut. This
    is the angle objective: the correct tilt is the one that lines the bands up
    so the gap is at its deepest and the flanking bands at their tallest. The
    cut position is NOT free here - it is pinned to the global two-band
    anchor - so the sweep can only answer "which tilt", never wander off to a
    wide empty region, which is how the free-position variants failed.
    """
    grid, dens = density(u)
    if grid is None or dens.max() <= 0:
        return -np.inf
    i = int(np.clip(np.searchsorted(grid, x_cut), 1, len(grid) - 2))
    w = int(look_um / STEP)
    lo, hi = max(0, i - w), min(len(dens), i + w + 1)
    if i - lo < 5 or hi - i < 5:
        return -np.inf
    return float(min(dens[lo:i].max(), dens[i + 1:hi].max()) - dens[i])


def evaluate_angle(x, yc, ang, anchor0, mcfg, min_sep):
    """Place the midline at one candidate angle and measure the result.

    The position is re-derived IN the rotated frame: the anchor was measured on
    the unrotated density, which is a projection along a different axis, so a
    tilted line keeping it would carry an offset belonging to the vertical one.
    A re-fit that moves further than `v3_max_recenter_um` from the vertical
    anchor is not used.

    Returns the centre, and the width of the low-density run the line sits in.
    A width of 0 means the density where the line lands is at band level, i.e.
    the midline is populated rather than empty.
    """
    t = np.radians(ang)
    u = x * np.cos(t) - yc * np.sin(t)
    g, d = density(u)
    if g is None:
        return None

    xc, anch, moved = anchor0["x"], anchor0, 0.0
    if abs(ang) > 1e-6:
        rot = two_band_midpoint(u, min_sep)
        lim = float(mcfg.get("v3_max_recenter_um", 60.0))
        if rot is not None and abs(rot["x"] - anchor0["x"]) <= lim:
            xc, anch = float(rot["x"]), rot
            moved = float(rot["x"] - anchor0["x"])
        elif rot is not None:
            moved = np.nan          # re-centre rejected; keep the vertical anchor

    thr = float(mcfg.get("v3_gap_frac", 0.30)) * max(anch["hL"], anch["hR"])
    i = int(np.clip(np.searchsorted(g, xc), 1, len(g) - 2))
    if d[i] >= thr:
        gap = 0.0                   # the line is sitting on band-level density
    else:
        lo = hi = i
        while lo > 0 and d[lo - 1] < thr:
            lo -= 1
        while hi < len(d) - 1 and d[hi + 1] < thr:
            hi += 1
        gap = float(g[hi] - g[lo])
    # How raised the density is where the line lands, as a fraction of the
    # WEAKER band. A gap can pass the threshold above and still sit on a small
    # local peak when one band is much fainter than the other - which is the
    # case on MUT 20250702_f3 (438 nuclei against 2218 debris, the worst ratio
    # in the set). Reported so those movies get flagged for a look by eye.
    on_band = float(d[i] / max(1e-9, min(anch["hL"], anch["hR"])))
    return dict(ang=float(ang), x_center=float(xc), gap=gap, moved=moved,
                on_band=on_band, sharp=gap_sharpness(u, xc))


def fit_midline_v3(spots, tcfg, mcfg, min_sep=60.0):
    """Two-band anchor for the position, constrained rotation sweep for tilt."""
    amin = float(tcfg.get("min_area_um2", 20.0))
    amax = float(tcfg.get("max_area_um2", 60.0))
    T = int(spots["frame"].max()) + 1
    e = spots[spots["frame"] <= int(np.floor(T * float(mcfg.get("fit_early_frac", 0.33))))]
    nuc = e[(e["area_um2"] >= amin) & (e["area_um2"] <= amax)]
    out = {"ok": False, "flag": "", "n_nuclei": int(len(nuc))}
    if len(nuc) < 50:
        out["flag"] = "too few nuclei"
        return out

    x = nuc["x_um"].to_numpy(float)
    y = nuc["y_um"].to_numpy(float)
    # Rotation pivot: centre of the y RANGE, not median(y). The nuclei are
    # skewed down the field, so median(y) put the pivot at y~560 of a 60-650
    # field on WT 20250616 and any tilt swung the whole line off the midline
    # even when the angle itself was fine.
    ymid = float(0.5 * (np.percentile(y, 5) + np.percentile(y, 95)))
    yc = y - ymid

    anchor0 = two_band_midpoint(x, min_sep)
    if anchor0 is None:
        out["flag"] = "no two bands globally"
        return out

    def _sep(ang):
        """Band separation at this angle - the angle objective.

        Density AT the midline is not usable here: on the movies where the
        midline is populated (MUT 20250709_f3, both MUT 20250702) the only way
        to lower it is to smear the projection, so a cut-based score runs to a
        wrong angle and destroys the far band. Scoring the bands themselves has
        no such degeneracy - smearing lowers the score.
        """
        t = np.radians(ang)
        r = two_band_midpoint(x * np.cos(t) - yc * np.sin(t), min_sep)
        return r["score"] if r is not None else np.nan

    curve = np.array([(ang, _sep(ang))
                      for ang in np.arange(-ANG_MAX, ANG_MAX + 1e-9, ANG_STEP)], float)
    if not np.isfinite(curve[:, 1]).any():
        out["flag"] = "no usable density at any angle"
        return out

    vert = evaluate_angle(x, yc, 0.0, anchor0, mcfg, min_sep)
    free = evaluate_angle(x, yc, float(curve[int(np.nanargmax(curve[:, 1])), 0]),
                          anchor0, mcfg, min_sep)

    # The swept tilt is used. Checked against hand-drawn midline regions on
    # four movies, its angle matches the drawn corridor's long axis on all
    # four, including the two where it was previously vetoed: MUT 20250702_f2
    # (drawn axis about -10 deg, swept -11) and _f3 (about -7.4, swept -9).
    #
    # A gap width of 0 means the midline is populated rather than empty, which
    # at 13-17 hpf is expected: the LPM is converging. It is reported as a
    # warning so those movies are easy to spot, and does not override the fit.
    # v3_gap_guard turns it into a veto for anyone who wants that behaviour.
    pick, why = (free if (mcfg.get("v3_allow_tilt", True) and free is not None)
                 else vert), ""
    if pick is not vert and pick["gap"] <= 0:
        why = ("no empty gap at %+.1f deg - cells sit on the midline here "
               "(converging LPM); tilt kept" % pick["ang"])
    if mcfg.get("v3_gap_guard", False) and pick is not vert and pick["gap"] <= 0:
        pick, why = vert, ("tilt %+.1f deg vetoed by v3_gap_guard" % free["ang"])

    lvl = float(mcfg.get("v3_on_band_warn", 0.50))
    if pick["on_band"] > lvl and pick["gap"] > 0:
        why = ((why + "; ") if why else "") + (
            "line sits on raised density (%.0f%% of the weaker band) - check by eye"
            % (100 * pick["on_band"]))
    out["flag"] = why

    t = np.radians(pick["ang"])
    out.update(ok=True, tilt_deg=pick["ang"], x_center=pick["x_center"],
               gap_width_um=pick["gap"], contrast=pick["sharp"],
               recentre_um=pick["moved"], on_band_frac=pick["on_band"],
               x_center_vertical=vert["x_center"], gap_vertical_um=vert["gap"],
               tilt_free_deg=(free["ang"] if free else np.nan),
               x_center_free=(free["x_center"] if free else np.nan),
               gap_free_um=(free["gap"] if free else np.nan),
               left_peak=anchor0["left"], right_peak=anchor0["right"],
               sharp_vertical=vert["sharp"],
               p0=np.array([pick["x_center"] / np.cos(t), ymid]),
               v=np.array([np.sin(t), np.cos(t)]), curve=curve)
    return out


def line_x_at_y(p0, v, yy):
    if abs(v[1]) < 1e-9:
        return np.full_like(yy, p0[0], dtype=float)
    return p0[0] + (yy - p0[1]) * (v[0] / v[1])


def plot_movie(name, spots, tcfg, old, v3, band_half, out_png, man=None):
    amin = float(tcfg.get("min_area_um2", 20.))
    amax = float(tcfg.get("max_area_um2", 60.))
    T = int(spots["frame"].max()) + 1
    e = spots[spots["frame"] <= int(np.floor(T * 0.33))]
    nuc = e[(e.area_um2 >= amin) & (e.area_um2 <= amax)]
    deb = e[(e.area_um2 < amin) | (e.area_um2 > amax)]

    fig, (ax, axd, axa) = plt.subplots(
        1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1, 1]})
    ax.scatter(deb.x_um, deb.y_um, s=3, c="0.8", label="other (n=%d)" % len(deb))
    ax.scatter(nuc.x_um, nuc.y_um, s=7, c="#4D4D4D",
               label="nucleus-sized (n=%d)" % len(nuc))
    yy = np.linspace(e.y_um.min(), e.y_um.max(), 50)

    if old is not None:
        ax.plot(line_x_at_y(old["p0"], old["v"], yy), yy, color="#CC3333", lw=1.5,
                alpha=.8, label="fit recorded in the results tree")
    if v3.get("ok"):
        xc = v3["x_center"]
        ax.axvline(xc, color="#117733", lw=1, ls="-", alpha=.35)
        xx = line_x_at_y(v3["p0"], v3["v"], yy)
        ax.plot(xx, yy, color="#117733", lw=2.5,
                label="v3 x=%.0f, tilt %+.1f deg" % (xc, v3["tilt_deg"]))
        ax.plot(xx - band_half, yy, color="#117733", lw=1, ls=":")
        ax.plot(xx + band_half, yy, color="#117733", lw=1, ls=":")
        hw = 0.5 * v3.get("gap_width_um", 2 * band_half)
        ax.plot(xx - hw, yy, color="#117733", lw=1, ls="--", alpha=.5)
        ax.plot(xx + hw, yy, color="#117733", lw=1, ls="--", alpha=.5,
                label="measured gap %.0f um" % v3.get("gap_width_um", np.nan))
        af, xf = v3.get("tilt_free_deg", 0.0), v3.get("x_center_free", np.nan)
        if abs(af) > 1e-6 and np.isfinite(xf) and abs(v3["tilt_deg"] - af) > 1e-6:
            tf = np.radians(af)
            ax.plot(line_x_at_y(np.array([xf / np.cos(tf), v3["p0"][1]]),
                                np.array([np.sin(tf), np.cos(tf)]), yy), yy,
                    color="#6699CC", lw=2, ls="-.", alpha=.9,
                    label="tilted %+.1f deg, re-centred x=%.0f" % (af, xf))
    if man is not None:
        ax.plot(line_x_at_y(man["p0"], man["v"], yy), yy, color="#AA4499", lw=2.5,
                label="MANUAL x=%.0f %+.1f deg" % (man["x_center"], man["tilt_deg"]))

    ax.invert_yaxis()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    g0, d0 = density(nuc.x_um.to_numpy(float))
    axd.plot(g0, d0, color="0.6", lw=1, label="unrotated (x)")
    if v3.get("ok"):
        t = np.radians(v3["tilt_deg"])
        u = (nuc.x_um.to_numpy(float) * np.cos(t)
             - (nuc.y_um.to_numpy(float) - v3["p0"][1]) * np.sin(t))
        g, d = density(u)
        axd.plot(g, d, color="#117733",
                 label="rotated %+.1f deg" % v3["tilt_deg"])
        axd.axvline(v3["x_center"], color="#117733", lw=2)
        for k in ("left_peak", "right_peak"):
            axd.axvline(v3[k], color="#6699CC", lw=1, ls="--")
    axd.set_xlabel("across-midline coord (um)")
    axd.set_ylabel("smoothed count")
    axd.set_title("density across the midline, best angle", fontsize=9)
    axd.legend(fontsize=8, frameon=False)

    if v3.get("ok"):
        c = v3["curve"]
        axa.plot(c[:, 0], c[:, 1], color="#4D4D4D")
        axa.axvline(v3["tilt_deg"], color="#117733", lw=2, label="v3 best")
        axa.legend(fontsize=8, frameon=False)
    axa.set_xlabel("midline tilt (deg from vertical)")
    axa.set_ylabel("band separation")
    axa.set_title("angle sweep: is the tilt real?", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tilt", action="store_true",
                    help="use the swept tilt instead of a vertical midline")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    results = Path(cfg["project"]["results_root"])
    tcfg = cfg.get("tracking", {}) or {}
    mcfg = dict(cfg.get("midline_tls", {}) or {})
    if args.tilt:
        mcfg["v3_allow_tilt"] = True
        print("[tilt] using the swept tilt, re-centred at that angle")
    band_half = float(mcfg.get("band_half_width_um", 20.0))

    # Optional hand-set midlines for movies the automatic estimate cannot
    # resolve. Columns: group,movie,x_center_um,tilt_deg. Absent = no override.
    mancsv = results.parent / "_midline_v3_check" / "midline_manual.csv"
    mandf = pd.read_csv(mancsv) if mancsv.exists() else None
    if mandf is not None:
        print("[manual] %d override(s) from %s" % (len(mandf), mancsv.name))

    out_dir = results.parent / "_midline_v3_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for group in cfg["dataset"]["groups"]:
        gdir = results / group
        if not gdir.exists():
            continue
        for mdir in sorted(p for p in gdir.iterdir() if p.is_dir()):
            t = mdir / "trackpy"
            f = t / "spots_all_objects_roi.csv"
            if not f.exists():
                continue
            spots = pd.read_csv(f)

            # whatever fit the results tree already records, for reference
            old = None
            oi = t / "midline_tls_spots_filter_info.csv"
            if oi.exists():
                r = pd.read_csv(oi).iloc[0]
                old = {"p0": np.array([r.p0_x_um, r.p0_y_um]),
                       "v": np.array([r.v_x, r.v_y]),
                       "x_center": float(r.x_center_um), "ok": True}

            v3 = fit_midline_v3(spots, tcfg, mcfg)

            man = None
            if mandf is not None:
                mm = mandf[(mandf.group == group) & (mandf.movie == mdir.name)]
                if len(mm):
                    r = mm.iloc[0]
                    tt = np.radians(float(r.tilt_deg))
                    ym = float(spots.y_um.median())
                    man = {"x_center": float(r.x_center_um),
                           "tilt_deg": float(r.tilt_deg),
                           "p0": np.array([float(r.x_center_um) / np.cos(tt), ym]),
                           "v": np.array([np.sin(tt), np.cos(tt)])}

            name = "%s__%s" % (group, mdir.name)
            plot_movie(name, spots, tcfg, old, v3, band_half,
                       out_dir / (name + ".png"), man=man)

            rows.append(dict(
                group=group, movie=mdir.name,
                stored_x=(old["x_center"] if old else np.nan),
                v3_x=v3.get("x_center", np.nan),
                v3_tilt=v3.get("tilt_deg", np.nan),
                v3_tilt_free=v3.get("tilt_free_deg", np.nan),
                v3_x_vert=v3.get("x_center_vertical", np.nan),
                v3_gap_vert_um=v3.get("gap_vertical_um", np.nan),
                v3_x_free=v3.get("x_center_free", np.nan),
                v3_gap_free_um=v3.get("gap_free_um", np.nan),
                v3_recentre_um=v3.get("recentre_um", np.nan),
                v3_on_band_frac=v3.get("on_band_frac", np.nan),
                v3_left=v3.get("left_peak", np.nan),
                v3_right=v3.get("right_peak", np.nan),
                v3_gap_um=v3.get("gap_width_um", np.nan),
                contrast=v3.get("contrast", np.nan),
                n_nuclei=v3["n_nuclei"], n_debris=int(len(spots) - v3["n_nuclei"]),
                manual_x=(man["x_center"] if man else np.nan),
                manual_tilt=(man["tilt_deg"] if man else np.nan),
                ok=v3["ok"], flag=v3["flag"]))
            r = rows[-1]
            print("  %-44s stored=%6.0f fit=%6.0f(%+5.1f) "
                  "gap=%5.0fum %s"
                  % (name, r["stored_x"],
                     r["v3_x"], r["v3_tilt"], r["v3_gap_um"], v3["flag"]))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "midline_v3_comparison.csv", index=False)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    idx = np.arange(len(df))
    a1.plot(idx, df.stored_x, "o-", color="#CC3333",
            label="recorded in the results tree")
    a1.plot(idx, df.v3_x, "s-", color="#117733", label="v3")
    a1.plot(idx, df.v3_left, ".", color="#6699CC", label="band peaks")
    a1.plot(idx, df.v3_right, ".", color="#6699CC")
    a1.set_ylabel("x centre (um)")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("midline position")
    a2.bar(idx, df.v3_tilt, .5, color="#117733", label="tilt")
    a2.axhline(0, color="0.5", lw=1)
    a2.set_ylabel("tilt (deg from vertical)")
    a2.legend(frameon=False, fontsize=8)
    a2.set_xticks(idx)
    a2.set_xticklabels(["%s/%s" % (g[0], m[:16])
                        for g, m in zip(df.group, df.movie)],
                       rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "_summary_v3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n[OK] %d movies -> %s" % (len(df), out_dir))


if __name__ == "__main__":
    main()
