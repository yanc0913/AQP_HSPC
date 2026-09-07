# build_source_data.py
# -*- coding: utf-8 -*-
"""
Assemble the Source Data workbook for the paper.

One sheet per FIGURE, not per panel: everything belonging to a figure sits
together, sections stacked down the sheet, each with its own heading and notes.
There is no README sheet - Nature Communications source data files do not
normally carry one, so every explanation lives in the sheet it applies to.

Each sheet carries, for the panels it covers:
  - the per-embryo values that are the dots on the plots (embryo is the unit of
    analysis; cells are rolled up before any test)
  - the derived quantity actually plotted where the figure shows one (log2 fold
    change for intensity, difference for event rate), computed from the same
    vehicle mean the figure used, so the arithmetic can be checked
  - the per-cell values underneath, since the roll-up is otherwise invisible
  - the statistics: test, n, exact p

Only quantities a panel actually reports are included. The pipeline computes
more than the paper shows - peak duration, a 30 min event window, per-embryo
means of the time-course panels, GsMTx4-vs-Yoda1 and between-genotype
comparisons - and none of that is carried here.

E3 vs ISO appears with TWO different p-values by design - Welch t in the main
figure, which analyses only that pair, and one-way ANOVA with Holm in the
supplementary figure, which analyses E3/ISO/BDM as a family of three. Both are
listed, side by side, with the panel each belongs to, so the difference reads
as a design choice rather than an inconsistency.

Run:  python build_source_data.py            (needs the calcium_gcamp env)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths_local import (DATASET_ROOTS, MAIN_FIG_DIRS, DATA_BASE,
                         LPM_GROUP_COMPARE)

# The LPM pipeline lives in its own repo and writes its own results tree;
# only its group-comparison folder is read here.
LPM = Path(LPM_GROUP_COMPARE)
OUT = DATA_BASE / "_SourceData" / "Source_Data.xlsx"
WSUF = "20"
AMP = "embryo_mean_amplitude_post_20min"
EVT = "mean_events_per_cell_per_20min_post"
PRETTY = {AMP: "Ca2+ intensity (F/F0)", EVT: "Ca2+ events per cell per 20 min"}


# ----------------------------------------------------------------- writing --
class Sheet:
    """A sheet built as stacked sections: heading, notes, table."""

    def __init__(self, writer, name):
        self.writer, self.name, self.row = writer, name, 0
        self.widths = {}

    def _put(self, text, col=0):
        pd.DataFrame([[text]]).to_excel(
            self.writer, sheet_name=self.name, startrow=self.row,
            startcol=col, index=False, header=False)
        self.row += 1

    def heading(self, text):
        if self.row:
            self.row += 1
        self._put(text)

    def note(self, text):
        self._put(text)

    def table(self, df, note=None):
        if note:
            self.note(note)
        df.to_excel(self.writer, sheet_name=self.name, startrow=self.row,
                    index=False)
        for i, c in enumerate(df.columns):
            w = max(len(str(c)), int(df[c].astype(str).str.len().max() or 0))
            self.widths[i] = min(42, max(self.widths.get(i, 10), w + 2))
        self.row += len(df) + 2

    def finish(self):
        ws = self.writer.sheets[self.name]
        from openpyxl.utils import get_column_letter
        for i, w in self.widths.items():
            ws.column_dimensions[get_column_letter(i + 1)].width = w


def emb(dskey, subtree="_py_out_%smin" % WSUF):
    return pd.read_csv(DATASET_ROOTS[dskey] / subtree / "tables"
                       / "Q2_embryo_summary_cells.csv")


def cells(dskey, subtree="_py_out_%smin" % WSUF):
    return pd.read_csv(DATASET_ROOTS[dskey] / subtree / "tables"
                       / "Q2_events_cells.csv")


def pairwise(dskey, subtree="_py_out_%smin" % WSUF):
    return pd.read_excel(DATASET_ROOTS[dskey] / subtree / "tables"
                         / "Q2_stats_pvalues.xlsx", sheet_name="pairwise")


# Columns that survive into the workbook. The pipeline computes more than the
# paper plots - peak duration, the 30 min event window, the active/inactive
# flag - and Source Data should carry only what a figure actually shows.
CELL_COLS = ["dataset", "batch_id", "pair_id", "condition", "genotype",
             "embryo_id", "roi_name", "cell_class", "n_events_post20min",
             "events_per_cell_per_20min_post", "mean_amplitude_post_20min",
             "pre_threshold"]


def plotted_only(st, metric_col="metric"):
    """Drop statistics for quantities no panel reports.

    duration            no figure plots peak duration
    foldchange_G_vs_Y   GsMTx4-vs-Yoda1 on the folds; Fig 5e-h draws only the
                        drug-vs-its-own-vehicle p above each box
    """
    st = st[st[metric_col].astype(str) != "duration"]
    st = st[~st.test.astype(str).str.startswith("foldchange")]
    return st.reset_index(drop=True)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def derive(e, vehicle_means):
    """Add the plotted quantity: log2 ratio for intensity, difference for rate.

    vehicle_means maps (dataset, cell_class, metric, vehicle) -> mean, taken
    from the figure's own output so these columns reproduce the published dots
    exactly rather than being recomputed by a slightly different route.
    """
    out = []
    for _, r in e.iterrows():
        row = r.to_dict()
        for metric, mode, col in ((AMP, "log2", "log2_fold_change_vs_vehicle"),
                                  (EVT, "diff", "difference_vs_vehicle")):
            key = (r["dataset"], r["cell_class"], metric, r.get("vehicle_for_row"))
            vm = vehicle_means.get(key)
            v = r.get(metric)
            if vm is None or not np.isfinite(v) or not np.isfinite(vm):
                row[col] = np.nan
                continue
            row[col] = (np.log2(v / vm) if mode == "log2" and vm > 0 and v > 0
                        else (v - vm) if mode == "diff" else np.nan)
        out.append(row)
    return pd.DataFrame(out)


# -------------------------------------------------------------- Fig 2 (LPM) --
def sheet_fig2(writer):
    s = Sheet(writer, "Fig 2")
    ts = pd.read_csv(LPM / "aligned_time_series_per_movie.csv")
    mv = pd.read_csv(LPM / "all_movies_summary.csv")
    mv = mv.rename(columns={"movie": "embryo"})

    s.heading("Figure 2c-h  |  LPM cell tracking, 13-17 hpf")
    s.note("Tg(fli1:H2B-EGFP) wild type vs aqp1a1 rk28/rk28. One movie = one "
           "embryo; the embryo is the unit of analysis.")
    s.note("Movies are aligned on ABSOLUTE developmental time, using the hpf "
           "window in each file name, not by stretching to a common frame "
           "count. Movies cover different windows (13-16, 13-17, 14-17 hpf), "
           "so the number of embryos contributing changes along the axis - the "
           "'n movies at this hpf' column below gives it at every point.")
    s.note("Midline detection: the exclusion band is centred on the gap "
           "between the two bilateral nuclear bands. See the Methods.")

    box = mv[["group", "embryo", "n_frames", "n_tracks", "n_burst_events",
              "mean_median_speed_um_per_min", "burst_per_track_ratio"]].copy()
    s.heading("Fig 2d and 2g  |  one value per embryo (the dots)")
    s.note("2d = mean_median_speed_um_per_min; 2g = burst_per_track_ratio "
           "(nuclear fragmentation events per track). n_tracks and "
           "n_burst_events are the two terms of that ratio.")
    s.table(box)

    rows = []
    for panel, col in (("2d", "mean_median_speed_um_per_min"),
                       ("2g", "burst_per_track_ratio")):
        a = mv.loc[mv.group == "WT", col]
        b = mv.loc[mv.group == "MUT", col]
        rows.append(dict(panel=panel, quantity=col, unit="",
                         n_WT=len(a), n_MUT=len(b),
                         mean_WT=a.mean(), mean_MUT=b.mean(),
                         test="Welch two-sided t-test", p_value=welch(a, b)))
    st = pd.DataFrame(rows).sort_values("panel")
    s.heading("Statistics")
    s.note("2d and 2g are the two panels that report a p-value. Panels 2c, "
           "2e, 2f and 2h are time courses plotted as mean +- SD across "
           "embryos and carry no test.")
    s.table(st)

    n_at = (ts.dropna(subset=["median_area_um2"])
            .groupby(["hpf", "group"]).movie.nunique().unstack("group")
            .fillna(0).astype(int).reset_index()
            .rename(columns={"WT": "n movies at this hpf (WT)",
                             "MUT": "n movies at this hpf (MUT)"}))
    long = ts.merge(n_at, on="hpf", how="left")
    keep = ["group", "movie", "hpf", "median_speed_um_per_min",
            "median_area_um2", "n_bursts", "n_tracked",
            "n movies at this hpf (WT)", "n movies at this hpf (MUT)"]
    s.heading("Fig 2c, 2e, 2f, 2h  |  per-embryo time series (the mean +- SD "
              "bands are computed from these)")
    s.table(long[keep].sort_values(["group", "movie", "hpf"]))
    s.finish()


# --------------------------------------------- Fig 5e-h + Supp Fig 9c-f,k-n --
def sheet_yoda(writer):
    s = Sheet(writer, "Fig 5e-h + Supp Fig 9c-n")
    fc = pd.read_csv(MAIN_FIG_DIRS["Yoda_GsMTx"] / "main_foldchange.csv")
    vm = {(r.timepoint, r.cell_class, r.metric, r.vehicle): r.vehicle_mean
          for r in fc.itertuples()}
    veh_of = {"Yoda": "DMSO", "GsMTx": "E3", "DMSO": "DMSO", "E3": "E3"}

    e = []
    for ds, tag in (("30hpf", "30hpf"), ("48hpf", "48hpf")):
        d = emb(ds)
        d.insert(0, "dataset", tag)
        e.append(d)
    e = pd.concat(e, ignore_index=True)
    e["vehicle_for_row"] = e.condition.map(veh_of)
    e = derive(e, vm)

    s.heading("Figure 5e-h and Supplementary Figure 9c-f, 9k-n  |  Ca2+ in "
              "vDA cells, 30 and 48 hpf")
    s.note("Tg(fli1:Gal4FF);Tg(UAS:GCaMP7a);Tg(fli1:Lifeact-mCherry). Same "
           "embryos in both figures: Supp Fig 9 shows the values, Fig 5 shows "
           "the drug effect derived from them.")
    s.note("Cells are classified as elongated (flat) or round. Values are the "
           "mean over the cells of one embryo; the embryo is the unit of "
           "analysis and every dot on the plots is one embryo.")
    s.note("Ca2+ intensity is F/F0 with F0 = median of the entire pre-drug "
           "phase, over a 20 min post-drug window. Events are single frames "
           "above pre_median + 2 x robust SD of the pre phase.")
    s.note("Fig 5e,f plot log2(drug / mean of its own vehicle). Fig 5g,h plot "
           "(drug - mean of its own vehicle), a difference and not a ratio "
           "because control event rates reach zero and a ratio would diverge. "
           "Both columns are below, with the vehicle mean used.")

    cols = ["dataset", "cell_class", "condition", "genotype", "embryo_id",
            "n_cells", AMP, EVT, "vehicle_for_row",
            "log2_fold_change_vs_vehicle", "difference_vs_vehicle"]
    tab = e[cols].rename(columns={AMP: PRETTY[AMP], EVT: PRETTY[EVT],
                                  "vehicle_for_row": "vehicle_used"})
    s.heading("Per-embryo values (the dots in Supp Fig 9c-f, 9k-n and Fig 5e-h)")
    s.table(tab.sort_values(["dataset", "cell_class", "condition", "embryo_id"]))

    vmt = pd.DataFrame([dict(dataset=k[0], cell_class=k[1],
                             quantity=PRETTY.get(k[2], k[2]), vehicle=k[3],
                             vehicle_mean=v) for k, v in vm.items()])
    s.heading("Vehicle means used to derive the two columns above")
    s.table(vmt.sort_values(["dataset", "cell_class", "quantity"]))

    st = []
    for ds in ("30hpf", "48hpf"):
        p = pairwise(ds)
        p.insert(0, "dataset", ds)
        st.append(p)
    st = pd.concat(st, ignore_index=True)
    st = plotted_only(st).rename(columns={"metric": "quantity",
                                          "test": "test_and_correction"})
    s.heading("Statistics")
    s.note("Planned drug-vs-vehicle comparisons only. DMSO/Yoda1 and E3/GsMTx4 "
           "have different vehicles, so each is its own family and no "
           "multiplicity correction applies (test = welch_none).")
    s.note("The same p-value is printed in Supp Fig 9 and above the "
           "corresponding bar in Fig 5e-h.")
    s.table(st)

    cl = []
    for ds in ("30hpf", "48hpf"):
        d = cells(ds)
        d.insert(0, "dataset", ds)
        cl.append(d)
    s.heading("Per-cell values underlying the per-embryo means")
    s.note("One row per cell. Not plotted; included so the roll-up to embryo "
           "means can be checked. pre_threshold is the event threshold for "
           "that cell, so the event counts can be re-derived.")
    s.table(pd.concat(cl, ignore_index=True)[CELL_COLS])
    s.finish()


# ----------------------------------------- Fig 5i-l + Supp Fig 10d-g, l-o --
def sheet_iso(writer):
    s = Sheet(writer, "Fig 5i-l + Supp Fig 10")
    x = pd.ExcelFile(MAIN_FIG_DIRS["ISO_MIC_Piezo_MAIN"] / "main_iso_foldchange.xlsx")
    per = x.parse("per_group")
    vm = {(r.dataset, r.genotype, r.cell_class, r.metric, r.vehicle): r.vehicle_mean
          for r in per.itertuples()}

    e = []
    for ds, tag in (("ISO", "48hpf E3/ISO/BDM"), ("Piezo", "48hpf MIC/piezo crispant")):
        d = emb(ds)
        d.insert(0, "dataset", tag)
        e.append(d)
    e = pd.concat(e, ignore_index=True)
    e["vehicle_for_row"] = "E3"

    key = {}
    for k, v in vm.items():
        key[(("48hpf E3/ISO/BDM" if k[0] == "ISO" else
              "48hpf MIC/piezo crispant"), k[2], k[3], k[4])] = v
    # per-genotype vehicle means in the MIC/piezo set
    gvm = {}
    for r in per.itertuples():
        gvm[(r.genotype, r.cell_class, r.metric)] = r.vehicle_mean

    rows = []
    for _, r in e.iterrows():
        row = r.to_dict()
        g = r["genotype"]
        for metric, mode, col in ((AMP, "log2", "log2_fold_change_vs_E3"),
                                  (EVT, "diff", "difference_vs_E3")):
            base = gvm.get((g, r["cell_class"], metric))
            v = r.get(metric)
            if base is None or not np.isfinite(v) or not np.isfinite(base):
                row[col] = np.nan
            elif mode == "log2":
                row[col] = np.log2(v / base) if v > 0 and base > 0 else np.nan
            else:
                row[col] = v - base
            if r["condition"] == "BDM":
                row[col] = np.nan
        rows.append(row)
    e = pd.DataFrame(rows)

    s.heading("Figure 5i-l and Supplementary Figure 10  |  Ca2+ response to "
              "isoprenaline, 48 hpf")
    s.note("Two experiments. E3 / ISO / BDM in wild type (Supp Fig 10a-g), and "
           "MIC vs piezo crispant each with E3 and ISO (Supp Fig 10h-o). "
           "Fig 5i-l shows the ISO effect from both, side by side.")
    s.note("Every embryo is normalised to the mean of ITS OWN genotype's E3 "
           "group, so the MIC and piezo bars each carry their own baseline. "
           "BDM has no normalised bar in any panel - Supp Fig 10d-g plots its "
           "raw values - so the two derived columns are left empty for it.")

    cols = ["dataset", "cell_class", "genotype", "condition", "embryo_id",
            "n_cells", AMP, EVT, "log2_fold_change_vs_E3", "difference_vs_E3"]
    s.heading("Per-embryo values (the dots in Supp Fig 10d-g, 10l-o and Fig 5i-l)")
    s.table(e[cols].rename(columns={AMP: PRETTY[AMP], EVT: PRETTY[EVT]})
            .sort_values(["dataset", "cell_class", "genotype", "condition"]))

    # ---- the two conventions, side by side
    sub = pairwise("ISO", "_py_out_%smin_E3_ISO" % WSUF)
    sub = sub[~sub.test.astype(str).str.startswith("foldchange")].copy()
    sub = sub[(sub.group1.isin(["E3", "ISO"])) & (sub.group2.isin(["E3", "ISO"]))]
    sub.insert(0, "reported_in", "Fig 5i-l, WT bar")
    sub.insert(1, "analysis", "E3 vs ISO only (two groups)")

    three = pairwise("ISO")
    three = three[~three.test.astype(str).str.startswith("foldchange")].copy()
    three.insert(0, "reported_in", "Supp Fig 10d-g")
    three.insert(1, "analysis", "E3 / ISO / BDM (three groups)")

    both = plotted_only(pd.concat([sub, three], ignore_index=True))
    s.heading("Statistics - E3 vs ISO in wild type is reported TWICE, on "
              "purpose")
    s.note("The main figure asks only whether ISO differs from E3, so that "
           "panel analyses the two groups alone: Welch two-sided t-test, no "
           "correction.")
    s.note("The supplementary figure presents E3, ISO and BDM together, so it "
           "analyses them as one family: one-way ANOVA followed by pairwise "
           "Welch tests with Holm-Bonferroni correction across the three "
           "comparisons.")
    s.note("Both are listed below with the panel each belongs to. The two "
           "p-values differ because the analyses differ, not because the data "
           "differ - the underlying embryos are identical.")
    s.table(both)

    aov = pd.read_excel(DATASET_ROOTS["Piezo"] / ("_py_out_%smin" % WSUF)
                        / "tables" / "Q2_stats_pvalues.xlsx",
                        sheet_name="twoway_anova")
    aov = aov[aov.metric.astype(str) != "duration"].reset_index(drop=True)
    aov.insert(0, "reported_in", "Supp Fig 10l-o, and Fig 5i-l MIC and piezo bars")
    s.heading("Statistics - MIC vs piezo crispant, E3 vs ISO: two-way ANOVA")
    s.note("Type-II two-way ANOVA on per-embryo values. geno = genotype (MIC "
           "vs piezo crispant), drug = treatment (E3 vs ISO), gxd = their "
           "interaction; eta2p is partial eta squared.")
    s.table(aov)

    pz = plotted_only(pairwise("Piezo"))
    pz.insert(0, "reported_in", "Supp Fig 10l-o, and Fig 5i-l MIC and piezo bars")
    s.heading("Statistics - MIC vs piezo crispant, E3 vs ISO: Tukey HSD")
    s.note("Post-hoc for the ANOVA above. The MIC|E3 vs MIC|ISO and Piezo|E3 "
           "vs Piezo|ISO rows are the p-values printed on the MIC and piezo "
           "bars of Fig 5i-l.")
    s.table(pz)

    cl = []
    for ds, tag in (("ISO", "48hpf E3/ISO/BDM"),
                    ("Piezo", "48hpf MIC/piezo crispant")):
        d = cells(ds)
        d.insert(0, "dataset", tag)
        cl.append(d)
    s.heading("Per-cell values underlying the per-embryo means")
    s.note("One row per cell. Not plotted; included so the roll-up to embryo "
           "means can be checked.")
    s.table(pd.concat(cl, ignore_index=True)[CELL_COLS])
    s.finish()


# ------------------------------------------------------------ Supp Fig 9b --
def sheet_q1(writer):
    s = Sheet(writer, "Supp Fig 9b")
    q = pd.read_csv(MAIN_FIG_DIRS["Q1_ratio"] / "main_q1_ratio.csv")
    s.heading("Supplementary Figure 9b  |  vDA / dDA GCaMP ratio, 30 vs 48 hpf")
    s.note("Ratio of the ventral to the dorsal dorsal-aorta band, control "
           "(E3) embryos only. One value per embryo.")
    s.note("A ratio above 1 means the ventral wall is brighter, i.e. Ca2+ "
           "activity is polarised to the side where haematopoietic cells "
           "emerge.")
    s.note("The panel uses the PRE-drug recording, before any compound is "
           "added, so the ratio is a property of the untreated embryo. The "
           "pipeline also writes a post-drug version of the same ratio; that "
           "is not this panel and is not included here.")
    pre = q[q.phase == "pre"]
    s.table(pre[["stage", "dataset", "batch_id", "embryo_id", "condition",
                 "genotype", "n_frames", "ratio_median"]]
            .sort_values(["stage", "embryo_id"]))
    a = pre.loc[pre.stage == "30hpf", "ratio_median"]
    b = pre.loc[pre.stage == "48hpf", "ratio_median"]
    s.heading("Statistics")
    s.table(pd.DataFrame([dict(comparison="30 hpf vs 48 hpf",
                               n_30hpf=len(a), n_48hpf=len(b),
                               median_30hpf=a.median(), median_48hpf=b.median(),
                               mean_30hpf=a.mean(), mean_48hpf=b.mean(),
                               test="Welch two-sided t-test",
                               p_value=welch(a, b))]))
    s.finish()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        sheet_fig2(w)
        sheet_yoda(w)
        sheet_iso(w)
        sheet_q1(w)
    mb = OUT.stat().st_size / 2 ** 20
    print("[OK] %s  (%.1f MB)" % (OUT, mb))
    x = pd.ExcelFile(OUT)
    for sh in x.sheet_names:
        d = x.parse(sh, header=None)
        print("   %-28s %5d rows" % (sh, len(d)))


if __name__ == "__main__":
    main()
