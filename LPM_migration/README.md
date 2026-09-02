# LPM migration tracking pipeline

Segmentation → tracking → group comparison for zebrafish **lateral plate
mesoderm (LPM)** nuclei in timelapse movies, comparing wild type against
`aqp1a.1-/-` over 13–17 hpf.

This code produces Figure 2c–h of Kondrychyn *et al.*, *"Cellular hydraulics
ensures robust endothelial-to-haematopoietic transition"*.

> **Code only.** Raw movies and analysis outputs are not in this repository.
> Copy `config.example.yaml` → `config.yaml` (git-ignored) and point it at your
> own data.

---

## 1. Pipeline

```
raw timelapse .tif
       │
       ▼
segment_stardist.py     StarDist2D per frame ──► <movie>_labels.tif  (T,Y,X uint16)
       ▼
track_from_labels.py    labels ──► region properties ──► ROI crop ──►
                        MIDLINE band removal ──► area/density filters ──►
                        trackpy linking ──► gap closing ──► burst detection
       ▼
compare_groups.py       per-movie summaries ──► WT vs mutant figures + stats
```

`run_pipeline.py` runs segmentation + tracking for every movie in every group,
then the group comparison.

| Script | Role |
|---|---|
| `run_pipeline.py` | Orchestrator; writes `pipeline_status.csv` |
| `segment_stardist.py` | StarDist2D segmentation → label stack |
| `track_from_labels.py` | The core. Everything from label stack to per-movie metrics |
| `compare_groups.py` | Per-movie summaries → boxplots, mean±SD time series, Welch t |
| `make_qc_stack_from_labels.py` | 5-channel ImageJ QC hyperstack (labels / spots / tracked / ROI / trajectories) |
| `visualize_tracks_on_labels.py` | Track-overlay movie for visual inspection |
| `run_midline_v3.py` | Re-run tracking into a **new** results tree, reusing existing label stacks |
| `midline_v3_check.py` | Read-only midline diagnostics: per-movie fit figures |
| `burst_timing_check.py` | Per-movie traces, movies-per-timepoint, burst-timing distribution |
| `check_results_text_claims.py` | Recomputes the quantities stated about Fig 2c-h from the tables |
| `check_panel_h_confound.py` | Nuclei counted with and without tracking, for comparison |
| `compare_results_trees.py` | Panel-by-panel diff between two results trees |

---

## 2. Setup

Requires Python 3.9+. **StarDist needs TensorFlow** — install a build matching
your platform and GPU first, then:

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml          # then edit the two paths
```

Edit `project.data_root` and `project.results_root`. Movies are expected under
`data_root/<group input_dir>/`.

Movie **filenames carry the developmental window**, e.g.
`20250616-wt_13-17hpf.tif`. `compare_groups.py` parses `13-17hpf` to place each
movie on an absolute hpf axis. Without that token a movie falls back to
end-alignment, which is almost never what you want — name your files.

---

## 3. Running

```bash
python run_pipeline.py --config config.yaml
python run_pipeline.py --config config.yaml --only_group WT
python run_pipeline.py --config config.yaml --only_movie 20250616-wt_13-17hpf
python run_pipeline.py --config config.yaml --skip_compare
```

### Re-running without re-segmenting

`run_pipeline.py` always re-runs StarDist. When only tracking parameters
changed, that is hours of GPU time and gigabytes of labels wasted. Use:

```bash
python run_midline_v3.py --config config_new.yaml \
    --labels_from "<results root that already has seg/ label stacks>" \
    --python "<python with stardist and trackpy>"
```

It points `track_from_labels.py` at the existing labels and writes everything
else into the new `results_root`. It **refuses to run** if the two roots are the
same, so an existing tree cannot be overwritten by accident. Copy the `seg/`
folders across afterwards if you want the new tree to stand alone.

---

## 4. The midline, and why it matters

Objects inside a band around the midline are removed, in every frame, for both
genotypes. That band has to be in the right place.

**How it is found (`method: bilateral`, the default):**

1. Build the x density from **nucleus-sized objects only**
   (`tracking.min_area_um2 .. max_area_um2`). The all-object profile is
   dominated by sub-nuclear debris, which is densest exactly at the midline.
2. Decide **which two peaks are the bands** by scoring every *pair* on
   (a) the depth of the valley between them and (b) how completely the pair
   brackets the tissue. Taking the two tallest peaks fails whenever a movie has
   three — two bands plus the converging cells sitting on the midline.
3. Get the tilt from a rotation sweep scored on the same band-separation
   criterion, pivoted on the **centre of the y range** (not `median(y)`, which is
   skewed by the nuclei distribution), re-deriving the position in the rotated
   frame.

`method: mode_tls` is an earlier, simpler estimator kept as an option. It is not
recommended: it assumes the midline is the densest column of objects, which does
not hold for a bilateral tissue.

**A gap of zero is not a failure.** At 13–17 hpf the LPM is converging, so some
movies genuinely have cells *on* the midline. The fit reports
`cells_on_midline` rather than rejecting those movies.

`band_half_width_um` (default 20) is a real analysis choice, not a display
setting: the midline is populated, so a wider band removes more of the very
population being measured.

**Check it by eye before trusting a run:**

```bash
python midline_v3_check.py --config config.yaml
```

Writes, into a **new** directory beside the results tree, one diagnostic figure
per movie (nuclei scatter with every candidate midline, the density across the
midline, and the angle sweep) plus a comparison csv. It reads only; nothing
under the results tree is touched. A `midline_manual.csv` with columns
`group,movie,x_center_um,tilt_deg` overrides individual movies, drawn in purple
alongside the automatic fit so a hand-set value can be checked before use.

---

## 5. Configuration

Everything lives in `config.yaml`:

| Block | Controls |
|---|---|
| `project` | data and results roots |
| `dataset.groups` | group name → input folder and glob |
| `segmentation` | StarDist model, probability and NMS thresholds |
| `roi` | automatic crop around the tissue |
| `midline_tls` | method, band half-width, tilt sweep, gap fraction |
| `tracking` | pixel size, frame interval, area limits, search range, memory, minimum track length |
| `gap_closing` | maximum gap, displacement, cost weights |
| `burst_detection` | the fragmentation criteria (see below) |
| `compare` | hpf window, group order, pre-burst aggregation |

### Burst (nuclear fragmentation) detection

A burst is a big nucleus disappearing and being replaced by several small
fragments. Two settings carry the meaning:

- `big_area_um2` — the parent must be at least this large.
- `frag_area_ratio_max` — each fragment must be at most this fraction of the
  parent. **It must be below 0.5**, otherwise a normal mitotic division, whose
  two daughters are each about half the parent, is counted as fragmentation.
  Default 0.30.

---

## 6. Analysis notes

- **The movie (= embryo) is the unit of analysis.** Per-movie summaries are
  compared between groups with Welch's t-test; individual nuclei are never the
  unit.
- **The x-axis is absolute developmental time.** Each movie is placed on an hpf
  axis using the window in its filename, not stretched to a common frame count,
  so a given x position means the same developmental stage in every embryo.
- Speed skips the first frame by default: there is no preceding frame to
  difference against.

---

## 7. Reproducing the published figures

The published run used `method: bilateral`, `band_half_width_um: 20`. To
regenerate Figure 2c–h:

```bash
python run_pipeline.py --config config.yaml
python midline_v3_check.py --config config.yaml     # then look at the figures
```

Group figures and tables land in `<results_root>/_group_compare/`:
`all_movies_summary.csv` (one row per movie — the boxplot dots) and
`aligned_time_series_per_movie.csv` (the mean±SD bands).

---

## License

MIT — see [LICENSE](../LICENSE) at the repository root.
