import argparse
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import imageio.v3 as iio


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_frame(labels_2d, tracks_seg, *, vmin, vmax, linewidth, markersize, title):
    """
    Render one RGB frame using Agg canvas (backend-independent).
    """
    fig = Figure(figsize=(6, 6), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    ax.imshow(labels_2d, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    # draw track tails + current points
    for tid, g in tracks_seg.groupby("track_id"):
        g = g.sort_values("frame")
        ax.plot(g["x_px"], g["y_px"], linewidth=linewidth)
        gt = g[g["frame"] == tracks_seg["frame"].max()]
        if not gt.empty:
            ax.plot(gt["x_px"], gt["y_px"], marker="o", markersize=markersize)

    canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels_path", required=True)
    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    viz = cfg.get("visualization", {})
    tcfg = cfg["tracking"]
    px_um = float(tcfg["px_um"])

    labels = tiff.imread(str(Path(args.labels_path)))
    if labels.ndim != 3:
        raise ValueError(f"Expected (T,Y,X) labels, got {labels.shape}")

    tracks = pd.read_csv(args.tracks_csv)
    if tracks.empty:
        print("[WARN] tracks_csv is empty, skip visualization.")
        return

    # convert um -> px for plotting on label image
    tracks = tracks.copy()
    tracks["x_px"] = tracks["x_um"] / px_um
    tracks["y_px"] = tracks["y_um"] / px_um

    T = labels.shape[0]
    tail = int(viz.get("tail_length", 30))
    fps = int(viz.get("fps", 10))
    max_tracks = int(viz.get("max_tracks", 200))
    linewidth = float(viz.get("linewidth", 1.0))
    markersize = float(viz.get("markersize", 2.0))
    downsample = int(viz.get("downsample", 1))
    p_low = float(viz.get("contrast_p_low", 1))
    p_high = float(viz.get("contrast_p_high", 99))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmin = np.percentile(labels, p_low)
    vmax = np.percentile(labels, p_high)

    keep_ids = sorted(tracks["track_id"].unique())[:max_tracks]
    tracks = tracks[tracks["track_id"].isin(keep_ids)].copy()

    frames = []
    for t in range(0, T, downsample):
        t0 = max(0, t - tail)
        seg = tracks[(tracks["frame"] >= t0) & (tracks["frame"] <= t)]
        rgb = render_frame(
            labels[t],
            seg,
            vmin=vmin,
            vmax=vmax,
            linewidth=linewidth,
            markersize=markersize,
            title=f"t={t}",
        )
        frames.append(rgb)

    iio.imwrite(out_path, frames, fps=fps)
    print(f"[OK] wrote overlay video: {out_path}")


if __name__ == "__main__":
    main()
