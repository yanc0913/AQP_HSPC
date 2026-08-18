import argparse
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import tifffile as tiff


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_points(frame_img: np.ndarray, xs: np.ndarray, ys: np.ndarray, value: int, radius: int = 1):
    """
    Draw points (optionally with small radius) onto a 2D image.
    xs, ys are in pixel coordinates (float allowed).
    """
    H, W = frame_img.shape
    xs_i = np.rint(xs).astype(int)
    ys_i = np.rint(ys).astype(int)

    for x, y in zip(xs_i, ys_i):
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        if radius <= 0:
            frame_img[y, x] = value
        else:
            y0 = max(0, y - radius)
            y1 = min(H - 1, y + radius)
            x0 = max(0, x - radius)
            x1 = min(W - 1, x + radius)
            frame_img[y0:y1 + 1, x0:x1 + 1] = value

def draw_polyline_points(img2d: np.ndarray, xs: np.ndarray, ys: np.ndarray, value: int, step: float = 0.5, radius: int = 0):
    """Draw a polyline by interpolating points between consecutive vertices."""
    if len(xs) < 2:
        return
    for i in range(len(xs) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        dist = float(np.hypot(x1 - x0, y1 - y0))
        n = max(2, int(dist / step) + 1)
        xi = np.linspace(x0, x1, n)
        yi = np.linspace(y0, y1, n)
        draw_points(img2d, xi, yi, value=value, radius=radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels_path", required=True, help="(T,Y,X) labels tif")
    ap.add_argument("--track_dir", required=True, help="trackpy output dir containing csv files")
    ap.add_argument("--out_path", required=True, help="output QC hyperstack tif")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    px_um = float(cfg["tracking"]["px_um"])

    qcfg = cfg.get("qc_stack", {})
    spot_value = int(qcfg.get("spot_value", 20000))
    track_value = int(qcfg.get("track_value", 60000))
    roi_value = int(qcfg.get("roi_value", 10000))
    radius_px = int(qcfg.get("radius_px", 1))

    traj_value = int(qcfg.get("traj_value", 45000))
    traj_tail = int(qcfg.get("traj_tail", 30))
    traj_step = float(qcfg.get("traj_step_px", 0.5))

    labels_path = Path(args.labels_path)
    track_dir = Path(args.track_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = tiff.imread(str(labels_path))
    if labels.ndim != 3:
        raise ValueError(f"Expected labels (T,Y,X), got {labels.shape}")
    T, H, W = labels.shape

    # CSV paths (produced by your track_from_labels.py)
    spots_all_csv = track_dir / "spots_all_objects.csv"         # ROI-filtered all objects
    spots_track_csv = track_dir / "spots_for_tracking.csv"      # ROI-filtered + area filtered
    tracks_csv = track_dir / "tracks_linked.csv"
    roi_csv = track_dir / "safety_roi_um.csv"

    if not spots_all_csv.exists():
        raise FileNotFoundError(f"Missing: {spots_all_csv}")
    if not spots_track_csv.exists():
        raise FileNotFoundError(f"Missing: {spots_track_csv}")
    if not tracks_csv.exists():
        raise FileNotFoundError(f"Missing: {tracks_csv}")

    spots_all = pd.read_csv(spots_all_csv)
    spots_track = pd.read_csv(spots_track_csv)
    tracks = pd.read_csv(tracks_csv)

    # Prepare 5-channel stack: (T,C,Y,X)
    # C0 labels (uint16), C1 all spots, C2 tracked spots, C3 ROI mask, C4 trajectories
    stack = np.zeros((T, 5, H, W), dtype=np.uint16)

    # C0 labels (clip to uint16 range just in case)
    stack[:, 0] = np.clip(labels, 0, 65535).astype(np.uint16)

    # C3 ROI mask (if exists)
    if roi_csv.exists():
        roi = pd.read_csv(roi_csv).iloc[0].to_dict()
        x0_um, x1_um, y0_um, y1_um = roi["x0_um"], roi["x1_um"], roi["y0_um"], roi["y1_um"]
        x0 = int(np.floor(x0_um / px_um))
        x1 = int(np.ceil(x1_um / px_um))
        y0 = int(np.floor(y0_um / px_um))
        y1 = int(np.ceil(y1_um / px_um))
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))
        if x1 > x0 and y1 > y0:
            stack[:, 3, y0:y1, x0:x1] = np.uint16(roi_value)

    # C1 all spots (per-frame)
    for f in range(T):
        sf = spots_all[spots_all["frame"] == f]
        if len(sf) == 0:
            continue
        xs = (sf["x_um"].to_numpy() / px_um)
        ys = (sf["y_um"].to_numpy() / px_um)
        draw_points(stack[f, 1], xs, ys, value=spot_value, radius=radius_px)
        

    # C2 tracked spots (per-frame; use spots_for_tracking OR tracks_linked)
    # Here use tracks_linked for "what actually linked"
    for f in range(T):
        tf = tracks[tracks["frame"] == f]
        if len(tf) == 0:
            continue
        xs = (tf["x_um"].to_numpy() / px_um)
        ys = (tf["y_um"].to_numpy() / px_um)
        draw_points(stack[f, 2], xs, ys, value=track_value, radius=radius_px)

    # C4 trajectories (tail) using tracks_linked.csv
    for f in range(T):
        t0 = max(0, f - traj_tail)
        seg = tracks[(tracks["frame"] >= t0) & (tracks["frame"] <= f)]
        if len(seg) == 0:
            continue
        for tid, g in seg.groupby("track_id"):
            g = g.sort_values("frame")
            xs = (g["x_um"].to_numpy() / px_um)
            ys = (g["y_um"].to_numpy() / px_um)
            draw_polyline_points(stack[f, 4], xs, ys, value=traj_value, step=traj_step, radius=0)


    # Save as ImageJ hyperstack
    # axes: T C Y X
    tiff.imwrite(
        str(out_path),
        stack,
        imagej=True,
        metadata={"axes": "TCYX"},
        compression="zlib",
    )
    print(f"[OK] Wrote ImageJ QC hyperstack: {out_path}")


if __name__ == "__main__":
    main()
