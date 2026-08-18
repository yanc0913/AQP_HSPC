import argparse
from pathlib import Path
import yaml
import numpy as np
import tifffile as tiff
from csbdeep.utils import normalize
from stardist.models import StarDist2D


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_TYX(img: np.ndarray, time_axis: int) -> np.ndarray:
    if img.ndim == 2:
        return img[np.newaxis, ...]
    if img.ndim != 3:
        raise ValueError(f"Expect 2D or 3D tif, got shape {img.shape}")
    if time_axis != 0:
        img = np.moveaxis(img, time_axis, 0)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    scfg = cfg["segmentation"]

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    seg_dir = out_dir / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    img = tiff.imread(str(input_path))
    img = ensure_TYX(img, int(scfg.get("time_axis", 0)))
    T, Y, X = img.shape

    model_name = scfg.get("model", "2D_versatile_fluo")
    prob = float(scfg.get("prob_thresh", 0.4))
    nms = float(scfg.get("nms_thresh", 0.2))
    labels_suffix = scfg.get("labels_suffix", "_labels.tif")

    print(f"[INFO] Loading StarDist model: {model_name}")
    model = StarDist2D.from_pretrained(model_name)

    labels_stack = np.zeros((T, Y, X), dtype=np.uint16)

    for t in range(T):
        im = img[t]
        im_norm = normalize(im, 1, 99.8, axis=None)
        labels, _ = model.predict_instances(im_norm, prob_thresh=prob, nms_thresh=nms)
        labels_stack[t] = labels.astype(np.uint16)
        if (t + 1) % 10 == 0 or t == T - 1:
            print(f"[INFO] frame {t+1}/{T}")

    out_labels = seg_dir / f"{input_path.stem}{labels_suffix}"
    tiff.imwrite(str(out_labels), labels_stack, imagej=True)
    print(f"[OK] Saved labels: {out_labels}")


if __name__ == "__main__":
    main()
