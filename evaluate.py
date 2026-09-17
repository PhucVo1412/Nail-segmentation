"""
Stage 4 (training_plan.md): evaluate the trained model on the held-out test
set (per-image mIoU, gate ~0.85) and sanity-check it end-to-end against the
existing classical-CV baseline (output.jpg) on images/img1.jpg, img2.jpg.

Usage:
    python evaluate.py --model models/nail_seg.keras --data dataset/test --out eval_out
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from nail_lib import FINGERS, crop_finger, detect_landmarks, local_to_image_points
from nail_seg_model import CUSTOM_OBJECTS, per_image_iou

COLORS = {
    "thumb": (255, 0, 255), "index": (0, 255, 255), "middle": (0, 255, 0),
    "ring": (255, 128, 0), "pinky": (255, 0, 0),
}


def load_test_set(test_dir: Path):
    img_dir, mask_dir = test_dir / "images", test_dir / "masks"
    names = sorted(p.name for p in img_dir.glob("*.png"))
    imgs, masks = [], []
    for name in names:
        img = cv2.imread(str(img_dir / name))  # BGR, matches training pipeline
        mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        imgs.append(img)
        masks.append(mask)
    return names, imgs, masks


def make_grid(img_bgr, gt_mask, pred_mask):
    h, w = img_bgr.shape[:2]

    def overlay(mask, color):
        out = img_bgr.copy()
        colored = np.zeros_like(out)
        colored[mask > 0] = color
        return cv2.addWeighted(out, 0.6, colored, 0.4, 0)

    gt_panel = overlay(gt_mask, (0, 255, 0))
    pred_panel = overlay(pred_mask, (0, 0, 255))
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    return np.hstack([img_bgr, sep, gt_panel, sep, pred_panel])


def evaluate_test_set(model, data_dir: Path, out_dir: Path, crop_size: int, batch_size: int = 16):
    names, imgs, masks = load_test_set(data_dir)
    if not names:
        print(f"No crops found under {data_dir}")
        return

    x = np.stack([cv2.resize(im, (crop_size, crop_size)) for im in imgs]).astype(np.float32) / 255.0
    y_true = np.stack([(m > 127).astype(np.float32)[..., None] for m in masks])

    y_pred = model.predict(x, batch_size=batch_size, verbose=0)
    ious = per_image_iou(y_true, y_pred)

    per_finger = defaultdict(list)
    for name, iou in zip(names, ious):
        finger = name.rsplit("__", 1)[-1].replace(".png", "")
        per_finger[finger].append(float(iou))

    print(f"\n--- Test set: {len(names)} crops ---")
    print(f"Mean per-image IoU: {ious.mean():.4f}  (gate: ~0.85 per training_plan.md)")
    for finger, vals in sorted(per_finger.items()):
        print(f"  {finger:8s}: mean IoU {np.mean(vals):.4f} (n={len(vals)})")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, img, mask, pred, iou in zip(names, imgs, masks, y_pred, ious):
        pred_mask = (cv2.resize(pred, (img.shape[1], img.shape[0])) > 0.5).astype(np.uint8) * 255
        grid = make_grid(img, mask, pred_mask)
        cv2.imwrite(str(out_dir / f"{Path(name).stem}_iou{iou:.2f}.png"), grid)
    print(f"Wrote qualitative grids to {out_dir}")

    return ious.mean()


def run_end_to_end(model, image_path: Path, out_prefix: Path, crop_size: int):
    """Swap the trained model in for segment_nail() and run the full
    detect->crop->segment->map-back pipeline from main.py, for visual
    comparison against the existing GrabCut baseline (output.jpg)."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read {image_path}")
        return

    landmarks = detect_landmarks(image)
    if landmarks is None:
        print(f"No hand detected in {image_path}")
        return

    overlay = image.copy()
    results = {}
    for name, (tip_idx, joint_idx) in FINGERS.items():
        tip, joint = landmarks[tip_idx], landmarks[joint_idx]
        crop_info = crop_finger(image, tip, joint)
        if crop_info is None:
            continue
        crop = crop_info["crop"]
        h0, w0 = crop.shape[:2]
        x = cv2.resize(crop, (crop_size, crop_size)).astype(np.float32)[None] / 255.0
        pred = model.predict(x, verbose=0)[0, ..., 0]
        pred_mask = (cv2.resize(pred, (w0, h0)) > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True).reshape(-1, 2)
        if len(approx) < 3:
            continue

        img_poly = local_to_image_points(approx, crop_info["offset"], crop_info["M"])
        img_poly_int = np.round(img_poly).astype(np.int32)
        color = COLORS[name]
        cv2.polylines(overlay, [img_poly_int], isClosed=True, color=color, thickness=2)
        results[name] = img_poly_int.tolist()

    blended = cv2.addWeighted(overlay, 0.85, image, 0.15, 0)
    cv2.imwrite(f"{out_prefix}.jpg", blended)
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{image_path.name}] detected {len(results)}/5 nails -> {out_prefix}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/nail_seg.keras")
    ap.add_argument("--data", default="dataset/test")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--crop-size", type=int, default=128)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, custom_objects=CUSTOM_OBJECTS)
    out_dir = Path(args.out)

    evaluate_test_set(model, Path(args.data), out_dir, args.crop_size)

    images_dir = Path(__file__).parent / "images"
    for img_name in ("img1.jpg", "img2.jpg"):
        img_path = images_dir / img_name
        if img_path.exists():
            run_end_to_end(model, img_path, out_dir / f"e2e_{Path(img_name).stem}", args.crop_size)
    print(f"\nCompare e2e_*.jpg in {out_dir} against the existing output.jpg (GrabCut baseline).")


if __name__ == "__main__":
    main()
