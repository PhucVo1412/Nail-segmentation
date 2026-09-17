"""
Stage 1 (training_plan.md): build (crop_img, crop_mask) pairs for training,
using the *exact same* crop_finger() used at inference time in main.py so the
model trains on the distribution it will see in production (see nail_lib.py
docstring).

Only the vpapenko/nails-segmentation-dataset (nails_segmentation/, CC0-1.0)
is registered below. nails/nails/ (likely Golbstein/Fingernails-Segmentation,
no LICENSE file) is deliberately NOT registered -- add it here only once its
license status is resolved.

Usage:
    python prepare_dataset.py --sources vpapenko --out dataset
"""

import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from nail_lib import FINGERS, crop_finger, detect_landmarks

ROOT = Path(__file__).parent


@dataclass
class DatasetSource:
    name: str
    image_dir: Path
    mask_dir: Path
    image_ext: str = ".jpg"
    mask_ext: str = ".jpg"


SOURCES = {
    "vpapenko": DatasetSource(
        name="vpapenko",
        image_dir=ROOT / "nails_segmentation" / "images",
        mask_dir=ROOT / "nails_segmentation" / "labels",
    ),
}


def list_pairs(source: DatasetSource):
    pairs = []
    for img_path in sorted(source.image_dir.glob(f"*{source.image_ext}")):
        mask_path = source.mask_dir / img_path.name
        if not mask_path.exists():
            print(f"  [{source.name}] no mask for {img_path.name}, skipping")
            continue
        pairs.append((img_path, mask_path))
    return pairs


def process_pair(source_name, img_path, mask_path, crop_size, min_fg_frac, sample_idx, stats):
    image = cv2.imread(str(img_path))  # BGR
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        stats["read_failed"] += 1
        return []

    if sample_idx < 5:
        print(f"  [sanity check] {img_path.name}: mask unique values "
              f"(sample) = {np.unique(mask)[:10]}, min={mask.min()}, max={mask.max()}")

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Masks are lossy JPEG -- binarize with a threshold, don't assume clean 0/255.
    mask_bin = (mask > 127).astype(np.uint8) * 255

    landmarks = detect_landmarks(image)
    if landmarks is None:
        stats["no_hand"] += 1
        return []

    out_crops = []
    for finger, (tip_idx, joint_idx) in FINGERS.items():
        tip, joint = landmarks[tip_idx], landmarks[joint_idx]

        crop_img_info = crop_finger(image, tip, joint, interp=cv2.INTER_LINEAR)
        crop_mask_info = crop_finger(mask_bin, tip, joint, interp=cv2.INTER_NEAREST)
        if crop_img_info is None or crop_mask_info is None:
            stats["geometry_failed"] += 1
            continue

        img_crop = cv2.resize(crop_img_info["crop"], (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        mask_crop = cv2.resize(crop_mask_info["crop"], (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        mask_crop = (mask_crop > 127).astype(np.uint8) * 255

        fg_frac = float((mask_crop == 255).mean())
        if fg_frac < min_fg_frac:
            stats["empty_mask"] += 1
            continue

        out_crops.append((finger, img_crop, mask_crop, fg_frac))
        stats["kept"] += 1

    return out_crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["vpapenko"], choices=list(SOURCES.keys()))
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--min-fg-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = ROOT / args.out
    for split in ("train", "val", "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "masks").mkdir(parents=True, exist_ok=True)

    stats = {"no_hand": 0, "geometry_failed": 0, "empty_mask": 0, "kept": 0, "read_failed": 0}
    all_pairs = []  # (source_name, img_path, mask_path)
    for source_name in args.sources:
        source = SOURCES[source_name]
        pairs = list_pairs(source)
        print(f"[{source_name}] found {len(pairs)} image/mask pairs")
        all_pairs.extend((source_name,) + p for p in pairs)

    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)

    n = len(all_pairs)
    n_val = round(n * args.val_frac)
    n_test = round(n * args.test_frac)
    n_train = n - n_val - n_test
    split_of_image = {}
    for i, (source_name, img_path, _) in enumerate(all_pairs):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        split_of_image[(source_name, img_path)] = split

    manifest_rows = []
    sample_idx = 0
    for source_name, img_path, mask_path in all_pairs:
        split = split_of_image[(source_name, img_path)]
        crops = process_pair(source_name, img_path, mask_path, args.crop_size,
                              args.min_fg_frac, sample_idx, stats)
        sample_idx += 1

        for finger, img_crop, mask_crop, fg_frac in crops:
            out_name = f"{source_name}__{img_path.stem}__{finger}.png"
            cv2.imwrite(str(out_root / split / "images" / out_name), img_crop)
            cv2.imwrite(str(out_root / split / "masks" / out_name), mask_crop)
            manifest_rows.append({
                "split": split, "source": source_name, "image_stem": img_path.stem,
                "finger": finger, "fg_frac": round(fg_frac, 4), "file": out_name,
            })

    with open(out_root / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "source", "image_stem", "finger", "fg_frac", "file"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n--- Summary ---")
    print(f"Source images: {n} (no-hand skipped: {stats['no_hand']}, read failed: {stats['read_failed']})")
    print(f"Finger crops attempted: {n * 5 - stats['no_hand'] * 5}")
    print(f"  geometry failed: {stats['geometry_failed']}")
    print(f"  empty mask filtered: {stats['empty_mask']}")
    print(f"  kept: {stats['kept']}")
    for split in ("train", "val", "test"):
        count = sum(1 for r in manifest_rows if r["split"] == split)
        print(f"  {split}: {count} crops")
    print(f"\nWrote dataset to {out_root}")


if __name__ == "__main__":
    main()
