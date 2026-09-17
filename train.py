"""
Stage 2 (augmentation) + Stage 3 (train baseline), training_plan.md.

Reads the crops written by prepare_dataset.py, trains the mini U-Net from
nail_seg_model.py with on-the-fly augmentation, BCE+Dice loss, early stopping
on val IoU.

Usage:
    python train.py --data dataset --out models/nail_seg.keras
"""

import argparse
from pathlib import Path

import tensorflow as tf

from nail_seg_model import build_unet, bce_dice_loss

AUTOTUNE = tf.data.AUTOTUNE


def list_pairs(data_dir: Path, split: str):
    img_dir = data_dir / split / "images"
    mask_dir = data_dir / split / "masks"
    img_paths = sorted(str(p) for p in img_dir.glob("*.png"))
    mask_paths = [str(mask_dir / Path(p).name) for p in img_paths]
    return img_paths, mask_paths


def load_pair(img_path, mask_path):
    img_bytes = tf.io.read_file(img_path)
    # Crops were written with cv2.imwrite (BGR bytes) -- decode_png just returns
    # the stored channels in file order, so no BGR/RGB conversion here keeps
    # channel order identical to main.py's inference path.
    img = tf.image.decode_png(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0,1]

    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_bytes, channels=1)
    mask = tf.cast(mask > 127, tf.float32)
    return img, mask


def make_augmenter(crop_size):
    geo_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(15 / 360, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.1, fill_mode="reflect"),
    ])

    def augment(img, mask):
        combined = tf.concat([img, mask], axis=-1)
        combined = geo_aug(combined)
        img_aug, mask_aug = combined[..., :3], combined[..., 3:]
        mask_aug = tf.cast(mask_aug > 0.5, tf.float32)

        img_aug = tf.image.random_brightness(img_aug, 0.15)
        img_aug = tf.image.random_contrast(img_aug, 0.85, 1.15)
        img_aug = tf.image.random_hue(img_aug, 0.03)
        img_aug = tf.image.random_saturation(img_aug, 0.8, 1.2)
        img_aug = tf.clip_by_value(img_aug, 0.0, 1.0)

        img_aug.set_shape([crop_size, crop_size, 3])
        mask_aug.set_shape([crop_size, crop_size, 1])
        return img_aug, mask_aug

    return augment


def make_dataset(data_dir, split, crop_size, batch_size, augment_fn=None, shuffle=False):
    img_paths, mask_paths = list_pairs(data_dir, split)
    if not img_paths:
        raise RuntimeError(f"No crops found for split '{split}' under {data_dir} -- "
                            f"run prepare_dataset.py first.")
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths), seed=42)
    ds = ds.map(load_pair, num_parallel_calls=AUTOTUNE)
    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, len(img_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--base-filters", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/nail_seg.keras")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--run-name", default="run1")
    args = ap.parse_args()

    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    data_dir = Path(args.data)
    augment_fn = make_augmenter(args.crop_size)
    train_ds, n_train = make_dataset(data_dir, "train", args.crop_size, args.batch_size,
                                      augment_fn=augment_fn, shuffle=True)
    val_ds, n_val = make_dataset(data_dir, "val", args.crop_size, args.batch_size)
    print(f"train crops: {n_train}, val crops: {n_val}")

    model = build_unet(input_shape=(args.crop_size, args.crop_size, 3), base_filters=args.base_filters)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=bce_dice_loss,
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="iou")],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out_path), monitor="val_iou", mode="max",
                                            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_iou", mode="max", patience=15,
                                          restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_iou", mode="max", factor=0.5,
                                              patience=7, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=str(Path(args.logdir) / args.run_name)),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    print(f"\nBest model saved to {out_path}")


if __name__ == "__main__":
    main()
