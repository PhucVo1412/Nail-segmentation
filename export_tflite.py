"""
Stage 5 (training_plan.md): export the trained Keras model to TFLite
(float16 first, optional int8), re-verify mIoU doesn't collapse, and write
the input/output spec for the mobile team.

Usage:
    python export_tflite.py --model models/nail_seg.keras --data dataset/test --out models/nail_seg_fp16.tflite
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from nail_seg_model import CUSTOM_OBJECTS, per_image_iou


def load_test_arrays(test_dir: Path, crop_size: int):
    img_dir, mask_dir = test_dir / "images", test_dir / "masks"
    names = sorted(p.name for p in img_dir.glob("*.png"))
    xs, ys = [], []
    for name in names:
        img = cv2.imread(str(img_dir / name))
        mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        xs.append(cv2.resize(img, (crop_size, crop_size)))
        ys.append((cv2.resize(mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)[..., None])
    x = np.stack(xs).astype(np.float32) / 255.0
    y = np.stack(ys)
    return x, y


def run_tflite(interpreter, x):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.resize_tensor_input(input_details["index"], x.shape)
    interpreter.allocate_tensors()
    in_dtype = input_details["dtype"]
    interpreter.set_tensor(input_details["index"], x.astype(in_dtype))
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])


def convert_float16(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def convert_int8(model, rep_images):
    def rep_dataset():
        for img in rep_images:
            yield [img[None].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/nail_seg.keras")
    ap.add_argument("--data", default="dataset/test")
    ap.add_argument("--train-data", default="dataset/train", help="source for int8 representative dataset")
    ap.add_argument("--out", default="models/nail_seg_fp16.tflite")
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--try-int8", action="store_true")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, custom_objects=CUSTOM_OBJECTS)
    x_test, y_test = load_test_arrays(Path(args.data), args.crop_size)

    keras_pred = model.predict(x_test, verbose=0)
    keras_iou = per_image_iou(y_test, keras_pred).mean()
    print(f"Keras float32 mean IoU: {keras_iou:.4f}")

    fp16_bytes = convert_float16(model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(fp16_bytes)
    print(f"Wrote {out_path} ({len(fp16_bytes) / 1024:.1f} KB)")

    interp = tf.lite.Interpreter(model_content=fp16_bytes)
    fp16_pred = run_tflite(interp, x_test)
    fp16_iou = per_image_iou(y_test, fp16_pred).mean()
    print(f"TFLite float16 mean IoU: {fp16_iou:.4f} (drop: {keras_iou - fp16_iou:.4f})")

    chosen_path = out_path
    if args.try_int8:
        train_x, _ = load_test_arrays(Path(args.train_data), args.crop_size)
        rep_images = train_x[:200]
        int8_bytes = convert_int8(model, rep_images)
        int8_path = out_path.with_name(out_path.stem.replace("fp16", "int8") + ".tflite")
        int8_path.write_bytes(int8_bytes)
        print(f"Wrote {int8_path} ({len(int8_bytes) / 1024:.1f} KB)")

        interp_int8 = tf.lite.Interpreter(model_content=int8_bytes)
        in_detail = interp_int8.get_input_details()[0]
        scale, zero_point = in_detail["quantization"]
        x_q = (x_test / scale + zero_point).astype(np.uint8) if scale else x_test.astype(np.uint8)
        int8_pred_raw = run_tflite(interp_int8, x_q)
        out_detail = interp_int8.get_output_details()[0]
        out_scale, out_zero = out_detail["quantization"]
        int8_pred = (int8_pred_raw.astype(np.float32) - out_zero) * out_scale if out_scale else int8_pred_raw.astype(np.float32)
        int8_iou = per_image_iou(y_test, int8_pred).mean()
        print(f"TFLite int8 mean IoU: {int8_iou:.4f} (drop: {keras_iou - int8_iou:.4f})")

        if keras_iou - int8_iou > keras_iou - fp16_iou + 0.02:
            print("int8 degrades accuracy noticeably more than float16 -- keeping float16 as the default.")
        else:
            chosen_path = int8_path

    spec = {
        "input_shape": [1, args.crop_size, args.crop_size, 3],
        "input_normalization": "[0, 1] float32 (pixel / 255.0)",
        "input_channel_order": "BGR (matches nail_lib.py / OpenCV, NOT RGB -- do not convert)",
        "output_shape": [1, args.crop_size, args.crop_size, 1],
        "output_activation": "sigmoid",
        "output_threshold": 0.5,
        "note": "Input must be a per-finger crop produced by nail_lib.crop_finger(), "
                "not a full-frame image -- replicate that crop geometry natively on mobile.",
        "chosen_model_file": chosen_path.name,
    }
    spec_path = out_path.parent / "nail_seg_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"Wrote spec to {spec_path}")


if __name__ == "__main__":
    main()
