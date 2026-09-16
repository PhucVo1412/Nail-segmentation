"""
Demo: detect 5 fingernail contour points from a hand photo, using only
off-the-shelf / pretrained pieces (no custom training):

  1. MediaPipe HandLandmarker (pretrained, Google) -> 21 hand landmarks
  2. Per-finger crop + rotate so the finger points "up"
  3. Classical CV segmentation inside the crop (GrabCut, OpenCV) to separate
     the nail blob from skin/background  <-- placeholder for the trained
     mini U-Net / TFLite segmentation model discussed in the proposal doc
  4. cv2.findContours + approxPolyDP -> simplified polygon points
  5. Map points back to original image coordinates, draw + export JSON

Run:
    python3 nail_points_demo.py <input_image> [output_prefix]

Requires: mediapipe, opencv-python, numpy
Needs models/hand_landmarker.task next to this script (see download note
at the bottom of this file / the chat message that shipped with this demo).

Landmark detection + per-finger crop/coordinate-mapping live in nail_lib.py,
shared with the training data-prep scripts (see training_plan.md) so a
trained model is trained on exactly the crops it sees at inference time.
"""

import sys
import json

import cv2
import numpy as np

from nail_lib import (
    FINGERS,
    MODEL_PATH,
    crop_finger,
    detect_landmarks,
    local_to_image_points,
)

GRABCUT_ITERS = 5
POLY_EPS_FRAC = 0.02   # approxPolyDP epsilon as a fraction of contour perimeter

COLORS = {
    "thumb":  (255, 0, 255),
    "index":  (0, 255, 255),
    "middle": (0, 255, 0),
    "ring":   (255, 128, 0),
    "pinky":  (255, 0, 0),
}


def segment_nail(crop):
    """Classical-CV stand-in for the trained segmentation model.

    Plain rect-seeded GrabCut turned out unreliable here (its internal GMM
    init has a randomized k-means step, so the *same* crop could segment on
    one run and fail on the next -- fine for a person cutout, not for an
    object this small). Instead: sample the crop's outer ring as a "skin"
    color reference, score every pixel by how far it is from that reference
    (deterministic), Otsu-threshold that distance map for a first guess, and
    only then hand GrabCut a real mask to refine -- much more stable.
    """
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    border = max(3, int(min(h, w) * 0.10))
    ring = np.zeros((h, w), np.uint8)
    ring[:border, :] = 1
    ring[-border:, :] = 1
    ring[:, :border] = 1
    ring[:, -border:] = 1

    ref_pixels = lab[ring == 1]
    mean, std = ref_pixels.mean(axis=0), ref_pixels.std(axis=0) + 1e-6
    dist = np.sqrt((((lab - mean) / std) ** 2).sum(axis=2))
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg_guess = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc_mask[fg_guess == 255] = cv2.GC_PR_FGD
    gc_mask[ring == 1] = cv2.GC_BGD  # the padded border is deterministically "skin/background"

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, gc_mask, None, bgd_model, fgd_model,
                    GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
        fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg = fg_guess  # GrabCut refinement failed; fall back to the Otsu guess

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.02 * w * h:
        return None

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, POLY_EPS_FRAC * peri, True)
    return approx.reshape(-1, 2)  # local (x, y) in crop coordinates


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 nail_points_demo.py <input_image> [output_prefix]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "nail_points_out"

    if not MODEL_PATH.exists():
        print(f"Missing model file: {MODEL_PATH}\n"
              f"Download the pretrained hand_landmarker.task from Google's "
              f"MediaPipe model index (ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) "
              f"and place it there.")
        sys.exit(1)

    image = cv2.imread(in_path)
    if image is None:
        print(f"Could not read image: {in_path}")
        sys.exit(1)

    landmarks = detect_landmarks(image)
    if landmarks is None:
        print("No hand detected.")
        sys.exit(1)

    overlay = image.copy()
    results = {}

    for name, (tip_idx, joint_idx) in FINGERS.items():
        tip, joint = landmarks[tip_idx], landmarks[joint_idx]
        crop_info = crop_finger(image, tip, joint)
        if crop_info is None:
            print(f"[{name}] crop failed, skipping")
            continue

        local_poly = segment_nail(crop_info["crop"])
        if local_poly is None or len(local_poly) < 3:
            print(f"[{name}] segmentation failed, skipping")
            continue

        img_poly = local_to_image_points(local_poly, crop_info["offset"], crop_info["M"])
        img_poly_int = np.round(img_poly).astype(np.int32)

        color = COLORS[name]
        cv2.polylines(overlay, [img_poly_int], isClosed=True, color=color, thickness=2)
        for (x, y) in img_poly_int:
            cv2.circle(overlay, (int(x), int(y)), 3, color, -1)
        label_pt = tuple(np.round(tip).astype(int))
        cv2.putText(overlay, name, label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        h, w = image.shape[:2]
        results[name] = {
            "points_px": img_poly_int.tolist(),
            "points_norm": (img_poly / np.array([w, h])).round(4).tolist(),
        }

    blended = cv2.addWeighted(overlay, 0.85, image, 0.15, 0)
    out_img_path = f"{out_prefix}.jpg"
    out_json_path = f"{out_prefix}.json"
    cv2.imwrite(out_img_path, blended)
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Detected {len(results)}/5 nails.")
    print(f"Saved overlay -> {out_img_path}")
    print(f"Saved points  -> {out_json_path}")


if __name__ == "__main__":
    main()