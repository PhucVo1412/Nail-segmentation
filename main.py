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
"""

import sys
import json
import math
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# (tip landmark index, nearest-joint landmark index) per finger, per MediaPipe
# Hands topology: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
FINGERS = {
    "thumb":  (4, 3),
    "index":  (8, 7),
    "middle": (12, 11),
    "ring":   (16, 15),
    "pinky":  (20, 19),
}

# Tunable geometry (fractions of the tip-joint landmark segment length).
# Note: the tip landmark (4/8/12/16/20) sits ON the nail, not below it -- for
# long/extension nails it can even be mid-nail -- so the crop must extend
# generously *beyond* the tip too, not just up to it.
UP_FACTOR = 0.9        # crop extent beyond the tip landmark (toward the free edge)
DOWN_FACTOR = 0.6      # crop extent back toward the joint (covers the cuticle/base)
WIDTH_FACTOR = 0.65    # crop half-width across the finger axis
GRABCUT_ITERS = 5
POLY_EPS_FRAC = 0.02   # approxPolyDP epsilon as a fraction of contour perimeter

COLORS = {
    "thumb":  (255, 0, 255),
    "index":  (0, 255, 255),
    "middle": (0, 255, 0),
    "ring":   (255, 128, 0),
    "pinky":  (255, 0, 0),
}


def detect_landmarks(image_bgr):
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    h, w = image_bgr.shape[:2]
    pts = result.hand_landmarks[0]
    return np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float32)


def crop_finger(image_bgr, tip, joint):
    """Rotate the image so the tip-joint axis points straight up, then crop
    an axis-aligned box around the (overshot) tip. Returns the crop and the
    inverse-mapping needed to bring local points back to image coordinates."""
    d = tip - joint
    seg_len = float(np.linalg.norm(d))
    if seg_len < 1e-3:
        return None

    angle_current = math.degrees(math.atan2(d[1], d[0]))  # image coords, y down
    # we want d to end up pointing to (0, -1) i.e. angle -90
    rot_deg = -90 - angle_current

    h, w = image_bgr.shape[:2]
    center = (float(tip[0]), float(tip[1]))
    M = cv2.getRotationMatrix2D(center, -rot_deg, 1.0)  # cv2: positive = CCW
    rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # tip stays exactly at `center` after rotation (it's the rotation pivot);
    # after rotation the finger axis points toward -y, so "beyond the tip"
    # is smaller y and "back toward the joint" is larger y.
    cx, cy = center
    box_w = seg_len * WIDTH_FACTOR

    x0, x1 = int(cx - box_w), int(cx + box_w)
    y0, y1 = int(cy - seg_len * UP_FACTOR), int(cy + seg_len * DOWN_FACTOR)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None

    crop = rotated[y0:y1, x0:x1].copy()
    return {
        "crop": crop,
        "M": M,               # rotation matrix used on the full image
        "offset": (x0, y0),   # crop origin within the rotated image
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


def local_to_image_points(points_local, offset, M):
    """crop-local -> rotated-image -> original-image coordinates."""
    x0, y0 = offset
    pts_rot = points_local.astype(np.float64) + np.array([x0, y0])
    Minv = cv2.invertAffineTransform(M)
    ones = np.ones((pts_rot.shape[0], 1))
    pts_h = np.hstack([pts_rot, ones])
    pts_img = pts_h @ Minv.T
    return pts_img


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