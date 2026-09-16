"""
Shared hand-landmark detection + per-finger crop/coordinate-mapping logic.

Used by both the inference pipeline (main.py) and the training data-prep
scripts (see training_plan.md, Giai doan 0/1) so a trained segmentation
model is trained on exactly the crops it will see at inference time --
same MediaPipe landmarks, same crop geometry, same rotation.
"""

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


def detect_landmarks(image_bgr, model_path=MODEL_PATH):
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
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


def crop_finger(image, tip, joint, interp=cv2.INTER_LINEAR):
    """Rotate `image` so the tip-joint axis points straight up, then crop an
    axis-aligned box around the (overshot) tip. Returns the crop and the
    inverse-mapping needed to bring local points back to image coordinates.

    `image` can be a BGR photo or a single-channel mask (e.g. when reusing
    this during training data-prep to crop a ground-truth mask the same way
    as its source photo) -- pass `interp=cv2.INTER_NEAREST` for masks so the
    warp doesn't blur binary values.
    """
    d = tip - joint
    seg_len = float(np.linalg.norm(d))
    if seg_len < 1e-3:
        return None

    angle_current = math.degrees(math.atan2(d[1], d[0]))  # image coords, y down
    # we want d to end up pointing to (0, -1) i.e. angle -90
    rot_deg = -90 - angle_current

    h, w = image.shape[:2]
    center = (float(tip[0]), float(tip[1]))
    M = cv2.getRotationMatrix2D(center, -rot_deg, 1.0)  # cv2: positive = CCW
    rotated = cv2.warpAffine(image, M, (w, h), flags=interp,
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


def local_to_image_points(points_local, offset, M):
    """crop-local -> rotated-image -> original-image coordinates."""
    x0, y0 = offset
    pts_rot = points_local.astype(np.float64) + np.array([x0, y0])
    Minv = cv2.invertAffineTransform(M)
    ones = np.ones((pts_rot.shape[0], 1))
    pts_h = np.hstack([pts_rot, ones])
    pts_img = pts_h @ Minv.T
    return pts_img
