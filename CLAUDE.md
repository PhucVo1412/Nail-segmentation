# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file research prototype (`main.py`) that detects the contour/outline points of all 5 fingernails from a hand photo, using only pretrained/off-the-shelf pieces — no custom-trained model yet. It exists to demonstrate the pipeline end-to-end before investing in training a real nail-segmentation model (see `readme.md` for the full research context and decisions).

The broader goal (from `readme.md`): a VR/AI app needs a model (`.tflite` or similar) that outputs points outlining each of the 5 fingernails from a live camera feed, so the app can draw polygons over them (AR nail try-on / nail art). The chosen technical direction is **segmentation + contour extraction** (model outputs a per-finger nail mask, app code extracts contour points), not a model that directly regresses fixed landmark points.

## Running

```
python main.py <input_image> [output_prefix]
```

- Requires `models/hand_landmarker.task` next to the script (pretrained MediaPipe Hand Landmarker model — already present in this repo).
- Dependencies: `mediapipe`, `opencv-python`, `numpy` — already installed in `.venv`. Activate it or invoke `.venv/Scripts/python.exe` directly.
- Sample input images are in `images/` (`img1.jpg`, `img2.jpg`).
- Output: `<output_prefix>.jpg` (overlay of detected nail polygons on the original image) and `<output_prefix>.json` (per-finger polygon points, both pixel and normalized coordinates).

There is no build step, test suite, or linter configured in this repo.

## Architecture (pipeline in `main.py`)

The whole thing is one linear pipeline run per image, per finger:

1. **`detect_landmarks`** — runs MediaPipe `HandLandmarker` (pretrained) on the full image, returns all 21 hand landmarks in pixel coordinates.
2. **`crop_finger`** — for one finger, takes the tip landmark and the nearest joint landmark, rotates the *entire original image* around the tip point so the tip→joint axis points straight up, then crops an axis-aligned box around the tip. Returns the crop plus the rotation matrix and crop offset, both needed later to map points back. Key gotcha documented in code: the tip landmark sits ON/mid the nail (not below it), so the crop must extend generously *beyond* the tip too, not just toward the joint — tunable via `UP_FACTOR`/`DOWN_FACTOR`/`WIDTH_FACTOR`.
3. **`segment_nail`** — **placeholder for a real trained segmentation model** (a mini U-Net/TFLite model, per the proposal in `readme.md`). Currently classical CV: samples the crop's outer ring as a skin-color reference, scores every pixel by color distance (LAB space) from that reference, Otsu-thresholds it for a deterministic seed mask, then refines with `GrabCut` (`GC_INIT_WITH_MASK`). Plain rect-seeded GrabCut was tried first and rejected — its internal GMM/k-means init is randomized, so identical input produced different results across runs; seeding it with a deterministic mask fixed that. Ends with `findContours` + `approxPolyDP` to get a simplified polygon in crop-local coordinates.
4. **`local_to_image_points`** — maps polygon points from crop-local → rotated-image → original-image coordinates, undoing the crop offset and rotation from step 2 (via `cv2.invertAffineTransform`).
5. **`main`** — orchestrates the above per finger (`FINGERS` dict maps finger name to MediaPipe tip/joint landmark index pairs), draws overlays, and writes the `.jpg`/`.json` outputs.

**When replacing the placeholder with a real model**: only `segment_nail()` should need to change (swap the classical-CV body for a TFLite interpreter call that takes the crop and returns a binary mask) — the crop/rotate and coordinate-mapping logic around it is meant to stay as-is.

## Key context from readme.md

- Dataset situation: no proprietary dataset yet; candidate public sources are small (50–379 images) and likely not diverse enough (skin tone, lighting, polish color) for production — listed as a phase-2 task.
- Reference production architecture: Banuba's "Nail Polish Try-On" paper (arXiv:1906.02222) — ~94.5 mIoU at 29.8ms/frame on iPad Pro; also predicts base→tip direction alongside the mask to keep polygon orientation consistent.
- Target export format: `.tflite` for Android; convert to CoreML/ONNX if iOS is also needed. The contour→polygon step is meant to live in app code, not inside the model graph.
