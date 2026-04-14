# Pose Focus Batch Pipeline

## Overview

This repo runs a **focus-first, pose-second** batch pipeline over a folder of input videos and writes four primary outputs per clip:

- `*_pose_overlay.mp4`
- `*_pose_debug.json`
- `*_focus_debug.json`
- `*_pose_final.jsonl`

The goal is **not** generic multi-person pose estimation. The goal is to produce pose for **one intended subject per frame**, keep that subject stable over time, and write outputs that are easy to review and easy to consume downstream.

At a high level, each clip goes through this sequence:

1. detect candidate people
2. build lightweight tracks
3. choose one focus subject
4. smooth and stabilize that subject selection
5. run pose only on that selected subject ROI
6. write overlay, debug JSON, focus debug JSON, and final JSONL

Primary entrypoint:

- `scripts/pose_batch_final_v2.py`

## Philosophy

This project is intentionally **focus-first, pose-second**.

The hard part is usually **not** whether a pose model can output 33 landmarks. The hard part is whether those landmarks belong to the **correct person** and stay attached to that same person across frames.

That leads to a few design rules:

- **One subject only**
- **Continuity matters more than framewise greediness**
- **The focus selector matters more than the raw pose model**
- **Every run must be debuggable**
- **Pragmatism beats purity**

This is why the repo keeps separate artifacts for pose output and focus decision-making.

## What each major component is doing

### Qwen

Qwen is used as a **focus arbiter**, not as the pose model.

Specifically:

- candidate people are detected and tracked first
- those candidate tracks are scored heuristically
- when the top candidates are too close to call, the pipeline builds a labeled board image showing the competing tracks
- that board is sent to `qwen3-vl:30b` through Ollama
- Qwen chooses **which candidate person should stay the focus**

So Qwen is answering a question like:

> "Out of these 2 or 3 candidate people, which one is the intended main subject for this clip right now?"

Qwen is **not** being used to infer 33 keypoints. It is only helping with ambiguous subject selection.

### MediaPipe object detector

The pre-pose person detection step uses a vendored MediaPipe-based object detector derived from the `pyautoflip` side of the earlier work.

That detector is used to:

- find `person` candidates
- provide bounding boxes
- support lightweight track construction
- give the focus selector something concrete to work with before pose

This detector runs before pose and is part of the **subject restriction** step.

### MediaPipe Tasks PoseLandmarker

The pose stage uses **MediaPipe Tasks PoseLandmarker**.

That is the part that actually produces the body landmarks.

It is used after focus has already been chosen:

- take the chosen subject box
- expand it slightly into an ROI
- run PoseLandmarker on that ROI
- convert ROI-relative pose points back to full-frame normalized coordinates
- write those coordinates to JSONL and debug JSON
- draw them back onto the overlay video

So the pose model is intentionally only run on the selected subject, not on every visible person.

### Python-side tracking and smoothing

The pipeline also does a meaningful amount of work in Python before the pose model is called.

That logic includes:

- H.264 normalization for troublesome inputs
- reduced-cadence person detection
- IoU-based lightweight tracking
- track persistence / miss tolerance
- switch hysteresis
- focus lock logic
- EMA box smoothing

This is what keeps the chosen subject from flickering between people every few frames.

## Workflow

For each input video:

### 1. Input scan

Read `.mp4` files from:

- `data/in/youtube_tests`

### 2. Working-video normalization

If the source is not already H.264, create an H.264 working copy under that clip's `work/` folder.

This was added because AV1 decode was unreliable in the active Python/OpenCV path during development.

### 3. Candidate person detection

Use the vendored MediaPipe object detector.

Important deviation from a naive/default path:

- detection does **not** need to happen on every frame
- only `person` detections are retained
- the detector is used as a pre-pose filter, not as an end in itself

### 4. Tracking

Build lightweight tracks with IoU matching.

Each track keeps:

- recent box history
- smoothed box state
- hit count
- miss count
- last-seen frame

### 5. Focus selection

Score recent candidate tracks using a weighted combination of:

- presence
- average area
- motion
- center distance

When the top candidates are too close, call Qwen with a visual board for arbitration.

### 6. Focus stabilization

Apply several anti-flicker measures:

- focus lock persistence
- missing-subject grace period
- switch confirmation / hysteresis
- EMA smoothing on the chosen subject box

This is one of the biggest changes from the default "largest person each frame" approach.

### 7. Pose estimation

Run MediaPipe Tasks `PoseLandmarker` on the chosen subject ROI.

Then:

- map ROI coordinates back into full-frame normalized coordinates
- keep the 33-point pose definition
- store the coordinates in the final JSONL shape expected downstream

### 8. Output writing

Write:

- `*_pose_overlay.mp4`
- `*_pose_debug.json`
- `*_focus_debug.json`
- `*_pose_final.jsonl`

## What changed from the naive/default path

Compared with "run pose on everything" or "pick the biggest person every frame", this repo changes several important things:

- `pyautoflip`-derived logic is used selectively for **pre-pose detection**, not as an untouched full AutoFlip pipeline
- MediaPipe **Tasks** replaced older `solutions`-style assumptions
- non-H.264 inputs are normalized to a working copy first
- subject identity is stabilized with **track lock, hysteresis, and smoothing**
- Qwen is used only when the focus decision is ambiguous
- pose is run only on the chosen subject ROI

## Repo layout

```text
scripts/
  pose_batch_final_v2.py
  setup_models.sh
  run_batch.sh

src/pose_pipeline/
  mediapipe_object_detector.py

data/in/youtube_tests/
  *.mp4

data/out/final_batch_v2/
  <clip_name>/
    <clip_name>_pose_overlay.mp4
    <clip_name>_pose_debug.json
    <clip_name>_focus_debug.json
    <clip_name>_pose_final.jsonl
    work/
      <clip_name>_h264.mp4
    qwen/
      optional arbitration boards

models/mp_tasks/object_detector/
  efficientdet_lite0.tflite

models/mp_tasks/pose_landmarker/
  pose_landmarker_lite.task
```

## Final JSONL shape

Each frame-level tag looks like this:

```json
{
  "type": "tag",
  "data": {
    "tag": "track_0",
    "start_time": 0,
    "end_time": 33,
    "track": "pose_detection",
    "frame_info": {
      "frame_idx": 0,
      "box": {
        "x1": 0.123456,
        "x2": 0.345678,
        "y1": 0.111111,
        "y2": 0.888888
      }
    },
    "additional_info": {
      "pose": {
        "Nose": [0.5, 0.2],
        "LEyeIn": [0.49, 0.19]
      },
      "other_info": {
        "focus_subject_id": "track_0",
        "focus_method": "pyautoflip_mpdet_lock_hysteresis_qwen30b_v2",
        "profile": "generic_single_subject"
      }
    },
    "source_media": "/abs/path/to/input.mp4"
  }
}
```

A final `progress` line is also appended per source file.

## How to run

```bash
conda activate pose-gpu
cd /home/elv-ryan/projects/pose_focus_tagger

bash scripts/setup_models.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/run_batch.sh
```

Outputs appear under:

```text
data/out/final_batch_v2/
```

## Requirements

Python requirements are listed in `requirements.txt`.

Non-Python requirements:

- system FFmpeg with AV1 decode support
- Ollama running locally if `QWEN_ENABLED = True`
- NVIDIA GPU recommended for the broader pipeline runtime

## GitHub notes

This repo intentionally ignores:

- raw videos
- generated outputs
- downloaded model assets
- temporary work folders
- older baselines and archives

That keeps the repo smaller and safer to push.
