#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p \
  "$ROOT/models/mp_tasks/object_detector" \
  "$ROOT/models/mp_tasks/pose_landmarker"

if [[ -f "$ROOT/baselines/9-16-conversion-joe/models/mp_tasks/object_detector/efficientdet_lite0.tflite" ]]; then
  cp "$ROOT/baselines/9-16-conversion-joe/models/mp_tasks/object_detector/efficientdet_lite0.tflite" \
     "$ROOT/models/mp_tasks/object_detector/efficientdet_lite0.tflite"
elif [[ ! -f "$ROOT/models/mp_tasks/object_detector/efficientdet_lite0.tflite" ]]; then
  wget -O "$ROOT/models/mp_tasks/object_detector/efficientdet_lite0.tflite" \
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
fi

if [[ ! -f "$ROOT/models/mp_tasks/pose_landmarker/pose_landmarker_lite.task" ]]; then
  wget -O "$ROOT/models/mp_tasks/pose_landmarker/pose_landmarker_lite.task" \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
fi

echo "models ready"
