import json
import importlib.util
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_detector_path = Path("baselines/pyautoflip-main/pyautoflip/detection/mediapipe_object_detector.py")
spec = importlib.util.spec_from_file_location("mp_objdet", _detector_path)
mp_objdet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp_objdet)
ObjectDetector = mp_objdet.ObjectDetector

VIDEO_PATH = "data/in/work/1WBaw2e9Zrg_first10s_h264.mp4"
POSE_MODEL_PATH = "models/mp_tasks/pose_landmarker/pose_landmarker_lite.task"

OUT_DIR = Path("data/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASENAME = Path(VIDEO_PATH).stem
OVERLAY_PATH = OUT_DIR / f"{BASENAME}_pose_overlay.mp4"
DEBUG_JSON_PATH = OUT_DIR / f"{BASENAME}_pose_debug.json"
JSONL_PATH = OUT_DIR / f"{BASENAME}_pose_tags.jsonl"

DETECT_EVERY_N = 3
BOX_EXPAND = 0.20

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

POSE_CONNECTIONS = list(vision.PoseLandmarksConnections.POSE_LANDMARKS)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0

def det_to_xyxy(det, frame_w, frame_h):
    x1 = det["x"] * frame_w
    y1 = det["y"] * frame_h
    x2 = x1 + det["width"] * frame_w
    y2 = y1 + det["height"] * frame_h
    return [x1, y1, x2, y2]

def expand_box(box, frame_w, frame_h, frac=0.20):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * frac
    pad_y = bh * frac
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(frame_w, int(round(x2 + pad_x)))
    y2 = min(frame_h, int(round(y2 + pad_y)))
    return [x1, y1, x2, y2]

def choose_subject_box(person_dets, frame_w, frame_h, prev_box):
    if not person_dets:
        return prev_box

    boxes = [det_to_xyxy(d, frame_w, frame_h) for d in person_dets]

    if prev_box is None:
        areas = [max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
        return boxes[int(np.argmax(areas))]

    scores = []
    for b in boxes:
        area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        scores.append(3.0 * iou_xyxy(prev_box, b) + 0.000001 * area)
    return boxes[int(np.argmax(scores))]

def box_norm(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    return {
        "x1": x1 / frame_w,
        "y1": y1 / frame_h,
        "x2": x2 / frame_w,
        "y2": y2 / frame_h,
    }

def draw_overlay(frame, box, landmarks):
    h, w = frame.shape[:2]

    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "focus_subject",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if landmarks:
        pts = []
        for lm in landmarks:
            px = int(round(lm["x"] * w))
            py = int(round(lm["y"] * h))
            pts.append((px, py))

        for conn in POSE_CONNECTIONS:
            a = conn.start
            b = conn.end
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)

        for px, py in pts:
            cv2.circle(frame, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)

    return frame

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_ms = int(round(1000.0 / fps)) if fps > 0 else 33

writer = cv2.VideoWriter(
    str(OVERLAY_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 30.0,
    (frame_w, frame_h),
)

detector = ObjectDetector(
    model_asset_path=str(Path("baselines/9-16-conversion-joe/models/mp_tasks/object_detector/efficientdet_lite0.tflite").resolve())
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

frames_out = []
jsonl_records = []
prev_box = None

with PoseLandmarker.create_from_options(options) as landmarker:
    for frame_idx in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = int(round(frame_idx * 1000.0 / fps)) if fps > 0 else frame_idx * 33

        run_det = (frame_idx % DETECT_EVERY_N == 0) or (prev_box is None)
        person_dets = []

        if run_det:
            all_dets = detector.detect(frame)
            person_dets = [d for d in all_dets if d.get("class") == "person"]
            prev_box = choose_subject_box(person_dets, frame_w, frame_h, prev_box)

        use_box = prev_box
        landmarks_out = []

        if use_box is not None:
            x1, y1, x2, y2 = expand_box(use_box, frame_w, frame_h, frac=BOX_EXPAND)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    crop_h, crop_w = crop.shape[:2]
                    for i, lm in enumerate(lms):
                        gx = (x1 + lm.x * crop_w) / frame_w
                        gy = (y1 + lm.y * crop_h) / frame_h
                        landmarks_out.append({
                            "id": i,
                            "name": LANDMARK_NAMES[i],
                            "x": float(gx),
                            "y": float(gy),
                            "z": float(lm.z),
                            "visibility": float(getattr(lm, "visibility", 0.0)),
                            "presence": float(getattr(lm, "presence", 0.0)),
                        })

        overlay = frame.copy()
        overlay = draw_overlay(overlay, use_box, landmarks_out)
        writer.write(overlay)

        box_record = box_norm(use_box, frame_w, frame_h) if use_box is not None else None

        frame_record = {
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms,
            "focus_subject_id": "track_0",
            "focus_method": "pyautoflip_mediapipe_person_largest_then_iou_v0",
            "subject_box": box_record,
            "landmarks": landmarks_out,
        }
        frames_out.append(frame_record)

        jsonl_records.append({
            "type": "tag",
            "data": {
                "tag": "pose_focus_subject",
                "track": "pose_landmarks",
                "start_time": timestamp_ms,
                "end_time": timestamp_ms + frame_ms,
                "source_media": VIDEO_PATH,
                "frame_info": {
                    "frame_idx": frame_idx,
                    "box": box_record,
                },
                "additional_info": {
                    "focus_subject_id": "track_0",
                    "focus_method": "pyautoflip_mediapipe_person_largest_then_iou_v0",
                    "landmarks": landmarks_out,
                },
            },
        })

jsonl_records.append({
    "type": "progress",
    "data": {
        "source_media": VIDEO_PATH
    }
})

with open(DEBUG_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "source_video": VIDEO_PATH,
        "focus_subject_id": "track_0",
        "frame_count": len(frames_out),
        "frames": frames_out,
    }, f, indent=2)

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for row in jsonl_records:
        f.write(json.dumps(row) + "\n")

cap.release()
writer.release()

print(f"wrote overlay: {OVERLAY_PATH}")
print(f"wrote debug json: {DEBUG_JSON_PATH}")
print(f"wrote tag jsonl: {JSONL_PATH}")
print(f"frames: {len(frames_out)}")
print(f"frames_with_pose: {sum(1 for x in frames_out if x['landmarks'])}")
