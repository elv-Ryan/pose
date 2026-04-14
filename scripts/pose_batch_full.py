import base64
import importlib.util
import json
import math
import os
import re
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path("/home/elv-ryan/projects/pose_focus_tagger")
INPUT_DIR = PROJECT_ROOT / "data/in/youtube_tests"
OUTPUT_ROOT = PROJECT_ROOT / "data/out/full_batch"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

POSE_MODEL_PATH = (PROJECT_ROOT / "models/mp_tasks/pose_landmarker/pose_landmarker_lite.task").resolve()
DETECTOR_MODEL_PATH = (PROJECT_ROOT / "baselines/9-16-conversion-joe/models/mp_tasks/object_detector/efficientdet_lite0.tflite").resolve()

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3-vl:30b"
QWEN_ENABLED = True

DEFAULT_PROFILE = "generic_single_subject"
PROFILE_OVERRIDES = {
    # "a0XBHsSOEos": "dance",
    # "PjAFwiq9YvU": "rap",
    # "1WBaw2e9Zrg": "sports_downhill",
}

PROFILE_PRESETS = {
    "generic_single_subject": {
        "presence_w": 0.55,
        "area_w": 0.25,
        "motion_w": 0.10,
        "center_w": -0.15,
        "prompt": "Choose the one main person to keep as the focus subject for this clip. Favor continuity and the clearly intended primary subject.",
    },
    "dance": {
        "presence_w": 0.55,
        "area_w": 0.20,
        "motion_w": 0.15,
        "center_w": -0.20,
        "prompt": "Choose the primary dancer or performer to keep as the focus subject. Favor the main performer, not background dancers or audience.",
    },
    "rap": {
        "presence_w": 0.60,
        "area_w": 0.25,
        "motion_w": 0.05,
        "center_w": -0.20,
        "prompt": "Choose the main rapper or front performer to keep as the focus subject. Favor the central lead performer, not side people or crowd.",
    },
    "sports_downhill": {
        "presence_w": 0.45,
        "area_w": 0.15,
        "motion_w": 0.35,
        "center_w": -0.05,
        "prompt": "Choose the intended downhill athlete to keep as the focus subject. Favor the main rider or skater, not spectators or bystanders.",
    },
}

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

ANALYSIS_MAX_SIDE = 960
DETECT_FPS = 6.0
SELECT_EVERY_SEC = 2.0
RECENT_WINDOW_SEC = 2.0
SWITCH_CONFIRM_EVENTS = 2
FOCUS_MISSING_GRACE_SEC = 1.0
BOX_EXPAND = 0.20
TRACK_MAX_MISSES = 12
TRACK_IOU_THRESH = 0.20
TRACK_SMOOTH_ALPHA = 0.35
FOCUS_BOX_EMA_ALPHA = 0.25
QWEN_AMBIGUITY_MARGIN = 0.12
QWEN_TOP_K = 3

_detector_path = PROJECT_ROOT / "baselines/pyautoflip-main/pyautoflip/detection/mediapipe_object_detector.py"
spec = importlib.util.spec_from_file_location("mp_objdet", _detector_path)
mp_objdet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp_objdet)
ObjectDetector = mp_objdet.ObjectDetector


def run(cmd):
    subprocess.run(cmd, check=True)


def ffprobe_json(path: Path):
    cmd = [
        "/usr/bin/ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def ensure_h264(src: Path, work_dir: Path) -> Path:
    meta = ffprobe_json(src)
    streams = meta.get("streams", [])
    v0 = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v0 is None:
        raise RuntimeError(f"no video stream in {src}")
    codec = v0.get("codec_name", "")
    if codec == "h264":
        return src
    dst = work_dir / f"{src.stem}_h264.mp4"
    if dst.exists():
        return dst
    run([
        "/usr/bin/ffmpeg",
        "-y",
        "-i", str(src),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        str(dst),
    ])
    return dst


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


def resize_for_detection(frame):
    h, w = frame.shape[:2]
    max_side = max(h, w)
    scale = min(1.0, ANALYSIS_MAX_SIDE / max_side)
    if scale >= 0.999:
        return frame, 1.0
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h)), scale


def det_to_xyxy(det, det_w, det_h, inv_scale):
    x1 = det["x"] * det_w
    y1 = det["y"] * det_h
    x2 = x1 + det["width"] * det_w
    y2 = y1 + det["height"] * det_h
    return [
        float(x1 * inv_scale),
        float(y1 * inv_scale),
        float(x2 * inv_scale),
        float(y2 * inv_scale),
    ]


def expand_box(box, frame_w, frame_h, frac=0.20):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * frac
    pad_y = bh * frac
    return [
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(frame_w), x2 + pad_x),
        min(float(frame_h), y2 + pad_y),
    ]


def box_norm(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    return {
        "x1": float(x1 / frame_w),
        "y1": float(y1 / frame_h),
        "x2": float(x2 / frame_w),
        "y2": float(y2 / frame_h),
    }


def clamp_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(frame_w - 1), x1))
    y1 = max(0.0, min(float(frame_h - 1), y1))
    x2 = max(x1 + 1.0, min(float(frame_w), x2))
    y2 = max(y1 + 1.0, min(float(frame_h), y2))
    return [x1, y1, x2, y2]


class Track:
    def __init__(self, track_id, box, frame_idx):
        self.track_id = track_id
        self.last_box = np.array(box, dtype=np.float32)
        self.smooth_box = np.array(box, dtype=np.float32)
        self.last_seen = frame_idx
        self.hits = 1
        self.misses = 0
        self.history = {frame_idx: [float(x) for x in box]}

    def update(self, box, frame_idx):
        box = np.array(box, dtype=np.float32)
        self.last_box = box
        self.smooth_box = TRACK_SMOOTH_ALPHA * box + (1.0 - TRACK_SMOOTH_ALPHA) * self.smooth_box
        self.last_seen = frame_idx
        self.hits += 1
        self.misses = 0
        self.history[frame_idx] = [float(x) for x in self.smooth_box.tolist()]

    def mark_missed(self):
        self.misses += 1

    def current_box(self):
        return [float(x) for x in self.smooth_box.tolist()]


def greedy_assign(active_tracks, det_boxes):
    pairs = []
    for tid, tr in active_tracks.items():
        for di, box in enumerate(det_boxes):
            pairs.append((iou_xyxy(tr.current_box(), box), tid, di))
    pairs.sort(reverse=True, key=lambda x: x[0])

    assigned_tracks = set()
    assigned_dets = set()
    matches = []

    for iou, tid, di in pairs:
        if iou < TRACK_IOU_THRESH:
            continue
        if tid in assigned_tracks or di in assigned_dets:
            continue
        assigned_tracks.add(tid)
        assigned_dets.add(di)
        matches.append((tid, di))

    unmatched_tracks = [tid for tid in active_tracks.keys() if tid not in assigned_tracks]
    unmatched_dets = [di for di in range(len(det_boxes)) if di not in assigned_dets]
    return matches, unmatched_tracks, unmatched_dets


def track_stats(track, frame_start, frame_end, frame_w, frame_h):
    keys = [k for k in sorted(track.history.keys()) if frame_start <= k <= frame_end]
    if not keys:
        return None

    boxes = [track.history[k] for k in keys]
    centers = [((b[0] + b[2]) * 0.5 / frame_w, (b[1] + b[3]) * 0.5 / frame_h) for b in boxes]
    areas = [((b[2] - b[0]) * (b[3] - b[1])) / (frame_w * frame_h) for b in boxes]

    motion = 0.0
    if len(centers) > 1:
        diffs = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            diffs.append(math.sqrt(dx * dx + dy * dy))
        motion = float(np.mean(diffs))

    center_dist = float(np.mean([math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) for cx, cy in centers]))
    presence = len(keys)

    return {
        "track_id": track.track_id,
        "presence": presence,
        "area_mean": float(np.mean(areas)),
        "motion_mean": motion,
        "center_dist": center_dist,
        "last_seen": track.last_seen,
        "box": track.current_box(),
    }


def score_track(stats, profile_name):
    p = PROFILE_PRESETS[profile_name]
    score = 0.0
    score += p["presence_w"] * min(1.0, stats["presence"] / 8.0)
    score += p["area_w"] * min(1.0, stats["area_mean"] / 0.20)
    score += p["motion_w"] * min(1.0, stats["motion_mean"] / 0.06)
    score += p["center_w"] * stats["center_dist"]
    return float(score)


def make_qwen_board(frame, candidates, out_path):
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    for c in candidates:
        x1, y1, x2, y2 = [int(v) for v in c["box"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            canvas,
            f"TRACK_{c['track_id']}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    full_w = 960
    full = cv2.resize(canvas, (full_w, int(round(h * full_w / w))))
    crops = []
    crop_h = 220
    crop_w = max(220, full_w // max(1, len(candidates)))

    for c in candidates:
        x1, y1, x2, y2 = [int(v) for v in c["box"]]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (crop_w, crop_h))
        cv2.putText(
            crop,
            f"TRACK_{c['track_id']}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        crops.append(crop)

    if crops:
        strip = np.hstack(crops)
        if strip.shape[1] < full.shape[1]:
            pad = full.shape[1] - strip.shape[1]
            strip = cv2.copyMakeBorder(strip, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif strip.shape[1] > full.shape[1]:
            strip = cv2.resize(strip, (full.shape[1], strip.shape[0]))
        board = np.vstack([full, strip])
    else:
        board = full

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), board)
    return out_path


def qwen_choose_focus(board_path, candidates, profile_name, current_focus_id):
    prompt = PROFILE_PRESETS[profile_name]["prompt"]
    candidate_list = ", ".join([f"TRACK_{c['track_id']}" for c in candidates])
    current_focus = f"TRACK_{current_focus_id}" if current_focus_id is not None else "NONE"
    text = (
        f"{prompt}\n"
        f"Current focus: {current_focus}\n"
        f"Candidates shown: {candidate_list}\n"
        "Return exactly one line with only the chosen track id in this format:\n"
        "TRACK_<number>\n"
    )

    with open(board_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": text,
                "images": [image_b64],
            }
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "")
        m = re.search(r"TRACK_(\d+)", content)
        if not m:
            return None, content
        return int(m.group(1)), content
    except Exception as e:
        return None, repr(e)


def draw_overlay(frame, focus_box, landmarks, focus_track_id):
    h, w = frame.shape[:2]

    if focus_box is not None:
        x1, y1, x2, y2 = [int(v) for v in focus_box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            frame,
            f"focus TRACK_{focus_track_id}",
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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


def process_video(src_path: Path):
    base = src_path.stem
    profile_name = PROFILE_OVERRIDES.get(base, DEFAULT_PROFILE)
    out_dir = OUTPUT_ROOT / base
    work_dir = out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = ensure_h264(src_path, work_dir)

    overlay_path = out_dir / f"{base}_pose_overlay.mp4"
    debug_json_path = out_dir / f"{base}_pose_debug.json"
    tags_jsonl_path = out_dir / f"{base}_pose_tags.jsonl"
    focus_debug_path = out_dir / f"{base}_focus_debug.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_ms = int(round(1000.0 / fps))
    detect_every_n = max(1, int(round(fps / DETECT_FPS)))
    select_every_n = max(1, int(round(fps * SELECT_EVERY_SEC)))
    recent_window_n = max(1, int(round(fps * RECENT_WINDOW_SEC)))
    focus_missing_grace_n = max(1, int(round(fps * FOCUS_MISSING_GRACE_SEC)))

    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    detector = ObjectDetector(model_asset_path=str(DETECTOR_MODEL_PATH))

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    RunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    active_tracks = {}
    all_tracks = {}
    next_track_id = 0

    current_focus_id = None
    current_focus_last_seen = -10**9
    focus_box_ema = None
    pending_switch_id = None
    pending_switch_count = 0

    frames_out = []
    jsonl_records = []
    focus_events = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(round(frame_idx * 1000.0 / fps))
            run_det = (frame_idx % detect_every_n == 0)

            if run_det:
                det_frame, scale = resize_for_detection(frame)
                inv_scale = 1.0 / scale
                det_h, det_w = det_frame.shape[:2]
                raw_dets = detector.detect(det_frame)
                person_dets = [d for d in raw_dets if d.get("class") == "person"]
                det_boxes = [det_to_xyxy(d, det_w, det_h, inv_scale) for d in person_dets]

                matches, unmatched_tracks, unmatched_dets = greedy_assign(active_tracks, det_boxes)

                for tid, di in matches:
                    active_tracks[tid].update(det_boxes[di], frame_idx)

                for tid in unmatched_tracks:
                    active_tracks[tid].mark_missed()

                for di in unmatched_dets:
                    t = Track(next_track_id, det_boxes[di], frame_idx)
                    active_tracks[next_track_id] = t
                    all_tracks[next_track_id] = t
                    next_track_id += 1

                to_drop = [tid for tid, tr in active_tracks.items() if tr.misses > TRACK_MAX_MISSES]
                for tid in to_drop:
                    del active_tracks[tid]

            should_select = (
                frame_idx == 0
                or frame_idx % select_every_n == 0
                or (current_focus_id is None)
                or (frame_idx - current_focus_last_seen > focus_missing_grace_n)
            )

            if should_select:
                recent_start = max(0, frame_idx - recent_window_n)
                candidates = []
                for tid, tr in all_tracks.items():
                    stats = track_stats(tr, recent_start, frame_idx, frame_w, frame_h)
                    if not stats:
                        continue
                    if stats["presence"] < 2:
                        continue
                    stats["score"] = score_track(stats, profile_name)
                    candidates.append(stats)

                candidates.sort(key=lambda x: x["score"], reverse=True)

                if candidates:
                    top_candidates = candidates[:QWEN_TOP_K]
                    chosen_id = top_candidates[0]["track_id"]
                    qwen_text = None
                    qwen_board_path = None

                    ambiguous = len(top_candidates) > 1 and abs(top_candidates[0]["score"] - top_candidates[1]["score"]) < QWEN_AMBIGUITY_MARGIN

                    if QWEN_ENABLED and ambiguous:
                        qwen_board_path = out_dir / "qwen" / f"{base}_frame_{frame_idx:06d}.jpg"
                        make_qwen_board(frame, top_candidates, qwen_board_path)
                        qwen_id, qwen_text = qwen_choose_focus(qwen_board_path, top_candidates, profile_name, current_focus_id)
                        valid_ids = {c["track_id"] for c in top_candidates}
                        if qwen_id in valid_ids:
                            chosen_id = qwen_id

                    force_reselect = current_focus_id is None or (frame_idx - current_focus_last_seen > focus_missing_grace_n)

                    if force_reselect:
                        current_focus_id = chosen_id
                        pending_switch_id = None
                        pending_switch_count = 0
                    else:
                        if chosen_id == current_focus_id:
                            pending_switch_id = None
                            pending_switch_count = 0
                        else:
                            if pending_switch_id == chosen_id:
                                pending_switch_count += 1
                            else:
                                pending_switch_id = chosen_id
                                pending_switch_count = 1
                            if pending_switch_count >= SWITCH_CONFIRM_EVENTS:
                                current_focus_id = chosen_id
                                pending_switch_id = None
                                pending_switch_count = 0

                    focus_events.append({
                        "frame_idx": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "profile": profile_name,
                        "chosen_focus_id": current_focus_id,
                        "candidate_scores": [
                            {
                                "track_id": c["track_id"],
                                "score": c["score"],
                                "presence": c["presence"],
                                "area_mean": c["area_mean"],
                                "motion_mean": c["motion_mean"],
                                "center_dist": c["center_dist"],
                            }
                            for c in top_candidates
                        ],
                        "qwen_board": str(qwen_board_path) if qwen_board_path else None,
                        "qwen_raw": qwen_text,
                    })

            focus_box = None
            if current_focus_id is not None and current_focus_id in all_tracks:
                tr = all_tracks[current_focus_id]
                if frame_idx - tr.last_seen <= focus_missing_grace_n:
                    focus_box = tr.current_box()
                    current_focus_last_seen = tr.last_seen

            if focus_box is not None:
                focus_box = clamp_box(focus_box, frame_w, frame_h)
                if focus_box_ema is None:
                    focus_box_ema = np.array(focus_box, dtype=np.float32)
                else:
                    focus_box_ema = FOCUS_BOX_EMA_ALPHA * np.array(focus_box, dtype=np.float32) + (1.0 - FOCUS_BOX_EMA_ALPHA) * focus_box_ema
                focus_box = clamp_box([float(x) for x in focus_box_ema.tolist()], frame_w, frame_h)

            landmarks_out = []
            if focus_box is not None:
                roi = expand_box(focus_box, frame_w, frame_h, frac=BOX_EXPAND)
                roi = clamp_box(roi, frame_w, frame_h)
                x1, y1, x2, y2 = [int(round(v)) for v in roi]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    if result.pose_landmarks:
                        crop_h, crop_w = crop.shape[:2]
                        for i, lm in enumerate(result.pose_landmarks[0]):
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
            overlay = draw_overlay(overlay, focus_box, landmarks_out, current_focus_id)
            writer.write(overlay)

            box_record = box_norm(focus_box, frame_w, frame_h) if focus_box is not None else None

            frame_record = {
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "focus_subject_id": None if current_focus_id is None else f"track_{current_focus_id}",
                "focus_method": "pyautoflip_mpdet_lock_hysteresis_qwen30b_v1",
                "profile": profile_name,
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
                    "source_media": str(src_path),
                    "frame_info": {
                        "frame_idx": frame_idx,
                        "box": box_record,
                    },
                    "additional_info": {
                        "focus_subject_id": None if current_focus_id is None else f"track_{current_focus_id}",
                        "focus_method": "pyautoflip_mpdet_lock_hysteresis_qwen30b_v1",
                        "profile": profile_name,
                        "landmarks": landmarks_out,
                    },
                },
            })

            frame_idx += 1

    jsonl_records.append({
        "type": "progress",
        "data": {
            "source_media": str(src_path),
        },
    })

    with open(debug_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_video": str(src_path),
            "working_video": str(video_path),
            "focus_subject_id": None if current_focus_id is None else f"track_{current_focus_id}",
            "profile": profile_name,
            "frame_count": len(frames_out),
            "frames": frames_out,
        }, f, indent=2)

    with open(tags_jsonl_path, "w", encoding="utf-8") as f:
        for row in jsonl_records:
            f.write(json.dumps(row) + "\n")

    with open(focus_debug_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_video": str(src_path),
            "working_video": str(video_path),
            "profile": profile_name,
            "focus_events": focus_events,
        }, f, indent=2)

    cap.release()
    writer.release()

    frames_with_pose = sum(1 for x in frames_out if x["landmarks"])
    print(f"[done] {src_path.name}")
    print(f"  profile: {profile_name}")
    print(f"  overlay: {overlay_path}")
    print(f"  debug_json: {debug_json_path}")
    print(f"  tags_jsonl: {tags_jsonl_path}")
    print(f"  focus_debug: {focus_debug_path}")
    print(f"  frames: {len(frames_out)}")
    print(f"  frames_with_pose: {frames_with_pose}")


def main():
    videos = sorted(INPUT_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no mp4 files found in {INPUT_DIR}")
    for v in videos:
        process_video(v)


if __name__ == "__main__":
    main()
