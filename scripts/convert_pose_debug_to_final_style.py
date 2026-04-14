import json
import statistics
from pathlib import Path

ROOT = Path("/home/elv-ryan/projects/pose_focus_tagger/data/out/full_batch")

POSE_KEY_ORDER = [
    "Nose",
    "LEyeIn", "LEye", "LEyeOut",
    "REyeIn", "REye", "REyeOut",
    "LEar", "REar",
    "MouthL", "MouthR",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LWrist", "RWrist",
    "LPinky", "RPinky",
    "LIndex", "RIndex",
    "LThumb", "RThumb",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
    "LHeel", "RHeel",
    "LFootIdx", "RFootIdx",
]

LANDMARK_NAME_MAP = {
    "nose": "Nose",
    "left_eye_inner": "LEyeIn",
    "left_eye": "LEye",
    "left_eye_outer": "LEyeOut",
    "right_eye_inner": "REyeIn",
    "right_eye": "REye",
    "right_eye_outer": "REyeOut",
    "left_ear": "LEar",
    "right_ear": "REar",
    "mouth_left": "MouthL",
    "mouth_right": "MouthR",
    "left_shoulder": "LShoulder",
    "right_shoulder": "RShoulder",
    "left_elbow": "LElbow",
    "right_elbow": "RElbow",
    "left_wrist": "LWrist",
    "right_wrist": "RWrist",
    "left_pinky": "LPinky",
    "right_pinky": "RPinky",
    "left_index": "LIndex",
    "right_index": "RIndex",
    "left_thumb": "LThumb",
    "right_thumb": "RThumb",
    "left_hip": "LHip",
    "right_hip": "RHip",
    "left_knee": "LKnee",
    "right_knee": "RKnee",
    "left_ankle": "LAnkle",
    "right_ankle": "RAnkle",
    "left_heel": "LHeel",
    "right_heel": "RHeel",
    "left_foot_index": "LFootIdx",
    "right_foot_index": "RFootIdx",
}

def round_or_none(v):
    if v is None:
        return None
    return round(float(v), 6)

def infer_frame_delta_ms(frames):
    if len(frames) < 2:
        return 33
    ts = [int(f.get("timestamp_ms", 0)) for f in frames]
    diffs = [b - a for a, b in zip(ts[:-1], ts[1:]) if (b - a) > 0]
    if not diffs:
        return 33
    return int(statistics.median(diffs))

def make_pose_dict(landmarks):
    pose = {k: None for k in POSE_KEY_ORDER}
    for lm in landmarks or []:
        src_name = lm.get("name")
        dst_name = LANDMARK_NAME_MAP.get(src_name)
        if not dst_name:
            continue
        pose[dst_name] = [
            round_or_none(lm.get("x")),
            round_or_none(lm.get("y")),
        ]
    return pose

def make_box(box):
    if not box:
        return {"x1": None, "x2": None, "y1": None, "y2": None}
    return {
        "x1": round_or_none(box.get("x1")),
        "x2": round_or_none(box.get("x2")),
        "y1": round_or_none(box.get("y1")),
        "y2": round_or_none(box.get("y2")),
    }

def convert_one(debug_path: Path):
    with open(debug_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        print(f"skip empty: {debug_path}")
        return

    delta_ms = infer_frame_delta_ms(frames)
    source_media = data.get("source_video") or data.get("working_video") or debug_path.name
    focus_subject_id = data.get("focus_subject_id") or "focus_subject"
    profile = data.get("profile")
    out_path = debug_path.with_name(debug_path.name.replace("_pose_debug.json", "_pose_final_style.jsonl"))

    with open(out_path, "w", encoding="utf-8") as out:
        for i, fr in enumerate(frames):
            start_time = int(fr.get("timestamp_ms", 0))
            if i + 1 < len(frames):
                end_time = int(frames[i + 1].get("timestamp_ms", start_time + delta_ms))
            else:
                end_time = start_time + delta_ms

            message = {
                "type": "tag",
                "data": {
                    "tag": focus_subject_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "track": "pose_detection",
                    "frame_info": {
                        "frame_idx": int(fr.get("frame_idx", i)),
                        "box": make_box(fr.get("subject_box")),
                    },
                    "additional_info": {
                        "pose": make_pose_dict(fr.get("landmarks")),
                        "other_info": {
                            "focus_subject_id": fr.get("focus_subject_id", focus_subject_id),
                            "focus_method": fr.get("focus_method"),
                            "profile": fr.get("profile", profile),
                        },
                    },
                    "source_media": source_media,
                },
            }
            out.write(json.dumps(message) + "\n")

        progress = {
            "type": "progress",
            "data": {
                "source_media": source_media
            }
        }
        out.write(json.dumps(progress) + "\n")

    print(f"wrote: {out_path}")

def main():
    debug_files = sorted(ROOT.glob("*/*_pose_debug.json"))
    if not debug_files:
        raise SystemExit(f"no *_pose_debug.json files found under {ROOT}")
    for debug_path in debug_files:
        convert_one(debug_path)

if __name__ == "__main__":
    main()
