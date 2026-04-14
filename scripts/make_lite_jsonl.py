import json
import math
from pathlib import Path

ROOT = Path("/home/elv-ryan/projects/pose_focus_tagger/data/out/final_batch_v2")

def round_sig(x, sig=3):
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x == 0.0:
            return 0.0
        return float(f"{x:.{sig}g}")
    return x

def transform_numbers(obj):
    if isinstance(obj, dict):
        return {k: transform_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [transform_numbers(v) for v in obj]
    return round_sig(obj, 3)

def has_null(obj):
    if obj is None:
        return True
    if isinstance(obj, dict):
        return any(has_null(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_null(v) for v in obj)
    return False

def strip_fields(tag_obj):
    data = tag_obj.get("data", {})

    data.pop("track", None)
    data.pop("tag", None)

    addl = data.get("additional_info", {})
    other = addl.get("other_info", {})

    other.pop("focus_subject_id", None)
    other.pop("focus_method", None)
    other.pop("profile", None)

    if not other:
        addl.pop("other_info", None)
    else:
        addl["other_info"] = other

    if addl:
        data["additional_info"] = addl
    else:
        data.pop("additional_info", None)

    tag_obj["data"] = data
    return tag_obj

def convert_file(src: Path):
    dst = src.with_name(src.name.replace("_pose_final.jsonl", "_pose_lite.jsonl"))

    kept = 0
    dropped = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            row_type = obj.get("type")

            if row_type == "tag":
                obj = strip_fields(obj)
                if has_null(obj):
                    dropped += 1
                    continue

            obj = transform_numbers(obj)
            fout.write(json.dumps(obj, separators=(",", ":")) + "\n")
            kept += 1

    print(f"wrote: {dst}")
    print(f"  kept: {kept}")
    print(f"  dropped: {dropped}")

def main():
    files = sorted(ROOT.glob("*/*_pose_final.jsonl"))
    if not files:
        raise SystemExit(f"no *_pose_final.jsonl files found under {ROOT}")

    for src in files:
        convert_file(src)

if __name__ == "__main__":
    main()
