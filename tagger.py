import json
import subprocess
import sys
import os
from dataclasses import dataclass

from common_ml.tagging.messages import Tag
from common_ml.tagging.run_helpers import run_default, catch_errors, get_params
from common_ml.tagging.producer import AVModel
from dacite import from_dict

def run_script(video_path: str, output_dir: str) -> None:

    # use sys.executable to make sure subprocesses use the same python executable
    scripts = [
        f"{sys.executable} scripts/pose_batch_final_v2.py --video {video_path} --output-path {output_dir}",
    ]
    for script in scripts:
        try:
            print(f"running {script}")
            subprocess.run(script.split(), check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Script failed (exit code {e.returncode}): {script}") from e

def load_tags(source_media: str, path: str) -> list[Tag]:
    """loads tags from json """
    tags = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if data["type"] != "tag":
                continue
            data = data["data"]
            data["source_media"] = source_media
            data["tag"] = ""
            if "other_info" in data["additional_info"]:
                del data["additional_info"]["other_info"]
            tags.append(
                from_dict(
                    data_class=Tag,
                    data=data
                )      
            )

    return tags

def downsample(tags: list[Tag], fps: int) -> list[Tag]:
    if len(tags) < 2:
        return tags
    
    tags = sorted(tags, key=lambda t: t.start_time)

    assert tags[1].start_time > 0
    assert tags[1].frame_info
    approximate_fps = tags[1].frame_info.frame_idx / (tags[1].start_time / 1000)
    
    downsample_ratio = int(approximate_fps / fps)
    assert downsample_ratio > 0

    out = []
    for i, t in enumerate(tags):
        if i % downsample_ratio == 0:
            out.append(t)

    return out


class PoseModel(AVModel):
    def __init__(self, fps: int):
        self.fps = fps

    def tag(self, fpath: str) -> list[Tag]:
        run_script(fpath, "out")
        # assume tags will go in tags.jsonl
        tags = load_tags(fpath, os.path.join("out", "tags.jsonl"))
        tags = downsample(tags, fps=self.fps)
        return tags
    
@dataclass
class RuntimeArgs:
    fps: int = 5

def main():
    catch_errors()

    params = from_dict(data_class=RuntimeArgs, data=get_params())

    producer = PoseModel(params.fps)
    run_default(producer)

if __name__ == "__main__":
    main()