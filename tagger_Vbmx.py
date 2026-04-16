import json
import subprocess
import sys
import os

from common_ml.tagging.messages import Tag
from common_ml.tagging.run_helpers import run_default, catch_errors
from common_ml.tagging.producer import AVModel
from dacite import from_dict

def run_script(video_path: str, output_dir: str) -> None:

    # use sys.executable to make sure subprocesses use the same python executable
    scripts = [
        f"{sys.executable} scripts/pose_batch_final_Vbmx.py --video {video_path} --output-path {output_dir}",
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
            tags.append(
                from_dict(
                    data_class=Tag,
                    data=data
                )      
            )

    return tags

class PoseModelVbmx(AVModel):
    def tag(self, fpath: str) -> list[Tag]:
        run_script(fpath, "out")
        # assume tags will go in tags.jsonl
        tags = load_tags(fpath, os.path.join("out", "tags.jsonl"))
        return tags
        
def main():
    catch_errors()

    producer = PoseModelVbmx()
    run_default(producer)

if __name__ == "__main__":
    main()