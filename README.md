# yt-mp4-toolkit

Tiny Python helpers for making a clean MP4 dataset quickly:

- Download video-only MP4s capped at 720p via `yt-dlp` (no audio track)
- Cut random silent clips from those MP4s via `ffmpeg`
- Optionally transcode clips to a fixed size/FPS (useful for ML baselines)

The repo includes a few example files under `downloads/`, `clips/`, and `clips_resized/` so you can sanity-check the pipeline end-to-end.

## Repo layout

```
.
├── 720p_noaudio_YT_downloader.py   # fetch 720p (or best <=720p) video-only MP4
├── mp4_clipper.py                 # make random silent clips from MP4s
├── clips_resizer.py               # optional: transcode clips to fixed size/FPS
├── downloads/                     # example full MP4s
├── clips/                         # example clips
└── clips_resized/                 # example resized clips
```

## Requirements

- Python 3
- `yt-dlp`
- `ffmpeg` (and `ffprobe`)

On macOS with Homebrew:

```bash
brew install yt-dlp ffmpeg
```

Quick check:

```bash
yt-dlp --version
ffmpeg -version
ffprobe -version
```

## Quickstart

1) Download a couple of MP4s (video-only, no audio) into `downloads/`:

```bash
python3 720p_noaudio_YT_downloader.py "<url1>" "<url2>"
```

2) Make random 20-second clips from the first 60 seconds of each MP4:

```bash
python3 mp4_clipper.py --in-dir downloads --out-dir clips --clip-len 20 --first-seconds 60 --reencode
```

3) Optional: standardize clips to a fixed resolution and FPS:

```bash
python3 clips_resizer.py --in-dir clips --out-dir clips_resized --width 832 --height 480 --fps 16
```

## Scripts

### 1) `720p_noaudio_YT_downloader.py`

Downloads the best video-only stream at 720p when available (otherwise best <=720p) and outputs MP4s under `downloads/`.

Output naming format:

- `downloads/<title> [<id>].mp4`

If you need to inspect available formats for a URL:

```bash
yt-dlp -F "<url>"
```

### 2) `mp4_clipper.py`

Creates a random silent clip from each `.mp4` in the input directory.

Key flags:

- `--clip-len` (default: 20 seconds)
- `--first-seconds` (default: 30 seconds). Start time is sampled from `[0, first-seconds]`, but clamped so the clip fits.
- `--reencode` for accurate cuts (slower). Without it, it tries stream copy and cuts can land on keyframes.

Example:

```bash
python3 mp4_clipper.py --in-dir downloads --out-dir clips --clip-len 20 --first-seconds 60 --reencode
```

### 3) `clips_resizer.py` (optional)

Transcodes every clip in a directory to a consistent format.

Defaults:

- 832x480
- 16 fps
- CRF 23, preset `veryfast`

Example:

```bash
python3 clips_resizer.py --in-dir clips --out-dir clips_resized
```

## Notes

- The scripts currently assume Homebrew installs at `/opt/homebrew/bin` (Apple Silicon macOS). If your binaries live elsewhere, edit the `YT_DLP`, `FFMPEG`, or `FFPROBE` constants near the top of each script.
- If you plan to commit MP4s, Git LFS is worth considering due to GitHub file size limits.
