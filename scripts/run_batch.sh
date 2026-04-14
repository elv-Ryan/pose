#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CUDA_VISIBLE_DEVICES:=1}"

python scripts/pose_batch_final_v2.py
