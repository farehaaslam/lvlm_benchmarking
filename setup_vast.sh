#!/usr/bin/env bash
# One-shot setup on a fresh vast.ai instance:
#   bash setup_vast.sh
# then (inside tmux so an SSH drop does not kill the run):
#   tmux new -s qwen
#   uv run qwen32B_vllm.py --limit 1 2>&1 | tee smoke.log   # quick check
#   uv run qwen32B_vllm.py 2>&1 | tee run.log               # full run (resumes)
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"

echo "==> GPUs"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRIVER_MAJOR" -lt 575 ]; then
    echo "[WARNING] Driver $DRIVER_MAJOR < 575: the CUDA 12.9 torch/vLLM wheels will not run. Pick another instance."
fi

echo "==> Disk (model needs ~70GB, plus videos)"
df -h .

echo "==> Installing uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Installing Python deps"
uv sync

echo "==> Downloading $MODEL in the background"
uv run hf download "$MODEL" > model_download.log 2>&1 &
MODEL_PID=$!

echo "==> Downloading questions"
uv run download_dataset.py

if [ -z "$(ls dataset/videos 2>/dev/null)" ]; then
    echo "==> Downloading videos"
    # or: rclone copy gdrive:gurrt/gurrt_dataset/videos dataset/videos
    uv run download_video.py
fi

echo "==> Waiting for model download"
if ! wait "$MODEL_PID"; then
    echo "[ERROR] Model download failed, see model_download.log"
    exit 1
fi

echo "==> Checking video <-> question pairs"
for v in dataset/videos/*; do
    q="dataset/questions/$(basename "${v%.*}").csv"
    [ -f "$q" ] || echo "[WARNING] missing questions for $(basename "$v")"
done
echo "videos: $(ls dataset/videos | wc -l) | question files: $(ls dataset/questions/*.csv | wc -l)"

echo "==> Setup done. Next:  tmux new -s qwen  then  uv run qwen32B_vllm.py 2>&1 | tee run.log"
