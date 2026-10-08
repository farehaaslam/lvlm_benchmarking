#!/usr/bin/env bash
# One-shot setup for video_mmmu_gemma/vllm_eval.py (Gemma 4 12B BF16 on Video-MMMU) on a fresh rented GPU box:
#   HF_TOKEN=hf_xxx bash setup_gemma_vast.sh
# Needs an HF token whose account has accepted the Gemma license on huggingface.co/google/gemma-4-12B-it.
# then (inside tmux so an SSH drop does not kill the run):
#   tmux new -s gemma
#   cd video_mmmu_gemma
#   uv run --no-sync python vllm_eval.py --limit_videos 3 2>&1 | tee smoke.log   # quick check
#   uv run --no-sync python vllm_eval.py 2>&1 | tee run.log                      # full run (resumes)
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-google/gemma-4-12B-it}"
DATA_DIR=dataset/videommmu

echo "==> GPUs"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRIVER_MAJOR" -lt 575 ]; then
    echo "[WARNING] Driver $DRIVER_MAJOR < 575: the CUDA 12.9 torch/vLLM wheels will not run. Pick another instance."
fi

echo "==> Disk (model ~25GB + videos ~15GB zipped, ~30GB unzipped)"
df -h .

echo "==> System packages"
command -v unzip >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq unzip tmux)

echo "==> Installing uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Installing Python deps"
uv sync

if [ -n "${HF_TOKEN:-}" ]; then
    uv run hf auth login --token "$HF_TOKEN"
fi

echo "==> Downloading $MODEL in the background"
uv run hf download "$MODEL" > model_download.log 2>&1 &
MODEL_PID=$!

echo "==> Downloading Video-MMMU (parquet questions + video zips)"
uv run hf download lmms-lab/VideoMMMU --repo-type dataset --local-dir "$DATA_DIR"

echo "==> Unzipping videos"
for z in "$DATA_DIR"/*.zip; do
    marker="$z.unzipped"
    [ -f "$marker" ] && continue
    unzip -q -o "$z" -d "$DATA_DIR" && touch "$marker"
done

echo "==> Waiting for model download"
if ! wait "$MODEL_PID"; then
    echo "[ERROR] Model download failed (license not accepted / no HF_TOKEN?), see model_download.log"
    exit 1
fi

echo "==> Checking questions <-> videos"
cd video_mmmu_gemma
uv run --no-sync python vllm_eval.py --summary_only | head -3

echo "==> Setup done. Next:  tmux new -s gemma  then  cd video_mmmu_gemma && uv run --no-sync python vllm_eval.py --limit_videos 3"
