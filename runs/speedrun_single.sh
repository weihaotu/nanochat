#!/bin/bash

# Single-GPU speedrun for nanochat
# Tuned for: 1x NVIDIA A100-SXM4-80GB
#
# Notes vs runs/speedrun.sh (the original 8xH100 version):
#   1) `torchrun --standalone --nproc_per_node=8 -m xxx -- args` is replaced with
#      plain `python -m xxx args`. The training code automatically falls back to
#      gradient accumulation when world_size == 1, so results are ~identical,
#      just ~8x slower in wall-clock time.
#   2) `--device-batch-size` is kept at 16. A100-80GB has the same VRAM as H100,
#      so the per-device batch size from the original script still fits. If you
#      ever OOM, step down: 16 -> 8 -> 4 -> 2 -> 1.
#   3) `--fp8` is removed. FP8 tensor cores only exist on Hopper (H100/H200).
#      On A100 (Ampere) the code falls back to bf16 by default (see README's
#      "Precision / dtype" section), which is the right choice here.
#
# Expected runtime: roughly 24h+ end-to-end on a single A100-80GB (vs ~3h on 8xH100).

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Route HuggingFace traffic through a mirror when huggingface.co is unreachable.
# nanochat.dataset was patched to read HF_ENDPOINT, and huggingface_hub/datasets
# (used by SFT for SmolTalk/MMLU/GSM8K) honor it natively.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# -----------------------------------------------------------------------------
# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Reset the report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining) -- single GPU, no torchrun, no fp8
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

python -m scripts.base_train --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT
# S3 from this network is slow (~35 KB/s); give plenty of room and retry.
curl -L --connect-timeout 30 --max-time 600 --retry 3 --retry-delay 5 \
    -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft --device-batch-size=16 --run=$WANDB_RUN
python -m scripts.chat_eval -i sft

# Chat with the model:
#   python -m scripts.chat_cli -p "Why is the sky blue?"
#   python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate
