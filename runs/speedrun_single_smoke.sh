#!/bin/bash
# Smoke test for the single-GPU pipeline (mirrors runs/speedrun_single.sh).
# Goal: exercise every stage (tokenizer -> pretrain -> eval -> SFT -> chat eval -> report)
# at the smallest possible scale to verify scripts work end-to-end on 1x GPU.
# Expected runtime: ~10-20 minutes (most time is dataset/HF downloads on first run).
#
# Run from repo root:
#   bash runs/speedrun_single_smoke.sh
#
# Outputs go to ~/.cache/nanochat just like the real run. After verifying success,
# you can wipe ~/.cache/nanochat/base_checkpoints/d4_smoke etc. and run the real script.

set -e  # fail fast on any error

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

# Route HuggingFace traffic through a mirror (huggingface.co is unreachable here).
# Affects: nanochat.dataset (patched to read HF_ENDPOINT) AND huggingface_hub/datasets
# libraries used by SFT (SmolTalk/MMLU/GSM8K) and any HF model loads.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Activate the existing venv (assume already created via `uv sync --extra gpu`).
source .venv/bin/activate

WANDB_RUN=dummy

# -----------------------------------------------------------------------------
echo "[smoke] === reset report ==="
python -m nanochat.report reset

# -----------------------------------------------------------------------------
echo "[smoke] === download 2 data shards ==="
python -m nanochat.dataset -n 2

# -----------------------------------------------------------------------------
echo "[smoke] === train tokenizer (tiny: 50M chars, vocab=8192) ==="
python -m scripts.tok_train --max-chars=50000000 --vocab-size=8192
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
echo "[smoke] === pretrain tiny base model (depth=4, 20 iters) ==="
# Disable eval/sample/core-metric during training; we hit them as separate steps below.
python -m scripts.base_train \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=2 \
    --total-batch-size=1024 \
    --num-iterations=20 \
    --target-param-data-ratio=-1 \
    --warmup-steps=2 \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --eval-tokens=1024 \
    --model-tag=d4_smoke \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
echo "[smoke] === base_eval (bpb only, tiny budget) ==="
python -m scripts.base_eval \
    --model-tag=d4_smoke \
    --eval=bpb,sample \
    --device-batch-size=2 \
    --split-tokens=2048 \
    --max-per-task=4

# -----------------------------------------------------------------------------
echo "[smoke] === download identity conversations ==="
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    # S3 from this network is slow (~35 KB/s); give plenty of room and retry.
    curl -L --connect-timeout 30 --max-time 600 --retry 3 --retry-delay 5 \
        -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# -----------------------------------------------------------------------------
echo "[smoke] === SFT (10 iters, no MMLU/GSM8K epochs) ==="
python -m scripts.chat_sft \
    --model-tag=d4_smoke \
    --device-batch-size=2 \
    --total-batch-size=1024 \
    --max-seq-len=512 \
    --num-iterations=10 \
    --eval-every=-1 \
    --chatcore-every=-1 \
    --mmlu-epochs=0 \
    --gsm8k-epochs=0 \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
echo "[smoke] === chat_eval (1 task, 2 problems) ==="
python -m scripts.chat_eval -i sft -a SpellingBee -x 2 -b 2 -g d4_smoke

# -----------------------------------------------------------------------------
echo "[smoke] === generate report ==="
python -m nanochat.report generate

echo "[smoke] === DONE ==="