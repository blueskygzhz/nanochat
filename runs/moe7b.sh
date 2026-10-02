#!/bin/bash

# Train a ~7B-total / ~1.35B-active DeepSeek-V2 style MoE with nanochat (pretrain + SFT).
# Designed for one 8XH100 80GB node. Budget: ~50B tokens, ~4.7e20 FLOPs.
# At ~25% MFU that is roughly 3 days of wall clock; check the first few hundred steps' tok/sec + ETA.
#
# Launch (in a screen session, this takes days):
#   WANDB_RUN=moe7b screen -L -Logfile runs/moe7b.log -S moe7b bash runs/moe7b.sh
#
# Model (depth 24, d_model 2048, 16 q-heads / 4 kv-heads, seq 2048):
#   layer 0: dense MLP; layers 1..23: MoE with 48 routed experts (top-6) + 2 shared, expert hidden 1408
#   total params 7.26B | active transformer matrices 1.35B | ~9.5e9 training FLOPs / token
#
# Memory per GPU (no FSDP, params/grads replicated, optimizer state sharded ZeRO-2 style):
#   fp32 params ~27GB + fp32 grads ~27GB + Muon/AdamW state ~4GB + activations (with checkpointing)
#   => fits 80GB with --device-batch-size=4. If you OOM, drop to 2; if you have headroom, try 8.

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
NPROC=${NPROC:-8}
MODEL_TAG=${MODEL_TAG:-moe7b}
WANDB_RUN=${WANDB_RUN:-dummy}

# -----------------------------------------------------------------------------
# Python venv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Data + tokenizer
# ~50B training tokens after BOS-bestfit cropping => ~1300 shards; download 1400 for margin (~140GB disk).
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 1400 -w 16 &
DATASET_DOWNLOAD_PID=$!
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# -----------------------------------------------------------------------------
# Pretraining
# --target-param-data-ratio is applied to *active* params (1.35B + lm_head) => 35 x 1.416B ~= 49.6B tokens
# --total-batch-size 2M tokens => ~23.6K steps; grad accumulation = 2M / (4 * 2048 * 8) = 32 micro-steps
# --fp8 only converts attention + shared experts + dense MLP (routed experts stay bf16, see base_train.py)
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --run=$WANDB_RUN \
    --model-tag=$MODEL_TAG \
    --depth=24 \
    --aspect-ratio=85 \
    --head-dim=128 \
    --n-kv-head=4 \
    --max-seq-len=2048 \
    --n-routed-experts=48 \
    --n-shared-experts=2 \
    --num-experts-per-tok=6 \
    --moe-intermediate-mult=0.6875 \
    --first-k-dense-replace=1 \
    --aux-loss-alpha=0.001 \
    --target-param-data-ratio=35 \
    --total-batch-size=2097152 \
    --device-batch-size=4 \
    --activation-checkpointing \
    --fp8 \
    --warmup-steps=500 \
    --eval-every=1000 \
    --eval-tokens=10485760 \
    --core-metric-every=5000 \
    --sample-every=5000 \
    --save-every=2000

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- --model-tag=$MODEL_TAG --device-batch-size=4

# -----------------------------------------------------------------------------
# SFT (inherits seq len / batch sizes / LRs / activation checkpointing from the pretrain checkpoint)
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- --run=$WANDB_RUN --model-tag=$MODEL_TAG
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i sft --model-tag=$MODEL_TAG

# Talk to it:
# python -m scripts.chat_cli --model-tag=$MODEL_TAG -p "Why is the sky blue?"
