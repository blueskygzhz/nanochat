#!/bin/bash

# Experimental 7.26B-total MoE, NOT a 7B-dense model.
# 24 layers, width 2048, 16 query / 4 KV heads, 48 routed (top-6) + 2 shared experts.
# Reference hardware: 8 x 80GB CUDA GPUs. Memory fit and throughput require a smoke run.
# Weights/gradients remain replicated; optimizer state is sharded. No expert parallelism.
# The ~50B-token budget is an initial experiment, not a convergence guarantee.
# Modes: check (default, meta only), prepare (downloads data), smoke, train, eval, sft.
# Set NANOCHAT_BASE_DIR to a persistent disk with space for data and multiple ~60GB checkpoints.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
MODE=${1:-check}
if (( $# > 0 )); then shift; fi
NPROC=${NPROC:-8}
ATTENTION_TYPE=${ATTENTION_TYPE:-gqa}
DEFAULT_TAG=moe7b
if [[ "$ATTENTION_TYPE" == mla ]]; then DEFAULT_TAG=moe7b-mla; fi
MODEL_TAG=${MODEL_TAG:-$DEFAULT_TAG}
WANDB_RUN=${WANDB_RUN:-dummy}
DEVICE_BATCH=${DEVICE_BATCH:-1}
SEQ_LEN=${SEQ_LEN:-2048}
TOTAL_BATCH=${TOTAL_BATCH:-2097152}
MUON_BUCKET_MB=${MUON_BUCKET_MB:-128}
LOSS_CHUNK_SIZE=${LOSS_CHUNK_SIZE:-512}
FP8=${FP8:-0}
RESUME_STEP=${RESUME_STEP:--1}
if (( NPROC <= 0 || 48 % NPROC != 0 || DEVICE_BATCH <= 0 || SEQ_LEN <= 1 )); then
    echo "NPROC must divide 48; DEVICE_BATCH > 0 and SEQ_LEN > 1 are required" >&2
    exit 2
fi
if [[ "$MODE" == prepare ]]; then
    python -m nanochat.dataset -n 8
    if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" || ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]]; then
        python -m scripts.tok_train
    fi
    python -m nanochat.dataset -n "${SHARDS:-1400}" -w "${DOWNLOAD_WORKERS:-8}"
    exit 0
fi
common=(
    --depth=24 --aspect-ratio=85 --head-dim=128 --n-kv-head=4
    --attention-type="$ATTENTION_TYPE" --q-lora-rank="${Q_LORA_RANK:-0}" --kv-lora-rank="${KV_LORA_RANK:-512}"
    --qk-nope-head-dim="${QK_NOPE_HEAD_DIM:-128}" --qk-rope-head-dim="${QK_ROPE_HEAD_DIM:-64}" --v-head-dim="${V_HEAD_DIM:-128}"
    --max-seq-len="$SEQ_LEN" --window-pattern="${WINDOW_PATTERN:-SSSL}"
    --n-routed-experts=48 --n-shared-experts=2 --num-experts-per-tok=6
    --moe-intermediate-mult=0.6875 --first-k-dense-replace=1 --aux-loss-alpha=0.001
    --device-batch-size="$DEVICE_BATCH" --activation-checkpointing
    --loss-chunk-size="$LOSS_CHUNK_SIZE" --muon-bucket-mb="$MUON_BUCKET_MB"
    --target-param-data-ratio="${PARAM_DATA_RATIO:-35}" --warmup-steps=500
)
if [[ "$FP8" == 1 ]]; then common+=(--fp8); fi
if [[ "${NO_COMPILE:-0}" == 1 ]]; then common+=(--no-compile); fi
launch=(python -m torch.distributed.run --standalone --nproc_per_node="$NPROC")
case "$MODE" in
    check)
        python -m scripts.base_train "${common[@]}" --device-type=cpu --dry-run "$@"
        ;;
    smoke)
        "${launch[@]}" -m scripts.base_train "${common[@]}" --device-type=cuda \
            --model-tag="${MODEL_TAG}-smoke" --run=dummy --num-iterations="${SMOKE_STEPS:-5}" \
            --total-batch-size="$((DEVICE_BATCH * SEQ_LEN * NPROC))" --warmup-steps=2 \
            --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 "$@"
        ;;
    train)
        "${launch[@]}" -m scripts.base_train "${common[@]}" --device-type=cuda \
            --model-tag="$MODEL_TAG" --run="$WANDB_RUN" --total-batch-size="$TOTAL_BATCH" \
            --resume-from-step="$RESUME_STEP" --eval-every=1000 --eval-tokens=10485760 \
            --core-metric-every=5000 --sample-every=5000 --save-every=2000 "$@"
        ;;
    eval)
        "${launch[@]}" -m scripts.base_eval --model-tag="$MODEL_TAG" --device-batch-size="$DEVICE_BATCH" "$@"
        ;;
    sft)
        "${launch[@]}" -m scripts.chat_sft --model-tag="$MODEL_TAG" --run="$WANDB_RUN" "$@"
        ;;
    *) echo "Usage: bash runs/moe7b.sh {check|prepare|smoke|train|eval|sft} [extra arguments]" >&2; exit 2 ;;
esac
