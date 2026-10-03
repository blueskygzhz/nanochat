#!/bin/bash
# The full nanochat pipeline, from scratch, on one CPU core.
#
#   bash runs/speedrun.sh
#
# This is the from-scratch counterpart of upstream's runs/speedrun.sh. Same stages,
# vastly smaller scale: upstream trains a GPT-2 capability model on 8xH100 in ~2 hours,
# this trains a 229K-parameter model on a CPU in ~2 minutes. The point is that every
# stage is the real algorithm, with no framework underneath.
#
# Stages:
#   1. pretrain a base model on next-token prediction, reporting bits-per-byte
#   2. evaluate it: bpb against the known entropy floor, CORE-style multiple choice
#   3. finetune it on conversations, training only on assistant tokens
#   4. talk to it

set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer the project venv over whatever `python` happens to be on PATH (which may be
# an old system interpreter without numpy).
if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi
if ! "$PY" -c 'import numpy' 2>/dev/null; then
    echo "error: $PY cannot import numpy. Run 'uv sync --group dev' first." >&2
    exit 1
fi
echo "interpreter: $PY ($("$PY" -V 2>&1))"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"

DEPTH="${DEPTH:-4}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-400}"
SFT_STEPS="${SFT_STEPS:-600}"

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

banner "1/4  Pretrain (depth=$DEPTH, $PRETRAIN_STEPS steps)"
"$PY" -m scripts.base_train \
    --depth "$DEPTH" \
    --num-iterations "$PRETRAIN_STEPS" \
    --eval-every 100 \
    --run base

banner "2/4  Evaluate the base model"
"$PY" -m scripts.base_eval --run base

banner "3/4  Finetune on conversations ($SFT_STEPS steps)"
"$PY" -m scripts.chat_sft \
    --source base \
    --num-iterations "$SFT_STEPS" \
    --eval-every 200 \
    --run sft

banner "4/4  Talk to it"
for q in "2+3" "7+8" "9+9"; do
    printf 'You: %-5s Bot: ' "$q"
    "$PY" -m scripts.chat_cli --run sft -p "$q"
done

banner "Done"
echo "Interactive chat:  $PY -m scripts.chat_cli --run sft"
echo "Checkpoints:       $NANOCHAT_BASE_DIR/checkpoints/{base,sft}"
