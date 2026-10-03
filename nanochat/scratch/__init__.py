"""
`nanochat.scratch` -- nanochat's full training and evaluation stack, from scratch.

    from nanochat.scratch import GPT, GPTConfig, setup_optimizer, Engine, evaluate_bpb

The package imports nothing but numpy. numpy supplies an n-dimensional array and a
BLAS matmul; the autograd engine, the module system, the layers, the optimizers, the
model, the inference engine, the metrics and the training loop are all written here:

    tensor.py      reverse-mode autograd      (replaces torch.Tensor + torch.autograd)
    nn.py          module system and layers   (replaces torch.nn)
    model.py       the GPT architecture       (replaces nanochat/gpt.py)
    optim.py       AdamW and Muon             (replaces torch.optim + nanochat/optim.py)
    engine.py      KV-cache inference         (replaces nanochat/engine.py)
    eval.py        bits-per-byte and CORE     (replaces nanochat/{loss,core}_eval.py)
    checkpoint.py  save/load/resume           (replaces nanochat/checkpoint_manager.py)
    data.py        tokenizer and batching     (replaces nanochat/dataloader.py)

The pipeline, end to end:

    python -m scripts.base_train     # pretrain, with checkpoints and val bpb
    python -m scripts.base_eval      # bits-per-byte + CORE-style task accuracy
    python -m scripts.chat_sft       # finetune on conversations
    python -m scripts.chat_cli       # talk to it

This is a learning artefact, not a replacement for a GPU stack. It is single-threaded
float32 on CPU with naive O(T^2) attention, so it is many orders of magnitude slower,
and it has no FlashAttention, FP8, bf16, kernel fusion or distributed training.
"""

from nanochat.scratch.checkpoint import (
    build_model, find_last_step, list_steps, load_checkpoint, load_model, save_checkpoint,
)
from nanochat.scratch.data import (
    ByteTokenizer, Dataset, addition_entropy_floor, addition_pairs, build_corpus,
    corpus_spec, make_addition_corpus,
)
from nanochat.scratch.engine import Engine, KVCache, sample_next_token
from nanochat.scratch.eval import evaluate_bpb, evaluate_task, token_bytes_table
from nanochat.scratch.model import GPT, GPTConfig, MoE, MoEGate
from nanochat.scratch.optim import AdamW, Muon, MuonAdamW, polar_express, setup_optimizer
from nanochat.scratch.tensor import Tensor, cross_entropy, no_grad, rms_norm, softmax

__all__ = [
    # engine / autograd
    "Tensor", "no_grad", "softmax", "rms_norm", "cross_entropy",
    # model
    "GPT", "GPTConfig", "MoE", "MoEGate",
    # optimizers
    "AdamW", "Muon", "MuonAdamW", "setup_optimizer", "polar_express",
    # data
    "ByteTokenizer", "Dataset", "make_addition_corpus", "addition_entropy_floor",
    "addition_pairs", "corpus_spec", "build_corpus",
    # inference
    "Engine", "KVCache", "sample_next_token",
    # evaluation
    "evaluate_bpb", "evaluate_task", "token_bytes_table",
    # checkpoints
    "save_checkpoint", "load_checkpoint", "build_model", "load_model",
    "find_last_step", "list_steps",
]
