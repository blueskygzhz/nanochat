# nanochat (from scratch, no framework)

This is a fork of [nanochat](https://github.com/karpathy/nanochat) with **PyTorch removed entirely**. Nothing here imports a deep learning framework. The autograd engine, the module system, the layers, the optimizers, the tokenizer and the model are all written out by hand.

numpy is the only runtime dependency that does any math, and it supplies exactly two things: an n-dimensional array and a BLAS matrix multiply. It provides no automatic differentiation, no layers, no optimizers and no model — that is all code in this repo.

```bash
uv sync --group dev
source .venv/bin/activate
python -m scripts.scratch_train
```

```
from-scratch nanochat | dense | 229,458 params | 126,000 train tokens
task: 2-digit addition | entropy floor: 0.6579 nats/token | uniform-byte baseline: 5.5452
--------------------------------------------------------------------
step    0 | train 5.5446 | val 5.5042 |    0.4s | gap to floor +4.8464
step  100 | train 0.7795 | val 0.7681 |   11.3s | gap to floor +0.1103
step  199 | train 0.6908 | val 0.6953 |   22.2s | gap to floor +0.0374
--------------------------------------------------------------------
greedy addition accuracy: 17/20
```

The loss starts at 5.5446, which is `ln(256)` — a model that knows nothing about bytes. It ends 0.037 nats above the information-theoretic floor of the task, in 22 seconds on one CPU core.

## Read this first: what this fork is and is not

The upstream project trains a GPT-2 capability model on an 8×H100 node in under two hours. **This fork cannot do that, and never will.** Removing PyTorch deleted 9,118 lines across 27 files, and with them:

- GPU training of any kind (no CUDA kernels)
- distributed training (NCCL cannot be implemented in Python)
- FlashAttention, FP8, bf16, `torch.compile`
- the inference engine with KV caching, SFT, RL
- MLA, MTP, and the 7.26B-total MoE run
- GPT-2 / GPT-3 parity, the CORE evaluation, and the speedrun leaderboard

What is left runs single-threaded float32 on CPU with naive O(T²) attention. It is many orders of magnitude slower than the real thing. Its demonstrated capability is a 229K-parameter model that learns one-digit addition in 22 seconds.

**If you want to train a usable language model, use [upstream nanochat](https://github.com/karpathy/nanochat).** This fork exists to make every step of training readable — there is no layer you cannot step into with a debugger.

## What's in it

| File | Replaces | Lines | What's in it |
|------|----------|-------|--------------|
| `nanochat/scratch/tensor.py` | `torch.Tensor`, `torch.autograd` | 660 | Reverse-mode AD: per-op backward closures, reverse-topological backward pass, broadcasting adjoints, fused `softmax`/`cross_entropy`/`rms_norm`, and the `topk`/`index_add` scatter-gather pair the MoE router needs |
| `nanochat/scratch/nn.py` | `torch.nn` | 257 | `Module.__setattr__` parameter/submodule registration, `named_parameters`, `state_dict`, `Linear`, `Embedding`, naive attention with GQA and sliding-window masks |
| `nanochat/scratch/model.py` | `nanochat/gpt.py` | 425 | RMSNorm, RoPE + QK-norm, ResFormer value embeddings, embedding smear, per-layer resid/x0 scalars, mid-layer backout, ReLU² FFNs, DeepSeek-V2 MoE with load-balancing aux loss, tanh-softcapped logits |
| `nanochat/scratch/optim.py` | `torch.optim`, `nanochat/optim.py` | 226 | AdamW with decoupled decay, and Muon (Polar Express orthogonalization, no SVD) |
| `nanochat/scratch/data.py` | `nanochat/dataloader.py` | 77 | Byte tokenizer and batch sampler |
| `nanochat/bpe.py` | `rustbpe`, `tiktoken` | 402 | Byte-level BPE: training (merge counting with incremental updates) and inference, standard library only |

## Usage

```bash
python -m scripts.scratch_train                        # dense
python -m scripts.scratch_train --n-routed-experts 4   # MoE
python -m scripts.scratch_train --text-file book.txt   # your own text
python -m scripts.scratch_train --help                 # all knobs
```

The default task is two-digit addition (`"7+5=12;"`). That choice is deliberate: its entropy is known exactly, so there is a real target to hit rather than just a loss curve that goes down. Every character of a line is determined by the two operands, so the only information in the stream is those operands:

\[
H = \frac{2\ln 10}{7} = 0.6579 \text{ nats/token}
\]

A model at ~0.66 has learned to carry. A model at ~2.3 has only learned character frequencies.

## How the gradients are verified

There is no framework to compare against, so correctness is established from first principles. For every operation, the analytic backward is checked against a central finite difference of the forward:

\[
\frac{df}{dx} \approx \frac{f(x+h) - f(x-h)}{2h} \qquad \text{error } O(h^2)
\]

Finite differences are meaningless in float32 (roundoff swamps the signal), so `tests/test_scratch.py` flips the engine to float64 for those checks via `nanochat.scratch.tensor.set_dtype`.

Verification layers, weakest to strongest:

1. **op forward values** against closed-form numpy expressions (49 ops)
2. **op gradients** against central finite differences
3. **whole-model gradients** against finite differences — dense, MoE, and MoE with `norm_topk_prob`, covering the top-k gather, the expert-order sort and the scatter-add combine
4. **optimizer steps** against hand-written reference updates (AdamW matches the textbook update to `rtol=1e-9` over 5 steps)
5. **it learns**: the loss reaches the known entropy floor and the model does the arithmetic correctly

Plus two guards that keep the build honest: no source file may contain `import torch`, and `nanochat/scratch/` may import nothing but numpy and the standard library.

```bash
pytest -m "not slow"   # 229 tests, ~5s
pytest -m slow         # 2 real training runs, ~85s
```

## Tokenizer

`nanochat/bpe.py` is a hand-written byte-level BPE that handles both training and inference — no `rustbpe`, no `tiktoken`. It is the default and needs no extra dependencies.

The Rust stack is still supported as an optional backend, because those two packages are the only way to *verify* the hand-written one:

```bash
uv sync --extra fast-tokenizer     # installs rustbpe + tiktoken
NANOCHAT_BPE_BACKEND=rust python -m scripts.tok_train
```

`tests/test_bpe.py` asserts that our training produces the **same merge order** as rustbpe and that our encoder produces the **same token ids** as tiktoken on identical ranks. Without the extra, those tests skip and everything else works unchanged. Pure Python is roughly 100x slower to train and 6x slower to encode, which is why the fast backend remains available.

## File structure

```
.
├── nanochat
│   ├── bpe.py                      # Hand-written byte-level BPE (training + inference)
│   ├── common.py                   # Logging, base dir, locked download
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── execution.py                # Sandboxed Python execution (for the humaneval task)
│   ├── scratch                     # The training stack, from scratch
│   │   ├── data.py                 # Byte tokenizer + batch sampler
│   │   ├── model.py                # The GPT architecture
│   │   ├── nn.py                   # Module/Parameter system and layers
│   │   ├── optim.py                # AdamW + Muon
│   │   └── tensor.py               # Reverse-mode autograd engine
│   └── tokenizer.py                # Tokenizer wrapper, GPT-4 style special tokens
├── scripts
│   ├── scratch_train.py            # Train a model end to end
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks                           # Eval task data loaders (arc, gsm8k, mmlu, ...)
└── tests
    ├── test_bpe.py                 # BPE training/encoding, rustbpe+tiktoken parity
    ├── test_execution.py           # Sandboxed code execution
    ├── test_scratch.py             # Autograd, layers, model, optimizers, convergence
    ├── test_tasks.py               # Task slicing, mixtures, HubDataset
    └── test_tokenizer.py           # BPE round-trips, chat rendering
```

`tasks/` and `nanochat/dataset.py` survive untouched — they are data plumbing (pyarrow, HTTP, sharding) and never depended on a framework. They currently have no consumer, since evaluating a model needs an inference engine that was part of the deleted stack.

## Acknowledgements

This is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat). All architecture ideas, the training recipe, and the tokenizer design are from upstream; this fork only reimplements them without a framework. The MoE layer follows DeepSeek-V2, Muon is from [Keller Jordan](https://kellerjordan.github.io/posts/muon/) with [Polar Express](https://arxiv.org/pdf/2505.16932) coefficients, and the speculative-decoding math is [Leviathan et al.](https://arxiv.org/abs/2211.17192).

## License

MIT
