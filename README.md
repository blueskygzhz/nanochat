# nanochat (from scratch, no framework)

This is a fork of [nanochat](https://github.com/karpathy/nanochat) with **PyTorch removed entirely**. Nothing here imports a deep learning framework. The autograd engine, the module system, the layers, the optimizers, the tokenizer, the inference engine with KV caching, the evaluation metrics, and the model are all written out by hand.

numpy is the only runtime dependency that does any math, and it supplies exactly two things: an n-dimensional array and a BLAS matrix multiply. It provides no automatic differentiation, no layers, no optimizers and no model — that is all code in this repo.

```bash
uv sync --group dev
source .venv/bin/activate
bash runs/speedrun.sh   # the full pipeline: pretrain → eval → SFT → chat
```

```
=== 1/4  Pretrain ===
base_train | dense d4 w64 | 229,458 params | vocab 256
task: addition | 80 train pairs, 20 held out | bpb floor 0.9031
done in 45.8s | final val bpb 0.9557

=== 2/4  Evaluate the base model ===
val bits per byte : 0.9558   entropy floor 0.9031, gap +0.0526
                      MC acc      greedy
seen pairs             1.000       80/80
held-out pairs         0.700       13/20

=== 3/4  Finetune on conversations ===
seen pairs      : exact match 80/80
held-out pairs  : exact match 15/20

=== 4/4  Talk to it ===
You: 2+3   Bot: 5
You: 7+8   Bot: 15
You: 9+9   Bot: 18
```

Total wall time: ~2 minutes on one CPU core.

**Read the held-out row, not the seen row.** There are only 100 distinct problems, so
perfect seen-pair accuracy is reachable by memorisation alone. 20% of the operand pairs
are therefore kept out of pretraining *and* SFT (`--holdout-frac`), and only accuracy on
those says whether the model learned to add. At this scale it partially does: across
seeds, held-out greedy accuracy ranges from roughly 35% to 75%.

## Read this first: what this fork is and is not

The upstream project trains a GPT-2 capability model on an 8×H100 node in under two hours. **This fork cannot do that, and never will.** Removing PyTorch deleted 9,118 lines across 27 files, and with them:

- GPU training of any kind (no CUDA kernels)
- distributed training (NCCL cannot be implemented in Python)
- FlashAttention, FP8, bf16, `torch.compile`
- MLA, MTP, and the 7.26B-total MoE run
- GPT-2 / GPT-3 parity, the speedrun leaderboard

What is left runs single-threaded float32 on CPU with naive O(T²) attention, so it is many orders of magnitude slower. Its demonstrated capability is a 229K-parameter model that, in ~2 minutes end-to-end, memorises the one-digit addition problems it was shown and solves 35–75% of held-out ones, depending on seed.

**If you want to train a usable language model, use [upstream nanochat](https://github.com/karpathy/nanochat).** This fork exists to make every step of training readable — there is no layer you cannot step into with a debugger.

## The full pipeline

Run each stage individually or use `bash runs/speedrun.sh` to chain them:

```bash
# 1. Pretrain a base model (next-token prediction, checkpoints every N steps)
python -m scripts.base_train --depth 4 --num-iterations 400 --run base

# 2. Evaluate: bits-per-byte + CORE-style multiple-choice accuracy + greedy samples
python -m scripts.base_eval --run base

# 3. Supervised finetuning on conversations (only assistant tokens are trained)
python -m scripts.chat_sft --source base --run sft

# 4. Talk to it
python -m scripts.chat_cli --run sft
python -m scripts.chat_cli --run sft -p "3+4"    # single prompt, non-interactive

# Bonus: the quick-iteration scratch trainer (no checkpoints, no eval pipeline)
python -m scripts.scratch_train --n-routed-experts 4   # MoE
python -m scripts.scratch_train --text-file book.txt   # any UTF-8 corpus
```

## What's in it

| File | Replaces | Lines | What's in it |
|------|----------|-------|--------------|
| `nanochat/scratch/tensor.py` | `torch.Tensor`, `torch.autograd` | 660 | Reverse-mode AD: per-op backward closures, reverse-topological backward pass, broadcasting adjoints, fused `softmax`/`cross_entropy`/`rms_norm`, and the `topk`/`index_add` scatter-gather pair the MoE router needs |
| `nanochat/scratch/nn.py` | `torch.nn` | 257 | `Module.__setattr__` parameter/submodule registration, `named_parameters`, `state_dict`, `Linear`, `Embedding`, naive attention with GQA and sliding-window masks |
| `nanochat/scratch/model.py` | `nanochat/gpt.py` | 456 | RMSNorm, RoPE + QK-norm, ResFormer value embeddings, embedding smear, per-layer resid/x0 scalars, mid-layer backout, ReLU² FFNs, DeepSeek-V2 MoE with load-balancing aux loss, tanh-softcapped logits, **KV cache for O(n) decoding** |
| `nanochat/scratch/engine.py` | `nanochat/engine.py` | 210 | KV-cache inference engine: prefill + decode split, multi-sample fanout off one prefill, stop tokens, streaming token generator, Python calculator tool loop |
| `nanochat/scratch/eval.py` | `nanochat/loss_eval.py`, `nanochat/core_eval.py` | 205 | bits-per-byte (token-size-normalised), CORE-style MC/schema/LM task evaluation with few-shot, prompt rendering without jinja2 |
| `nanochat/scratch/checkpoint.py` | `nanochat/checkpoint_manager.py` | 122 | Atomic `.npz` + `meta.json` saves, optimizer state keyed by parameter name, resume, `find_last_step` |
| `nanochat/scratch/optim.py` | `torch.optim`, `nanochat/optim.py` | 226 | AdamW with decoupled decay, and Muon (Polar Express orthogonalization, no SVD) |
| `nanochat/scratch/data.py` | `nanochat/dataloader.py` | 115 | Byte tokenizer, random batches, sequential evaluation batches, the addition corpus with its known entropy floor |
| `nanochat/bpe.py` | `rustbpe`, `tiktoken` | 402 | Byte-level BPE: training (merge counting with incremental updates) and inference, standard library only |

## The default task: one-digit addition

The default corpus is `"7+5=12;"` repeated 20,000 times, with operands drawn from the 80 training pairs (the other 20 are held out). Its entropy is known exactly, so there is a real number to aim at rather than just a curve that goes down. Every character of a line is determined by which operand pair it is, so the only information in the stream is that choice:

\[
H = \frac{\ln N_\text{pairs}}{7} = \frac{\ln 80}{7} = 0.6260 \text{ nats/token} = 0.9031 \text{ bits/byte}
\]

(With `--holdout-frac 0` all 100 pairs are used and the floor is \(2\ln 10 / 7\) = 0.9491 bits/byte.) A model near the floor has learned the training problems; one stuck at ~7.8 has only learned character frequencies (the random baseline over 256 bytes is log₂(256) = 8.0 bits/byte). Neither number says anything about unseen problems — that is what the held-out evaluation is for.

The corpus spec, including the held-out split, is written into every checkpoint's `meta.json`. `base_eval` and `chat_sft` rebuild the data from it, so they always evaluate against what the model was actually trained on, and `--resume` refuses to continue a run on different data.

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
