# nanochat (from scratch, no framework)

This is a fork of [nanochat](https://github.com/karpathy/nanochat) with **PyTorch removed entirely**. Nothing here imports a deep learning framework. The autograd engine, the module system, the layers, the optimizers, the tokenizer, the inference engine with KV caching, the evaluation metrics, and the model are all written out by hand.

numpy is the only runtime dependency that does any math, and it supplies exactly two things: an n-dimensional array and a BLAS matrix multiply. It provides no automatic differentiation, no layers, no optimizers and no model — that is all code in this repo.

```bash
uv sync --group dev
source .venv/bin/activate
bash runs/speedrun.sh   # the full pipeline: pretrain → eval → SFT → chat
```

```
=== 1/5  Pretrain ===
base_train | dense d4 w64 | 231,186 params | byte tokenizer, vocab 265
task: addition | 80 train pairs, 20 held out | floor 0.6260 nats/byte = 0.9031 bpb
done in 40s | final val bpb 0.9556

=== 2/5  Evaluate the base model ===
val bits per byte : 0.9556   entropy floor 0.9031, gap +0.0524
                      MC acc      greedy
seen pairs             1.000       80/80
held-out pairs         0.700       13/20

=== 3/5  Finetune on conversations ===
seen pairs      : exact match 80/80
held-out pairs  : exact match 15/20

=== 4/5  Evaluate the chat model ===
                    turn 1    turn 2    pass@4
seen pairs           80/80     80/80     80/80
held-out pairs       15/20     15/20     15/20

=== 5/5  Talk to it ===
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
- MLA and the 7.26B-total MoE run (MTP has since been re-added, DeepSeek-V3 style)
- GPT-2 / GPT-3 parity, the speedrun leaderboard

What is left runs single-threaded float32 on CPU with naive O(T²) attention, so it is many orders of magnitude slower. Its demonstrated capability is a 229K-parameter model that, in ~2 minutes end-to-end, memorises the one-digit addition problems it was shown and solves 35–75% of held-out ones, depending on seed.

**If you want to train a usable language model, use [upstream nanochat](https://github.com/karpathy/nanochat).** This fork exists to make every step of training readable — there is no layer you cannot step into with a debugger.

## The full pipeline

Run each stage individually or use `bash runs/speedrun.sh` to chain them:

```bash
# 0. (optional) Train a BPE tokenizer -- offline, on any local text file
python -m scripts.tok_train --text-file book.txt --vocab-size 512

# 1. Pretrain a base model (next-token prediction, checkpoints every N steps)
python -m scripts.base_train --depth 4 --num-iterations 400 --run base
python -m scripts.base_train --tokenizer bpe --run base   # ...with the BPE tokenizer

# 2. Evaluate: bits-per-byte + CORE-style multiple-choice accuracy + greedy decoding
python -m scripts.base_eval --run base

# 3. Supervised finetuning on conversations (only assistant tokens are trained)
python -m scripts.chat_sft --source base --run sft

# 4. Evaluate the chat model: generated answers, second-turn answers, pass@k
python -m scripts.chat_eval --run sft
python -m scripts.chat_eval --run sft --tasks all --max-problems 100   # + ARC/MMLU/GSM8K/HumanEval

# 5. Talk to it -- multi-turn; oldest exchanges are dropped when the context fills
python -m scripts.chat_cli --run sft
python -m scripts.chat_cli --run sft -p "3+4"    # single prompt, non-interactive

# Variants
python -m scripts.base_train --n-routed-experts 4 --run moe   # MoE
python -m scripts.base_train --n-mtp 2 --run mtp              # + multi-token prediction
python -m scripts.base_train --text-file book.txt --run book  # any UTF-8 corpus
```

With `--n-mtp D` the model trains D DeepSeek-V3 MTP modules (loss `L + λ/D·ΣL_k`), and
`Engine` then decodes speculatively by default: the MTP chain drafts D tokens, one target
forward verifies them, and the Leviathan et al. acceptance rule keeps the output
distribution exactly the model's own — for greedy, any temperature and top-k, any draft
temperature (`draft_temperature=`), and per-sample temperatures (`temperature=[0, 0.7, 1]`).
`base_eval` reports the acceptance rate and checks greedy output is token-identical.

`TOKENIZER=bpe bash runs/speedrun.sh` runs the whole thing with a BPE tokenizer trained on this README.

Every checkpoint's `meta.json` records the corpus (with its held-out split) and the tokenizer; a BPE tokenizer is copied into the run directory. Every later stage reads both from there, so evaluation always uses the data split and vocabulary the model was actually trained with — re-running `tok_train` cannot change the tokens under an existing model, and `--resume` refuses a different corpus or tokenizer.

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
| `nanochat/chat_format.py` | — | 250 | The chat layout shared by SFT and inference: message validation, prefill, loss masks, reply parsing into text/tool parts with a stop reason, history truncation |

**Special tokens: structure is never text.** Both tokenizers reserve ids for `SPECIAL_TOKENS` (BOS, turn markers, tool markers) — the byte tokenizer has 256 byte ids plus 9 specials = 265. Text encoding is ordinary-only, so no string, including a literal `<|assistant_end|>`, can produce a special id: a user cannot forge a turn, and a `;` or a newline is just a character. This is the same separation Anthropic made in moving from the Text Completions API (turns written as `\n\nHuman:` / `\n\nAssistant:` text) to the Messages API (structured messages, rendered by the server), and `chat_format` follows the Messages API's documented semantics: roles alternate starting with the user, a final assistant message is a prefill that may not end in whitespace, replies end with an end-of-turn token (a reply may contain newlines) and come back as structured content with a stop reason (`end_turn` / `max_tokens`). Checkpoints trained with the old 256-id byte tokenizer (BOS = `;`, turns as `U:`/`A:` text) still load: their tokenizer spec has no `vocab_size`, which selects `LegacyByteTokenizer`.

**Memory and speed.** Like torch, `backward()` frees each interior gradient as soon as it has been propagated and releases the graph (the closures holding saved activations) unless `retain_graph=True`; a second backward through a freed graph raises instead of silently mis-accumulating. Attention runs GQA as a plain batched matmul (query heads sharing a KV head are stacked as rows, so K/V are never copied), the causal mask is fused into the softmax, and `relu²` is one op. Against the previous version, a d8/T=256 training step uses 63% less peak memory (1770 → 651 MB) and runs 13–15% faster; float64 gradients agree to 4e-9 relative.

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
│   ├── chat_format.py              # Chat layout + loss mask, shared by SFT and inference
│   ├── common.py                   # Logging, base dir, locked download
│   ├── dataset.py                  # Download/read utils for pretraining data (stdlib HTTP)
│   ├── execution.py                # Sandboxed Python execution (for the humaneval task)
│   ├── scratch                     # The training stack, from scratch
│   │   ├── checkpoint.py           # Atomic save/load/resume
│   │   ├── data.py                 # Byte tokenizer, corpus specs, held-out split, batching
│   │   ├── engine.py               # KV-cache inference engine + calculator tool loop
│   │   ├── eval.py                 # Bits-per-byte, CORE-style task evaluation
│   │   ├── model.py                # The GPT architecture
│   │   ├── nn.py                   # Module/Parameter system and layers
│   │   ├── optim.py                # AdamW + Muon
│   │   └── tensor.py               # Reverse-mode autograd engine
│   └── tokenizer.py                # Tokenizer wrapper, special tokens, tokenizer specs
├── runs
│   └── speedrun.sh                 # The whole pipeline (TOKENIZER=bpe for BPE)
├── scripts
│   ├── base_train.py               # 1. Pretrain
│   ├── base_eval.py                # 2. Evaluate the base model
│   ├── chat_sft.py                 # 3. Finetune on conversations
│   ├── chat_eval.py                # 4. Evaluate the chat model
│   ├── chat_benchmarks.py          #    ARC/MMLU/GSM8K/HumanEval scoring for chat_eval --tasks
│   ├── chat_cli.py                 # 5. Talk to it
│   ├── chat_cai.py                 # post: SL-CAI (critique -> revision -> finetune)
│   ├── chat_pm.py                  # post: preference models from AI feedback
│   ├── chat_rl.py                  # post: PPO with KL penalty
│   ├── posttrain_common.py         #       shared helpers for the three above
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it (local file or parquet shards)
├── tasks                           # Eval task data loaders (arc, gsm8k, mmlu, ...)
└── tests
    ├── test_benchmarks.py          # Benchmark scoring, offline (fake hub data, stub model)
    ├── test_bpe.py                 # BPE training/encoding, rustbpe+tiktoken parity
    ├── test_chat_format.py         # Chat layout, injection resistance, reply parsing
    ├── test_rlhf.py                # PM loss, GAE, PPO clipping, KL rewards, CAI feedback
    ├── test_execution.py           # Sandboxed code execution
    ├── test_pipeline.py            # KV cache, engine, eval, checkpoints, tokenizers, chat
    ├── test_scratch.py             # Autograd, layers, model, optimizers, convergence
    ├── test_tasks.py               # Task slicing, mixtures, HubDataset
    └── test_tokenizer.py           # BPE round-trips, chat rendering
```

## Post-training: Constitutional AI and RLHF

`POSTTRAIN=1 bash runs/speedrun.sh` adds three stages after SFT, following the two papers in which Anthropic published its post-training recipe — *Training a Helpful and Harmless Assistant with RLHF* (arXiv:2204.05862) and *Constitutional AI* (arXiv:2212.08073). Anthropic has not published how any specific Claude model (3.x or later) was post-trained beyond saying it uses Constitutional AI and RLHF, plus the "character" variant described in *Claude's Character* (2024); those published methods are what is implemented.

```bash
python -m scripts.chat_cai --source sft --run cai          # SL-CAI: critique -> revision -> finetune
python -m scripts.chat_pm  --source cai --run pm           # comparisons + AI feedback -> 2 preference models
python -m scripts.chat_rl  --source cai --pm pm --run rl   # PPO on r_PM - lambda * KL(pi || pi_0)
```

| Paper | Here |
|---|---|
| Constitution: principles, each with a comparison question, a critique request and a revision request | `nanochat/constitution.py` |
| SL-CAI: sample, critique and revise against a randomly drawn principle (repeatedly), finetune on revisions mixed with helpful data | `scripts/chat_cai.py` |
| RL-CAI feedback: multiple-choice prompt, one random principle per comparison, soft labels from normalised `(A)`/`(B)` log-probs, optional 0.4–0.6 clamp for CoT labels | `LMFeedback.compare`, `chat_pm --clamp` |
| PM: policy-family transformer + scalar head on the last token, Bradley-Terry loss (soft-label form) | `RewardModel`, `preference_loss` |
| Robustness: two PMs on disjoint halves of the comparisons; optimise one, score with the other | `chat_pm` (train/test), `chat_rl` (PM-train / PM-test) |
| RL: PPO with reward `r_PM − λ·KL(π‖π₀)`, π₀ = the SL-CAI model; reward vs √KL as the diagnostic | `chat_rl`, `kl_penalized_rewards`, `ppo_policy_loss`, `gae` |
| Character training: self-ranked replies → preference model | `ranking_to_comparisons` feeds the same PM loss |

**The judge is a stand-in.** The papers' feedback model is a large language model reading the conversation. A 230K-parameter model trained on addition cannot judge or revise anything, so by default (`--feedback rule`) each principle is checked and revised programmatically; every script says so in its output. `LMFeedback` implements the real prompt and soft labels (tested against a stub model) and is used with `--feedback lm:<run>` once there is a model capable of the job. Everything else — principle sampling, soft labels, PM, PPO — is identical either way.

**What actually happens at this scale** (measured, default pipeline): SL-CAI fixes the few wrong answers on seen pairs (78 → 80/80; held-out 5 → 7/20). The PMs reach 0.79–0.90 accuracy on comparisons they never saw. PPO then has almost nothing left to improve — the SL-CAI policy already samples gold replies 94–99% of the time — and at the papers' λ_KL = 0.001 it does what the papers warn about: PM reward rises while gold accuracy collapses from 0.94 to 0.06 (the policy learns to answer "1" to everything), with PM-test rising too, because both PMs share the same blind spot. Hence `chat_rl` defaults to λ = 0.2, which holds gold accuracy at 0.94–1.00; the full sweep is in its docstring. That is the honest result: the machinery is the published one, and it reproduces the published failure mode, but at this scale RL does not make the model better.

## Standard benchmarks

`chat_eval --tasks` runs the benchmarks in `tasks/` (data downloaded from the HuggingFace hub on first use and cached): ARC and MMLU by argmax over the answer letters' logits, GSM8K and HumanEval by greedy decoding checked with the task's own answer extraction / sandboxed test execution. Each score is printed beside its chance baseline and as a centered accuracy, whose mean is upstream's ChatCORE. On the default model, 100 problems each:

```
task                n     acc  chance  centered   cropped
ARC-Easy          100   0.220   0.249    -0.039   100/100
ARC-Challenge     100   0.270   0.250    +0.027   100/100
MMLU              100   0.320   0.250    +0.093   100/100
GSM8K             100   0.000   0.000    +0.000   100/100
HumanEval         100   0.000   0.000    +0.000   100/100
ChatCORE                                 +0.016
```

That is chance — a 229K-parameter model trained on one-digit addition knows nothing about science questions or code, and every prompt had to be cropped to fit its 64-token context. The point of running them is that the harness is real: the same commands score a capable model correctly (reference answers score 100% on all five tasks).

## Packaging

`pyproject.toml` builds a wheel containing the `nanochat` package only (hatchling). `uv sync` installs it in editable mode, which also puts the checkout on `sys.path`, so `scripts/` and `tasks/` stay runnable as `python -m scripts.<name>`; they are deliberately not shipped in the wheel, where they would occupy the generic top-level names `scripts` and `tasks`.

## Acknowledgements

This is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat). All architecture ideas, the training recipe, and the tokenizer design are from upstream; this fork only reimplements them without a framework. The MoE layer follows DeepSeek-V2, Muon is from [Keller Jordan](https://kellerjordan.github.io/posts/muon/) with [Polar Express](https://arxiv.org/pdf/2505.16932) coefficients, and the speculative-decoding math is [Leviathan et al.](https://arxiv.org/abs/2211.17192).

## License

MIT
