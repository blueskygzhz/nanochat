"""
Evaluate a base model: `python -m scripts.base_eval`

The from-scratch counterpart of nanochat's `scripts/base_eval.py`. Reports:

  1. **val bits-per-byte** -- the tokenizer-independent compression metric, walked
     over the held-out split in order (not random windows) so the number is stable.
  2. **CORE-style task accuracy** -- each candidate continuation is scored by its mean
     autoregressive loss and the lowest wins. No generation, no answer parsing, so it
     works on a base model.
  3. **greedy exact match** -- decoding through the KV-cache engine.

(2) and (3) are reported twice: on operand pairs the model was trained on, and on
pairs held out of training. Seen-pair accuracy measures memorisation; only the
held-out number measures whether the model learned to add.

The corpus (and the held-out split) is rebuilt from the spec recorded in the
checkpoint, so the evaluation always matches what the model was actually trained on.

    python -m scripts.base_eval --run base
    python -m scripts.base_eval --run base --step 100
"""

import argparse
import math
import os
import random

from nanochat.common import get_base_dir
from nanochat.scratch import (
    Dataset, Engine, build_corpus, corpus_spec, encode_corpus, evaluate_bpb, evaluate_task,
    list_steps, load_model, token_bytes_table,
)
from nanochat.tokenizer import load_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default="base", help="checkpoint subdirectory")
    p.add_argument("--step", type=int, default=None, help="step to load (default: latest)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=-1, help="-1 => the whole val split")
    p.add_argument("--text-file", type=str, default=None,
                   help="override the corpus recorded in the checkpoint")
    p.add_argument("--seed", type=int, default=0, help="seeds the MC distractors")
    return p.parse_args()


def resolve_corpus(meta, text_file=None):
    """The corpus spec to evaluate on: an explicit override, else the checkpoint's own."""
    if text_file:
        return corpus_spec(text_file=text_file)
    if "data" in meta:
        return meta["data"]
    # Checkpoints written before the spec was recorded used the full 100-pair corpus
    print("warning: checkpoint has no data spec; assuming the legacy addition corpus "
          "with no held-out pairs")
    return corpus_spec(holdout_frac=0.0)


def addition_mc_items(pairs, seed=0):
    """One multiple-choice item per operand pair.

    Each item asks for `a+b=` and offers the true sum against a near-miss distractor,
    which is what makes it informative: a model that has learned the digit
    distribution but not the table will score ~50%.
    """
    rng = random.Random(seed)
    items = []
    for a, b in pairs:
        true = f"{a + b:02d};"
        wrong = f"{(a + b + rng.choice([-1, 1])) % 100:02d};"
        if wrong == true:
            wrong = f"{(a + b + 2) % 100:02d};"
        choices = [true, wrong]
        gold = 0
        if rng.random() < 0.5:                 # shuffle so position carries no signal
            choices, gold = [wrong, true], 1
        items.append({"query": f"{a}+{b}=", "choices": choices, "gold": gold})
    return items


def greedy_exact_match(engine, tokenizer, pairs):
    """Decode `a+b=` greedily and compare against the zero-padded sum.

    Generates exactly as many tokens as the answer has under this tokenizer: 3 for
    bytes (`0`, `7`, `;`), typically 2 for BPE (`07`, `;`).
    """
    correct = 0
    for a, b in pairs:
        want = f"{a + b:02d};"
        ids = tokenizer.encode(f"{a}+{b}=", prepend="<|bos|>")
        out = engine.generate_batch(ids, max_tokens=len(tokenizer.encode(want)), temperature=0.0)
        correct += tokenizer.decode(out[0]) == want
    return correct


def main():
    args = parse_args()
    checkpoints = os.path.join(get_base_dir(), "checkpoints", args.run)
    if not list_steps(checkpoints):
        raise SystemExit(f"no checkpoints in {checkpoints}; run scripts.base_train first")

    model, meta = load_model(checkpoints, args.step)
    tokenizer = load_tokenizer(meta.get("tokenizer"))
    seq_len = model.config.sequence_len

    spec = resolve_corpus(meta, args.text_file)
    text, info = build_corpus(spec)
    floor = info["floor"]
    dataset = Dataset(encode_corpus(text, tokenizer, spec))

    print(f"base_eval | run={args.run} step={meta['step']} | "
          f"d{model.config.n_layer} w{model.config.n_embd} | {model.num_parameters():,} params "
          f"| {meta.get('tokenizer', {'kind': 'byte'})['kind']} tokenizer")
    print(f"corpus: {spec}")
    print("-" * 76)

    # 1) bits per byte over the held-out split, in order
    token_bytes = token_bytes_table(tokenizer)
    available = dataset.num_sequential_batches(args.batch_size, seq_len, "val")
    steps = available if args.eval_batches < 0 else min(args.eval_batches, available)
    bpb = evaluate_bpb(model, dataset.sequential_batches(args.batch_size, seq_len, "val"),
                       steps, token_bytes)
    print(f"val bits per byte : {bpb:.4f}   ({steps} batches, "
          f"{steps * args.batch_size * seq_len:,} tokens)")
    if floor is not None:
        bpb_floor = floor / math.log(2)
        print(f"  entropy floor   : {bpb_floor:.4f}   gap {bpb - bpb_floor:+.4f}")
    print(f"  random baseline : {math.log2(tokenizer.get_vocab_size()):.4f}")

    engine = Engine(model, tokenizer)
    if info["train_pairs"] is None:
        # A free-text corpus has no task; show a continuation of held-out text instead
        prompt = tokenizer.decode(dataset.val[:16].tolist())
        room = seq_len - len(tokenizer.encode(prompt, prepend="<|bos|>"))
        sample = engine.generate_text(prompt, max_tokens=max(1, min(48, room)),
                                      temperature=0.8, top_k=40)
        print(f"\nsample: {prompt!r} -> {sample!r}")
        return

    # 2) + 3) on seen and on held-out operand pairs
    mc_meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": ""}
    print(f"\n{'':18s}{'MC acc':>10s}{'greedy':>12s}")
    for name, pairs in (("seen pairs", info["train_pairs"]),
                        ("held-out pairs", info["heldout_pairs"])):
        if not pairs:
            print(f"{name:18s}{'n/a':>10s}{'n/a':>12s}   (no pairs held out)")
            continue
        acc = evaluate_task(model, tokenizer, addition_mc_items(pairs, args.seed), mc_meta)
        correct = greedy_exact_match(engine, tokenizer, pairs)
        print(f"{name:18s}{acc:10.3f}{f'{correct}/{len(pairs)}':>12s}")
    print("(MC chance is 0.500. Only the held-out row measures generalisation.)")


if __name__ == "__main__":
    main()
