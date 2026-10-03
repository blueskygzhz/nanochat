"""
Evaluate a base model: `python -m scripts.base_eval`

The from-scratch counterpart of nanochat's `scripts/base_eval.py`. Reports:

  1. **val bits-per-byte** -- the tokenizer-independent compression metric, walked
     over the held-out split in order (not random windows) so the number is stable.
  2. **CORE-style task accuracy** -- each candidate continuation is scored by its mean
     autoregressive loss and the lowest wins. No generation, no answer parsing, so it
     works on a base model.
  3. **samples** -- greedy and sampled completions, through the KV-cache engine.

    python -m scripts.base_eval --run base
    python -m scripts.base_eval --run base --step 100
"""

import argparse
import math
import os

from nanochat.common import get_base_dir
from nanochat.scratch import (
    ByteTokenizer, Dataset, Engine, addition_entropy_floor, evaluate_bpb, evaluate_task,
    list_steps, load_model, make_addition_corpus, token_bytes_table,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default="base", help="checkpoint subdirectory")
    p.add_argument("--step", type=int, default=None, help="step to load (default: latest)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=-1, help="-1 => the whole val split")
    p.add_argument("--text-file", type=str, default=None)
    p.add_argument("--corpus-lines", type=int, default=20000)
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def addition_tasks(n=100, seed=0):
    """A multiple-choice task built from the addition corpus.

    Each item asks for `a+b=` and offers the true sum against a near-miss distractor,
    which is what makes it informative: a model that has learned the digit
    distribution but not carrying will score ~50%.
    """
    import random
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        a, b = rng.randrange(10), rng.randrange(10)
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


def main():
    args = parse_args()
    checkpoints = os.path.join(get_base_dir(), "checkpoints", args.run)
    if not list_steps(checkpoints):
        raise SystemExit(f"no checkpoints in {checkpoints}; run scripts.base_train first")

    model, meta = load_model(checkpoints, args.step)
    tokenizer = ByteTokenizer()
    seq_len = model.config.sequence_len

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
        floor = None
    else:
        text = make_addition_corpus(args.corpus_lines, seed=args.seed)
        floor = addition_entropy_floor()
    dataset = Dataset.from_text(text, tokenizer)

    print(f"base_eval | run={args.run} step={meta['step']} | "
          f"d{model.config.n_layer} w{model.config.n_embd} | {model.num_parameters():,} params")
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

    # 2) CORE-style multiple choice
    tasks = addition_tasks(args.max_examples, seed=args.seed)
    acc = evaluate_task(model, tokenizer, tasks,
                        {"task_type": "multiple_choice", "num_fewshot": 0,
                         "continuation_delimiter": ""},
                        max_examples=args.max_examples)
    print(f"\naddition (MC)     : {acc:.3f}   (chance 0.500, n={len(tasks)})")

    # 3) generation, through the KV-cache engine
    engine = Engine(model, tokenizer)
    print("\nsamples (greedy):")
    correct = 0
    trials = addition_tasks(10, seed=args.seed + 1)
    for item in trials:
        prompt = item["query"]
        ids = tokenizer.encode(prompt, prepend="<|bos|>")
        got = tokenizer.decode(engine.generate_batch(ids, max_tokens=3, temperature=0.0)[0])
        want = item["choices"][item["gold"]]
        ok = got == want
        correct += ok
        print(f"  {prompt}{got!r}{'' if ok else f'  (want {want!r})'}")
    print(f"\ngreedy exact match: {correct}/{len(trials)}")


if __name__ == "__main__":
    main()
