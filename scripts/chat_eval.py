"""
Evaluate a chat (SFT) model: `python -m scripts.chat_eval`

The from-scratch counterpart of nanochat's `scripts/chat_eval.py`. Where `base_eval`
scores a base model by likelihood (no generation needed), a chat model is judged on
what it *says*: each question is rendered in the chat format, the reply is decoded
greedily through the KV-cache engine until the end-of-turn token, and compared with
the answer. Upstream does this for GSM8K / HumanEval / MMLU; at this scale the task
is the arithmetic the model was finetuned on.

Reported separately for operand pairs seen in training and pairs held out of both
pretraining and SFT. Only the latter measures generalisation.

It also checks the chat machinery itself, independent of how good the model is:
  - **multi-turn**: the same questions asked as the second turn of a conversation,
    after an earlier exchange. A model that only works on turn one has learned a
    prompt shape, not a conversation format.
  - **sampling**: pass@k at temperature > 0 through multi-sample generation.

    python -m scripts.chat_eval --run sft
"""

import argparse
import os

from nanochat.chat_format import render_prompt, reply_stop_tokens
from nanochat.common import get_base_dir
from nanochat.scratch import Engine, addition_pairs, list_steps, load_model
from nanochat.tokenizer import load_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default="sft", help="checkpoint subdirectory")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=4, help="k for pass@k")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def reply(engine, tokenizer, messages, max_tokens=8, **kwargs):
    """Greedy (by default) assistant replies to a history ending in a user turn."""
    ids = render_prompt(tokenizer, messages)
    stop = set(reply_stop_tokens(tokenizer))
    kwargs.setdefault("temperature", 0.0)
    outs = engine.generate_batch(ids, max_tokens=max_tokens, stop_tokens=sorted(stop), **kwargs)
    return [tokenizer.decode([t for t in out if t not in stop]).strip() for out in outs]


def chat_exact_match(engine, tokenizer, pairs, history=()):
    """How many `a+b` questions get exactly `a+b` back, optionally after `history`."""
    correct = 0
    for a, b in pairs:
        messages = list(history) + [{"role": "user", "content": f"{a}+{b}"}]
        correct += reply(engine, tokenizer, messages)[0] == str(a + b)
    return correct


def pass_at_k(engine, tokenizer, pairs, k, temperature, seed):
    """Fraction of questions where at least one of k sampled replies is right."""
    hits = 0
    for i, (a, b) in enumerate(pairs):
        answers = reply(engine, tokenizer, [{"role": "user", "content": f"{a}+{b}"}],
                        num_samples=k, temperature=temperature, seed=seed + i)
        hits += str(a + b) in answers
    return hits


def main():
    args = parse_args()
    checkpoints = os.path.join(get_base_dir(), "checkpoints", args.run)
    if not list_steps(checkpoints):
        raise SystemExit(f"no checkpoints in {checkpoints}; run scripts.chat_sft first")
    model, meta = load_model(checkpoints, args.step)
    tokenizer = load_tokenizer(meta.get("tokenizer"))
    engine = Engine(model, tokenizer)

    spec = meta.get("data") or {}
    if spec.get("kind") == "addition":
        seen, held = addition_pairs(spec["holdout_frac"], spec["seed"])
    else:
        print("warning: checkpoint has no addition split; every pair counts as seen")
        seen, held = addition_pairs(0.0)

    # An earlier exchange from the *seen* pairs, so turn two is the only new variable
    a0, b0 = seen[0]
    history = [{"role": "user", "content": f"{a0}+{b0}"},
               {"role": "assistant", "content": str(a0 + b0)}]

    print(f"chat_eval | run={args.run} step={meta['step']} | {model.num_parameters():,} params "
          f"| {(meta.get('tokenizer') or {'kind': 'byte'})['kind']} tokenizer")
    print("-" * 72)
    header = f"{'':16s}{'turn 1':>10s}{'turn 2':>10s}{f'pass@{args.num_samples}':>10s}"
    print(header)
    for name, pairs in (("seen pairs", seen), ("held-out pairs", held)):
        if not pairs:
            print(f"{name:16s}{'n/a':>10s}{'n/a':>10s}{'n/a':>10s}")
            continue
        n = len(pairs)
        t1 = chat_exact_match(engine, tokenizer, pairs)
        t2 = chat_exact_match(engine, tokenizer, pairs, history=history)
        pk = pass_at_k(engine, tokenizer, pairs, args.num_samples, args.temperature, args.seed)
        print(f"{name:16s}{f'{t1}/{n}':>10s}{f'{t2}/{n}':>10s}{f'{pk}/{n}':>10s}")
    print(f"(turn 1/2: greedy exact match; pass@{args.num_samples}: "
          f"T={args.temperature}. Only held-out rows measure generalisation.)")


if __name__ == "__main__":
    main()
