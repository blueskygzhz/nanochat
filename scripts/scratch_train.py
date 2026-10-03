"""
Train nanochat's GPT with zero torch: `python -m scripts.scratch_train`

Everything below runs on `nanochat.scratch`, which is numpy plus a hand-written
autograd engine. It is slow (CPU, float32, naive attention), so the defaults are
sized to finish in about a minute.

The default task is two-digit addition, whose entropy is known exactly, so the run
reports how close the model got to the information-theoretic floor instead of just
printing a loss curve. Point `--text-file` at anything to train on real text instead.

Examples
--------
    python -m scripts.scratch_train                          # dense, addition
    python -m scripts.scratch_train --n-routed-experts 4     # MoE
    python -m scripts.scratch_train --text-file book.txt --n-layer 6
"""

import argparse
import math
import time

import numpy as np

from nanochat.scratch import (
    Dataset, GPT, GPTConfig, addition_entropy_floor, make_addition_corpus, no_grad, setup_optimizer,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-kv-head", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--sequence-len", type=int, default=64)
    p.add_argument("--window-pattern", type=str, default="SSSL")
    # moe
    p.add_argument("--n-routed-experts", type=int, default=0, help="0 => dense model")
    p.add_argument("--n-shared-experts", type=int, default=0)
    p.add_argument("--num-experts-per-tok", type=int, default=2)
    p.add_argument("--aux-loss-alpha", type=float, default=0.001)
    # optimization
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--matrix-lr", type=float, default=0.03)
    p.add_argument("--embedding-lr", type=float, default=0.2)
    p.add_argument("--unembedding-lr", type=float, default=0.02)
    p.add_argument("--scalar-lr", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=8)
    # data
    p.add_argument("--text-file", type=str, default=None, help="train on this UTF-8 file instead")
    p.add_argument("--corpus-lines", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def lr_scale(step, total, warmup):
    """Linear warmup then cosine decay to 10% of peak."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


@no_grad()
def evaluate(model, dataset, args, rng):
    model.eval()
    losses = [model(*dataset.get_batch(args.batch_size, args.sequence_len, rng, "val")).item()
              for _ in range(args.eval_batches)]
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
        floor = None
    else:
        text = make_addition_corpus(args.corpus_lines, seed=args.seed)
        floor = addition_entropy_floor()
    dataset = Dataset.from_text(text)

    config = GPTConfig(
        sequence_len=args.sequence_len, vocab_size=256,
        n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head, n_embd=args.n_embd,
        window_pattern=args.window_pattern,
        n_routed_experts=args.n_routed_experts, n_shared_experts=args.n_shared_experts,
        num_experts_per_tok=args.num_experts_per_tok, aux_loss_alpha=args.aux_loss_alpha,
    )
    model = GPT(config)
    optimizer = setup_optimizer(model,
                                unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr,
                                matrix_lr=args.matrix_lr, scalar_lr=args.scalar_lr)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    kind = "MoE" if args.n_routed_experts > 0 else "dense"
    print(f"from-scratch nanochat | {kind} | {model.num_parameters():,} params | "
          f"{len(dataset.train):,} train tokens")
    print(f"tokens/step: {args.batch_size * args.sequence_len:,} | steps: {args.steps}")
    if floor is not None:
        print(f"task: 2-digit addition | entropy floor: {floor:.4f} nats/token "
              f"| uniform-byte baseline: {math.log(256):.4f}")
    print("-" * 68)

    t0 = time.time()
    for step in range(args.steps):
        scale = lr_scale(step, args.steps, args.warmup)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * scale

        x, y = dataset.get_batch(args.batch_size, args.sequence_len, rng)
        loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            val = evaluate(model, dataset, args, np.random.default_rng(1234))
            extra = f" | gap to floor {val - floor:+.4f}" if floor is not None else ""
            print(f"step {step:4d} | train {loss.item():.4f} | val {val:.4f}"
                  f" | lr x{scale:.2f} | {time.time() - t0:6.1f}s{extra}")

    print("-" * 68)
    from nanochat.scratch import ByteTokenizer
    tokenizer = ByteTokenizer()
    if floor is None:
        out = model.generate(tokenizer.encode(text[:16]), 64, temperature=0.8, top_k=40, seed=args.seed)
        print("sample:", repr(tokenizer.decode(out)))
    else:
        # Greedy-decode held-out sums and check the digits are actually right.
        trials = [(int(a), int(b)) for a, b in rng.integers(0, 10, (20, 2))]
        decoded = []
        for a, b in trials:
            prompt = f"{a}+{b}="
            out = model.generate(tokenizer.encode(prompt), 3, temperature=0.0)
            decoded.append((prompt, tokenizer.decode(out)[len(prompt):], f"{a + b:02d};"))
        correct = sum(got == want for _, got, want in decoded)
        print(f"greedy addition accuracy: {correct}/{len(decoded)}")
        for prompt, got, want in decoded[:5]:
            print(f"  {prompt}{got}  (want {want})")


if __name__ == "__main__":
    main()
