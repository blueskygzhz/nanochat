"""
Pretrain a base model: `python -m scripts.base_train`

The from-scratch counterpart of nanochat's `scripts/base_train.py`. Same pipeline
shape -- warmup/cosine schedule, gradient accumulation, periodic held-out evaluation
in bits-per-byte, checkpointing, and resume -- at a scale that finishes on a CPU.

    python -m scripts.base_train --depth 4 --num-iterations 400
    python -m scripts.base_train --resume                      # continue from the last step
    python -m scripts.base_train --text-file book.txt          # your own corpus

`--depth` is the single complexity dial, as upstream: it sets the number of layers and
derives width, heads and the learning rate from it, so there is one number to turn.

On the default addition corpus, `--holdout-frac` of the operand pairs never appear in
training. The corpus spec (including that split) is written into every checkpoint,
so `base_eval` and `chat_sft` evaluate on exactly the pairs this run never saw.

`--tokenizer bpe` uses the tokenizer trained by `scripts.tok_train`. It is copied into
the run directory and recorded in every checkpoint, so later stages always decode with
the vocabulary the model was trained on, even if `tok_train` is re-run.
"""

import argparse
import math
import os
import time

import numpy as np

from nanochat.common import get_base_dir
from nanochat.scratch import (
    Dataset, GPT, GPTConfig, build_corpus, corpus_spec, encode_corpus, evaluate_bpb,
    find_last_step, load_checkpoint, no_grad, save_checkpoint, setup_optimizer,
    token_bytes_table,
)
from nanochat.tokenizer import load_tokenizer, snapshot_tokenizer, tokenizer_spec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # model: one dial, as upstream
    p.add_argument("--depth", type=int, default=4, help="number of layers; width is derived from it")
    p.add_argument("--sequence-len", type=int, default=64)
    p.add_argument("--window-pattern", type=str, default="SSSL")
    # moe
    p.add_argument("--n-routed-experts", type=int, default=0, help="0 => dense model")
    p.add_argument("--n-shared-experts", type=int, default=0)
    p.add_argument("--num-experts-per-tok", type=int, default=2)
    p.add_argument("--aux-loss-alpha", type=float, default=0.001)
    # optimization
    p.add_argument("--num-iterations", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16, help="rows per micro-batch")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--matrix-lr", type=float, default=0.03)
    p.add_argument("--embedding-lr", type=float, default=0.2)
    p.add_argument("--unembedding-lr", type=float, default=0.02)
    p.add_argument("--scalar-lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--final-lr-frac", type=float, default=0.1)
    # eval / checkpointing
    p.add_argument("--eval-every", type=int, default=100, help="-1 to disable")
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--save-every", type=int, default=-1, help="-1 => only at the end")
    p.add_argument("--run", type=str, default="base", help="checkpoint subdirectory name")
    p.add_argument("--resume", action="store_true", help="resume from the last checkpoint")
    # data
    p.add_argument("--tokenizer", type=str, default="byte", choices=["byte", "bpe"],
                   help="byte: 256 raw bytes, no training needed. bpe: from scripts.tok_train")
    p.add_argument("--tokenizer-dir", type=str, default=None,
                   help="BPE tokenizer location (default: $NANOCHAT_BASE_DIR/tokenizer)")
    p.add_argument("--text-file", type=str, default=None)
    p.add_argument("--corpus-lines", type=int, default=20000)
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="fraction of addition operand pairs kept out of training entirely")
    p.add_argument("--seed", type=int, default=0, help="seeds init, data and the held-out split")
    return p.parse_args()


def derive_config(args, vocab_size):
    """Width, heads and head_dim from depth alone.

    Keeps head_dim at 16 and scales width with depth, which is the usual
    compute-optimal shape (aspect ratio roughly constant). n_kv_head is half of
    n_head, i.e. 2x GQA sharing.
    """
    depth = args.depth
    n_head = max(2, depth)
    n_embd = max(24, 16 * n_head)
    return GPTConfig(
        sequence_len=args.sequence_len, vocab_size=vocab_size,
        n_layer=depth, n_head=n_head, n_kv_head=max(1, n_head // 2), n_embd=n_embd,
        window_pattern=args.window_pattern,
        n_routed_experts=args.n_routed_experts, n_shared_experts=args.n_shared_experts,
        num_experts_per_tok=args.num_experts_per_tok, aux_loss_alpha=args.aux_loss_alpha,
    )


def lr_scale(step, total, warmup_ratio, final_frac):
    """Linear warmup, then cosine decay to `final_frac` of peak."""
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


@no_grad()
def eval_bpb(model, dataset, token_bytes, args, split="val"):
    model.eval()
    n = dataset.num_sequential_batches(args.batch_size, args.sequence_len, split)
    steps = min(args.eval_batches, n)
    if steps == 0:
        model.train()
        return float("nan")
    bpb = evaluate_bpb(model, dataset.sequential_batches(args.batch_size, args.sequence_len, split),
                       steps, token_bytes)
    model.train()
    return bpb


def main():
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = os.path.join(get_base_dir(), "checkpoints", args.run)
    os.makedirs(out_dir, exist_ok=True)
    spec = corpus_spec(args.text_file, args.corpus_lines, args.seed, args.holdout_frac)

    # On resume, the checkpoint's own tokenizer snapshot wins: the vocabulary must not
    # change underneath a partially trained model.
    resume_step = find_last_step(out_dir) if args.resume else None
    if args.resume and resume_step is None:
        print(f"--resume: nothing to resume in {out_dir}, starting fresh")
    if resume_step is not None:
        from nanochat.scratch import load_meta
        tok_spec = load_meta(out_dir, resume_step).get("tokenizer") or tokenizer_spec("byte")
        if tok_spec["kind"] != args.tokenizer:
            raise SystemExit(f"--resume: checkpoint uses the {tok_spec['kind']} tokenizer, "
                             f"but --tokenizer {args.tokenizer} was given")
    else:
        tok_spec = snapshot_tokenizer(tokenizer_spec(args.tokenizer, args.tokenizer_dir), out_dir)
    tokenizer = load_tokenizer(tok_spec)

    text, info = build_corpus(spec)
    floor = info["floor"]
    dataset = Dataset(encode_corpus(text, tokenizer, spec))
    token_bytes = token_bytes_table(tokenizer)

    config = derive_config(args, tokenizer.get_vocab_size())
    model = GPT(config, seed=args.seed)
    optimizer = setup_optimizer(
        model, unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr, scalar_lr=args.scalar_lr, weight_decay=args.weight_decay)

    start_step = 0
    if resume_step is not None:
        model, meta = load_checkpoint(out_dir, resume_step, model=model, optimizer=optimizer)
        # Resuming onto different data would silently produce a model trained on
        # a mixture that no checkpoint describes, and could leak held-out pairs.
        if meta.get("data") not in (None, spec):
            raise SystemExit(f"--resume: checkpoint was trained on {meta['data']}, "
                             f"but this run is configured for {spec}")
        start_step = meta["step"] + 1
        print(f"resumed from step {meta['step']} (val bpb {meta.get('val_bpb', float('nan')):.4f})")

    base_lrs = [g["lr"] for g in optimizer.param_groups]
    tokens_per_step = args.batch_size * args.sequence_len * args.grad_accum_steps
    kind = "MoE" if args.n_routed_experts > 0 else "dense"

    print(f"base_train | {kind} d{config.n_layer} w{config.n_embd} | "
          f"{model.num_parameters():,} params | {tok_spec['kind']} tokenizer, "
          f"vocab {config.vocab_size}")
    print(f"{len(dataset.train):,} train / {len(dataset.val):,} val tokens | "
          f"{tokens_per_step:,} tokens/step x {args.num_iterations} steps")
    if floor is not None:
        # The floor is per *byte* (BOS is special and counts zero bytes), so it holds
        # for any tokenizer -- which is exactly why bpb is the metric.
        print(f"task: addition | {len(info['train_pairs'])} train pairs, "
              f"{len(info['heldout_pairs'])} held out | floor {floor:.4f} nats/byte "
              f"= {floor / math.log(2):.4f} bpb")
    print(f"checkpoints -> {out_dir}")
    print("-" * 76)

    t0 = time.time()
    bpb = None
    for step in range(start_step, args.num_iterations):
        scale = lr_scale(step, args.num_iterations, args.warmup_ratio, args.final_lr_frac)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * scale

        # Gradient accumulation: average the loss over micro-batches so the gradient
        # matches one big batch rather than being grad_accum_steps times too large.
        optimizer.zero_grad()
        total = 0.0
        for _ in range(args.grad_accum_steps):
            loss = model(*dataset.get_batch(args.batch_size, args.sequence_len, rng))
            (loss / args.grad_accum_steps).backward()
            total += loss.item() / args.grad_accum_steps
        optimizer.step()

        is_last = step == args.num_iterations - 1
        bpb = None  # evaluated at most once per step, then shared by log and checkpoint
        if (args.eval_every > 0 and step % args.eval_every == 0) or is_last:
            bpb = eval_bpb(model, dataset, token_bytes, args)
            print(f"step {step:5d} | train {total:.4f} | val bpb {bpb:.4f} "
                  f"| lr x{scale:.2f} | {time.time() - t0:6.1f}s")
        if (args.save_every > 0 and step % args.save_every == 0) or is_last:
            if bpb is None:
                bpb = eval_bpb(model, dataset, token_bytes, args)
            save_checkpoint(out_dir, step, model, optimizer,
                            meta={"val_bpb": bpb, "train_loss": total, "depth": args.depth,
                                  "seed": args.seed, "data": spec, "tokenizer": tok_spec})

    print("-" * 76)
    final = bpb if bpb is not None else eval_bpb(model, dataset, token_bytes, args)
    print(f"done in {time.time() - t0:.1f}s | final val bpb {final:.4f}")
    if floor is not None:
        print(f"gap to bpb floor: {final - floor / math.log(2):+.4f}")
    print(f"last checkpoint: step_{find_last_step(out_dir):06d} in {out_dir}")
    print("next: python -m scripts.base_eval --run", args.run)


if __name__ == "__main__":
    main()
