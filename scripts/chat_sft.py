"""
Supervised finetuning on conversations: `python -m scripts.chat_sft`

The from-scratch counterpart of nanochat's `scripts/chat_sft.py`.

The one idea that matters here is the **loss mask**. A conversation contains both the
user's turn and the assistant's, but we only want to train on the assistant's tokens --
training on the user's turn teaches the model to invent user messages. So targets at
every non-assistant position are set to `-1`, which `cross_entropy` ignores.

`nanochat/tokenizer.py:render_conversation` already returns exactly that mask, so this
script is mostly packing: fit conversations into fixed-length rows, pad the remainder,
and mask the padding too.

    python -m scripts.chat_sft --source base --run sft
"""

import argparse
import math
import os
import time

import numpy as np

from nanochat.common import get_base_dir
from nanochat.scratch import (
    ByteTokenizer, Engine, addition_pairs, load_model, no_grad,
    save_checkpoint, setup_optimizer,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=str, default="base", help="pretrained run to start from")
    p.add_argument("--source-step", type=int, default=None)
    p.add_argument("--run", type=str, default="sft", help="output run name")
    p.add_argument("--num-iterations", type=int, default=600,
                   help="600 is where this task converges; 150 leaves single-digit "
                        "answers off-by-one, because pretraining used zero-padded "
                        "two-digit sums and the chat format drops the padding")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--matrix-lr", type=float, default=0.005, help="lower than pretraining")
    p.add_argument("--embedding-lr", type=float, default=0.02)
    p.add_argument("--unembedding-lr", type=float, default=0.005)
    p.add_argument("--scalar-lr", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------------
# A tiny built-in conversation dataset.
#
# Upstream pulls SmolTalk from HuggingFace. At this scale that is pointless, so the
# task is arithmetic Q&A: it is checkable, and it shares a domain with what the base
# model was pretrained on, which is what makes finetuning show up as a real change.
#
# Conversations are drawn only from the operand pairs the base model was pretrained
# on. The held-out pairs stay unseen through both stages, so they remain a clean test.

def make_conversations(n=400, seed=0, pairs=None):
    import random
    rng = random.Random(seed)
    pairs = pairs or [(a, b) for a in range(10) for b in range(10)]
    out = []
    for _ in range(n):
        a, b = rng.choice(pairs)
        out.append([
            {"role": "user", "content": f"{a}+{b}"},
            {"role": "assistant", "content": f"{a + b}"},
        ])
    return out


def render_conversation(tokenizer, conversation):
    """Token ids plus a 1/0 mask marking which positions to train on.

    `ByteTokenizer` has no chat special tokens, so for it we lay out the turns by hand
    in the same shape the real tokenizer uses and mask everything but the assistant's
    reply. With a trained BPE tokenizer this delegates to `render_conversation`, which
    produces the identical contract.
    """
    if hasattr(tokenizer, "render_conversation"):
        return tokenizer.render_conversation(conversation)

    ids, mask = [tokenizer.get_bos_token_id()], [0]
    for message in conversation:
        body = tokenizer.encode(message["content"])
        if message["role"] == "user":
            ids += tokenizer.encode("U:") + body + tokenizer.encode("\n")
            mask += [0] * (len(body) + 3)
        else:
            prefix = tokenizer.encode("A:")
            suffix = tokenizer.encode("\n")
            ids += prefix + body + suffix
            mask += [0] * len(prefix) + [1] * len(body) + [1] * len(suffix)
    return ids, mask


def pack_batch(tokenizer, conversations, batch_size, seq_len, rng):
    """Pack conversations into (B, T) rows, masking prompts and padding alike."""
    pad = tokenizer.get_bos_token_id()
    rows, masks = [], []
    for _ in range(batch_size):
        row, row_mask = [], []
        while len(row) < seq_len + 1:
            ids, mask = render_conversation(tokenizer, conversations[rng.integers(len(conversations))])
            room = seq_len + 1 - len(row)
            row += ids[:room]
            row_mask += mask[:room]
        rows.append(row[:seq_len + 1])
        masks.append(row_mask[:seq_len + 1])

    arr = np.array(rows, dtype=np.int64)
    m = np.array(masks, dtype=np.int64)
    x = arr[:, :-1]
    y = arr[:, 1:].copy()
    y[m[:, 1:] == 0] = -1   # the loss mask, shifted to line up with targets
    return x, y


@no_grad()
def eval_loss(model, tokenizer, conversations, args, seq_len, batches=5):
    model.eval()
    rng = np.random.default_rng(1234)
    losses = [model(*pack_batch(tokenizer, conversations, args.batch_size, seq_len, rng)).item()
              for _ in range(batches)]
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    base_dir = get_base_dir()
    src_dir = os.path.join(base_dir, "checkpoints", args.source)
    out_dir = os.path.join(base_dir, "checkpoints", args.run)
    os.makedirs(out_dir, exist_ok=True)

    model, meta = load_model(src_dir, args.source_step)
    model.train()
    tokenizer = ByteTokenizer()
    seq_len = model.config.sequence_len

    # Inherit the base model's held-out split so SFT never sees those pairs either
    spec = meta.get("data")
    if spec is not None and spec.get("kind") == "addition":
        train_pairs, held_pairs = addition_pairs(spec["holdout_frac"], spec["seed"])
    else:
        print("warning: source checkpoint has no addition split; nothing is held out")
        train_pairs, held_pairs = addition_pairs(0.0)
    train_convs = make_conversations(400, seed=args.seed, pairs=train_pairs)
    val_convs = make_conversations(80, seed=args.seed + 999, pairs=train_pairs)

    optimizer = setup_optimizer(
        model, unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr, scalar_lr=args.scalar_lr)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    print(f"chat_sft | from {args.source} step {meta['step']} | {model.num_parameters():,} params")
    print(f"{len(train_convs)} train / {len(val_convs)} val conversations | "
          f"{args.num_iterations} steps")
    print(f"checkpoints -> {out_dir}")
    print("-" * 72)

    t0 = time.time()
    for step in range(args.num_iterations):
        t = step / max(args.num_iterations - 1, 1)
        scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * scale

        x, y = pack_batch(tokenizer, train_convs, args.batch_size, seq_len, rng)
        loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.num_iterations - 1:
            val = eval_loss(model, tokenizer, val_convs, args, seq_len)
            print(f"step {step:5d} | train {loss.item():.4f} | val {val:.4f} "
                  f"| lr x{scale:.2f} | {time.time() - t0:6.1f}s")

    save_checkpoint(out_dir, args.num_iterations - 1, model, optimizer,
                    meta={"source": args.source, "source_step": meta["step"],
                          "data": spec,
                          "val_loss": eval_loss(model, tokenizer, val_convs, args, seq_len)})
    print("-" * 72)

    # Does it answer in the finetuned format -- on pairs it trained on, and on pairs
    # neither pretraining nor SFT ever showed it?
    model.eval()
    engine = Engine(model, tokenizer)
    for name, pairs in (("seen pairs", train_pairs), ("held-out pairs", held_pairs)):
        if not pairs:
            print(f"{name:16s}: n/a (no pairs held out)")
            continue
        correct = chat_exact_match(engine, tokenizer, pairs)
        print(f"{name:16s}: exact match {correct}/{len(pairs)}")
    print("(only the held-out row measures generalisation)")
    print(f"next: python -m scripts.chat_cli --run {args.run}")


def chat_exact_match(engine, tokenizer, pairs):
    """Greedy-decode the assistant turn for `a+b` and compare against the sum."""
    correct = 0
    for a, b in pairs:
        ids, _ = render_conversation(tokenizer, [{"role": "user", "content": f"{a}+{b}"}])
        ids = ids + tokenizer.encode("A:")
        got = tokenizer.decode(engine.generate_batch(ids, max_tokens=4, temperature=0.0)[0])
        correct += got.split("\n")[0].strip() == str(a + b)
    return correct


if __name__ == "__main__":
    main()
