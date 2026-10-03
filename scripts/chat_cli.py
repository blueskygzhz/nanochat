"""
Talk to a finetuned model: `python -m scripts.chat_cli`

The from-scratch counterpart of nanochat's `scripts/chat_cli.py`. A real multi-turn
conversation: every turn is answered with the whole history in context, and tokens
are streamed from the KV-cache engine as they are sampled.

When the history no longer fits in the model's context window, the oldest exchanges
are dropped (whole user+assistant pairs, so the history still starts with the user).

    python -m scripts.chat_cli --run sft
    python -m scripts.chat_cli --run sft -p "3+4"        # single prompt, then exit
    python -m scripts.chat_cli --run sft --temperature 0  # greedy

In interactive mode, `/clear` starts a new conversation and an empty line quits.
"""

import argparse
import os
import sys

from nanochat.chat_format import fit_history, reply_stop_tokens
from nanochat.common import get_base_dir
from nanochat.scratch import Engine, list_steps, load_model
from nanochat.tokenizer import load_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default="sft", help="checkpoint subdirectory")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("-p", "--prompt", type=str, default=None, help="answer one prompt and exit")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def respond(engine, tokenizer, messages, args, out=sys.stdout):
    """Stream the assistant's reply to `messages` (ending in a user turn) and return it.

    Returns `(reply_text, dropped)` where `dropped` is how many old messages had to be
    cut to fit the context.
    """
    seq_len = engine.model.config.sequence_len
    max_tokens = min(args.max_tokens, seq_len - 2)
    kept, ids = fit_history(tokenizer, messages, budget=seq_len - max_tokens)
    stop = set(reply_stop_tokens(tokenizer))

    pieces, shown = [], ""
    for token, _ in engine.generate(ids, max_tokens=max_tokens, temperature=args.temperature,
                                    top_k=args.top_k, seed=args.seed, stop_tokens=sorted(stop)):
        if token in stop:
            break
        pieces.append(token)
        # Re-decode the whole reply each step and print only the newly resolved text.
        # A byte-level token can be half of a UTF-8 character, so decoding
        # token-by-token would emit replacement characters that later turn out wrong.
        text = tokenizer.decode(pieces)
        if text.endswith("\ufffd"):
            continue  # incomplete character, wait for the next token
        out.write(text[len(shown):])
        out.flush()
        shown = text
    return tokenizer.decode(pieces), len(messages) - len(kept)


def main():
    args = parse_args()
    checkpoints = os.path.join(get_base_dir(), "checkpoints", args.run)
    if not list_steps(checkpoints):
        raise SystemExit(f"no checkpoints in {checkpoints}; run scripts.chat_sft first")

    model, meta = load_model(checkpoints, args.step)
    tokenizer = load_tokenizer(meta.get("tokenizer"))
    engine = Engine(model, tokenizer)

    if args.prompt is not None:
        respond(engine, tokenizer, [{"role": "user", "content": args.prompt}], args)
        print()
        return

    print(f"chat_cli | run={args.run} step={meta['step']} | {model.num_parameters():,} params "
          f"| context {model.config.sequence_len} tokens")
    print(f"temperature={args.temperature}  (this model was finetuned on 'a+b' arithmetic)")
    print("/clear for a new conversation, empty line or Ctrl-C to quit.\n")
    history = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        if user == "/clear":
            history = []
            print("(new conversation)")
            continue
        history.append({"role": "user", "content": user})
        print("Bot: ", end="")
        try:
            reply, dropped = respond(engine, tokenizer, history, args)
        except ValueError as e:  # a single message longer than the context
            history.pop()
            print(f"\n(error: {e})")
            continue
        print()
        if dropped:
            history = history[dropped:]
            print(f"(context full: dropped the {dropped // 2} oldest exchange(s))")
        history.append({"role": "assistant", "content": reply.strip()})


if __name__ == "__main__":
    main()
