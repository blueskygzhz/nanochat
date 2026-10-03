"""
Talk to a finetuned model: `python -m scripts.chat_cli`

The from-scratch counterpart of nanochat's `scripts/chat_cli.py`. Streams tokens from
the KV-cache engine as they are sampled.

    python -m scripts.chat_cli --run sft
    python -m scripts.chat_cli --run sft -p "3+4"        # single prompt, then exit
    python -m scripts.chat_cli --run sft --temperature 0  # greedy
"""

import argparse
import os
import sys

from nanochat.common import get_base_dir
from nanochat.scratch import ByteTokenizer, Engine, list_steps, load_model


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


def build_prompt(tokenizer, user_text):
    """Render a user turn and prime the assistant, matching chat_sft's layout."""
    from scripts.chat_sft import render_conversation
    ids, _ = render_conversation(tokenizer, [{"role": "user", "content": user_text}])
    return ids + tokenizer.encode("A:")


def respond(engine, tokenizer, user_text, args):
    ids = build_prompt(tokenizer, user_text)
    newline = tokenizer.encode("\n")[0]
    pieces, shown = [], ""
    for token, _ in engine.generate(ids, max_tokens=args.max_tokens,
                                    temperature=args.temperature, top_k=args.top_k,
                                    seed=args.seed, stop_tokens=[newline]):
        if token == newline:
            break
        pieces.append(token)
        # Re-decode the whole run each step and print only the newly resolved text.
        # A single byte can be half of a UTF-8 character, so decoding token-by-token
        # would emit replacement characters that later turn out to be wrong.
        text = tokenizer.decode(pieces)
        if text.endswith("\ufffd"):
            continue  # incomplete character, wait for the next byte
        sys.stdout.write(text[len(shown):])
        sys.stdout.flush()
        shown = text
    return shown


def main():
    args = parse_args()
    checkpoints = os.path.join(get_base_dir(), "checkpoints", args.run)
    if not list_steps(checkpoints):
        raise SystemExit(f"no checkpoints in {checkpoints}; run scripts.chat_sft first")

    model, meta = load_model(checkpoints, args.step)
    model.eval()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    if args.prompt is not None:
        text = respond(engine, tokenizer, args.prompt, args)
        print()
        return

    print(f"chat_cli | run={args.run} step={meta['step']} | {model.num_parameters():,} params")
    print(f"temperature={args.temperature}  (this model was finetuned on 'a+b' arithmetic)")
    print("Ctrl-C or empty line to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        print("Bot: ", end="")
        respond(engine, tokenizer, user, args)
        print()


if __name__ == "__main__":
    main()
