"""
Train a BPE tokenizer (GPT-4 style split pattern, our own BPE implementation).

    python -m scripts.tok_train --text-file book.txt --vocab-size 512    # local, offline
    python -m scripts.tok_train --vocab-size 32768                       # ClimbMix parquets

The tokenizer is written to `$NANOCHAT_BASE_DIR/tokenizer` (or `--out-dir`), together
with `token_bytes.npy`, the per-token byte counts that bits-per-byte needs. Then:

    python -m scripts.base_train --tokenizer bpe --text-file book.txt
"""
import argparse
import os
import time

import numpy as np

from nanochat.common import get_base_dir
from nanochat.scratch.eval import token_bytes_table
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--text-file', type=str, default=None,
                   help='train on this UTF-8 file instead of the downloaded parquet shards')
    p.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
    p.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
    p.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size including special tokens (default: 32768 = 2^15)')
    p.add_argument('--out-dir', type=str, default=None, help='default: $NANOCHAT_BASE_DIR/tokenizer')
    return p.parse_args()


def text_iterator(args):
    """Documents, each cropped to --doc-cap characters, stopping after --max-chars.

    A text file is split into paragraphs so --doc-cap applies per paragraph, as it
    does per document for the parquet shards.
    """
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            docs = [d for d in f.read().split("\n\n") if d]
        batches = [docs]
    else:
        from nanochat.dataset import parquets_iter_batched
        batches = parquets_iter_batched(split="train")
    nchars = 0
    for batch in batches:
        for doc in batch:
            doc = doc[:args.doc_cap]
            nchars += len(doc)
            yield doc
            if nchars > args.max_chars:
                return


def main():
    args = parse_args()
    floor = 256 + len(SPECIAL_TOKENS)
    if args.vocab_size < floor:
        raise SystemExit(f"--vocab-size must be at least {floor} "
                         f"(256 byte tokens + {len(SPECIAL_TOKENS)} special tokens)")
    print(f"source: {args.text_file or 'parquet shards'} | vocab_size: {args.vocab_size:,} "
          f"| doc_cap: {args.doc_cap:,} | max_chars: {args.max_chars:,}")

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(args), args.vocab_size)
    print(f"Training time: {time.time() - t0:.2f}s")
    if tokenizer.get_vocab_size() < args.vocab_size:
        print(f"note: the corpus ran out of pairs to merge; vocab is "
              f"{tokenizer.get_vocab_size()}, not {args.vocab_size}")

    tokenizer_dir = args.out_dir or os.path.join(get_base_dir(), "tokenizer")
    tokenizer.save(tokenizer_dir)

    # Round-trip sanity check, including bytes that are not valid standalone UTF-8
    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
    assert tokenizer.decode(tokenizer.encode(test_text)) == test_text

    # Per-token byte counts, for a bits-per-byte metric that is invariant to vocab
    # size. Special tokens count 0. Same function the evaluation code uses, so the
    # cached file can never disagree with what eval would compute.
    token_bytes = token_bytes_table(tokenizer).astype(np.int32)
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.npy")
    np.save(token_bytes_path, token_bytes)
    print(f"Saved token_bytes to {token_bytes_path}")

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            sample = f.read()
        n_bytes = len(sample.encode("utf-8"))
        n_tokens = len(tokenizer.encode(sample))
        print(f"compression on the training file: {n_bytes / max(n_tokens, 1):.2f} bytes/token")


if __name__ == "__main__":
    main()
