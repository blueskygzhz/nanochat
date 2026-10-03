"""
Data plumbing for the from-scratch trainer: a byte-level tokenizer and a batch sampler.

The default corpus is two-digit addition (`"7+5=12;"`). That choice is on purpose: its
entropy is known exactly, so training has a *checkable* target rather than just a
loss curve that goes down. Every character of a line is a deterministic function of
the line's two operands, so the only information in the stream is those operands:

    H(line) = 2 * ln(10) = 4.6052 nats  over 7 characters
    => floor = 4.6052 / 7 = 0.6579 nats/token

A model that reaches ~0.66 has learned the table. One that sits at ~2.3 has only
learned the character frequencies.

**Memorisation vs generalisation.** There are only 100 distinct problems, so a
model can reach the floor by memorising all of them, and any evaluation drawn from
the same 100 problems cannot tell the difference. `addition_pairs` therefore holds
out a fraction of the `(a, b)` pairs: they never appear in training, and the
evaluation scripts report accuracy on seen and held-out pairs separately. Only the
held-out number says anything about whether the model learned to add.
"""

import os

import numpy as np

__all__ = [
    "ByteTokenizer", "make_addition_corpus", "addition_entropy_floor", "addition_pairs",
    "corpus_spec", "build_corpus", "Dataset",
]

ADDITION_LINE_LEN = 7  # e.g. "7+5=12;"


class ByteTokenizer:
    """UTF-8 bytes as tokens. Vocabulary is exactly 256, no training required.

    This is the degenerate case of the BPE tokenizer in `nanochat/bpe.py`: no merges,
    one token per byte. It keeps this package free of any tokenizer dependency.

    It also implements the small slice of the `RustBPETokenizer` surface that
    `nanochat.scratch.eval` and `nanochat.scratch.engine` call, so they can be
    exercised without training a real tokenizer first.

    **On BOS.** A 256-entry byte vocabulary has no spare id for a real special token,
    so BOS has to be an actual byte. The choice matters more than it looks: prepending
    a byte the model never saw during training puts it immediately out of distribution
    and measurably degrades generation (on the addition task, greedy accuracy drops
    from 99/100 to 83/100 with a NUL byte as BOS). So BOS is `;`, which terminates
    every record in the addition corpus and is therefore exactly the context the model
    sees before a fresh problem mid-stream. `get_special_tokens()` is empty, so `;`
    still counts toward bits-per-byte like any other byte.
    """

    vocab_size = 256
    BOS_ID = ord(";")

    def encode(self, text, prepend=None, append=None):
        ids = list(text.encode("utf-8"))
        if prepend is not None:
            ids = [self.encode_special(prepend)] + ids
        if append is not None:
            ids = ids + [self.encode_special(append)]
        return ids

    def __call__(self, texts, **kwargs):
        return [self.encode(t, **kwargs) for t in texts]

    def decode(self, tokens):
        return bytes(int(t) & 0xFF for t in tokens).decode("utf-8", errors="replace")

    def decode_single_token_bytes(self, token_id):
        return bytes([int(token_id) & 0xFF])

    def encode_special(self, name):
        if name != "<|bos|>":
            raise KeyError(f"Unknown special token: {name}")
        return self.BOS_ID

    def get_bos_token_id(self):
        return self.BOS_ID

    def get_special_tokens(self):
        return set()

    def get_vocab_size(self):
        return self.vocab_size


def addition_pairs(holdout_frac=0.2, seed=0):
    """Split the 100 `(a, b)` operand pairs into `(train_pairs, heldout_pairs)`.

    Deterministic in `seed`, so every script that knows the split parameters (they
    are recorded in the checkpoint's meta) reconstructs exactly the same split.
    """
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError("holdout_frac must be in [0, 1)")
    pairs = [(a, b) for a in range(10) for b in range(10)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    n_held = int(round(len(pairs) * holdout_frac))
    held = sorted(pairs[i] for i in order[:n_held])
    train = sorted(pairs[i] for i in order[n_held:])
    return train, held


def make_addition_corpus(n_lines=20000, seed=0, pairs=None):
    """`a+b=cc;` lines with the sum zero-padded to 2 digits.

    With `pairs=None` the operands are uniform single digits. Otherwise each line is
    drawn uniformly from `pairs`, which is how held-out pairs are kept out of training.
    """
    rng = np.random.default_rng(seed)
    if pairs is None:
        a = rng.integers(0, 10, n_lines)
        b = rng.integers(0, 10, n_lines)
    else:
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.ndim != 2 or len(pairs) == 0:
            raise ValueError("pairs must be a non-empty list of (a, b)")
        chosen = pairs[rng.integers(0, len(pairs), n_lines)]
        a, b = chosen[:, 0], chosen[:, 1]
    return "".join(f"{x}+{y}={x + y:02d};" for x, y in zip(a, b))


def addition_entropy_floor(n_pairs=100):
    """Cross-entropy a perfect model would reach, in nats/token.

    The only information in a line is which operand pair it is, so the floor is
    ln(n_pairs) spread over the line's characters. 100 pairs gives 2*ln(10)/7.
    """
    return float(np.log(n_pairs)) / ADDITION_LINE_LEN


# ----------------------------------------------------------------------------
# corpus provenance
#
# A checkpoint is only interpretable together with the data it was trained on. The
# training script records a small JSON-able spec in the checkpoint's meta, and every
# downstream script rebuilds the corpus (and the held-out split) from that spec rather
# than from its own command-line defaults.

def corpus_spec(text_file=None, corpus_lines=20000, seed=0, holdout_frac=0.2):
    if text_file:
        return {"kind": "text", "path": os.path.abspath(text_file)}
    return {"kind": "addition", "corpus_lines": int(corpus_lines), "seed": int(seed),
            "holdout_frac": float(holdout_frac)}


def build_corpus(spec):
    """Rebuild a corpus from its spec.

    Returns `(text, info)`. For the addition task `info` carries the entropy floor
    and the seen/held-out pair split; for a text file both are None.
    """
    kind = spec.get("kind")
    if kind == "text":
        with open(spec["path"], encoding="utf-8") as f:
            return f.read(), {"floor": None, "train_pairs": None, "heldout_pairs": None}
    if kind == "addition":
        train, held = addition_pairs(spec["holdout_frac"], spec["seed"])
        text = make_addition_corpus(spec["corpus_lines"], seed=spec["seed"], pairs=train)
        return text, {"floor": addition_entropy_floor(len(train)),
                      "train_pairs": train, "heldout_pairs": held}
    raise ValueError(f"unknown corpus kind: {kind!r}")


class Dataset:
    """A flat token array plus a random-window batch sampler.

    `get_batch` returns `(x, y)` where `y` is `x` shifted left by one, which is the
    standard next-token objective: position t predicts token t+1.
    """

    def __init__(self, tokens, split=0.9):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        n = int(len(self.tokens) * split)
        self.train = self.tokens[:n]
        self.val = self.tokens[n:]

    @classmethod
    def from_text(cls, text, tokenizer=None, split=0.9):
        tokenizer = tokenizer or ByteTokenizer()
        return cls(tokenizer.encode(text), split=split)

    def get_batch(self, batch_size, seq_len, rng, split="train"):
        data = self.train if split == "train" else self.val
        if len(data) < seq_len + 1:
            raise ValueError(f"{split} split has {len(data)} tokens, need > {seq_len}")
        starts = rng.integers(0, len(data) - seq_len - 1, batch_size)
        x = np.stack([data[s:s + seq_len] for s in starts])
        y = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
        return x, y

    def batches(self, batch_size, seq_len, split="train", seed=0):
        """Endless random-window batch iterator. This is the shape `evaluate_bpb` wants."""
        rng = np.random.default_rng(seed)
        while True:
            yield self.get_batch(batch_size, seq_len, rng, split)

    def sequential_batches(self, batch_size, seq_len, split="val"):
        """Non-overlapping windows in order, stopping at the end of the split.

        Random windows are fine for a training signal but make evaluation noisy and
        double-count tokens. For a reproducible held-out number, walk the split once.
        """
        data = self.train if split == "train" else self.val
        stride = batch_size * seq_len
        usable = (len(data) - 1) // stride * stride
        for start in range(0, usable, stride):
            chunk = data[start:start + stride + 1]
            x = np.stack([chunk[i * seq_len:(i + 1) * seq_len] for i in range(batch_size)])
            y = np.stack([chunk[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(batch_size)])
            yield x, y

    def num_sequential_batches(self, batch_size, seq_len, split="val"):
        data = self.train if split == "train" else self.val
        stride = batch_size * seq_len
        return (len(data) - 1) // stride
