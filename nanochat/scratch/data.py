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

from nanochat.tokenizer import SPECIAL_TOKENS

__all__ = [
    "ByteTokenizer", "LegacyByteTokenizer", "make_addition_corpus", "addition_entropy_floor",
    "addition_pairs", "corpus_spec", "build_corpus", "Dataset",
]

ADDITION_LINE_LEN = 7  # e.g. "7+5=12;"
N_BYTES = 256


class ByteTokenizer:
    """UTF-8 bytes as tokens, plus reserved special tokens. No training required.

    ids 0..255 are the raw bytes; ids 256..264 are `SPECIAL_TOKENS` (BOS, the chat turn
    markers, the tool-call markers), in that order -- the same names the BPE tokenizer
    uses, so the chat format and the engine treat both tokenizers identically.

    **Structure cannot be written as text.** `encode` maps a string to its UTF-8 bytes
    and nothing else, so every id it can produce is < 256: no input -- including the
    literal characters "<|assistant_end|>" -- encodes to a special id. Special ids only
    enter a sequence when code asks for one by name (`encode_special`, `prepend=`,
    the chat renderer). That is the property that makes a user message unable to
    forge a turn boundary. `decode` renders a special id as its name, like tiktoken,
    which is for debugging only; the chat layer never turns special ids into text.

    This is the degenerate case of the BPE tokenizer in `nanochat/bpe.py` (no merges),
    and implements the slice of the `RustBPETokenizer` surface the rest of the code
    calls, so everything runs without training a tokenizer first.
    """

    vocab_size = N_BYTES + len(SPECIAL_TOKENS)
    BOS_ID = N_BYTES + SPECIAL_TOKENS.index("<|bos|>")

    def __init__(self):
        self._special_ids = {name: N_BYTES + i for i, name in enumerate(SPECIAL_TOKENS)}
        self._special_names = {i: name for name, i in self._special_ids.items()}

    def encode(self, text, prepend=None, append=None):
        if isinstance(text, list):
            return [self.encode(t, prepend=prepend, append=append) for t in text]
        ids = list(text.encode("utf-8"))
        if prepend is not None:
            ids = [prepend if isinstance(prepend, int) else self.encode_special(prepend)] + ids
        if append is not None:
            ids = ids + [append if isinstance(append, int) else self.encode_special(append)]
        return ids

    def __call__(self, texts, **kwargs):
        return [self.encode(t, **kwargs) for t in texts]

    def decode_single_token_bytes(self, token_id):
        token_id = int(token_id)
        if 0 <= token_id < N_BYTES:
            return bytes([token_id])
        name = self._special_names.get(token_id)
        if name is None:
            raise KeyError(f"unknown token id: {token_id}")
        return name.encode("utf-8")

    def decode(self, tokens):
        tokens = [int(t) for t in tokens]
        if all(0 <= t < N_BYTES for t in tokens):   # the common case, no specials
            return bytes(tokens).decode("utf-8", errors="replace")
        return b"".join(map(self.decode_single_token_bytes, tokens)).decode("utf-8", errors="replace")

    def encode_special(self, name):
        try:
            return self._special_ids[name]
        except KeyError:
            raise KeyError(f"Unknown special token: {name!r}") from None

    def get_bos_token_id(self):
        return self.BOS_ID

    def get_special_tokens(self):
        return set(SPECIAL_TOKENS)

    def get_vocab_size(self):
        return self.vocab_size


class LegacyByteTokenizer:
    """The original 256-id byte tokenizer, kept only to load checkpoints trained with it.

    With no spare ids it has no special tokens, so BOS is the byte `;` (the addition
    record terminator) and chat turns can only be written as text (`U:` / `A:` /
    newline). Both are forgeable by input text -- a `;` in a document *is* a document
    boundary, a "\\nA:" in a user message *is* a fake assistant turn. That is why it was
    replaced; see `ByteTokenizer`. Checkpoints record which one they were trained with
    (`nanochat.tokenizer.tokenizer_spec`), so old runs still load and decode correctly.
    """

    vocab_size = N_BYTES
    BOS_ID = ord(";")

    def encode(self, text, prepend=None, append=None):
        if isinstance(text, list):
            return [self.encode(t, prepend=prepend, append=append) for t in text]
        ids = list(text.encode("utf-8"))
        if prepend is not None:
            ids = [self.encode_special(prepend)] + ids
        if append is not None:
            ids = ids + [self.encode_special(append)]
        return ids

    def __call__(self, texts, **kwargs):
        return [self.encode(t, **kwargs) for t in texts]

    def decode(self, tokens):
        return bytes(int(t) for t in tokens).decode("utf-8", errors="replace")

    def decode_single_token_bytes(self, token_id):
        return bytes([int(token_id)])

    def encode_special(self, name):
        if name != "<|bos|>":
            raise KeyError(f"Unknown special token: {name!r}")
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


def has_dedicated_bos(tokenizer):
    """True if BOS is a special token rather than an ordinary byte (LegacyByteTokenizer)."""
    return "<|bos|>" in tokenizer.get_special_tokens()


def encode_corpus(text, tokenizer, spec):
    """Corpus text -> token ids, with BOS placed where the model will later see it.

    Generation and evaluation prompts start with BOS, so the model must see BOS in
    training at the same kind of position, or every prompt starts out of distribution.
      - A tokenizer with a dedicated BOS (ByteTokenizer, BPE): prepend it to every
        document. For the addition corpus a document is one `a+b=cc;` record; a text
        file is one document. BOS is special, so it counts zero bytes towards bpb and
        the bits-per-byte floor is unchanged.
      - LegacyByteTokenizer: BOS is the byte `;`, the addition record terminator, so
        the raw text already has it in exactly the right places. Encode as is.
    """
    if not has_dedicated_bos(tokenizer):
        return tokenizer.encode(text)
    bos = tokenizer.get_bos_token_id()
    if spec.get("kind") != "addition":
        return [bos] + tokenizer.encode(text)
    cache, ids = {}, []
    for i in range(0, len(text), ADDITION_LINE_LEN):  # only 100 distinct records
        record = text[i:i + ADDITION_LINE_LEN]
        if record not in cache:
            cache[record] = [bos] + tokenizer.encode(record)
        ids.extend(cache[record])
    return ids


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
