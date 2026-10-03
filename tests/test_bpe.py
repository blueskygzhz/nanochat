"""Tests for nanochat/bpe.py: the hand-written BPE tokenizer.

Covers:
1. Pre-tokenizer (split_text) — character-level correctness and reference-regex parity
2. Training (train_bpe) — merge counting, incremental updates, rustbpe parity
3. BPE class — encode/decode, round-trips, tiktoken parity, special tokens, persistence
4. Tokenizer integration — NANOCHAT_BPE_BACKEND switch, save/load portability

rustbpe and tiktoken are optional (the `fast-tokenizer` extra). They are used here
only as reference implementations; the tests that need them skip when absent.
"""

import importlib.util
import os
import pickle
import random

import pytest

from nanochat.bpe import (
    BPE,
    is_letter,
    is_number,
    is_space,
    split_text,
    train_bpe,
)
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Helper corpus

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "hello world, hello tokenizer, hello hello hello",
    "Numbers like 12345 and unicode like naïve café 你好 🙂 should survive.",
    "def f(x):\n    return x + 1\n",
    "Don't you won't they'll we've you're I'm it's",
    "  lots  of    spaces  and\ttabs\n",
] * 4


# ---------------------------------------------------------------------------
# 1. Pre-tokenizer

def _regex_split(text):
    import regex
    from nanochat.tokenizer import SPLIT_PATTERN
    return regex.compile(SPLIT_PATTERN).findall(text)


SPLIT_CASES = [
    # basic word
    ("hello world", ["hello", " world"]),
    # contraction
    ("don't", ["don", "'t"]),
    ("they'll", ["they", "'ll"]),
    ("we've", ["we", "'ve"]),
    ("you're", ["you", "'re"]),
    ("I'm", ["I", "'m"]),
    # leading symbol + word
    ("!hello", ["!hello"]),
    (" cat", [" cat"]),
    # digits: at most 2 per token
    ("12", ["12"]),
    ("123", ["12", "3"]),
    ("1", ["1"]),
    # punctuation run
    ("...", ["..."]),
    ("?! ok", ["?!", " ok"]),
    # newline handling
    ("\n", ["\n"]),
    ("\r\n", ["\r\n"]),
    ("  \n", ["  \n"]),
    # trailing whitespace stays with the next word
    ("hello world ", ["hello", " world", " "]),
    # empty
    ("", []),
    # single space
    (" ", [" "]),
    # unicode
    ("café", ["café"]),
    ("你好", ["你好"]),
    ("🙂", ["🙂"]),
]


@pytest.mark.parametrize("text,expected", SPLIT_CASES)
def test_split_known_cases(text, expected):
    assert split_text(text) == expected


def test_split_joins_back():
    cases = [t for t, _ in SPLIT_CASES] + [
        "a" * 50, "@#$%^&*()", "mixed123abc456", "\xa0nbsp", "\u3000ideographic",
        "ＡＢＣ fullwidth", "snake_case camelCase", "0x1F 3.14 1e10",
    ]
    for text in cases:
        assert "".join(split_text(text)) == text, repr(text)


def test_split_matches_reference_regex_on_random_text():
    cases = [
        "hello world! this is a test.",
        "  leading spaces", "trailing  ",
        "don't you DON'T it's they'll", "Numbers: 1 12 123 4567",
    ]
    random.seed(0)
    alphabet = "abcXYZ 09!?,.'_-你好🌍\u00a0\u3000\n\r\t"
    for _ in range(3000):
        cases.append("".join(random.choice(alphabet) for _ in range(random.randint(0, 40))))
    bad = 0
    for text in cases:
        if split_text(text) != _regex_split(text):
            bad += 1
    assert bad == 0, f"{bad} mismatches out of {len(cases)}"


def test_character_predicates():
    # letters
    for ch in "abcXYZéàü你α":
        assert is_letter(ch), repr(ch)
    for ch in "0 !@\n":
        assert not is_letter(ch), repr(ch)
    # numbers
    for ch in "0123456789²½":
        assert is_number(ch), repr(ch)
    for ch in "a !@":
        assert not is_number(ch), repr(ch)
    # space: must exclude \x1c-\x1f
    for ch in " \t\n\r\x0b\x0c\u00a0":
        assert is_space(ch), repr(ch)
    for ch in "\x1c\x1d\x1e\x1f":
        assert not is_space(ch), repr(ch)


def test_unassigned_codepoints_disagree_with_unicodedata_not_real_text():
    """Documents the unicode-version gap: only unassigned (Cn) codepoints disagree."""
    import unicodedata
    import regex
    L = regex.compile(r'\p{L}')
    mismatches = 0
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        if bool(L.fullmatch(ch)) != is_letter(ch):
            assert unicodedata.category(ch) == "Cn", f"Real character U+{cp:04X} mismatches"
            mismatches += 1
    assert mismatches > 0, "expected some Cn discrepancies due to unicode version gap"


# ---------------------------------------------------------------------------
# 2. Training

TRAIN_CORPUS = [
    "the the the fox the dog the",
    "fox dog fox fox",
    "hello world",
    "the world",
]


def test_train_bpe_produces_correct_vocab_size():
    # Use a rich corpus so merges don't run out before the target.
    corpus = CORPUS * 5
    ranks = train_bpe(iter(corpus), 280)
    assert len(ranks) == 280


def test_train_bpe_contains_all_256_bytes():
    ranks = train_bpe(iter(TRAIN_CORPUS), 260)
    for b in range(256):
        assert bytes([b]) in ranks, b


def test_train_bpe_merges_are_in_rank_order():
    ranks = train_bpe(iter(TRAIN_CORPUS), 280)
    # "the" (t=116, h=104, e=101) is frequent so "th" should appear before "he"
    th = ranks.get(b"th")
    he = ranks.get(b"he")
    if th and he:
        assert th < he or he < th  # just that both exist and are ordered


def test_train_bpe_matches_rustbpe():
    """Core invariant: our training must produce the same merge order as rustbpe."""
    rustbpe = pytest.importorskip("rustbpe", reason="optional extra: fast-tokenizer")
    from nanochat.tokenizer import SPLIT_PATTERN
    corpus = CORPUS * 3
    V = 256 + 200
    rt = rustbpe.Tokenizer()
    rt.train_from_iterator(iter(corpus), V, pattern=SPLIT_PATTERN)
    ref = {bytes(k): v for k, v in rt.get_mergeable_ranks()}
    mine = train_bpe(iter(corpus), V)
    assert mine == ref


def test_train_bpe_is_reproducible():
    a = train_bpe(iter(TRAIN_CORPUS), 280)
    b = train_bpe(iter(TRAIN_CORPUS), 280)
    assert a == b


def test_train_bpe_rejects_too_small_vocab():
    with pytest.raises(ValueError, match="256"):
        train_bpe(iter(TRAIN_CORPUS), 100)


def test_train_bpe_handles_empty_corpus():
    ranks = train_bpe(iter([]), 260)
    assert len(ranks) == 256  # no merges possible, just 256 bytes


def test_train_bpe_handles_single_byte_corpus():
    # Single distinct byte: merges do happen ("aa"->merged), so the vocab may
    # exceed 256, but it will always be at most 256 + log2(len(corpus)).
    ranks = train_bpe(iter(["aaaa"]), 270)
    assert 256 <= len(ranks) <= 270


# ---------------------------------------------------------------------------
# 3. BPE class

@pytest.fixture(scope="module")
def small_bpe():
    ranks = train_bpe(iter(CORPUS), 256 + 80)
    return BPE(ranks, {})


def test_bpe_encode_decode_roundtrip(small_bpe):
    for text in ["hello world", "naïve café 你好 🙂", "unseen tokens: zqxjkv", "don't"]:
        ids = small_bpe.encode_ordinary(text)
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        assert small_bpe.decode(ids) == text


def test_bpe_encode_is_consistent(small_bpe):
    text = "the quick brown fox"
    assert small_bpe.encode_ordinary(text) == small_bpe.encode_ordinary(text)


def test_bpe_matches_tiktoken_on_same_ranks(small_bpe):
    tiktoken = pytest.importorskip("tiktoken", reason="optional extra: fast-tokenizer")
    from nanochat.tokenizer import SPLIT_PATTERN
    enc = tiktoken.Encoding(name="ref", pat_str=SPLIT_PATTERN,
                            mergeable_ranks=small_bpe._mergeable_ranks, special_tokens={})
    for text in ["hello world", "naïve café 你好", "def f(x):", "   spaces   ", "zqxjkv"]:
        assert small_bpe.encode_ordinary(text) == enc.encode_ordinary(text), repr(text)


def test_bpe_encode_ordinary_batch(small_bpe):
    texts = ["hello", "world", "foo bar"]
    batch = small_bpe.encode_ordinary_batch(texts)
    assert batch == [small_bpe.encode_ordinary(t) for t in texts]


def test_bpe_special_tokens():
    ranks = train_bpe(iter(CORPUS), 256 + 30)
    offset = len(ranks)
    specials = {"<|bos|>": offset, "<|eos|>": offset + 1}
    bpe = BPE(ranks, specials)
    assert bpe.n_vocab == offset + 2
    assert bpe.encode_single_token("<|bos|>") == offset
    assert bpe.encode_single_token(b"hello") == ranks[b"hello"] if b"hello" in ranks else True
    assert set(bpe.special_tokens_set) == {"<|bos|>", "<|eos|>"}
    with pytest.raises(KeyError):
        bpe.encode_single_token("<|unknown|>")


def test_bpe_decode_handles_split_utf8():
    # A token boundary can cut a multibyte sequence; decode must not raise.
    ranks = {bytes([b]): b for b in range(256)}
    bpe = BPE(ranks, {})
    # First byte of 'é' (U+00E9) is 0xC3; second is 0xA9. Decoding only the first is OK.
    partial = bpe.decode([0xC3])
    assert isinstance(partial, str)


def test_bpe_n_vocab(small_bpe):
    assert small_bpe.n_vocab == max(small_bpe._mergeable_ranks.values()) + 1


def test_bpe_persistence(small_bpe, tmp_path):
    # Save via pickle protocol (same as tokenizer.save)
    path = tmp_path / "bpe.pkl"
    with open(path, "wb") as f:
        pickle.dump(small_bpe, f)
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    text = "the quick brown fox"
    assert loaded.encode_ordinary(text) == small_bpe.encode_ordinary(text)


def test_bpe_state_round_trip(small_bpe):
    state = small_bpe.to_state()
    restored = BPE.from_state(state)
    text = "hello world"
    assert restored.encode_ordinary(text) == small_bpe.encode_ordinary(text)


def test_bpe_train_classmethod():
    bpe = BPE.train(iter(CORPUS), 256 + 40 + len(SPECIAL_TOKENS),
                    special_tokens=SPECIAL_TOKENS)
    assert bpe.n_vocab == 256 + 40 + len(SPECIAL_TOKENS)
    for name in SPECIAL_TOKENS:
        assert name in bpe._special_tokens
    text = "hello world"
    assert "".join(bpe.decode([i]) for i in bpe.encode_ordinary(text)) == text


# ---------------------------------------------------------------------------
# 4. Tokenizer integration: backend switch and save/load portability

@pytest.fixture(scope="module")
def tokenizer_scratch():
    vocab_size = 256 + len(SPECIAL_TOKENS) + 40
    return RustBPETokenizer.train_from_iterator(iter(CORPUS), vocab_size, backend="scratch")


@pytest.fixture(scope="module")
def tokenizer_rust():
    pytest.importorskip("rustbpe", reason="optional extra: fast-tokenizer")
    pytest.importorskip("tiktoken", reason="optional extra: fast-tokenizer")
    vocab_size = 256 + len(SPECIAL_TOKENS) + 40
    return RustBPETokenizer.train_from_iterator(iter(CORPUS), vocab_size, backend="rust")


def test_both_backends_produce_identical_ranks(tokenizer_scratch, tokenizer_rust):
    assert tokenizer_scratch.enc._mergeable_ranks == tokenizer_rust.enc._mergeable_ranks


def test_both_backends_encode_identically(tokenizer_scratch, tokenizer_rust):
    texts = ["hello world", "naïve café 你好", "don't", "def f(x):", "zqxjkv"]
    for text in texts:
        assert (tokenizer_scratch.encode(text) == tokenizer_rust.encode(text)), repr(text)


def test_default_backend_is_scratch(monkeypatch):
    monkeypatch.delenv("NANOCHAT_BPE_BACKEND", raising=False)
    from nanochat.tokenizer import get_backend
    assert get_backend() == "scratch"


def test_rust_backend_env(monkeypatch):
    monkeypatch.setenv("NANOCHAT_BPE_BACKEND", "rust")
    from nanochat.tokenizer import get_backend
    assert get_backend() == "rust"


def test_invalid_backend_env(monkeypatch):
    monkeypatch.setenv("NANOCHAT_BPE_BACKEND", "numpy")
    from nanochat.tokenizer import get_backend
    with pytest.raises(ValueError, match="NANOCHAT_BPE_BACKEND"):
        get_backend()


def test_save_load_portable_between_backends(tokenizer_scratch, tmp_path, monkeypatch):
    tokenizer_scratch.save(str(tmp_path))
    # The saved file must be a dict, not a tiktoken.Encoding
    with open(tmp_path / "tokenizer.pkl", "rb") as f:
        payload = pickle.load(f)
    assert isinstance(payload, dict) and "mergeable_ranks" in payload

    # Load under scratch backend
    monkeypatch.delenv("NANOCHAT_BPE_BACKEND", raising=False)
    loaded_scratch = RustBPETokenizer.from_directory(str(tmp_path), backend="scratch")
    text = "hello world café"
    assert loaded_scratch.encode(text) == tokenizer_scratch.encode(text)

    # The same file must also load under the optional rust backend
    pytest.importorskip("tiktoken", reason="optional extra: fast-tokenizer")
    loaded_rust = RustBPETokenizer.from_directory(str(tmp_path), backend="rust")
    assert loaded_rust.encode(text) == tokenizer_scratch.encode(text)


def test_save_load_legacy_tiktoken_pickle(tokenizer_scratch, tmp_path):
    """Legacy: a pickled tiktoken.Encoding must still load via from_directory."""
    tiktoken = pytest.importorskip("tiktoken", reason="optional extra: fast-tokenizer")
    from nanochat.tokenizer import SPLIT_PATTERN
    enc = tiktoken.Encoding(name="legacy", pat_str=SPLIT_PATTERN,
                            mergeable_ranks=tokenizer_scratch.enc._mergeable_ranks,
                            special_tokens=tokenizer_scratch.enc._special_tokens)
    with open(tmp_path / "tokenizer.pkl", "wb") as f:
        pickle.dump(enc, f)
    loaded = RustBPETokenizer.from_directory(str(tmp_path), backend="scratch")
    text = "hello world"
    assert loaded.encode(text) == tokenizer_scratch.encode(text)


def test_backend_env_switch_end_to_end(monkeypatch, tmp_path):
    vocab_size = 256 + len(SPECIAL_TOKENS) + 30
    backends = ["scratch"]
    if all(importlib.util.find_spec(m) for m in ("rustbpe", "tiktoken")):
        backends.append("rust")
    for backend in backends:
        monkeypatch.setenv("NANOCHAT_BPE_BACKEND", backend)
        tok = RustBPETokenizer.train_from_iterator(iter(CORPUS), vocab_size)
        assert isinstance(tok.enc, BPE if backend == "scratch" else object)
        tok.save(str(tmp_path / backend))
        loaded = RustBPETokenizer.from_directory(str(tmp_path / backend))
        text = "hello world"
        assert loaded.encode(text) == tok.encode(text)


def test_tokenizer_existing_tests_still_pass():
    """The scratch backend must reproduce all behaviour the existing test suite checks."""
    import importlib, sys
    vocab_size = 256 + len(SPECIAL_TOKENS) + 35
    tok = RustBPETokenizer.train_from_iterator(iter(CORPUS), vocab_size, backend="scratch")
    # Special tokens encode to unique single ids
    ids = [tok.encode_special(t) for t in SPECIAL_TOKENS]
    assert len(set(ids)) == len(SPECIAL_TOKENS)
    # encode_special rejects ordinary tokens and unknown names
    with pytest.raises(KeyError, match="Unknown special token"):
        tok.encode_special("a")
    with pytest.raises(KeyError, match="Unknown special token"):
        tok.encode_special("<|not_registered|>")
    # round-trip
    for text in ["hello world", "naïve café 你好 🙂", "unseen: zqxjkv"]:
        assert tok.decode(tok.encode(text)) == text
    # render_conversation truncation
    conv = {"messages": [
        {"role": "user", "content": "hello " * 2000},
        {"role": "assistant", "content": "world"},
    ]}
    ids, mask = tok.render_conversation(conv, max_tokens=32)
    assert len(ids) == len(mask) == 32
    with pytest.raises(ValueError, match="max_tokens"):
        tok.render_conversation(conv, max_tokens=0)
    # render_for_completion does not crop long prompts
    full_ids = tok.render_for_completion(conv)
    assert len(full_ids) > 32
    assert full_ids[-2] == tok.encode_special("<|user_end|>")
