"""Hand-written byte-level BPE: replaces both rustbpe (training) and tiktoken (inference).

Uses only the Python standard library. Nothing here calls into Rust, and the whole
algorithm is readable Python, which is the point: this is the part of the stack a
learner should be able to step through in a debugger.

Three pieces, in dependency order:

1. `split_text`  The GPT-4 style pre-tokenizer, hand-written as a scanner instead of a
                 regex. Text is first cut into chunks (roughly "words"), and merges are
                 never allowed to cross a chunk boundary. This is what stops the
                 tokenizer from learning a single token for " the end of the" and keeps
                 digits from gluing to letters.
2. `train_bpe`   Byte-pair encoding training: start from the 256 raw bytes, then
                 repeatedly merge the most frequent adjacent pair. The merge order *is*
                 the vocabulary; merge i gets token id 256+i.
3. `BPE`         Encode/decode. Encoding re-applies the learned merges to new text, always
                 taking the lowest-rank (earliest-learned) pair first. That ordering is
                 what makes encoding deterministic and consistent with training.

Compatibility: `BPE` is a drop-in for the subset of `tiktoken.Encoding` this repo uses, so
it can be swapped in behind `RustBPETokenizer` without touching call sites.

On Unicode: the character predicates below come from `unicodedata`, which tracks the
Unicode version of the running CPython (15.0.0 here). The `regex` module bundles newer
tables, so the two disagree on ~22k code points that are unassigned in 15.0.0 but assigned
as letters later. Real text contains no unassigned code points, so splits agree on
everything you will actually tokenize; `tests/test_bpe.py` pins this explicitly.

Performance: on warm (cached) text this runs about 6x slower than tiktoken; on cold
diverse text (short random words) about 12x slower. The pure-Python training loop is
about 100x slower than rustbpe at scale. For learning and debugging these numbers are
fine; for tokenizing a 400B-token corpus during pretraining, set
NANOCHAT_BPE_BACKEND=rust.
"""

import unicodedata
from collections import Counter

# `\s` in the regex module does NOT include the four ASCII information separators, but
# Python's str.isspace() does. Matching the reference pre-tokenizer means excluding them.
_NOT_REALLY_SPACE = frozenset("\x1c\x1d\x1e\x1f")
_NEWLINES = frozenset("\r\n")
_CONTRACTION_1 = frozenset("sdmt")      # 's 'd 'm 't
_CONTRACTION_2 = frozenset(("ll", "ve", "re"))


def is_space(ch):
    """Equivalent to `\\s` in the reference pattern."""
    return ch.isspace() and ch not in _NOT_REALLY_SPACE


def is_letter(ch):
    """Equivalent to `\\p{L}`: any Unicode letter (Lu, Ll, Lt, Lm, Lo)."""
    return unicodedata.category(ch)[0] == "L"


def is_number(ch):
    """Equivalent to `\\p{N}`: any Unicode number (Nd, Nl, No)."""
    return unicodedata.category(ch)[0] == "N"


def _match_chunk(text, i):
    """Return the end index of the chunk starting at `i`.

    This hand-codes the alternatives of nanochat's split pattern, in order. A regex
    alternation is leftmost-*first* (not longest), so the order of these branches is part
    of the specification, not a detail. Two subtleties are called out inline: possessive
    quantifiers (which must not backtrack) and the trailing-whitespace lookahead.
    """
    n = len(text)
    ch = text[i]

    # 1) '(?i:[sdmt]|ll|ve|re)  -- contractions, so "don't" -> "don" + "'t"
    if ch == "'" and i + 1 < n:
        if text[i + 1].lower() in _CONTRACTION_1:
            return i + 2
        if i + 2 < n and text[i + 1 : i + 3].lower() in _CONTRACTION_2:
            return i + 3

    # 2) [^\r\n\p{L}\p{N}]?+\p{L}+  -- a word, optionally with one leading symbol (" cat").
    # The `?+` is POSSESSIVE: once that leading character is consumed it is never given
    # back, so if no letters follow, this whole alternative fails rather than retrying
    # with zero characters. Falling through (instead of looping) is what implements that.
    k = i
    if ch not in _NEWLINES and not is_letter(ch) and not is_number(ch):
        k += 1
    if k < n and is_letter(text[k]):
        while k < n and is_letter(text[k]):
            k += 1
        return k

    # 3) \p{N}{1,2}  -- at most two digits per token, so numbers stay granular
    if is_number(ch):
        return i + 2 if (i + 1 < n and is_number(text[i + 1])) else i + 1

    # 4) ' ?[^\s\p{L}\p{N}]++[\r\n]*  -- punctuation runs, plus any trailing newlines
    k = i + 1 if ch == " " else i
    if k < n and not is_space(text[k]) and not is_letter(text[k]) and not is_number(text[k]):
        while k < n and not is_space(text[k]) and not is_letter(text[k]) and not is_number(text[k]):
            k += 1
        while k < n and text[k] in _NEWLINES:
            k += 1
        return k

    # Everything below is whitespace handling. Find the extent of the run once.
    run_end = i
    while run_end < n and is_space(text[run_end]):
        run_end += 1

    # 5) \s*[\r\n]  -- greedy `\s*` then a required newline, which after backtracking means
    # "through the LAST newline in this run". Keeps blank lines as single tokens.
    last_newline = -1
    for m in range(i, run_end):
        if text[m] in _NEWLINES:
            last_newline = m
    if last_newline >= 0:
        return last_newline + 1

    # 6) \s+(?!\S)  -- whitespace NOT followed by a visible character. Greedy, so it first
    # tries the whole run; if a visible character follows, it backtracks one, leaving that
    # last space to be picked up by alternative 2 as part of the next word (" cat").
    if run_end > i:
        if run_end == n:
            return run_end          # run reaches end of string: take all of it
        if run_end - 1 > i:
            return run_end - 1      # hand the final space to the following word
        # A single space before a word: alternative 6 cannot match, so fall through to 7.

    # 7) \s+  -- any remaining whitespace
    return run_end if run_end > i else i + 1


def split_text(text):
    """Cut text into pre-token chunks. Merges never cross these boundaries."""
    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = _match_chunk(text, i)
        chunks.append(text[i:j])
        i = j
    return chunks


# -----------------------------------------------------------------------------
# Training


def _count_pairs(symbols, counts, into):
    """Accumulate weighted adjacent-pair counts for one word."""
    for pair in zip(symbols, symbols[1:]):
        into[pair] = into.get(pair, 0) + counts


def _merge_symbols(symbols, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `symbols` with `new_id`."""
    out = []
    i, n = 0, len(symbols)
    first, second = pair
    while i < n:
        if i + 1 < n and symbols[i] == first and symbols[i + 1] == second:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(text_iterator, vocab_size, verbose_every=0):
    """Learn `vocab_size - 256` merges and return {token_bytes: rank}.

    The algorithm: count how often each pre-token chunk appears, represent each unique
    chunk as a list of byte ids, then repeatedly merge the most frequent adjacent pair
    across the whole corpus. Ties go to the lexicographically smallest pair purely so
    runs are reproducible.

    Only unique chunks are stored (with multiplicities), which is why a corpus of
    millions of words collapses to a far smaller working set. Pair counts are updated
    incrementally: after a merge, only the words that actually contained that pair are
    revisited, instead of rescanning the corpus.
    """
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")

    # 1) Pre-tokenize and tally identical chunks.
    chunk_counts = Counter()
    for text in text_iterator:
        chunk_counts.update(split_text(text))

    # 2) Each unique chunk becomes a list of byte ids, carrying its own frequency.
    words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
    weights = list(chunk_counts.values())

    # 3) Global pair counts, plus which words contain each pair (so updates stay local).
    pair_counts = {}
    pair_words = {}
    for index, (symbols, weight) in enumerate(zip(words, weights)):
        local = {}
        _count_pairs(symbols, weight, local)
        for pair, count in local.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_words.setdefault(pair, set()).add(index)

    vocab = {bytes([b]): b for b in range(256)}
    merges = []
    num_merges = vocab_size - 256

    for step in range(num_merges):
        if not pair_counts:
            break  # corpus fully merged: every word is a single token
        # Most frequent pair; the pair itself breaks ties for determinism.
        best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
        if pair_counts[best] <= 0:
            break
        new_id = 256 + step
        merges.append(best)

        # Re-merge only the affected words, and diff their pair contributions.
        for index in list(pair_words.get(best, ())):
            symbols = words[index]
            weight = weights[index]
            before = {}
            _count_pairs(symbols, weight, before)
            merged = _merge_symbols(symbols, best, new_id)
            words[index] = merged
            after = {}
            _count_pairs(merged, weight, after)
            for pair in before.keys() | after.keys():
                delta = after.get(pair, 0) - before.get(pair, 0)
                if delta:
                    pair_counts[pair] = pair_counts.get(pair, 0) + delta
                if after.get(pair, 0):
                    pair_words.setdefault(pair, set()).add(index)
                elif pair in pair_words:
                    pair_words[pair].discard(index)
                if pair_counts.get(pair, 0) <= 0:
                    pair_counts.pop(pair, None)
                    pair_words.pop(pair, None)
        pair_counts.pop(best, None)
        pair_words.pop(best, None)

        if verbose_every and (step + 1) % verbose_every == 0:
            print(f"  merge {step + 1}/{num_merges}: {best} -> {new_id}")

    # 4) Merge list -> {bytes: rank}. A merge's bytes are its two parts concatenated.
    id_to_bytes = {b: bytes([b]) for b in range(256)}
    for step, (first, second) in enumerate(merges):
        new_id = 256 + step
        token = id_to_bytes[first] + id_to_bytes[second]
        id_to_bytes[new_id] = token
        vocab[token] = new_id
    return vocab


# -----------------------------------------------------------------------------
# Encoding / decoding


class BPE:
    """A trained BPE tokenizer. Drop-in for the `tiktoken.Encoding` API used in this repo.

    Attribute names with a leading underscore (`_mergeable_ranks`, `_special_tokens`)
    mirror tiktoken's, because surrounding code reads them.
    """

    def __init__(self, mergeable_ranks, special_tokens, name="handwritten"):
        self.name = name
        self._mergeable_ranks = dict(mergeable_ranks)
        self._special_tokens = dict(special_tokens)
        self._decoder = {rank: token for token, rank in self._mergeable_ranks.items()}
        if len(self._decoder) != len(self._mergeable_ranks):
            raise ValueError("mergeable_ranks maps two tokens to the same id")
        self._special_decoder = {i: name.encode("utf-8") for name, i in self._special_tokens.items()}
        self.max_token_value = max(
            max(self._mergeable_ranks.values(), default=-1),
            max(self._special_tokens.values(), default=-1),
        )
        self._cache = {}

    # ---- construction -------------------------------------------------------

    @classmethod
    def train(cls, text_iterator, vocab_size, special_tokens=(), verbose_every=0):
        """Train on text, then append special tokens after the learned vocabulary."""
        ordinary_size = vocab_size - len(special_tokens)
        if ordinary_size < 256:
            raise ValueError(
                f"need room for 256 byte tokens plus {len(special_tokens)} special tokens, "
                f"so vocab_size must be at least {256 + len(special_tokens)}, got {vocab_size}")
        ranks = train_bpe(text_iterator, ordinary_size, verbose_every=verbose_every)
        offset = len(ranks)
        specials = {name: offset + i for i, name in enumerate(special_tokens)}
        return cls(ranks, specials)

    # ---- tiktoken-compatible surface ---------------------------------------

    @property
    def n_vocab(self):
        return self.max_token_value + 1

    @property
    def special_tokens_set(self):
        return set(self._special_tokens)

    def encode_single_token(self, text):
        """Id of a single whole token, by its bytes or its special-token name."""
        if isinstance(text, str):
            if text in self._special_tokens:
                return self._special_tokens[text]
            text = text.encode("utf-8")
        if text in self._mergeable_ranks:
            return self._mergeable_ranks[text]
        raise KeyError(f"not a single token: {text!r}")

    def decode_single_token_bytes(self, token_id):
        if token_id in self._decoder:
            return self._decoder[token_id]
        if token_id in self._special_decoder:
            return self._special_decoder[token_id]
        raise KeyError(f"unknown token id: {token_id}")

    def _encode_chunk(self, piece):
        """BPE-merge the bytes of one pre-token chunk.

        Repeatedly merge the adjacent pair with the LOWEST rank, i.e. the merge that was
        learned earliest. Applying merges in learned order is what makes encoding agree
        with training; merging greedily by length instead would give different ids.
        """
        if len(piece) == 1:
            return [self._mergeable_ranks[piece]]
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        ranks = self._mergeable_ranks
        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            best_rank, best_index = None, -1
            for index in range(len(parts) - 1):
                rank = ranks.get(parts[index] + parts[index + 1])
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, index
            if best_index < 0:
                break  # no learned merge applies to any remaining pair
            parts[best_index : best_index + 2] = [parts[best_index] + parts[best_index + 1]]

        ids = []
        for part in parts:
            rank = ranks.get(part)
            if rank is None:
                # Unreachable with a vocabulary that contains all 256 bytes, but a
                # truncated vocabulary should fail loudly rather than emit garbage.
                raise KeyError(f"no token covers the bytes {part!r}")
            ids.append(rank)
        if len(piece) <= 64:  # bound the cache: long chunks are rare and rarely repeat
            self._cache[piece] = ids
        return ids

    def encode_ordinary(self, text):
        """Encode text, treating special-token strings as ordinary text."""
        ids = []
        for chunk in split_text(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode_ordinary_batch(self, texts, num_threads=None):
        # num_threads is accepted for API compatibility. Python's GIL means threads do not
        # speed up this pure-Python loop, so it is deliberately ignored rather than faked.
        return [self.encode_ordinary(text) for text in texts]

    def decode_bytes(self, ids):
        out = bytearray()
        for token_id in ids:
            out += self.decode_single_token_bytes(token_id)
        return bytes(out)

    def decode(self, ids):
        # errors="replace" matches tiktoken: a token boundary can split a UTF-8 sequence,
        # so decoding an arbitrary id subset must not raise.
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ---- persistence --------------------------------------------------------

    def to_state(self):
        """A plain-dict form that pickles without depending on this class's layout."""
        return {
            "name": self.name,
            "mergeable_ranks": self._mergeable_ranks,
            "special_tokens": self._special_tokens,
        }

    @classmethod
    def from_state(cls, state):
        return cls(state["mergeable_ranks"], state["special_tokens"], name=state.get("name", "handwritten"))

    def __getstate__(self):
        return self.to_state()

    def __setstate__(self, state):
        self.__init__(state["mergeable_ranks"], state["special_tokens"],
                      name=state.get("name", "handwritten"))
