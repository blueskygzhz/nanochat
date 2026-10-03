"""
BPE Tokenizer in the style of GPT-4.

By default both training and inference use the hand-written BPE in `nanochat/bpe.py`
(pure Python, standard library only), so the whole tokenizer is steppable in a debugger.

The Rust stack (rustbpe for training, tiktoken for inference) stays available behind
`NANOCHAT_BPE_BACKEND=rust`, because pure Python encodes roughly 100x slower and
pretraining tokenizes on the fly. Both backends are verified to produce *identical*
merge ranks and identical token ids in `tests/test_bpe.py`, so the choice is purely
speed, never behaviour.
"""

import os
import copy
import pickle

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
# `nanochat/bpe.py` hand-codes this same pattern as a scanner; it only needs to be a regex
# string for the optional rustbpe/tiktoken backend.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Backend selection

BACKENDS = ("scratch", "rust")


def get_backend():
    """Which BPE implementation to use: 'scratch' (hand-written) or 'rust'."""
    backend = os.environ.get("NANOCHAT_BPE_BACKEND", "scratch").lower()
    if backend not in BACKENDS:
        raise ValueError(f"NANOCHAT_BPE_BACKEND must be one of {BACKENDS}, got {backend!r}")
    return backend


def _build_rust_encoding(mergeable_ranks, special_tokens):
    """Wrap ranks in a tiktoken Encoding (the fast inference backend)."""
    import tiktoken
    return tiktoken.Encoding(
        name="rustbpe",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
        special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
    )


def _build_encoding(mergeable_ranks, special_tokens, backend=None):
    backend = backend or get_backend()
    if backend == "rust":
        return _build_rust_encoding(mergeable_ranks, special_tokens)
    from nanochat.bpe import BPE
    return BPE(mergeable_ranks, special_tokens)


class RustBPETokenizer:
    """Tokenizer wrapper. Named for history; the default backend is the hand-written BPE.

    Holds an encoding object exposing a small tiktoken-compatible surface
    (`encode_ordinary`, `decode`, `_mergeable_ranks`, ...). `nanochat.bpe.BPE` implements
    that surface, so this class is backend-agnostic.
    """

    def __init__(self, enc, bos_token):
        self.enc = enc
        # Only registered special tokens may be looked up by name: tiktoken's
        # encode_single_token prefers an ordinary token with identical bytes.
        self._special_token_ids = dict(enc._special_tokens)
        ordinary = getattr(enc, "_mergeable_ranks", {})
        collisions = sorted(name for name in self._special_token_ids if name.encode("utf-8") in ordinary)
        if collisions:
            raise ValueError(f"Special tokens also exist as ordinary BPE tokens: {collisions}")
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size, backend=None):
        backend = backend or get_backend()
        # the special tokens are inserted after training, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        if backend == "rust":
            import rustbpe
            tokenizer = rustbpe.Tokenizer()
            tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
            mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
        else:
            from nanochat.bpe import train_bpe
            mergeable_ranks = train_bpe(text_iterator, vocab_size_no_special)
        collisions = [name for name in SPECIAL_TOKENS if name.encode("utf-8") in mergeable_ranks]
        if collisions:
            raise ValueError(f"Trained vocabulary contains special token byte strings: {collisions}")
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        return cls(_build_encoding(mergeable_ranks, special_tokens, backend), "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir, backend=None):
        """Load a saved tokenizer, accepting both the portable and the legacy format.

        Saved files are backend-independent: only the ranks and special tokens are stored,
        and the encoding object is rebuilt locally. Legacy checkpoints hold a pickled
        `tiktoken.Encoding`, which is still readable.
        """
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "mergeable_ranks" in payload:
            enc = _build_encoding(payload["mergeable_ranks"], payload["special_tokens"], backend)
        else:
            # Legacy: a pickled tiktoken.Encoding. Pull the ranks out and rebuild, so the
            # configured backend still applies.
            enc = _build_encoding(payload._mergeable_ranks, payload._special_tokens, backend)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        # Loading OpenAI's published vocabularies needs tiktoken regardless of backend:
        # the ranks ship inside that package. The hand-written BPE can still *use* them.
        import tiktoken
        published = tiktoken.get_encoding(tiktoken_name)
        enc = _build_encoding(published._mergeable_ranks, published._special_tokens)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    def encode_special(self, text):
        try:
            return self._special_token_ids[text]
        except KeyError:
            raise KeyError(f"Unknown special token: {text!r}") from None

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def decode_single_token_bytes(self, token_id):
        return self.enc.decode_single_token_bytes(token_id)

    def save(self, tokenizer_dir):
        # Save ranks + special tokens, not the encoding object. This keeps the file
        # portable between backends and independent of any library's pickle layout.
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        payload = {
            "mergeable_ranks": dict(self.enc._mergeable_ranks),
            "special_tokens": dict(self.enc._special_tokens),
        }
        with open(pickle_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.

        The layout lives in `nanochat/chat_format.py`, shared with the byte tokenizer
        and with inference, so training and inference cannot drift apart.
        """
        from nanochat.chat_format import render_conversation
        return render_conversation(self, conversation["messages"], max_tokens=max_tokens)

    def _render_conversation_upstream(self, conversation, max_tokens=2048):
        """Upstream's original renderer, kept verbatim as the reference that
        `tests/test_chat_format.py` checks `chat_format` against."""
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs); None keeps everything
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive or None")
            ids = ids[:max_tokens]
            mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Never silently crop a prompt: the caller's context limit applies to the full prompt.
        ids, mask = self.render_conversation(conversation, max_tokens=None)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)


# -----------------------------------------------------------------------------
# Tokenizer provenance
#
# A checkpoint's embedding table is meaningless without the exact vocabulary it was
# trained with, so the training scripts record a small JSON-able spec in meta.json and
# every downstream script loads the tokenizer from that spec. For BPE, the tokenizer
# files are *copied* into the run directory: re-running tok_train later must not
# silently change the vocabulary under an already-trained model.

TOKENIZER_KINDS = ("byte", "bpe")


def tokenizer_spec(kind="byte", tokenizer_dir=None):
    if kind not in TOKENIZER_KINDS:
        raise ValueError(f"tokenizer kind must be one of {TOKENIZER_KINDS}, got {kind!r}")
    if kind == "byte":
        # vocab_size distinguishes the byte tokenizer with special tokens (265) from the
        # legacy one (256), whose specs were written without a vocab_size
        from nanochat.scratch.data import ByteTokenizer
        return {"kind": "byte", "vocab_size": ByteTokenizer.vocab_size}
    if tokenizer_dir is None:
        from nanochat.common import get_base_dir
        tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    return {"kind": "bpe", "dir": os.path.abspath(tokenizer_dir)}


def load_tokenizer(spec=None):
    """Build the tokenizer a spec describes.

    `None`, or a byte spec without a vocab_size, means the legacy 256-id byte tokenizer:
    that is what every checkpoint written before these specs existed was trained with.
    """
    spec = spec or {"kind": "byte"}
    if spec["kind"] == "byte":
        from nanochat.scratch.data import ByteTokenizer, LegacyByteTokenizer
        vocab_size = spec.get("vocab_size")
        if vocab_size is None or vocab_size == LegacyByteTokenizer.vocab_size:
            return LegacyByteTokenizer()
        if vocab_size == ByteTokenizer.vocab_size:
            return ByteTokenizer()
        raise ValueError(f"no byte tokenizer has vocab_size {vocab_size}")
    if spec["kind"] == "bpe":
        if not os.path.exists(os.path.join(spec["dir"], "tokenizer.pkl")):
            raise FileNotFoundError(
                f"no BPE tokenizer in {spec['dir']}. Train one first, e.g.\n"
                f"    python -m scripts.tok_train --text-file book.txt --vocab-size 512")
        tokenizer = RustBPETokenizer.from_directory(spec["dir"])
        expected = spec.get("vocab_size")
        if expected is not None and tokenizer.get_vocab_size() != expected:
            raise ValueError(f"tokenizer in {spec['dir']} has vocab "
                             f"{tokenizer.get_vocab_size()}, the checkpoint expects {expected}")
        return tokenizer
    raise ValueError(f"unknown tokenizer kind: {spec['kind']!r}")


def snapshot_tokenizer(spec, run_dir):
    """Copy a BPE tokenizer into `run_dir/tokenizer` and return the spec pointing there.

    Records the vocabulary size too, so a mismatch is caught on load rather than
    surfacing as an embedding shape error (or, worse, as silently wrong tokens).
    """
    if spec["kind"] == "byte":
        return dict(spec)
    import shutil
    dest = os.path.abspath(os.path.join(run_dir, "tokenizer"))
    if os.path.abspath(spec["dir"]) != dest:
        shutil.copytree(spec["dir"], dest, dirs_exist_ok=True)
    vocab_size = RustBPETokenizer.from_directory(dest).get_vocab_size()
    return {"kind": "bpe", "dir": dest, "vocab_size": vocab_size}
