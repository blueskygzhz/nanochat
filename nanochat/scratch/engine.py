"""
Inference engine with a KV cache. The from-scratch counterpart of `nanochat/engine.py`.

Why a cache at all: a naive sampler re-runs the whole prefix every step, so emitting
n tokens costs O(n^2) forward passes worth of work. Attention's keys and values for a
position never change once computed, so caching them turns each step into a single
query against a growing prefix -- O(n) total.

What this keeps from the upstream engine:
  - prefill / decode split, with batched multi-sample generation off one prefill
  - per-row stop conditions and the Python-calculator tool loop
  - the `<|python_start|> ... <|python_end|>` -> `<|output_start|> ... <|output_end|>`
    protocol, so a model can call out mid-generation

What it drops: FlashAttention's in-place cache kernels (we keep plain arrays),
speculative decoding with an MTP draft head (there is no MTP head here), and the
MLA compressed cache.
"""

import numpy as np

from nanochat.scratch.tensor import Tensor, no_grad

__all__ = ["KVCache", "Engine", "sample_next_token", "use_calculator"]


# ----------------------------------------------------------------------------
# tool use

def use_calculator(expr):
    """Evaluate a simple arithmetic expression, or return None if it is not safe.

    Deliberately not `eval` on arbitrary input: only digits and the four operators
    are allowed through, which rules out attribute access, calls and comprehensions.
    """
    expr = expr.replace(",", "").strip()
    if not expr or any(c not in "0123456789+-*/(). " for c in expr):
        return None
    if "**" in expr:  # cheap exponentiation is an easy denial of service
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except Exception:
        return None
    return result if isinstance(result, (int, float)) and abs(result) < 1e15 else None


# ----------------------------------------------------------------------------

class KVCache:
    """Pre-allocated key/value storage, shaped (n_layers, B, T, H, D).

    `prev_embedding` is the other piece of carried state: the model's embedding smear
    mixes in the previous token's pre-smear embedding, and at decode time that token
    is in the previous step, not in the current batch.
    """

    def __init__(self, batch_size, n_kv_head, seq_len, head_dim, n_layers, dtype=np.float32):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = n_layers
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        shape = (n_layers, batch_size, seq_len, n_kv_head, head_dim)
        self.k_cache = np.zeros(shape, dtype=dtype)
        self.v_cache = np.zeros(shape, dtype=dtype)
        self.pos = 0
        self.prev_embedding = None
        self._writes_this_step = 0

    @classmethod
    def from_config(cls, config, batch_size, seq_len=None, dtype=np.float32):
        return cls(batch_size, config.n_kv_head, seq_len or config.sequence_len,
                   config.head_dim, config.n_layer, dtype)

    def reset(self):
        self.pos = 0
        self.prev_embedding = None
        self._writes_this_step = 0

    def get_pos(self):
        return self.pos

    def append(self, layer_idx, k, v):
        """Write this step's K/V for one layer, return the full prefix including it."""
        T = k.shape[1]
        if self.pos + T > self.max_seq_len:
            raise ValueError(f"KV cache capacity exceeded: {self.pos + T} > {self.max_seq_len}")
        if k.shape[0] != self.batch_size:
            raise ValueError(f"batch mismatch: cache {self.batch_size}, got {k.shape[0]}")
        lo, hi = self.pos, self.pos + T
        self.k_cache[layer_idx, :, lo:hi] = k
        self.v_cache[layer_idx, :, lo:hi] = v
        return self.k_cache[layer_idx, :, :hi], self.v_cache[layer_idx, :, :hi]

    def advance(self, num_tokens):
        """Called once per forward, after every layer has written its slice."""
        self.pos += num_tokens

    def truncate(self, length, prev_embedding=None):
        """Roll the cache back, e.g. to discard a rejected speculative suffix."""
        if not 0 <= length <= self.pos:
            raise ValueError("cannot extend a cache by truncating it")
        self.pos = length
        self.prev_embedding = None if length == 0 else prev_embedding

    def expand(self, batch_size):
        """Copy a batch=1 cache out to `batch_size` rows.

        This is what makes num_samples>1 cheap: prefill the shared prompt once, then
        fan the result out so each sample continues from the same prefix.
        """
        if self.batch_size != 1:
            raise ValueError("can only expand a batch=1 cache")
        out = KVCache(batch_size, self.n_kv_head, self.max_seq_len, self.head_dim,
                      self.n_layers, self.k_cache.dtype)
        out.k_cache[:] = np.repeat(self.k_cache, batch_size, axis=1)
        out.v_cache[:] = np.repeat(self.v_cache, batch_size, axis=1)
        out.pos = self.pos
        if self.prev_embedding is not None:
            out.prev_embedding = np.repeat(self.prev_embedding, batch_size, axis=0)
        return out


# ----------------------------------------------------------------------------
# sampling

def sampling_distribution(logits, temperature, top_k=None):
    """Turn raw logits into the probability vector we actually sample from.

    Computed in float64 after subtracting the max: the exponentials underflow to 0
    rather than overflowing to inf, so the result is always a valid distribution.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0 for sampling (use argmax for greedy)")
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        kth = np.partition(logits, -k, axis=-1)[..., -k, None]
        logits = np.where(logits < kth, -np.inf, logits)
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample one token per row. temperature == 0 means greedy (argmax)."""
    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = logits[None, :]
    if temperature == 0.0:
        return logits.argmax(axis=-1)
    probs = sampling_distribution(logits, temperature, top_k)
    return np.array([rng.choice(probs.shape[-1], p=row) for row in probs])


# ----------------------------------------------------------------------------

class RowState:
    """Per-sample generation state: the tokens so far, and whether we're in a tool call."""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.completed = False
        self.in_python_block = False
        self.python_expr = []


class Engine:
    """Wraps a model and a tokenizer into a sampler with a KV cache."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # -- the two phases ------------------------------------------------------

    @no_grad()
    def prefill(self, tokens, kv_cache):
        """Run the prompt through in one forward, return logits for the last position."""
        idx = np.asarray(tokens, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        return self.model(idx, kv_cache=kv_cache).data[:, -1, :]

    @no_grad()
    def decode_one(self, last_tokens, kv_cache):
        """Advance one step: feed only the newest token, read logits for the next."""
        idx = np.asarray(last_tokens, dtype=np.int64).reshape(-1, 1)
        return self.model(idx, kv_cache=kv_cache).data[:, -1, :]

    # -- generation ----------------------------------------------------------

    @no_grad()
    def generate(self, tokens, max_tokens=None, num_samples=1, temperature=1.0, top_k=None,
                 seed=42, stop_tokens=None, use_tools=False, max_seq_len=None):
        """Sample `num_samples` continuations of `tokens`.

        Yields `(token, sample_index)` as tokens are produced, so a caller can stream.
        The prompt is prefilled once at batch=1 and the cache is then fanned out.
        """
        rng = np.random.default_rng(seed)
        prompt = list(tokens)
        if len(prompt) < 2:
            raise ValueError("prompt must have at least 2 tokens (the smear reads the previous one)")
        cap = max_seq_len or self.model.config.sequence_len
        if max_tokens is None:
            max_tokens = cap - len(prompt)
        if len(prompt) + max_tokens > cap:
            raise ValueError(f"prompt + max_tokens ({len(prompt) + max_tokens}) exceeds {cap}")

        stop = set(stop_tokens or [])
        tool_ids = self._tool_ids() if use_tools else None

        # Prefill once, then expand so all samples share the prompt's compute
        shared = KVCache.from_config(self.model.config, 1, cap, self.model.wte.weight.data.dtype)
        logits = self.prefill(prompt, shared)
        kv_cache = shared if num_samples == 1 else shared.expand(num_samples)
        logits = np.repeat(logits, num_samples, axis=0) if num_samples > 1 else logits

        rows = [RowState(prompt) for _ in range(num_samples)]
        for _ in range(max_tokens):
            next_tokens = sample_next_token(logits, rng, temperature, top_k)
            forced = [None] * num_samples

            for i, row in enumerate(rows):
                if row.completed:
                    forced[i] = self.tokenizer.get_bos_token_id()  # padding, not emitted
                    continue
                token = int(next_tokens[i])
                if tool_ids is not None:
                    token = self._step_tools(row, token, tool_ids)
                row.tokens.append(token)
                yield token, i
                if token in stop:
                    row.completed = True

            if all(r.completed for r in rows):
                break
            feed = [forced[i] if forced[i] is not None else rows[i].tokens[-1]
                    for i in range(num_samples)]
            logits = self.decode_one(feed, kv_cache)

    def generate_batch(self, tokens, **kwargs):
        """Collect `generate` into one token list per sample."""
        num_samples = kwargs.get("num_samples", 1)
        out = [[] for _ in range(num_samples)]
        for token, i in self.generate(tokens, **kwargs):
            out[i].append(token)
        return out

    @no_grad()
    def generate_text(self, prompt, max_tokens=64, **kwargs):
        """Convenience: str -> str, stopping at BOS."""
        ids = self.tokenizer.encode(prompt, prepend="<|bos|>")
        kwargs.setdefault("stop_tokens", [self.tokenizer.get_bos_token_id()])
        completion = self.generate_batch(ids, max_tokens=max_tokens, **kwargs)[0]
        stop = set(kwargs["stop_tokens"])
        completion = [t for t in completion if t not in stop]
        return self.tokenizer.decode(completion)

    # -- the calculator tool loop -------------------------------------------

    def _tool_ids(self):
        names = ["<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]
        try:
            return {n: self.tokenizer.encode_special(n) for n in names}
        except KeyError as e:
            raise ValueError(f"tokenizer lacks the tool special tokens: {e}") from None

    def _step_tools(self, row, token, tool_ids):
        """Track the python block and, on close, splice the result back in.

        The model emits `<|python_start|> expr <|python_end|>`; we evaluate `expr` and
        the caller sees `<|output_start|> result <|output_end|>` appear next, exactly
        as if the model had produced it.
        """
        if token == tool_ids["<|python_start|>"]:
            row.in_python_block = True
            row.python_expr = []
        elif token == tool_ids["<|python_end|>"] and row.in_python_block:
            row.in_python_block = False
            expr = self.tokenizer.decode(row.python_expr)
            result = use_calculator(expr)
            row.python_expr = []
            if result is not None:
                row.pending_output = self.tokenizer.encode(str(result))
        elif row.in_python_block:
            row.python_expr.append(token)
        return token
