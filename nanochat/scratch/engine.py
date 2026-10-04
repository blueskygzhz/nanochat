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
  - speculative decoding with the model's MTP modules as the draft (Leviathan et al.
    2023 / Chen et al. 2023): exact for any temperature and top-k, see
    `speculative_accept`

What it drops: FlashAttention's in-place cache kernels (we keep plain arrays) and the
MLA compressed cache.
"""

from collections import deque

import numpy as np

from nanochat.scratch import nn
from nanochat.scratch.tensor import Tensor, no_grad

__all__ = ["KVCache", "Engine", "sample_next_token", "use_calculator",
           "token_distribution", "speculative_accept"]


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


def _row_temperatures(temperature, n):
    """A scalar or a length-n sequence of temperatures -> (n,) float array, validated."""
    t = np.asarray(temperature, dtype=np.float64)
    if t.ndim == 0:
        t = np.full(n, float(t))
    elif t.shape != (n,):
        raise ValueError(f"need one temperature per row: got {t.shape[0]} for {n} rows")
    if (t < 0).any() or not np.isfinite(t).all():
        raise ValueError("temperatures must be finite and >= 0")
    return t


def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample one token per row. `temperature` is a scalar or one value per row;
    a row with temperature 0 is greedy (argmax)."""
    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = logits[None, :]
    temps = _row_temperatures(temperature, logits.shape[0])
    out = logits.argmax(axis=-1)
    for i in np.flatnonzero(temps > 0):  # rows in order: one rng draw per sampled row
        p = sampling_distribution(logits[i], temps[i], top_k)
        out[i] = rng.choice(p.shape[-1], p=p)
    return out


def token_distribution(logits, temperature, top_k=None):
    """The exact distribution a sampler with these settings draws from, as an array.

    Temperature 0 is greedy, i.e. a one-hot on the argmax. Representing it as a
    distribution (rather than as a special case) is what lets speculative decoding
    treat greedy and sampled decoding -- for the target and for the draft, in any
    combination -- with a single acceptance rule.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0.0:
        p = np.zeros_like(logits)
        np.put_along_axis(p, logits.argmax(axis=-1)[..., None], 1.0, axis=-1)
        return p
    return sampling_distribution(logits, temperature, top_k)


def speculative_accept(p, q, drafts, rng):
    """Speculative sampling (Leviathan et al. 2023, Chen et al. 2023), one round.

    p:      (d+1, V) target distributions; row i is the target's distribution for the
            token after the first i drafts (the verify forward produces all of them).
    q:      (d, V) the distributions the drafts were actually sampled from.
    drafts: (d,) the drafted tokens, drafts[i] ~ q[i].

    Draft i is accepted with probability min(1, p_i(x) / q_i(x)). At the first rejection
    the replacement is drawn from the residual norm(max(0, p_i - q_i)) and the round
    ends; if every draft is accepted, a bonus token is drawn from p_d. Either way each
    emitted token is distributed exactly as if sampled from the target alone:

        P(emit x) = q(x) min(1, p(x)/q(x)) + (1 - sum_y min(p(y), q(y))) r(x) = p(x).

    So the output distribution does not depend on q at all -- q (the draft model, its
    temperature) only changes how many tokens each round yields.

    Returns `(tokens, n_accepted)`; len(tokens) == n_accepted + 1.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d = len(drafts)
    if p.shape[0] != d + 1 or q.shape[0] != d:
        raise ValueError(f"need d+1 target and d draft rows, got {p.shape[0]} and {q.shape[0]} for d={d}")
    out = []
    for i in range(d):
        x = int(drafts[i])
        px, qx = p[i, x], q[i, x]
        # qx > 0 always holds for a token actually drawn from q; the guard keeps the
        # rule well-defined if a caller passes a draft q could not have produced.
        if qx > 0 and rng.random() < min(1.0, px / qx):
            out.append(x)
            continue
        residual = np.maximum(p[i] - q[i], 0.0)
        z = residual.sum()
        # z == 0 only if p == q, where rejection has probability 0; fall back to p
        out.append(int(rng.choice(p.shape[1], p=residual / z if z > 0 else p[i])))
        return out, i
    out.append(int(rng.choice(p.shape[1], p=p[d])))
    return out, d


# ----------------------------------------------------------------------------

class RowState:
    """Per-sample generation state.

    `forced` is the queue of tokens the engine will emit next *instead of* sampling.
    The tool loop fills it with `<|output_start|> result <|output_end|>`; they are
    yielded and fed back through the model like any other token, so the KV cache
    holds them and the model conditions on the tool's answer.
    """

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.completed = False
        self.in_python_block = False
        self.python_expr = []
        self.forced = deque()


class Engine:
    """Wraps a model and a tokenizer into a sampler with a KV cache.

    If the model has MTP modules (`config.n_mtp > 0`), `generate` decodes
    speculatively by default: the MTP chain drafts up to n_mtp tokens, one target
    forward verifies them all, and `speculative_accept` keeps the output distribution
    exactly the target's. `spec_stats` records drafted/accepted counts of the last call.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.spec_stats = None

    @property
    def acceptance_rate(self):
        """Fraction of drafted tokens accepted in the last speculative `generate`."""
        s = self.spec_stats
        return s["accepted"] / s["drafted"] if s and s["drafted"] else float("nan")

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
                 seed=42, stop_tokens=None, use_tools=False, max_seq_len=None,
                 speculative=None, draft_temperature=None):
        """Sample `num_samples` continuations of `tokens`.

        Yields `(token, sample_index)` as tokens are produced, so a caller can stream.
        The prompt is prefilled once at batch=1 and the cache is then fanned out.

        `temperature` is a scalar or one value per sample (0 = greedy), so one call can
        mix e.g. a greedy row with sampled ones. `speculative=None` means "on if the
        model has MTP modules and tools are off"; the output distribution is the same
        either way (greedy rows give identical tokens). `draft_temperature` (scalar or
        per sample, default: the sample's own temperature) only sets how the MTP draft
        proposes tokens -- e.g. 0 drafts greedily even when the target samples.
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
        temps = _row_temperatures(temperature, num_samples)

        stop = set(stop_tokens or [])
        has_mtp = getattr(self.model.config, "n_mtp", 0) > 0
        if speculative is None:
            speculative = has_mtp and not use_tools
        if speculative:
            if not has_mtp:
                raise ValueError("speculative decoding needs a model with MTP modules (n_mtp > 0)")
            if use_tools:
                raise ValueError("speculative decoding does not support the tool loop")
            draft_temps = temps if draft_temperature is None else _row_temperatures(
                draft_temperature, num_samples)
            yield from self._generate_speculative(prompt, max_tokens, temps, draft_temps,
                                                  top_k, rng, stop, cap)
            return
        if draft_temperature is not None:
            raise ValueError("draft_temperature only applies to speculative decoding")
        tool_ids = self._tool_ids() if use_tools else None

        # Prefill once, then expand so all samples share the prompt's compute
        shared = KVCache.from_config(self.model.config, 1, cap, self.model.wte.weight.data.dtype)
        logits = self.prefill(prompt, shared)
        kv_cache = shared if num_samples == 1 else shared.expand(num_samples)
        logits = np.repeat(logits, num_samples, axis=0) if num_samples > 1 else logits

        pad = self.tokenizer.get_bos_token_id()
        rows = [RowState(prompt) for _ in range(num_samples)]
        for emitted in range(max_tokens):
            next_tokens = sample_next_token(logits, rng, temps, top_k)

            for i, row in enumerate(rows):
                if row.completed:
                    continue
                # A queued tool result takes precedence over the model's own sample
                token = row.forced.popleft() if row.forced else int(next_tokens[i])
                if tool_ids is not None:
                    self._step_tools(row, token, tool_ids)
                row.tokens.append(token)
                yield token, i
                if token in stop:
                    row.completed = True

            if all(r.completed for r in rows):
                break
            if emitted == max_tokens - 1:
                break  # the next logits would never be sampled from; don't compute them
            # Finished rows still occupy a batch slot; feed them padding, never emitted
            feed = [pad if r.completed else r.tokens[-1] for r in rows]
            logits = self.decode_one(feed, kv_cache)

    # -- speculative decoding with the MTP draft ------------------------------

    def _generate_speculative(self, prompt, max_tokens, temps, draft_temps, top_k, rng, stop, cap):
        """Speculative generation, one sample at a time off a shared prefill.

        Rows accept different numbers of drafts per round, so their caches would sit at
        different positions; `KVCache` has a single position, hence one row at a time.
        Each row still reuses the prompt's prefill (the cache is copied, not recomputed).
        """
        model = self.model
        self.spec_stats = {"rounds": 0, "drafted": 0, "accepted": 0, "emitted": 0}
        dtype = model.wte.weight.data.dtype
        shared = KVCache.from_config(model.config, 1, cap, dtype)
        hidden = model.forward_hidden(np.asarray([prompt], dtype=np.int64), shared)
        first_logits = model.logits(hidden[:, -1:]).data[0, -1]
        for i in range(len(temps)):
            yield from ((tok, i) for tok in self._speculate_row(
                prompt, shared.expand(1), hidden.data, first_logits, temps[i], draft_temps[i],
                top_k, rng, stop, max_tokens, cap))

    def _speculate_row(self, prompt, kv, hbuf, first_logits, temperature, draft_temperature,
                       top_k, rng, stop, max_tokens, cap):
        """Draft with the MTP chain, verify with one target forward, accept exactly.

        Invariants between rounds, with `tokens` the committed sequence of length N:
          - the main cache holds tokens[:N-1] (the newest token is not fed yet);
          - the MTP cache holds positions [0, L) whose inputs are all committed;
          - `hbuf` holds the main hidden states of positions [L, N-1), the MTP inputs
            still to be processed.
        """
        model, D = self.model, self.model.config.n_mtp
        mkv = KVCache(1, model.config.n_kv_head, cap, model.config.head_dim, D, hbuf.dtype)
        stats = self.spec_stats
        tokens = list(prompt)
        V = first_logits.shape[-1]

        first = int(rng.choice(V, p=token_distribution(first_logits, temperature, top_k)))
        tokens.append(first)
        emitted = 1
        stats["emitted"] += 1
        yield first
        if first in stop:
            return

        while emitted < max_tokens:
            N = len(tokens)
            # Room for d drafts: they all get emitted at most (plus one corrected/bonus
            # token), and the verify forward writes d+1 positions starting at N-1.
            d = min(D, max_tokens - emitted - 1, cap - N)
            drafts, q = [], np.zeros((0, V))
            if d > 0:
                drafts, q, hbuf = self._draft(tokens, hbuf, mkv, draft_temperature, top_k, rng)
                drafts, q = drafts[:d], q[:d]

            feed = [tokens[-1]] + drafts
            h = model.forward_hidden(np.asarray([feed], dtype=np.int64), kv)
            p = token_distribution(model.logits(h).data[0], temperature, top_k)   # (d+1, V)
            new, n_acc = speculative_accept(p, q, drafts, rng)

            if n_acc < d:   # roll the main cache back to the last accepted fed token
                kept = feed[n_acc]
                kv.truncate(N + n_acc, prev_embedding=model.embed_tokens(
                    np.asarray([[kept]], dtype=np.int64)).data)
            hbuf = np.concatenate([hbuf, h.data[:, :n_acc + 1]], axis=1)

            stats["rounds"] += 1
            stats["drafted"] += d
            stats["accepted"] += n_acc
            for tok in new:
                tokens.append(tok)
                emitted += 1
                stats["emitted"] += 1
                yield tok
                if tok in stop:
                    return

    def _draft(self, tokens, hbuf, mkv, temperature, top_k, rng):
        """Run all n_mtp modules at the newest position and sample one draft from each.

        Module k at position i reads t_{i+k}. The pending positions [L, N-1) are run
        through every module in one go (module k's tokens for the last k-1 of them are
        drafts); afterwards the MTP cache is rolled back to the positions whose inputs
        are all committed, so a rejected draft never contaminates it.
        """
        model, D = self.model, self.model.config.n_mtp
        N, L = len(tokens), mkv.get_pos()
        R = N - 1 - L
        if hbuf.shape[1] != R:
            raise RuntimeError(f"MTP input buffer out of sync: {hbuf.shape[1]} != {R}")
        stream = list(tokens)
        h = Tensor(hbuf)
        drafts, q = [], []
        for depth in range(D):   # all depths, so every MTP cache layer advances in step
            k = depth + 1
            tok = np.asarray([stream[L + k:N - 1 + k]], dtype=np.int64)
            h = model.mtp_module_forward(depth, h, tok, T0=L, kv_cache=mkv)
            logits = model.logits(nn.norm(h[:, -1:])).data[0, -1]
            qd = token_distribution(logits, temperature, top_k)
            x = int(rng.choice(qd.shape[-1], p=qd))
            drafts.append(x)
            q.append(qd)
            stream.append(x)
        mkv.advance(R)
        # Depth k at position j used t_{j+k}: committed for every depth iff j <= N-1-D
        safe = max(L, N - D)
        mkv.truncate(safe)
        return drafts, np.stack(q), hbuf[:, safe - L:]

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
        """Track the python block and, on close, queue the result to be spliced in.

        The model emits `<|python_start|> expr <|python_end|>`; we evaluate `expr` and
        the caller sees `<|output_start|> result <|output_end|>` appear next, exactly
        as if the model had produced it. An expression the calculator refuses yields
        no output block, and the model simply carries on sampling.
        """
        if token == tool_ids["<|python_start|>"]:
            row.in_python_block = True
            row.python_expr = []
        elif token == tool_ids["<|python_end|>"] and row.in_python_block:
            row.in_python_block = False
            result = use_calculator(self.tokenizer.decode(row.python_expr))
            row.python_expr = []
            if result is not None:
                row.forced.append(tool_ids["<|output_start|>"])
                row.forced.extend(self.tokenizer.encode(str(result)))
                row.forced.append(tool_ids["<|output_end|>"])
        elif row.in_python_block:
            row.python_expr.append(token)
