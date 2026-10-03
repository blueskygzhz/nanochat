"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

def _truncate_cache(cache, length, prev_embedding):
    """Discard speculative suffix and restore the normalized pre-smear embedding."""
    if not 0 <= length <= cache.get_pos():
        raise ValueError('Cannot extend a cache by truncating it')
    if length > 0 and (prev_embedding is None or prev_embedding.shape[:2] != (cache.batch_size, 1)):
        raise ValueError('Truncation requires the last retained token embedding')
    cache.cache_seqlens.fill_(length)
    if cache.attention_type == 'mla':
        cache._pos = length
    cache.prev_embedding = None if length == 0 else prev_embedding.detach().clone()


# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    attention_type = "gqa"

    @classmethod
    def from_config(cls, config, batch_size, seq_len, device, dtype):
        if getattr(config, 'attention_type', 'gqa') == 'mla':
            return MLAKVCache(batch_size, seq_len, config.n_layer, config.kv_lora_rank,
                              config.qk_rope_head_dim, device, dtype)
        return cls(batch_size, config.n_kv_head, seq_len, config.n_embd // config.n_head,
                   config.n_layer, device, dtype)

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def truncate(self, length, prev_embedding=None):
        _truncate_cache(self, length, prev_embedding)

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

class MLAKVCache:
    """Uniform-position cache storing only normalized KV latents and rotated shared keys."""
    attention_type = "mla"

    def __init__(self, batch_size, seq_len, num_layers, kv_rank, rope_dim, device, dtype):
        if min(batch_size, seq_len, num_layers, kv_rank, rope_dim) <= 0:
            raise ValueError('MLA cache dimensions must be positive')
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.kv_rank = kv_rank
        self.rope_dim = rope_dim
        self.latent_cache = torch.zeros(num_layers, batch_size, seq_len, kv_rank, device=device, dtype=dtype)
        self.rope_cache = torch.zeros(num_layers, batch_size, seq_len, rope_dim, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.prev_embedding = None
        self._pos = 0

    def truncate(self, length, prev_embedding=None):
        _truncate_cache(self, length, prev_embedding)

    def get_pos(self):
        return self._pos

    def reset(self):
        self._pos = 0
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def advance(self, num_tokens):
        if num_tokens < 0 or self._pos + num_tokens > self.max_seq_len:
            raise ValueError('MLA cache capacity exceeded')
        self._pos += num_tokens
        self.cache_seqlens.fill_(self._pos)

    def get_layer_cache(self, layer_idx):
        return self.latent_cache[layer_idx], self.rope_cache[layer_idx]

    def write_layer(self, layer_idx, latent, rope):
        if torch.is_grad_enabled():
            raise ValueError('MLA cache writes require no_grad or inference_mode')
        T = latent.size(1)
        if latent.shape != (self.batch_size, T, self.kv_rank) or rope.shape != (self.batch_size, T, self.rope_dim):
            raise ValueError('MLA cache shape mismatch')
        if (latent.device != self.latent_cache.device or latent.dtype != self.latent_cache.dtype
                or rope.device != self.rope_cache.device or rope.dtype != self.rope_cache.dtype):
            raise ValueError('MLA cache device/dtype mismatch')
        if self._pos + T > self.max_seq_len:
            raise ValueError('MLA cache capacity exceeded')
        self.latent_cache[layer_idx, :, self._pos:self._pos + T].copy_(latent)
        self.rope_cache[layer_idx, :, self._pos:self._pos + T].copy_(rope)

    def prefill(self, other):
        if not isinstance(other, MLAKVCache) or (self.n_layers, self.kv_rank, self.rope_dim) != (other.n_layers, other.kv_rank, other.rope_dim):
            raise ValueError('Incompatible MLA cache layouts')
        if self._pos != 0 or other._pos > self.max_seq_len or other.batch_size not in (1, self.batch_size):
            raise ValueError('Invalid MLA cache prefix copy')
        if self.latent_cache.dtype != other.latent_cache.dtype or self.latent_cache.device != other.latent_cache.device:
            raise ValueError('MLA cache device/dtype mismatch')
        pos = other._pos
        self.latent_cache[:, :, :pos].copy_(other.latent_cache[:, :, :pos])
        self.rope_cache[:, :, :pos].copy_(other.rope_cache[:, :, :pos])
        self.advance(pos)
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()


# -----------------------------------------------------------------------------
def sampling_distribution(logits, temperature, top_k):
    """Dense 1-D distribution the sampler draws from, after temperature and top-k.

    temperature=0 returns the degenerate one-hot argmax distribution, which makes
    speculative sampling collapse to exact greedy verification.
    """
    logits = logits.detach().float().reshape(-1)
    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs
    probs = F.softmax(logits / temperature, dim=-1)
    if top_k is not None and 0 < top_k < probs.numel():
        # identical to a softmax over the top-k logits, which is what sample_next_token does
        values, idx = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs).scatter_(0, idx, values / values.sum())
    return probs


def residual_distribution(target_probs, draft_probs):
    """norm(max(0, p - q)): the distribution a rejected draft token is replaced from."""
    residual = (target_probs - draft_probs).clamp_min(0)
    total = residual.sum()
    if total <= 0:
        return target_probs # p == q up to float noise: nothing left to correct towards
    return residual / total


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

    def commit(self, token, tokenizer, special):
        python_start, python_end, output_start, output_end, assistant_end, bos = special
        self.current_tokens.append(token)
        if token in (assistant_end, bos):
            self.completed = True
        if token == python_start:
            self.in_python_block = True
            self.python_expr_tokens = []
        elif token == python_end and self.in_python_block:
            self.in_python_block = False
            if self.python_expr_tokens:
                result = use_calculator(tokenizer.decode(self.python_expr_tokens))
                if result is not None:
                    self.forced_tokens.extend([output_start, *tokenizer.encode(str(result)), output_end])
            self.python_expr_tokens = []
        elif self.in_python_block:
            self.python_expr_tokens.append(token)

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42,
                 speculative=False, stats=None):
        """Generate tokens; speculative mode verifies one MTP proposal per step against the target."""
        assert isinstance(tokens, list) and tokens and isinstance(tokens[0], int), "expecting non-empty list of ints"
        if speculative:
            if num_samples != 1:
                raise ValueError('MTP speculative decoding supports only num_samples=1')
            if temperature < 0:
                raise ValueError('temperature must be non-negative')
            if not getattr(self.model.config, 'mtp_enabled', False):
                raise ValueError('Speculative decoding requires a checkpoint trained with --mtp')
            if self.model.training:
                raise ValueError('Speculative decoding requires model.eval()')
            rng = torch.Generator(device=self.model.get_device())
            rng.manual_seed(seed)
            yield from self._generate_speculative(tokens, max_tokens, temperature, top_k, rng, stats)
            return
        if max_tokens is not None and max_tokens <= 0:
            return
        device = self.model.get_device()
        # Allocate the KV cache in the compute dtype so it matches what the forward pass emits
        dtype = COMPUTE_DTYPE
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_cache_prefill = KVCache.from_config(m, batch_size=1, seq_len=len(tokens), device=device, dtype=dtype)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache.from_config(m, batch_size=num_samples, seq_len=kv_length_hint, device=device, dtype=dtype)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.commit(next_token, self.tokenizer, (python_start, python_end, output_start, output_end, assistant_end, bos))

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            if all(state.completed for state in row_states) or (max_tokens is not None and num_generated >= max_tokens):
                break
            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    @torch.inference_mode()
    def _generate_speculative(self, tokens, max_tokens, temperature, top_k, rng, stats):
        """Verify [target token, one MTP proposal] together; never commit an unverified proposal.

        temperature=0 is exact greedy verification (accept iff the draft is the target argmax).
        Otherwise this is speculative sampling (Leviathan et al. 2023 / Chen et al. 2023): the
        draft b ~ q is accepted with probability min(1, p(b)/q(b)), and a rejection is replaced
        by a draw from norm(max(0, p - q)), so every committed token is an exact sample of the
        target distribution p. Note that this preserves the output *distribution*, not the RNG
        stream: rejections consume extra randomness, so the token sequence does not match the
        non-speculative sampler for the same seed (it does for temperature=0 or top_k=1, where
        both distributions are one-hot).
        """
        stats = {} if stats is None else stats
        stats.clear()
        stats.update(draft_calls=0, draft_tokens=0, accepted_draft_tokens=0, verification_calls=0,
                     target_calls=0, target_tokens=0, generated_tokens=0, forced_tokens=0)
        budget = max_tokens if max_tokens is not None else max(0, self.model.config.sequence_len - len(tokens))
        if budget <= 0:
            return
        device = self.model.get_device()
        distribution = lambda logits: sampling_distribution(logits, temperature, top_k)
        if temperature == 0:
            draw = lambda probs: int(probs.argmax().item())
        else:
            draw = lambda probs: int(torch.multinomial(probs, 1, generator=rng).item())
        special = tuple(self.tokenizer.encode_special(s) for s in
                        ('<|python_start|>', '<|python_end|>', '<|output_start|>', '<|output_end|>', '<|assistant_end|>'))
        special = (*special, self.tokenizer.get_bos_token_id())
        tool_boundaries = special[:4]
        cache = KVCache.from_config(self.model.config, 1, len(tokens) + budget, device, COMPUTE_DTYPE)
        prompt = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, hidden = self.model.forward_with_hidden(prompt, kv_cache=cache)
        next_probs, hidden = distribution(logits[0, -1]), hidden[:, -1:]
        stats['target_calls'] += 1
        stats['target_tokens'] += len(tokens)
        state = RowState(tokens.copy())
        while stats['generated_tokens'] < budget and not state.completed:
            forced = bool(state.forced_tokens)
            token = state.forced_tokens.popleft() if forced else draw(next_probs)
            state.commit(token, self.tokenizer, special)
            stats['generated_tokens'] += 1
            stats['forced_tokens'] += int(forced)
            yield [token], [0 if forced else 1]
            if state.completed or stats['generated_tokens'] >= budget:
                return
            ids = torch.tensor([[token]], dtype=torch.long, device=device)
            draft, draft_probs = None, None
            if not forced and not state.in_python_block and not state.forced_tokens and token not in tool_boundaries:
                draft_probs = distribution(self.model.mtp_logits(hidden, ids)[0, -1])
                draft = draw(draft_probs)
                stats['draft_calls'] += 1
                if draft in tool_boundaries:
                    draft = None
            if draft is None:
                logits, hidden = self.model.forward_with_hidden(ids, kv_cache=cache)
                next_probs, hidden = distribution(logits[0, -1]), hidden[:, -1:]
                stats['target_calls'] += 1
                stats['target_tokens'] += 1
                continue
            pos = cache.get_pos()
            pair = torch.tensor([[token, draft]], dtype=torch.long, device=device)
            verified_logits, verified_hidden = self.model.forward_with_hidden(pair, kv_cache=cache)
            stats['target_calls'] += 1
            stats['target_tokens'] += 2
            stats['verification_calls'] += 1
            stats['draft_tokens'] += 1
            target_probs = distribution(verified_logits[0, 0])
            ratio = (target_probs[draft] / draft_probs[draft]).item()
            # short circuits keep temperature=0 and hopeless drafts from consuming randomness
            accepted = ratio >= 1.0 or (ratio > 0.0 and torch.rand((), generator=rng, device=device).item() < ratio)
            if accepted:
                next_probs, hidden = distribution(verified_logits[0, 1]), verified_hidden[:, 1:]
                state.commit(draft, self.tokenizer, special)
                stats['accepted_draft_tokens'] += 1
                stats['generated_tokens'] += 1
                yield [draft], [1]
            else:
                cache.truncate(pos + 1, self.model.embed_tokens(ids))
                next_probs, hidden = residual_distribution(target_probs, draft_probs), verified_hidden[:, :1]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
