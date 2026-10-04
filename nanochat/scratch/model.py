"""
nanochat's GPT, rebuilt on the from-scratch autograd engine.

This mirrors the architecture in `nanochat/gpt.py` -- RMSNorm everywhere, RoPE with
QK-norm, GQA, ResFormer value embeddings with a learned gate, embedding smear,
per-layer residual/x0 scalars, mid-layer backout, ReLU-squared FFNs, and
tanh-softcapped logits.

The MoE layer follows DeepSeek-V2's reference `modeling_deepseek.py` exactly: softmax
router, greedy or group-limited top-k, `norm_topk_prob` / `routed_scaling_factor`,
sequence- or batch-level balance loss attached through `AddAuxiliaryLoss`, SwiGLU
experts plus a fused shared expert, normal(0, initializer_range) expert init and a
kaiming-uniform router. The dense FFN can be switched to SwiGLU via `hidden_act`.

Deliberately *not* mirrored, because they are properties of the GPU stack rather
than of the model: FlashAttention (we use the naive O(T^2) form), FP8 matmuls,
bf16 compute, torch.compile, activation checkpointing, MLA, and the MTP head.
"""

from dataclasses import dataclass

import numpy as np

from nanochat.scratch import nn
from nanochat.scratch.tensor import (
    Tensor, add_aux_loss, cat, cross_entropy, get_dtype, index_add, no_grad, softmax, topk,
)

ACTIVATIONS = ("relu2", "silu")  # silu => SwiGLU (gate/up/down), as DeepSeek's hidden_act
TOPK_METHODS = ("greedy", "group_limited_greedy")


@dataclass
class GPTConfig:
    sequence_len: int = 64
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 64
    window_pattern: str = "SSSL"
    # MoE (n_routed_experts <= 0 => every layer is a dense MLP)
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    num_experts_per_tok: int = 2
    moe_intermediate_mult: float = 0.6875
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0
    aux_loss_alpha: float = 0.001
    seq_aux: bool = True
    scoring_func: str = "softmax"
    topk_method: str = "greedy"         # V2-Lite: greedy; V2: group_limited_greedy
    n_group: int = 1                    # V2: 8
    topk_group: int = 1                 # V2: 3
    moe_hidden_act: str = "silu"        # expert FFN; silu => SwiGLU, as DeepSeek
    initializer_range: float = 0.02     # std of DeepSeek's normal init (experts)
    # dense FFN
    hidden_act: str = "relu2"           # nanochat; "silu" => DeepSeek's SwiGLU MLP
    intermediate_size: int | None = None  # dense FFN width; None => 4 * n_embd
    # Multi-token prediction (DeepSeek-V3 sec. 2.2). n_mtp = D sequential modules; module
    # k predicts token t+k+1. They add (mtp_loss_weight / D) * sum_k L_k to the training
    # loss and serve as the draft model for speculative decoding in `Engine`.
    n_mtp: int = 0
    mtp_loss_weight: float = 0.3        # lambda; V3 uses 0.3, then 0.1 late in training

    def __post_init__(self):
        if self.n_mtp < 0:
            raise ValueError("n_mtp must be >= 0")
        if self.mtp_loss_weight < 0:
            raise ValueError("mtp_loss_weight must be >= 0")
        for name in ("hidden_act", "moe_hidden_act"):
            if getattr(self, name) not in ACTIVATIONS:
                raise ValueError(f"{name} must be one of {ACTIVATIONS}")
        if self.scoring_func != "softmax":
            raise ValueError(f"unsupported scoring_func {self.scoring_func!r} (DeepSeek-V2: softmax)")
        if self.topk_method not in TOPK_METHODS:
            raise ValueError(f"topk_method must be one of {TOPK_METHODS}")
        if self.n_routed_experts > 0 and self.topk_method == "group_limited_greedy":
            if self.n_routed_experts % self.n_group:
                raise ValueError("n_routed_experts must be divisible by n_group")
            if not 1 <= self.topk_group <= self.n_group:
                raise ValueError("topk_group must be in [1, n_group]")
            if self.num_experts_per_tok > self.topk_group * (self.n_routed_experts // self.n_group):
                raise ValueError("num_experts_per_tok exceeds the experts in topk_group groups")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.n_embd < 24:
            raise ValueError("n_embd must be >= 24 (the smear gate reads 24 channels)")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if not all(c in "SL" for c in self.window_pattern.upper()):
            raise ValueError("window_pattern may only contain S and L")

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


def has_ve(layer_idx, n_layer):
    """Value embeddings on alternating layers; the last layer always gets one."""
    return layer_idx % 2 == (n_layer - 1) % 2


def moe_intermediate_size(config):
    # The real model rounds up to a multiple of 128 for tensor cores. There are no
    # tensor cores here, so we round to 8 to keep tiny configs actually tiny.
    hidden = int(round(config.moe_intermediate_mult * config.n_embd))
    return max(8, -(-hidden // 8) * 8)


def is_moe_layer(layer_idx, config):
    return (config.n_routed_experts > 0
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0)


# ----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    VE_GATE_CHANNELS = 12

    def __init__(self, config, layer_idx, use_ve=None):
        super().__init__()
        self.layer_idx = layer_idx  # also this layer's slot in a KVCache
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        if use_ve is None:
            use_ve = has_ve(layer_idx, config.n_layer)
        self.ve_gate = nn.Linear(self.VE_GATE_CHANNELS, config.n_kv_head) if use_ve else None

    def forward(self, x, ve, cos_sin, window, kv_cache=None):
        B, T, _ = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): blend in the value embedding through a
        # per-head, input-dependent gate in (0, 3).
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3.0 * self.ve_gate(x[..., :self.VE_GATE_CHANNELS]).sigmoid()
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = nn.apply_rotary_emb(q, cos, sin)
        k = nn.apply_rotary_emb(k, cos, sin)
        q, k = nn.norm(q, scale=1.2), nn.norm(k, scale=1.2)  # QK norm + sharper attention

        if kv_cache is None:
            return self.c_proj(nn.attention(q, k, v, window=window).reshape(B, T, -1))

        # Decoding: append this step's K/V to the cache and attend over the prefix.
        # `offset` tells the mask where these queries sit in absolute position, which
        # is what keeps the causal/window geometry correct.
        offset = kv_cache.get_pos()
        k_all, v_all = kv_cache.append(self.layer_idx, k.data, v.data)
        # A sliding-window layer can only see keys j >= offset - window, so read just
        # that slice: per-step attention cost is O(window) instead of O(prefix), and
        # in steady state the (shifted) mask is the same every step, so it is cached.
        start = max(0, offset - window) if window >= 0 else 0
        y = nn.attention(q, Tensor(k_all[:, start:]), Tensor(v_all[:, start:]),
                         window=window, offset=offset - start)
        return self.c_proj(y.reshape(B, T, -1))


class MLP(nn.Module):
    """Dense FFN, also the body of every MoE expert.

    act="relu2": nanochat's `c_proj(relu(c_fc(x))^2)`.
    act="silu":  DeepSeek's `DeepseekV2MLP`, `down_proj(silu(gate_proj(x)) * up_proj(x))`,
                 with the same parameter names.
    """

    def __init__(self, config, intermediate_size=None, act=None):
        super().__init__()
        hidden = 4 * config.n_embd if intermediate_size is None else intermediate_size
        self.act = config.hidden_act if act is None else act
        if self.act == "silu":
            self.gate_proj = nn.Linear(config.n_embd, hidden)
            self.up_proj = nn.Linear(config.n_embd, hidden)
            self.down_proj = nn.Linear(hidden, config.n_embd)
        else:
            self.c_fc = nn.Linear(config.n_embd, hidden)
            self.c_proj = nn.Linear(hidden, config.n_embd)

    def forward(self, x):
        if self.act == "silu":
            return self.down_proj(nn.swiglu(self.gate_proj(x), self.up_proj(x)))
        return self.c_proj(nn.relu_squared(self.c_fc(x)))

    def in_projs(self):
        return (self.gate_proj, self.up_proj) if self.act == "silu" else (self.c_fc,)

    def out_proj(self):
        return self.down_proj if self.act == "silu" else self.c_proj


class MoEGate(nn.Module):
    """The router. Picks top-k experts per token and produces the load-balancing loss.

    Note what is and is not differentiable here: the *indices* are discrete and carry
    no gradient, the *weights* do. The auxiliary loss is differentiable only through
    the mean softmax score (`Pi`); the dispatch counts (`fi`) are constants, because
    they come from scatter-adding ones at the chosen indices.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        if not 1 <= self.top_k <= self.n_routed_experts:
            raise ValueError("top_k must be in [1, n_routed_experts]")
        self.weight = nn.Parameter(np.zeros((self.n_routed_experts, config.n_embd)))

    def forward(self, x):
        B, T, C = x.shape
        N, E = B * T, self.n_routed_experts
        flat = x.reshape(N, C)
        scores = softmax(flat @ self.weight.mT, axis=-1)          # (N, E)

        if self.topk_method == "group_limited_greedy":
            # Device-limited routing (DeepSeek-V2 sec. 2.2.2): experts are split into
            # n_group contiguous groups, each scored by its best expert; only the top
            # topk_group groups stay eligible. Masked scores become 0, so they never win
            # and get no gradient -- the same as DeepSeek's masked_fill(..., 0.0).
            group_scores = scores.data.reshape(N, self.n_group, -1).max(axis=-1)
            group_idx = np.argpartition(-group_scores, self.topk_group - 1, axis=-1)[:, :self.topk_group]
            group_mask = np.zeros_like(group_scores)
            np.put_along_axis(group_mask, group_idx, 1.0, axis=-1)
            score_mask = np.repeat(group_mask, E // self.n_group, axis=-1)   # (N, E)
            topk_weight, topk_idx = topk(scores * score_mask, self.top_k, axis=-1)
        else:
            topk_weight, topk_idx = topk(scores, self.top_k, axis=-1)  # (N, k)

        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(axis=-1, keepdims=True) + 1e-20)
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        aux_loss = None
        if self.training and self.alpha > 0.0:
            if self.seq_aux:
                # Per-sequence balance: dispatch fraction within each sequence, averaged over batch
                idx_per_seq = topk_idx.reshape(B, T * self.top_k)
                ce = np.zeros((B, E), dtype=np.float32)
                np.add.at(ce, (np.arange(B)[:, None], idx_per_seq), 1.0)
                ce /= (T * self.top_k / E)
                mean_scores = scores.reshape(B, T, E).mean(axis=1)   # (B, E)
                aux_loss = (mean_scores * ce).sum(axis=1).mean() * self.alpha
            else:
                # Global balance over all tokens: sum_i P_i * f_i
                assignments = topk_idx.reshape(-1)
                counts = np.bincount(assignments, minlength=E).astype(np.float32)
                fi = (counts / assignments.size) * E
                aux_loss = (scores.mean(axis=0) * fi).sum() * self.alpha

        return topk_idx, topk_weight, aux_loss


class MoE(nn.Module):
    """DeepSeek-V2 MoE (`DeepseekV2MoE`): top-k routed experts plus always-on shared experts.

    As in DeepSeek's reference code, the auxiliary loss is attached to the routed
    output with `AddAuxiliaryLoss` (`add_aux_loss`): the forward is the identity, and
    the backward injects a gradient of exactly 1.0 into the aux loss. So it is not part
    of the returned loss value, it is optimised by any backward pass through this layer,
    and it is *not* scaled down by `loss / grad_accum_steps`. `self.aux_loss` keeps the
    value for logging.
    """

    def __init__(self, config):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        inter = moe_intermediate_size(config)
        act = config.moe_hidden_act
        self.experts = nn.ModuleList([MLP(config, inter, act) for _ in range(self.n_routed_experts)])
        self.gate = MoEGate(config)
        # n shared experts == one MLP n times as wide (the hidden units just concatenate)
        self.shared_experts = (MLP(config, inter * config.n_shared_experts, act)
                               if config.n_shared_experts > 0 else None)
        self.aux_loss = None

    def forward(self, x):
        B, T, C = x.shape
        topk_idx, topk_weight, self.aux_loss = self.gate(x)
        y = self._dispatch(x.reshape(B * T, C), topk_idx, topk_weight).view(B, T, C)
        if self.training and self.aux_loss is not None:
            y = add_aux_loss(y, self.aux_loss)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y

    def _dispatch(self, x_flat, topk_idx, topk_weight):
        """Sort (token, slot) pairs by expert, run each expert on one contiguous slice,
        then scatter-add back. Same math as DeepSeek's `moe_infer`.

        The scatter-add is what makes the backward work: a token routed to k experts
        gathers gradient from all k of its slots, which is precisely the adjoint of
        `index_add`."""
        N, k = topk_idx.shape
        flat_expert = topk_idx.reshape(-1)
        order = np.argsort(flat_expert, kind="stable")      # group pairs by expert
        tok_idx = order // k                                # source token of each pair
        w = topk_weight.reshape(N * k)[order].reshape(N * k, 1)
        counts = np.bincount(flat_expert, minlength=self.n_routed_experts)

        xs = x_flat[tok_idx]                                # tokens in expert order
        outs, start = [], 0
        for e, c in enumerate(counts):
            if c > 0:
                outs.append(self.experts[e](xs[start:start + c]))
                start += c
        y = cat(outs, axis=0) * w
        return index_add(x_flat.shape, tok_idx, y)


class Block(nn.Module):
    def __init__(self, config, layer_idx, use_ve=None, moe=None):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, use_ve=use_ve)
        if moe is None:
            moe = is_moe_layer(layer_idx, config)
        self.mlp = MoE(config) if moe else MLP(config, config.intermediate_size)

    def forward(self, x, ve, cos_sin, window, kv_cache=None):
        x = x + self.attn(nn.norm(x), ve, cos_sin, window, kv_cache)
        x = x + self.mlp(nn.norm(x))
        return x


class MTPModule(nn.Module):
    """One multi-token-prediction depth, as in DeepSeek-V3 (sec. 2.2, fig. 3).

        h'_i = proj([ RMSNorm(h_i^{k-1}) ; RMSNorm(Emb(t_{i+k})) ])
        h_i^k = Block(h'_i)                     -> shared head predicts t_{i+k+1}

    The embedding and output head are the main model's (shared, not copied). The block
    has full-context attention, no value embedding, and the same FFN kind as the main
    model's last layer (MoE for an MoE model, as in V3). `depth` (0-based) is also the
    block's slot in the separate MTP `KVCache` used while drafting.
    """

    def __init__(self, config, depth):
        super().__init__()
        self.proj = nn.Linear(2 * config.n_embd, config.n_embd)
        self.block = Block(config, depth, use_ve=False,
                           moe=is_moe_layer(config.n_layer - 1, config))


# ----------------------------------------------------------------------------

class GPT(nn.Module):
    SMEAR_GATE_CHANNELS = 24

    def __init__(self, config, seed=0):
        super().__init__()
        self.config = config
        self.seed = seed
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.resid_lambdas = nn.Parameter(np.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(np.zeros(config.n_layer))
        self.smear_gate = nn.Linear(self.SMEAR_GATE_CHANNELS, 1)
        self.smear_lambda = nn.Parameter(np.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * np.ones(1))
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, config.n_kv_head * config.head_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Only registered when used, so models without MTP keep their exact state dict
        if config.n_mtp > 0:
            self.mtp = nn.ModuleList([MTPModule(config, k) for k in range(config.n_mtp)])
        else:
            self.mtp = None
        self.mtp_loss_values = None  # per-depth MTP losses of the last training forward
        self.cos, self.sin = self._precompute_rotary(config.sequence_len, config.head_dim)
        self.init_weights()

    # -- setup ---------------------------------------------------------------

    def _compute_window_sizes(self, config):
        """Per-layer left-window sizes. L = full context, S = quarter context.
        The pattern tiles across layers and the last layer is always L."""
        pattern = config.window_pattern.upper()
        short = max(1, config.sequence_len // 4)
        sizes = [(-1 if pattern[i % len(pattern)] == "L" else short)
                 for i in range(config.n_layer)]
        sizes[-1] = -1
        return sizes

    def _precompute_rotary(self, seq_len, head_dim, base=100000.0):
        channels = np.arange(0, head_dim, 2, dtype=np.float32)
        inv_freq = 1.0 / (base ** (channels / head_dim))
        freqs = np.outer(np.arange(seq_len, dtype=np.float32), inv_freq)
        # (1, T, 1, head_dim/2) so they broadcast over batch and heads
        cos = Tensor(np.cos(freqs)[None, :, None, :])
        sin = Tensor(np.sin(freqs)[None, :, None, :])
        return cos, sin

    @no_grad()
    def init_weights(self):
        """The whole initialisation scheme in one function, mirroring nanochat.

        The two choices worth noticing: every attention and dense-FFN output
        projection starts at exactly zero (so those sublayers are identity maps at
        step 0 and the residual stream is clean), and `resid_lambdas`/`x0_lambdas`
        decay with depth so early layers lean on the embedding and deep layers lean on
        the residual. MoE layers instead use DeepSeek-V2's scheme: N(0, 0.02) experts
        and a kaiming-uniform router.

        Every random draw comes from `self.seed`, so two models with the same config
        and seed are bit-identical and different seeds give independent inits.
        """
        rng = np.random.default_rng(self.seed)
        dt = get_dtype()
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5  # sqrt(3) makes Uniform match Normal's std

        self.wte.weight.data = rng.normal(0.0, 0.8, self.wte.weight.shape).astype(dt)
        self.lm_head.weight.data = rng.normal(0.0, 0.001, self.lm_head.weight.shape).astype(dt)

        for block in self.h:
            self._init_block(block, rng)

        n_layer = self.config.n_layer
        denom = max(n_layer - 1, 1)
        self.resid_lambdas.data = np.array(
            [1.15 - 0.10 * i / denom for i in range(n_layer)], dtype=dt)
        self.x0_lambdas.data = np.array(
            [0.20 - 0.15 * i / denom for i in range(n_layer)], dtype=dt)
        self.smear_lambda.data = np.zeros(1, dtype=dt)
        self.backout_lambda.data = np.full(1, 0.2, dtype=dt)
        self.smear_gate.weight.data = rng.uniform(
            0.0, 0.02, self.smear_gate.weight.shape).astype(dt)
        for ve in self.value_embeds.values():
            ve.weight.data = rng.uniform(-s, s, ve.weight.shape).astype(dt)

        # MTP draws come last so a model without MTP initialises exactly as before
        if self.mtp is not None:
            s2 = 3 ** 0.5 * (2 * n_embd) ** -0.5  # proj reads [h ; emb], fan_in 2C
            for module in self.mtp:
                module.proj.weight.data = rng.uniform(-s2, s2, module.proj.weight.shape).astype(dt)
                self._init_block(module.block, rng)

    def _init_block(self, block, rng):
        """Attention: uniform q/k/v, zero output projection. FFN: nanochat's scheme for a
        dense MLP (zero output projection), DeepSeek-V2's for MoE."""
        dt = get_dtype()
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5  # sqrt(3) makes Uniform match Normal's std
        for lin in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
            lin.weight.data = rng.uniform(-s, s, lin.weight.shape).astype(dt)
        block.attn.c_proj.weight.data = np.zeros_like(block.attn.c_proj.weight.data)
        if block.attn.ve_gate is not None:
            block.attn.ve_gate.weight.data = rng.uniform(
                0.0, 0.02, block.attn.ve_gate.weight.shape).astype(dt)

        if isinstance(block.mlp, MoE):
            # DeepSeek's `_init_weights`: every expert Linear ~ N(0, initializer_range)
            experts = list(block.mlp.experts)
            if block.mlp.shared_experts is not None:
                experts.append(block.mlp.shared_experts)
            std = self.config.initializer_range
            for expert in experts:
                for lin in (*expert.in_projs(), expert.out_proj()):
                    lin.weight.data = rng.normal(0.0, std, lin.weight.shape).astype(dt)
            # The router is a bare Parameter, which `_init_weights` skips: it keeps
            # kaiming_uniform_(a=sqrt(5)), i.e. Uniform(+-1/sqrt(fan_in))
            block.mlp.gate.weight.data = rng.uniform(
                -n_embd ** -0.5, n_embd ** -0.5, block.mlp.gate.weight.shape).astype(dt)
        else:
            for lin in block.mlp.in_projs():
                lin.weight.data = rng.uniform(-s * 0.4, s * 0.4, lin.weight.shape).astype(dt)
            out = block.mlp.out_proj()
            out.weight.data = np.zeros_like(out.weight.data)

    # -- forward -------------------------------------------------------------

    def embed_tokens(self, idx):
        return nn.norm(self.wte(idx))

    def forward_hidden(self, idx, kv_cache=None):
        idx = np.asarray(idx, dtype=np.int64)
        B, T = idx.shape
        # With a cache, the rotary tables must cover the absolute positions T0..T0+T
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        if T0 + T > self.config.sequence_len:
            raise ValueError(
                f"sequence length {T0 + T} exceeds rotary cache {self.config.sequence_len}")
        if kv_cache is None and T < 2:
            raise ValueError("forward needs T >= 2 (the smear reads the previous token)")
        cos_sin = (self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T])

        x = self.embed_tokens(idx)
        # Smear: mix the previous token's embedding in, gated. Cheap bigram information.
        # The pre-smear embedding of the last token is cache state: at the next decode
        # step there is no in-batch predecessor to read, so it has to be carried over.
        if kv_cache is None:
            gate = self.smear_lambda * self.smear_gate(x[:, 1:, :self.SMEAR_GATE_CHANNELS]).sigmoid()
            x = cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], axis=1)
        else:
            prev = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :].data.copy()
            if T > 1:
                gate = self.smear_lambda * self.smear_gate(x[:, 1:, :self.SMEAR_GATE_CHANNELS]).sigmoid()
                first = x[:, :1]
                if prev is not None:
                    first_gate = self.smear_lambda * self.smear_gate(
                        first[:, :, :self.SMEAR_GATE_CHANNELS]).sigmoid()
                    first = first + first_gate * Tensor(prev)
                x = cat([first, x[:, 1:] + gate * x[:, :-1]], axis=1)
            elif prev is not None:
                gate = self.smear_lambda * self.smear_gate(x[:, :, :self.SMEAR_GATE_CHANNELS]).sigmoid()
                x = x + gate * Tensor(prev)

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        for i, block in enumerate(self.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda * x_backout
        if kv_cache is not None:
            kv_cache.advance(T)
        return nn.norm(x)

    def logits(self, hidden):
        """Softcap the logits with 15*tanh(z/15): keeps them in a sane range without
        a hard clip, which matters because the head starts at ~zero init."""
        return 15.0 * (self.lm_head(hidden) / 15.0).tanh()

    def collect_aux_loss(self):
        """Sum of the MoE load-balancing losses from the most recent forward.

        For logging only: the losses are already wired into the graph by
        `add_aux_loss`, so adding this to the objective would count them twice.
        """
        total = None
        for m in self.modules():
            if isinstance(m, MoE) and m.aux_loss is not None:
                total = m.aux_loss if total is None else total + m.aux_loss
        return total

    # -- multi-token prediction ----------------------------------------------

    def mtp_module_forward(self, depth, h_prev, tokens, T0=0, kv_cache=None):
        """Run MTP module `depth` (0-based) over a contiguous run of positions.

        h_prev: (B, L, C) -- the main model's hidden states for depth 0, the previous
                module's outputs otherwise, at absolute positions T0 .. T0+L-1.
        tokens: (B, L) ints -- t_{i+depth+1} for each of those positions i.
        Returns the module's output states (B, L, C); `self.logits(nn.norm(.))` turns
        them into the distribution over t_{i+depth+2}.

        With `kv_cache` (an MTP cache, one layer per depth) the block attends over the
        cached prefix too; T0 must then equal the cache position. The caller advances it.
        """
        module = self.mtp[depth]
        L = h_prev.shape[1]
        if T0 + L > self.config.sequence_len:
            raise ValueError(f"MTP positions up to {T0 + L} exceed rotary cache {self.config.sequence_len}")
        x = module.proj(cat([nn.norm(h_prev), self.embed_tokens(tokens)], axis=-1))
        cos_sin = (self.cos[:, T0:T0 + L], self.sin[:, T0:T0 + L])
        return module.block(x, None, cos_sin, -1, kv_cache)

    def mtp_logits(self, idx, hidden=None):
        """Teacher-forced MTP predictions for a (B, T) batch.

        Returns one (B, T - k, V) logits tensor per depth k = 1..D: entry i of depth k
        predicts token t_{i+k+1}, from the main hidden state at i and the true tokens
        t_{i+1} .. t_{i+k} (the causal chain of V3's fig. 3). Every input token is inside
        `idx`, so only the targets need the token after the window.
        """
        if self.mtp is None:
            return []
        idx = np.asarray(idx, dtype=np.int64)
        T = idx.shape[1]
        h = self.forward_hidden(idx) if hidden is None else hidden
        out = []
        for depth in range(self.config.n_mtp):
            k = depth + 1
            L = T - k
            if L <= 0:
                break
            h = self.mtp_module_forward(depth, h[:, :L], idx[:, k:])
            out.append(self.logits(nn.norm(h)))
        return out

    def mtp_losses(self, idx, targets, hidden=None):
        """Per-depth MTP cross-entropy: depth k's position i is scored on t_{i+k+1},
        which is `targets[:, i + k]` (targets are idx shifted left by one)."""
        targets = np.asarray(targets)
        losses = []
        for depth, logits in enumerate(self.mtp_logits(idx, hidden)):
            k = depth + 1
            B, L, V = logits.shape
            losses.append(cross_entropy(logits.reshape(B * L, V), targets[:, k:].reshape(-1),
                                        ignore_index=-1))
        return losses

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        hidden = self.forward_hidden(idx, kv_cache)
        logits = self.logits(hidden)
        if targets is None:
            return logits
        B, T, V = logits.shape
        loss = cross_entropy(logits.reshape(B * T, V), np.asarray(targets).reshape(-1),
                             ignore_index=-1, reduction=loss_reduction)
        if loss_reduction == "none":
            loss = loss.reshape(B, T)  # (B, T) per-token losses, for bits-per-byte
        # As in DeepSeek, the returned loss is the LM loss only; the MoE balance losses
        # reach the gradient through `add_aux_loss` inside each MoE layer.
        #
        # MTP (V3 eq. 25): in training, add (lambda / D) * sum_k L_k. It is a real
        # objective, so -- unlike the balance loss -- it is part of the returned value
        # and scales with `loss / grad_accum_steps`. Eval mode and per-token losses
        # (bits-per-byte) stay pure next-token, so metrics are comparable with and
        # without MTP. The parts are kept in `mtp_loss_values` for logging.
        self.mtp_loss_values = None
        if (self.mtp is not None and self.training and kv_cache is None
                and loss_reduction == "mean" and self.config.mtp_loss_weight > 0):
            mtp = self.mtp_losses(idx, targets, hidden)
            if mtp:
                self.mtp_loss_values = [m.item() for m in mtp]
                total = mtp[0]
                for m in mtp[1:]:
                    total = total + m
                loss = loss + total * (self.config.mtp_loss_weight / len(mtp))
        return loss

    # -- inference -----------------------------------------------------------

    @no_grad()
    def generate(self, prompt, max_tokens, temperature=1.0, top_k=None, seed=0, use_cache=True):
        """Autoregressive sampling for one sequence.

        With `use_cache` (the default) the prompt is prefilled once into a `KVCache`
        and every later step feeds only the newest token, so each step attends one
        query against the cached prefix: O(n) forwards' worth of work instead of
        O(n^2). The cache reproduces the full forward exactly (rotary offset, window
        masks and the smear's carried embedding all live in it), so both paths emit the
        same tokens.

        Past `sequence_len` the context slides, as without a cache: the rotary table
        only covers positions < sequence_len, so the last `sequence_len` tokens are
        re-prefilled at positions 0.. -- the cache cannot be shifted in place because
        every cached key is rotated by its absolute position.

        `use_cache=False` keeps the reference O(n^2) loop that re-runs the whole window.
        """
        rng = np.random.default_rng(seed)
        tokens = list(prompt)
        if max_tokens <= 0:
            return tokens
        if not tokens:
            raise ValueError("generate needs a non-empty prompt")
        cap = self.config.sequence_len
        cache = None
        if use_cache:
            from nanochat.scratch.engine import KVCache
            cache = KVCache.from_config(self.config, batch_size=1, dtype=self.wte.weight.data.dtype)

        def next_logits():
            if cache is None:
                window = tokens[-cap:]
                return self.forward(np.array([window], dtype=np.int64)).data[0, -1]
            if cache.get_pos() == 0 or cache.get_pos() >= cap:
                cache.reset()                                    # (re)prefill the window
                window = tokens[-cap:]
                return self.forward(np.array([window], dtype=np.int64), kv_cache=cache).data[0, -1]
            return self.forward(np.array([[tokens[-1]]], dtype=np.int64), kv_cache=cache).data[0, -1]

        for _ in range(max_tokens):
            logits = next_logits()
            if temperature == 0.0:
                nxt = int(np.argmax(logits))
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = np.partition(logits, -top_k)[-top_k]
                    logits = np.where(logits < kth, -np.inf, logits)
                p = np.exp(logits.astype(np.float64) - logits.max())
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            tokens.append(nxt)
        return tokens
