"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW
from nanochat.mla import MultiHeadLatentAttention, LatentRMSNorm

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # ---- Mixture-of-Experts, DeepSeek-V2 style ----
    # n_routed_experts <= 0 => no MoE at all, every layer is a dense MLP (fully backward compatible).
    # Reference config (DeepSeek-V2-Lite): 64 routed + 2 shared experts, top-6, moe_intermediate 1408/2048.
    n_routed_experts: int = 0             # routed experts per MoE layer (0 => dense model)
    n_shared_experts: int = 0             # always-on shared experts (0 => disabled)
    num_experts_per_tok: int = 6          # top-k routed experts per token
    moe_intermediate_mult: float = 0.6875 # expert FFN hidden = mult * n_embd (1408/2048 in V2-Lite)
    first_k_dense_replace: int = 1        # the first K layers stay dense
    moe_layer_freq: int = 1               # of the remaining layers, every moe_layer_freq-th is MoE
    topk_method: str = "greedy"           # "greedy" | "group_limited_greedy"
    n_group: int = 1                      # expert groups (for group_limited_greedy)
    topk_group: int = 1                   # groups kept per token (for group_limited_greedy)
    norm_topk_prob: bool = False          # renormalize top-k weights to sum to 1
    routed_scaling_factor: float = 1.0    # scales routed output when norm_topk_prob is False
    aux_loss_alpha: float = 0.001         # load-balancing auxiliary loss weight
    seq_aux: bool = True                  # per-sequence aux loss (vs global over all tokens)
    attention_type: str = "gqa"
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    def __post_init__(self):
        if self.attention_type not in ("gqa", "mla"):
            raise ValueError("attention_type must be gqa or mla")
        if self.attention_type == "mla":
            if self.q_lora_rank < 0 or min(self.kv_lora_rank, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim) <= 0:
                raise ValueError("MLA dimensions must be positive (q_lora_rank may be zero)")
            if self.qk_rope_head_dim % 2:
                raise ValueError("MLA rotary dimension must be even")

    @property
    def rotary_dim(self):
        return self.qk_rope_head_dim if self.attention_type == "mla" else self.n_embd // self.n_head


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Dense FFN, also used as the body of every MoE expert (DeepSeek reuses one MLP class too)."""
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        hidden = 4 * config.n_embd if intermediate_size is None else intermediate_size
        self.c_fc = Linear(config.n_embd, hidden, bias=False)
        self.c_proj = Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


def moe_intermediate_size(config):
    """Expert FFN hidden dim: mult * n_embd, rounded up to a multiple of 128 for tensor cores."""
    hidden = int(round(config.moe_intermediate_mult * config.n_embd))
    return max(128, -(-hidden // 128) * 128)


def is_moe_layer(layer_idx, config):
    """DeepSeek-V2 layer placement: dense for the first K layers, then every moe_layer_freq-th layer."""
    return (config.n_routed_experts > 0
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0)


class MoEGate(nn.Module):
    """
    Router ("gate") of a MoE layer, mirroring DeepSeek-V2's MoEGate.

    Returns (topk_idx, topk_weight, aux_loss). Scores are always computed in fp32.
    The weight is a raw nn.Parameter (not our Linear) on purpose: it keeps routing out
    of the FP8 conversion pass and out of the Muon matrix groups, both of which would
    hurt a tiny, precision-sensitive matrix.
    """
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        assert 1 <= self.top_k <= self.n_routed_experts, "top_k must be in [1, n_routed_experts]"
        if self.topk_method == "group_limited_greedy":
            assert self.n_group > 0, "n_group must be positive"
            assert self.n_routed_experts % self.n_group == 0, "n_routed_experts must be divisible by n_group"
            assert 1 <= self.topk_group <= self.n_group, "topk_group must be in [1, n_group]"
            assert self.top_k <= self.topk_group * (self.n_routed_experts // self.n_group), "selected groups contain fewer than top_k experts"
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.n_embd)))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        # Gating scores in fp32 for a numerically stable softmax regardless of activation dtype
        logits = F.linear(hidden_states.float(), self.weight.float(), None)  # (N, E)
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        # Select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            # Score each group by its best expert, keep topk_group groups, then top-k within them
            group_scores = scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values  # (N, n_group)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(
                bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
            ).reshape(bsz * seq_len, -1)
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        else:
            raise NotImplementedError(f"unsupported topk_method: {self.topk_method}")

        # Normalize gate weights to sum to 1, or else apply the routed scaling factor
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        # Expert-level load-balancing auxiliary loss (training only)
        if self.training and self.alpha > 0.0:
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                # Per-sequence balance: dispatch fractions within each sequence, then averaged over the batch
                scores_for_seq_aux = scores.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * self.top_k, device=hidden_states.device),
                ).div_(seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Global balance over all tokens: sum_i P_i * f_i
                assignments = topk_idx_for_aux_loss.reshape(-1)
                counts = scores.new_zeros(self.n_routed_experts)
                counts.scatter_add_(0, assignments, scores.new_ones(assignments.numel()))
                ce = counts / assignments.numel()
                Pi = scores.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class MoE(nn.Module):
    """
    DeepSeek-V2 style MoE layer: n_routed_experts routed via top-k, plus optional
    always-on shared experts. Drop-in replacement for MLP.

    Deviation from DeepSeek's reference code (deliberate): instead of injecting the
    auxiliary loss through an AddAuxiliaryLoss autograd hack, we stash it on
    self.aux_loss and GPT.forward adds it to the main loss. The hack hard-codes an
    incoming gradient of 1.0, which would ignore nanochat's `loss / grad_accum_steps`
    scaling and silently over-weight the aux loss by grad_accum_steps.
    """
    def __init__(self, config):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        inter = moe_intermediate_size(config)
        self.experts = nn.ModuleList([MLP(config, intermediate_size=inter) for _ in range(self.n_routed_experts)])
        self.gate = MoEGate(config)
        # Shared experts are merged into one wider MLP, exactly as DeepSeek does
        self.shared_experts = MLP(config, intermediate_size=inter * config.n_shared_experts) if config.n_shared_experts > 0 else None
        self.aux_loss = None  # set every training forward, collected by GPT.forward

    def forward(self, x):
        identity = x
        B, T, C = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        self.aux_loss = aux_loss
        x_flat = x.view(-1, C)
        y = self._dispatch(x_flat, topk_idx, topk_weight).view(B, T, C)
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    # Excluded from torch.compile on purpose: per-expert token counts change every step, so
    # under dynamic=False each new shape would trigger a recompile (until the cache limit is
    # hit and dynamo silently falls back to eager anyway). Everything around it still compiles.
    @torch.compiler.disable
    def _dispatch(self, x_flat, topk_idx, topk_weight):
        """Sort (token, slot) pairs by expert, run each expert on one contiguous slice, scatter back.

        Same math as DeepSeek's moe_infer path. Exactly one GPU->CPU sync per layer (the
        per-expert counts), instead of one per expert. Experts that receive no tokens are not
        called at all, so their .grad stays None (handled by MuonAdamW._grad).
        """
        N, k = topk_idx.shape
        flat_expert = topk_idx.reshape(-1)                         # (N*k,) expert id of each (token, slot)
        order = flat_expert.argsort(stable=True)                   # group pairs by expert
        tok_idx = order // k                                       # source token of each sorted pair
        w = topk_weight.reshape(-1)[order].to(x_flat.dtype).unsqueeze(-1)  # (N*k, 1)
        counts = torch.bincount(flat_expert, minlength=self.n_routed_experts).tolist()  # the one sync
        xs = x_flat[tok_idx]                                       # (N*k, C) tokens in expert order
        ys, start = [], 0
        for e, c in enumerate(counts):
            if c > 0:
                ys.append(self.experts[e](xs[start:start + c]))
                start += c
        y = torch.cat(ys, dim=0) * w
        return torch.zeros_like(x_flat).index_add_(0, tok_idx, y)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = (MultiHeadLatentAttention(config, layer_idx, Linear, apply_rotary_emb)
                     if config.attention_type == "mla" else CausalSelfAttention(config, layer_idx))
        self.mlp = MoE(config) if is_moe_layer(layer_idx, config) else MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer)
                                          if config.attention_type == "gqa" and has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, config.rotary_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        # Activation checkpointing (runtime flag, not part of the config/checkpoint): recompute each
        # Block in backward instead of storing its activations. ~30% slower, needed for big models.
        self.activation_checkpointing = False
        self.loss_chunk_size = 0

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            if isinstance(block.attn, MultiHeadLatentAttention):
                block.attn.init_weights()
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            if isinstance(block.mlp, MoE):
                # Every expert (routed and shared) mirrors the dense MLP init
                experts = list(block.mlp.experts)
                if block.mlp.shared_experts is not None:
                    experts.append(block.mlp.shared_experts)
                for expert in experts:
                    torch.nn.init.uniform_(expert.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
                    torch.nn.init.zeros_(expert.c_proj.weight)
                # Router: bound = 1/sqrt(fan_in), i.e. exactly DeepSeek's kaiming_uniform_(a=sqrt(5))
                torch.nn.init.uniform_(block.mlp.gate.weight, -(n_embd**-0.5), n_embd**-0.5)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if isinstance(block.attn, CausalSelfAttention) and block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.config.rotary_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        if self.config.attention_type == "mla":
            # Model FLOPs: excludes checkpoint recomputation, SDPA padding and softmax.
            return 3 * self.estimate_prefill_flops(self.config.sequence_len) / self.config.sequence_len
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params(active=True) + attn_flops
        return num_flops_per_token

    def num_matmul_params(self, active=False):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.

        active=True counts only the params a single token actually flows through, which
        for a MoE layer is top_k of n_routed_experts (plus the always-on shared experts).
        Use active=True for any FLOPs/MFU math, total for memory/checkpoint math.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, (nn.Linear, MoEGate)))
        if active:
            for block in self.transformer.h:
                if isinstance(block.mlp, MoE):
                    routed = sum(p.numel() for e in block.mlp.experts for p in (e.c_fc.weight, e.c_proj.weight))
                    n, k = block.mlp.n_routed_experts, block.mlp.num_experts_per_tok
                    matmul_params -= round(routed * (n - k) / n) # skipped (inactive) experts
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        if self.config.attention_type == "mla":
            width = 2 * self.config.kv_lora_rank + self.config.qk_rope_head_dim
            attended = sum(context_len if w < 0 else min(context_len, w + 1) for w, _ in self.window_sizes)
            return 2 * self.num_matmul_params(active=True) + 2 * h * width * attended
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params(active=True) + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        mla = self.config.attention_type == "mla"
        pair_flops = 2 * h * (self.config.qk_nope_head_dim + self.config.qk_rope_head_dim + self.config.v_head_dim) if mla else 4 * h * q
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window + 1, num_tokens) if mla and window >= 0 else min(window, num_tokens)
            if window < 0:
                w = num_tokens
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += pair_flops * attended_tokens
        prefill_flops = 2 * self.num_matmul_params(active=True) * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to store one token of persistent KV cache, per row (all layers)."""
        if self.config.attention_type == "mla":
            return self.config.n_layer * (self.config.kv_lora_rank + self.config.qk_rope_head_dim) * COMPUTE_DTYPE.itemsize
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Logical cache reads per decode step, not measured HBM traffic.

        MLA consumes latent twice (scores and weighted sum), rotary keys once.
        Actual traffic depends on head reuse, kernel tiling and temporary tensors.
        """
        if self.config.attention_type == "mla":
            width = 2 * self.config.kv_lora_rank + self.config.qk_rope_head_dim
            attended = sum(context_len if w < 0 else min(context_len, w + 1) for w, _ in self.window_sizes)
            return width * COMPUTE_DTYPE.itemsize * attended
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        # For MoE models, the params a single token actually flows through ("active" / sparse count).
        # Identical to transformer_matrices for dense models.
        inactive = 0
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE):
                routed = sum(p.numel() for e in block.mlp.experts for p in (e.c_fc.weight, e.c_proj.weight))
                n, k = block.mlp.n_routed_experts, block.mlp.num_experts_per_tok
                inactive += round(routed * (n - k) / n)
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'transformer_matrices_active': transformer_matrices - inactive,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5, router_lr=None, muon_bucket_mb=0):
        if muon_bucket_mb < 0:
            raise ValueError("muon_bucket_mb must be non-negative")
        model_dim = self.config.n_embd

        # MoE routers get AdamW, not Muon: they are tiny (n_experts x n_embd), and orthogonalized
        # updates make little sense for a routing matrix. Also keeps them out of the shape-stacked
        # Muon groups. No weight decay (decaying the router just drags routing back to uniform).
        router_params = [m.weight for m in self.modules() if isinstance(m, MoEGate)]
        latent_norm_params = [m.weight for m in self.modules() if isinstance(m, LatentRMSNorm)]
        router_ids = {id(p) for p in router_params}
        adamw_ids = router_ids | {id(p) for p in latent_norm_params}

        # Separate out all parameters into groups
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in adamw_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params) + len(router_params) + len(latent_norm_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if router_params:
            lr = (matrix_lr if router_lr is None else router_lr) * dmodel_lr_scale
            param_groups.append(dict(kind='adamw', params=router_params, lr=lr, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        if latent_norm_params:
            param_groups.append(dict(kind='adamw', params=latent_norm_params, lr=matrix_lr * dmodel_lr_scale,
                                     betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Buckets preserve matrix boundaries; one matrix is the minimum allocation unit.
        world_size = get_dist_info()[3]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            bucket_size = len(group_params)
            if muon_bucket_mb > 0:
                p = group_params[0]
                capacity = max(1, int(muon_bucket_mb * 1024**2) // (p.numel() * p.element_size()))
                bucket_size = max(world_size, (capacity // world_size) * world_size)
            for start in range(0, len(group_params), bucket_size):
                param_groups.append(dict(
                    kind='muon', params=group_params[start:start + bucket_size], lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
                ))

        optimizer = MuonAdamW(param_groups, memory_efficient=muon_bucket_mb > 0)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def collect_aux_loss(self):
        """Sum the MoE load-balancing aux losses stashed by the most recent forward.
        Returns None for dense models or in eval mode. Also useful for logging it separately."""
        aux = None
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE) and block.mlp.aux_loss is not None:
                aux = block.mlp.aux_loss if aux is None else aux + block.mlp.aux_loss
        return aux

    def _compute_logits(self, x):
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        return 15 * torch.tanh(logits / 15)

    def _loss_chunk(self, x, targets, reduction):
        return F.cross_entropy(self._compute_logits(x), targets, ignore_index=-1, reduction=reduction)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        if T0 + T > self.cos.size(1):
            raise ValueError("Sequence exceeds rotary cache capacity")
        if kv_cache is not None:
            if self.config.attention_type == "mla" and torch.is_grad_enabled():
                raise ValueError("MLA cache is inference-only; disable gradient tracking")
            if getattr(kv_cache, 'attention_type', 'gqa') != self.config.attention_type:
                raise ValueError("KV cache attention type does not match model")
            if kv_cache.batch_size != B or kv_cache.n_layers != self.config.n_layer:
                raise ValueError("KV cache batch/layer shape mismatch")
            if T0 + T > kv_cache.max_seq_len:
                raise ValueError("KV cache capacity exceeded")
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                first = x[:, :1]
                if x_pre_smear is not None:
                    first_gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(first[:, :, :24]))
                    first = first + first_gate * x_pre_smear
                x = torch.cat([first, x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        use_ckpt = self.activation_checkpointing and self.training and kv_cache is None and torch.is_grad_enabled()
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            if use_ckpt:
                # Routing is deterministic given the inputs, so the recompute picks the same experts
                x = torch.utils.checkpoint.checkpoint(block, x, ve, cos_sin, self.window_sizes[i], None, use_reentrant=False)
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        if targets is None:
            return self._compute_logits(x)

        chunk_size = self.loss_chunk_size
        if chunk_size > 0:
            x_flat = x.reshape(-1, x.size(-1))
            targets_flat = targets.reshape(-1)
            reduction = 'none' if loss_reduction == 'none' else 'sum'
            if loss_reduction not in ('none', 'sum', 'mean'):
                raise ValueError(f"Unsupported loss reduction: {loss_reduction}")
            losses = []
            for start in range(0, x_flat.size(0), chunk_size):
                chunk_args = (x_flat[start:start + chunk_size], targets_flat[start:start + chunk_size], reduction)
                if self.training and torch.is_grad_enabled():
                    loss_chunk = torch.utils.checkpoint.checkpoint(
                        self._loss_chunk, *chunk_args, use_reentrant=False,
                    )
                else:
                    loss_chunk = self._loss_chunk(*chunk_args)
                losses.append(loss_chunk)
            loss = torch.cat(losses) if reduction == 'none' else torch.stack(losses).sum()
            if loss_reduction == 'mean':
                loss = loss / (targets_flat != -1).sum()
        else:
            logits = self._compute_logits(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)

        if loss_reduction == 'mean':
            aux = self.collect_aux_loss()
            if aux is not None:
                loss = loss + aux.to(loss.dtype)
        return loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
