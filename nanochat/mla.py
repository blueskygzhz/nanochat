"""DeepSeek-style multi-head latent attention with a compressed inference cache.

Training/prefill expands K/V for PyTorch SDPA. Cached continuations absorb the
K up-projection into Q and apply the V up-projection after attention, without
reconstructing historical per-head K/V. This is a reference backend, not FlashMLA.
RoPE uses nanochat's rotation convention; official DeepSeek checkpoints are not
interchangeable. Value embeddings and full-head QK normalization are not used.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LatentRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x.float(), (x.size(-1),), self.weight.float(), self.eps).to(x.dtype)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config, layer_idx, linear_cls, rotary_fn):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.kv_rank = config.kv_lora_rank
        self.nope_dim = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.v_dim = config.v_head_dim
        self.qk_dim = self.nope_dim + self.rope_dim
        self.scale = self.qk_dim ** -0.5
        self.rotary_fn = rotary_fn
        if config.q_lora_rank > 0:
            self.q_a_proj = linear_cls(config.n_embd, config.q_lora_rank, bias=False)
            self.q_norm = LatentRMSNorm(config.q_lora_rank)
            self.q_b_proj = linear_cls(config.q_lora_rank, self.n_head * self.qk_dim, bias=False)
        else:
            self.q_proj = linear_cls(config.n_embd, self.n_head * self.qk_dim, bias=False)
        self.kv_a_proj = linear_cls(config.n_embd, self.kv_rank + self.rope_dim, bias=False)
        self.kv_norm = LatentRMSNorm(self.kv_rank)
        self.kv_b_proj = linear_cls(self.kv_rank, self.n_head * (self.nope_dim + self.v_dim), bias=False)
        self.c_proj = linear_cls(self.n_head * self.v_dim, config.n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                bound = (3 / module.in_features) ** 0.5
                nn.init.uniform_(module.weight, -bound, bound)
            elif isinstance(module, LatentRMSNorm):
                nn.init.ones_(module.weight)
        nn.init.zeros_(self.c_proj.weight)

    def _project(self, x, cos_sin):
        B, T, _ = x.shape
        q = self.q_b_proj(self.q_norm(self.q_a_proj(x))) if hasattr(self, 'q_a_proj') else self.q_proj(x)
        q_nope, q_rope = q.view(B, T, self.n_head, self.qk_dim).split((self.nope_dim, self.rope_dim), dim=-1)
        latent, k_rope = self.kv_a_proj(x).split((self.kv_rank, self.rope_dim), dim=-1)
        latent = self.kv_norm(latent)
        cos, sin = cos_sin
        q_rope = self.rotary_fn(q_rope, cos, sin)
        k_rope = self.rotary_fn(k_rope.unsqueeze(2), cos, sin).squeeze(2)
        return q_nope, q_rope, latent, k_rope

    @staticmethod
    def _mask(query_len, key_len, query_start, key_start, window, device):
        rows = torch.arange(query_start, query_start + query_len, device=device)[:, None]
        cols = torch.arange(key_start, key_start + key_len, device=device)[None, :]
        mask = cols <= rows
        if window >= 0:
            mask = mask & (rows - cols <= window)
        return mask

    def _expanded_attention(self, q_nope, q_rope, latent, k_rope, window):
        B, T, _ = latent.shape
        kv = self.kv_b_proj(latent).view(B, T, self.n_head, self.nope_dim + self.v_dim)
        k_nope, v = kv.split((self.nope_dim, self.v_dim), dim=-1)
        q = torch.cat((q_nope, q_rope), dim=-1).transpose(1, 2)
        k = torch.cat((k_nope, k_rope.unsqueeze(2).expand(-1, -1, self.n_head, -1)), dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        # Matching head dimensions lets SDPA select its flash backend where supported.
        if self.v_dim < self.qk_dim:
            v = F.pad(v, (0, self.qk_dim - self.v_dim))
        mask = None if window < 0 or window >= T - 1 else self._mask(T, T, 0, 0, window, q.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None,
                                          dropout_p=0.0, scale=self.scale)
        return y[..., :self.v_dim].transpose(1, 2)

    def _absorbed_attention(self, q_nope, q_rope, latent, k_rope, query_start, key_start, window):
        weights = self.kv_b_proj.weight.to(q_nope.dtype).view(self.n_head, self.nope_dim + self.v_dim, self.kv_rank)
        q_latent = torch.einsum('bthd,hdr->bthr', q_nope, weights[:, :self.nope_dim])
        scores = torch.einsum('bthr,bsr->bhts', q_latent, latent).float()
        scores = scores + torch.einsum('bthd,bsd->bhts', q_rope, k_rope).float()
        mask = self._mask(q_nope.size(1), latent.size(1), query_start, key_start, window, q_nope.device)
        scores = (scores * self.scale).masked_fill(~mask, float('-inf'))
        probs = scores.softmax(dim=-1).to(q_nope.dtype)
        context = torch.einsum('bhts,bsr->bthr', probs, latent)
        return torch.einsum('bthr,hdr->bthd', context, weights[:, self.nope_dim:])

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if ve is not None:
            raise ValueError('MLA does not support value embeddings')
        if kv_cache is not None and torch.is_grad_enabled():
            raise ValueError('MLA KV cache is inference-only; use torch.no_grad() or inference_mode()')
        B, T, _ = x.shape
        q_nope, q_rope, latent, k_rope = self._project(x, cos_sin)
        window = window_size[0]
        if kv_cache is None:
            y = self._expanded_attention(q_nope, q_rope, latent, k_rope, window)
        else:
            pos = kv_cache.get_pos()
            kv_cache.write_layer(self.layer_idx, latent, k_rope)
            if pos == 0:
                y = self._expanded_attention(q_nope, q_rope, latent, k_rope, window)
            else:
                start = max(0, pos - window) if window >= 0 else 0
                cached_latent, cached_rope = kv_cache.get_layer_cache(self.layer_idx)
                y = self._absorbed_attention(q_nope, q_rope, cached_latent[:, start:pos + T],
                                             cached_rope[:, start:pos + T], pos, start, window)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        return self.c_proj(y.reshape(B, T, self.n_head * self.v_dim))
