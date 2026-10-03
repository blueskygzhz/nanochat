"""MLA equations, compressed-cache decoding, training and compatibility regressions."""
import copy
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from nanochat.engine import Engine, KVCache, MLAKVCache
from nanochat.gpt import GPT, GPTConfig, apply_rotary_emb
from nanochat.mla import LatentRMSNorm

COMPUTE_DTYPE = torch.float32


@pytest.fixture(autouse=True)
def cpu_reference_dtype(monkeypatch):
    monkeypatch.setattr('nanochat.gpt.COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr('nanochat.engine.COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr('nanochat.optim.COMPUTE_DTYPE', torch.float32)


def config(**kwargs):
    values = dict(sequence_len=32, vocab_size=128, n_layer=3, n_head=2, n_kv_head=1, n_embd=64,
                  window_pattern='L', attention_type='mla', q_lora_rank=16, kv_lora_rank=16,
                  qk_nope_head_dim=12, qk_rope_head_dim=8, v_head_dim=10)
    values.update(kwargs)
    return GPTConfig(**values)


def build(cfg, nonzero=True):
    with torch.device('meta'):
        model = GPT(cfg)
    model.to_empty(device='cpu')
    model.init_weights()
    if nonzero:
        with torch.no_grad():
            for p in model.parameters():
                if p.ndim == 2:
                    torch.nn.init.normal_(p, std=0.08)
            model.smear_lambda.fill_(0.4)
    return model


def new_cache(model, batch, length):
    return KVCache.from_config(model.config, batch, length, 'cpu', COMPUTE_DTYPE)


def naive_attention(attn, x, cos_sin, window):
    B, T, _ = x.shape
    def project(inp, module):
        return F.linear(inp, module.weight.to(inp.dtype))
    def rms(inp, module):
        normalized = inp.float() * torch.rsqrt(inp.float().square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * module.weight.float()).to(inp.dtype)
    if hasattr(attn, 'q_a_proj'):
        q = project(rms(project(x, attn.q_a_proj), attn.q_norm), attn.q_b_proj)
    else:
        q = project(x, attn.q_proj)
    q = q.view(B, T, attn.n_head, attn.qk_dim)
    q_nope, q_rope = q.split((attn.nope_dim, attn.rope_dim), -1)
    compressed = project(x, attn.kv_a_proj)
    latent, rope = compressed.split((attn.kv_rank, attn.rope_dim), -1)
    kv = project(rms(latent, attn.kv_norm), attn.kv_b_proj).view(B, T, attn.n_head, -1)
    k_nope, v = kv.split((attn.nope_dim, attn.v_dim), -1)
    cos, sin = cos_sin
    q = torch.cat((q_nope, apply_rotary_emb(q_rope, cos, sin)), -1)
    rope = apply_rotary_emb(rope.unsqueeze(2), cos, sin).expand(-1, -1, attn.n_head, -1)
    k = torch.cat((k_nope, rope), -1)
    scores = torch.einsum('bthd,bshd->bhts', q, k).float() * (attn.qk_dim ** -0.5)
    positions = torch.arange(T)
    distance = positions[:, None] - positions[None, :]
    mask = distance >= 0
    if window >= 0:
        mask &= distance <= window
    probs = scores.masked_fill(~mask, float('-inf')).softmax(-1).to(x.dtype)
    y = torch.einsum('bhts,bshd->bthd', probs, v).reshape(B, T, -1)
    return project(y, attn.c_proj)


@pytest.mark.parametrize('q_rank', [0, 16])
@pytest.mark.parametrize('window', [-1, 0, 3])
@pytest.mark.parametrize('v_dim', [10, 24])
def test_expanded_matches_independent_equations_and_gradients(q_rank, window, v_dim):
    model = build(config(q_lora_rank=q_rank, v_head_dim=v_dim))
    attn = model.transformer.h[0].attn
    x = torch.randn(2, 7, 64, requires_grad=True)
    cos_sin = model.cos[:, :7], model.sin[:, :7]
    actual = attn(x, None, cos_sin, (window, 0), None)
    expected = naive_attention(attn, x, cos_sin, window)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
    inputs = (x, *attn.parameters())
    a = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), inputs)
    for got, want in zip(a, b):
        torch.testing.assert_close(got, want, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize('q_rank', [0, 16])
@pytest.mark.parametrize('moe', [False, True])
@pytest.mark.parametrize('window', [-1, 0, 3])
def test_cached_chunks_and_single_tokens_match_full_forward(q_rank, moe, window):
    model = build(config(q_lora_rank=q_rank, n_routed_experts=4 if moe else 0,
                         num_experts_per_tok=2, n_shared_experts=1)).eval()
    model.window_sizes = [(window, 0)] * model.config.n_layer
    ids = torch.randint(0, 128, (2, 15))
    with torch.no_grad():
        expected = model(ids)
        cache = new_cache(model, 2, 15)
        parts = []
        start = 0
        for length in (4, 1, 3, 1, 6):
            parts.append(model(ids[:, start:start + length], kv_cache=cache))
            start += length
            assert cache.get_pos() == start
            assert (cache.cache_seqlens == start).all()
        torch.testing.assert_close(torch.cat(parts, 1), expected, rtol=3e-4, atol=3e-6)
        cache.reset()
        pieces = [model(ids[:, t:t + 1], kv_cache=cache) for t in range(15)]
        torch.testing.assert_close(torch.cat(pieces, 1), expected, rtol=3e-4, atol=3e-6)


def test_prefix_copy_broadcast_reset_and_no_historical_kv_expansion(monkeypatch):
    model = build(config()).eval()
    prompt = torch.randint(0, 128, (1, 5))
    suffix = torch.randint(0, 128, (3, 2))
    with torch.no_grad():
        source = new_cache(model, 1, 5)
        model(prompt, kv_cache=source)
        target = new_cache(model, 3, 7)
        target.prefill(source)
        expected = model(torch.cat((prompt.expand(3, -1), suffix), 1))[:, -2:]
        def unexpected(*args, **kwargs):
            raise AssertionError('cached continuation must not expand historical K/V')
        for block in model.transformer.h:
            monkeypatch.setattr(block.attn.kv_b_proj, 'forward', unexpected)
        torch.testing.assert_close(model(suffix, kv_cache=target), expected, rtol=3e-4, atol=3e-6)
    assert not hasattr(target, 'k_cache') and not hasattr(target, 'v_cache')
    assert target.latent_cache.shape == (3, 3, 7, 16)
    assert target.rope_cache.shape == (3, 3, 7, 8)
    stored = target.latent_cache.numel() + target.rope_cache.numel()
    assert stored * COMPUTE_DTYPE.itemsize == 3 * 7 * model.kv_bytes_per_token()
    target.reset()
    assert target.get_pos() == 0 and target.prev_embedding is None


def test_checkpointing_chunked_loss_and_optimizer(monkeypatch):
    import nanochat.optim as optim
    monkeypatch.setattr(optim, 'adamw_step_fused', optim.adamw_step_fused._torchdynamo_orig_callable)
    monkeypatch.setattr(optim, 'muon_step_fused', optim.muon_step_fused._torchdynamo_orig_callable)
    plain = build(config(n_routed_experts=4, num_experts_per_tok=2))
    ckpt = copy.deepcopy(plain)
    ckpt.activation_checkpointing = True
    ckpt.loss_chunk_size = 5
    ids = torch.randint(0, 128, (2, 9))
    targets = ids.clone()
    targets[:, :3] = -1
    a, b = plain(ids, targets), ckpt(ids, targets)
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b)
    for p, q in zip(plain.parameters(), ckpt.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=3e-6)
    optimizer = ckpt.setup_optimizer(muon_bucket_mb=1)
    norms = {id(m.weight) for m in ckpt.modules() if isinstance(m, LatentRMSNorm)}
    seen = []
    for group in optimizer.param_groups:
        for p in group['params']:
            seen.append(id(p))
            if id(p) in norms:
                assert group['kind'] == 'adamw'
            if group['kind'] == 'muon':
                assert p.ndim == 2
    assert len(seen) == len(set(seen)) == len(list(ckpt.parameters()))
    norm = ckpt.transformer.h[0].attn.kv_norm.weight
    before = norm.detach().clone()
    optimizer.step()
    assert not torch.equal(before, norm)
    assert all(torch.isfinite(p).all() for p in ckpt.parameters())


@pytest.mark.parametrize('overrides', [dict(attention_type='bad'), dict(q_lora_rank=-1), dict(kv_lora_rank=0),
                                      dict(qk_rope_head_dim=3), dict(qk_nope_head_dim=0), dict(v_head_dim=-1)])
def test_config_rejects_invalid_dimensions(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


def test_cache_layout_capacity_and_inference_only_errors():
    model = build(config()).eval()
    x = torch.ones(1, 2, dtype=torch.long)
    cache = new_cache(model, 1, 2)
    with pytest.raises(ValueError, match='inference'):
        model(x, kv_cache=cache)
    with torch.no_grad():
        model(x, kv_cache=cache)
        with pytest.raises(ValueError, match='capacity'):
            model(x[:, :1], kv_cache=cache)
        gqa = KVCache(1, 1, 4, 32, 3, 'cpu', COMPUTE_DTYPE)
        with pytest.raises(ValueError, match='attention'):
            model(x, kv_cache=gqa)
        with pytest.raises(ValueError, match='layouts'):
            cache.prefill(gqa)
        wrong = MLAKVCache(1, 4, 3, 17, 8, 'cpu', COMPUTE_DTYPE)
        with pytest.raises(ValueError, match='shape'):
            model(x, kv_cache=wrong)


def test_old_config_keeps_gqa_and_value_embeddings():
    legacy = dict(sequence_len=32, vocab_size=128, n_layer=2, n_head=2, n_kv_head=1, n_embd=64, window_pattern='L')
    model = build(GPTConfig(**legacy), nonzero=False)
    assert model.config.attention_type == 'gqa'
    assert len(model.value_embeds) == 1
    assert isinstance(new_cache(model, 1, 16), KVCache)
    mla = build(config(), nonzero=False)
    assert len(mla.value_embeds) == 0
    assert mla.cos.size(-1) == 4
    assert all(torch.equal(m.weight, torch.ones_like(m.weight)) for m in mla.modules() if isinstance(m, LatentRMSNorm))
    assert GPTConfig(**asdict(mla.config)) == mla.config


def test_engine_selects_compressed_cache_for_multiple_samples():
    from nanochat.tokenizer import SPECIAL_TOKENS
    class Tokenizer:
        def encode_special(self, name):
            return 100 + SPECIAL_TOKENS.index(name)
        def get_bos_token_id(self):
            return self.encode_special('<|bos|>')
    model = build(config()).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    sequences, masks = Engine(model, Tokenizer()).generate_batch([1, 2, 3], num_samples=3, max_tokens=4, temperature=0)
    assert sequences == [[1, 2, 3, 0, 0, 0, 0]] * 3
    assert masks == [[0, 0, 0, 1, 1, 1, 1]] * 3


def test_flops_and_cache_metrics_follow_latent_dimensions():
    model = build(config())
    model.window_sizes = [(-1, 0), (0, 0), (3, 0)]
    cfg = model.config
    params = model.num_matmul_params(active=True)
    pairs = sum(sum(min(t + 1, w + 1) if w >= 0 else t + 1 for t in range(8)) for w, _ in model.window_sizes)
    pair_flops = 2 * cfg.n_head * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
    assert model.estimate_prefill_flops(8) == 2 * params * 8 + pair_flops * pairs
    assert model.estimate_flops() == 3 * model.estimate_prefill_flops(cfg.sequence_len) / cfg.sequence_len
    attended = 8 + 1 + 4
    width = 2 * cfg.kv_lora_rank + cfg.qk_rope_head_dim
    assert model.estimate_decode_flops(8) == 2 * params + 2 * cfg.n_head * width * attended
    assert model.kv_read_bytes(8) == width * 4 * attended
    assert model.kv_bytes_per_token() == cfg.n_layer * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 4


def test_legacy_checkpoint_without_mla_fields_loads(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from nanochat import checkpoint_manager as manager
    cfg = config(attention_type='gqa')
    original = build(cfg).eval()
    legacy = asdict(cfg)
    for name in ('attention_type', 'q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim'):
        legacy.pop(name)
    manager.save_checkpoint(tmp_path, 1, original.state_dict(), None, {'model_config': legacy})
    monkeypatch.setattr(manager, 'get_tokenizer', lambda: SimpleNamespace(get_vocab_size=lambda: cfg.vocab_size))
    loaded, _, _ = manager.build_model(tmp_path, 1, torch.device('cpu'), 'eval')
    ids = torch.randint(0, cfg.vocab_size, (1, 7))
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids), original(ids))


def test_legacy_checkpoint_without_scalar_features_loads(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from nanochat import checkpoint_manager as manager
    cfg = config(attention_type='gqa')
    original = build(cfg).eval()
    with torch.no_grad(): # a model from before resid/x0/smear/backout existed
        original.resid_lambdas.fill_(1.0)
        original.x0_lambdas.zero_()
        original.smear_lambda.zero_()
        original.backout_lambda.zero_()
    legacy_keys = ('resid_lambdas', 'x0_lambdas', 'smear_gate.weight', 'smear_lambda', 'backout_lambda')
    state = {k: v for k, v in original.state_dict().items() if k not in legacy_keys}
    manager.save_checkpoint(tmp_path, 1, state, None, {'model_config': asdict(cfg)})
    monkeypatch.setattr(manager, 'get_tokenizer', lambda: SimpleNamespace(get_vocab_size=lambda: cfg.vocab_size))
    loaded, _, _ = manager.build_model(tmp_path, 1, torch.device('cpu'), 'eval')
    ids = torch.randint(0, cfg.vocab_size, (1, 7))
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids), original(ids))
    # assign=True keeps patched tensors where they are created, so they must follow the checkpoint device
    patched = {'lm_head.weight': torch.empty(2, device='meta')}
    manager._patch_missing_keys(patched, cfg)
    assert all(v.device.type == 'meta' for v in patched.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_bfloat16_cached_decode(monkeypatch):
    monkeypatch.setattr('nanochat.gpt.COMPUTE_DTYPE', torch.bfloat16)
    cfg = config(qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32, kv_lora_rank=32)
    with torch.device('meta'):
        model = GPT(cfg)
    model.to_empty(device='cuda')
    model.init_weights()
    model.eval()
    with torch.no_grad():
        for block in model.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.04)
        ids = torch.randint(0, 128, (1, 11), device='cuda')
        expected = model(ids)
        cache = KVCache.from_config(cfg, 1, 11, 'cuda', torch.bfloat16)
        actual = torch.cat([model(ids[:, :5], kv_cache=cache)] +
                           [model(ids[:, t:t + 1], kv_cache=cache) for t in range(5, 11)], 1)
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.004)
