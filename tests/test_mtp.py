"""MTP supervision and single-draft greedy speculative decoding regressions."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nanochat import checkpoint_manager, engine as engine_module, optim
from nanochat.engine import Engine, KVCache, MLAKVCache
from nanochat.gpt import GPT, GPTConfig
from scripts.infer_bench import bench_speculative


class Tokenizer:
    names = ('<|python_start|>', '<|python_end|>', '<|output_start|>', '<|output_end|>',
             '<|assistant_end|>', '<|bos|>', '<|user_start|>', '<|user_end|>', '<|assistant_start|>')

    def encode_special(self, name):
        return 256 + self.names.index(name)

    def get_bos_token_id(self):
        return self.encode_special('<|bos|>')

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode()

    def get_vocab_size(self):
        return 272


@pytest.fixture(autouse=True)
def reference_dtype(monkeypatch):
    monkeypatch.setattr('nanochat.gpt.COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr('nanochat.engine.COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr('nanochat.optim.COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr(optim, 'adamw_step_fused', optim.adamw_step_fused._torchdynamo_orig_callable)
    monkeypatch.setattr(optim, 'muon_step_fused', optim.muon_step_fused._torchdynamo_orig_callable)


def make_model(attention='gqa', mtp=True, moe=False):
    cfg = GPTConfig(sequence_len=64, vocab_size=272, n_layer=2, n_head=2, n_kv_head=1, n_embd=64,
                    attention_type=attention, window_pattern='L', kv_lora_rank=16, q_lora_rank=16,
                    qk_nope_head_dim=12, qk_rope_head_dim=8, v_head_dim=10,
                    n_routed_experts=4 if moe else 0, num_experts_per_tok=2, n_shared_experts=1,
                    mtp_enabled=mtp, mtp_loss_weight=0.2, mtp_bos_token_id=261)
    torch.manual_seed(123)
    with torch.device('meta'):
        model = GPT(cfg)
    model.to_empty(device='cpu')
    model.init_weights()
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2:
                torch.nn.init.normal_(p, std=0.07)
        model.smear_lambda.fill_(0.5)
    return model


@pytest.mark.parametrize('attention', ['gqa', 'mla'])
def test_mtp_loss_matches_explicit_row_shift_and_backprop(attention):
    model = make_model(attention, moe=True)
    rows = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [11, 12, 13, 14, 15, 16, 17]])
    idx, targets = rows[:, :-1], rows[:, 1:].clone()
    targets[0, :2] = -1
    hidden = model.forward_hidden(idx)
    losses = []
    for b in range(2):
        for t in range(idx.size(1) - 1):
            if targets[b, t] >= 0 and targets[b, t + 1] >= 0:
                logits = model.mtp_logits(hidden[b, t:t + 1], idx[b, t + 1:t + 2])
                losses.append(F.cross_entropy(logits, targets[b, t + 1:t + 2]))
    expected = torch.stack(losses).mean()
    actual = model._compute_mtp_loss(hidden, idx, targets)
    torch.testing.assert_close(actual, expected)
    total = model(idx, targets)
    aux = model.collect_aux_loss()
    mtp = model.collect_mtp_loss()
    ce = F.cross_entropy(model._compute_logits(hidden).reshape(-1, 272), targets.reshape(-1), ignore_index=-1)
    torch.testing.assert_close(total, ce + aux + expected * 0.2)
    torch.testing.assert_close(mtp, expected.detach() * 0.2)
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.mtp.parameters())
    before = model.mtp.fuse.weight.detach().clone()
    optimizer = model.setup_optimizer(muon_bucket_mb=1)
    seen = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(seen) == len(set(seen)) == len(list(model.parameters()))
    optimizer.step()
    assert not torch.equal(before, model.mtp.fuse.weight)


def test_mtp_masks_prompt_tool_padding_and_bos_boundaries():
    model = make_model()
    idx = torch.tensor([[1, 2, 3, 261, 4, 5], [11, 12, 13, 14, 15, 16]])
    targets = torch.tensor([[2, 3, 261, 4, 5, 6], [-1, 13, -1, 15, 16, -1]])
    expected = torch.tensor([[3, -1, -1, 5, 6], [-1, -1, -1, 16, -1]])
    assert torch.equal(model._mtp_targets(idx, targets), expected)
    hidden = model.forward_hidden(idx)
    empty = targets.fill_(-1)
    loss = model._compute_mtp_loss(hidden, idx, empty)
    assert loss.item() == 0 and loss.requires_grad
    loss.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.mtp.parameters())


@pytest.mark.parametrize('attention', ['gqa', 'mla'])
def test_chunked_mtp_checkpoint_and_gradient_accumulation(attention):
    plain = make_model(attention)
    chunked = copy.deepcopy(plain)
    chunked.loss_chunk_size = 5
    chunked.activation_checkpointing = True
    tokens = torch.randint(0, 255, (2, 10))
    x, y = tokens[:, :-1], tokens[:, 1:]
    a, b = plain(x, y), chunked(x, y)
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b)
    for p, q in zip(plain.parameters(), chunked.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=4e-4, atol=3e-6)
    chunked.zero_grad(set_to_none=True)
    for row in range(2):
        (chunked(x[row:row + 1], y[row:row + 1]) / 2).backward()
    for p, q in zip(plain.parameters(), chunked.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=4e-4, atol=3e-6)


@pytest.mark.parametrize('reduction', ['none', 'sum'])
def test_nonmean_and_eval_do_not_use_mtp(reduction):
    model = make_model()
    ids = torch.randint(0, 255, (1, 9))
    x, y = ids[:, :-1], ids[:, 1:]
    loss = model(x, y, loss_reduction=reduction)
    expected = F.cross_entropy(model(x).reshape(-1, 272), y.reshape(-1), reduction=reduction)
    torch.testing.assert_close(loss, expected)
    assert model.collect_mtp_loss() is None
    model(x, y)
    assert model.collect_mtp_loss() is not None
    model.eval()
    torch.testing.assert_close(model(x, y), F.cross_entropy(model(x).reshape(-1, 272), y.reshape(-1)))
    assert model.collect_mtp_loss() is None


@pytest.mark.parametrize('attention', ['gqa', 'mla'])
def test_mtp_counts_and_disabled_checkpoint_compatibility(attention, tmp_path, monkeypatch):
    enabled = make_model(attention)
    disabled = make_model(attention, mtp=False)
    assert enabled.num_mtp_params() == 10 * 64**2
    assert enabled.num_scaling_params()['total'] - disabled.num_scaling_params()['total'] == enabled.num_mtp_params()
    assert enabled.num_scaling_params()['transformer_matrices_active'] == disabled.num_scaling_params()['transformer_matrices_active']
    assert enabled.num_matmul_params(active=True) == disabled.num_matmul_params(active=True)
    assert enabled.estimate_decode_flops(8) == disabled.estimate_decode_flops(8)
    assert enabled.estimate_prefill_flops(8) == disabled.estimate_prefill_flops(8)
    expected_extra = 3 * enabled.estimate_mtp_flops() * 63 / 64
    assert enabled.estimate_flops() - disabled.estimate_flops() == expected_extra
    old_config = asdict(disabled.config)
    for key in ('mtp_enabled', 'mtp_loss_weight', 'mtp_bos_token_id'):
        old_config.pop(key)
    checkpoint_manager.save_checkpoint(tmp_path, 1, disabled.state_dict(), None, {'model_config': old_config})
    monkeypatch.setattr(checkpoint_manager, 'get_tokenizer', lambda: Tokenizer())
    loaded, _, _ = checkpoint_manager.build_model(tmp_path, 1, torch.device('cpu'), 'eval')
    disabled.eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(disabled(ids), loaded(ids))
    assert loaded.mtp is None
    assert not any(name.startswith('mtp.') for name in loaded.state_dict())


@pytest.mark.parametrize('attention', ['gqa', 'mla'])
@pytest.mark.parametrize('window', [-1, 0, 3])
def test_rollback_restores_cache_and_nonzero_smear(attention, window):
    model = make_model(attention).eval()
    model.window_sizes = [(window, 0)] * model.config.n_layer
    cache = KVCache.from_config(model.config, 1, 20, 'cpu', torch.float32)
    clean = KVCache.from_config(model.config, 1, 20, 'cpu', torch.float32)
    prefix = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        model(prefix, kv_cache=cache)
        model(prefix, kv_cache=clean)
        model(torch.tensor([[4, 99]]), kv_cache=cache)
        model(torch.tensor([[4]]), kv_cache=clean)
        cache.truncate(4, model.embed_tokens(torch.tensor([[4]])))
        assert cache.get_pos() == 4 and (cache.cache_seqlens == 4).all()
        torch.testing.assert_close(cache.prev_embedding, clean.prev_embedding)
        a = model(torch.tensor([[5, 6]]), kv_cache=cache)
        b = model(torch.tensor([[5, 6]]), kv_cache=clean)
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6)
        with pytest.raises(ValueError):
            cache.truncate(7)
        with pytest.raises(ValueError):
            cache.truncate(4)
        cache.truncate(0)
        assert cache.get_pos() == 0 and cache.prev_embedding is None


@pytest.mark.parametrize('attention', ['gqa', 'mla'])
@pytest.mark.parametrize('policy', ['model', 'oracle', 'wrong'])
def test_real_model_speculative_matches_baseline(attention, policy, monkeypatch):
    model = make_model(attention, moe=True).eval()
    model.window_sizes = [(3, 0), (-1, 0)]
    engine = Engine(model, Tokenizer())
    prompt = [10, 20, 30, 40]
    baseline = list(engine.generate(prompt, max_tokens=18, temperature=0))
    original_forward = model.forward_with_hidden
    state = {}
    def record(ids, kv_cache):
        pos = kv_cache.get_pos()
        state['tokens'] = state.get('tokens', [])[:pos] + ids[0].tolist()
        state['cache'] = kv_cache
        return original_forward(ids, kv_cache)
    monkeypatch.setattr(model, 'forward_with_hidden', record)
    if policy != 'model':
        def draft(hidden, next_ids):
            history = state['tokens'][:state['cache'].get_pos()] + next_ids[0].tolist()
            logits = model(torch.tensor([history]))[:, -1:]
            if policy == 'wrong':
                target = logits[0, 0].argmax().item()
                wrong = (target + 1) % 250
                logits = torch.full_like(logits, -100)
                logits[0, 0, wrong] = 100
            return logits
        monkeypatch.setattr(model, 'mtp_logits', draft)
    stats = {}
    actual = list(engine.generate(prompt, max_tokens=18, temperature=0, speculative=True, stats=stats))
    assert actual == baseline
    assert stats['generated_tokens'] == len(actual)
    if policy == 'wrong':
        assert stats['draft_tokens'] > 0 and stats['accepted_draft_tokens'] == 0
    if policy == 'oracle':
        assert stats['accepted_draft_tokens'] == stats['draft_tokens'] > 0


class ScriptedModel:
    """Deterministic target/draft stream to exercise every commit boundary."""
    training = False
    def __init__(self, output, drafts, attention):
        self.output = output
        self.drafts = drafts
        self.config = SimpleNamespace(mtp_enabled=True, sequence_len=64, attention_type=attention,
                                      n_head=2, n_kv_head=1, n_embd=8, n_layer=1,
                                      kv_lora_rank=4, qk_rope_head_dim=2)
        self.last_cache = None
        self.call_lengths = []

    def get_device(self):
        return torch.device('cpu')

    def embed_tokens(self, ids):
        return ids.float().unsqueeze(-1).expand(-1, -1, 8)

    def forward_with_hidden(self, ids, kv_cache):
        start = kv_cache.get_pos()
        self.last_cache = kv_cache
        self.call_lengths.append(ids.size(1))
        logits = torch.full((*ids.shape, 272), -100.)
        hidden = torch.zeros(*ids.shape, 8)
        for t in range(ids.size(1)):
            output_idx = start + t
            token = self.output[output_idx] if output_idx < len(self.output) else 260
            logits[0, t, token] = 100
            hidden[0, t, 0] = output_idx
        kv_cache.prev_embedding = self.embed_tokens(ids[:, -1:])
        kv_cache.advance(ids.size(1))
        return logits, hidden

    def forward(self, ids, kv_cache):
        return self.forward_with_hidden(ids, kv_cache)[0]

    def mtp_logits(self, hidden, next_ids):
        next_idx = int(hidden[0, -1, 0]) + 1
        token = self.output[next_idx] if next_idx < len(self.output) else 260
        token = self.drafts.get(next_idx, token)
        logits = torch.full((1, 1, 272), -100.)
        logits[0, 0, token] = 100
        return logits


@pytest.mark.parametrize('attention', ['gqa', 'mla'])
@pytest.mark.parametrize('budget', [0, 1, 2, 3, 4, 8])
@pytest.mark.parametrize('drafts', [{}, {1: 99, 2: 99, 3: 99}, {1: 99}])
def test_accept_reject_eos_and_budgets(attention, budget, drafts):
    model = ScriptedModel([10, 20, 30, 260], drafts, attention)
    engine = Engine(model, Tokenizer())
    expected = list(engine.generate([261], max_tokens=budget, temperature=0))
    stats = {}
    model.call_lengths.clear()
    actual = list(engine.generate([261], max_tokens=budget, temperature=0, speculative=True, stats=stats))
    assert actual == expected
    assert len(actual) <= budget
    assert stats['generated_tokens'] == len(actual)
    if budget == 0:
        assert stats['target_calls'] == 0
    elif budget == 1:
        assert stats['target_calls'] == 1
    if budget >= 4 and not drafts:
        assert stats['accepted_draft_tokens'] == 2
        assert stats['target_calls'] == 3
    if budget >= 4 and len(drafts) == 3:
        assert stats['accepted_draft_tokens'] == 0
    if budget >= 4 and drafts == {1: 99}:
        assert stats['accepted_draft_tokens'] == 1
    if budget > 0:
        assert model.last_cache.get_pos() <= 1 + len(actual)


@pytest.mark.parametrize('attention', ['gqa', 'mla'])
def test_tools_execute_once_only_after_commit(attention, monkeypatch):
    calls = []
    monkeypatch.setattr(engine_module, 'use_calculator', lambda text: calls.append(text) or '5')
    # A hallucinated tool-ending draft at index 1 must not execute a tool.
    output = [65, 66, 256, 50, 43, 51, 257, 99, 99, 99, 67, 260]
    model = ScriptedModel(output, {1: 257}, attention)
    engine = Engine(model, Tokenizer())
    baseline = list(engine.generate([261], temperature=0, max_tokens=20))
    assert calls == ['2+3']
    calls.clear()
    stats = {}
    actual = list(engine.generate([261], temperature=0, max_tokens=20, speculative=True, stats=stats))
    assert actual == baseline
    assert calls == ['2+3']
    assert [c[0] for c, m in actual if m == [0]] == [258, 53, 259]
    assert stats['forced_tokens'] == 3
    calls.clear()
    results, masks = engine.generate_batch([261], temperature=0, max_tokens=20, speculative=True)
    assert results[0][-1] == 67 and 260 not in results[0]
    assert masks[0].count(0) == 4  # prompt and three tool-output tokens
    assert calls == ['2+3']


def test_speculative_requires_mtp_greedy_single_row_eval():
    model = make_model(mtp=False).eval()
    with pytest.raises(ValueError, match='checkpoint'):
        list(Engine(model, Tokenizer()).generate([1, 2], temperature=0, speculative=True))
    model = make_model().eval()
    engine = Engine(model, Tokenizer())
    with pytest.raises(ValueError, match='temperature'):
        list(engine.generate([1, 2], temperature=0.7, speculative=True))
    with pytest.raises(ValueError, match='num_samples'):
        list(engine.generate([1, 2], num_samples=2, temperature=0, speculative=True))
    model.train()
    with pytest.raises(ValueError, match='eval'):
        list(engine.generate([1, 2], temperature=0, speculative=True))


@pytest.mark.parametrize('terminal', [260, 261])
@pytest.mark.parametrize('attention', ['gqa', 'mla'])
def test_rejected_terminal_draft_never_ends_output(terminal, attention):
    model = ScriptedModel([10, 20, 30, terminal], {1: terminal}, attention)
    engine = Engine(model, Tokenizer())
    stats = {}
    result = list(engine.generate([261], max_tokens=8, temperature=0, speculative=True, stats=stats))
    assert [column[0] for column, _ in result] == [10, 20, 30, terminal]
    assert stats['draft_tokens'] > stats['accepted_draft_tokens']
    model = ScriptedModel([terminal], {}, attention)
    stats = {}
    assert list(Engine(model, Tokenizer()).generate([261], max_tokens=8, temperature=0, speculative=True, stats=stats)) == [([terminal], [1])]
    assert stats['draft_calls'] == 0 and stats['target_calls'] == 1


def test_mtp_does_not_read_future_tokens():
    model = make_model().eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 4, 90, 91]])
    with torch.no_grad():
        # At t=2, only h_2 and the same x_3=4 are available; x_4 and beyond must not leak.
        a_logits = model.mtp_logits(model.forward_hidden(a)[:, 2:3], a[:, 3:4])
        b_logits = model.mtp_logits(model.forward_hidden(b)[:, 2:3], b[:, 3:4])
        torch.testing.assert_close(a_logits, b_logits)


@pytest.mark.parametrize('attention', ['gqa', 'mla'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_bfloat16_mtp_training_and_speculation(attention, monkeypatch):
    monkeypatch.setattr('nanochat.gpt.COMPUTE_DTYPE', torch.bfloat16)
    monkeypatch.setattr('nanochat.engine.COMPUTE_DTYPE', torch.bfloat16)
    model = make_model(attention).to('cuda')
    model.activation_checkpointing = True
    model.loss_chunk_size = 5
    ids = torch.randint(0, 255, (2, 10), device='cuda')
    loss = model(ids[:, :-1], ids[:, 1:])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.mtp.parameters())
    model.eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    engine = Engine(model, Tokenizer())
    baseline = list(engine.generate([1, 2, 3], max_tokens=10, temperature=0))
    stats = {}
    assert list(engine.generate([1, 2, 3], max_tokens=10, temperature=0, speculative=True, stats=stats)) == baseline
    assert stats['accepted_draft_tokens'] > 0


def test_speculative_benchmark_uses_whole_stream():
    model = ScriptedModel([10, 20, 30, 260], {}, 'mla')
    result = bench_speculative(Engine(model, Tokenizer()), [261], 8, True)
    assert result['output_ids'] == [10, 20, 30, 260]
    assert result['generated_tokens'] == 4
    assert result['acceptance_rate'] == 1
    assert result['stats']['target_calls'] == 3
    assert result['elapsed_sec'] >= result['ttft_sec'] > 0
    assert result['tokens_per_sec'] == 4 / result['elapsed_sec']
