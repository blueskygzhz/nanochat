"""
Tests for the DeepSeek-V2 style MoE implementation in nanochat/gpt.py.

Runs on CPU (attention falls back to SDPA), no GPU required:
    python -m pytest tests/test_moe.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from nanochat.gpt import (
    GPT, GPTConfig, MLP, MoE, MoEGate, is_moe_layer, moe_intermediate_size,
)


def make_config(**kw):
    """Small model so tests stay fast. Defaults to a MoE config."""
    base = dict(
        sequence_len=64, vocab_size=128, n_layer=4, n_head=2, n_kv_head=2, n_embd=64,
        window_pattern="L",
        n_routed_experts=8, n_shared_experts=2, num_experts_per_tok=2,
        moe_intermediate_mult=0.6875, first_k_dense_replace=1, moe_layer_freq=1,
        aux_loss_alpha=0.001, seq_aux=True,
    )
    base.update(kw)
    return GPTConfig(**base)


def build_model(config):
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


# ---------------------------------------------------------------- layer placement

def test_layer_placement_follows_deepseek_rule():
    config = make_config(n_layer=6, first_k_dense_replace=1, moe_layer_freq=1)
    # layer 0 dense, layers 1..5 MoE
    assert [is_moe_layer(i, config) for i in range(6)] == [False, True, True, True, True, True]

    # moe_layer_freq=2 => only even layers (and >= first_k_dense_replace) are MoE
    config = make_config(n_layer=6, first_k_dense_replace=1, moe_layer_freq=2)
    assert [is_moe_layer(i, config) for i in range(6)] == [False, False, True, False, True, False]


def test_dense_model_is_unchanged_when_moe_disabled():
    config = make_config(n_routed_experts=0)
    model = build_model(config)
    assert all(isinstance(b.mlp, MLP) for b in model.transformer.h)
    assert model.collect_aux_loss() is None
    # active == total params for a dense model
    counts = model.num_scaling_params()
    assert counts['transformer_matrices'] == counts['transformer_matrices_active']
    assert model.num_matmul_params(active=True) == model.num_matmul_params()


def test_moe_blocks_are_constructed_correctly():
    config = make_config(n_layer=4)
    model = build_model(config)
    assert isinstance(model.transformer.h[0].mlp, MLP)       # first_k_dense_replace=1
    for i in (1, 2, 3):
        mlp = model.transformer.h[i].mlp
        assert isinstance(mlp, MoE)
        assert len(mlp.experts) == config.n_routed_experts
        assert mlp.shared_experts is not None
        inter = moe_intermediate_size(config)
        assert mlp.experts[0].c_fc.weight.shape == (inter, config.n_embd)
        # shared experts are merged into one wider MLP (DeepSeek behaviour)
        assert mlp.shared_experts.c_fc.weight.shape == (inter * config.n_shared_experts, config.n_embd)


# ---------------------------------------------------------------- gate / routing

def test_gate_shapes_and_topk_normalization():
    config = make_config(num_experts_per_tok=3, norm_topk_prob=True)
    gate = MoEGate(config)
    torch.nn.init.uniform_(gate.weight, -0.1, 0.1)
    gate.eval()
    x = torch.randn(2, 5, config.n_embd)
    idx, w, aux = gate(x)
    assert idx.shape == (10, 3) and w.shape == (10, 3)
    assert idx.min() >= 0 and idx.max() < config.n_routed_experts
    # norm_topk_prob=True => weights sum to 1 per token
    torch.testing.assert_close(w.sum(-1), torch.ones(10), rtol=1e-5, atol=1e-5)
    assert aux is None  # eval mode => no aux loss


def test_routed_scaling_factor_applies_when_not_normalizing():
    config = make_config(num_experts_per_tok=2, norm_topk_prob=False, routed_scaling_factor=2.5)
    gate = MoEGate(config)
    torch.nn.init.uniform_(gate.weight, -0.1, 0.1)
    x = torch.randn(2, 4, config.n_embd)
    _, w_scaled, _ = gate(x)

    gate.routed_scaling_factor = 1.0
    _, w_plain, _ = gate(x)
    torch.testing.assert_close(w_scaled, w_plain * 2.5)


def test_group_limited_greedy_restricts_experts_to_selected_groups():
    # 8 experts in 4 groups of 2; keep only 1 group => the 2 chosen experts must share a group
    config = make_config(n_routed_experts=8, n_group=4, topk_group=1,
                         num_experts_per_tok=2, topk_method="group_limited_greedy")
    gate = MoEGate(config)
    torch.nn.init.uniform_(gate.weight, -1.0, 1.0)
    x = torch.randn(3, 7, config.n_embd)
    idx, _, _ = gate(x)
    group_of = idx // (config.n_routed_experts // config.n_group)
    # every token's experts come from a single group
    assert (group_of[:, 0] == group_of[:, 1]).all()


def test_gate_rejects_bad_config():
    with pytest.raises(AssertionError):
        MoEGate(make_config(n_routed_experts=4, num_experts_per_tok=8))      # top_k > n_experts
    with pytest.raises(AssertionError):
        MoEGate(make_config(n_routed_experts=8, n_group=3, topk_method="group_limited_greedy"))
    with pytest.raises(NotImplementedError):
        gate = MoEGate(make_config(topk_method="greedy"))
        gate.topk_method = "nope"
        gate(torch.randn(1, 2, 64))


# ---------------------------------------------------------------- dispatch correctness

def test_dispatch_matches_explicit_per_token_reference():
    """The vectorized index_add_ dispatch must equal a naive per-token loop."""
    torch.manual_seed(0)
    config = make_config(n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=0,
                         norm_topk_prob=True)
    moe = MoE(config)
    for e in moe.experts:
        torch.nn.init.normal_(e.c_fc.weight, std=0.2)
        torch.nn.init.normal_(e.c_proj.weight, std=0.2)  # non-zero so output is non-trivial
    torch.nn.init.uniform_(moe.gate.weight, -0.5, 0.5)

    x = torch.randn(2, 6, config.n_embd)
    got = moe(x)

    # Reference: recompute routing, then run each token through its experts one at a time
    idx, w, _ = moe.gate(x)
    x_flat = x.view(-1, config.n_embd)
    want = torch.zeros_like(x_flat)
    for n in range(x_flat.shape[0]):
        for slot in range(config.num_experts_per_tok):
            e = idx[n, slot].item()
            want[n] += w[n, slot].to(x.dtype) * moe.experts[e](x_flat[n:n + 1])[0]
    torch.testing.assert_close(got.view(-1, config.n_embd), want, rtol=1e-4, atol=1e-5)


def test_shared_experts_are_added_on_top_of_routed_output():
    torch.manual_seed(0)
    config = make_config(n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=2)
    moe = MoE(config)
    for m in list(moe.experts) + [moe.shared_experts]:
        torch.nn.init.normal_(m.c_fc.weight, std=0.2)
        torch.nn.init.normal_(m.c_proj.weight, std=0.2)
    torch.nn.init.uniform_(moe.gate.weight, -0.5, 0.5)

    x = torch.randn(2, 5, config.n_embd)
    with_shared = moe(x)
    shared_only = moe.shared_experts(x)
    moe.shared_experts = None
    routed_only = moe(x)
    torch.testing.assert_close(with_shared, routed_only + shared_only, rtol=1e-4, atol=1e-5)


def test_only_topk_experts_receive_gradient():
    """A token must not touch experts it wasn't routed to."""
    torch.manual_seed(0)
    config = make_config(n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=0,
                         aux_loss_alpha=0.0)
    moe = MoE(config)
    for e in moe.experts:
        torch.nn.init.normal_(e.c_fc.weight, std=0.2)
        torch.nn.init.normal_(e.c_proj.weight, std=0.2)
    torch.nn.init.uniform_(moe.gate.weight, -0.5, 0.5)

    x = torch.randn(1, 1, config.n_embd)  # single token
    idx, _, _ = moe.gate(x)
    chosen = set(idx[0].tolist())
    moe(x).sum().backward()

    touched = {e for e, expert in enumerate(moe.experts)
               if expert.c_fc.weight.grad is not None and expert.c_fc.weight.grad.abs().sum() > 0}
    assert touched == chosen, f"routed to {chosen} but gradients reached {touched}"


# ---------------------------------------------------------------- auxiliary loss

def test_aux_loss_only_in_training_mode():
    config = make_config(n_routed_experts=4, num_experts_per_tok=2, aux_loss_alpha=0.01)
    moe = MoE(config)
    torch.nn.init.uniform_(moe.gate.weight, -0.5, 0.5)
    x = torch.randn(2, 4, config.n_embd)

    moe.eval()
    moe(x)
    assert moe.aux_loss is None

    moe.train()
    moe(x)
    assert moe.aux_loss is not None and moe.aux_loss.requires_grad

    # alpha=0 disables it even in train mode
    moe.gate.alpha = 0.0
    moe(x)
    assert moe.aux_loss is None


def test_aux_loss_is_minimized_by_balanced_routing():
    """Global aux loss equals alpha*E*sum(P_i*f_i); perfectly balanced routing hits its floor alpha."""
    config = make_config(n_routed_experts=4, num_experts_per_tok=1, seq_aux=False, aux_loss_alpha=1.0)
    gate = MoEGate(config)
    gate.train()
    n_embd, E = config.n_embd, config.n_routed_experts

    # Build a gate whose routing is exactly uniform: identical rows => all scores equal
    torch.nn.init.zeros_(gate.weight)
    x = torch.randn(1, 64, n_embd)
    _, _, aux_uniform = gate(x)
    # balanced: P_i = 1/E, f_i = E * (1/E) = 1 => alpha * sum(1/E * 1) ... = alpha
    torch.testing.assert_close(aux_uniform, torch.tensor(1.0), rtol=1e-4, atol=1e-4)

    # Now force everything to one expert: scores one-hot on expert 0 => aux = alpha * E
    with torch.no_grad():
        gate.weight.zero_()
        gate.weight[0] = 50.0  # huge logit for expert 0 regardless of input sign
    x_pos = torch.rand(1, 64, n_embd)  # positive inputs => expert 0 wins
    _, _, aux_collapsed = gate(x_pos)
    assert aux_collapsed > aux_uniform
    torch.testing.assert_close(aux_collapsed, torch.tensor(float(E)), rtol=1e-3, atol=1e-3)


def test_seq_aux_and_global_aux_both_produce_scalars():
    x = torch.randn(3, 8, 64)
    for seq_aux in (True, False):
        gate = MoEGate(make_config(n_routed_experts=4, num_experts_per_tok=2,
                                   seq_aux=seq_aux, aux_loss_alpha=0.01))
        torch.nn.init.uniform_(gate.weight, -0.5, 0.5)
        gate.train()
        _, _, aux = gate(x)
        assert aux.ndim == 0 and torch.isfinite(aux)


def test_aux_loss_reaches_router_gradient():
    config = make_config(n_routed_experts=4, num_experts_per_tok=2, aux_loss_alpha=1.0)
    gate = MoEGate(config)
    torch.nn.init.uniform_(gate.weight, -0.5, 0.5)
    gate.train()
    _, _, aux = gate(torch.randn(2, 8, config.n_embd))
    aux.backward()
    assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------- full model integration

def test_full_model_forward_backward_and_loss_includes_aux():
    torch.manual_seed(0)
    config = make_config(n_layer=4, aux_loss_alpha=1.0)  # big alpha so the effect is visible
    model = build_model(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    loss = model(idx, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)

    aux = model.collect_aux_loss()
    assert aux is not None
    # loss returned == cross entropy + aux, so subtracting aux must leave a plain CE value
    assert loss.item() > aux.item()

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    # routers must have learned something from the aux loss alone
    for block in model.transformer.h:
        if isinstance(block.mlp, MoE):
            assert block.mlp.gate.weight.grad.abs().sum() > 0


def test_aux_loss_not_added_for_non_mean_reduction():
    """loss_reduction='none' must stay per-token: a scalar aux would broadcast and corrupt it."""
    config = make_config(n_layer=3, aux_loss_alpha=1.0)
    model = build_model(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    per_token = model(idx, targets, loss_reduction='none')
    assert per_token.shape == (16,)
    # compare against a manual CE to make sure nothing was added
    logits = model(idx)
    want = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                           ignore_index=-1, reduction='none')
    torch.testing.assert_close(per_token, want, rtol=1e-4, atol=1e-5)


def test_eval_mode_forward_has_no_aux_and_is_deterministic():
    config = make_config(n_layer=3)
    model = build_model(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 12))
    with torch.no_grad():
        a = model(idx)
        b = model(idx)
    assert model.collect_aux_loss() is None
    torch.testing.assert_close(a, b)


def test_parameter_accounting_and_optimizer_grouping():
    config = make_config(n_layer=4)
    model = build_model(config)

    counts = model.num_scaling_params()  # contains the internal total==sum assert
    # MoE keeps only top_k of n_routed_experts active => active < total
    assert counts['transformer_matrices_active'] < counts['transformer_matrices']
    assert model.num_matmul_params(active=True) < model.num_matmul_params()
    # FLOPs must be based on the active (sparse) count
    assert model.estimate_flops() < 6 * model.num_matmul_params()

    # setup_optimizer asserts full parameter coverage internally
    optimizer = model.setup_optimizer()
    seen = sum(len(g['params']) for g in optimizer.param_groups)
    assert seen == len(list(model.parameters()))

    # routers must be in an AdamW group, never in a Muon group
    router_ids = {id(b.mlp.gate.weight) for b in model.transformer.h if isinstance(b.mlp, MoE)}
    assert len(router_ids) == 3  # layers 1,2,3
    for g in optimizer.param_groups:
        if g['kind'] == 'muon':
            assert not (router_ids & {id(p) for p in g['params']})
    adamw_ids = {id(p) for g in optimizer.param_groups if g['kind'] == 'adamw' for p in g['params']}
    assert router_ids <= adamw_ids


def test_optimizer_step_updates_moe_params():
    torch.manual_seed(0)
    config = make_config(n_layer=3)
    model = build_model(config)
    model.train()
    optimizer = model.setup_optimizer()

    moe = next(b.mlp for b in model.transformer.h if isinstance(b.mlp, MoE))
    router_before = moe.gate.weight.detach().clone()
    # NOTE: c_proj is zero-initialized (repo-wide convention), so on the very first step
    # c_fc receives exactly zero gradient. c_proj is what moves first.
    proj_before = moe.experts[0].c_proj.weight.detach().clone()
    fc_before = moe.experts[0].c_fc.weight.detach().clone()

    def train_step():
        idx = torch.randint(0, config.vocab_size, (2, 16))
        targets = torch.randint(0, config.vocab_size, (2, 16))
        model(idx, targets).backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)

    train_step()
    assert not torch.equal(router_before, moe.gate.weight), "router did not update"
    assert not torch.equal(proj_before, moe.experts[0].c_proj.weight), "expert c_proj did not update"

    train_step()  # now c_proj != 0, so gradient can flow back into c_fc
    assert not torch.equal(fc_before, moe.experts[0].c_fc.weight), "expert c_fc did not update"


def test_optimizer_handles_experts_that_received_no_tokens():
    """An unrouted expert is absent from the graph (p.grad is None); the optimizer must cope.

    Regression test: Muon stacks p.grad across a shape group, which used to crash on None.
    """
    torch.manual_seed(0)
    # Many experts, very few tokens => several experts are guaranteed to get nothing
    config = make_config(n_layer=2, n_routed_experts=16, num_experts_per_tok=1, n_shared_experts=0)
    model = build_model(config)
    model.train()
    optimizer = model.setup_optimizer()

    idx = torch.randint(0, config.vocab_size, (1, 4))  # only 4 tokens for 16 experts
    targets = torch.randint(0, config.vocab_size, (1, 4))
    model(idx, targets).backward()

    moe = next(b.mlp for b in model.transformer.h if isinstance(b.mlp, MoE))
    missing = [e for e, ex in enumerate(moe.experts) if ex.c_fc.weight.grad is None]
    assert missing, "test setup failed: expected at least one expert with no gradient"

    optimizer.step()  # must not raise
    model.zero_grad(set_to_none=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_init_weights_zeroes_expert_output_projections():
    config = make_config(n_layer=3)
    model = build_model(config)
    for block in model.transformer.h:
        if isinstance(block.mlp, MoE):
            for e in list(block.mlp.experts) + [block.mlp.shared_experts]:
                assert e.c_proj.weight.abs().sum() == 0      # zero-init output proj, like dense MLP
                assert e.c_fc.weight.abs().sum() > 0
            # router is small-random (kaiming a=sqrt(5) equivalent), not zero
            w = block.mlp.gate.weight
            assert w.abs().sum() > 0 and w.abs().max() <= config.n_embd ** -0.5 + 1e-6


def test_activation_checkpointing_matches_plain_backward():
    """Recomputing blocks in backward must give identical loss and gradients (routing is deterministic)."""
    torch.manual_seed(0)
    config = make_config(n_layer=4, aux_loss_alpha=0.01)
    model = build_model(config)
    for block in model.transformer.h:  # make c_proj non-zero so every param gets a real gradient
        mlps = list(block.mlp.experts) + [block.mlp.shared_experts] if isinstance(block.mlp, MoE) else [block.mlp]
        for m in mlps:
            torch.nn.init.normal_(m.c_proj.weight, std=0.05)
        torch.nn.init.normal_(block.attn.c_proj.weight, std=0.05)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    def run(ckpt):
        model.zero_grad(set_to_none=True)
        model.activation_checkpointing = ckpt
        loss = model(idx, targets)
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return loss.detach(), grads

    loss_a, grads_a = run(False)
    loss_b, grads_b = run(True)
    torch.testing.assert_close(loss_a, loss_b)
    assert grads_a.keys() == grads_b.keys()
    for n in grads_a:
        torch.testing.assert_close(grads_a[n], grads_b[n], rtol=1e-4, atol=1e-6, msg=n)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_chunked_loss_matches_full_loss_and_gradients(reduction, checkpointing):
    torch.manual_seed(7)
    model = build_model(make_config(n_layer=2, vocab_size=131))
    model.activation_checkpointing = checkpointing
    for p in model.parameters():
        if p.ndim == 2:
            torch.nn.init.normal_(p, std=0.05)
    idx = torch.randint(0, 131, (2, 8))
    targets = torch.randint(0, 131, (2, 8))
    targets[0, :5] = -1

    def run(chunk_size):
        model.zero_grad(set_to_none=True)
        model.loss_chunk_size = chunk_size
        loss = model(idx, targets, loss_reduction=reduction)
        loss.sum().backward()
        return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    expected, grads = run(0)
    actual, chunk_grads = run(5)
    torch.testing.assert_close(actual, expected)
    assert grads.keys() == chunk_grads.keys()
    for name in grads:
        torch.testing.assert_close(chunk_grads[name], grads[name], rtol=2e-4, atol=2e-6, msg=name)
    model.eval()
    with torch.no_grad():
        model.loss_chunk_size = 0
        expected_eval = model(idx, targets, loss_reduction=reduction)
        model.loss_chunk_size = 5
        torch.testing.assert_close(model(idx, targets, loss_reduction=reduction), expected_eval)


def test_fp8_conversion_preserves_flop_counts():
    from nanochat.fp8 import convert_to_float8_training
    model = build_model(make_config(n_layer=2))
    before = (model.num_matmul_params(), model.num_matmul_params(active=True), model.estimate_flops(),
              model.estimate_decode_flops(16), model.estimate_prefill_flops(16))
    convert_to_float8_training(model)
    after = (model.num_matmul_params(), model.num_matmul_params(active=True), model.estimate_flops(),
             model.estimate_decode_flops(16), model.estimate_prefill_flops(16))
    assert after == before


@pytest.mark.parametrize('kw', [dict(num_experts_per_tok=0), dict(n_group=0),
                                dict(topk_group=0), dict(n_group=4, topk_group=1, num_experts_per_tok=3)])
def test_invalid_router_config_fails_early(kw):
    with pytest.raises(AssertionError):
        MoEGate(make_config(topk_method='group_limited_greedy', **kw))


def test_global_aux_matches_one_hot_value_and_gradient():
    gate = MoEGate(make_config(seq_aux=False))
    torch.nn.init.normal_(gate.weight, std=0.05)
    x = torch.randn(2, 9, 64, requires_grad=True)
    idx, _, actual = gate(x)
    scores = F.linear(x.reshape(-1, 64), gate.weight).softmax(-1)
    fractions = F.one_hot(idx.reshape(-1), gate.n_routed_experts).float().mean(0)
    expected = gate.alpha * gate.n_routed_experts * (scores.mean(0) * fractions).sum()
    torch.testing.assert_close(actual, expected)
    got = torch.autograd.grad(actual, (x, gate.weight), retain_graph=True)
    want = torch.autograd.grad(expected, (x, gate.weight))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b)


def test_bucketed_optimizer_matches_unbucketed_and_resumes(monkeypatch):
    import copy
    import nanochat.optim as optim
    monkeypatch.setattr(optim, 'adamw_step_fused', optim.adamw_step_fused._torchdynamo_orig_callable)
    monkeypatch.setattr(optim, 'muon_step_fused', optim.muon_step_fused._torchdynamo_orig_callable)
    torch.manual_seed(9)
    plain = build_model(make_config(n_layer=2))
    bucketed = copy.deepcopy(plain)
    a = plain.setup_optimizer()
    b = bucketed.setup_optimizer(muon_bucket_mb=0.04)
    assert b.memory_efficient
    assert len(b.param_groups) > len(a.param_groups)
    for _ in range(3):
        for i, (p, q) in enumerate(zip(plain.parameters(), bucketed.parameters())):
            grad = None if i % 7 == 0 else torch.randn_like(p)
            p.grad = grad
            q.grad = None if grad is None else grad.clone()
        a.step()
        b.step()
        for p, q in zip(plain.parameters(), bucketed.parameters()):
            torch.testing.assert_close(p, q)
        restored = bucketed.setup_optimizer(muon_bucket_mb=0.04)
        restored.load_state_dict(copy.deepcopy(b.state_dict()))
        b = restored


def test_generate_works_with_moe():
    config = make_config(n_layer=3)
    model = build_model(config)
    model.eval()
    out = list(model.generate([1, 2, 3], max_tokens=4, temperature=0))
    assert len(out) == 4 and all(0 <= t < config.vocab_size for t in out)
