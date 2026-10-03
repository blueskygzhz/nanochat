"""
Tests for `nanochat.scratch`, the whole training stack.

There is no torch in this repo any more, so there is no autograd oracle to compare
against. Gradient correctness is instead established from first principles, which is
actually the stronger claim: for every operation we check that the analytic backward
equals a central finite difference of the forward.

    df/dx ~= (f(x + h) - f(x - h)) / 2h     error O(h^2)

Finite differences in float32 are useless (the roundoff floor swamps the signal), so
`float64_engine` flips the engine's element type to float64 for those tests.

Layers of verification, weakest to strongest:
  1. op-level forward values vs closed-form numpy expressions
  2. op-level gradients vs central finite differences
  3. whole-model gradients vs central finite differences (dense and MoE)
  4. optimizer updates vs hand-computed reference steps
  5. the thing actually learns: loss reaches a known entropy floor
"""

import importlib.util  # noqa: F401  (kept: used by test_bpe-style spec probes)
import math

import numpy as np
import pytest

import nanochat.scratch.nn as snn
from nanochat.scratch import tensor as st
from nanochat.scratch.data import (
    ADDITION_LINE_LEN, ByteTokenizer, Dataset, addition_entropy_floor, make_addition_corpus,
)
from nanochat.scratch.model import GPT, GPTConfig, MoE
from nanochat.scratch.optim import AdamW, Muon, polar_express, setup_optimizer
from nanochat.scratch.tensor import Tensor


def test_no_torch_distribution_is_installed():
    """torch must not be an installed distribution.

    Note this deliberately does not use `importlib.util.find_spec`: an uninstall can
    leave an empty directory behind in site-packages, which Python then reports as a
    namespace package even though no torch code exists. The distribution metadata is
    the authoritative answer.
    """
    import importlib.metadata as md
    for name in ("torch", "kernels", "wandb"):
        with pytest.raises(md.PackageNotFoundError):
            md.distribution(name)


def test_no_module_imports_torch():
    """The real regression guard: no source file may import torch.

    If someone re-adds `import torch` anywhere, this fails loudly instead of
    silently reintroducing the dependency.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(r"^\s*(?:import\s+torch|from\s+torch)\b", re.MULTILINE)
    offenders = []
    for path in root.rglob("*.py"):
        # Skip the venv and any cache/dot directory (e.g. leftover .pytest_cache
        # inductor artifacts, which are generated files, not project source).
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"these files import torch: {offenders}"


def test_scratch_package_imports_only_numpy():
    """nanochat.scratch must depend on nothing but numpy and the standard library.

    Checked against `sys.stdlib_module_names` rather than a hand-maintained allowlist,
    so adding a stdlib import does not require editing this test, while adding a new
    third-party dependency still fails.
    """
    import pathlib
    import re
    import sys
    root = pathlib.Path(__file__).resolve().parent.parent / "nanochat" / "scratch"
    pattern = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w.]*)", re.MULTILINE)
    allowed_third_party = {"numpy"}
    offenders = []
    for path in sorted(root.glob("*.py")):
        for mod in pattern.findall(path.read_text(encoding="utf-8")):
            top = mod.split(".")[0]
            if top in ("nanochat", *allowed_third_party) or top in sys.stdlib_module_names:
                continue
            offenders.append(f"{path.name}: {mod}")
    assert not offenders, f"non-stdlib, non-numpy imports: {offenders}"


@pytest.fixture
def float64_engine():
    """Run the engine in float64 so finite differences are meaningful."""
    st.set_dtype(np.float64)
    yield
    st.set_dtype(np.float32)


# ----------------------------------------------------------------------------
# gradient checking machinery

def _grad_check(fn, shapes, positive=False, seed=0, n_probes=25, eps=1e-6, tol=1e-6):
    """Verify fn's analytic gradients against central finite differences.

    The output is reduced to a scalar with fixed random weights, so the check covers
    every output element rather than just their sum.
    """
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(s) for s in shapes]
    if positive:
        arrays = [np.abs(a) + 0.5 for a in arrays]

    inputs = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    probe = rng.standard_normal(fn(*inputs).shape)

    def loss_of(tensors):
        return float((fn(*tensors).data * probe).sum())

    out = fn(*inputs)
    (out * Tensor(probe)).sum().backward()
    analytic = [t.grad.copy() if t.grad is not None else None for t in inputs]

    for i, arr in enumerate(arrays):
        assert analytic[i] is not None, f"input {i} received no gradient"
        flat = arr.size
        idxs = rng.choice(flat, size=min(n_probes, flat), replace=False)
        for flat_idx in idxs:
            idx = np.unravel_index(flat_idx, arr.shape)
            probes = [Tensor(a.copy()) for a in arrays]
            original = arr[idx]

            probes[i].data[idx] = original + eps
            plus = loss_of(probes)
            probes[i].data[idx] = original - eps
            minus = loss_of(probes)

            numeric = (plus - minus) / (2 * eps)
            expected = analytic[i][idx]
            scale = max(abs(numeric), abs(expected), 1e-4)
            assert abs(numeric - expected) / scale < tol, (
                f"input {i}{idx}: analytic {expected:.12f} vs numeric {numeric:.12f}")


def _check_forward(fn, shapes, reference, positive=False, seed=0):
    """Check an op's forward values against a closed-form numpy expression."""
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(s) for s in shapes]
    if positive:
        arrays = [np.abs(a) + 0.5 for a in arrays]
    got = fn(*[Tensor(a.copy()) for a in arrays])
    want = reference(*arrays)
    assert tuple(got.shape) == tuple(np.shape(want)), f"{got.shape} != {np.shape(want)}"
    np.testing.assert_allclose(got.data, want, atol=1e-10, rtol=1e-8)


# name, fn, shapes, positive
OPS = [
    ("add",               lambda a, b: a + b,                       [(3, 4), (3, 4)], False),
    ("add_broadcast",     lambda a, b: a + b,                       [(2, 3, 4), (4,)], False),
    ("add_broadcast_mid", lambda a, b: a + b,                       [(2, 3, 4), (1, 3, 1)], False),
    ("sub",               lambda a, b: a - b,                       [(3, 4), (3, 4)], False),
    ("rsub",              lambda a: 2.0 - a,                        [(3, 4)], False),
    ("mul",               lambda a, b: a * b,                       [(3, 4), (3, 4)], False),
    ("mul_broadcast",     lambda a, b: a * b,                       [(2, 3, 4), (3, 1)], False),
    ("mul_scalar",        lambda a: a * 3.5,                        [(3, 4)], False),
    ("div",               lambda a, b: a / b,                       [(3, 4), (3, 4)], True),
    ("rdiv",              lambda a: 2.0 / a,                        [(3, 4)], True),
    ("pow",               lambda a: a ** 3,                         [(3, 4)], False),
    ("neg",               lambda a: -a,                             [(3, 4)], False),
    ("matmul_2d",         lambda a, b: a @ b,                       [(3, 5), (5, 4)], False),
    ("matmul_batched",    lambda a, b: a @ b,                       [(2, 3, 5), (2, 5, 4)], False),
    ("matmul_broadcast",  lambda a, b: a @ b,                       [(2, 6, 3, 5), (5, 4)], False),
    ("exp",               lambda a: a.exp(),                        [(3, 4)], False),
    ("log",               lambda a: a.log(),                        [(3, 4)], True),
    ("sqrt",              lambda a: a.sqrt(),                       [(3, 4)], True),
    ("rsqrt",             lambda a: a.rsqrt(),                      [(3, 4)], True),
    ("tanh",              lambda a: a.tanh(),                       [(3, 4)], False),
    ("sigmoid",           lambda a: a.sigmoid(),                    [(3, 4)], False),
    ("relu_squared",      lambda a: snn.relu_squared(a),            [(3, 4)], False),
    ("square",            lambda a: a.square(),                     [(3, 4)], False),
    ("sum_all",           lambda a: a.sum(),                        [(3, 4)], False),
    ("sum_axis",          lambda a: a.sum(axis=1),                  [(3, 4, 5)], False),
    ("sum_keepdims",      lambda a: a.sum(axis=-1, keepdims=True),  [(3, 4)], False),
    ("mean_all",          lambda a: a.mean(),                       [(3, 4)], False),
    ("mean_axis",         lambda a: a.mean(axis=1),                 [(3, 4, 5)], False),
    ("reshape",           lambda a: a.reshape(6, 2),                [(3, 4)], False),
    ("view_infer",        lambda a: a.view(3, -1),                  [(3, 4, 5)], False),
    ("permute",           lambda a: a.permute(2, 0, 1),             [(2, 3, 4)], False),
    ("swapaxes",          lambda a: a.swapaxes(0, 2),               [(2, 3, 4)], False),
    ("mT",                lambda a: a.mT,                           [(2, 3, 4)], False),
    ("getitem_slice",     lambda a: a[:, 1:3],                      [(4, 5)], False),
    ("getitem_int",       lambda a: a[2],                           [(4, 5)], False),
    ("getitem_ellipsis",  lambda a: a[..., :2],                     [(2, 3, 5)], False),
    ("unsqueeze",         lambda a: a.unsqueeze(-1),                [(3, 4)], False),
    ("squeeze",           lambda a: a.squeeze(1),                   [(3, 1, 4)], False),
    ("expand",            lambda a: a.expand(3, 5, 4),              [(3, 1, 4)], False),
    ("softmax",           lambda a: st.softmax(a, -1),              [(3, 7)], False),
    ("softmax_axis0",     lambda a: st.softmax(a, 0),               [(3, 7)], False),
    ("rms_norm",          lambda a: st.rms_norm(a, 1e-6),           [(3, 4, 5)], False),
    ("norm_fn",           lambda a: snn.norm(a),                    [(2, 3, 5)], False),
    ("cat",               lambda a, b: st.cat([a, b], axis=1),      [(2, 3), (2, 5)], False),
    ("cat_last",          lambda a, b: st.cat([a, b], axis=-1),     [(2, 3, 4), (2, 3, 2)], False),
    ("stack",             lambda a, b: st.stack([a, b], axis=1),    [(2, 3), (2, 3)], False),
    ("chained",           lambda a, b: ((a @ b).tanh() * 2.0).sum(axis=-1).mean(), [(3, 5), (5, 4)], False),
    ("fanout",            lambda a: (a * a + a).sum(),              [(3, 4)], False),
    ("deep_chain",        lambda a: ((a.tanh() + a).square() * a).mean(), [(3, 4)], False),
]


@pytest.mark.parametrize("name,fn,shapes,positive", OPS, ids=[o[0] for o in OPS])
def test_op_gradient_matches_finite_difference(float64_engine, name, fn, shapes, positive):
    """Every primitive's analytic backward equals a central finite difference."""
    _grad_check(fn, shapes, positive=positive)


# ----------------------------------------------------------------------------
# forward values against closed forms
#
# Finite differences only validate that backward is consistent with forward. These
# pin down that forward itself computes the intended function.

def _np_softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def test_softmax_forward_matches_closed_form(float64_engine):
    _check_forward(lambda a: st.softmax(a, -1), [(4, 7)], lambda a: _np_softmax(a, -1))
    _check_forward(lambda a: st.softmax(a, 0), [(4, 7)], lambda a: _np_softmax(a, 0))


def test_softmax_is_a_probability_distribution(float64_engine):
    p = st.softmax(Tensor(np.random.default_rng(0).standard_normal((5, 9)) * 50), -1)
    np.testing.assert_allclose(p.data.sum(-1), 1.0, atol=1e-12)
    assert (p.data >= 0).all()


def test_rms_norm_forward_matches_closed_form(float64_engine):
    _check_forward(lambda a: st.rms_norm(a, 1e-6), [(3, 4, 5)],
                   lambda a: a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-6))


def test_rms_norm_output_has_unit_rms(float64_engine):
    y = st.rms_norm(Tensor(np.random.default_rng(0).standard_normal((4, 64)) * 7.0), 1e-12)
    np.testing.assert_allclose(np.sqrt((y.data ** 2).mean(-1)), 1.0, rtol=1e-6)


def test_sigmoid_and_tanh_forward(float64_engine):
    _check_forward(lambda a: a.sigmoid(), [(4, 5)], lambda a: 1.0 / (1.0 + np.exp(-a)))
    _check_forward(lambda a: a.tanh(), [(4, 5)], np.tanh)


def test_sigmoid_does_not_overflow():
    x = Tensor(np.array([-1e4, -40.0, 0.0, 40.0, 1e4], dtype=np.float32))
    y = x.sigmoid()
    assert np.isfinite(y.data).all()
    np.testing.assert_allclose(y.data, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)


def test_relu_and_clamp_min_forward():
    x = Tensor(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]))
    np.testing.assert_allclose(x.relu().data, [0, 0, 0, 0.5, 2.0])
    np.testing.assert_allclose(snn.relu_squared(x).data, [0, 0, 0, 0.25, 4.0])
    np.testing.assert_allclose(x.clamp_min(-1.0).data, [-1, -0.5, 0, 0.5, 2.0])


def test_max_forward_and_tie_splitting():
    """Ties split the gradient evenly across the tied entries."""
    x = Tensor(np.array([[1.0, 3.0, 3.0], [5.0, 2.0, 1.0]]), requires_grad=True)
    out = x.max(axis=-1)
    np.testing.assert_allclose(out.data, [3.0, 5.0])
    out.backward(np.ones(2))
    np.testing.assert_allclose(x.grad, [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])


def test_relu_gradient_at_the_kink_is_zero():
    """ReLU is not differentiable at 0; we define the subgradient there as 0.
    Finite differences cannot check this, so it is pinned down explicitly."""
    x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
    x.relu().sum().backward()
    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0])


def test_matmul_forward_against_explicit_loops(float64_engine):
    """Guards the batched/broadcast matmul against an unambiguous reference."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((2, 3, 4)), rng.standard_normal((2, 4, 5))
    got = (Tensor(a) @ Tensor(b)).data
    want = np.zeros((2, 3, 5))
    for n in range(2):
        for i in range(3):
            for j in range(5):
                want[n, i, j] = sum(a[n, i, k] * b[n, k, j] for k in range(4))
    np.testing.assert_allclose(got, want, atol=1e-12)


# ----------------------------------------------------------------------------
# indexing, masking, scatter/gather

def test_getitem_duplicate_indices_accumulate():
    """A gather that repeats a row must scatter-ADD, not overwrite. This is the
    embedding backward: a token appearing twice must collect both gradients."""
    x = Tensor(np.zeros((3, 4)), requires_grad=True)
    x[np.array([0, 2, 0, 0, 1])].backward(np.ones((5, 4)))
    np.testing.assert_allclose(x.grad, [[3.0] * 4, [1.0] * 4, [1.0] * 4])


def test_getitem_slice_gradient_is_placed_correctly():
    x = Tensor(np.zeros((4, 5)), requires_grad=True)
    x[:, 1:3].backward(np.ones((4, 2)))
    expected = np.zeros((4, 5))
    expected[:, 1:3] = 1.0
    np.testing.assert_allclose(x.grad, expected)


def test_cat_splits_gradient_back_to_each_input():
    a = Tensor(np.zeros((2, 3)), requires_grad=True)
    b = Tensor(np.zeros((2, 5)), requires_grad=True)
    g = np.arange(16, dtype=np.float64).reshape(2, 8)
    st.cat([a, b], axis=1).backward(g)
    np.testing.assert_allclose(a.grad, g[:, :3])
    np.testing.assert_allclose(b.grad, g[:, 3:])


def test_stack_routes_gradient_per_slice():
    a = Tensor(np.zeros((2, 3)), requires_grad=True)
    b = Tensor(np.zeros((2, 3)), requires_grad=True)
    g = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    st.stack([a, b], axis=1).backward(g)
    np.testing.assert_allclose(a.grad, g[:, 0])
    np.testing.assert_allclose(b.grad, g[:, 1])


def test_masked_fill_blocks_gradient_where_masked():
    mask = np.array([[True, False, False], [False, True, False]])
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    out = st.masked_fill(x, mask, -1e9)
    np.testing.assert_allclose(out.data, [[-1e9, 1.0, 1.0], [1.0, -1e9, 1.0]])
    out.backward(np.ones((2, 3)))
    np.testing.assert_allclose(x.grad, (~mask).astype(float))


def test_where_routes_gradient_to_the_selected_branch():
    mask = np.array([[True, False, False], [False, True, False]])
    a = Tensor(np.full((2, 3), 7.0), requires_grad=True)
    b = Tensor(np.full((2, 3), -7.0), requires_grad=True)
    out = st.where(mask, a, b)
    np.testing.assert_allclose(out.data, np.where(mask, 7.0, -7.0))
    out.backward(np.ones((2, 3)))
    np.testing.assert_allclose(a.grad, mask.astype(float))
    np.testing.assert_allclose(b.grad, (~mask).astype(float))


def test_topk_selects_the_largest_and_routes_gradient_to_them():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 9))
    t = Tensor(x.copy(), requires_grad=True)
    vals, idx = st.topk(t, 3, axis=-1)

    assert vals.shape == (5, 3) and idx.shape == (5, 3)
    # the selected set must be exactly the 3 largest of each row
    for r in range(5):
        assert set(idx[r]) == set(np.argsort(-x[r])[:3])
        assert all(len(set(idx[r])) == 3 for _ in [0])
    np.testing.assert_allclose(np.sort(vals.data, -1), np.sort(np.sort(x, -1)[:, -3:], -1))

    vals.backward(np.ones((5, 3)))
    expected = np.zeros_like(x)
    np.put_along_axis(expected, idx, 1.0, axis=-1)
    np.testing.assert_allclose(t.grad, expected)


def test_index_add_is_scatter_add_and_its_adjoint_is_gather(float64_engine):
    rng = np.random.default_rng(0)
    src_arr = rng.standard_normal((7, 4))
    idx = np.array([0, 0, 2, 1, 2, 2, 0])

    src = Tensor(src_arr.copy(), requires_grad=True)
    out = st.index_add((3, 4), idx, src)
    want = np.zeros((3, 4))
    for i, row in zip(idx, src_arr):
        want[i] += row
    np.testing.assert_allclose(out.data, want, atol=1e-12)

    g = rng.standard_normal((3, 4))
    out.backward(g)
    np.testing.assert_allclose(src.grad, g[idx])  # adjoint of scatter-add is gather


# ----------------------------------------------------------------------------
# cross entropy

def test_cross_entropy_forward_matches_closed_form():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((12, 9)) * 4.0
    targets = rng.integers(0, 9, 12)
    logp = np.log(_np_softmax(logits, -1))
    want = -logp[np.arange(12), targets]

    got_none = st.cross_entropy(Tensor(logits), targets, reduction="none")
    np.testing.assert_allclose(got_none.data, want, rtol=1e-6)
    np.testing.assert_allclose(st.cross_entropy(Tensor(logits), targets).item(), want.mean(), rtol=1e-6)
    np.testing.assert_allclose(
        st.cross_entropy(Tensor(logits), targets, reduction="sum").item(), want.sum(), rtol=1e-6)


def test_cross_entropy_ignore_index_excludes_from_value_and_gradient():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((6, 5))
    targets = rng.integers(0, 5, 6)
    targets[[1, 4]] = -1

    t = Tensor(logits.copy(), requires_grad=True)
    out = st.cross_entropy(t, targets, ignore_index=-1)
    out.backward()
    # ignored rows contribute nothing and receive no gradient
    np.testing.assert_allclose(t.grad[[1, 4]], 0.0)
    # and the mean is over the 4 valid rows only
    logp = np.log(_np_softmax(logits, -1))
    valid = [i for i in range(6) if targets[i] != -1]
    want = np.mean([-logp[i, targets[i]] for i in valid])
    np.testing.assert_allclose(out.item(), want, rtol=1e-6)


def test_cross_entropy_gradient_matches_finite_difference(float64_engine):
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 7, 9)
    targets[3] = -1
    _grad_check(lambda a: st.cross_entropy(a, targets, ignore_index=-1), [(9, 7)], seed=1)


def test_cross_entropy_equals_log_vocab_at_uniform_logits():
    """A model that has learned nothing must sit at ln(V)."""
    V = 256
    out = st.cross_entropy(Tensor(np.zeros((8, V))), np.arange(8) % V)
    np.testing.assert_allclose(out.item(), math.log(V), rtol=1e-6)


def test_cross_entropy_is_stable_on_huge_logits():
    """A naive exp(logits) would overflow to inf here."""
    logits = np.array([[1e4, -1e4, 0.0], [-1e4, 1e4, 0.0]], dtype=np.float32)
    t = Tensor(logits.copy(), requires_grad=True)
    out = st.cross_entropy(t, np.array([0, 1]))
    assert np.isfinite(out.data).all() and out.item() < 1e-3
    out.backward()
    assert np.isfinite(t.grad).all()


# ----------------------------------------------------------------------------
# engine semantics

def test_no_grad_blocks_graph_construction():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    with st.no_grad():
        y = x * 3.0
    assert not y.requires_grad and y._backward is None
    assert st.is_grad_enabled()
    assert (x * 3.0).requires_grad


def test_detach_cuts_the_graph():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    (x.detach() * 2.0).sum().backward()
    assert x.grad is None


def test_gradients_accumulate_across_backward_calls():
    x = Tensor(np.ones(3), requires_grad=True)
    (x * 2.0).sum().backward()
    np.testing.assert_allclose(x.grad, 2.0)
    (x * 2.0).sum().backward()
    np.testing.assert_allclose(x.grad, 4.0)  # accumulated, not replaced
    x.zero_grad()
    assert x.grad is None


def test_backward_releases_the_graph_and_interior_grads():
    """Only leaves keep .grad, and the graph (which holds the saved activations) is
    dropped, so a `loss` still in scope does not pin the whole forward pass."""
    x = Tensor(np.ones(3), requires_grad=True)
    h = x * 2.0
    loss = (h * h).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, 8.0)
    assert h.grad is None and loss.grad is None
    assert h._parents == () and loss._parents == ()


def test_second_backward_through_a_released_graph_raises():
    x = Tensor(np.ones(3), requires_grad=True)
    loss = (x * 2.0).sum()
    loss.backward()
    with pytest.raises(RuntimeError, match="second time"):
        loss.backward()


def test_retain_graph_allows_a_second_backward():
    x = Tensor(np.ones(3), requires_grad=True)
    loss = (x * 2.0).sum()
    loss.backward(retain_graph=True)
    loss.backward()
    np.testing.assert_allclose(x.grad, 4.0)


def test_first_gradient_write_does_not_alias_between_parents():
    """`a + b` hands the same incoming gradient to both parents. If the first write
    aliased it instead of copying, a later accumulation into one would leak into the
    other."""
    a = Tensor(np.ones(2), requires_grad=True)
    b = Tensor(np.ones(2), requires_grad=True)
    ((a + b) * 1.0 + a * 1.0).sum().backward()
    np.testing.assert_allclose(a.grad, 2.0)
    np.testing.assert_allclose(b.grad, 1.0)


def test_backward_on_non_scalar_without_grad_raises():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(RuntimeError, match="non-scalar"):
        (x * 2.0).backward()


def test_deep_graph_does_not_hit_recursion_limit():
    """The backward pass uses an explicit stack, so depth is bounded by memory only."""
    x = Tensor(np.ones(4), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.sum().backward()
    assert np.isfinite(x.grad).all()


def test_dtype_switch_round_trips():
    assert st.get_dtype() == np.float32
    st.set_dtype(np.float64)
    try:
        assert Tensor(np.ones(2)).data.dtype == np.float64
    finally:
        st.set_dtype(np.float32)
    assert Tensor(np.ones(2)).data.dtype == np.float32


# ----------------------------------------------------------------------------
# layers

def test_linear_forward_and_gradients(float64_engine):
    lin = snn.Linear(3, 5, bias=True)
    rng = np.random.default_rng(0)
    lin.weight.data = rng.standard_normal((5, 3))
    lin.bias.data = rng.standard_normal(5)
    x_arr = rng.standard_normal((2, 4, 3))

    x = Tensor(x_arr.copy(), requires_grad=True)
    out = lin(x)
    np.testing.assert_allclose(out.data, x_arr @ lin.weight.data.T + lin.bias.data, atol=1e-10)

    # gradients w.r.t. input and both parameters
    g = rng.standard_normal(out.shape)
    out.backward(g)
    np.testing.assert_allclose(x.grad, g @ lin.weight.data, atol=1e-10)
    np.testing.assert_allclose(lin.weight.grad,
                               g.reshape(-1, 5).T @ x_arr.reshape(-1, 3), atol=1e-10)
    np.testing.assert_allclose(lin.bias.grad, g.reshape(-1, 5).sum(0), atol=1e-10)


def test_linear_without_bias_has_no_bias_parameter():
    lin = snn.Linear(3, 5, bias=False)
    assert lin.bias is None
    assert [n for n, _ in lin.named_parameters()] == ["weight"]


def test_embedding_backward_accumulates_repeats():
    emb = snn.Embedding(5, 3)
    out = emb(np.array([[1, 1, 4]]))
    assert out.shape == (1, 3, 3)
    out.backward(np.ones(out.shape))
    expected = np.zeros((5, 3))
    expected[1] = 2.0  # token 1 appears twice
    expected[4] = 1.0
    np.testing.assert_allclose(emb.weight.grad, expected)


def test_embedding_forward_is_a_lookup():
    emb = snn.Embedding(6, 4)
    idx = np.array([[3, 0], [5, 3]])
    np.testing.assert_allclose(emb(idx).data, emb.weight.data[idx])


def test_causal_window_mask_content():
    m = snn.causal_window_mask(4, 4, window=1)
    expected = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0]], dtype=bool)
    assert np.array_equal(m, expected)
    assert np.array_equal(snn.causal_window_mask(3, 3, window=-1), np.triu(np.ones((3, 3), bool), 1))


def test_causal_window_mask_with_offset():
    """Decoding with a cache: a query at absolute position 4 may see keys 0..4."""
    m = snn.causal_window_mask(1, 6, window=-1, offset=4)
    np.testing.assert_array_equal(m, [[False] * 5 + [True]])


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
@pytest.mark.parametrize("window", [-1, 3])
def test_attention_forward_matches_explicit_reference(window, n_kv_head):
    """Independent numpy implementation of attention, including GQA head sharing."""
    B, T, H, D = 2, 7, 4, 8
    rng = np.random.default_rng(0)
    q = rng.standard_normal((B, T, H, D))
    k = rng.standard_normal((B, T, n_kv_head, D))
    v = rng.standard_normal((B, T, n_kv_head, D))

    mask = snn.causal_window_mask(T, T, window)
    want = np.zeros((B, T, H, D))
    rep = H // n_kv_head
    for b in range(B):
        for h in range(H):
            kv = h // rep  # which kv head this query head reads
            scores = (q[b, :, h, :] @ k[b, :, kv, :].T) / math.sqrt(D)
            scores = np.where(mask, -1e9, scores)
            want[b, :, h, :] = _np_softmax(scores, -1) @ v[b, :, kv, :]

    got = snn.attention(Tensor(q), Tensor(k), Tensor(v), window=window)
    np.testing.assert_allclose(got.data, want, atol=1e-6)


def test_attention_gradients_match_finite_difference(float64_engine):
    _grad_check(lambda q, k, v: snn.attention(q, k, v, window=2),
                [(1, 5, 2, 4), (1, 5, 2, 4), (1, 5, 2, 4)], n_probes=12, tol=1e-5)


@pytest.mark.parametrize("n_kv_head", [2, 1])
def test_gqa_attention_gradients_match_finite_difference(float64_engine, n_kv_head):
    """The grouped path stacks query heads as rows against a shared K/V. K and V must
    receive the *sum* of the gradients from every query head that reads them."""
    _grad_check(lambda q, k, v: snn.attention(q, k, v, window=2),
                [(2, 5, 4, 4), (2, 5, n_kv_head, 4), (2, 5, n_kv_head, 4)],
                n_probes=12, tol=1e-5)


def test_masked_softmax_matches_masked_fill_and_blocks_gradient(float64_engine):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5))
    mask = snn.causal_window_mask(3, 5, window=1, offset=2)
    fused = st.softmax(Tensor(x), axis=-1, mask=mask).data
    reference = _np_softmax(np.where(mask, -1e9, x), -1)
    np.testing.assert_allclose(fused, reference, atol=1e-12)
    assert (fused[mask] == 0).all()

    t = Tensor(x.copy(), requires_grad=True)
    (st.softmax(t, axis=-1, mask=mask) * Tensor(rng.standard_normal((3, 5)))).sum().backward()
    assert (t.grad[mask] == 0).all(), "blocked scores must receive exactly zero gradient"
    _grad_check(lambda a: st.softmax(a, axis=-1, mask=mask), [(3, 5)])


def test_attention_is_causal():
    """Changing a later token must not alter an earlier output."""
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((1, 6, 2, 4)) for _ in range(3))
    base = snn.attention(Tensor(q), Tensor(k), Tensor(v)).data
    k2, v2 = k.copy(), v.copy()
    k2[0, 5] += 10.0
    v2[0, 5] += 10.0
    perturbed = snn.attention(Tensor(q), Tensor(k2), Tensor(v2)).data
    np.testing.assert_allclose(base[0, :5], perturbed[0, :5], atol=1e-10)
    assert not np.allclose(base[0, 5], perturbed[0, 5])


def test_attention_rejects_bad_head_counts():
    q = Tensor(np.zeros((1, 2, 4, 2)))
    kv = Tensor(np.zeros((1, 2, 3, 2)))
    with pytest.raises(ValueError, match="divisible"):
        snn.attention(q, kv, kv)


def test_apply_rotary_emb_matches_closed_form(float64_engine):
    B, T, H, D = 2, 5, 3, 8
    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, T, H, D))
    cos = rng.standard_normal((1, T, 1, D // 2))
    sin = rng.standard_normal((1, T, 1, D // 2))
    d = D // 2
    want = np.concatenate([x[..., :d] * cos + x[..., d:] * sin,
                           x[..., :d] * (-sin) + x[..., d:] * cos], axis=3)
    got = snn.apply_rotary_emb(Tensor(x), Tensor(cos), Tensor(sin))
    np.testing.assert_allclose(got.data, want, atol=1e-10)


def test_rotary_emb_preserves_norm_and_relative_angle():
    """A real rotation: it must preserve vector norms, and the dot product between
    q and k must depend only on their relative position."""
    T, D = 8, 16
    model = GPT(GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=32,
                          sequence_len=T, vocab_size=16))
    cos, sin = model.cos, model.sin
    rng = np.random.default_rng(0)
    vec = rng.standard_normal((1, 1, 1, D))
    x = np.broadcast_to(vec, (1, T, 1, D)).copy()
    y = snn.apply_rotary_emb(Tensor(x), cos, sin).data
    np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)
    # same offset => same inner product, at every absolute position
    dots = [float((y[0, i] * y[0, i + 2]).sum()) for i in range(T - 2)]
    np.testing.assert_allclose(dots, dots[0], rtol=1e-4)


# ----------------------------------------------------------------------------
# module system

def test_module_registration_and_state_dict():
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=16)
    model = GPT(config)
    names = [n for n, _ in model.named_parameters()]

    assert len(names) == len(set(names)), "parameter names must be unique"
    assert "wte.weight" in names and "lm_head.weight" in names
    assert "h.0.attn.c_q.weight" in names and "h.1.mlp.c_fc.weight" in names
    assert "resid_lambdas" in names and "backout_lambda" in names
    assert any(n.startswith("value_embeds.") for n in names)
    assert not any("cos" in n or "sin" in n for n in names), "rotary tables are buffers"
    assert model.num_parameters() == sum(p.numel() for p in model.parameters())

    sd = model.state_dict()
    reloaded = GPT(config)
    reloaded.load_state_dict(sd)
    for (n, a), (_, b) in zip(model.named_parameters(), reloaded.named_parameters()):
        np.testing.assert_array_equal(a.data, b.data, err_msg=n)


def test_load_state_dict_rejects_missing_and_unexpected_keys():
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=16)
    model = GPT(config)
    sd = model.state_dict()
    with pytest.raises(KeyError, match="missing"):
        model.load_state_dict({k: v for k, v in sd.items() if k != "wte.weight"})
    with pytest.raises(KeyError, match="unexpected"):
        model.load_state_dict({**sd, "nope": np.zeros(1)})
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_state_dict({**sd, "wte.weight": np.zeros((1, 1))})


def test_state_dict_is_a_copy_not_a_view():
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=16)
    model = GPT(config)
    sd = model.state_dict()
    model.wte.weight.data[0, 0] += 123.0
    assert sd["wte.weight"][0, 0] != model.wte.weight.data[0, 0]


def test_reassigning_an_attribute_unregisters_the_parameter():
    m = snn.Linear(3, 4)
    assert "weight" in dict(m.named_parameters())
    m.weight = None
    assert "weight" not in dict(m.named_parameters())


def test_modulelist_and_moduledict_register_children():
    ml = snn.ModuleList([snn.Linear(2, 2), snn.Linear(2, 2)])
    assert len(ml) == 2 and len(list(ml.named_parameters())) == 2
    assert {n for n, _ in ml.named_parameters()} == {"0.weight", "1.weight"}
    md = snn.ModuleDict({"a": snn.Linear(2, 2)})
    assert "a" in md and [n for n, _ in md.named_parameters()] == ["a.weight"]


def test_train_eval_propagates_and_gates_aux_loss():
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=3, num_experts_per_tok=2)
    model = GPT(config)
    assert all(m.training for m in model.modules())
    model.eval()
    assert not any(m.training for m in model.modules())

    x = np.random.default_rng(0).integers(0, 16, (2, 8))
    model(x, x)
    assert model.collect_aux_loss() is None, "no load-balancing loss in eval mode"
    model.train()
    model(x, x)
    assert model.collect_aux_loss() is not None


def test_zero_grad_clears_gradients():
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=16)
    model = GPT(config)
    x = np.random.default_rng(0).integers(0, 16, (2, 8))
    model(x, x).backward()
    assert any(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


# ----------------------------------------------------------------------------
# whole-model gradient check against finite differences

def _finite_difference_check(config, n_probes=40, eps=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    model = GPT(config)
    x = rng.integers(0, config.vocab_size, (2, config.sequence_len))
    y = rng.integers(0, config.vocab_size, (2, config.sequence_len))

    loss = model(x, y)
    model.zero_grad()
    loss.backward()
    analytic = {n: p.grad.copy() for n, p in model.named_parameters()}

    params = list(model.named_parameters())
    for _ in range(n_probes):
        name, p = params[rng.integers(len(params))]
        idx = np.unravel_index(rng.integers(p.numel()), p.data.shape)
        original = p.data[idx]

        p.data[idx] = original + eps
        plus = model(x, y).item()
        p.data[idx] = original - eps
        minus = model(x, y).item()
        p.data[idx] = original

        numeric = (plus - minus) / (2 * eps)
        expected = analytic[name][idx]
        scale = max(abs(numeric), abs(expected), 1e-3)
        assert abs(numeric - expected) / scale < 2e-4, (
            f"{name}{idx}: analytic {expected:.10f} vs numeric {numeric:.10f}")


def test_dense_model_gradients_match_finite_differences(float64_engine):
    config = GPTConfig(n_layer=3, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, window_pattern="SL")
    _finite_difference_check(config)


def test_moe_model_gradients_match_finite_differences(float64_engine):
    """Covers router weights, every expert, and the shared expert -- i.e. the top-k
    gather, the expert-order sort, and the scatter-add combine."""
    config = GPTConfig(n_layer=3, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=4, n_shared_experts=1,
                       num_experts_per_tok=2, aux_loss_alpha=0.01)
    _finite_difference_check(config, seed=1)


def test_moe_gradients_with_norm_topk_prob(float64_engine):
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=2, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=4, num_experts_per_tok=3,
                       norm_topk_prob=True, seq_aux=False, aux_loss_alpha=0.02)
    _finite_difference_check(config, seed=2)


# ----------------------------------------------------------------------------
# model specifics

def test_logits_are_softcapped():
    """15*tanh(z/15) must keep logits bounded however large the projection gets."""
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=16)
    model = GPT(config)
    model.lm_head.weight.data *= 1e6
    logits = model.logits(Tensor(np.random.default_rng(0).standard_normal((2, 8, 24)) * 100))
    assert np.abs(logits.data).max() <= 15.0 + 1e-6


def test_untrained_model_sits_at_log_vocab():
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=64)
    model = GPT(config)
    rng = np.random.default_rng(0)
    loss = model(rng.integers(0, 64, (4, 8)), rng.integers(0, 64, (4, 8)))
    assert abs(loss.item() - math.log(64)) < 0.1


def test_window_sizes_follow_the_pattern_and_last_layer_is_full():
    config = GPTConfig(n_layer=5, n_head=2, n_kv_head=1, n_embd=24, sequence_len=64,
                       vocab_size=16, window_pattern="SL")
    model = GPT(config)
    assert model.window_sizes == [16, -1, 16, -1, -1]  # pattern tiled, last forced to L


def test_value_embeddings_are_on_alternating_layers_including_the_last():
    config = GPTConfig(n_layer=5, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=16)
    model = GPT(config)
    assert sorted(model.value_embeds.keys()) == ["0", "2", "4"]
    assert str(config.n_layer - 1) in model.value_embeds


def test_config_validation_rejects_bad_shapes():
    for kwargs in [
        {"n_embd": 25, "n_head": 2},            # n_embd not divisible by n_head
        {"n_head": 3, "n_kv_head": 2},          # n_head not divisible by n_kv_head
        {"n_embd": 16},                         # < 24, the smear gate needs 24 channels
        {"n_embd": 30, "n_head": 2},            # head_dim 15 is odd -> rotary impossible
        {"window_pattern": "SLX"},              # bad pattern character
    ]:
        base = {"n_layer": 2, "n_head": 2, "n_kv_head": 1, "n_embd": 24,
                "sequence_len": 8, "vocab_size": 16}
        with pytest.raises(ValueError):
            GPTConfig(**{**base, **kwargs})


def test_config_accepts_a_valid_shape():
    cfg = GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=64, sequence_len=32, vocab_size=256)
    assert cfg.head_dim == 16


def test_forward_rejects_too_long_and_too_short_sequences():
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=16)
    model = GPT(config)
    with pytest.raises(ValueError, match="exceeds rotary"):
        model(np.zeros((1, 9), dtype=np.int64))
    with pytest.raises(ValueError, match="T >= 2"):
        model(np.zeros((1, 1), dtype=np.int64))


def test_generate_is_deterministic_at_temperature_zero():
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=16, vocab_size=32)
    model = GPT(config)
    a = model.generate([1, 2, 3], 8, temperature=0.0)
    b = model.generate([1, 2, 3], 8, temperature=0.0)
    assert a == b and len(a) == 11 and a[:3] == [1, 2, 3]


def test_generate_respects_top_k_and_seed():
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=16, vocab_size=32)
    model = GPT(config)
    a = model.generate([1, 2, 3], 8, temperature=1.0, top_k=5, seed=7)
    b = model.generate([1, 2, 3], 8, temperature=1.0, top_k=5, seed=7)
    c = model.generate([1, 2, 3], 8, temperature=1.0, top_k=5, seed=8)
    assert a == b, "same seed must reproduce"
    assert all(0 <= t < 32 for t in a)
    assert a != c, "different seeds should diverge"


def test_generate_does_not_build_a_graph():
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=1, n_embd=24, sequence_len=16, vocab_size=32)
    model = GPT(config)
    model.generate([1, 2, 3], 4, temperature=0.0)
    assert all(p.grad is None for p in model.parameters())


# ----------------------------------------------------------------------------
# MoE specifics

def test_moe_routes_every_token_to_exactly_top_k_experts():
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=5, num_experts_per_tok=2)
    model = GPT(config)
    moe = next(m for m in model.modules() if isinstance(m, MoE))
    idx, weight, aux = moe.gate(Tensor(np.random.default_rng(0).standard_normal((2, 8, 24))))
    assert idx.shape == (16, 2) and weight.shape == (16, 2)
    assert np.all((idx >= 0) & (idx < 5))
    assert all(len(set(row)) == 2 for row in idx), "a token must not pick one expert twice"
    assert aux is not None and aux.item() > 0


def test_moe_norm_topk_prob_weights_sum_to_one():
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=5, num_experts_per_tok=3,
                       norm_topk_prob=True)
    model = GPT(config)
    moe = next(m for m in model.modules() if isinstance(m, MoE))
    _, weight, _ = moe.gate(Tensor(np.random.default_rng(0).standard_normal((2, 8, 24))))
    np.testing.assert_allclose(weight.data.sum(-1), 1.0, atol=1e-5)


def test_moe_routed_scaling_factor_is_applied():
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=4, num_experts_per_tok=2,
                       norm_topk_prob=False, routed_scaling_factor=2.5)
    model = GPT(config)
    moe = next(m for m in model.modules() if isinstance(m, MoE))
    moe.gate.weight.data = np.zeros_like(moe.gate.weight.data)  # uniform scores = 1/4
    _, weight, _ = moe.gate(Tensor(np.random.default_rng(0).standard_normal((1, 8, 24))))
    np.testing.assert_allclose(weight.data, 0.25 * 2.5, rtol=1e-6)


def test_moe_balanced_router_hits_the_aux_loss_minimum():
    """With uniform scores and uniform dispatch, sum_i P_i * f_i == 1, so the
    auxiliary loss equals alpha exactly. That is its floor."""
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=4, num_experts_per_tok=4,
                       aux_loss_alpha=0.5, seq_aux=False)
    model = GPT(config)
    moe = next(m for m in model.modules() if isinstance(m, MoE))
    moe.gate.weight.data = np.zeros_like(moe.gate.weight.data)
    _, _, aux = moe.gate(Tensor(np.random.default_rng(0).standard_normal((2, 8, 24))))
    np.testing.assert_allclose(aux.item(), 0.5, rtol=1e-5)


def test_moe_aux_loss_penalises_imbalance():
    """A router that sends every token to one expert must score worse than a
    balanced one. This is the property the aux loss exists to enforce.

    The loss is alpha * sum_i P_i * f_i, where P is the mean softmax score and f is
    the dispatch fraction scaled by E. Because sum_i f_i == E always, a *uniform* P
    gives exactly alpha no matter how skewed the dispatch is -- the penalty only
    bites when high scores coincide with heavy dispatch. Collapsed onto one expert
    that product is 1 * E, so the loss is alpha * E.

    Input is all-ones so the routing logits are deterministic (logit_e = sum(W[e])).
    """
    E, alpha = 4, 1.0
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=E, num_experts_per_tok=1,
                       aux_loss_alpha=alpha, seq_aux=False)
    model = GPT(config)
    moe = next(m for m in model.modules() if isinstance(m, MoE))
    x = Tensor(np.ones((2, 8, 24)))

    moe.gate.weight.data = np.zeros_like(moe.gate.weight.data)
    _, _, balanced = moe.gate(x)
    np.testing.assert_allclose(balanced.item(), alpha, rtol=1e-5)

    # Make expert 0 win every token by a mile
    moe.gate.weight.data[0] = 100.0
    idx, _, collapsed = moe.gate(x)
    assert set(idx.reshape(-1).tolist()) == {0}, "setup failed: routing not collapsed"
    np.testing.assert_allclose(collapsed.item(), alpha * E, rtol=1e-4)
    assert collapsed.item() > balanced.item()


def test_moe_dispatch_equals_a_naive_per_token_loop(float64_engine):
    """The sort/slice/scatter-add dispatch must equal the obvious slow version."""
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=4, num_experts_per_tok=2)
    model = GPT(config)
    moe = next(m for m in model.modules() if isinstance(m, MoE))
    # zero-init c_proj would make every expert output 0, so randomize
    rng = np.random.default_rng(0)
    for e in moe.experts:
        e.c_proj.weight.data = rng.standard_normal(e.c_proj.weight.shape) * 0.1

    x_flat = Tensor(rng.standard_normal((16, 24)))
    idx, weight, _ = moe.gate(Tensor(x_flat.data.reshape(2, 8, 24)))
    got = moe._dispatch(x_flat, idx, weight).data

    want = np.zeros((16, 24))
    for token in range(16):
        for slot in range(2):
            expert = moe.experts[idx[token, slot]]
            y = expert(Tensor(x_flat.data[token:token + 1])).data[0]
            want[token] += weight.data[token, slot] * y
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_dense_and_moe_layer_placement():
    config = GPTConfig(n_layer=4, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                       vocab_size=16, n_routed_experts=2, first_k_dense_replace=1)
    model = GPT(config)
    assert [isinstance(b.mlp, MoE) for b in model.h] == [False, True, True, True]


def test_moe_gate_rejects_bad_top_k():
    with pytest.raises(ValueError, match="top_k"):
        GPT(GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8,
                      vocab_size=16, n_routed_experts=2, num_experts_per_tok=5))


# ----------------------------------------------------------------------------
# optimizers

def test_adamw_first_step_matches_hand_computation():
    """Step 1 of Adam with bias correction reduces to lr * sign(g) (up to eps)."""
    g = np.array([2.0, -3.0, 0.5])
    p = snn.Parameter(np.zeros(3))
    opt = AdamW([p], lr=0.1, betas=(0.9, 0.95), eps=1e-12, weight_decay=0.0)
    p.grad = g.copy()
    opt.step()
    # m1_hat = g, v1_hat = g^2  =>  update = lr * g / |g| = lr * sign(g)
    np.testing.assert_allclose(p.data, -0.1 * np.sign(g), rtol=1e-6)


def test_adamw_matches_a_reference_implementation(float64_engine):
    """Multi-step comparison against the textbook update written out separately."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 4))
    grads = [rng.standard_normal((6, 4)) for _ in range(5)]
    lr, b1, b2, eps, wd = 0.01, 0.9, 0.95, 1e-10, 0.1

    p = snn.Parameter(w.copy())
    opt = AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

    ref, m, v = w.copy(), np.zeros_like(w), np.zeros_like(w)
    for t, g in enumerate(grads, start=1):
        p.grad = g.copy()
        opt.step()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        ref -= lr * wd * ref                                   # decoupled decay
        ref -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        np.testing.assert_allclose(p.data, ref, rtol=1e-9, atol=1e-12)


def test_adamw_weight_decay_shrinks_a_zero_gradient_parameter():
    p = snn.Parameter(np.ones(3))
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = np.zeros(3)
    opt.step()
    np.testing.assert_allclose(p.data, 1.0 - 0.1 * 0.5, rtol=1e-6)


def test_adamw_reaches_the_minimum_of_a_quadratic():
    """min ||x - target||^2. Adam's step near the optimum is ~lr regardless of how
    small the gradient is (m/sqrt(v) -> sign), so a decaying lr is what lands it."""
    target = np.array([1.0, -2.0, 0.5])
    p = snn.Parameter(np.zeros(3))
    opt = AdamW([p], lr=0.05)
    for i in range(800):
        opt.param_groups[0]["lr"] = 0.05 * (0.5 ** (i // 100))
        p.grad = 2.0 * (p.data - target)
        opt.step()
    np.testing.assert_allclose(p.data, target, atol=1e-3)


def test_optimizer_skips_parameters_without_gradients():
    """MoE experts that received no tokens have grad None and must be left alone."""
    a, b = snn.Parameter(np.ones(3)), snn.Parameter(np.ones(3))
    opt = AdamW([a, b], lr=0.1)
    a.grad = np.ones(3)
    opt.step()
    assert not np.allclose(a.data, 1.0)
    np.testing.assert_allclose(b.data, 1.0)


@pytest.mark.parametrize("shape", [(8, 8), (12, 4), (4, 12)])
def test_polar_express_conditions_the_update(shape):
    """The point of Muon. Polar Express is an *approximate* orthogonalisation (5
    polynomial steps, no SVD), so singular values are not each exactly 1. What it
    does guarantee: a badly spread spectrum comes out nearly flat, and the final
    Muon+ renormalisation makes the RMS singular value exactly 1."""
    rng = np.random.default_rng(0)
    k = min(shape)
    u, _ = np.linalg.qr(rng.standard_normal((shape[0], k)))
    v, _ = np.linalg.qr(rng.standard_normal((shape[1], k)))
    g = ((u * np.geomspace(1e-3, 1.0, k)) @ v.T).astype(np.float32)

    out = polar_express(g)
    assert out.shape == shape
    sv_in = np.linalg.svd(g, compute_uv=False)
    sv_out = np.linalg.svd(out, compute_uv=False)

    assert sv_in.max() / sv_in.min() > 100, f"input should be ill-conditioned: {sv_in}"
    assert sv_out.max() / sv_out.min() < 1.5, f"spectrum not flattened: {sv_out}"
    np.testing.assert_allclose(np.sqrt((sv_out ** 2).mean()), 1.0, rtol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(out), math.sqrt(min(shape)), rtol=1e-4)


def test_polar_express_preserves_direction():
    """An already-orthogonal input should come back essentially unchanged."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    out = polar_express(q.astype(np.float32))
    assert abs(float((out * q).sum()) / 6.0) > 0.98


def test_polar_express_handles_zero_gradient():
    np.testing.assert_array_equal(polar_express(np.zeros((4, 5), np.float32)), np.zeros((4, 5)))


def test_polar_express_rejects_non_matrices():
    with pytest.raises(ValueError, match="2-D"):
        polar_express(np.zeros(5, np.float32))


def test_muon_rejects_non_matrices():
    with pytest.raises(ValueError, match="2-D"):
        Muon([snn.Parameter(np.zeros(5))])


def test_muon_update_is_scale_invariant():
    """Orthogonalisation throws away the gradient's magnitude: scaling the gradient
    by 1000 must not change the step. That is the defining property of Muon.

    Tolerance is loose because `polar_express` runs in float32 and its five
    polynomial steps have large coefficients (8.2, -22.5, 15.9), which amplifies
    rounding. There is also an absolute `+ 1e-6` in its normalisation, which does
    not scale with the input. The invariance is exact in exact arithmetic.
    """
    rng = np.random.default_rng(0)
    g = rng.standard_normal((6, 6))

    def step_with(grad):
        p = snn.Parameter(np.zeros((6, 6)))
        opt = Muon([p], lr=0.1, momentum=0.0, nesterov=False)
        p.grad = grad.copy()
        opt.step()
        return p.data.copy()

    base = step_with(g)
    for factor in (1000.0, 0.001):
        other = step_with(g * factor)
        rel = np.linalg.norm(other - base) / np.linalg.norm(base)
        assert rel < 5e-3, f"scaling by {factor} changed the update by {rel:.2e} relative"


def test_muon_decreases_a_quadratic_loss():
    rng = np.random.default_rng(0)
    target = rng.standard_normal((6, 6))
    p = snn.Parameter(np.zeros((6, 6)))
    opt = Muon([p], lr=0.05, momentum=0.9)
    losses = []
    for _ in range(200):
        diff = p.data - target
        losses.append(float((diff ** 2).sum()))
        p.grad = 2.0 * diff
        opt.step()
    assert losses[-1] < losses[0] * 0.05


def test_setup_optimizer_splits_parameters_correctly():
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=16)
    model = GPT(config)
    opt = setup_optimizer(model)

    muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    adamw_ids = {id(p) for g in opt.adamw.param_groups for p in g["params"]}
    assert not muon_ids & adamw_ids, "a parameter must not be claimed by both optimizers"
    assert muon_ids | adamw_ids == {id(p) for p in model.parameters()}, "every parameter covered"

    named = dict(model.named_parameters())
    assert id(named["wte.weight"]) in adamw_ids
    assert id(named["lm_head.weight"]) in adamw_ids
    assert id(named["resid_lambdas"]) in adamw_ids
    assert id(named["h.0.attn.c_q.weight"]) in muon_ids
    assert all(p.data.ndim == 2 for g in opt.muon.param_groups for p in g["params"])


def test_setup_optimizer_zero_grad_clears_everything():
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=1, n_embd=24, sequence_len=8, vocab_size=16)
    model = GPT(config)
    opt = setup_optimizer(model)
    model(np.zeros((2, 8), dtype=np.int64), np.zeros((2, 8), dtype=np.int64)).backward()
    opt.zero_grad()
    assert all(p.grad is None for p in model.parameters())


# ----------------------------------------------------------------------------
# data

def test_byte_tokenizer_round_trips():
    tok = ByteTokenizer()
    for text in ["hello", "7+5=12;", "héllo wörld", ""]:
        assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == 256 + 9  # bytes + reserved special tokens


def test_byte_tokenizer_is_byte_level():
    tok = ByteTokenizer()
    assert tok.encode("é") == [0xC3, 0xA9]  # two UTF-8 bytes, two tokens
    assert all(0 <= t < 256 for t in tok.encode("héllo wörld 你好"))


def test_addition_corpus_shape_and_entropy():
    corpus = make_addition_corpus(100, seed=0)
    assert len(corpus) == 100 * ADDITION_LINE_LEN
    for i in range(0, len(corpus), ADDITION_LINE_LEN):
        line = corpus[i:i + ADDITION_LINE_LEN]
        a, b = int(line[0]), int(line[2])
        assert line[1] == "+" and line[3] == "=" and line[6] == ";"
        assert int(line[4:6]) == a + b
    np.testing.assert_allclose(addition_entropy_floor(), 2 * math.log(10) / 7, rtol=1e-9)


def test_dataset_batches_are_shifted_by_one():
    ds = Dataset.from_text(make_addition_corpus(500, seed=0))
    x, y = ds.get_batch(4, 16, np.random.default_rng(0))
    assert x.shape == (4, 16) and y.shape == (4, 16)
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
    assert len(ds.val) > 0


def test_dataset_train_val_are_disjoint():
    ds = Dataset.from_text(make_addition_corpus(500, seed=0), split=0.9)
    assert len(ds.train) + len(ds.val) == len(ds.tokens)
    np.testing.assert_array_equal(np.concatenate([ds.train, ds.val]), ds.tokens)


def test_dataset_rejects_too_long_a_sequence():
    ds = Dataset.from_text("short")
    with pytest.raises(ValueError, match="need >"):
        ds.get_batch(2, 64, np.random.default_rng(0))


# ----------------------------------------------------------------------------
# it actually learns

def _train(model, dataset, steps, batch_size, seq_len, seed=0, **lrs):
    opt = setup_optimizer(model, **lrs)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        loss = model(*dataset.get_batch(batch_size, seq_len, rng))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_model_can_overfit_a_single_batch():
    """The sharpest end-to-end signal: if gradients and the optimizer are right, a
    model with more capacity than the batch should drive the loss to ~0."""
    config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=32, sequence_len=16, vocab_size=16)
    model = GPT(config)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 16, (2, 16)), rng.integers(0, 16, (2, 16))

    opt = setup_optimizer(model, matrix_lr=0.05, embedding_lr=0.2,
                          unembedding_lr=0.05, scalar_lr=0.05)
    first = model(x, y).item()
    for _ in range(300):
        loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first > 2.0, f"untrained model should start near ln(16)={math.log(16):.2f}, got {first}"
    assert loss.item() < 0.05, f"failed to overfit: {first:.4f} -> {loss.item():.4f}"


@pytest.mark.slow
def test_dense_model_approaches_the_entropy_floor():
    """Train on two-digit addition and check the loss gets close to 2*ln(10)/7."""
    dataset = Dataset.from_text(make_addition_corpus(20000, seed=0))
    config = GPTConfig(n_layer=4, n_head=4, n_kv_head=2, n_embd=64, sequence_len=64, vocab_size=256)
    model = GPT(config)
    losses = _train(model, dataset, steps=400, batch_size=16, seq_len=64,
                    matrix_lr=0.03, embedding_lr=0.2, unembedding_lr=0.02, scalar_lr=0.05)

    floor = addition_entropy_floor()
    final = float(np.mean(losses[-20:]))
    assert losses[0] > 4.0, "should start near ln(256)"
    assert final < floor + 0.15, f"final loss {final:.4f} vs floor {floor:.4f}"

    # And it should actually be able to do the arithmetic, greedily.
    tok = ByteTokenizer()
    correct = sum(tok.decode(model.generate(tok.encode(f"{a}+{b}="), 3, temperature=0.0))[-3:]
                  == f"{a + b:02d};" for a in range(10) for b in range(10))
    assert correct >= 90, f"only {correct}/100 sums correct"


@pytest.mark.slow
def test_moe_model_trains_and_keeps_experts_balanced():
    dataset = Dataset.from_text(make_addition_corpus(20000, seed=0))
    config = GPTConfig(n_layer=4, n_head=4, n_kv_head=2, n_embd=64, sequence_len=64,
                       vocab_size=256, n_routed_experts=4, n_shared_experts=1,
                       num_experts_per_tok=2, aux_loss_alpha=0.01)
    model = GPT(config)
    losses = _train(model, dataset, steps=300, batch_size=16, seq_len=64,
                    matrix_lr=0.03, embedding_lr=0.2, unembedding_lr=0.02, scalar_lr=0.05)
    assert float(np.mean(losses[-20:])) < losses[0] - 2.0

    # The load-balancing loss should keep dispatch far from collapsed-onto-one-expert.
    moe = next(m for m in model.modules() if isinstance(m, MoE))
    x, _ = dataset.get_batch(8, 64, np.random.default_rng(99))
    idx, _, _ = moe.gate(model.embed_tokens(x))
    share = np.bincount(idx.reshape(-1), minlength=config.n_routed_experts)
    share = share / share.sum()
    assert share.max() < 0.75, f"routing collapsed: {share}"
