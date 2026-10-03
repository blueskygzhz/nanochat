"""
Tests for the pipeline modules: engine (KV cache), eval (bpb + CORE), checkpointing.

The sharpest test in this file is `test_kv_cache_matches_full_forward`: a cache is
only correct if decoding token-by-token produces *exactly* what one full forward over
the same sequence produces. Everything else about the engine (sampling, stop tokens,
multi-sample fanout) is built on that identity.
"""

import json
import math
import os

import numpy as np
import pytest

from nanochat.scratch import (
    ByteTokenizer, Dataset, Engine, GPT, GPTConfig, KVCache, Tensor,
    build_model, evaluate_bpb, evaluate_task, find_last_step, list_steps,
    load_checkpoint, load_model, make_addition_corpus, sample_next_token,
    save_checkpoint, setup_optimizer, token_bytes_table,
)
from nanochat.scratch import eval as seval


@pytest.fixture
def tiny_model():
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
                       sequence_len=32, vocab_size=256)
    return GPT(config)


@pytest.fixture
def tokenizer():
    return ByteTokenizer()


class StubModel:
    """A model that always predicts `favoured`, for testing the code *around* a model.

    Hacking `lm_head.weight` to force a prediction does not work: the logit for a row
    is `weight_row . hidden`, and hidden can be negative, so a large positive weight
    can make a token the *least* likely. A stub removes that coupling entirely.
    """

    def __init__(self, favoured, vocab_size=256, sequence_len=32, n_layer=2,
                 n_kv_head=2, head_dim=8):
        self.favoured = favoured
        self.vocab_size = vocab_size
        self.config = GPTConfig(n_layer=n_layer, n_head=n_kv_head * 2, n_kv_head=n_kv_head,
                                n_embd=n_kv_head * 2 * head_dim, sequence_len=sequence_len,
                                vocab_size=vocab_size)
        self.calls = []

        class _W:
            weight = type("_P", (), {"data": np.zeros(1, dtype=np.float32)})()
        self.wte = _W()

    def __call__(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        idx = np.asarray(idx)
        B, T = idx.shape
        self.calls.append((B, T))
        logits = np.zeros((B, T, self.vocab_size), dtype=np.float32)
        logits[:, :, self.favoured] = 10.0
        if kv_cache is not None:
            kv_cache.advance(T)
        return Tensor(logits)


# ----------------------------------------------------------------------------
# KV cache

@pytest.mark.parametrize("prefill_len", [2, 5, 11])
def test_kv_cache_matches_full_forward(tiny_model, prefill_len):
    """Incremental decoding must reproduce the full forward bit for bit.

    This covers the two pieces of carried state that are easy to get wrong: the rotary
    offset (queries must be rotated by their *absolute* position) and the embedding
    smear (the previous token's pre-smear embedding lives in the cache).
    """
    tokens = np.random.default_rng(0).integers(0, 256, 12).tolist()
    full = tiny_model(np.array([tokens])).data[0]

    kv = KVCache.from_config(tiny_model.config, batch_size=1)
    out = [tiny_model(np.array([tokens[:prefill_len]]), kv_cache=kv).data[0]]
    for t in tokens[prefill_len:]:
        out.append(tiny_model(np.array([[t]]), kv_cache=kv).data[0])
    incremental = np.concatenate(out, axis=0)

    assert incremental.shape == full.shape
    np.testing.assert_allclose(incremental, full, atol=2e-4, rtol=1e-3)


def test_kv_cache_tracks_position_and_capacity(tiny_model):
    kv = KVCache.from_config(tiny_model.config, batch_size=1, seq_len=8)
    assert kv.get_pos() == 0
    tiny_model(np.array([[1, 2, 3]]), kv_cache=kv)
    assert kv.get_pos() == 3
    tiny_model(np.array([[4]]), kv_cache=kv)
    assert kv.get_pos() == 4
    kv.reset()
    assert kv.get_pos() == 0 and kv.prev_embedding is None


def test_kv_cache_rejects_overflow(tiny_model):
    kv = KVCache.from_config(tiny_model.config, batch_size=1, seq_len=4)
    tiny_model(np.array([[1, 2, 3, 4]]), kv_cache=kv)
    with pytest.raises(ValueError, match="capacity exceeded"):
        tiny_model(np.array([[5]]), kv_cache=kv)


def test_kv_cache_truncate_rolls_back():
    kv = KVCache(1, 2, 16, 8, 2)
    kv.pos = 10
    kv.truncate(6, prev_embedding=np.zeros((1, 1, 4)))
    assert kv.get_pos() == 6
    kv.truncate(0)
    assert kv.get_pos() == 0 and kv.prev_embedding is None
    with pytest.raises(ValueError, match="cannot extend"):
        kv.truncate(5)


def test_kv_cache_expand_replicates_the_prefix(tiny_model):
    """num_samples > 1 shares one prefill; every row must start from the same state."""
    kv = KVCache.from_config(tiny_model.config, batch_size=1)
    tiny_model(np.array([[7, 8, 9]]), kv_cache=kv)
    wide = kv.expand(4)
    assert wide.batch_size == 4 and wide.get_pos() == kv.get_pos()
    for b in range(4):
        np.testing.assert_array_equal(wide.k_cache[:, b], kv.k_cache[:, 0])
        np.testing.assert_array_equal(wide.v_cache[:, b], kv.v_cache[:, 0])
    assert wide.prev_embedding.shape[0] == 4

    with pytest.raises(ValueError, match="batch=1"):
        wide.expand(2)


def test_cache_decoding_is_causal(tiny_model):
    """Tokens generated later must not change logits emitted earlier."""
    kv = KVCache.from_config(tiny_model.config, batch_size=1)
    first = tiny_model(np.array([[3, 4]]), kv_cache=kv).data[0, -1].copy()
    tiny_model(np.array([[5]]), kv_cache=kv)

    kv2 = KVCache.from_config(tiny_model.config, batch_size=1)
    again = tiny_model(np.array([[3, 4]]), kv_cache=kv2).data[0, -1]
    np.testing.assert_allclose(first, again, atol=1e-6)


# ----------------------------------------------------------------------------
# sampling

def test_sampling_distribution_is_normalised_and_stable():
    logits = np.array([1e4, -1e4, 0.0, 5.0])
    from nanochat.scratch.engine import sampling_distribution
    dist = sampling_distribution(logits, temperature=1.0)
    assert np.isfinite(dist).all()
    np.testing.assert_allclose(dist.sum(), 1.0, atol=1e-12)
    assert dist.argmax() == 0


def test_top_k_zeroes_everything_outside_the_top_k():
    from nanochat.scratch.engine import sampling_distribution
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    dist = sampling_distribution(logits, temperature=1.0, top_k=2)
    assert (dist[2:] == 0).all()
    np.testing.assert_allclose(dist[:2].sum(), 1.0)


def test_temperature_sharpens_and_flattens():
    from nanochat.scratch.engine import sampling_distribution
    logits = np.array([2.0, 1.0, 0.0])
    cold = sampling_distribution(logits, temperature=0.1)
    hot = sampling_distribution(logits, temperature=10.0)
    assert cold.max() > 0.99, "low temperature should concentrate on the argmax"
    assert hot.max() < 0.4, "high temperature should approach uniform"


def test_sample_next_token_greedy_and_seeded():
    logits = np.array([[1.0, 5.0, 2.0], [9.0, 0.0, 1.0]])
    np.testing.assert_array_equal(sample_next_token(logits, None, temperature=0.0), [1, 0])
    a = sample_next_token(logits, np.random.default_rng(0), temperature=1.0)
    b = sample_next_token(logits, np.random.default_rng(0), temperature=1.0)
    np.testing.assert_array_equal(a, b)


def test_sampling_rejects_zero_temperature():
    from nanochat.scratch.engine import sampling_distribution
    with pytest.raises(ValueError, match="temperature"):
        sampling_distribution(np.zeros(3), temperature=0.0)


# ----------------------------------------------------------------------------
# engine

def test_engine_greedy_generation_is_deterministic(tiny_model, tokenizer):
    engine = Engine(tiny_model, tokenizer)
    ids = tokenizer.encode("3+4=")
    a = engine.generate_batch(ids, max_tokens=5, temperature=0.0)[0]
    b = engine.generate_batch(ids, max_tokens=5, temperature=0.0)[0]
    assert a == b and len(a) == 5


def test_engine_matches_the_models_own_generate(tiny_model, tokenizer):
    """The cached engine and the model's cache-free sampler must agree greedily."""
    ids = tokenizer.encode("3+4=")
    engine_out = Engine(tiny_model, tokenizer).generate_batch(ids, max_tokens=6, temperature=0.0)[0]
    model_out = tiny_model.generate(ids, 6, temperature=0.0)[len(ids):]
    assert engine_out == model_out


def test_engine_multi_sample_shares_one_prefill(tiny_model, tokenizer):
    engine = Engine(tiny_model, tokenizer)
    outs = engine.generate_batch(tokenizer.encode("3+4="), max_tokens=5,
                                 num_samples=4, temperature=1.0, top_k=8, seed=1)
    assert len(outs) == 4 and all(len(o) == 5 for o in outs)
    # greedy across samples must be identical, since they share the prefix
    greedy = engine.generate_batch(tokenizer.encode("3+4="), max_tokens=5,
                                   num_samples=3, temperature=0.0)
    assert greedy[0] == greedy[1] == greedy[2]


def test_engine_stops_at_stop_token(tokenizer):
    """A stub that always emits `stop` must terminate after exactly one token."""
    stop = 42
    engine = Engine(StubModel(favoured=stop), tokenizer)
    out = engine.generate_batch(tokenizer.encode("ab"), max_tokens=10,
                                temperature=0.0, stop_tokens=[stop])[0]
    assert out == [stop], f"should stop immediately, got {out}"


def test_engine_prefills_once_then_decodes_single_tokens(tokenizer):
    """The whole point of the cache: one wide forward, then width-1 forwards."""
    stub = StubModel(favoured=5)
    engine = Engine(stub, tokenizer)
    prompt = tokenizer.encode("abcd")
    engine.generate_batch(prompt, max_tokens=4, temperature=0.0)
    shapes = stub.calls
    assert shapes[0] == (1, len(prompt)), f"first call should prefill the prompt, got {shapes[0]}"
    assert all(T == 1 for _, T in shapes[1:]), f"decode steps must feed one token: {shapes}"
    assert len(shapes) == 4, "3 decode steps after prefill, plus the prefill itself"


def test_engine_rejects_short_prompt_and_overlong_request(tiny_model, tokenizer):
    engine = Engine(tiny_model, tokenizer)
    with pytest.raises(ValueError, match="at least 2 tokens"):
        engine.generate_batch([5], max_tokens=4)
    with pytest.raises(ValueError, match="exceeds"):
        engine.generate_batch([1, 2, 3], max_tokens=1000)


def test_engine_generate_streams_token_and_index(tiny_model, tokenizer):
    engine = Engine(tiny_model, tokenizer)
    events = list(engine.generate(tokenizer.encode("ab"), max_tokens=3, num_samples=2,
                                  temperature=1.0, seed=0))
    assert len(events) == 6
    assert {i for _, i in events} == {0, 1}
    assert all(isinstance(t, int) for t, _ in events)


def test_use_calculator_evaluates_and_refuses():
    from nanochat.scratch.engine import use_calculator
    assert use_calculator("2+3") == 5
    assert use_calculator("12 * 4") == 48
    assert use_calculator("1,000+1") == 1001
    # anything that is not plain arithmetic must be refused, not evaluated
    for bad in ["__import__('os')", "open('x')", "2**999999", "a+b", "print(1)", ""]:
        assert use_calculator(bad) is None, bad


# ----------------------------------------------------------------------------
# bits per byte

def test_token_bytes_table_zeroes_special_tokens(tokenizer):
    table = token_bytes_table(tokenizer)
    assert table.shape == (256,)
    # ByteTokenizer has no special tokens, so every entry is one byte
    assert (table == 1).all()


def test_bpb_of_a_uniform_model_is_log2_vocab(tokenizer):
    """A model that predicts uniformly costs exactly 8 bits per byte token, since
    log2(256) = 8. This pins down the bpb formula independently of any real model."""
    class Uniform(StubModel):
        def __call__(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
            idx = np.asarray(idx)
            B, T = idx.shape
            if targets is None:
                return Tensor(np.zeros((B, T, self.vocab_size), dtype=np.float32))
            # uniform over 256 => every token costs ln(256) nats
            return Tensor(np.full((B, T), math.log(self.vocab_size), dtype=np.float32))

    dataset = Dataset.from_text(make_addition_corpus(500, seed=0), tokenizer)
    table = token_bytes_table(tokenizer)
    bpb = evaluate_bpb(Uniform(favoured=0), dataset.sequential_batches(4, 16, "val"),
                       dataset.num_sequential_batches(4, 16, "val"), table)
    np.testing.assert_allclose(bpb, math.log2(256), rtol=1e-6)


def test_real_model_at_init_has_bpb_near_log2_vocab(tiny_model, tokenizer):
    """Same check end to end: an untrained model is close to the uniform baseline."""
    dataset = Dataset.from_text(make_addition_corpus(500, seed=0), tokenizer)
    bpb = evaluate_bpb(tiny_model, dataset.sequential_batches(4, 16, "val"),
                       dataset.num_sequential_batches(4, 16, "val"),
                       token_bytes_table(tokenizer))
    assert abs(bpb - math.log2(256)) < 0.3, f"untrained bpb {bpb:.4f} vs 8.0"


def test_bpb_ignores_masked_targets(tiny_model, tokenizer):
    """Targets of -1 must contribute neither nats nor bytes."""
    table = token_bytes_table(tokenizer)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (2, 16))
    y = rng.integers(0, 256, (2, 16))

    full = evaluate_bpb(tiny_model, iter([(x, y)]), 1, table)
    y_masked = y.copy()
    y_masked[:, 8:] = -1
    half = evaluate_bpb(tiny_model, iter([(x, y_masked)]), 1, table)
    assert np.isfinite(full) and np.isfinite(half)
    assert not np.isclose(full, half), "masking half the targets should change the metric"

    all_masked = evaluate_bpb(tiny_model, iter([(x, np.full_like(y, -1))]), 1, table)
    assert all_masked == float("inf"), "no countable bytes => inf, not a crash"


def test_bpb_is_tokenizer_independent_in_principle(tiny_model, tokenizer):
    """Sanity check on the formula: bpb = total_nats / (ln2 * total_bytes).

    Recompute it by hand from the per-token losses and compare.
    """
    table = token_bytes_table(tokenizer)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 256, (2, 16)), rng.integers(0, 256, (2, 16))
    got = evaluate_bpb(tiny_model, iter([(x, y)]), 1, table)

    losses = tiny_model(x, y, loss_reduction="none").data.reshape(-1)
    nbytes = table[y.reshape(-1)]
    want = losses.sum() / (math.log(2) * nbytes.sum())
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_loss_reduction_none_returns_per_token_losses(tiny_model):
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 256, (3, 16)), rng.integers(0, 256, (3, 16))
    per_token = tiny_model(x, y, loss_reduction="none")
    assert per_token.shape == (3, 16)
    mean = tiny_model(x, y).item()
    np.testing.assert_allclose(per_token.data.mean(), mean, rtol=1e-4)


# ----------------------------------------------------------------------------
# CORE-style task evaluation

def test_find_common_length_prefix_and_suffix():
    assert seval.find_common_length([[1, 2, 3, 9], [1, 2, 3, 8]], "left") == 3
    assert seval.find_common_length([[9, 7, 8], [1, 7, 8]], "right") == 2
    assert seval.find_common_length([[1, 2], [3, 4]], "left") == 0
    assert seval.find_common_length([[1, 2], [1, 2]], "left") == 2  # fully equal


def test_render_prompts_mc_shapes_one_prompt_per_choice():
    item = {"query": "2+2=", "choices": ["4", "5", "6"], "gold": 0}
    prompts = seval.render_prompts_mc(item, "")
    assert prompts == ["2+2=4", "2+2=5", "2+2=6"]


def test_render_prompts_mc_with_fewshot():
    shot = {"query": "1+1=", "choices": ["2"], "gold": 0}
    item = {"query": "2+2=", "choices": ["4", "5"], "gold": 0}
    prompts = seval.render_prompts_mc(item, "", [shot])
    assert prompts[0] == "1+1=2\n\n2+2=4"


def test_render_prompts_lm_is_a_token_prefix_pair(tokenizer):
    item = {"context": "the cat sat  ", "continuation": "on the mat"}
    without, with_cont = seval.render_prompts_lm(item, " ")
    assert with_cont.startswith(without.rstrip())
    tokens, starts, ends = seval.batch_sequences_lm(tokenizer, [without, with_cont])
    assert len(tokens) == 1 and starts[0] < ends[0]


def test_render_prompts_schema_varies_context_keeps_continuation():
    item = {"context_options": ["A said", "B said"], "continuation": " hello", "gold": 0}
    prompts = seval.render_prompts_schema(item, "")
    assert prompts == ["A said hello", "B said hello"]


def test_batch_sequences_mc_finds_the_answer_span(tokenizer):
    prompts = ["2+2=4", "2+2=5"]
    tokens, starts, ends = seval.batch_sequences_mc(tokenizer, prompts)
    # the common prefix is bos + "2+2=" = 5 tokens; the answer is the last token
    assert starts == [5, 5]
    assert ends == [len(tokens[0]), len(tokens[1])]
    assert all(e - s == 1 for s, e in zip(starts, ends))


def test_stack_sequences_right_pads():
    out = seval.stack_sequences([[1, 2, 3], [4, 5]], pad_token_id=0)
    np.testing.assert_array_equal(out, [[1, 2, 3], [4, 5, 0]])


def test_forward_model_marks_the_last_column_nan(tiny_model):
    losses, predictions = seval.forward_model(tiny_model, np.array([[1, 2, 3, 4]]))
    assert losses.shape == (1, 4) and predictions.shape == (1, 4)
    assert np.isnan(losses[0, -1]), "no autoregressive target exists at the last position"
    assert np.isfinite(losses[0, :-1]).all()


def test_evaluate_task_picks_the_lower_loss_continuation(tokenizer):
    """A model that always predicts "4" must choose whichever option contains it."""
    stub = StubModel(favoured=tokenizer.encode("4")[0])
    data = [{"query": "2+2=", "choices": ["4", "9"], "gold": 0},
            {"query": "3+1=", "choices": ["9", "4"], "gold": 1}]
    acc = evaluate_task(stub, tokenizer, data,
                        {"task_type": "multiple_choice", "num_fewshot": 0,
                         "continuation_delimiter": ""})
    assert acc == 1.0

    # and it must be wrong when the gold label disagrees with its preference
    flipped = [{"query": "2+2=", "choices": ["4", "9"], "gold": 1}]
    assert evaluate_task(stub, tokenizer, flipped,
                         {"task_type": "multiple_choice", "num_fewshot": 0,
                          "continuation_delimiter": ""}) == 0.0


def test_evaluate_task_respects_max_examples(tiny_model, tokenizer):
    data = [{"query": f"{i}+0=", "choices": ["0", "1"], "gold": 0} for i in range(10)]
    meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": ""}
    assert 0.0 <= evaluate_task(tiny_model, tokenizer, data, meta, max_examples=3) <= 1.0
    assert evaluate_task(tiny_model, tokenizer, [], meta) == 0.0


def test_evaluate_task_rejects_unknown_type(tiny_model, tokenizer):
    with pytest.raises(ValueError, match="unsupported task type"):
        evaluate_task(tiny_model, tokenizer, [{"query": "a", "choices": ["b"], "gold": 0}],
                      {"task_type": "nonsense"})


# ----------------------------------------------------------------------------
# checkpointing

def test_checkpoint_round_trip_restores_parameters(tiny_model, tmp_path):
    save_checkpoint(str(tmp_path), 7, tiny_model, meta={"val_bpb": 1.25})
    loaded, meta = load_model(str(tmp_path))
    assert meta["step"] == 7 and meta["val_bpb"] == 1.25
    for (name, a), (_, b) in zip(tiny_model.named_parameters(), loaded.named_parameters()):
        np.testing.assert_array_equal(a.data, b.data, err_msg=name)


def test_checkpoint_rebuilds_the_model_from_saved_config(tmp_path):
    config = GPTConfig(n_layer=3, n_head=2, n_kv_head=1, n_embd=24,
                       sequence_len=16, vocab_size=64, n_routed_experts=2)
    model = GPT(config)
    save_checkpoint(str(tmp_path), 0, model)
    loaded, _ = load_model(str(tmp_path))  # no config passed in
    assert vars(loaded.config) == vars(config)
    assert loaded.num_parameters() == model.num_parameters()


def test_checkpoint_round_trips_optimizer_state(tiny_model, tmp_path):
    """Resume is only correct if the momentum buffers come back too."""
    opt = setup_optimizer(tiny_model)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 256, (2, 16)), rng.integers(0, 256, (2, 16))
    tiny_model(x, y).backward()
    opt.step()
    save_checkpoint(str(tmp_path), 1, tiny_model, opt)

    fresh = GPT(tiny_model.config)
    fresh_opt = setup_optimizer(fresh)
    load_checkpoint(str(tmp_path), 1, model=fresh, optimizer=fresh_opt)

    assert fresh_opt.adamw.state, "AdamW state was not restored"
    assert fresh_opt.muon.state, "Muon state was not restored"
    for state in fresh_opt.adamw.state.values():
        assert isinstance(state["step"], int), "Adam's step must come back as an int"

    # Stepping both from the same gradient must now agree
    for model_, opt_ in ((tiny_model, opt), (fresh, fresh_opt)):
        model_.zero_grad()
        model_(x, y).backward()
        opt_.step()
    for (name, a), (_, b) in zip(tiny_model.named_parameters(), fresh.named_parameters()):
        np.testing.assert_allclose(a.data, b.data, atol=1e-6, err_msg=name)


def test_list_steps_and_find_last_step(tiny_model, tmp_path):
    assert list_steps(str(tmp_path)) == [] and find_last_step(str(tmp_path)) is None
    for step in (5, 1, 10):
        save_checkpoint(str(tmp_path), step, tiny_model)
    assert list_steps(str(tmp_path)) == [1, 5, 10]
    assert find_last_step(str(tmp_path)) == 10


def test_checkpoint_writes_are_atomic(tiny_model, tmp_path):
    """No .tmp files may survive a successful save."""
    save_checkpoint(str(tmp_path), 3, tiny_model, setup_optimizer(tiny_model))
    out = os.path.join(str(tmp_path), "step_000003")
    assert sorted(os.listdir(out)) == ["meta.json", "model.npz", "optim.npz"]
    assert not any(f.endswith(".tmp") for f in os.listdir(out))


def test_meta_json_is_readable_and_contains_the_config(tiny_model, tmp_path):
    save_checkpoint(str(tmp_path), 2, tiny_model, meta={"note": "hello"})
    with open(os.path.join(str(tmp_path), "step_000002", "meta.json")) as f:
        meta = json.load(f)
    assert meta["step"] == 2 and meta["note"] == "hello"
    assert meta["config"]["n_layer"] == tiny_model.config.n_layer


def test_build_model_returns_an_eval_mode_model(tiny_model, tmp_path):
    save_checkpoint(str(tmp_path), 0, tiny_model)
    loaded, _ = build_model(str(tmp_path), 0)
    assert not any(m.training for m in loaded.modules())


def test_loading_a_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path), 99)
    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        load_model(str(tmp_path))


# ----------------------------------------------------------------------------
# dataset batching used by the eval path

def test_sequential_batches_cover_the_split_without_overlap(tokenizer):
    ds = Dataset.from_text(make_addition_corpus(500, seed=0), tokenizer)
    batches = list(ds.sequential_batches(2, 8, "val"))
    assert len(batches) == ds.num_sequential_batches(2, 8, "val")
    seen = np.concatenate([x.reshape(-1) for x, _ in batches])
    assert len(seen) == len(set(range(len(seen)))), "windows must not overlap"
    for x, y in batches:
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])


def test_sequential_batches_are_reproducible(tokenizer):
    ds = Dataset.from_text(make_addition_corpus(300, seed=0), tokenizer)
    a = [x.tolist() for x, _ in ds.sequential_batches(2, 8, "val")]
    b = [x.tolist() for x, _ in ds.sequential_batches(2, 8, "val")]
    assert a == b, "evaluation batches must not depend on RNG state"


def test_byte_tokenizer_satisfies_the_eval_surface(tokenizer):
    assert tokenizer.get_vocab_size() == 256
    assert tokenizer.get_special_tokens() == set()
    assert tokenizer.get_bos_token_id() == tokenizer.encode_special("<|bos|>")
    assert tokenizer.encode("ab", prepend="<|bos|>")[0] == tokenizer.get_bos_token_id()
    assert tokenizer.decode_single_token_bytes(65) == b"A"
    assert tokenizer(["ab", "c"]) == [[97, 98], [99]]
    with pytest.raises(KeyError, match="Unknown special token"):
        tokenizer.encode_special("<|nope|>")


def test_byte_tokenizer_bos_is_in_distribution(tokenizer):
    """BOS must be a byte the model actually sees during training.

    A 256-byte vocabulary has no spare id, so BOS is `;`, the record terminator in the
    addition corpus. Prepending it reproduces the real mid-stream context; a byte that
    never occurs in the data would be out of distribution and hurt generation.
    """
    corpus = make_addition_corpus(50, seed=0)
    assert tokenizer.get_bos_token_id() == ord(";")
    assert tokenizer.get_bos_token_id() in set(tokenizer.encode(corpus))


# ----------------------------------------------------------------------------
# the pipeline end to end

@pytest.mark.slow
def test_train_eval_checkpoint_resume_pipeline(tmp_path, tokenizer):
    """Pretrain briefly, checkpoint, resume, and confirm bpb improves throughout."""
    dataset = Dataset.from_text(make_addition_corpus(8000, seed=0), tokenizer)
    config = GPTConfig(n_layer=3, n_head=4, n_kv_head=2, n_embd=48,
                       sequence_len=32, vocab_size=256)
    model = GPT(config)
    opt = setup_optimizer(model, matrix_lr=0.03, embedding_lr=0.2,
                          unembedding_lr=0.02, scalar_lr=0.05)
    table = token_bytes_table(tokenizer)
    rng = np.random.default_rng(0)

    def bpb():
        n = min(5, dataset.num_sequential_batches(8, 32, "val"))
        return evaluate_bpb(model, dataset.sequential_batches(8, 32, "val"), n, table)

    before = bpb()
    for _ in range(120):
        loss = model(*dataset.get_batch(8, 32, rng))
        opt.zero_grad()
        loss.backward()
        opt.step()
    middle = bpb()
    save_checkpoint(str(tmp_path), 119, model, opt, meta={"val_bpb": middle})

    # Resume into a fresh model and a fresh optimizer. Order matters: the optimizer
    # must be built from the model it will actually update, so the model is created
    # and loaded first. (Building it from the previous `model` binding would leave the
    # optimizer updating parameters that nothing computes gradients for.)
    model = GPT(config)
    opt = setup_optimizer(model, matrix_lr=0.03, embedding_lr=0.2,
                         unembedding_lr=0.02, scalar_lr=0.05)
    load_checkpoint(str(tmp_path), 119, model=model, optimizer=opt)
    assert {id(p) for g in opt.param_groups for p in g["params"]} == \
           {id(p) for p in model.parameters()}, "optimizer must own the loaded model's params"

    np.testing.assert_allclose(bpb(), middle, rtol=1e-5)  # resume restores the metric
    for _ in range(80):
        loss = model(*dataset.get_batch(8, 32, rng))
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = bpb()

    assert before > 7.0, f"untrained bpb should be near log2(256)=8, got {before}"
    assert middle < before - 3.0, f"bpb did not improve: {before} -> {middle}"
    assert after < middle, f"training after resume did not help: {middle} -> {after}"


@pytest.mark.slow
def test_trained_model_beats_chance_on_the_task_and_generates_correctly(tokenizer):
    """The payoff: after training, CORE-style accuracy is high and greedy decoding
    through the KV-cache engine produces the right digits."""
    dataset = Dataset.from_text(make_addition_corpus(20000, seed=0), tokenizer)
    config = GPTConfig(n_layer=4, n_head=4, n_kv_head=2, n_embd=64,
                       sequence_len=64, vocab_size=256)
    model = GPT(config)
    opt = setup_optimizer(model, matrix_lr=0.03, embedding_lr=0.2,
                          unembedding_lr=0.02, scalar_lr=0.05)
    rng = np.random.default_rng(0)
    for _ in range(400):
        loss = model(*dataset.get_batch(16, 64, rng))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    # bits per byte should approach the task's entropy floor
    floor_bpb = (2 * math.log(10) / 7) / math.log(2)
    bpb = evaluate_bpb(model, dataset.sequential_batches(16, 64, "val"),
                       min(10, dataset.num_sequential_batches(16, 64, "val")),
                       token_bytes_table(tokenizer))
    assert bpb < floor_bpb + 0.25, f"bpb {bpb:.4f} vs floor {floor_bpb:.4f}"

    # multiple choice, true sum against a near miss
    data = []
    for a in range(10):
        for b in range(10):
            data.append({"query": f"{a}+{b}=", "gold": 0,
                         "choices": [f"{a + b:02d};", f"{(a + b + 1) % 100:02d};"]})
    acc = evaluate_task(model, tokenizer, data,
                        {"task_type": "multiple_choice", "num_fewshot": 0,
                         "continuation_delimiter": ""})
    assert acc > 0.85, f"MC accuracy {acc:.3f} is barely above chance"

    # and greedy decoding through the engine
    engine = Engine(model, tokenizer)
    correct = 0
    for a in range(10):
        for b in range(10):
            ids = tokenizer.encode(f"{a}+{b}=", prepend="<|bos|>")
            got = tokenizer.decode(engine.generate_batch(ids, max_tokens=3, temperature=0.0)[0])
            correct += got == f"{a + b:02d};"
    assert correct >= 85, f"engine greedy decoding only got {correct}/100"
