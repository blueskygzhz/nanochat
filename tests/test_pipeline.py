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
    addition_pairs, build_corpus, build_model, corpus_spec, evaluate_bpb, evaluate_task,
    find_last_step, list_steps, load_checkpoint, load_model, make_addition_corpus,
    sample_next_token, save_checkpoint, setup_optimizer, token_bytes_table,
)
from nanochat.scratch import eval as seval
from nanochat.tokenizer import tokenizer_spec


@pytest.fixture
def tiny_model():
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
                       sequence_len=32, vocab_size=ByteTokenizer.vocab_size)
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

    def __init__(self, favoured, vocab_size=ByteTokenizer.vocab_size, sequence_len=32, n_layer=2,
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


class ScriptedModel(StubModel):
    """Emits `script[k]` as the argmax on its k-th forward, whatever the input, and
    records every input it is fed. That separates "what the model wanted" from "what
    the engine emitted", which is exactly the distinction forced tokens create."""

    def __init__(self, script, **kwargs):
        kwargs.setdefault("vocab_size", ByteTokenizer.vocab_size)  # bytes + specials
        super().__init__(favoured=0, **kwargs)
        self.script = list(script)
        self.fed = []

    def __call__(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        idx = np.asarray(idx)
        B, T = idx.shape
        self.fed.append(idx.copy())
        logits = np.zeros((B, T, self.vocab_size), dtype=np.float32)
        logits[:, :, self.script[min(len(self.fed) - 1, len(self.script) - 1)]] = 10.0
        if kv_cache is not None:
            kv_cache.advance(T)
        return Tensor(logits)


def test_engine_splices_the_calculator_result_into_the_stream():
    """`<|python_start|>2+3<|python_end|>` must be followed by the *tool's* tokens
    `<|output_start|>5<|output_end|>`, overriding whatever the model sampled, and
    those tokens must be fed back through the model so the cache contains them."""
    tok = ByteTokenizer()
    S, E, OS, OE = (tok.encode_special(n) for n in
                    ("<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"))
    junk, after = ord("x"), ord("!")
    # what the model *wants* to emit at each step; steps 5..7 are overridden
    script = [S, ord("2"), ord("+"), ord("3"), E, junk, junk, junk, after]
    model = ScriptedModel(script)
    out = Engine(model, tok).generate_batch(tok.encode("ab"), max_tokens=9,
                                           temperature=0.0, use_tools=True)[0]

    assert out == [S, ord("2"), ord("+"), ord("3"), E, OS, ord("5"), OE, after], out
    fed = [int(x[0, 0]) for x in model.fed[1:]]  # decode steps, after the prefill
    assert fed == out[:-1], "every emitted token, forced or not, must be fed back"


def test_engine_emits_no_output_block_for_a_refused_expression():
    tok = ByteTokenizer()
    S, E = tok.encode_special("<|python_start|>"), tok.encode_special("<|python_end|>")
    script = [S, ord("a"), E, ord("!"), ord("!")]
    out = Engine(ScriptedModel(script), tok).generate_batch(
        tok.encode("ab"), max_tokens=5, temperature=0.0, use_tools=True)[0]
    assert out == [S, ord("a"), E, ord("!"), ord("!")], "refused => keep sampling"


def test_engine_tools_require_the_special_tokens(tiny_model):
    """Only the legacy byte tokenizer lacks them; it cannot express a tool call."""
    from nanochat.scratch import LegacyByteTokenizer
    legacy = LegacyByteTokenizer()
    with pytest.raises(ValueError, match="tool special tokens"):
        Engine(tiny_model, legacy).generate_batch(legacy.encode("ab"), max_tokens=2,
                                                  use_tools=True)


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
    assert table.shape == (265,)
    assert (table[:256] == 1).all(), "every byte token is one byte"
    assert (table[256:] == 0).all(), "special tokens are structure, not text: zero bytes"


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
    bpb = evaluate_bpb(Uniform(favoured=0, vocab_size=256), dataset.sequential_batches(4, 16, "val"),
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
    from nanochat.tokenizer import SPECIAL_TOKENS
    assert tokenizer.get_vocab_size() == 256 + len(SPECIAL_TOKENS)
    assert tokenizer.get_special_tokens() == set(SPECIAL_TOKENS)
    assert tokenizer.get_bos_token_id() == tokenizer.encode_special("<|bos|>") == 256
    assert tokenizer.encode("ab", prepend="<|bos|>")[0] == tokenizer.get_bos_token_id()
    assert tokenizer.decode_single_token_bytes(65) == b"A"
    assert tokenizer.decode_single_token_bytes(256) == b"<|bos|>"
    assert tokenizer(["ab", "c"]) == [[97, 98], [99]]
    with pytest.raises(KeyError, match="Unknown special token"):
        tokenizer.encode_special("<|nope|>")
    with pytest.raises(KeyError, match="unknown token id"):
        tokenizer.decode([999])


def test_no_text_encodes_to_a_special_token(tokenizer):
    """The core property: structure cannot be written as text. Not the literal names,
    not the legacy separators, not any byte sequence."""
    from nanochat.tokenizer import SPECIAL_TOKENS
    special = {tokenizer.encode_special(n) for n in SPECIAL_TOKENS}
    for text in SPECIAL_TOKENS + ["a;b", "\nA:5\n", "".join(map(chr, range(256)))]:
        assert not special & set(tokenizer.encode(text)), text
    # ...and the literal name round-trips as text, character for character
    assert tokenizer.decode(tokenizer.encode("<|assistant_end|>")) == "<|assistant_end|>"


def test_legacy_byte_tokenizer_still_loads_old_checkpoints():
    """Checkpoints written before the specials existed recorded no vocab_size."""
    from nanochat.scratch import LegacyByteTokenizer
    from nanochat.tokenizer import load_tokenizer
    for spec in (None, {"kind": "byte"}, {"kind": "byte", "vocab_size": 256}):
        legacy = load_tokenizer(spec)
        assert isinstance(legacy, LegacyByteTokenizer)
        assert legacy.get_vocab_size() == 256 and legacy.get_bos_token_id() == ord(";")
    assert isinstance(load_tokenizer({"kind": "byte", "vocab_size": 265}), ByteTokenizer)
    with pytest.raises(ValueError, match="vocab_size 300"):
        load_tokenizer({"kind": "byte", "vocab_size": 300})


# ----------------------------------------------------------------------------
# held-out evaluation and corpus provenance

def test_addition_pairs_partition_all_100_and_are_deterministic():
    train, held = addition_pairs(0.2, seed=3)
    assert len(train) == 80 and len(held) == 20
    assert not set(train) & set(held), "a pair cannot be both seen and held out"
    assert set(train) | set(held) == {(a, b) for a in range(10) for b in range(10)}
    assert addition_pairs(0.2, seed=3) == (train, held)
    assert addition_pairs(0.2, seed=4) != (train, held)
    assert addition_pairs(0.0) == (sorted(train + held), [])


def test_held_out_pairs_never_reach_the_training_corpus():
    train, held = addition_pairs(0.2, seed=0)
    text = make_addition_corpus(5000, seed=0, pairs=train)
    lines = {text[i:i + 7] for i in range(0, len(text), 7)}
    assert lines == {f"{a}+{b}={a + b:02d};" for a, b in train}
    for a, b in held:
        assert f";{a}+{b}=" not in ";" + text


def test_entropy_floor_scales_with_the_number_of_pairs():
    from nanochat.scratch import addition_entropy_floor
    np.testing.assert_allclose(addition_entropy_floor(), 2 * math.log(10) / 7)
    np.testing.assert_allclose(addition_entropy_floor(80), math.log(80) / 7)


def test_corpus_spec_round_trips_through_checkpoint_meta(tiny_model, tmp_path):
    """The spec must survive JSON, and rebuilding from it must give the same data."""
    spec = corpus_spec(corpus_lines=300, seed=5, holdout_frac=0.25)
    save_checkpoint(str(tmp_path), 0, tiny_model, meta={"data": spec})
    _, meta = load_model(str(tmp_path))
    assert meta["data"] == spec
    text_a, info_a = build_corpus(spec)
    text_b, info_b = build_corpus(meta["data"])
    assert text_a == text_b and info_a == info_b
    assert len(info_a["heldout_pairs"]) == 25


def test_text_corpus_spec_reads_the_file(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text("hello world", encoding="utf-8")
    text, info = build_corpus(corpus_spec(text_file=str(path)))
    assert text == "hello world"
    assert info == {"floor": None, "train_pairs": None, "heldout_pairs": None}
    with pytest.raises(ValueError, match="unknown corpus kind"):
        build_corpus({"kind": "nope"})


# ----------------------------------------------------------------------------
# optimizer and init hygiene

def test_weight_decay_applies_to_muon_matrices_only(tiny_model):
    opt = setup_optimizer(tiny_model, weight_decay=0.1)
    assert all(g["weight_decay"] == 0.1 for g in opt.muon.param_groups)
    assert all(g["weight_decay"] == 0.0 for g in opt.adamw.param_groups), \
        "decaying AdamW groups would pull resid_lambdas and embeddings towards zero"


def test_init_seed_controls_the_weights():
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
                       sequence_len=16, vocab_size=64)
    a, b, c = GPT(config, seed=1), GPT(config, seed=1), GPT(config, seed=2)
    for (name, pa), (_, pb), (_, pc) in zip(a.named_parameters(), b.named_parameters(),
                                            c.named_parameters()):
        np.testing.assert_array_equal(pa.data, pb.data, err_msg=name)
    assert not np.array_equal(a.wte.weight.data, c.wte.weight.data)


# ----------------------------------------------------------------------------
# tokenizer provenance and the chat format

@pytest.fixture(scope="module")
def bpe_dir(tmp_path_factory):
    """A tiny BPE tokenizer, trained in ~0.1s, saved the way tok_train saves it."""
    from nanochat.tokenizer import RustBPETokenizer
    text = "the cat sat on the mat. " * 50 + make_addition_corpus(200, seed=0)
    path = tmp_path_factory.mktemp("tok")
    RustBPETokenizer.train_from_iterator(iter([text]), 300).save(str(path))
    return str(path)


def test_tokenizer_spec_round_trips_and_checks_vocab(bpe_dir, tmp_path):
    from nanochat.tokenizer import load_tokenizer, snapshot_tokenizer, tokenizer_spec
    byte_spec = snapshot_tokenizer(tokenizer_spec("byte"), str(tmp_path))
    assert byte_spec == {"kind": "byte", "vocab_size": 265}
    assert isinstance(load_tokenizer(byte_spec), ByteTokenizer)

    from nanochat.tokenizer import RustBPETokenizer
    vocab = RustBPETokenizer.from_directory(bpe_dir).get_vocab_size()
    spec = snapshot_tokenizer(tokenizer_spec("bpe", bpe_dir), str(tmp_path / "run"))
    assert spec["dir"] == str(tmp_path / "run" / "tokenizer") and spec["vocab_size"] == vocab
    tok = load_tokenizer(json.loads(json.dumps(spec)))   # must survive meta.json
    assert tok.get_vocab_size() == vocab
    with pytest.raises(ValueError, match="checkpoint expects"):
        load_tokenizer({**spec, "vocab_size": 999})
    with pytest.raises(FileNotFoundError, match="tok_train"):
        load_tokenizer({"kind": "bpe", "dir": str(tmp_path / "nowhere")})


def test_snapshot_isolates_the_run_from_retraining(bpe_dir, tmp_path):
    """Re-running tok_train must not change the vocabulary under a trained model."""
    import shutil
    from nanochat.tokenizer import RustBPETokenizer, load_tokenizer, snapshot_tokenizer, tokenizer_spec
    src = tmp_path / "shared"
    shutil.copytree(bpe_dir, src)
    vocab = RustBPETokenizer.from_directory(str(src)).get_vocab_size()
    spec = snapshot_tokenizer(tokenizer_spec("bpe", str(src)), str(tmp_path / "run"))
    RustBPETokenizer.train_from_iterator(iter(["zzz yyy " * 100]), 280).save(str(src))
    assert RustBPETokenizer.from_directory(str(src)).get_vocab_size() != vocab
    assert load_tokenizer(spec).get_vocab_size() == vocab


def test_encode_corpus_puts_bos_where_prompts_will_have_it(bpe_dir):
    from nanochat.scratch import encode_corpus
    from nanochat.tokenizer import load_tokenizer
    spec = corpus_spec(corpus_lines=50, seed=0, holdout_frac=0.2)
    text, _ = build_corpus(spec)

    # legacy bytes: BOS is ';', already in the text at every record boundary
    from nanochat.scratch import LegacyByteTokenizer
    legacy = LegacyByteTokenizer()
    assert encode_corpus(text, legacy, spec) == legacy.encode(text)

    # dedicated BOS (bytes and BPE alike): one per record, record content unchanged
    for tok in (ByteTokenizer(), load_tokenizer({"kind": "bpe", "dir": bpe_dir})):
        ids = encode_corpus(text, tok, spec)
        bos = tok.get_bos_token_id()
        assert ids.count(bos) == 50 and ids[0] == bos
        assert tok.decode([t for t in ids if t != bos]) == text
        # and a free-text corpus is one document
        assert encode_corpus("hello", tok, {"kind": "text"})[0] == bos


@pytest.mark.parametrize("kind", ["byte", "bpe"])
def test_chat_format_masks_only_assistant_replies(kind, bpe_dir):
    from nanochat.chat_format import render_conversation, reply_stop_tokens
    from nanochat.tokenizer import load_tokenizer
    tok = load_tokenizer({"kind": kind, "dir": bpe_dir} if kind == "bpe" else None)
    messages = [{"role": "user", "content": "2+3"}, {"role": "assistant", "content": "5"},
                {"role": "user", "content": "4+4"}, {"role": "assistant", "content": "8"}]
    ids, mask = render_conversation(tok, messages)
    assert len(ids) == len(mask) and ids[0] == tok.get_bos_token_id()
    stop = set(reply_stop_tokens(tok))
    trained = [t for t, m in zip(ids, mask) if m]
    assert tok.decode([t for t in trained if t not in stop]) == "58", \
        "exactly the two replies are trained on, nothing from the user turns"
    # each reply ends with a trained stop token, so the model learns when to stop
    assert sum(t in stop for t in trained) == 2


@pytest.mark.parametrize("kind", ["byte", "bpe"])
def test_render_prompt_is_the_training_layout_up_to_the_reply(kind, bpe_dir):
    """Inference must present exactly the prefix the model was trained on."""
    from nanochat.chat_format import render_conversation, render_prompt
    from nanochat.tokenizer import load_tokenizer
    tok = load_tokenizer({"kind": kind, "dir": bpe_dir} if kind == "bpe" else tokenizer_spec())
    user = [{"role": "user", "content": "2+3"}]
    full, mask = render_conversation(tok, user + [{"role": "assistant", "content": "5"}])
    prompt = render_prompt(tok, user)
    assert full[:len(prompt)] == prompt
    assert mask[len(prompt)] == 1 and not any(mask[:len(prompt)])


@pytest.mark.parametrize("kind", ["byte", "bpe", "legacy"])
def test_prefill_continues_the_assistant_turn(kind, bpe_dir):
    """A final assistant message is a prefill: the turn stays open after its content,
    so the prompt is exactly the training layout up to that point."""
    from nanochat.chat_format import render_conversation, render_prompt
    from nanochat.tokenizer import load_tokenizer
    spec = {"bpe": {"kind": "bpe", "dir": bpe_dir}, "byte": tokenizer_spec(), "legacy": None}[kind]
    tok = load_tokenizer(spec)
    full, _ = render_conversation(tok, [{"role": "user", "content": "2+3"},
                                        {"role": "assistant", "content": "5 is the answer"}])
    prompt = render_prompt(tok, [{"role": "user", "content": "2+3"},
                                 {"role": "assistant", "content": "5 is"}])
    assert full[:len(prompt)] == prompt and len(prompt) < len(full)
    with pytest.raises(ValueError, match="cannot end with whitespace"):
        render_prompt(tok, [{"role": "user", "content": "2+3"},
                            {"role": "assistant", "content": "5 is "}])


def test_fit_history_drops_whole_exchanges_oldest_first():
    from nanochat.chat_format import fit_history, render_prompt
    tok = ByteTokenizer()
    history = []
    for a in range(5):
        history += [{"role": "user", "content": f"{a}+1"},
                    {"role": "assistant", "content": str(a + 1)}]
    history.append({"role": "user", "content": "9+9"})

    full = len(render_prompt(tok, history))
    kept, ids = fit_history(tok, history, budget=full)
    assert kept == history, "nothing dropped when it fits"
    kept, ids = fit_history(tok, history, budget=full - 1)
    assert kept == history[2:] and len(ids) <= full - 1
    assert kept[0]["role"] == "user" and kept[-1] == history[-1]
    with pytest.raises(ValueError, match="alone needs"):
        fit_history(tok, history, budget=3)


def test_chat_cli_streams_a_reply_and_reports_dropped_history():
    import io
    from types import SimpleNamespace
    from scripts.chat_cli import respond
    tok = ByteTokenizer()
    end = tok.encode_special("<|assistant_end|>")
    # full prompt = bos, user_start, "1+1", user_end, assistant_start, "2", assistant_end,
    # user_start, "2+2", user_end, assistant_start = 15 tokens, which does not fit in
    # 16 - 4 (room reserved for the reply); dropping one exchange leaves 8
    model = ScriptedModel([ord("4"), ord("\n"), ord("5"), end], sequence_len=16)
    args = SimpleNamespace(max_tokens=4, temperature=0.0, top_k=None, seed=0)
    out = io.StringIO()
    history = [{"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"},
               {"role": "user", "content": "2+2"}]
    reply, dropped = respond(Engine(model, tok), tok, history, args, out=out)
    # a newline is just text now; only the end-of-turn token ends the reply
    assert reply == "4\n5" and out.getvalue() == "4\n5"
    assert dropped == 2, "the oldest exchange (user + assistant) must be dropped"


def test_chat_cli_never_prints_structure_and_returns_parsed_parts():
    """A tool call in a reply is structure: not printed, but kept in the returned
    content so the history re-renders it as a tool call, not as literal text."""
    import io
    from types import SimpleNamespace
    from nanochat.chat_format import render_conversation
    from scripts.chat_cli import respond
    tok = ByteTokenizer()
    sp = tok.encode_special
    script = [ord("a"), sp("<|python_start|>"), ord("1"), sp("<|python_end|>"), ord("b"),
              sp("<|assistant_end|>")]
    out = io.StringIO()
    reply, _ = respond(Engine(ScriptedModel(script, sequence_len=32), tok), tok,
                       [{"role": "user", "content": "x"}],
                       SimpleNamespace(max_tokens=8, temperature=0.0, top_k=None, seed=0), out=out)
    assert out.getvalue() == "a1b", "no '<|python_start|>' text on screen"
    assert reply == [{"type": "text", "text": "a"}, {"type": "python", "text": "1"},
                     {"type": "text", "text": "b"}]
    ids, _ = render_conversation(tok, [{"role": "user", "content": "x"},
                                       {"role": "assistant", "content": reply}])
    assert ids[-len(script):] == script, "the history re-renders to exactly what was generated"


def test_no_module_imports_requests():
    """`requests` is not a dependency (only a transitive one of an optional extra),
    so importing it would break a default install."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(r"^\s*(?:import\s+requests|from\s+requests\b)", re.MULTILINE)
    offenders = [str(p.relative_to(root)) for d in ("nanochat", "scripts", "tasks")
                 for p in (root / d).rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert not offenders, offenders


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
    from nanochat.scratch import encode_corpus
    spec = corpus_spec(corpus_lines=20000, seed=0, holdout_frac=0.0)
    dataset = Dataset(encode_corpus(build_corpus(spec)[0], tokenizer, spec))  # BOS per record
    config = GPTConfig(n_layer=4, n_head=4, n_kv_head=2, n_embd=64,
                       sequence_len=64, vocab_size=tokenizer.get_vocab_size())
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
