"""
Tests for the post-training machinery: nanochat/scratch/rlhf.py (preference model,
PPO) and nanochat/constitution.py (principles, AI feedback).

Each quantity is checked against a closed form or a hand-computed value, so the tests
pin down the algorithms independently of whether any training run succeeds.
"""

import math

import numpy as np
import pytest

from nanochat.constitution import (
    ARITHMETIC_CONSTITUTION, COT_CLAMP, MC_TEMPLATE, LMFeedback, Response, RuleFeedback,
    clamp_label, format_conversation, ranking_to_comparisons, sample_principle,
)
from nanochat.scratch import ByteTokenizer, Engine, GPT, GPTConfig, Tensor
from nanochat.scratch import tensor as st
from nanochat.scratch.rlhf import (
    RewardModel, build_ppo_batch, clone_model, gae, kl_penalized_rewards,
    load_reward_model, make_trunk_head_optimizer, pad_sequences, ppo_policy_loss,
    preference_loss, rollout, save_reward_model, sequence_logprobs, value_loss, whiten,
)

TOK = ByteTokenizer()
PAD = TOK.get_bos_token_id()
PRINCIPLES = {p.name: p for p in ARITHMETIC_CONSTITUTION}


def tiny_config(**kw):
    return GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=32, sequence_len=32,
                     vocab_size=TOK.vocab_size, **kw)


@pytest.fixture
def float64_engine():
    st.set_dtype(np.float64)
    yield
    st.set_dtype(np.float32)


# ----------------------------------------------------------------------------
# preference model

def test_soft_bradley_terry_loss_matches_the_closed_form(float64_engine):
    rm = RewardModel(tiny_config())
    rm.head.weight.data = np.random.default_rng(0).standard_normal(rm.head.weight.shape) * 0.5
    a = [TOK.encode("ab"), TOK.encode("abcd")]
    b = [TOK.encode("xy"), TOK.encode("x")]
    p = np.array([0.8, 0.25])
    loss, _, margin = preference_loss(rm, a, b, p, PAD)

    def log_sigmoid(z):
        return -np.log1p(np.exp(-z))
    want = -np.mean(p * log_sigmoid(margin) + (1 - p) * log_sigmoid(-margin))
    np.testing.assert_allclose(loss.item(), want, rtol=1e-10)


def test_hard_labels_reduce_to_the_hh_paper_loss(float64_engine):
    """With p = 1 the loss is log(1 + exp(r_bad - r_good)), as in arXiv:2204.05862."""
    rm = RewardModel(tiny_config())
    rm.head.weight.data = np.random.default_rng(1).standard_normal(rm.head.weight.shape)
    loss, _, margin = preference_loss(rm, [TOK.encode("good")], [TOK.encode("bad")], [1.0], PAD)
    np.testing.assert_allclose(loss.item(), math.log1p(math.exp(-margin[0])), rtol=1e-10)


def test_untrained_pm_scores_everything_zero_so_loss_is_ln2():
    rm = RewardModel.from_policy(GPT(tiny_config()))
    loss, _, margin = preference_loss(rm, [TOK.encode("ab")], [TOK.encode("cd")], [0.7], PAD)
    assert np.all(margin == 0) and loss.item() == pytest.approx(math.log(2), rel=1e-6)


def test_pm_score_reads_the_last_real_token_not_padding():
    rm = RewardModel(tiny_config())
    rm.head.weight.data = np.ones_like(rm.head.weight.data)
    short, long = TOK.encode("ab"), TOK.encode("abcdef")
    idx, last = pad_sequences([short, long], PAD)
    batched = rm.score(idx, last).data
    alone = rm.score(np.array([short]), [len(short) - 1]).data
    np.testing.assert_allclose(batched[0], alone[0], rtol=1e-5)


def test_pm_learns_a_preference_and_generalises_to_unseen_pairs():
    """Prefer sequences ending in 'y' over 'n'. Train on some prefixes, test on others."""
    rng = np.random.default_rng(0)
    rm = RewardModel.from_policy(GPT(tiny_config()))
    opt = make_trunk_head_optimizer(rm, dict(matrix_lr=0.01, embedding_lr=0.05,
                                             unembedding_lr=0.01, scalar_lr=0.01), head_lr=0.05)
    prefixes = ["ab", "cd", "ef", "gh", "ij", "kl", "mn", "op"]
    for _ in range(60):
        ps = [prefixes[i] for i in rng.integers(0, 6, 8)]
        flip = rng.random(8) < 0.5
        a = [TOK.encode(p + ("n" if f else "y")) for p, f in zip(ps, flip)]
        b = [TOK.encode(p + ("y" if f else "n")) for p, f in zip(ps, flip)]
        loss, _, _ = preference_loss(rm, a, b, np.where(flip, 0.1, 0.9), PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, acc, _ = preference_loss(rm, [TOK.encode(p + "y") for p in prefixes[6:]],
                                [TOK.encode(p + "n") for p in prefixes[6:]], [0.9, 0.9], PAD)
    assert acc == 1.0


def test_reward_model_save_load_round_trip(tmp_path):
    rm = RewardModel(tiny_config())
    rm.head.weight.data = np.full_like(rm.head.weight.data, 0.3)
    save_reward_model(str(tmp_path), rm, meta={"feedback": "rule"})
    loaded, meta = load_reward_model(str(tmp_path))
    assert meta["feedback"] == "rule"
    for (n, a), (_, b) in zip(rm.named_parameters(), loaded.named_parameters()):
        np.testing.assert_array_equal(a.data, b.data, err_msg=n)


# ----------------------------------------------------------------------------
# rollouts and PPO

def test_ppo_batch_marks_exactly_the_response_tokens():
    policy = GPT(tiny_config())
    ro = rollout(Engine(policy, TOK), TOK, [[{"role": "user", "content": "1+1"}],
                                            [{"role": "user", "content": "12+3"}]],
                 max_tokens=4, num_samples=2)
    full, last, x, y = build_ppo_batch(ro, PAD)
    for b, r in enumerate(ro):
        assert list(y[b][y[b] != -1]) == r["response_ids"]
        assert last[b] == len(r["prompt_ids"]) + len(r["response_ids"]) - 1
    np.testing.assert_array_equal(x, full[:, :-1])


def test_kl_rewards_spend_kl_per_token_and_pay_the_pm_at_the_end():
    mask = np.array([[0, 1, 1, 1], [0, 0, 1, 0]], bool)
    lp = np.array([[0, -1.0, -2.0, -0.5], [0, 0, -3.0, 0]])
    lp_ref = np.array([[0, -1.5, -1.0, -0.5], [0, 0, -1.0, 0]])
    rewards, kl = kl_penalized_rewards(np.array([2.0, -1.0]), lp, lp_ref, mask, kl_coef=0.1)
    np.testing.assert_allclose(rewards, [[0, -0.05, 0.1, 2.0], [0, 0, 0.2 - 1.0, 0]])
    np.testing.assert_allclose(kl, [-0.5, -2.0])


def test_gae_matches_hand_computation():
    r = np.array([[0.0, 0.0, 1.0, 0.0]])
    v = np.array([[0.2, 0.5, 0.7, 9.9]])   # last position is masked out
    mask = np.array([[1, 1, 1, 0]], bool)
    adv, ret = gae(r, v, mask, gamma=1.0, lam=1.0)
    np.testing.assert_allclose(adv[0, :3], [0.8, 0.5, 0.3])   # Monte Carlo return - V
    assert adv[0, 3] == 0 and ret[0, 3] == 0
    adv, _ = gae(r, v, mask, gamma=1.0, lam=0.0)               # one-step TD
    np.testing.assert_allclose(adv[0, :3], [0.5 - 0.2, 0.7 - 0.5, 1.0 - 0.7])


def test_whiten_uses_only_masked_positions():
    x = np.array([[1.0, 2.0, 3.0, 100.0]])
    w = whiten(x, np.array([[1, 1, 1, 0]], bool))
    assert w[0, 3] == 0
    np.testing.assert_allclose(w[0, :3].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(w[0, :3].std(), 1, atol=1e-6)


def test_ppo_clipping_removes_the_gradient_outside_the_trust_region(float64_engine):
    """Positive advantage: tokens whose ratio already exceeds 1+eps get no gradient;
    tokens inside the region get d/dlogp = -A/n * ratio."""
    lp_new = Tensor(np.array([[0.0, 0.5, 0.05]]), requires_grad=True)
    lp_old = np.zeros((1, 3))
    adv = np.ones((1, 3))
    mask = np.ones((1, 3), bool)
    loss, stats = ppo_policy_loss(lp_new, lp_old, adv, mask, clip=0.2)
    loss.backward()
    ratio = np.exp(lp_new.data[0])
    np.testing.assert_allclose(lp_new.grad[0], [-ratio[0] / 3, 0.0, -ratio[2] / 3])
    assert stats["clipfrac"] == pytest.approx(1 / 3)


def test_ppo_loss_is_minus_mean_advantage_at_ratio_one():
    policy = GPT(tiny_config())
    ro = rollout(Engine(policy, TOK), TOK, [[{"role": "user", "content": "1+1"}]], max_tokens=4)
    _, _, x, y = build_ppo_batch(ro, PAD)
    lp = sequence_logprobs(policy, x, y)
    mask = y != -1
    adv = np.where(mask, np.linspace(-1, 2, mask.size).reshape(mask.shape), 0)
    loss, stats = ppo_policy_loss(lp, lp.data.astype(np.float64), adv, mask)
    assert loss.item() == pytest.approx(-adv[mask].mean(), rel=1e-5)
    assert stats["approx_kl"] == pytest.approx(0, abs=1e-6)


def test_value_loss_is_half_mse_over_actions():
    v = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
    loss = value_loss(v, np.array([[0.0, 2.0, 0.0]]), np.array([[1, 1, 0]], bool))
    assert loss.item() == pytest.approx(0.5 * (1.0 + 0.0) / 2)


def test_ppo_steps_increase_the_probability_of_a_rewarded_reply():
    """End to end on one prompt: reward one specific reply, and PPO must make it
    more likely without touching the frozen reference copy."""
    rng = np.random.default_rng(0)
    policy = GPT(tiny_config())
    reference = clone_model(policy)
    from nanochat.scratch import setup_optimizer
    opt = setup_optimizer(policy, matrix_lr=0.01, embedding_lr=0.05, unembedding_lr=0.01, scalar_lr=0.01)
    prompt = [{"role": "user", "content": "x"}]
    target = TOK.encode("7")[0]

    def p_target():
        ro = {"prompt_ids": __import__("nanochat.chat_format", fromlist=["render_prompt"])
              .render_prompt(TOK, prompt), "response_ids": [target]}
        _, _, x, y = build_ppo_batch([ro], PAD)
        return float(np.exp(sequence_logprobs(policy, x, y).data[y != -1][0]))

    before = p_target()
    ref_before = {n: p.data.copy() for n, p in reference.named_parameters()}
    for it in range(15):
        ro = rollout(Engine(policy, TOK), TOK, [prompt] * 16, max_tokens=1, seed=it)
        _, _, x, y = build_ppo_batch(ro, PAD)
        mask = y != -1
        reward = np.array([1.0 if r["response_ids"][0] == target else 0.0 for r in ro])
        lp_old = sequence_logprobs(policy, x, y).data.astype(np.float64)
        rewards, _ = kl_penalized_rewards(reward, lp_old, lp_old, mask, 0.0)
        adv, _ = gae(rewards, np.zeros_like(rewards), mask)
        for _ in range(2):
            loss, _ = ppo_policy_loss(sequence_logprobs(policy, x, y), lp_old,
                                      whiten(adv, mask) + 1e-3 * rng.standard_normal(adv.shape) * mask,
                                      mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert p_target() > before
    for n, p in reference.named_parameters():
        np.testing.assert_array_equal(p.data, ref_before[n], err_msg=n)


# ----------------------------------------------------------------------------
# constitution and feedback

MSG = [{"role": "user", "content": "3+4"}]


def test_rule_feedback_soft_labels_per_principle():
    fb = RuleFeedback(confidence=0.9)
    right, wrong, chatty, cut = (Response("7"), Response("8"), Response("it is 7"),
                                 Response("7", "max_tokens"))
    assert fb.compare(MSG, right, wrong, PRINCIPLES["correct"]) == pytest.approx(0.9)
    assert fb.compare(MSG, wrong, right, PRINCIPLES["correct"]) == pytest.approx(0.1)
    assert fb.compare(MSG, right, chatty, PRINCIPLES["concise"]) == pytest.approx(0.9)
    assert fb.compare(MSG, right, cut, PRINCIPLES["complete"]) == pytest.approx(0.9)
    # a principle that does not separate the pair gives no preference
    assert fb.compare(MSG, right, wrong, PRINCIPLES["concise"]) == 0.5
    # and the CoT clamp bounds the label
    assert fb.compare(MSG, right, wrong, PRINCIPLES["correct"], clamp=COT_CLAMP) == 0.6


def test_rule_revision_fixes_what_the_principle_names():
    fb = RuleFeedback()
    critique, revision = fb.revise(MSG, Response("8"), PRINCIPLES["correct"])
    assert revision == "7" and "incorrect" in critique
    assert fb.revise(MSG, Response("so 7!"), PRINCIPLES["concise"])[1] == "7"
    assert fb.revise(MSG, Response("7"), PRINCIPLES["correct"])[1] == "7"


def test_principles_are_sampled_uniformly():
    rng = np.random.default_rng(0)
    counts = {p.name: 0 for p in ARITHMETIC_CONSTITUTION}
    for _ in range(3000):
        counts[sample_principle(ARITHMETIC_CONSTITUTION, rng).name] += 1
    assert all(abs(c / 3000 - 1 / 3) < 0.03 for c in counts.values())


def test_clamp_and_ranking_helpers():
    assert clamp_label(0.99, COT_CLAMP) == 0.6 and clamp_label(0.01, COT_CLAMP) == 0.4
    assert clamp_label(0.55, COT_CLAMP) == 0.55 and clamp_label(0.99) == 0.99
    pairs = ranking_to_comparisons(["a", "b", "c"], [2, 0, 1])   # c best, then a, then b
    assert pairs == [("c", "a", 1.0), ("c", "b", 1.0), ("a", "b", 1.0)]
    with pytest.raises(ValueError):
        ranking_to_comparisons(["a", "b"], [0, 0])


class OptionModel:
    """Stub LM: after the prompt, prefers the byte 'A' (or 'B') by a fixed margin."""

    def __init__(self, favour, margin=2.0):
        self.config = tiny_config()
        self.favour, self.margin = ord(favour), margin
        self.prompts = []

        class _W:
            weight = type("_P", (), {"data": np.zeros(1, dtype=np.float32)})()
        self.wte = _W()

    def __call__(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        idx = np.asarray(idx)
        self.prompts.append(idx[0].tolist())
        logits = np.zeros((*idx.shape, TOK.vocab_size), dtype=np.float32)
        logits[..., self.favour] = self.margin
        if kv_cache is not None:
            kv_cache.advance(idx.shape[1])
        return Tensor(logits)


def test_lm_feedback_uses_the_paper_template_and_option_probabilities():
    model = OptionModel("A", margin=2.0)
    model.config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=32, sequence_len=512,
                             vocab_size=TOK.vocab_size)
    fb = LMFeedback(model, TOK)
    a, b = Response("7"), Response("8")
    principle = PRINCIPLES["correct"]
    p = fb.compare(MSG, a, b, principle)
    # " (A)" and " (B)" differ in one byte; the stub gives 'A' a logit margin of 2
    assert p == pytest.approx(1 / (1 + math.exp(-2.0)), rel=1e-5)
    prompt_text = TOK.decode(model.prompts[0][1:])
    assert prompt_text.startswith(MC_TEMPLATE.format(
        conversation=format_conversation(MSG), principle=principle.comparison, a="7", b="8"))
    assert fb.compare(MSG, a, b, principle, clamp=COT_CLAMP) == 0.6
    assert LMFeedback(OptionModel("B"), TOK).compare(MSG, a, b, principle) < 0.5


def test_lm_feedback_revision_runs_critique_then_revision_turns():
    model = OptionModel("z")
    model.config = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=32, sequence_len=512,
                             vocab_size=TOK.vocab_size)
    critique, revision = LMFeedback(model, TOK, max_new_tokens=3).revise(
        MSG, Response("8"), PRINCIPLES["correct"])
    assert critique == "zzz" and revision == "zzz"
    last_prompt = TOK.decode([t for t in model.prompts[-3] if t < 256])
    assert PRINCIPLES["correct"].critique_request in last_prompt
    assert PRINCIPLES["correct"].revision_request in last_prompt
