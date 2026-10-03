"""
RLHF / RLAIF machinery, from scratch: a preference model and PPO with a KL penalty.

This follows the recipe Anthropic published in "Training a Helpful and Harmless
Assistant with RLHF" (Bai et al. 2022, arXiv:2204.05862) and reused for the RL stage of
"Constitutional AI" (Bai et al. 2022, arXiv:2212.08073):

  **Preference model (PM).** A transformer from the same family as the policy, with a
  scalar head read out at the *last* token. Trained on pairwise comparisons with the
  Bradley-Terry loss  log(1 + exp(r_bad - r_good)).  Constitutional AI trains on *soft*
  labels -- the feedback model's normalised probability that A is better -- so the
  loss here is the expectation of that BT loss under the label:
        L = -[ p log sigmoid(r_A - r_B) + (1 - p) log sigmoid(r_B - r_A) ]
  which reduces to the BT loss when p is 0 or 1.

  **RL.** PPO against the PM score, with a KL penalty to the initial policy:
        r_total = r_PM - lambda_KL * KL(pi || pi_0),     lambda_KL = 0.001 in the papers
  The KL is spent per token (r_t = -lambda * (log pi(a_t) - log pi_0(a_t))) and the PM
  score is added at the final token -- the usual way of turning that sequence-level
  objective into per-token rewards for PPO.

  **Diagnostics.** The HH paper reports PM reward growing roughly linearly in sqrt(KL)
  for most of training, and measures over-optimisation by training two PMs on disjoint
  halves of the comparisons, optimising against one and scoring with the other. Both
  are reported by `scripts/chat_rl.py`.

What is *not* public, and therefore not reproduced here: Anthropic's PPO
hyperparameters beyond lambda_KL, its value-function setup, and anything about how any
specific Claude model was post-trained. Choices made where the papers are silent are
marked as such below.
"""

import json
import os

import numpy as np

from nanochat.scratch import nn
from nanochat.scratch.checkpoint import _atomic_write
from nanochat.scratch.model import GPT, GPTConfig
from nanochat.scratch.optim import AdamW, MuonAdamW, setup_optimizer
from nanochat.scratch.tensor import Tensor, cross_entropy, no_grad, stack, where

__all__ = [
    "RewardModel", "clone_model", "preference_loss", "pad_sequences", "rollout",
    "build_ppo_batch", "sequence_logprobs", "kl_penalized_rewards", "gae",
    "whiten", "ppo_policy_loss", "value_loss", "save_reward_model", "load_reward_model",
    "make_trunk_head_optimizer", "score_rollouts",
]


def clone_model(model):
    """An independent copy of a GPT (same config, same weights, no shared buffers)."""
    copy = GPT(model.config)
    copy.load_state_dict(model.state_dict())
    return copy


# ----------------------------------------------------------------------------
# preference model

class RewardModel(nn.Module):
    """A GPT trunk with a scalar head: one number per position.

    Read at the last token, it is the preference model's score for a whole sequence.
    Read at every position, the same architecture is the PPO value function -- which is
    why the critic is initialised from the PM (as InstructGPT did; the Anthropic papers
    do not say how their value function was set up).
    """

    def __init__(self, config, seed=0):
        super().__init__()
        self.trunk = GPT(config, seed=seed)
        self.head = nn.Linear(config.n_embd, 1)
        # zero init: every sequence scores 0 before training, so the first updates are
        # driven by the labels rather than by a random head
        self.head.weight.data = np.zeros_like(self.head.weight.data)

    @property
    def config(self):
        return self.trunk.config

    @classmethod
    def from_policy(cls, policy):
        """Initialise the trunk from a (finetuned) language model, as the papers do."""
        rm = cls(policy.config)
        rm.trunk.load_state_dict(policy.state_dict())
        return rm

    def values(self, idx):
        """(B, T) -> (B, T) scalar per position."""
        idx = np.asarray(idx, dtype=np.int64)
        B, T = idx.shape
        return self.head(self.trunk.forward_hidden(idx)).reshape(B, T)

    def score(self, idx, last_index):
        """Score each row at its last real token. `last_index` is (B,) ints."""
        idx = np.asarray(idx, dtype=np.int64)
        return self.values(idx)[np.arange(idx.shape[0]), np.asarray(last_index)]


def preference_loss(rm, seqs_a, seqs_b, p_a, pad_id):
    """Soft-label Bradley-Terry loss. Returns `(loss_tensor, accuracy, margins)`.

    `p_a[i]` is the label's probability that sequence A is preferred. A and B are
    scored in one batch. Accuracy counts agreement with the label's direction, ignoring
    exact ties (p = 0.5).
    """
    n = len(seqs_a)
    p_a = np.asarray(p_a, dtype=np.float64)
    idx, last = pad_sequences(list(seqs_a) + list(seqs_b), pad_id)
    scores = rm.score(idx, last)
    s_a, s_b = scores[:n], scores[n:]
    # A 2-way softmax over (s_a, s_b) gives sigmoid(s_a - s_b), so the soft BT loss is a
    # label-weighted 2-class cross-entropy -- computed stably by the fused op
    logits = stack([s_a, s_b], axis=1)
    nll_a = cross_entropy(logits, np.zeros(n, dtype=np.int64), reduction="none")
    nll_b = cross_entropy(logits, np.ones(n, dtype=np.int64), reduction="none")
    loss = (nll_a * Tensor(p_a) + nll_b * Tensor(1.0 - p_a)).mean()
    margin = s_a.data - s_b.data
    decided = p_a != 0.5
    acc = float(np.mean((margin[decided] > 0) == (p_a[decided] > 0.5))) if decided.any() else float("nan")
    return loss, acc, margin


def make_trunk_head_optimizer(rm, trunk_lrs, head_lr):
    """nanochat's Muon/AdamW split for the trunk, plus AdamW for the scalar head.

    (`setup_optimizer` routes parameters by name, so it is applied to the trunk on its
    own; prefixed names would send the embeddings to Muon.)
    """
    trunk = setup_optimizer(rm.trunk, **trunk_lrs)
    head = AdamW([rm.head.weight], lr=head_lr)
    return MuonAdamW(trunk.muon, _MergedAdamW(trunk.adamw, head))


class _MergedAdamW:
    """Two AdamW instances behind one interface (param groups, step, state)."""

    def __init__(self, *opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    @property
    def state(self):
        merged = {}
        for o in self.opts:
            merged.update(o.state)
        return merged

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad()


def save_reward_model(out_dir, rm, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    _atomic_write(os.path.join(out_dir, "model.npz"), lambda f: np.savez(f, **rm.state_dict()))
    payload = {"config": vars(rm.config), **(meta or {})}
    _atomic_write(os.path.join(out_dir, "meta.json"),
                  lambda f: f.write(json.dumps(payload, indent=2).encode()))


def load_reward_model(out_dir):
    with open(os.path.join(out_dir, "meta.json"), "rb") as f:
        meta = json.loads(f.read())
    rm = RewardModel(GPTConfig(**meta["config"]))
    with np.load(os.path.join(out_dir, "model.npz")) as z:
        rm.load_state_dict({k: z[k] for k in z.files})
    return rm, meta


# ----------------------------------------------------------------------------
# rollouts

def pad_sequences(seqs, pad_id):
    """Right-pad to one (B, T) array. Returns `(array, last_index)`.

    Right padding is safe for a causal model: no real position can see the padding.
    """
    lengths = [len(s) for s in seqs]
    if min(lengths) < 1:
        raise ValueError("empty sequence")
    # the model needs T >= 2 (the embedding smear reads the previous position);
    # that applies to the padded batch, not to each row
    out = np.full((len(seqs), max(max(lengths), 2)), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out, np.asarray(lengths) - 1


def rollout(engine, tokenizer, prompts, max_tokens, temperature=1.0, seed=0, num_samples=1):
    """Sample replies. `prompts` are message lists ending in a user turn.

    Returns one dict per sample: prompt ids, response ids (including the end-of-turn
    token if one was emitted), and the reply parsed back into `content`/`stop_reason`.
    """
    from nanochat.chat_format import parse_reply, render_prompt, reply_stop_tokens
    stop = sorted(set(reply_stop_tokens(tokenizer)))
    out = []
    for i, messages in enumerate(prompts):
        prompt_ids = render_prompt(tokenizer, messages)
        samples = engine.generate_batch(prompt_ids, max_tokens=max_tokens, num_samples=num_samples,
                                        temperature=temperature, seed=seed + i, stop_tokens=stop)
        for response_ids in samples:
            content, stop_reason = parse_reply(tokenizer, response_ids)
            out.append({"messages": messages, "prompt_ids": prompt_ids,
                        "response_ids": list(response_ids), "content": content,
                        "stop_reason": stop_reason})
    return out


def build_ppo_batch(rollouts, pad_id):
    """Pack rollouts for teacher-forced scoring.

    Returns `(full, last_index, x, y)`: `full` is prompt+response (what the PM reads);
    `x = full[:, :-1]` and `y = full[:, 1:]` with every non-response target set to -1,
    so y[b, j] != -1 exactly at the actions the policy took.
    """
    seqs = [r["prompt_ids"] + r["response_ids"] for r in rollouts]
    full, last = pad_sequences(seqs, pad_id)
    x = full[:, :-1]
    y = np.full(x.shape, -1, dtype=np.int64)
    for b, r in enumerate(rollouts):
        start = len(r["prompt_ids"]) - 1          # target index of the first response token
        n = len(r["response_ids"])
        y[b, start:start + n] = full[b, start + 1:start + 1 + n]
    return full, last, x, y


def sequence_logprobs(model, x, y):
    """log pi(y_j | x_<=j) at every position, as a (B, T) Tensor (0 where y == -1)."""
    logits = model(np.asarray(x, dtype=np.int64))
    B, T, V = logits.shape
    nll = cross_entropy(logits.reshape(B * T, V), np.asarray(y).reshape(-1),
                        ignore_index=-1, reduction="none")
    return -nll.reshape(B, T)


# ----------------------------------------------------------------------------
# PPO

def kl_penalized_rewards(pm_scores, logp, logp_ref, mask, kl_coef):
    """Per-token rewards for r_total = r_PM - kl_coef * KL(pi || pi_0).

    Each action pays kl_coef * (log pi - log pi_0); the PM score arrives at the last
    action. Returns `(rewards, kl_per_sequence)`; the sum over a sequence of
    log pi - log pi_0 is the standard single-sample estimate of its KL.
    """
    mask = np.asarray(mask, dtype=bool)
    kl = np.where(mask, logp - logp_ref, 0.0)
    rewards = -kl_coef * kl
    for b in range(mask.shape[0]):
        idx = np.flatnonzero(mask[b])
        if len(idx):
            rewards[b, idx[-1]] += pm_scores[b]
    return rewards, kl.sum(axis=1)


def gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Generalised advantage estimation over each row's action positions.

    Returns `(advantages, returns)`. gamma = 1: an episode is one reply, and the
    reward that matters (the PM score) arrives at its end.
    """
    mask = np.asarray(mask, dtype=bool)
    adv = np.zeros_like(rewards, dtype=np.float64)
    for b in range(rewards.shape[0]):
        running, next_value = 0.0, 0.0
        for j in np.flatnonzero(mask[b])[::-1]:
            delta = rewards[b, j] + gamma * next_value - values[b, j]
            running = delta + gamma * lam * running
            adv[b, j] = running
            next_value = values[b, j]
    return adv, adv + np.where(mask, values, 0.0)


def whiten(x, mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 2:
        return np.where(mask, x, 0.0)
    m, s = x[mask].mean(), x[mask].std()
    return np.where(mask, (x - m) / (s + 1e-8), 0.0)


def ppo_policy_loss(logp_new, logp_old, advantages, mask, clip=0.2):
    """The clipped surrogate  -E[min(r A, clip(r, 1-eps, 1+eps) A)],  r = pi/pi_old.

    Returns `(loss_tensor, stats)`. Where the clipped term is the smaller one it is a
    constant (the ratio has left the trust region), so those tokens get no gradient --
    that is the whole mechanism, and it is what `where` expresses below.
    """
    mask = np.asarray(mask, dtype=bool)
    ratio = (logp_new - Tensor(np.where(mask, logp_old, 0.0))).exp()
    adv = np.where(mask, advantages, 0.0)
    unclipped = ratio * Tensor(adv)
    clipped = np.clip(ratio.data, 1.0 - clip, 1.0 + clip) * adv
    surrogate = where(unclipped.data <= clipped, unclipped, Tensor(clipped))
    n = max(int(mask.sum()), 1)
    loss = -(surrogate * Tensor(mask.astype(np.float64))).sum() / n
    r = ratio.data[mask]
    stats = {"clipfrac": float(np.mean(np.abs(r - 1.0) > clip)) if r.size else 0.0,
             "approx_kl": float(np.mean((r - 1.0) - np.log(r))) if r.size else 0.0}
    return loss, stats


def value_loss(values, returns, mask):
    """0.5 * mean squared error over action positions."""
    mask = np.asarray(mask, dtype=np.float64)
    err = values - Tensor(np.where(mask > 0, returns, 0.0))
    return (err * err * Tensor(mask)).sum() * (0.5 / max(mask.sum(), 1.0))


@no_grad()
def score_rollouts(rm, rollouts, pad_id):
    full, last, _, _ = build_ppo_batch(rollouts, pad_id)
    return rm.score(full, last).data.astype(np.float64)
