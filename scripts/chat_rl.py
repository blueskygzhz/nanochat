"""
RL from AI feedback: PPO against the preference model, with a KL penalty.
`python -m scripts.chat_rl --source cai --pm pm --run rl`

The RL stage of HH-RLHF / Constitutional AI (arXiv:2204.05862, arXiv:2212.08073):

    r_total = r_PM - lambda_KL * KL(pi || pi_0)

with pi_0 the policy RL starts from (the SL-CAI model) and PPO as the optimiser. Each
iteration samples replies to prompts from the seen operand pairs, scores them with the
"train" PM, spends the KL penalty per token, estimates advantages with GAE against a
value function initialised from the PM, and takes clipped PPO steps.

Logged every `--log-every` iterations, the quantities the papers use to read an RL run:

    PM-train   the reward being optimised
    PM-test    the same, from the PM trained on the other half of the comparisons.
               Rising together = real improvement; train up while test flattens or
               falls = the policy is exploiting PM-train (over-optimisation)
    KL, sqrtKL distance from pi_0. The HH paper finds PM reward ~ linear in sqrt(KL)
    gold       the fraction of replies the constitution's checks actually accept
               (available only because the toy domain is checkable)

**On --kl-coef.** The papers use 0.001, with 52B-parameter preference models trained
on ~10^5 comparisons. Here the PM is a 230K-parameter model trained on ~10^3, and at
0.001 the policy finds its blind spots within 60 iterations: on the default run PM
reward rises (1.78 -> 2.07) while the replies the constitution actually accepts fall
from 94% to 6% -- the policy learns to answer "1" to everything. Measured sweep (same
PM, 60 iterations; gold = replies passing every principle):

    kl_coef   PM-train      gold          greedy seen / held-out
    0.001     1.78 -> 2.07  0.94 -> 0.06  80 -> 1  /  7 -> 1
    0.05      1.78 -> 1.82  0.94 -> 0.91  80 -> 70 /  7 -> 4
    0.2       1.78 -> 1.79  0.94 -> 1.00  80 -> 78 /  7 -> 6

So the default is 0.2; pass --kl-coef 0.001 to reproduce the papers' setting and watch
over-optimisation happen. Note that PM-test rose along with PM-train in the 0.001 run:
two PMs trained from the same initialisation on the same kind of data share blind spots,
so at this scale only the gold check exposes the hacking.

Everything else -- learning rates, clip, epochs, GAE lambda, the value function -- the
papers do not specify; these are common PPO defaults.
"""

import argparse
import os
import time

import numpy as np

from nanochat.constitution import Response, RuleFeedback, ARITHMETIC_CONSTITUTION
from nanochat.scratch import Engine, save_checkpoint, setup_optimizer
from nanochat.scratch.rlhf import (
    RewardModel, build_ppo_batch, clone_model, gae, kl_penalized_rewards, load_reward_model,
    make_trunk_head_optimizer, ppo_policy_loss, rollout, sequence_logprobs,
    value_loss, whiten,
)
from nanochat.scratch.tensor import no_grad
from scripts.posttrain_common import accuracy_report, load_run, pair_split, prompts_for, run_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="cai", help="initial policy pi_0")
    p.add_argument("--pm", default="pm", help="preference-model run (with train/ and test/)")
    p.add_argument("--run", default="rl")
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32, help="rollouts per iteration")
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kl-coef", type=float, default=0.2,
                   help="lambda_KL. Papers: 0.001; see the docstring for why the default differs")
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--matrix-lr", type=float, default=0.001)
    p.add_argument("--embedding-lr", type=float, default=0.003)
    p.add_argument("--unembedding-lr", type=float, default=0.001)
    p.add_argument("--scalar-lr", type=float, default=0.002)
    p.add_argument("--value-lr", type=float, default=0.003)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def gold_rate(rollouts):
    """Fraction of replies passing every constitution check (correct, concise, complete)."""
    judge = RuleFeedback()
    ok = 0
    for r in rollouts:
        response = Response(r["content"] if isinstance(r["content"], str) else "", r["stop_reason"])
        verdicts = [judge.compare(r["messages"], response, Response("", "max_tokens"), p)
                    for p in ARITHMETIC_CONSTITUTION]
        ok += all(v > 0.5 for v in verdicts)
    return ok / max(len(rollouts), 1)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    policy, tokenizer, meta = load_run(args.source)
    seen, held = pair_split(meta)
    pad_id = tokenizer.get_bos_token_id()
    pm_train, pm_meta = load_reward_model(os.path.join(run_dir(args.pm), "train"))
    pm_test, _ = load_reward_model(os.path.join(run_dir(args.pm), "test"))
    if pm_train.config.vocab_size != policy.config.vocab_size:
        raise SystemExit("preference model and policy use different vocabularies")
    pm_train.eval()
    pm_test.eval()

    reference = clone_model(policy)   # pi_0, frozen
    reference.eval()
    critic = RewardModel(pm_train.config)
    critic.load_state_dict(pm_train.state_dict())
    critic.train()
    policy.train()
    engine = Engine(policy, tokenizer)

    pol_opt = setup_optimizer(policy, matrix_lr=args.matrix_lr, embedding_lr=args.embedding_lr,
                              unembedding_lr=args.unembedding_lr, scalar_lr=args.scalar_lr)
    val_opt = make_trunk_head_optimizer(
        critic, dict(matrix_lr=args.value_lr, embedding_lr=args.value_lr,
                     unembedding_lr=args.value_lr, scalar_lr=args.value_lr),
        head_lr=args.value_lr)

    print(f"chat_rl (PPO) | pi_0 = {args.source} step {meta['step']} | PM = {args.pm} "
          f"(feedback: {pm_meta.get('feedback')}, held-out acc {pm_meta.get('heldout_acc', float('nan')):.3f})")
    print(f"r = r_PM - {args.kl_coef} * KL | {args.batch_size} rollouts x {args.iterations} "
          f"iterations | clip {args.clip}, {args.ppo_epochs} epochs")
    policy.eval()
    before = accuracy_report(engine, tokenizer, seen, held)
    print("-" * 84)
    print(f"{'iter':>5s}{'PM-train':>10s}{'PM-test':>10s}{'KL':>8s}{'sqrtKL':>8s}"
          f"{'gold':>7s}{'clip':>7s}{'vloss':>8s}{'time':>8s}")

    t0 = time.time()
    history = []
    for it in range(args.iterations):
        # --- rollout (no grad) -------------------------------------------------
        policy.eval()
        idx = rng.integers(0, len(seen), args.batch_size)
        ro = rollout(engine, tokenizer, prompts_for([seen[i] for i in idx]),
                     max_tokens=args.max_tokens, temperature=args.temperature,
                     seed=args.seed * 100003 + it * 1009)
        full, last, x, y = build_ppo_batch(ro, pad_id)
        mask = y != -1
        with no_grad():
            logp_old = sequence_logprobs(policy, x, y).data.astype(np.float64)
            logp_ref = sequence_logprobs(reference, x, y).data.astype(np.float64)
            values = critic.values(x).data.astype(np.float64)
            r_train = pm_train.score(full, last).data.astype(np.float64)
            r_test = pm_test.score(full, last).data.astype(np.float64)
        rewards, kl = kl_penalized_rewards(r_train, logp_old, logp_ref, mask, args.kl_coef)
        adv, returns = gae(rewards, values, mask, gamma=1.0, lam=args.gae_lambda)
        adv = whiten(adv, mask)

        # --- PPO updates ---------------------------------------------------------
        policy.train()
        stats, vlosses = [], []
        for _ in range(args.ppo_epochs):
            loss, st = ppo_policy_loss(sequence_logprobs(policy, x, y), logp_old, adv, mask, args.clip)
            pol_opt.zero_grad()
            loss.backward()
            pol_opt.step()
            stats.append(st)
            vl = value_loss(critic.values(x), returns, mask)
            val_opt.zero_grad()
            vl.backward()
            val_opt.step()
            vlosses.append(vl.item())

        mean_kl = float(np.mean(kl))
        row = {"iter": it, "pm_train": float(r_train.mean()), "pm_test": float(r_test.mean()),
               "kl": mean_kl, "gold": gold_rate(ro)}
        history.append(row)
        if it % args.log_every == 0 or it == args.iterations - 1:
            print(f"{it:5d}{row['pm_train']:10.3f}{row['pm_test']:10.3f}{mean_kl:8.3f}"
                  f"{np.sqrt(max(mean_kl, 0.0)):8.3f}{row['gold']:7.2f}"
                  f"{np.mean([s['clipfrac'] for s in stats]):7.2f}{np.mean(vlosses):8.4f}"
                  f"{time.time() - t0:7.1f}s")

    policy.eval()
    after = accuracy_report(engine, tokenizer, seen, held)
    save_checkpoint(run_dir(args.run), args.iterations - 1, policy, pol_opt,
                    meta={"source": args.source, "stage": "rl", "pm": args.pm,
                          "kl_coef": args.kl_coef, "data": meta.get("data"),
                          "tokenizer": meta.get("tokenizer"), "history": history})
    print("-" * 84)
    first, final = history[0], history[-1]
    print(f"PM-train {first['pm_train']:+.3f} -> {final['pm_train']:+.3f} | "
          f"PM-test {first['pm_test']:+.3f} -> {final['pm_test']:+.3f} | "
          f"gold {first['gold']:.2f} -> {final['gold']:.2f}")
    print(f"greedy exact match   {'before':>8s}{'after':>8s}")
    print(f"  seen pairs         {f'{before[0]}/{len(seen)}':>8s}{f'{after[0]}/{len(seen)}':>8s}")
    if held:
        print(f"  held-out pairs     {f'{before[1]}/{len(held)}':>8s}{f'{after[1]}/{len(held)}':>8s}")
    print(f"next: python -m scripts.chat_eval --run {args.run}")


if __name__ == "__main__":
    main()
