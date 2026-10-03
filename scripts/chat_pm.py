"""
Train a preference model from AI feedback: `python -m scripts.chat_pm --source cai`

The RL-CAI data stage (Bai et al. 2022, arXiv:2212.08073), with the preference-model
training of the HH-RLHF paper (arXiv:2204.05862):

  1. sample two replies per prompt from the policy that RL will start from
  2. draw one principle per comparison, ask the feedback model the multiple-choice
     question, and keep its normalised probability as a *soft* label
     (`--clamp 0.4,0.6` applies the paper's clamp for chain-of-thought labels)
  3. initialise a preference model from the policy, add a scalar head, and train it
     on the comparisons with the soft Bradley-Terry loss

Two PMs are trained on disjoint halves of the comparisons -- the HH paper's
robustness setup. `chat_rl` optimises against the "train" PM and scores with the
"test" PM; when the two diverge, the policy is exploiting the PM rather than
improving (over-optimisation).
"""

import argparse
import os
import time

import numpy as np

from nanochat.constitution import Response, sample_principle
from nanochat.scratch import Engine
from nanochat.scratch.rlhf import (
    RewardModel, make_trunk_head_optimizer, preference_loss, rollout, save_reward_model,
)
from nanochat.scratch.tensor import no_grad
from scripts.posttrain_common import (
    describe_feedback, load_run, make_feedback, pair_split, parse_constitution, prompts_for,
    run_dir,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="cai", help="the policy RL will start from")
    p.add_argument("--run", default="pm")
    p.add_argument("--feedback", default="rule", help="'rule' (stand-in) or 'lm:<run>'")
    p.add_argument("--principles", default="")
    p.add_argument("--clamp", default="", help="e.g. 0.4,0.6 (the paper's CoT clamp)")
    p.add_argument("--num-prompts", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=1.2,
                   help="sampling temperature for the comparison pairs; diversity is "
                        "what makes comparisons informative")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--matrix-lr", type=float, default=0.003)
    p.add_argument("--embedding-lr", type=float, default=0.01)
    p.add_argument("--unembedding-lr", type=float, default=0.003)
    p.add_argument("--scalar-lr", type=float, default=0.005)
    p.add_argument("--head-lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def collect_comparisons(engine, tokenizer, feedback, constitution, prompts, temperature,
                        clamp, rng, seed):
    """Two samples per prompt, one AI label each. Identical pairs carry no preference
    and are dropped."""
    samples = rollout(engine, tokenizer, prompts, max_tokens=8, temperature=temperature,
                      seed=seed, num_samples=2)
    comparisons = []
    for s_a, s_b in zip(samples[0::2], samples[1::2]):
        if s_a["response_ids"] == s_b["response_ids"]:
            continue
        a = Response(s_a["content"] if isinstance(s_a["content"], str) else "", s_a["stop_reason"])
        b = Response(s_b["content"] if isinstance(s_b["content"], str) else "", s_b["stop_reason"])
        principle = sample_principle(constitution, rng)
        p_a = feedback.compare(s_a["messages"], a, b, principle, clamp=clamp)
        comparisons.append({
            "a": s_a["prompt_ids"] + s_a["response_ids"],
            "b": s_b["prompt_ids"] + s_b["response_ids"],
            "p_a": p_a, "principle": principle.name})
    return comparisons


def train_pm(policy, comparisons, args, pad_id, rng, label):
    rm = RewardModel.from_policy(policy)
    rm.train()
    opt = make_trunk_head_optimizer(
        rm, dict(matrix_lr=args.matrix_lr, embedding_lr=args.embedding_lr,
                 unembedding_lr=args.unembedding_lr, scalar_lr=args.scalar_lr),
        head_lr=args.head_lr)
    n = len(comparisons)
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        losses, accs = [], []
        for start in range(0, n, args.batch_size):
            batch = [comparisons[i] for i in order[start:start + args.batch_size]]
            loss, acc, _ = preference_loss(rm, [c["a"] for c in batch], [c["b"] for c in batch],
                                           [c["p_a"] for c in batch], pad_id)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if not np.isnan(acc):
                accs.append(acc)
        print(f"  PM[{label}] epoch {epoch} | loss {np.mean(losses):.4f} | "
              f"train acc {np.mean(accs) if accs else float('nan'):.3f}")
    rm.eval()
    return rm


@no_grad()
def evaluate_pm(rm, comparisons, pad_id, batch_size=64):
    accs, weights = [], []
    for start in range(0, len(comparisons), batch_size):
        batch = comparisons[start:start + batch_size]
        _, acc, _ = preference_loss(rm, [c["a"] for c in batch], [c["b"] for c in batch],
                                    [c["p_a"] for c in batch], pad_id)
        decided = sum(c["p_a"] != 0.5 for c in batch)
        if decided:
            accs.append(acc)
            weights.append(decided)
    return float(np.average(accs, weights=weights)) if accs else float("nan")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    clamp = tuple(float(v) for v in args.clamp.split(",")) if args.clamp else None
    policy, tokenizer, meta = load_run(args.source)
    seen, _ = pair_split(meta)
    feedback = make_feedback(args.feedback)
    constitution = parse_constitution(args.principles)
    pad_id = tokenizer.get_bos_token_id()

    print(f"chat_pm | policy {args.source} step {meta['step']}")
    print(f"feedback: {describe_feedback(feedback)}")
    print(f"constitution: {', '.join(p.name for p in constitution)} | clamp {clamp}")

    t0 = time.time()
    idx = rng.integers(0, len(seen), args.num_prompts)
    comparisons = collect_comparisons(Engine(policy, tokenizer), tokenizer, feedback, constitution,
                                      prompts_for([seen[i] for i in idx]), args.temperature,
                                      clamp, rng, args.seed)
    decided = sum(c["p_a"] != 0.5 for c in comparisons)
    print(f"{len(comparisons)} comparisons ({decided} with a preference) in {time.time() - t0:.1f}s")
    if decided < 20:
        raise SystemExit("too few informative comparisons; raise --num-prompts or --temperature")

    order = rng.permutation(len(comparisons))
    half = len(order) // 2
    splits = {"train": [comparisons[i] for i in order[:half]],
              "test": [comparisons[i] for i in order[half:]]}
    models = {}
    for name, data in splits.items():
        models[name] = train_pm(policy, data, args, pad_id, rng, name)

    print("-" * 72)
    print("PM accuracy on the *other* half's comparisons (held out from that PM):")
    for name, other in (("train", "test"), ("test", "train")):
        acc = evaluate_pm(models[name], splits[other], pad_id)
        print(f"  PM[{name}] on {other} half: {acc:.3f}")
        save_reward_model(os.path.join(run_dir(args.run), name), models[name],
                          meta={"source": args.source, "source_step": meta["step"],
                                "feedback": args.feedback, "clamp": clamp,
                                "constitution": [p.name for p in constitution],
                                "num_comparisons": len(splits[name]), "heldout_acc": acc,
                                "tokenizer": meta.get("tokenizer"), "data": meta.get("data")})
    print(f"saved -> {run_dir(args.run)}/{{train,test}}")
    print(f"next: python -m scripts.chat_rl --source {args.source} --pm {args.run}")


if __name__ == "__main__":
    main()
