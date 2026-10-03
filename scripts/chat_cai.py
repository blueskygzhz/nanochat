"""
SL-CAI: supervised learning from constitutional critique and revision.
`python -m scripts.chat_cai --source sft --run cai`

Stage one of Constitutional AI (Bai et al. 2022, arXiv:2212.08073):

  1. sample a reply from the current (helpful-only) model for each prompt
  2. draw a principle at random; ask for a critique of the reply against it, then a
     revision; repeat `--revisions` times, drawing a fresh principle each round
  3. finetune the *original* model on the final revisions, mixed with ordinary
     helpful conversations so that helpfulness is not traded away

The paper's point for the RL stage that follows: SL-CAI moves the policy onto the
distribution the preference model will reward, so RL has less exploring to do.

Prompts come only from the operand pairs seen in pretraining; held-out pairs stay
unseen through every stage, so they remain a clean test.
"""

import argparse
import math
import time

import numpy as np

from nanochat.constitution import Response, sample_principle
from nanochat.scratch import Engine, save_checkpoint, setup_optimizer
from scripts.chat_sft import make_conversations, pack_batch
from scripts.posttrain_common import (
    accuracy_report, describe_feedback, load_run, make_feedback, pair_split,
    parse_constitution, prompts_for, run_dir,
)
from nanochat.scratch.rlhf import rollout


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="sft", help="helpful-only model to start from")
    p.add_argument("--run", default="cai")
    p.add_argument("--feedback", default="rule", help="'rule' (stand-in) or 'lm:<run>'")
    p.add_argument("--principles", default="", help="comma list (default: whole constitution)")
    p.add_argument("--num-prompts", type=int, default=400)
    p.add_argument("--revisions", type=int, default=2, help="critique->revision rounds per reply")
    p.add_argument("--helpful-frac", type=float, default=0.5,
                   help="fraction of finetuning conversations that are plain helpful data")
    p.add_argument("--num-iterations", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--matrix-lr", type=float, default=0.003)
    p.add_argument("--embedding-lr", type=float, default=0.01)
    p.add_argument("--unembedding-lr", type=float, default=0.003)
    p.add_argument("--scalar-lr", type=float, default=0.005)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model, tokenizer, meta = load_run(args.source)
    seen, held = pair_split(meta)
    feedback = make_feedback(args.feedback)
    constitution = parse_constitution(args.principles)
    engine = Engine(model, tokenizer)

    print(f"chat_cai (SL-CAI) | from {args.source} step {meta['step']}")
    print(f"feedback: {describe_feedback(feedback)}")
    print(f"constitution: {', '.join(p.name for p in constitution)} | "
          f"{args.revisions} revision round(s) per reply")
    before = accuracy_report(engine, tokenizer, seen, held)

    # 1-2) sample, critique, revise
    t0 = time.time()
    idx = rng.integers(0, len(seen), args.num_prompts)
    prompts = prompts_for([seen[i] for i in idx])
    samples = rollout(engine, tokenizer, prompts, max_tokens=8, temperature=args.temperature,
                      seed=args.seed)
    revised, changed, example = [], 0, None
    for s in samples:
        text = s["content"] if isinstance(s["content"], str) else ""
        response = Response(text.strip(), s["stop_reason"])
        original = response
        for _ in range(args.revisions):
            principle = sample_principle(constitution, rng)
            critique, revision = feedback.revise(s["messages"], response, principle)
            response = Response(revision, "end_turn")
            if example is None and response.text != original.text:
                example = (s["messages"][0]["content"], original.text, principle.name,
                           critique, revision)
        changed += response.text != original.text or original.stop_reason != "end_turn"
        revised.append(s["messages"] + [{"role": "assistant", "content": response.text}])
    print(f"revised {len(revised)} replies in {time.time() - t0:.1f}s; "
          f"{changed} ({changed / len(revised):.0%}) were changed by revision")
    if example:
        q, orig, name, critique, revision = example
        print(f"  e.g. U:{q} | A:{orig!r} --[{name}]--> critique: {critique!r} -> A:{revision!r}")

    # 3) finetune the original model on the revisions, mixed with helpful data
    n_helpful = int(len(revised) * args.helpful_frac / max(1e-9, 1 - args.helpful_frac))
    data = revised + make_conversations(n_helpful, seed=args.seed + 1, pairs=seen)
    model.train()
    optimizer = setup_optimizer(model, matrix_lr=args.matrix_lr, embedding_lr=args.embedding_lr,
                                unembedding_lr=args.unembedding_lr, scalar_lr=args.scalar_lr)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    seq_len = model.config.sequence_len
    for step in range(args.num_iterations):
        scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(args.num_iterations - 1, 1)))
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * scale
        loss = model(*pack_batch(tokenizer, data, args.batch_size, seq_len, rng))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.num_iterations - 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")
    model.eval()

    after = accuracy_report(engine, tokenizer, seen, held)
    save_checkpoint(run_dir(args.run), args.num_iterations - 1, model, optimizer,
                    meta={"source": args.source, "stage": "sl-cai", "data": meta.get("data"),
                          "tokenizer": meta.get("tokenizer"), "feedback": args.feedback,
                          "constitution": [p.name for p in constitution]})
    print("-" * 72)
    print(f"{'':16s}{'before':>10s}{'after':>10s}")
    print(f"{'seen pairs':16s}{f'{before[0]}/{len(seen)}':>10s}{f'{after[0]}/{len(seen)}':>10s}")
    if held:
        print(f"{'held-out pairs':16s}{f'{before[1]}/{len(held)}':>10s}{f'{after[1]}/{len(held)}':>10s}")
    print(f"next: python -m scripts.chat_pm --source {args.run}")


if __name__ == "__main__":
    main()
