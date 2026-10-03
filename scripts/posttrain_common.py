"""
Shared plumbing for the post-training stages (chat_cai, chat_pm, chat_rl).
"""

import os

from nanochat.common import get_base_dir
from nanochat.constitution import ARITHMETIC_CONSTITUTION, LMFeedback, RuleFeedback
from nanochat.scratch import addition_pairs, list_steps, load_model
from nanochat.tokenizer import load_tokenizer


def run_dir(name):
    return os.path.join(get_base_dir(), "checkpoints", name)


def load_run(name, step=None):
    """Load a policy checkpoint and its tokenizer. Returns `(model, tokenizer, meta)`."""
    path = run_dir(name)
    if not list_steps(path):
        raise SystemExit(f"no checkpoints in {path}")
    model, meta = load_model(path, step)
    return model, load_tokenizer(meta.get("tokenizer")), meta


def pair_split(meta):
    """The (seen, held-out) operand pairs recorded with the checkpoint."""
    spec = meta.get("data") or {}
    if spec.get("kind") == "addition":
        return addition_pairs(spec["holdout_frac"], spec["seed"])
    print("warning: checkpoint has no addition split; nothing is held out")
    return addition_pairs(0.0)


def prompts_for(pairs):
    return [[{"role": "user", "content": f"{a}+{b}"}] for a, b in pairs]


def make_feedback(spec):
    """`rule` -> the programmatic stand-in; `lm:<run>` -> a checkpoint as the judge."""
    if spec == "rule":
        return RuleFeedback()
    if spec.startswith("lm:"):
        model, tokenizer, _ = load_run(spec[3:])
        return LMFeedback(model, tokenizer)
    raise SystemExit(f"--feedback must be 'rule' or 'lm:<run>', got {spec!r}")


def describe_feedback(feedback):
    if feedback.is_stand_in:
        return ("rule-based stand-in (a 230K-parameter model cannot judge text; each "
                "principle is checked programmatically)")
    return "language model (Constitutional AI multiple-choice prompt, soft labels)"


def parse_constitution(names):
    by_name = {p.name: p for p in ARITHMETIC_CONSTITUTION}
    if not names:
        return ARITHMETIC_CONSTITUTION
    picked = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in picked if n not in by_name]
    if unknown:
        raise SystemExit(f"unknown principles {unknown}; choose from {sorted(by_name)}")
    return tuple(by_name[n] for n in picked)


def accuracy_report(engine, tokenizer, seen, held):
    """Greedy exact match on seen and held-out pairs, as (seen_correct, held_correct)."""
    from scripts.chat_eval import chat_exact_match
    return (chat_exact_match(engine, tokenizer, seen),
            chat_exact_match(engine, tokenizer, held) if held else 0)
