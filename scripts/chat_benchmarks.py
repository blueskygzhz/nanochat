"""
The standard chat benchmarks from `tasks/` -- ARC, MMLU, GSM8K, HumanEval -- run
against a from-scratch model. Used by `scripts.chat_eval --tasks ...`.

Two kinds of task, scored the way upstream's chat_eval scores them:

  - **categorical** (ARC, MMLU): render the question, run one forward pass, and take
    the argmax of the next-token logits *restricted to the answer letters*. No
    sampling, no parsing; a model that has learned nothing scores exactly chance.
  - **generative** (GSM8K, HumanEval): decode the reply greedily, stopping at the end
    of the assistant turn, and hand the text to the task's own `evaluate` (answer
    extraction for GSM8K, sandboxed test execution for HumanEval). If the tokenizer
    has the tool tokens, the calculator tool loop is on, as GSM8K's data expects.

Every score is reported next to its chance baseline and as a *centered* accuracy,
(acc - chance) / (1 - chance), so 0 means "no better than guessing" whatever the
number of choices. The mean centered score over tasks is upstream's ChatCORE.

**The context window is the binding constraint.** The default model has a 64-token
context; most MMLU and GSM8K prompts are longer. Prompts that do not fit are cropped
from the left (BOS is kept, the start of the question is lost), and how many were
cropped is reported, because a score on cropped prompts is a score on a different,
harder task.
"""

import numpy as np

from nanochat.chat_format import render_prompt, reply_stop_tokens
from nanochat.scratch import no_grad

TASK_NAMES = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval")


def load_task(name):
    """Build a task by name. Imported lazily: each one downloads its data on first use."""
    if name in ("ARC-Easy", "ARC-Challenge"):
        from tasks.arc import ARC
        return ARC(name, "test")
    if name == "MMLU":
        from tasks.mmlu import MMLU
        return MMLU("all", "test")
    if name == "GSM8K":
        from tasks.gsm8k import GSM8K
        return GSM8K("main", "test")
    if name == "HumanEval":
        from tasks.humaneval import HumanEval
        return HumanEval()
    raise ValueError(f"unknown task {name!r}; choose from {', '.join(TASK_NAMES)}")


def parse_task_list(spec):
    """'all' or a comma-separated list -> validated task names, in canonical order."""
    if not spec:
        return []
    names = list(TASK_NAMES) if spec.strip().lower() == "all" else \
        [s.strip() for s in spec.split(",") if s.strip()]
    for n in names:
        if n not in TASK_NAMES:
            raise ValueError(f"unknown task {n!r}; choose from {', '.join(TASK_NAMES)} or 'all'")
    return names


def crop_prompt(ids, budget):
    """Keep BOS plus the last `budget - 1` tokens. Returns `(ids, was_cropped)`.

    The question's tail and the assistant-start marker are what the next token most
    depends on, so those are what survive; BOS stays because the model never saw a
    sequence that did not start with one.
    """
    if budget < 2:
        raise ValueError(f"context budget {budget} is too small for any prompt")
    if len(ids) <= budget:
        return list(ids), False
    return [ids[0]] + list(ids[-(budget - 1):]), True


def has_tool_tokens(tokenizer):
    try:
        tokenizer.encode_special("<|python_start|>")
        return True
    except KeyError:
        return False


@no_grad()
def run_categorical(model, tokenizer, task, max_problems=None):
    """Argmax over the answer letters' logits. Returns a result dict."""
    n = len(task) if max_problems is None else min(len(task), max_problems)
    seq_len = model.config.sequence_len
    correct = cropped = 0
    chance = 0.0
    for i in range(n):
        conversation = task[i]
        letters = list(conversation["letters"])
        letter_ids = []
        for letter in letters:
            ids = tokenizer.encode(letter)
            if len(ids) != 1:
                raise ValueError(f"answer letter {letter!r} is {len(ids)} tokens; need exactly 1")
            letter_ids.append(ids[0])
        prompt, was_cropped = crop_prompt(render_prompt(tokenizer, conversation["messages"][:-1]),
                                          seq_len)
        cropped += was_cropped
        logits = model(np.asarray([prompt], dtype=np.int64)).data[0, -1]
        prediction = letters[int(np.argmax(logits[letter_ids]))]
        correct += bool(task.evaluate(conversation, prediction))
        chance += 1.0 / len(letters)
    return _result(correct, n, chance / max(n, 1), cropped)


def run_generative(engine, tokenizer, task, max_problems=None, max_new_tokens=256):
    """Greedy decoding, scored by the task's own `evaluate`. Returns a result dict."""
    n = len(task) if max_problems is None else min(len(task), max_problems)
    seq_len = engine.model.config.sequence_len
    # Leave room for an answer: at most half the context goes to generation
    new_tokens = max(1, min(max_new_tokens, seq_len // 2))
    stop = set(reply_stop_tokens(tokenizer))
    use_tools = has_tool_tokens(tokenizer)
    correct = cropped = 0
    for i in range(n):
        conversation = task[i]
        prompt, was_cropped = crop_prompt(render_prompt(tokenizer, conversation["messages"][:-1]),
                                          seq_len - new_tokens)
        cropped += was_cropped
        out = engine.generate_batch(prompt, max_tokens=new_tokens, temperature=0.0,
                                    stop_tokens=sorted(stop), use_tools=use_tools)[0]
        completion = tokenizer.decode([t for t in out if t not in stop])
        correct += bool(task.evaluate(conversation, completion))
    return _result(correct, n, 0.0, cropped)


def _result(correct, n, chance, cropped):
    acc = correct / n if n else 0.0
    centered = (acc - chance) / (1.0 - chance) if chance < 1.0 else 0.0
    return {"correct": correct, "n": n, "accuracy": acc, "chance": chance,
            "centered": centered, "cropped": cropped}


def run_task(name, model, engine, tokenizer, max_problems=None, task=None):
    """Load (unless given) and evaluate one task, dispatching on its eval type."""
    task = task if task is not None else load_task(name)
    if task.eval_type == "categorical":
        return run_categorical(model, tokenizer, task, max_problems)
    if task.eval_type == "generative":
        return run_generative(engine, tokenizer, task, max_problems)
    raise ValueError(f"task {name} has unknown eval_type {task.eval_type!r}")


def format_results(results):
    """A table of per-task results plus the ChatCORE mean, as a list of lines."""
    lines = [f"{'task':15s}{'n':>6s}{'acc':>8s}{'chance':>8s}{'centered':>10s}{'cropped':>10s}"]
    for name, r in results.items():
        if "error" in r:
            lines.append(f"{name:15s}  failed: {r['error']}")
            continue
        cropped = f"{r['cropped']}/{r['n']}"
        lines.append(f"{name:15s}{r['n']:6d}{r['accuracy']:8.3f}{r['chance']:8.3f}"
                     f"{r['centered']:+10.3f}{cropped:>10s}")
    scored = [r["centered"] for r in results.values() if "error" not in r]
    if len(scored) > 1:
        lines.append(f"{'ChatCORE':15s}{'':6s}{'':8s}{'':8s}{float(np.mean(scored)):+10.3f}")
    return lines
