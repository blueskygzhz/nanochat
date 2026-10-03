"""
Evaluation: bits-per-byte, and the CORE-style task metric.

From-scratch counterpart of `nanochat/loss_eval.py` + `nanochat/core_eval.py`.

Two metrics, for two different questions.

**bits per byte** answers "how well does this model compress text?" The plain mean
loss is in nats *per token*, so it changes if you change the tokenizer -- a bigger
vocabulary packs more bytes into each token and the loss goes up even for an equally
good model. Dividing total nats by total *bytes* removes the tokenizer from the
comparison:

    bpb = sum(nats) / (ln(2) * sum(bytes))

**CORE** answers "can the model pick the right answer?" For a multiple-choice item we
score each candidate continuation by its mean autoregressive loss and take the lowest.
This needs no generation and no answer parsing, so it works on a base model.

The jinja2 templates upstream uses for prompt rendering are replaced with plain string
formatting, to keep the dependency list at numpy.
"""

import math
import random

import numpy as np

from nanochat.scratch.tensor import no_grad

__all__ = [
    "evaluate_bpb", "token_bytes_table",
    "render_prompts_mc", "render_prompts_schema", "render_prompts_lm",
    "find_common_length", "evaluate_task", "evaluate_example",
]


# ----------------------------------------------------------------------------
# bits per byte

def token_bytes_table(tokenizer):
    """Byte length of every token id, with special tokens set to 0 so they do not count.

    Uses the raw token bytes rather than `len(decode(id))`: a token can be a fragment
    of a multi-byte UTF-8 character, and decoding it alone would corrupt the count.
    """
    vocab_size = tokenizer.get_vocab_size()
    special = {tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()}
    table = np.zeros(vocab_size, dtype=np.int64)
    for tid in range(vocab_size):
        if tid not in special:
            table[tid] = len(tokenizer.decode_single_token_bytes(tid))
    return table


@no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """Bits per byte over `steps` batches drawn from the `batches` iterator.

    Three kinds of position are excluded from the metric, all via the same mechanism
    of contributing zero bytes: special tokens (table entry 0), tokens masked with
    `ignore_index=-1`, and padding.
    """
    token_bytes = np.asarray(token_bytes)
    total_nats, total_bytes = 0.0, 0
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        losses = model(x, y, loss_reduction="none").data.reshape(-1)
        y = np.asarray(y).reshape(-1)

        valid = y >= 0
        y_safe = np.where(valid, y, 0)
        num_bytes = np.where(valid, token_bytes[y_safe], 0)
        total_nats += float((losses * (num_bytes > 0)).sum())
        total_bytes += int(num_bytes.sum())

    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


# ----------------------------------------------------------------------------
# prompt rendering

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """One prompt per choice: the shared context, then each candidate answer."""
    shots = "".join(
        f"{ex['query']}{continuation_delimiter}{ex['choices'][ex['gold']]}\n\n"
        for ex in (fewshot_examples or [])
    )
    return [f"{shots}{item['query']}{continuation_delimiter}{choice}" for choice in item["choices"]]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """One prompt per context option, all sharing the same continuation."""
    shots = "".join(
        f"{ex['context_options'][ex['gold']]}{continuation_delimiter}{ex['continuation']}\n\n"
        for ex in (fewshot_examples or [])
    )
    return [f"{shots}{opt}{continuation_delimiter}{item['continuation']}"
            for opt in item["context_options"]]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Two prompts: without and with the continuation.

    Contexts are trimmed because several datasets store trailing whitespace, which
    would otherwise get absorbed into the next token and destroy the clean token-space
    prefix relationship the caller relies on.
    """
    shots = "".join(
        f"{ex['context'].strip()}{continuation_delimiter}{ex['continuation']}\n\n"
        for ex in (fewshot_examples or [])
    )
    without = f"{shots}{item['context'].strip()}{continuation_delimiter}".strip()
    with_cont = f"{shots}{item['context'].strip()}{continuation_delimiter}{item['continuation']}"
    return [without, with_cont]


def find_common_length(sequences, direction="left"):
    """Length of the common prefix ('left') or suffix ('right') across sequences."""
    min_len = min(len(s) for s in sequences)
    indices = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(indices):
        token = sequences[0][idx]
        if not all(s[idx] == token for s in sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Right-pad a list of token sequences into one (B, T) array."""
    bsz, seq_len = len(tokens), max(len(t) for t in tokens)
    out = np.full((bsz, seq_len), pad_token_id, dtype=np.int64)
    for i, t in enumerate(tokens):
        out[i, :len(t)] = t
    return out


def batch_sequences_mc(tokenizer, prompts):
    """Multiple choice: contexts match, continuations differ -> common prefix."""
    tokens = [tokenizer.encode(p, prepend="<|bos|>") for p in prompts]
    start = find_common_length(tokens, "left")
    return tokens, [start] * len(prompts), [len(t) for t in tokens]


def batch_sequences_schema(tokenizer, prompts):
    """Schema: contexts differ, continuation matches -> common suffix."""
    tokens = [tokenizer.encode(p, prepend="<|bos|>") for p in prompts]
    suffix = find_common_length(tokens, "right")
    ends = [len(t) for t in tokens]
    return tokens, [e - suffix for e in ends], ends


def batch_sequences_lm(tokenizer, prompts):
    """Language modeling: one sequence, scored over the continuation span."""
    without, with_cont = [tokenizer.encode(p, prepend="<|bos|>") for p in prompts]
    start, end = len(without), len(with_cont)
    if start >= end or without != with_cont[:start]:
        raise ValueError("the prompt without continuation must be a proper token prefix")
    return [with_cont], [start], [end]


# ----------------------------------------------------------------------------
# task evaluation

@no_grad()
def forward_model(model, input_ids):
    """Return per-position losses and argmax predictions for a (B, T) batch.

    The last column of losses is NaN: position T-1 predicts token T, which is not in
    the sequence, so there is no autoregressive target for it.
    """
    B, T = input_ids.shape
    logits = model(input_ids).data
    targets = np.roll(input_ids, -1, axis=1)

    # log_softmax, then gather the target's log-prob. Stable via the max subtraction.
    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    rows, cols = np.indices((B, T))
    losses = -logp[rows, cols, targets]
    losses[:, -1] = np.nan
    return losses, logits.argmax(axis=-1)


@no_grad()
def evaluate_example(idx, model, tokenizer, data, task_meta):
    """Score one example. Returns True if the model got it right."""
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta.get("num_fewshot", 0)
    delimiter = task_meta.get("continuation_delimiter", " ")

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available = [i for i in range(len(data)) if i != idx]
        fewshot_examples = [data[i] for i in rng.sample(available, num_fewshot)]

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, delimiter, fewshot_examples)
        tokens, starts, ends = batch_sequences_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, delimiter, fewshot_examples)
        tokens, starts, ends = batch_sequences_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, delimiter, fewshot_examples)
        tokens, starts, ends = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"unsupported task type: {task_type}")

    # Crop to what the model can forward, shifting the scored span with it
    max_len = model.config.sequence_len
    cropped_tokens, cropped_starts, cropped_ends = [], [], []
    for t, s, e in zip(tokens, starts, ends):
        if len(t) > max_len:
            shift = len(t) - max_len
            if s - shift < 1:
                raise ValueError("example does not fit: the scored span would be cropped away")
            cropped_tokens.append(t[-max_len:])
            cropped_starts.append(s - shift)
            cropped_ends.append(e - shift)
        else:
            cropped_tokens.append(t)
            cropped_starts.append(s)
            cropped_ends.append(e)

    input_ids = stack_sequences(cropped_tokens, tokenizer.get_bos_token_id())
    losses, predictions = forward_model(model, input_ids)

    # predictions[i] predicts input_ids[i+1], hence the si-1:ei-1 shift
    if task_type == "language_modeling":
        si, ei = cropped_starts[0], cropped_ends[0]
        return bool(np.all(predictions[0, si - 1:ei - 1] == input_ids[0, si:ei]))
    mean_losses = [float(np.mean(losses[i, s - 1:e - 1]))
                   for i, (s, e) in enumerate(zip(cropped_starts, cropped_ends))]
    return int(np.argmin(mean_losses)) == item["gold"]


def evaluate_task(model, tokenizer, data, task_meta, max_examples=None):
    """Accuracy over a task's examples."""
    n = len(data) if max_examples is None else min(len(data), max_examples)
    if n == 0:
        return 0.0
    correct = sum(evaluate_example(i, model, tokenizer, data, task_meta) for i in range(n))
    return correct / n
