"""
The chat wire format, shared by SFT (training) and chat_cli / evaluation (inference).

Training and inference have to agree on the layout token for token, which is why both
sides import it from here instead of each spelling it out.

Two tokenizers, two layouts:

  - a BPE tokenizer carries dedicated special tokens, so it uses upstream's layout
        <|bos|><|user_start|> ... <|user_end|><|assistant_start|> ... <|assistant_end|>
  - `ByteTokenizer` has 256 ids and none to spare, so turns are plain text after BOS
        U:<user text>\\n A:<assistant text>\\n
    with the newline ending the assistant's turn.

In both, only the assistant's reply (and its terminator) is trained on.
"""

__all__ = [
    "has_chat_tokens", "render_conversation", "render_prompt", "reply_stop_tokens",
    "fit_history",
]


def has_chat_tokens(tokenizer):
    try:
        tokenizer.encode_special("<|assistant_start|>")
        return True
    except KeyError:
        return False


def render_conversation(tokenizer, messages):
    """Token ids for a whole conversation, plus a 1/0 mask of positions to train on.

    `messages` alternates user/assistant, starting with the user.
    """
    messages = list(messages)
    if has_chat_tokens(tokenizer):
        return tokenizer.render_conversation({"messages": messages}, max_tokens=None)

    ids, mask = [tokenizer.get_bos_token_id()], [0]
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(f"message {i} is from {message['role']!r}, expected {expected!r}")
        body = tokenizer.encode(message["content"])
        newline = tokenizer.encode("\n")
        if message["role"] == "user":
            segment = tokenizer.encode("U:") + body + newline
            ids += segment
            mask += [0] * len(segment)
        else:
            prefix = tokenizer.encode("A:")
            ids += prefix + body + newline
            mask += [0] * len(prefix) + [1] * (len(body) + len(newline))
    return ids, mask


def render_prompt(tokenizer, messages):
    """A history ending in a user turn -> ids primed for the assistant's reply."""
    messages = list(messages)
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("the prompt must end with a user message")
    ids, _ = render_conversation(tokenizer, messages)
    if has_chat_tokens(tokenizer):
        return ids + [tokenizer.encode_special("<|assistant_start|>")]
    return ids + tokenizer.encode("A:")


def reply_stop_tokens(tokenizer):
    """The token(s) that end an assistant reply."""
    if has_chat_tokens(tokenizer):
        return [tokenizer.encode_special("<|assistant_end|>"), tokenizer.get_bos_token_id()]
    return tokenizer.encode("\n")


def fit_history(tokenizer, messages, budget):
    """Drop the oldest exchanges until the primed prompt fits in `budget` tokens.

    Returns `(kept_messages, prompt_ids)`. Exchanges are dropped whole (a user turn
    together with its reply) so the history always still starts with the user. Raises
    if even the latest user turn on its own does not fit.
    """
    messages = list(messages)
    while True:
        ids = render_prompt(tokenizer, messages)
        if len(ids) <= budget:
            return messages, ids
        if len(messages) <= 1:
            raise ValueError(f"the message alone needs {len(ids)} tokens, "
                             f"but only {budget} fit in the context")
        messages = messages[2:]
