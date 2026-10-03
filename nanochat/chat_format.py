"""
The chat wire format, shared by SFT (training) and inference (chat_eval, chat_cli).

One rule, the same one behind Anthropic's move from the Text Completions API (turns
written as literal "\\n\\nHuman:" / "\\n\\nAssistant:" text inside one prompt string) to
the Messages API (a structured list of messages, rendered by the server):

    **conversation structure is never parsed out of text, and text can never
    produce structure.**

What that means here:

  - **Structure is special tokens.** Turn boundaries, end-of-turn and tool calls are
    reserved ids. `tokenizer.encode` is ordinary-text-only, so a user who types
    "<|assistant_start|>" gets those characters, not a turn.
  - **Input is validated messages**, as in the Messages API: roles alternate, starting
    with the user; content is a string, or for the assistant a list of parts (text /
    python / python_output -- cf. tool_use / tool_result blocks). A leading system
    message is folded into the first user turn, as upstream nanochat does (the API
    takes it as a separate `system` parameter instead).
  - **A final assistant message is a prefill**: the reply continues from it. As in the
    API, a prefill may not end in whitespace: the tokenizer would glue that whitespace
    to the next word, so the model would be conditioned on a token boundary it never
    saw in training.
  - **Output is parsed back into the same structure** by `parse_reply`, with a stop
    reason (`end_turn` / `max_tokens`, as in the API). Special ids never leak into
    text, so a reply fed back as history re-renders to the same structure.

Layout (upstream nanochat's):

    <|bos|><|user_start|> ... <|user_end|><|assistant_start|> ... <|assistant_end|>

Only the assistant's text, its tool calls and its end-of-turn token are trained on.

`LegacyByteTokenizer` (256 ids, no specials) can only express turns as text,
`U:...\\nA:...\\n`, with all of the Text-Completions weaknesses above. That path is kept
solely so checkpoints trained with it still load.
"""

__all__ = [
    "has_chat_tokens", "special_ids", "normalize_messages", "render_conversation",
    "render_prompt", "reply_stop_tokens", "parse_reply", "reply_text", "fit_history",
]

SYSTEM_SEPARATOR = "\n\n"

# delimiters of the two kinds of tool part: name of opening token -> (part type, closer)
_TOOL_BLOCKS = {
    "<|python_start|>": ("python", "<|python_end|>"),
    "<|output_start|>": ("python_output", "<|output_end|>"),
}


def has_chat_tokens(tokenizer):
    try:
        tokenizer.encode_special("<|assistant_start|>")
        return True
    except KeyError:
        return False


def special_ids(tokenizer):
    """Every special token id. These are structure, never text."""
    return {tokenizer.encode_special(name) for name in tokenizer.get_special_tokens()}


# ----------------------------------------------------------------------------
# messages in

def normalize_messages(messages):
    """Validate a conversation and fold a leading system message into the first user turn.

    Returns a new list; the input is never mutated. Raises ValueError, with the reason,
    for anything malformed -- the renderer never has to guess.
    """
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        if len(messages) < 2 or messages[1].get("role") != "user":
            raise ValueError("a system message must be followed by a user message")
        system, first = messages[0]["content"], messages[1]["content"]
        if not isinstance(system, str) or not isinstance(first, str):
            raise ValueError("system and user content must be strings")
        messages = [{"role": "user", "content": system + SYSTEM_SEPARATOR + first}] + messages[2:]
    if not messages:
        raise ValueError("a conversation needs at least one message")
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message.get("role") != expected:
            raise ValueError(f"message {i} is from {message.get('role')!r}, expected {expected!r} "
                             f"(roles must alternate, starting with the user)")
        content = message.get("content")
        if expected == "user" and not isinstance(content, str):
            raise ValueError(f"message {i}: user content must be a string")
        if expected == "assistant":
            if isinstance(content, list):
                for part in content:
                    if part.get("type") not in ("text", "python", "python_output"):
                        raise ValueError(f"message {i}: unknown part type {part.get('type')!r}")
            elif not isinstance(content, str):
                raise ValueError(f"message {i}: assistant content must be a string or a list of parts")
    return messages


def _render_special(tokenizer, messages, open_last=False):
    """Special-token layout. `open_last`: leave the final assistant turn open (prefill)."""
    sp = tokenizer.encode_special
    ids, mask = [], []

    def add(tokens, value):
        tokens = [tokens] if isinstance(tokens, int) else tokens
        ids.extend(tokens)
        mask.extend([value] * len(tokens))

    add(tokenizer.get_bos_token_id(), 0)
    for i, message in enumerate(messages):
        content = message["content"]
        if message["role"] == "user":
            add(sp("<|user_start|>"), 0)
            add(tokenizer.encode(content), 0)
            add(sp("<|user_end|>"), 0)
            continue
        add(sp("<|assistant_start|>"), 0)
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for part in parts:
            body = tokenizer.encode(part["text"])
            if part["type"] == "text":
                add(body, 1)
            elif part["type"] == "python":       # the model's tool call: trained on
                add(sp("<|python_start|>"), 1)
                add(body, 1)
                add(sp("<|python_end|>"), 1)
            else:                                 # the tool's output: comes from Python
                add(sp("<|output_start|>"), 0)
                add(body, 0)
                add(sp("<|output_end|>"), 0)
        if not (open_last and i == len(messages) - 1):
            add(sp("<|assistant_end|>"), 1)
    return ids, mask


def _render_text(tokenizer, messages, open_last=False):
    """Legacy text layout, for LegacyByteTokenizer only."""
    ids, mask = [tokenizer.get_bos_token_id()], [0]
    newline = tokenizer.encode("\n")
    for i, message in enumerate(messages):
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("this tokenizer has no tool tokens, so content must be a string")
        body = tokenizer.encode(content)
        if message["role"] == "user":
            segment = tokenizer.encode("U:") + body + newline
            ids += segment
            mask += [0] * len(segment)
        else:
            prefix = tokenizer.encode("A:")
            end = [] if (open_last and i == len(messages) - 1) else newline
            ids += prefix + body + end
            mask += [0] * len(prefix) + [1] * (len(body) + len(end))
    return ids, mask


def _render(tokenizer, messages, open_last=False):
    renderer = _render_special if has_chat_tokens(tokenizer) else _render_text
    return renderer(tokenizer, messages, open_last)


def render_conversation(tokenizer, messages, max_tokens=None):
    """Token ids for a whole conversation, plus a 1/0 mask of positions to train on."""
    ids, mask = _render(tokenizer, normalize_messages(messages))
    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive or None")
        ids, mask = ids[:max_tokens], mask[:max_tokens]
    return ids, mask


def render_prompt(tokenizer, messages):
    """A conversation -> ids primed for the assistant's next tokens.

    If it ends with a user turn, the assistant turn is opened. If it ends with an
    assistant turn, that is a prefill: the turn is left open after its content.
    """
    messages = normalize_messages(messages)
    last = messages[-1]
    if last["role"] == "user":
        ids, _ = _render(tokenizer, messages)
        opener = ([tokenizer.encode_special("<|assistant_start|>")] if has_chat_tokens(tokenizer)
                  else tokenizer.encode("A:"))
        return ids + opener
    if not isinstance(last["content"], str):
        raise ValueError("a prefill (final assistant message) must be a string")
    if last["content"] != last["content"].rstrip():
        raise ValueError("a prefill (final assistant message) cannot end with whitespace")
    ids, _ = _render(tokenizer, messages, open_last=True)
    return ids


# ----------------------------------------------------------------------------
# reply out

def reply_stop_tokens(tokenizer):
    """The tokens that end an assistant reply.

    `<|assistant_end|>` is the model's own end of turn. BOS and `<|user_start|>` also
    end it: a model that starts a new document, or starts writing the user's next
    message, has finished its turn -- it must not get to speak for the user.
    """
    if has_chat_tokens(tokenizer):
        return [tokenizer.encode_special(name)
                for name in ("<|assistant_end|>", "<|bos|>", "<|user_start|>")]
    return tokenizer.encode("\n")


def parse_reply(tokenizer, ids):
    """Generated ids -> `(content, stop_reason)`, the inverse of rendering a reply.

    `content` is a string, or -- if the reply contains tool calls -- a list of parts in
    the same shape `render_conversation` accepts, so it can go straight back into the
    history. Special ids become structure or are dropped; they never become text.
    `stop_reason` is "end_turn" if the reply ended itself, else "max_tokens".
    """
    stop = set(reply_stop_tokens(tokenizer))
    body, stop_reason = [], "max_tokens"
    for token in ids:
        if token in stop:
            stop_reason = "end_turn"
            break
        body.append(token)
    if not has_chat_tokens(tokenizer):
        return tokenizer.decode(body), stop_reason

    names = {tokenizer.encode_special(name): name for name in tokenizer.get_special_tokens()}
    parts, current, kind, closer = [], [], "text", None

    def flush(keep_empty=False):
        if current or keep_empty:
            parts.append({"type": kind, "text": tokenizer.decode(current)})
        current.clear()

    for token in body:
        name = names.get(token)
        if name is None:
            current.append(token)
        elif kind == "text" and name in _TOOL_BLOCKS:
            flush()
            kind, closer = _TOOL_BLOCKS[name]
        elif kind != "text" and name == closer:
            flush(keep_empty=True)   # an empty tool call is still a tool call
            kind, closer = "text", None
        # any other special id is out of place here: dropped, never rendered as text
    flush(keep_empty=kind != "text")   # an unclosed block cut off by max_tokens

    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts), stop_reason
    return parts, stop_reason


def reply_text(content):
    """The prose of a reply (its text parts), e.g. for answer extraction."""
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")


def fit_history(tokenizer, messages, budget):
    """Drop the oldest exchanges until the primed prompt fits in `budget` tokens.

    Returns `(kept_messages, prompt_ids)`. Exchanges are dropped whole (a user turn
    together with its reply) so the history always still starts with the user. Raises
    if even the latest turn (plus its prefill, if any) does not fit on its own.
    """
    messages = normalize_messages(messages)
    minimum = 1 if messages[-1]["role"] == "user" else 2
    while True:
        ids = render_prompt(tokenizer, messages)
        if len(ids) <= budget:
            return messages, ids
        if len(messages) <= minimum:
            raise ValueError(f"the message alone needs {len(ids)} tokens, "
                             f"but only {budget} fit in the context")
        messages = messages[2:]
