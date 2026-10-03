"""
Constitutional AI: principles, and the AI feedback that applies them.

Follows "Constitutional AI: Harmlessness from AI Feedback" (Bai et al. 2022,
arXiv:2212.08073). The only human input is a list of principles -- the constitution.
They are used in two places:

  **SL-CAI (critique -> revision).** Sample a reply, then ask the model to critique it
  against a randomly drawn principle and to revise it accordingly; repeat; finetune on
  the final revisions (mixed with ordinary helpful data).

  **RL-CAI (AI feedback).** For a pair of sampled replies, a feedback model answers a
  multiple-choice question built from one randomly drawn principle:

        Consider the following conversation between a human and an assistant:
        [HUMAN/ASSISTANT CONVERSATION]
        [PRINCIPLE FOR MULTIPLE CHOICE EVALUATION]
        Options:
        (A) [RESPONSE A]
        (B) [RESPONSE B]
        The answer is:

  and the label is *soft*: the normalised probabilities of "(A)" and "(B)". With
  chain-of-thought feedback those probabilities are badly calibrated, so they are
  clamped into [0.4, 0.6]. The preference model is trained on these labels.

Anthropic's "character training" (Claude's Character, 2024) is described as a variant
of the same loop: the model writes messages relevant to a trait, writes several replies,
ranks its own replies by how well they fit the trait, and a preference model is trained
on the rankings. A ranking of k replies is k-1 or more pairwise comparisons, so it runs
through the same comparison/PM machinery.

Two feedback models are provided:

  - `LMFeedback` implements the above exactly with a language model: the multiple-choice
    prompt with soft labels from option log-probabilities, and critique/revision by
    generation. It needs a model that can actually read and judge text.
  - `RuleFeedback` is a stand-in for when no such model exists. A 230K-parameter model
    trained on one-digit addition cannot critique anything, so for the toy domain each
    principle comes with a programmatic check and a programmatic revision. Everything
    downstream (principle sampling, soft labels, clamping, PM training, PPO) is the same
    code either way; only the judge is substituted, and it is labelled as such in every
    script's output.
"""

import math
import re
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Principle", "Response", "ARITHMETIC_CONSTITUTION", "MC_TEMPLATE", "COT_CLAMP",
    "sample_principle", "format_conversation", "clamp_label", "RuleFeedback", "LMFeedback",
    "arithmetic_truth", "ranking_to_comparisons",
]

COT_CLAMP = (0.4, 0.6)  # the paper's clamp for chain-of-thought labels

MC_TEMPLATE = ("Consider the following conversation between a human and an assistant:\n"
               "{conversation}\n{principle}\nOptions:\n(A) {a}\n(B) {b}\nThe answer is:")


@dataclass(frozen=True)
class Principle:
    name: str
    comparison: str        # the multiple-choice question for RL-CAI feedback
    critique_request: str  # SL-CAI: appended as a user turn after the reply
    revision_request: str  # SL-CAI: appended after the critique


@dataclass(frozen=True)
class Response:
    text: str
    stop_reason: str = "end_turn"


# The constitution for the toy domain. The phrasing follows the paper's pattern (a
# comparison question; a critique request; a revision request); the content is about
# what this domain can express. Harmlessness principles need a model that understands
# requests well enough to recognise harm -- `LMFeedback` takes any list of principles.
ARITHMETIC_CONSTITUTION = (
    Principle(
        name="correct",
        comparison="Which of these assistant responses gives the correct answer to the "
                   "human's calculation?",
        critique_request="Identify whether the assistant's answer to the calculation is "
                         "correct, and if it is not, what the correct answer is.",
        revision_request="Please rewrite the assistant's response so that it gives the "
                         "correct answer."),
    Principle(
        name="concise",
        comparison="Which of these assistant responses answers with just the result, "
                   "without unnecessary text?",
        critique_request="Identify any text in the assistant's response that is not the "
                         "result the human asked for.",
        revision_request="Please rewrite the assistant's response so that it contains "
                         "only the result."),
    Principle(
        name="complete",
        comparison="Which of these assistant responses finishes its answer rather than "
                   "trailing off?",
        critique_request="Identify whether the assistant's response is complete or is "
                         "cut off before it finishes.",
        revision_request="Please rewrite the assistant's response so that it is complete."),
)


def sample_principle(constitution, rng):
    """One principle, uniformly at random: the paper draws one per comparison/revision."""
    return constitution[int(rng.integers(len(constitution)))]


def format_conversation(messages):
    lines = []
    for m in messages:
        who = "Human" if m["role"] == "user" else "Assistant"
        content = m["content"] if isinstance(m["content"], str) else \
            "".join(p["text"] for p in m["content"] if p["type"] == "text")
        lines.append(f"{who}: {content}")
    return "\n\n".join(lines)


def clamp_label(p, clamp=None):
    return p if clamp is None else min(max(p, clamp[0]), clamp[1])


def ranking_to_comparisons(responses, ranking):
    """A best-first ranking of k responses -> all k(k-1)/2 hard-labelled pairs.

    `ranking` lists indices into `responses`, best first. This is how a self-ranking
    (character training) feeds the same pairwise preference-model loss.
    """
    order = list(ranking)
    if sorted(order) != list(range(len(responses))):
        raise ValueError("ranking must be a permutation of the response indices")
    return [(responses[order[i]], responses[order[j]], 1.0)
            for i in range(len(order)) for j in range(i + 1, len(order))]


# ----------------------------------------------------------------------------
# stand-in feedback for the toy domain

def arithmetic_truth(messages):
    """The correct answer to the last user message, or None if it is not arithmetic."""
    from nanochat.scratch.engine import use_calculator
    result = use_calculator(messages[-1]["content"])
    if result is None:
        return None
    return str(int(result)) if float(result).is_integer() else str(result)


def _check(principle, messages, response):
    """A principle's verdict on one response, in [0, 1], or None if it does not apply."""
    text = response.text
    if principle.name == "correct":
        truth = arithmetic_truth(messages)
        return None if truth is None else float(text.strip() == truth)
    if principle.name == "concise":
        return float(bool(re.fullmatch(r"-?\d+(\.\d+)?", text)))
    if principle.name == "complete":
        return float(response.stop_reason == "end_turn")
    raise KeyError(f"no programmatic check for principle {principle.name!r}")


class RuleFeedback:
    """Programmatic stand-in for the feedback model, for domains where principles can
    be checked exactly. Produces the same kind of output as `LMFeedback`: a soft label
    for a comparison, and a (critique, revision) pair.

    `confidence` is the label given to a clear preference (the paper's labels come from
    a model's probabilities and are rarely exactly 0 or 1).
    """

    is_stand_in = True

    def __init__(self, confidence=0.9):
        if not 0.5 < confidence <= 1.0:
            raise ValueError("confidence must be in (0.5, 1]")
        self.confidence = confidence

    def compare(self, messages, a, b, principle, clamp=None):
        sa, sb = _check(principle, messages, a), _check(principle, messages, b)
        if sa is None or sb is None or sa == sb:
            p = 0.5
        else:
            p = 0.5 + (sa - sb) * (self.confidence - 0.5)
        return clamp_label(p, clamp)

    def revise(self, messages, response, principle):
        text = response.text
        if principle.name == "correct":
            truth = arithmetic_truth(messages)
            if truth is None or text.strip() == truth:
                return "The answer is correct.", text
            return f"The answer {text.strip()!r} is incorrect; the correct result is {truth}.", truth
        if principle.name == "concise":
            number = re.search(r"-?\d+(\.\d+)?", text)
            revised = number.group(0) if number else text.strip()
            critique = ("The response contains only the result." if revised == text
                        else "The response contains text other than the result.")
            return critique, revised
        if principle.name == "complete":
            # rendering a revision as a finished assistant turn completes it
            return ("The response is complete." if response.stop_reason == "end_turn"
                    else "The response is cut off."), text
        raise KeyError(f"no programmatic revision for principle {principle.name!r}")


# ----------------------------------------------------------------------------
# the real thing: a language model as the feedback model

class LMFeedback:
    """Constitutional AI feedback from a language model, as in the paper.

    `compare` renders the multiple-choice prompt (after optional few-shot examples)
    and returns the normalised probability of " (A)" versus " (B)" as the next text.
    `revise` appends the principle's critique request, generates a critique, appends
    the revision request and generates the revision.
    """

    is_stand_in = False

    def __init__(self, model, tokenizer, few_shot="", max_new_tokens=64):
        from nanochat.scratch.engine import Engine
        self.model, self.tokenizer = model, tokenizer
        self.engine = Engine(model, tokenizer)
        self.few_shot = few_shot
        self.max_new_tokens = max_new_tokens

    def comparison_prompt(self, messages, a, b, principle):
        return self.few_shot + MC_TEMPLATE.format(
            conversation=format_conversation(messages), principle=principle.comparison,
            a=a.text, b=b.text)

    def _continuation_logprob(self, prompt_ids, cont_ids):
        from nanochat.scratch.tensor import no_grad
        seq = list(prompt_ids) + list(cont_ids)
        budget = self.model.config.sequence_len
        if len(seq) > budget:  # keep BOS, drop the oldest text
            seq = seq[:1] + seq[-(budget - 1):]
        with no_grad():
            logits = self.model(np.asarray([seq], dtype=np.int64)).data[0].astype(np.float64)
        z = logits - logits.max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        n = len(cont_ids)
        positions = range(len(seq) - n - 1, len(seq) - 1)
        return float(sum(logp[pos, seq[pos + 1]] for pos in positions))

    def compare(self, messages, a, b, principle, clamp=None):
        tok = self.tokenizer
        prompt = [tok.get_bos_token_id()] + tok.encode(self.comparison_prompt(messages, a, b, principle))
        la = self._continuation_logprob(prompt, tok.encode(" (A)"))
        lb = self._continuation_logprob(prompt, tok.encode(" (B)"))
        p = 1.0 / (1.0 + math.exp(lb - la))
        return clamp_label(p, clamp)

    def _reply(self, messages):
        from nanochat.chat_format import parse_reply, render_prompt, reply_stop_tokens, reply_text
        ids = render_prompt(self.tokenizer, messages)
        out = self.engine.generate_batch(ids, max_tokens=self.max_new_tokens, temperature=0.0,
                                         stop_tokens=sorted(set(reply_stop_tokens(self.tokenizer))))[0]
        return reply_text(parse_reply(self.tokenizer, out)[0]).strip()

    def revise(self, messages, response, principle):
        history = list(messages) + [{"role": "assistant", "content": response.text},
                                    {"role": "user", "content": principle.critique_request}]
        critique = self._reply(history)
        history += [{"role": "assistant", "content": critique},
                    {"role": "user", "content": principle.revision_request}]
        return critique, self._reply(history)
