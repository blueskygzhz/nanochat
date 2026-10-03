"""
Tests for scripts/chat_benchmarks.py: running the tasks/ benchmarks against a model.

Offline: the real ARC / MMLU / GSM8K task classes are used, with the hub download
replaced by small in-memory tables, and the model is a stub with a known preference.
That pins down the scoring rules (letter restriction, chance, centering, prompt
cropping, answer extraction) independently of how good any model is.
"""

import numpy as np
import pyarrow as pa
import pytest

from nanochat.scratch import ByteTokenizer, Engine, GPTConfig, Tensor
from scripts import chat_benchmarks as cb
from tasks.common import HubDataset


class Prefers:
    """A model whose next-token logits always favour `token` (then `second`)."""

    def __init__(self, token, second=None, sequence_len=64):
        self.token, self.second = token, second
        self.config = GPTConfig(n_layer=1, n_head=2, n_kv_head=1, n_embd=32,
                                sequence_len=sequence_len, vocab_size=256)
        self.seen_lengths = []

        class _W:
            weight = type("_P", (), {"data": np.zeros(1, dtype=np.float32)})()
        self.wte = _W()

    def __call__(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        idx = np.asarray(idx)
        B, T = idx.shape
        self.seen_lengths.append(T)
        logits = np.zeros((B, T, 256), dtype=np.float32)
        logits[..., self.token] = 10.0
        if self.second is not None:
            logits[..., self.second] = 5.0
        if kv_cache is not None:
            kv_cache.advance(T)
        return Tensor(logits)


def _fake_hub(monkeypatch, module, rows):
    table = pa.Table.from_pylist(rows)
    monkeypatch.setattr(module, "load_hub_dataset", lambda *a, **k: HubDataset(table))


@pytest.fixture
def arc(monkeypatch):
    import tasks.arc
    rows = [{"question": f"q{i}", "answerKey": "AB"[i % 2],
             "choices": {"text": ["x", "y"], "label": ["A", "B"]}} for i in range(10)]
    _fake_hub(monkeypatch, tasks.arc, rows)
    return tasks.arc.ARC("ARC-Easy", "test")


@pytest.fixture
def mmlu(monkeypatch):
    import tasks.mmlu
    rows = [{"question": "long " * 40, "choices": ["a", "b", "c", "d"], "answer": 2,
             "subject": "s"} for _ in range(4)]
    _fake_hub(monkeypatch, tasks.mmlu, rows)
    return tasks.mmlu.MMLU("all", "test")


def test_categorical_picks_the_preferred_letter_and_centers_against_chance(arc):
    tok = ByteTokenizer()
    # half the answers are A, half B; always answering A is exactly chance
    r = cb.run_categorical(Prefers(ord("A")), tok, arc)
    assert r["n"] == 10 and r["correct"] == 5
    assert r["chance"] == pytest.approx(0.5) and r["centered"] == pytest.approx(0.0)


def test_categorical_ignores_tokens_that_are_not_answer_letters(arc):
    """The model's favourite token is 'z'; only its preference *among the letters*
    (B over A) may count."""
    r = cb.run_categorical(Prefers(ord("z"), second=ord("B")), ByteTokenizer(), arc)
    assert r["correct"] == 5  # the B-answer half


def test_categorical_respects_max_problems(arc):
    assert cb.run_categorical(Prefers(ord("A")), ByteTokenizer(), arc, max_problems=3)["n"] == 3


def test_long_prompts_are_cropped_to_the_context_and_counted(mmlu):
    tok = ByteTokenizer()
    model = Prefers(ord("C"), sequence_len=32)
    r = cb.run_categorical(model, tok, mmlu)
    assert r["cropped"] == 4 and r["correct"] == 4
    assert max(model.seen_lengths) == 32, "nothing longer than the context reaches the model"
    assert r["chance"] == pytest.approx(0.25) and r["centered"] == pytest.approx(1.0)


def test_crop_prompt_keeps_bos_and_the_tail():
    ids, cut = cb.crop_prompt([9, 1, 2, 3, 4, 5], budget=4)
    assert cut and ids == [9, 3, 4, 5]
    assert cb.crop_prompt([9, 1, 2], budget=4) == ([9, 1, 2], False)
    with pytest.raises(ValueError):
        cb.crop_prompt([1, 2, 3], budget=1)


def test_generative_scores_with_the_tasks_own_answer_extraction(monkeypatch):
    """GSM8K's evaluate looks for '#### <number>'. A model that emits '#### 7' and
    then ends its turn is right exactly on the problems whose answer is 7."""
    import tasks.gsm8k
    rows = [{"question": "q", "answer": "work <<3+4=7>>7\n#### 7"},
            {"question": "q", "answer": "work\n#### 8"}]
    _fake_hub(monkeypatch, tasks.gsm8k, rows)
    task = tasks.gsm8k.GSM8K("main", "test")

    reply = [ord(c) for c in "#### 7"] + [ord("\n")]

    class Scripted(Prefers):
        def __call__(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
            step = getattr(self, "_step", 0)
            self.token = reply[min(step, len(reply) - 1)]
            self._step = step + 1
            return super().__call__(idx, targets, kv_cache, loss_reduction)

    tok = ByteTokenizer()
    correct = 0
    for i in range(len(task)):
        model = Scripted(0)
        sub = type("One", (), {"__len__": lambda s: 1, "__getitem__": lambda s, j: task[i],
                               "evaluate": task.evaluate, "eval_type": "generative"})()
        correct += cb.run_generative(Engine(model, tok), tok, sub)["correct"]
    assert correct == 1


def test_parse_task_list():
    assert cb.parse_task_list("") == []
    assert cb.parse_task_list("all") == list(cb.TASK_NAMES)
    assert cb.parse_task_list("MMLU, GSM8K") == ["MMLU", "GSM8K"]
    with pytest.raises(ValueError, match="unknown task"):
        cb.parse_task_list("SuperGLUE")


def test_format_results_reports_failures_and_chatcore():
    results = {"A": cb._result(5, 10, 0.5, 2), "B": cb._result(0, 10, 0.0, 0),
               "C": {"error": "URLError: offline"}}
    text = "\n".join(cb.format_results(results))
    assert "failed: URLError: offline" in text
    assert "2/10" in text and "ChatCORE" in text


def test_sandbox_runs_ordinary_code_and_still_caps_memory():
    """Regression: the absolute 256MB RLIMIT_AS cap made *every* program fail with
    MemoryError on hosts where a bare interpreter already maps ~240MB, silently
    scoring every HumanEval answer as wrong. The budget is now on top of startup."""
    from nanochat.execution import execute_code
    assert execute_code("print(1 + 1)").stdout.strip() == "2"
    assert execute_code("x = bytearray(64 * 1024 * 1024)").success
    blocked = execute_code("x = bytearray(1024 * 1024 * 1024)")
    assert not blocked.success and blocked.memory_exceeded
