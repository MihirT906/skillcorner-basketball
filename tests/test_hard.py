import json
from pathlib import Path

import questions
from questions import EVAL_GAMES, MIX
from questions_hard import SUBTYPES, TEMPLATES, generate_eval_hard, generate_train_hard
from synthetic_traces_hard import PLAN, generate_traces_hard

HERE = Path(__file__).resolve().parent.parent


def read(path):
    return [json.loads(line) for line in (HERE / path).read_text().splitlines()]


def test_hard_templates():
    assert all(len(set(phrasings)) >= 3 for phrasings in TEMPLATES.values())
    assert set(SUBTYPES) == set(MIX) and sorted(k for keys in SUBTYPES.values() for k in keys) == sorted(TEMPLATES)
    assert set(PLAN) == set(TEMPLATES) and not set(TEMPLATES) & set(questions.TEMPLATES)


def test_hard_question_files_are_current_and_separate_from_the_basic_ones():
    train, evals = read("datasets/train_hard.jsonl"), read("datasets/eval_hard.jsonl")
    assert train == generate_train_hard() and evals == generate_eval_hard()
    assert len(train) == 240 and len(evals) == 40
    assert {r["game_id"] for r in evals} == set(EVAL_GAMES) and not {r["game_id"] for r in train} & set(EVAL_GAMES)
    basic = read("datasets/train.jsonl") + read("datasets/eval.jsonl")
    for key in ("id", "question"):
        values = [r[key] for r in train + evals + basic]
        assert len(set(values)) == len(values)


def test_every_hard_question_is_answerable_with_the_tools():
    """Train and eval: the planned tool calls replay cleanly and the graded answer matches ground truth."""
    traces, dropped = generate_traces_hard()
    assert dropped == [] and [t["id"] for t in traces] == [q["id"] for q in generate_train_hard()]
    eval_traces, eval_dropped = generate_traces_hard(generate_eval_hard(with_meta=True), "eval_hard.jsonl")
    assert eval_dropped == [] and len(eval_traces) == 40
    # more than one tool call, or arithmetic / extra filters on one
    assert sum(len(t["messages"][2]["tool_calls"]) == 2 for t in traces) >= 80


def test_hard_trace_files_are_current_and_exclude_eval_games():
    traces, _ = generate_traces_hard()
    assert read("traces/synthetic/traces_hard_synthetic.jsonl") == traces
    assert read("traces/synthetic/traces_hard_clean.jsonl") == [{"messages": t["messages"], "tools": t["tools"]} for t in traces]
    assert not any(str(game_id) in t["id"] for t in traces for game_id in EVAL_GAMES)
