import json

from agent import SYSTEM_PROMPT
from questions import EVAL_GAMES
from questions import TEMPLATES, generate_train
from synthetic_traces import ANSWER_PHRASES, CALL_PHRASES, PLAN, generate_traces, validate


def test_traces():
    traces, dropped = generate_traces()
    assert (traces, dropped) == generate_traces()  # fixed seed
    # every template has a plan, and every training question gets a valid trace
    assert set(PLAN) == set(TEMPLATES)
    assert dropped == [] and [t["id"] for t in traces] == [q["id"] for q in generate_train()]
    assert not any(str(game_id) in t["id"] for t in traces for game_id in EVAL_GAMES)
    for t in traces:
        roles = [m["role"] for m in t["messages"]]
        assert roles[:3] == ["system", "user", "assistant"] and roles[-1] == "assistant"
        assert set(roles[3:-1]) == {"tool"} and len(roles) - 4 == len(t["messages"][2]["tool_calls"])
        assert t["messages"][0]["content"] == SYSTEM_PROMPT
        assert t["messages"][2]["content"] and "FINAL ANSWER: " in t["messages"][-1]["content"].splitlines()[-1]
        json.dumps(t)
    comparisons = [t for t in traces if t["question_type"] == "comparison"]
    assert comparisons and all(len(t["messages"][2]["tool_calls"]) == 2 for t in comparisons)
    assert len(CALL_PHRASES) >= 5 and len(ANSWER_PHRASES) >= 5


def test_trace_files_on_disk_are_current():
    traces, _ = generate_traces()
    full = [json.loads(line) for line in open("traces/synthetic/traces_synthetic.jsonl")]
    clean = [json.loads(line) for line in open("traces/synthetic/traces_clean.jsonl")]
    assert full == traces
    assert clean == [{"messages": t["messages"], "tools": t["tools"]} for t in traces]
    assert all(list(line) == ["messages", "tools"] for line in clean)


def test_period_and_score_traces():
    traces, _ = generate_traces()
    metas = {q["id"]: q for q in generate_train(with_meta=True)}
    seen = set()
    for t in traces:
        q = metas[t["id"]]
        call = t["messages"][2]["tool_calls"][0]["function"]
        args = json.loads(call["arguments"])
        if q["meta"]["subtype"] == "player_points_in":
            seen.add("player_points_in")
            assert call["name"] == "get_box_score" and args["period"] == q["meta"]["period_num"] and "player" in args
        if q["answer_type"] == "score":
            seen.add("score")
            assert t["messages"][-1]["content"].splitlines()[-1] == f"FINAL ANSWER: {q['answer']}"
    assert seen == {"player_points_in", "score"}


def test_validate_catches_bad_traces():
    traces, _ = generate_traces()
    rows = {q["id"]: q for q in map(json.loads, open("datasets/train.jsonl"))}
    trace = json.loads(json.dumps(traces[0]))
    row = rows[trace["id"]]
    assert validate(trace, row) is None
    wrong_answer = json.loads(json.dumps(trace))
    wrong_answer["messages"][-1]["content"] = "FINAL ANSWER: 999"
    assert "ground truth" in validate(wrong_answer, row)
    wrong_result = json.loads(json.dumps(trace))
    wrong_result["messages"][3]["content"] = json.dumps({"teams": []})
    assert "differs" in validate(wrong_result, row)
