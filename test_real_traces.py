"""gen_real_traces and merge_traces, against a scripted fake client (no network)."""

import json

import pytest

from agent import SYSTEM_PROMPT
from gen_real_traces import HERE, real_trace
from merge_traces import merge
from test_agent import FakeClient, reply, tool_call

ROW = json.loads((HERE / "train.jsonl").read_text().splitlines()[0])


def run(final, calls=True):
    game_id = str(ROW["game_id"])
    replies = [reply(tool_calls=[tool_call(1, "get_box_score", game_id=game_id)])] if calls else []
    return real_trace(ROW, FakeClient(replies + [reply(content=final)]), "m")


def test_real_trace_kept_only_when_correct_and_tool_backed():
    trace, reason = run(f"FINAL ANSWER: {ROW['answer']}")
    assert reason is None and trace["id"] == ROW["id"] and trace["model"] == "m"
    assert [m["role"] for m in trace["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert trace["messages"][0]["content"] == SYSTEM_PROMPT
    assert run("FINAL ANSWER: 999")[0] is None
    assert run(f"FINAL ANSWER: {ROW['answer']}", calls=False) == (None, "answered without calling a tool")


def test_merge_prefers_real_and_refuses_eval_games():
    synthetic = [json.loads(line) for line in (HERE / "traces_synthetic.jsonl").read_text().splitlines()[:3]]
    real = [{**synthetic[0], "model": "m"}]
    assert merge(synthetic, real) == synthetic[1:] + real
    assert len(merge(synthetic, real, keep_both=True)) == 4
    with pytest.raises(ValueError):
        merge([{**synthetic[0], "id": "191313-player_stat-01"}], [])
    with pytest.raises(ValueError):
        merge([{**synthetic[0], "messages": [{"role": "system", "content": "other"}]}], [])
