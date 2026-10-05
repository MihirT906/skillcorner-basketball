"""Agent loop tests against a scripted fake client (no network)."""

import json
from types import SimpleNamespace as NS

from agent import OPENAI_TOOLS, run_agent
from grade import grade
from run_eval import summarize


def tool_call(i, name, **args):
    return NS(id=f"call_{i}", function=NS(name=name, arguments=json.dumps(args)))


def reply(content=None, tool_calls=None):
    usage = NS(prompt_tokens=100, completion_tokens=10, prompt_tokens_details=NS(cached_tokens=40))
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))], usage=usage)


class FakeClient:
    """Returns scripted replies in order and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


def test_schemas_convert_to_openai_format():
    assert [t["function"]["name"] for t in OPENAI_TOOLS] == [
        "get_score_by_period", "get_box_score", "get_team_stats", "get_shots", "get_picks"]
    assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in OPENAI_TOOLS)


def test_agent_calls_tool_and_answers():
    client = FakeClient([
        reply(tool_calls=[tool_call(1, "get_score_by_period", game_id="191313")]),
        reply("Zaragoza won 95-94.\nFINAL ANSWER: Casademont Zaragoza"),
    ])
    result = run_agent("Who won game 191313?", client=client, model="fake")
    assert [c["name"] for c in result["tool_calls"]] == ["get_score_by_period"]
    assert result["tool_calls"][0]["result"]["away"]["final"] == 95
    # the tool result went back to the model
    tool_msg = client.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["home"]["final"] == 94
    assert result["usage"] == {"input_tokens": 200, "cached_input_tokens": 80, "output_tokens": 20}
    assert grade(result["response"], "Casademont Zaragoza", "team")["correct"]


def test_tool_errors_are_returned_to_the_model():
    client = FakeClient([
        reply(tool_calls=[tool_call(1, "get_team_stats", game_id="191313", team="Lakers"),
                          tool_call(2, "no_such_tool"),
                          tool_call(3, "get_shots", game_id="191313", bogus=1)]),
        reply("FINAL ANSWER: unknown"),
    ])
    result = run_agent("?", client=client, model="fake")
    assert all("error" in c["result"] for c in result["tool_calls"])


def test_tool_call_limit():
    looping = [reply(tool_calls=[tool_call(i, "get_score_by_period", game_id="191313"),
                                 tool_call(f"{i}b", "get_picks", game_id="191313")]) for i in range(3)]
    client = FakeClient(looping + [reply("FINAL ANSWER: 94")])
    result = run_agent("?", client=client, model="fake", max_tool_calls=5)
    assert len(result["tool_calls"]) == 5
    # the sixth call was refused, and the next request no longer allows tools
    refused = client.requests[3]["messages"][-1]
    assert "limit" in json.loads(refused["content"])["error"]
    assert [r["tool_choice"] for r in client.requests] == ["auto", "auto", "auto", "none"]


def test_summarize():
    usage = {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5}
    results = [
        {"question_type": "leader", "correct": True, "n_tool_calls": 1, "latency_s": 2.0, "usage": usage},
        {"question_type": "leader", "correct": False, "n_tool_calls": 3, "latency_s": 4.0, "usage": usage},
    ]
    s = summarize(results, "fake")
    assert (s["accuracy"], s["avg_tool_calls"], s["avg_latency_s"], s["total_cost_usd"]) == (0.5, 2.0, 3.0, None)
    assert s["by_question_type"] == {"leader": {"n": 2, "correct": 1, "accuracy": 0.5}}


def test_run_eval_cli_points_at_any_endpoint(tmp_path, monkeypatch):
    import run_eval

    seen = {}
    make_client = run_eval.make_client

    def fake_client(base_url, api_key_env):
        seen.update(base_url=base_url, api_key_env=api_key_env)
        return FakeClient([reply(content="FINAL ANSWER: 0")] * 3)

    monkeypatch.setattr(run_eval, "make_client", fake_client)
    out = tmp_path / "out.json"
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--model", "m", "--base-url", "http://host/v1", "--api-key-env", "MY_KEY",
                                     "--questions", str(run_eval.HERE / "eval.jsonl"), "--limit", "3", "--workers", "1",
                                     "--out", str(out)])
    run_eval.main()
    assert seen == {"base_url": "http://host/v1", "api_key_env": "MY_KEY"}
    assert json.loads(out.read_text())["summary"]["n"] == 3

    monkeypatch.setenv("MY_KEY", "secret")
    client = make_client("http://host/v1", "MY_KEY")
    assert str(client.base_url) == "http://host/v1/" and client.api_key == "secret"
