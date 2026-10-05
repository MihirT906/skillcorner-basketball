"""Tool-calling stats agent on the OpenAI Chat Completions API, using the tools in tools.py."""

import json
import os
import time

from openai import OpenAI

from tools import TOOL_SCHEMAS, TOOLS

SYSTEM_PROMPT = (
    "You are a stats assistant for basketball broadcasters covering the Spanish Liga ACB. "
    "Answer using tools only, never guess. Keep the answer short. "
    "End with a final line: FINAL ANSWER: <answer>\n"
    "\n"
    "Format of <answer>:\n"
    "- Counts: a single integer, e.g. 12\n"
    "- Percentages: one decimal with %, e.g. 51.6%\n"
    "- Players and teams: the full name as returned by the tools\n"
    "- Score after a quarter: <home team> <points>-<points> <away team>"
)
MAX_TOOL_CALLS = 5

# TOOL_SCHEMAS is written in Anthropic's shape (name / description / input_schema).
OPENAI_TOOLS = [
    {"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": s["input_schema"]}}
    for s in TOOL_SCHEMAS
]


def default_model() -> str:
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise RuntimeError("Set OPENAI_MODEL (or pass model=...) to the OpenAI model the agent should use.")
    return model


def call_tool(name: str, arguments: str) -> dict:
    """Run one tool call. Bad tool names or arguments come back as {'error': ...} for the model to read."""
    if name not in TOOLS:
        return {"error": f"Unknown tool {name!r}. Available: {list(TOOLS)}"}
    try:
        return TOOLS[name](**json.loads(arguments or "{}"))
    except (TypeError, json.JSONDecodeError) as e:
        return {"error": f"Bad arguments for {name}: {e}"}


def run_agent(question: str, client=None, model: str | None = None, system_prompt: str = SYSTEM_PROMPT,
              max_tool_calls: int = MAX_TOOL_CALLS) -> dict:
    """Answer one question. Returns the final text plus the full conversation, tool calls, latency and token usage."""
    client = client or OpenAI()
    model = model or default_model()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    tool_calls = []
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    start = time.perf_counter()

    while True:
        # Once the budget is spent the model must answer with what it has.
        out_of_budget = len(tool_calls) >= max_tool_calls
        response = client.chat.completions.create(
            model=model, messages=messages, tools=OPENAI_TOOLS, tool_choice="none" if out_of_budget else "auto"
        )
        if response.usage:
            details = getattr(response.usage, "prompt_tokens_details", None)
            usage["input_tokens"] += response.usage.prompt_tokens
            usage["cached_input_tokens"] += getattr(details, "cached_tokens", 0) or 0
            usage["output_tokens"] += response.usage.completion_tokens
        message = response.choices[0].message
        if not message.tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            if len(tool_calls) >= max_tool_calls:
                result = {"error": f"Tool call limit ({max_tool_calls}) reached. Answer with the information you have."}
            else:
                result = call_tool(call.function.name, call.function.arguments)
                tool_calls.append({"name": call.function.name, "arguments": call.function.arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})

    messages.append({"role": "assistant", "content": message.content or ""})
    return {
        "response": message.content or "",
        "messages": messages,
        "tool_calls": tool_calls,
        "latency_s": round(time.perf_counter() - start, 2),
        "usage": usage,
        "model": model,
    }


if __name__ == "__main__":
    import sys

    result = run_agent(" ".join(sys.argv[1:]))
    for call in result["tool_calls"]:
        print(f"[tool] {call['name']}({call['arguments']})")
    print(result["response"])
