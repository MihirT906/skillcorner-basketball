"""Generate training traces by running a real model on the training questions.

The model answers each question in train.jsonl through the same agent loop as the eval. A trace
is kept only if its final answer grades correct against the ground truth, it made at least one
tool call, and no tool call returned an error. Eval games are refused outright.

    python gen_real_traces.py --model <model> [--base-url URL] [--api-key-env VAR] [--limit N]
        ->  traces_real.jsonl
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent import OPENAI_TOOLS, default_model, run_agent
from grade import grade
from questions import EVAL_GAMES
from run_eval import make_client

HERE = Path(__file__).resolve().parent


def real_trace(row: dict, client, model: str) -> tuple[dict | None, str | None]:
    """(trace, None) if the model's run is worth training on, otherwise (None, reason dropped)."""
    try:
        result = run_agent(row["question"], client=client, model=model)
    except Exception as e:  # one failed request should not sink the whole run
        return None, f"{type(e).__name__}: {e}"
    if not result["tool_calls"]:
        return None, "answered without calling a tool"
    if any("error" in call["result"] for call in result["tool_calls"]):
        return None, "a tool call returned an error"
    graded = grade(result["response"], row["answer"], row["answer_type"])
    if not graded["correct"]:
        return None, f"FINAL ANSWER {graded['model_answer']!r} does not match ground truth {row['answer']!r}"
    return {"id": row["id"], "question_type": row["question_type"], "model": model,
            "messages": result["messages"], "tools": OPENAI_TOOLS}, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. https://<host>/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    parser.add_argument("--questions", default=str(HERE / "train.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--out", default=str(HERE / "traces_real.jsonl"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model = args.model or default_model()
    rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()][: args.limit]
    leaked = sorted({row["game_id"] for row in rows} & set(EVAL_GAMES))
    if leaked:
        raise SystemExit(f"Refusing to generate training traces from eval games {leaked}.")

    client = make_client(args.base_url, args.api_key_env)
    with ThreadPoolExecutor(args.workers) as pool:
        outcomes = list(pool.map(lambda row: real_trace(row, client, model), rows))
    traces = [trace for trace, _ in outcomes if trace]
    Path(args.out).write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces))
    print(f"Wrote {len(traces)} of {len(rows)} traces to {args.out}; dropped {len(rows) - len(traces)}")
    print("Traces per question_type:", dict(Counter(t["question_type"] for t in traces)))
    for row, (trace, reason) in zip(rows, outcomes):
        if not trace:
            print(f"  dropped {row['id']}: {reason}")


if __name__ == "__main__":
    main()
