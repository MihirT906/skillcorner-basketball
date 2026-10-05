"""Run the agent over a question file, grade it, save the results and print a report.

    python run_eval.py --model <model> [--questions eval.jsonl] [--limit N] [--out baseline_eval.json]

Any OpenAI-compatible endpoint works: --base-url is the endpoint and --api-key-env names the
environment variable that holds its key.

    python run_eval.py --model <trained-model> --base-url https://<host>/v1 --api-key-env DOUBLEWORD_API_KEY
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from agent import default_model, run_agent
from grade import grade
from questions import MIX

HERE = Path(__file__).resolve().parent

# USD per 1M tokens: (input, cached input, output). OpenAI list prices; check them against
# openai.com/api/pricing, and add the model you run, before trusting the cost line.
PRICES_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "o3": (2.00, 0.50, 8.00),
    "o4-mini": (1.10, 0.275, 4.40),
}


def make_client(base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY") -> OpenAI:
    """Client for OpenAI, or for any OpenAI-compatible endpoint when base_url is given."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Set {api_key_env} to the API key for {base_url or 'OpenAI'}.")
    return OpenAI(base_url=base_url, api_key=api_key)


def cost_usd(model: str, usage: dict) -> float | None:
    if model not in PRICES_PER_MTOK:
        return None
    price_in, price_cached, price_out = PRICES_PER_MTOK[model]
    fresh = usage["input_tokens"] - usage["cached_input_tokens"]
    return (fresh * price_in + usage["cached_input_tokens"] * price_cached + usage["output_tokens"] * price_out) / 1e6


def run_one(row: dict, client, model: str) -> dict:
    try:
        result = run_agent(row["question"], client=client, model=model)
    except Exception as e:  # one failed request should not sink the whole run
        result = {"response": "", "tool_calls": [], "latency_s": None, "error": f"{type(e).__name__}: {e}",
                  "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}}
    graded = grade(result["response"], row["answer"], row["answer_type"])
    return {
        **row,
        "correct": graded["correct"],
        "model_answer": graded["model_answer"],
        "response": result["response"],
        "tool_calls": result["tool_calls"],
        "n_tool_calls": len(result["tool_calls"]),
        "latency_s": result["latency_s"],
        "usage": result["usage"],
        **({"error": result["error"]} if "error" in result else {}),
    }


def summarize(results: list[dict], model: str) -> dict:
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["correct"])
    usage = {k: sum(r["usage"][k] for r in results) for k in ("input_tokens", "cached_input_tokens", "output_tokens")}
    latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
    cost = cost_usd(model, usage)
    return {
        "model": model,
        "n": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": round(sum(r["correct"] for r in results) / len(results), 4),
        "by_question_type": {
            t: {"n": len(by_type[t]), "correct": sum(by_type[t]), "accuracy": round(sum(by_type[t]) / len(by_type[t]), 4)}
            for t in MIX if t in by_type
        },
        "avg_tool_calls": round(sum(r["n_tool_calls"] for r in results) / len(results), 2),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "usage": usage,
        "total_cost_usd": round(cost, 4) if cost is not None else None,
    }


def print_report(summary: dict, results: list[dict]) -> None:
    print(f"\nModel: {summary['model']}")
    print(f"{'question_type':<18}{'n':>4}{'correct':>9}{'accuracy':>10}")
    for qtype, s in summary["by_question_type"].items():
        print(f"{qtype:<18}{s['n']:>4}{s['correct']:>9}{s['accuracy']:>10.1%}")
    print(f"{'OVERALL':<18}{summary['n']:>4}{summary['correct']:>9}{summary['accuracy']:>10.1%}")
    print(f"\nAvg tool calls: {summary['avg_tool_calls']}   Avg latency: {summary['avg_latency_s']}s")
    cost = summary["total_cost_usd"]
    u = summary["usage"]
    print(f"Tokens: {u['input_tokens']:,} in ({u['cached_input_tokens']:,} cached), {u['output_tokens']:,} out   "
          f"Total cost: {'$%.4f' % cost if cost is not None else 'unknown (no price for this model in PRICES_PER_MTOK)'}")

    failures = [r for r in results if not r["correct"]]
    print(f"\nFailures: {len(failures)}")
    for r in failures:
        print(f"\n[{r['id']}] {r['question']}")
        print(f"  expected:     {r['answer']}")
        print(f"  model answer: {r['model_answer']}" + (f"   (error: {r['error']})" if "error" in r else ""))
        if not r["tool_calls"]:
            print("  tool calls:   none")
        for call in r["tool_calls"]:
            print(f"  tool call:    {call['name']}({call['arguments']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. https://<host>/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    parser.add_argument("--questions", default=str(HERE / "eval.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--out", default=str(HERE / "baseline_eval.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model = args.model or default_model()
    rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()][: args.limit]
    client = make_client(args.base_url, args.api_key_env)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda row: run_one(row, client, model), rows))
    summary = summarize(results, model)
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    print_report(summary, results)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
