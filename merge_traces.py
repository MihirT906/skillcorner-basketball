"""Merge synthetic and real traces into one training file with just {"messages", "tools"}.

One trace per question: a real trace replaces the synthetic trace for the same question id
(pass --keep-both to keep the two). Traces from eval games, or with a system prompt other than
the current one, stop the merge.

    python merge_traces.py [--synthetic traces_synthetic.jsonl] [--real traces_real.jsonl]
        ->  traces_merged.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from agent import SYSTEM_PROMPT
from questions import EVAL_GAMES

HERE = Path(__file__).resolve().parent


def read_traces(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def merge(synthetic: list[dict], real: list[dict], keep_both: bool = False) -> list[dict]:
    for trace in synthetic + real:
        if int(trace["id"].split("-")[0]) in EVAL_GAMES:
            raise ValueError(f"Trace {trace['id']} comes from an eval game")
        if trace["messages"][0] != {"role": "system", "content": SYSTEM_PROMPT}:
            raise ValueError(f"Trace {trace['id']} was made with a different system prompt")
    real_ids = {t["id"] for t in real}
    return [t for t in synthetic if keep_both or t["id"] not in real_ids] + real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=str(HERE / "traces_synthetic.jsonl"))
    parser.add_argument("--real", default=str(HERE / "traces_real.jsonl"))
    parser.add_argument("--out", default=str(HERE / "traces_merged.jsonl"))
    parser.add_argument("--keep-both", action="store_true", help="keep the synthetic trace even when a real one exists")
    args = parser.parse_args()

    synthetic, real = read_traces(Path(args.synthetic)), read_traces(Path(args.real))
    try:
        merged = merge(synthetic, real, args.keep_both)
    except ValueError as e:
        raise SystemExit(str(e))
    Path(args.out).write_text(
        "".join(json.dumps({"messages": t["messages"], "tools": t["tools"]}, ensure_ascii=False) + "\n" for t in merged)
    )
    print(f"Wrote {len(merged)} traces to {args.out} ({len(real)} real, {len(merged) - len(real)} synthetic)")
    print("Traces per question_type:", dict(Counter(t["question_type"] for t in merged)))


if __name__ == "__main__":
    main()
