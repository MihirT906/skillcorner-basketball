# Event runbook (Model Training Night)

Run everything from the project root with the venv active (`source .venv/bin/activate`).
Each step names what you should see before moving on.

Set these once, for whichever endpoint you are calling:

```bash
export API_KEY=...                  # the endpoint's key
BASE=https://<host>/v1              # omit --base-url entirely for OpenAI itself
MODEL=<model-name>
```

## What is where

| | Basic tier (one lookup per question) | Hard tier (filters, arithmetic, two calls) |
| --- | --- | --- |
| Training questions | `datasets/train.jsonl` (238) | `datasets/train_hard.jsonl` (240) |
| Eval questions (unseen games) | `datasets/eval.jsonl` (40) | `datasets/eval_hard.jsonl` (40) |
| Traces to upload | `traces/synthetic/traces_clean.jsonl` | `traces/synthetic/traces_hard_clean.jsonl` |

The two trace files are kept separate, one `{"messages", "tools"}` object per line, every trace
replayed through the tools and graded. Eval output goes in `results/`.

Frontier baseline, already run (`gpt-5.5`, 2026-10-05):

| Eval | Accuracy | Avg tool calls | Avg latency | Output |
| --- | --- | --- | --- | --- |
| Basic | 40/40 | 1.1 | 3.09 s | `results/baseline_gpt-5.5.json` |
| Hard | 40/40 | 1.4 | 3.54 s | `results/baseline_hard_gpt-5.5.json` |

## 0. Start training first

Upload `traces/synthetic/traces_clean.jsonl` to Overmind and kick off the post-training run.
`traces/synthetic/traces_hard_clean.jsonl` is the second file: add it to the same run if Overmind
takes more than one file, otherwise use it for a second run. Everything below can happen while
it trains.

## 1. Smoke test (1 minute)

```bash
pytest -q                                                    # 94 passed
python run_eval.py --model $MODEL --base-url $BASE --api-key-env API_KEY --limit 3 --out results/smoke.json
```

Expect a 3-question report with tool calls listed and no `error:` lines. If every question
fails with "tool calls: none", the endpoint is not doing tool calling: check the model name
and that the server has tool/function calling switched on.

## 2. Untuned small model (the "before" number)

Run the small model as handed out at the event, before any training, on both eval files:

```bash
python run_eval.py --model $MODEL --base-url $BASE --api-key-env API_KEY --out results/untuned_eval.json
python run_eval.py --model $MODEL --base-url $BASE --api-key-env API_KEY --questions datasets/eval_hard.jsonl --out results/untuned_hard_eval.json
```

40 questions each on the two unseen games (191313, 178442). Note accuracy per question type,
average latency and total cost. The cost line needs the model in `PRICES_PER_MTOK`
(`run_eval.py`); add it if it prints "unknown" (it does for `gpt-5.5`).

To redo the frontier baseline on another model, run the same two commands with that model and
`--out results/baseline_<model>.json` / `results/baseline_hard_<model>.json`.

## 3. Real traces from a frontier model (optional, improves the training set)

```bash
python gen_real_traces.py --model $MODEL --base-url $BASE --api-key-env API_KEY --limit 5   # try 5 first
python gen_real_traces.py --model $MODEL --base-url $BASE --api-key-env API_KEY             # all 238 basic
python gen_real_traces.py --model $MODEL --base-url $BASE --api-key-env API_KEY \
  --questions datasets/train_hard.jsonl --out traces/frontier/traces_hard_real.jsonl        # all 240 hard
```

Writes `traces/frontier/traces_real.jsonl` (and `traces_hard_real.jsonl`). Only traces with a
correct final answer, at least one tool call and no tool errors are kept; the rest are listed
as dropped with the reason. It refuses eval games.

## 4. Merge

```bash
python merge_traces.py          # -> traces/traces_merged.jsonl
python merge_traces.py --synthetic traces/synthetic/traces_hard_synthetic.jsonl \
  --real traces/frontier/traces_hard_real.jsonl --out traces/traces_hard_merged.jsonl
```

One trace per question: a real trace replaces the synthetic one for the same question
(`--keep-both` keeps the two). Upload the merged files to Overmind for a second run if
there is time; otherwise the run from step 0 stands.

## 5. Evaluate the trained model

```bash
python run_eval.py --model <trained-model> --base-url <doubleword-url>/v1 --api-key-env DOUBLEWORD_API_KEY --out results/trained_eval.json
python run_eval.py --model <trained-model> --base-url <doubleword-url>/v1 --api-key-env DOUBLEWORD_API_KEY \
  --questions datasets/eval_hard.jsonl --out results/trained_hard_eval.json
```

Compare three columns per eval file: untuned small model (step 2), trained model, and the
`gpt-5.5` baseline above. Accuracy per question type, latency, cost.

## 6. Live demo (game 191313, Breogan 94-95 Zaragoza)

```bash
OPENAI_BASE_URL=<doubleword-url>/v1 OPENAI_API_KEY=$DOUBLEWORD_API_KEY OPENAI_MODEL=<trained-model> \
  python agent.py "Game 191313: Leche Rio Breogan vs Casademont Zaragoza (2026-05-29). Which player scored the most points in the game?"
```

Demo questions with their answers are in `../README.md`.
