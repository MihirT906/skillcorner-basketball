# Event runbook (Model Training Night)

Run everything from this folder with the venv active (`source .venv/bin/activate`).
Each step names what you should see before moving on.

Set these once, for whichever endpoint you are calling:

```bash
export API_KEY=...                  # the endpoint's key
BASE=https://<host>/v1              # omit --base-url entirely for OpenAI itself
MODEL=<model-name>
```

## 0. Start training first

Upload `traces_clean.jsonl` (238 validated traces, `{"messages", "tools"}` per line) to Overmind
and kick off the post-training run. Everything below can happen while it trains.

## 1. Smoke test (1 minute)

```bash
pytest -q                                                    # 90 passed
python run_eval.py --model $MODEL --base-url $BASE --api-key-env API_KEY --limit 3 --out smoke.json
```

Expect a 3-question report with tool calls listed and no `error:` lines. If every question
fails with "tool calls: none", the endpoint is not doing tool calling: check the model name
and that the server has tool/function calling switched on.

## 2. Baseline eval (frontier model, if one is available)

```bash
python run_eval.py --model $MODEL --base-url $BASE --api-key-env API_KEY --out baseline_eval.json
```

40 questions on the two unseen games (191313, 178442). Note accuracy per question type,
average latency and total cost. The cost line needs the model in `PRICES_PER_MTOK`
(`run_eval.py`); add it if it prints "unknown".

## 3. Real traces from the frontier model (optional, improves the training set)

```bash
python gen_real_traces.py --model $MODEL --base-url $BASE --api-key-env API_KEY --limit 5   # try 5 first
python gen_real_traces.py --model $MODEL --base-url $BASE --api-key-env API_KEY             # all 238
```

Writes `traces_real.jsonl`. Only traces with a correct final answer, at least one tool call and
no tool errors are kept; the rest are listed as dropped with the reason. It refuses eval games.

## 4. Merge

```bash
python merge_traces.py          # -> traces_merged.jsonl
```

One trace per question: a real trace replaces the synthetic one for the same question
(`--keep-both` keeps the two). Upload `traces_merged.jsonl` to Overmind for a second run if
there is time; otherwise the run from step 0 stands.

## 5. Evaluate the trained model

```bash
python run_eval.py --model <trained-model> --base-url <doubleword-url>/v1 --api-key-env DOUBLEWORD_API_KEY --out trained_eval.json
```

Compare with `baseline_eval.json`: accuracy per question type, latency, cost.

## 6. Live demo (game 191313, Breogan 94-95 Zaragoza)

```bash
OPENAI_BASE_URL=<doubleword-url>/v1 OPENAI_API_KEY=$DOUBLEWORD_API_KEY OPENAI_MODEL=<trained-model> \
  python agent.py "Game 191313: Leche Rio Breogan vs Casademont Zaragoza (2026-05-29). Which player scored the most points in the game?"
```

Demo questions with their answers are in `README.md`.
