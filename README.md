# skillcorner-basketball

**A stats assistant for basketball broadcasters: ask about any ACB game, get the exact number.**

A tool-calling agent answers stat questions about Spanish Liga ACB 2025-26 games from
[SkillCorner's open data](https://github.com/SkillCorner/opendata-basketball). It never guesses:
it calls tools that query the game's events, then gives a short answer ending in
`FINAL ANSWER: <answer>`. The point of the project is that a small model post-trained on this one
narrow task can match a frontier model at a fraction of the cost and latency.

Six question types: `player_stat`, `team_stat`, `period_filtered`, `leader`, `comparison`, and
`tracking` (contested-shot FG% and pick-and-roll counts, which a normal box score does not have).

## How to run

```bash
git clone https://github.com/SkillCorner/opendata-basketball ../opendata-basketball   # data, as a sibling folder
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pytest -q
```

| Command | What it does |
| --- | --- |
| `python questions.py` | Writes `datasets/train.jsonl` and `datasets/eval.jsonl`, with answers computed from the data |
| `python synthetic_traces.py` | Writes `traces/synthetic/traces_synthetic.jsonl` and `traces/synthetic/traces_clean.jsonl` (ideal-agent traces, each one replayed and graded) |
| `python run_eval.py --model M [--base-url URL --api-key-env VAR] [--questions F] [--limit N]` | Runs the agent on the eval questions; reports accuracy per question type, latency and cost |
| `python gen_real_traces.py --model M [--base-url URL --api-key-env VAR]` | Writes `traces/frontier/traces_real.jsonl` from a real model's correct runs on the training questions |
| `python merge_traces.py` | Merges synthetic and real traces into `traces/traces_merged.jsonl` |
| `python agent.py "<question>"` | Asks one question (uses `OPENAI_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`) |

`run_eval.py` and `gen_real_traces.py` work with any OpenAI-compatible endpoint. The steps for the
event are in [EVENT_RUNBOOK.md](docs/EVENT_RUNBOOK.md); data quirks are in [SCHEMA_NOTES.md](docs/SCHEMA_NOTES.md).

## Game split

The model is trained on some games and evaluated on others, to show it learned the skill rather
than memorized a game.

| Split | Games | Questions |
| --- | --- | --- |
| Training | 114243, 114234, 114169, 114099, 114086, 179612, 184439, 188630 | 238 (`datasets/train.jsonl`), one validated trace each |
| Eval (never trained on) | 191313 Breogan 94-95 Zaragoza, 178442 Andorra 98-102 Bilbao (OT) | 40 (`datasets/eval.jsonl`) |

Ground truth always comes from code (`data.py`), never from an LLM. Scores are computed from shots
and free throws only.

## Answer format

The system prompt (identical in the training traces and at eval time) asks for:

- Counts: a single integer, e.g. `12`
- Percentages: one decimal with %, e.g. `51.6%`
- Players and teams: the full name as returned by the tools
- Score after a quarter: `<home team> <points>-<points> <away team>`

## Demo questions

All from the eval games. Prefix each with the game line, e.g.
`Game 191313: Leche Rio Breogan vs Casademont Zaragoza (2026-05-29).`

Game 191313, Leche Rio Breogan 94-95 Casademont Zaragoza:

| Question | Answer |
| --- | --- |
| Which player scored the most points in the game? | Francisco Alonso |
| How many points did Jaime Fernandez Manzanares score in the 1st quarter? | 3 |
| Which team was ahead at the end of Q1? | Leche Rio Breogan |
| What percentage of its field goal attempts did Casademont Zaragoza make? | 51.6% |
| What was Casademont Zaragoza's field goal percentage on contested shots? | 51.1% |
| What percentage of his contested field goal attempts did Justin Wright-Foreman make? | 44.4% |
| Which team recorded more offensive rebounds? | Leche Rio Breogan |

Game 178442, BC MoraBanc Andorra 98-102 Bilbao Basket (overtime):

| Question | Answer |
| --- | --- |
| Who was the game's leading scorer? | Justin Jaworski |
| What was BC MoraBanc Andorra's point total in the overtime period alone? | 11 |
| Which team was leading after Q3? | Bilbao Basket |
| What was Artem Pustovyi's field goal percentage on contested shots? | 66.7% |
