# Project brief: ACB Broadcaster Stats Assistant

## The hackathon
Tonight (Mon Oct 5, 5–8 PM, SF) is "Model Training Night," part of SF Tech Week,
hosted by Overmind, Daytona and Doubleword. Every participant ships a
post-trained model in one evening:
- Daytona: a sandbox where the agent runs
- Overmind: post-trains a small open-weight model on our agent traces and
  benchmarks it
- Doubleword: serves (hosts) the trained model
Judging: "No pitch decks. Best deployment wins." Prizes: $1,000 top prize;
hardware for everyone who kicks off a training run. Training must start as
early as possible after 5 PM, so everything needed for training must be
ready before we leave.

## Why this project
Frontier models are overkill for narrow, repetitive tasks. A small model
post-trained on one task can match them at a fraction of the cost and
latency. We prove that on a sports use case.

## The product
"A stats assistant for basketball broadcasters: ask about any ACB game, get
the exact number." A tool-calling agent answers stat questions about Spanish
Liga ACB 2025-26 games using SkillCorner's open data
(github.com/SkillCorner/opendata-basketball). It never guesses; it calls
tools that query the game's Dynamic Events, then gives a short answer ending
in "FINAL ANSWER: <answer>".

Scope is deliberately narrow (small models do well on narrow tasks). Six
question types only: player_stat, team_stat, period_filtered, leader,
comparison, tracking (contested-shot FG%, pick-and-roll counts: the data a
normal box score doesn't have, our differentiator). Out of scope: opinions,
narratives, raw tracking (speed/distance), pass receivers (noisy field).

## Key design decisions (keep these intact)
- Multi-game split to prove the model learned the SKILL, not memorized one
  game. Training games: 114243, 114234, 114169, 114099, 114086, 179612,
  184439, 188630. Eval games (never used for training): 191313 (Breogan
  94-95 Zaragoza, our demo game) and 178442 (Andorra 98-102 Bilbao).
  Never generate training traces from eval games.
- Ground truth is always computed by code from data.py, never by an LLM.
- Scores come from shots + free_throws only (per SkillCorner KNOWN_ISSUES).
- Synthetic traces: we have no LLM API key, so training traces are generated
  by script (an "ideal agent"), validated by replaying tool calls and
  grading the final answer. Only fully correct traces are kept.
- The system prompt and answer-format rules must be identical in training
  traces and at eval time.
- Answers must be auto-gradable: integer, percentage with one decimal and %,
  full player/team name as returned by tools, or a score string.

## What tonight looks like
1. Hand traces_clean.jsonl to Overmind and start post-training a small
   open model.
2. Evaluate the trained model with run_eval.py + grade.py on eval.jsonl
   (the two unseen games): accuracy per question type, latency, cost.
3. If a frontier model is available at the event, run the same eval on it
   as the baseline ("matches frontier accuracy at a fraction of the cost").
4. Deploy via Doubleword; live demo with questions about game 191313.

## Definition of done before leaving (~3:15 PM)
- All tests pass.
- traces_clean.jsonl exists with ~250 validated training traces
  (85 minimum if time is short).
- eval.jsonl unchanged except for fixes that make questions answerable.
- run_eval.py works with any OpenAI-compatible endpoint via --model and a
  base URL, so it can point at the trained model tonight.
- README.md: one-line pitch, how to run, game split, demo questions.
- Everything committed to git.

## Working rules
- Prefer small, verifiable changes; run tests after each.
- If a question can't be answered exactly by the tools, fix the tool or
  drop the question. Never keep traces that are right by luck.
- Stop and report at the checkpoints the user asks for.