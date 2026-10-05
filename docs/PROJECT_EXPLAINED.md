# The project, explained in plain terms

This is a walkthrough of what has been built, how each piece works, what results exist so far,
and what the trained model has to beat tonight.

## The idea in one paragraph

A broadcaster asks a question about a Spanish Liga ACB basketball game ("How many points did
Spissu score in the second quarter?") and gets the exact number. The assistant never answers from
memory: it calls a tool that looks the number up in the game data, then replies. A big frontier
model (gpt-5.5) already does this perfectly. The goal tonight is to show that a small model,
after being trained on examples of this one narrow task, can do it just as well while being
cheaper and faster.

## The pipeline at a glance

| Step | What happens | Script | Output |
| --- | --- | --- | --- |
| 1 | Load the game data and compute stats from it | `data.py`, `tools.py` | Five tools the assistant can call |
| 2 | Generate basic questions with correct answers | `questions.py` | 238 training + 40 eval questions |
| 3 | Generate hard questions with correct answers | `questions_hard.py` | 240 training + 40 eval questions |
| 4 | Generate training examples ("traces") | `synthetic_traces.py`, `synthetic_traces_hard.py` | 478 traces |
| 5 | Run a model on the eval questions and grade it | `run_eval.py`, `grade.py` | A results file per model |

No language model wrote any question, any answer or any training example. All of it comes from
Python code reading the game data.

## Step 1: the data

The data is SkillCorner's open basketball data, downloaded into a folder next to this one
(`../opendata-basketball`). It covers 10 games.

- **8 games are for training:** 114243, 114234, 114169, 114099, 114086, 179612, 184439, 188630.
- **2 games are held back for testing:** 191313 (Breogan 94-95 Zaragoza) and 178442 (Andorra
  98-102 Bilbao, overtime). The model never sees these during training. That is how we show it
  learned the skill and did not just memorise games.

For each game, `data.py` reads the list of events: every shot, free throw, rebound, turnover,
foul and pick-and-roll. From those events it computes box scores and scores by quarter. Points are
counted only from made shots and made free throws, because the data's own "points scored" field
is unreliable.

`tools.py` exposes this as five tools the assistant can call:

| Tool | What it returns |
| --- | --- |
| `get_score_by_period` | Each team's points per quarter |
| `get_box_score` | Player stats, for the whole game or one quarter |
| `get_team_stats` | Team totals, for the whole game or one quarter |
| `get_shots` | Makes, attempts and percentage, filterable by contested or uncontested, two or three points |
| `get_picks` | Number of pick-and-rolls a team ran |

## Step 2: the basic questions

`questions.py` works like a fill-in-the-blank generator. It has 27 question templates, each with
three wordings, such as "How many points did {player} score?".

For each game it:

1. Lists every question that could be asked, and computes the correct answer from the data.
2. Picks 30 at random for a training game, or 20 for a test game, in a fixed mix of six question
   types.
3. Skips questions with no clear answer, such as "who was the leading scorer?" when two players
   are tied.

The six question types are: player stat, team stat, period-filtered (one quarter), leader,
comparison, and tracking (contested shots and pick-and-rolls, which a normal box score does not
have).

Result: 238 training questions and 40 test questions.

### Example basic questions

One of each type, all from training game 114243 (BAXI Manresa vs Coviran Granada). Every question
starts with a line naming the game, so the model knows which game to look up.

| Type | Question | Correct answer |
| --- | --- | --- |
| Player stat | Beka Burjanadze finished with how many points? | 9 |
| Team stat | Coviran Granada finished with how many rebounds? | 54 |
| Period-filtered | How many points did BAXI Manresa have after the first half? | 41 |
| Leader | Which player scored the most points in the game? | Alfonso Plummer |
| Comparison | Between BAXI Manresa and Coviran Granada, which team recorded more turnovers? | Coviran Granada |
| Tracking | What was Alfonso Plummer's field goal percentage on contested shots? | 42.9 |

This is what one question looks like in `datasets/train.jsonl`:

```json
{
  "id": "114243-player_stat-01",
  "game_id": 114243,
  "question": "Game 114243: BAXI Manresa vs Coviran Granada (2025-10-11). Beka Burjanadze finished with how many points?",
  "answer": 9,
  "answer_type": "integer",
  "question_type": "player_stat"
}
```

`answer_type` tells the grader how to compare a reply with the answer (see Step 5).

## Step 3: the hard questions

Claude Code did not write the hard questions one by one. It wrote a second generator script,
`questions_hard.py`, which works the same way with 26 harder templates. These need two lookups,
some arithmetic, or several filters at once. Examples:

- "How many points did A and B score combined?"
- "Who was the team's leading scorer in the second half?"
- "What was the team's field goal percentage on contested two-point shots?"

Result: 240 training questions and 40 test questions.

### Example hard questions

One of each type, from the same game, with what makes each one harder.

| Type | Question | Correct answer | Why it is harder |
| --- | --- | --- | --- |
| Player stat | What was the combined point total of Zach Hankins and Eduardo Duran? | 10 | Two lookups, then add |
| Team stat | BAXI Manresa and Coviran Granada combined for how many turnovers? | 25 | Two lookups, then add |
| Period-filtered | What was Matt Thomas's point total in the second half alone? | 9 | Look up Q3 and Q4, then add |
| Leader | Who was Coviran Granada's leading scorer in the first quarter? | Matt Thomas | Team and quarter filters together |
| Comparison | Between BAXI Manresa and Coviran Granada, which team was more accurate on contested field goal attempts? | BAXI Manresa | Two lookups, then compare |
| Tracking | What percentage of its contested two-point attempts did BAXI Manresa make? | 39.3 | Contested and two-point filters together |

## Step 4: the training traces

A trace is one complete example conversation that the small model learns from:

1. The instructions (system prompt).
2. The question.
3. The assistant deciding which tool to call.
4. What the tool returned.
5. The assistant's short answer, ending in `FINAL ANSWER: <answer>`.

The usual way to get traces is to run a frontier model on the training questions and record what
it does. That was not possible when this was built (no API key), so a script plays the part of a
perfect assistant instead:

- **It already knows the right move.** Each question was built from a template, and the script
  has a plan for every template: which tool to call and how to read the answer from the result.
  For "combined points of A and B", the plan is: get A's box score, get B's box score, add the
  two point totals.
- **The tool output is real.** The script actually runs the tool, so the numbers in the trace are
  the true numbers from the game.
- **It varies the wording.** The assistant's sentences are picked from a handful of phrasings, and
  about a third of the time it uses a short name ("Rousselle") in place of the full name, so the
  model learns that both work.
- **Every trace is checked.** The tool calls are run again and must give the same result, and the
  final answer must match the correct answer. Anything that fails is thrown away. Nothing failed.

Result: 478 traces, one per training question, all from the 8 training games.

### Example trace: a basic question

This is trace `114243-tracking-01`. The system prompt comes first in the real file and is left out
here; the long explanatory notes inside the tool output are cut short with `...`.

**User**

> Game 114243: BAXI Manresa vs Coviran Granada (2025-10-11). What was Alfonso Plummer's field goal
> percentage on contested shots?

**Assistant** (decides to call a tool)

> Checking the shot data for game 114243.

```json
{"name": "get_shots", "arguments": {"game_id": "114243", "player": "Plummer", "contested": true}}
```

The script used the short name "Plummer" here. The tool still found the right player.

**Tool result** (real output)

```json
{"game_id": 114243, "filters": {"player": "Alfonso Plummer", "contested": true},
 "attempts": 7, "makes": 3, "fg_pct": 42.9, "points": 9, "note": "...", "contested_rule": "..."}
```

**Assistant** (final answer)

> Alfonso Plummer made 3 of 7 contested field goal attempts, which is 42.9%.
> FINAL ANSWER: 42.9%

### Example trace: a hard question

This is trace `114243-hard-period_filtered-05`. The tool gives points per quarter; the answer needs
a subtraction that the tool does not do.

**User**

> Game 114243: BAXI Manresa vs Coviran Granada (2025-10-11). By how many points was the leading
> team ahead after Q1?

**Assistant** (decides to call a tool)

> I'll get the period-by-period score to answer that.

```json
{"name": "get_score_by_period", "arguments": {"game_id": "114243"}}
```

**Tool result** (real output)

```json
{"game_id": 114243, "date": "2025-10-11", "periods": ["Q1", "Q2", "Q3", "Q4"], "overtime": false,
 "home": {"team": "BAXI Manresa", "final": 83, "by_period": {"Q1": 20, "Q2": 21, "Q3": 28, "Q4": 14}},
 "away": {"team": "Coviran Granada", "final": 68, "by_period": {"Q1": 11, "Q2": 13, "Q3": 24, "Q4": 20}}}
```

**Assistant** (final answer)

> Based on the period-by-period score, BAXI Manresa led Coviran Granada 20-11 after Q1, a margin of
> 9 points.
> FINAL ANSWER: 9

### The trace files

| File in `traces/synthetic/` | Traces | Contents |
| --- | --- | --- |
| `traces_clean.jsonl` | 238 | Basic questions, ready to upload for training |
| `traces_hard_clean.jsonl` | 240 | Hard questions, ready to upload for training |
| `traces_synthetic.jsonl`, `traces_hard_synthetic.jsonl` | 238, 240 | The same traces with the question id and type attached |

The basic and hard files are separate. To train on all 478, they need to be joined into one file.

### Traces from a real frontier model (not done)

`gen_real_traces.py` can do the usual version: run a real model on the training questions and keep
only the runs it got right. `merge_traces.py` then swaps those in for the scripted ones. Neither
has been run, so `traces/frontier/` is empty. It would cost roughly 500-600 API requests for the
basic set, or 1,100-1,300 for basic plus hard.

## Step 5: testing a model

There are no ready-made traces for the two test games. Only the questions and their correct
answers are stored. The conversation is produced live by whichever model is being tested.

For each test question, `run_eval.py`:

1. Sends the instructions and the question to the model.
2. Lets the model choose a tool, runs it for real, and passes the result back.
3. Takes the model's reply and grades it.
4. Saves the question, the tool calls, the reply, whether it was right, how long it took and how
   many tokens it used.

The instructions given to the model are word-for-word the same as in the training traces.

### Example: one test question answered by gpt-5.5

This is the record for `191313-hard-comparison-01` in `results/baseline_hard_gpt-5.5.json`. Nothing
here was scripted: gpt-5.5 chose the tool calls itself.

| | |
| --- | --- |
| Question | Game 191313: Leche Rio Breogan vs Casademont Zaragoza (2026-05-29). Which team had more turnovers in the 3rd quarter? |
| Correct answer | Leche Rio Breogan |
| Tool call 1 | `get_team_stats(game_id="191313", team="Leche Rio Breogan", period=3)` |
| Tool call 2 | `get_team_stats(game_id="191313", team="Casademont Zaragoza", period=3)` |
| Model's reply | Leche Rio Breogan had more 3rd-quarter turnovers. FINAL ANSWER: Leche Rio Breogan |
| Graded | Correct |
| Time | 3.77 s |
| Tokens | 3,430 in, 129 out |

Tonight's run produces the same kind of record for the trained small model, so the two can be
compared question by question.

### How a reply is graded

Grading is done by code in `grade.py`, with no language model judging. It reads the text after
`FINAL ANSWER:` and compares it to the stored correct answer. If that line is missing, the reply
is wrong.

| Kind of answer | Rule |
| --- | --- |
| A count | Must be exactly the right number ("20 points" and "twelve" are understood) |
| A percentage | Must be within 0.5 of the right value |
| A player or team name | Must match, ignoring capitals and accents; small typos and partial names pass |
| A score | Both teams with the right points, in either order |

Only the final answer is graded. The grader does not check which tools the model called or what it
wrote before the final line.

## Results so far

Only the frontier model has been tested. The small model has no results yet, because it has not
been trained.

| Test set | Model | Correct | Tool calls per question | Time per question | Input tokens | Output tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Basic, 40 questions | gpt-5.5 | 40/40 | 1.1 | 3.09 s | 138,322 | 2,845 |
| Hard, 40 questions | gpt-5.5 | 40/40 | 1.4 | 3.54 s | 141,215 | 4,106 |

gpt-5.5 was right on every question in all six question types. The full records are in
`results/baseline_gpt-5.5.json` and `results/baseline_hard_gpt-5.5.json`.

## What the trained model has to beat

gpt-5.5 got 80 out of 80, so it cannot be beaten on accuracy. The best possible result is a tie.
The case for the small model is therefore: **the same accuracy, for less money and less time.**

| Measure | gpt-5.5 | Target for the trained small model |
| --- | --- | --- |
| Accuracy, basic | 40/40 | 40/40, or very close |
| Accuracy, hard | 40/40 | 40/40, or very close |
| Time per question | 3.1-3.5 s | Lower |
| Cost per question | About 3,500 input tokens at gpt-5.5 prices | Lower |

## What is still missing

- **The trained model's results.** This is tonight's run: train on the traces, then run
  `run_eval.py` on both test sets.
- **The untrained small model's results.** Running the same small model before training would show
  that the training is what made the difference.
- **A dollar figure.** The results files record tokens but the cost field is empty, so the cost
  comparison still has to be worked out from each model's prices.
- **One combined training file**, if the plan is to train on basic and hard together.

## Known limits of the scripted traces

- The assistant's wording comes from a small set of fixed phrasings.
- Every trace goes right first time. The model never sees a tool error or a retry, so it has no
  example of how to recover from one.
