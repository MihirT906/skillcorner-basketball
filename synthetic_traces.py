"""Generate ideal-agent training traces for train.jsonl without calling any LLM.

Each trace is the conversation a perfect agent would have: system prompt, question, a tool
call with the right arguments, the real tool output, and a short answer ending in FINAL ANSWER.
Every trace is validated (tool calls replayed, answer graded) before it is written.

    python synthetic_traces.py   ->  traces/synthetic/traces_synthetic.jsonl
"""

import json
import random
from collections import Counter
from pathlib import Path

from agent import OPENAI_TOOLS, SYSTEM_PROMPT
from data import resolve_player, resolve_team
from grade import grade
from questions import SEED, generate_train
from tools import TOOLS

HERE = Path(__file__).resolve().parent
PARTIAL_NAME_RATE = 0.3  # how often a tool argument uses a partial name instead of the full one


def _plural(n, noun):
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _period_label(period_num):
    return f"Q{period_num}" if period_num <= 4 else f"OT{period_num - 4}"


def _players(result):
    return [p for team in result["teams"] for p in team["players"]]


def _top(result, stat):
    return max(_players(result), key=lambda p: p[stat])


def _points_through(side, period_num):
    # by_period is in playing order: Q1..Q4, then OT1..
    return sum(list(side["by_period"].values())[:period_num])


def _side(result, team):
    return result["home"] if result["home"]["team"] == team else result["away"]


# ---------------------------------------------------------------------------------------
# The plan: for every question template, which tool(s) to call with which arguments, and
# how to read the answer out of the tool result(s).
#
#   calls(m)        -> list of (tool_name, arguments) the ideal agent makes; `game_id` is added
#                      to every call. m is the question's meta: the entities it asks about.
#   answer(m, r)    -> (final answer, fact sentence) from the tool results r (one per call).
#
# question_type    template                  tool call(s)
# ---------------  ------------------------  ---------------------------------------------
# player_stat      player_points/threes/     get_box_score(player)
#                  rebounds/turnovers
# team_stat        team_turnovers/rebounds/  get_team_stats(team)
#                  fg_pct/3p_pct/fouls
# period_filtered  team_points_after,        get_score_by_period()
#                  leader_after,
#                  team_points_in,
#                  score_after
#                  player_points_in          get_box_score(player, period)
# leader           top_*_team                get_box_score(team)
#                  top_*_game                get_box_score()
# comparison       team_more                 get_team_stats(team a) + get_team_stats(team b)
#                  player_more_points        get_box_score(player a) + get_box_score(player b)
# tracking         contested_fg_pct_player   get_shots(player, contested=True)
#                  contested_fg_pct_team     get_shots(team, contested=True)
#                  team_picks                get_picks(team)
# ---------------------------------------------------------------------------------------

def _player_stat(stat, fact):
    return {
        "calls": lambda m: [("get_box_score", {"player": m["player"]})],
        "answer": lambda m, r: (_players(r[0])[0][stat], fact(m["player"], _players(r[0])[0][stat])),
    }


def _team_stat(stat, fact):
    return {
        "calls": lambda m: [("get_team_stats", {"team": m["team"]})],
        "answer": lambda m, r: (r[0][stat], fact(m["team"], r[0])),
    }


def _leader(stat, noun, team_scope):
    def answer(m, r):
        top = _top(r[0], stat)
        whom = m["team"] if team_scope else "all players"
        return top["player"], f"{top['player']} led {whom} with {_plural(top[stat], noun)}"

    return {
        "calls": lambda m: [("get_box_score", {"team": m["team"]} if team_scope else {})],
        "answer": answer,
    }


TEAM_MORE_STATS = {  # question wording -> get_team_stats field
    "rebounds": "REB",
    "turnovers": "TOV",
    "personal fouls": "PF",
    "made three-pointers": "3PM",
    "made free throws": "FTM",
    "offensive rebounds": "OREB",
}


def _team_more(m, r):
    stat = TEAM_MORE_STATS[m["stat"]]
    winner, loser = sorted(r, key=lambda t: -t[stat])
    return winner["team"], f"{winner['team']} had {winner[stat]} {m['stat']} to {loser['team']}'s {loser[stat]}"


def _player_more_points(m, r):
    winner, loser = sorted((_players(x)[0] for x in r), key=lambda p: -p["PTS"])
    return winner["player"], f"{winner['player']} scored {_plural(winner['PTS'], 'point')} to {loser['player']}'s {loser['PTS']}"


def _leader_after(m, r):
    home, away = (_points_through(r[0][side], m["period_num"]) for side in ("home", "away"))
    lead, trail = (r[0]["home"], r[0]["away"]) if home > away else (r[0]["away"], r[0]["home"])
    return lead["team"], f"{lead['team']} led {trail['team']} {max(home, away)}-{min(home, away)} after {m['period']}"


def _score_after(m, r):
    """(answer in the required '<home> <pts>-<pts> <away>' format, the same score in words)."""
    home, away = r[0]["home"], r[0]["away"]
    h, a = _points_through(home, m["period_num"]), _points_through(away, m["period_num"])
    return f"{home['team']} {h}-{a} {away['team']}", f"{home['team']} {h}, {away['team']} {a}"


def _contested(who):
    return {
        "calls": lambda m: [("get_shots", {who: m[who], "contested": True})],
        "answer": lambda m, r: (
            r[0]["fg_pct"],
            f"{m[who]} made {r[0]['makes']} of {r[0]['attempts']} contested field goal attempts, which is {r[0]['fg_pct']}%",
        ),
    }


PLAN = {
    # player_stat
    "player_points": _player_stat("PTS", lambda p, v: f"{p} scored {_plural(v, 'point')}"),
    "player_threes": _player_stat("3PM", lambda p, v: f"{p} made {_plural(v, 'three-pointer')}"),
    "player_rebounds": _player_stat("REB", lambda p, v: f"{p} grabbed {_plural(v, 'rebound')}"),
    "player_turnovers": _player_stat("TOV", lambda p, v: f"{p} committed {_plural(v, 'turnover')}"),
    # team_stat
    "team_turnovers": _team_stat("TOV", lambda t, s: f"{t} committed {_plural(s['TOV'], 'turnover')}"),
    "team_rebounds": _team_stat("REB", lambda t, s: f"{t} grabbed {_plural(s['REB'], 'rebound')}"),
    "team_fg_pct": _team_stat("FG%", lambda t, s: f"{t} shot {s['FG%']}% from the field ({s['FGM']} of {s['FGA']})"),
    "team_3p_pct": _team_stat("3P%", lambda t, s: f"{t} shot {s['3P%']}% from three-point range ({s['3PM']} of {s['3PA']})"),
    "team_fouls": _team_stat("PF", lambda t, s: f"{t} committed {_plural(s['PF'], 'personal foul')}"),
    # period_filtered
    "team_points_after": {
        "calls": lambda m: [("get_score_by_period", {})],
        "answer": lambda m, r: (
            _points_through(_side(r[0], m["team"]), m["period_num"]),
            f"{m['team']} had {_plural(_points_through(_side(r[0], m['team']), m['period_num']), 'point')} after {m['period']}",
        ),
    },
    "leader_after": {"calls": lambda m: [("get_score_by_period", {})], "answer": _leader_after},
    "team_points_in": {
        "calls": lambda m: [("get_score_by_period", {})],
        "answer": lambda m, r: (
            _side(r[0], m["team"])["by_period"][_period_label(m["period_num"])],
            f"{m['team']} scored {_plural(_side(r[0], m['team'])['by_period'][_period_label(m['period_num'])], 'point')} in {m['period']}",
        ),
    },
    "score_after": {
        "calls": lambda m: [("get_score_by_period", {})],
        "answer": lambda m, r: (_score_after(m, r)[0], f"the score after {m['period']} was {_score_after(m, r)[1]}"),
    },
    "player_points_in": {
        "calls": lambda m: [("get_box_score", {"player": m["player"], "period": m["period_num"]})],
        "answer": lambda m, r: (
            _players(r[0])[0]["PTS"],
            f"{m['player']} scored {_plural(_players(r[0])[0]['PTS'], 'point')} in {m['period']}",
        ),
    },
    # leader
    "top_scorer_team": _leader("PTS", "point", team_scope=True),
    "top_scorer_game": _leader("PTS", "point", team_scope=False),
    "top_rebounder_team": _leader("REB", "rebound", team_scope=True),
    "top_rebounder_game": _leader("REB", "rebound", team_scope=False),
    # comparison
    "team_more": {
        "calls": lambda m: [("get_team_stats", {"team": m["a"]}), ("get_team_stats", {"team": m["b"]})],
        "answer": _team_more,
    },
    "player_more_points": {
        "calls": lambda m: [("get_box_score", {"player": m["a"]}), ("get_box_score", {"player": m["b"]})],
        "answer": _player_more_points,
    },
    # tracking
    "contested_fg_pct_player": _contested("player"),
    "contested_fg_pct_team": _contested("team"),
    "team_picks": {
        "calls": lambda m: [("get_picks", {"team": m["team"]})],
        "answer": lambda m, r: (r[0]["picks"], f"{m['team']} ran {_plural(r[0]['picks'], 'pick-and-roll')}"),
    },
}

# What each tool's data is called in the assistant's sentences.
TOOL_NOUN = {
    "get_score_by_period": "period-by-period score",
    "get_box_score": "box score",
    "get_team_stats": "team stats",
    "get_shots": "shot data",
    "get_picks": "pick-and-roll data",
}

CALL_PHRASES = [
    "Let me pull the {noun} for that game.",
    "I'll check the {noun} for this game.",
    "Let me look up the {noun}.",
    "I'll get the {noun} to answer that.",
    "Checking the {noun} for game {game_id}.",
    "One moment, pulling the {noun}.",
]

ANSWER_PHRASES = [
    "{fact}.",
    "According to the {noun}, {fact}.",
    "Based on the {noun}, {fact}.",
    "Per the {noun}, {fact}.",
    "Looking at the {noun}, {fact}.",
    "{fact}, going by the {noun}.",
]


def partial_name(kind: str, name: str, game_id: int) -> str:
    """A shorter name the tools still resolve to the same team/player, or the full name if there is none."""
    words = name.split()
    short = words[-1] if len(words[-1]) >= 4 else words[0]
    if short == name:
        return name
    resolve = resolve_team if kind == "team" else resolve_player
    try:
        return short if resolve(short, game_id) == resolve(name, game_id) else name
    except LookupError:
        return name


def format_answer(answer, answer_type: str) -> str:
    return f"{answer}%" if answer_type == "percentage" else str(answer)


def build_trace(row: dict, rng: random.Random, plans=None) -> dict:
    meta, game_id = row["meta"], row["game_id"]
    plan = (plans or PLAN)[meta["subtype"]]

    calls = []
    for name, args in plan["calls"](meta):
        args = {
            k: partial_name(k, v, game_id) if k in ("team", "player") and rng.random() < PARTIAL_NAME_RATE else v
            for k, v in args.items()
        }
        calls.append((name, {"game_id": str(game_id), **args}))
    results = [TOOLS[name](**args) for name, args in calls]
    answer, fact = plan["answer"](meta, results)

    noun = TOOL_NOUN[calls[0][0]]
    tool_calls = [
        {
            "id": f"call_{rng.getrandbits(48):012x}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }
        for name, args in calls
    ]
    final = rng.choice(ANSWER_PHRASES).format(noun=noun, fact=fact)
    final = final[0].upper() + final[1:] + f"\nFINAL ANSWER: {format_answer(answer, row['answer_type'])}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": rng.choice(CALL_PHRASES).format(noun=noun, game_id=game_id), "tool_calls": tool_calls},
        *(
            {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)}
            for call, result in zip(tool_calls, results)
        ),
        {"role": "assistant", "content": final},
    ]
    return {"id": row["id"], "question_type": row["question_type"], "messages": messages, "tools": OPENAI_TOOLS}


def validate(trace: dict, row: dict) -> str | None:
    """None if the trace is sound, otherwise the reason it is not."""
    messages = trace["messages"]
    if messages[0] != {"role": "system", "content": SYSTEM_PROMPT} or messages[1]["content"] != row["question"]:
        return "system prompt or question does not match"
    tool_calls = messages[2]["tool_calls"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    if not tool_calls or len(tool_calls) != len(tool_messages):
        return "tool calls and tool results do not line up"
    for call, tool_message in zip(tool_calls, tool_messages):
        if call["id"] != tool_message["tool_call_id"]:
            return "tool call id mismatch"
        replayed = TOOLS[call["function"]["name"]](**json.loads(call["function"]["arguments"]))
        if "error" in replayed:
            return f"tool returned an error: {replayed['error']}"
        if json.loads(json.dumps(replayed)) != json.loads(tool_message["content"]):
            return "replayed tool result differs from the one in the trace"
    graded = grade(messages[-1]["content"], row["answer"], row["answer_type"])
    if not graded["correct"]:
        return f"FINAL ANSWER {graded['model_answer']!r} does not match ground truth {row['answer']!r}"
    return None


def generate_traces(rows=None, questions_file="train.jsonl", plans=None) -> tuple[list[dict], list[tuple[dict, str]]]:
    """(valid traces, [(question row, reason dropped)]) for every question in train.jsonl (or another question set)."""
    on_disk = {q["id"]: q for q in map(json.loads, (HERE / "datasets" / questions_file).read_text().splitlines())}
    rows = generate_train(with_meta=True) if rows is None else rows
    traces, dropped = [], []
    for row in rows:
        public = {k: v for k, v in row.items() if k != "meta"}
        if on_disk.get(row["id"]) != public:
            dropped.append((row, f"question does not match {questions_file} (regenerate it)"))
            continue
        try:
            trace = build_trace(row, random.Random(f"{SEED}-trace-{row['id']}"), plans)
            reason = validate(trace, row)
        except Exception as e:  # a trace that cannot be built is dropped, not fatal
            trace, reason = None, f"{type(e).__name__}: {e}"
        if reason:
            dropped.append((row, reason))
        else:
            traces.append(trace)
    return traces, dropped


if __name__ == "__main__":
    traces, dropped = generate_traces()
    for name, lines in (
        ("traces_synthetic.jsonl", traces),
        # Just the two keys a fine-tuning upload expects.
        ("traces_clean.jsonl", [{"messages": t["messages"], "tools": t["tools"]} for t in traces]),
    ):
        (HERE / "traces" / "synthetic" / name).write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines))
    print(f"Wrote {len(traces)} traces to traces/synthetic/; dropped {len(dropped)}")
    print("Traces per question_type:", dict(Counter(t["question_type"] for t in traces)))
    for row, reason in dropped:
        print(f"  dropped {row['id']}: {reason}")
