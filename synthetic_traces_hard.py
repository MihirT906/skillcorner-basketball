"""Ideal-agent training traces for the hard questions (train_hard.jsonl), without calling any LLM.

Same trace shape and validation as synthetic_traces.py; only the plans differ.

    python synthetic_traces_hard.py   ->  traces/synthetic/traces_hard_synthetic.jsonl
"""

import json
from collections import Counter

from questions_hard import HALVES, generate_train_hard
from synthetic_traces import HERE, TEAM_MORE_STATS, _period_label, _players, _plural, _points_through, _top, generate_traces

# ---------------------------------------------------------------------------------------
# question_type    template                   tool call(s)
# ---------------  -------------------------  --------------------------------------------
# player_stat      player_twos                get_box_score(player)            FGM - 3PM
#                  player_fg_pct              get_shots(player)
#                  players_combined_points    get_box_score(a) + get_box_score(b)
# team_stat        team_stat_diff / _total    get_team_stats(a) + get_team_stats(b)
#                  team_two_pt_pct            get_shots(team, three_pt=False)
# period_filtered  team_stat_in               get_team_stats(team, period)
#                  team_points_half,          get_score_by_period()
#                  margin_after
#                  player_points_half         get_box_score(player, period) x 2
#                  player_rebounds_in         get_box_score(player, period)
# leader           top_scorer_in_game         get_box_score(period)
#                  top_scorer_in_team         get_box_score(team, period)
#                  top_scorer_half_team       get_box_score(team, period) x 2
#                  top_threes_game            get_box_score()
# comparison       team_more_in               get_team_stats(a, period) + get_team_stats(b, period)
#                  quarter_winner             get_score_by_period()
#                  team_better_contested      get_shots(a, contested=True) + get_shots(b, contested=True)
#                  player_better_contested    get_shots(player a, contested=True) + (player b)
#                  team_more_picks            get_picks(a) + get_picks(b)
# tracking         contested_fg_pct_team_in   get_shots(team, period, contested=True)
#                  uncontested_fg_pct_*       get_shots(team | player, contested=False)
#                  contested_2p_pct_team      get_shots(team, three_pt=False, contested=True)
#                  contested_points_team      get_shots(team, contested=True)
#                  team_picks_in              get_picks(team, period)
# ---------------------------------------------------------------------------------------


def _both(tool, **extra):
    """The same tool once per team (or player) named in the question."""
    key = extra.pop("key", "team")
    return lambda m: [(tool, {key: m["a"], **extra, **({"period": m["period_num"]} if "period_num" in m else {})}),
                      (tool, {key: m["b"], **extra, **({"period": m["period_num"]} if "period_num" in m else {})})]


def _half_calls(who):
    return lambda m: [("get_box_score", {who: m[who], "period": p}) for p in HALVES[m["half_num"]][1]]


def _shot_pct(who, what, **filters):
    def calls(m):
        return [("get_shots", {who: m[who], **({"period": m["period_num"]} if "period_num" in m else {}), **filters})]

    def answer(m, r):
        where = f" in {m['period']}" if "period" in m else ""
        return r[0]["fg_pct"], f"{r[0]['filters'][who]} made {r[0]['makes']} of {r[0]['attempts']} {what}{where}, which is {r[0]['fg_pct']}%"

    return {"calls": calls, "answer": answer}


def _player_twos(m, r):
    p = _players(r[0])[0]
    twos = p["FGM"] - p["3PM"]
    return twos, f"{p['player']} made {_plural(p['FGM'], 'field goal')}, {p['3PM']} of them from three, so {_plural(twos, 'two-pointer')}"


def _combined_points(m, r):
    a, b = (_players(x)[0] for x in r)
    return a["PTS"] + b["PTS"], f"{a['player']} scored {a['PTS']} and {b['player']} scored {b['PTS']}, {a['PTS'] + b['PTS']} points combined"


def _stat_diff(m, r):
    stat = TEAM_MORE_STATS[m["stat"]]
    hi, lo = sorted(r, key=lambda t: -t[stat])
    return hi[stat] - lo[stat], f"{hi['team']} had {hi[stat]} {m['stat']} to {lo['team']}'s {lo[stat]}, a difference of {hi[stat] - lo[stat]}"


def _stat_total(m, r):
    stat = TEAM_MORE_STATS[m["stat"]]
    total = r[0][stat] + r[1][stat]
    return total, f"{r[0]['team']} had {r[0][stat]} {m['stat']} and {r[1]['team']} had {r[1][stat]}, {total} in total"


def _team_stat_in(m, r):
    value = r[0][TEAM_MORE_STATS[m["stat"]]]
    return value, f"{r[0]['team']} had {value} {m['stat']} in {m['period']}"


def _team_points_half(m, r):
    side = r[0]["home"] if r[0]["home"]["team"] == m["team"] else r[0]["away"]
    quarters = [side["by_period"][_period_label(p)] for p in HALVES[m["half_num"]][1]]
    return sum(quarters), f"{m['team']} scored {quarters[0]} and {quarters[1]} in the two quarters of {m['half']}, {sum(quarters)} points in all"


def _margin_after(m, r):
    home, away = (_points_through(r[0][side], m["period_num"]) for side in ("home", "away"))
    lead, trail = (r[0]["home"], r[0]["away"]) if home > away else (r[0]["away"], r[0]["home"])
    margin = abs(home - away)
    return margin, f"{lead['team']} led {trail['team']} {max(home, away)}-{min(home, away)} after {m['period']}, a margin of {_plural(margin, 'point')}"


def _player_points_half(m, r):
    quarters = [_players(x)[0]["PTS"] for x in r]
    return sum(quarters), f"{m['player']} scored {quarters[0]} and {quarters[1]} in the two quarters of {m['half']}, {sum(quarters)} points in all"


def _player_rebounds_in(m, r):
    p = _players(r[0])[0]
    return p["REB"], f"{p['player']} grabbed {_plural(p['REB'], 'rebound')} in {m['period']}"


def _top_scorer_in(m, r):
    top = _top(r[0], "PTS")
    whom = m.get("team", "all scorers")
    return top["player"], f"{top['player']} led {whom} with {_plural(top['PTS'], 'point')} in {m['period']}"


def _top_scorer_half(m, r):
    points = Counter()
    for result in r:
        for p in _players(result):
            points[p["player"]] += p["PTS"]
    player, pts = points.most_common(1)[0]
    return player, f"adding the two quarters, {player} led {m['team']} with {_plural(pts, 'point')} in {m['half']}"


def _top_threes(m, r):
    top = _top(r[0], "3PM")
    return top["player"], f"{top['player']} made the most three-pointers with {top['3PM']}"


def _team_more_in(m, r):
    stat = TEAM_MORE_STATS[m["stat"]]
    hi, lo = sorted(r, key=lambda t: -t[stat])
    return hi["team"], f"{hi['team']} had {hi[stat]} {m['stat']} in {m['period']} to {lo['team']}'s {lo[stat]}"


def _quarter_winner(m, r):
    label = _period_label(m["period_num"])
    hi, lo = sorted((r[0]["home"], r[0]["away"]), key=lambda s: -s["by_period"][label])
    return hi["team"], f"{hi['team']} outscored {lo['team']} {hi['by_period'][label]}-{lo['by_period'][label]} in {m['period']}"


def _better_contested(who):
    def answer(m, r):
        hi, lo = sorted(r, key=lambda s: -s["fg_pct"])
        return hi["filters"][who], (
            f"{hi['filters'][who]} shot {hi['fg_pct']}% on contested shots ({hi['makes']} of {hi['attempts']}) "
            f"to {lo['filters'][who]}'s {lo['fg_pct']}% ({lo['makes']} of {lo['attempts']})"
        )

    return answer


def _more_picks(m, r):
    hi, lo = sorted(r, key=lambda s: -s["picks"])
    return hi["filters"]["team"], f"{hi['filters']['team']} ran {hi['picks']} pick-and-rolls to {lo['filters']['team']}'s {lo['picks']}"


PLAN = {
    # player_stat
    "player_twos": {"calls": lambda m: [("get_box_score", {"player": m["player"]})], "answer": _player_twos},
    "player_fg_pct": _shot_pct("player", "field goal attempts"),
    "players_combined_points": {"calls": _both("get_box_score", key="player"), "answer": _combined_points},
    # team_stat
    "team_stat_diff": {"calls": _both("get_team_stats"), "answer": _stat_diff},
    "team_stat_total": {"calls": _both("get_team_stats"), "answer": _stat_total},
    "team_two_pt_pct": _shot_pct("team", "two-point attempts", three_pt=False),
    # period_filtered
    "team_stat_in": {"calls": lambda m: [("get_team_stats", {"team": m["team"], "period": m["period_num"]})], "answer": _team_stat_in},
    "team_points_half": {"calls": lambda m: [("get_score_by_period", {})], "answer": _team_points_half},
    "margin_after": {"calls": lambda m: [("get_score_by_period", {})], "answer": _margin_after},
    "player_points_half": {"calls": _half_calls("player"), "answer": _player_points_half},
    "player_rebounds_in": {
        "calls": lambda m: [("get_box_score", {"player": m["player"], "period": m["period_num"]})],
        "answer": _player_rebounds_in,
    },
    # leader
    "top_scorer_in_game": {"calls": lambda m: [("get_box_score", {"period": m["period_num"]})], "answer": _top_scorer_in},
    "top_scorer_in_team": {
        "calls": lambda m: [("get_box_score", {"team": m["team"], "period": m["period_num"]})],
        "answer": _top_scorer_in,
    },
    "top_scorer_half_team": {"calls": _half_calls("team"), "answer": _top_scorer_half},
    "top_threes_game": {"calls": lambda m: [("get_box_score", {})], "answer": _top_threes},
    # comparison
    "team_more_in": {"calls": _both("get_team_stats"), "answer": _team_more_in},
    "quarter_winner": {"calls": lambda m: [("get_score_by_period", {})], "answer": _quarter_winner},
    "team_better_contested": {"calls": _both("get_shots", contested=True), "answer": _better_contested("team")},
    "player_better_contested": {"calls": _both("get_shots", key="player", contested=True), "answer": _better_contested("player")},
    "team_more_picks": {"calls": _both("get_picks"), "answer": _more_picks},
    # tracking
    "contested_fg_pct_team_in": _shot_pct("team", "contested field goal attempts", contested=True),
    "uncontested_fg_pct_team": _shot_pct("team", "uncontested field goal attempts", contested=False),
    "uncontested_fg_pct_player": _shot_pct("player", "uncontested field goal attempts", contested=False),
    "contested_2p_pct_team": _shot_pct("team", "contested two-point attempts", three_pt=False, contested=True),
    "contested_points_team": {
        "calls": lambda m: [("get_shots", {"team": m["team"], "contested": True})],
        "answer": lambda m, r: (r[0]["points"], f"{r[0]['filters']['team']} scored {_plural(r[0]['points'], 'point')} on contested field goals"),
    },
    "team_picks_in": {
        "calls": lambda m: [("get_picks", {"team": m["team"], "period": m["period_num"]})],
        "answer": lambda m, r: (r[0]["picks"], f"{r[0]['filters']['team']} ran {_plural(r[0]['picks'], 'pick-and-roll')} in {m['period']}"),
    },
}


def generate_traces_hard(rows=None, questions_file="train_hard.jsonl"):
    """(valid traces, [(question row, reason dropped)]) for the hard training questions."""
    return generate_traces(generate_train_hard(with_meta=True) if rows is None else rows, questions_file, PLAN)


if __name__ == "__main__":
    traces, dropped = generate_traces_hard()
    for name, lines in (
        ("traces_hard_synthetic.jsonl", traces),
        ("traces_hard_clean.jsonl", [{"messages": t["messages"], "tools": t["tools"]} for t in traces]),
    ):
        (HERE / "traces" / "synthetic" / name).write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines))
    print(f"Wrote {len(traces)} hard traces to traces/synthetic/; dropped {len(dropped)}")
    print("Traces per question_type:", dict(Counter(t["question_type"] for t in traces)))
    for row, reason in dropped:
        print(f"  dropped {row['id']}: {reason}")
