"""Harder questions for the same six question types, with ground truth computed from the game data.

Where the basic set (questions.py) is one lookup per question, these need combined filters,
arithmetic across tool results, a leader within a period, or two tool calls.

Run `python questions_hard.py` to write train_hard.jsonl and eval_hard.jsonl to datasets/.
"""

import itertools
from collections import Counter

import pandas as pd

from data import box_score, resolve_player, scoring_events
from questions import (
    EVAL_GAMES,
    EVAL_PER_GAME,
    MIN_CONTESTED_ATTEMPTS,
    OUT_DIR,
    PERIOD_NAMES,
    TRAIN_GAMES,
    TRAIN_PER_GAME,
    fg_line,
    generate,
    pct,
    team_stats,
    unique_leader,
    write_jsonl,
)

HALVES = {1: ("the first half", (1, 2)), 2: ("the second half", (3, 4))}
PERIOD_STATS = ("rebounds", "turnovers", "personal fouls")  # team stats asked about per quarter
GAME_STATS = ("rebounds", "turnovers", "personal fouls", "made three-pointers", "made free throws", "offensive rebounds")

TEMPLATES = {
    # player_stat
    "player_twos": [
        "How many two-point field goals did {player} make?",
        "How many of {player}'s made field goals were two-pointers?",
        "{player} made how many two-point baskets?",
    ],
    "player_fg_pct": [
        "What was {player}'s field goal percentage?",
        "What percentage of his field goal attempts did {player} make?",
        "{player} shot what percentage from the field?",
    ],
    "players_combined_points": [
        "How many points did {a} and {b} score combined?",
        "What was the combined point total of {a} and {b}?",
        "{a} and {b} combined for how many points?",
    ],
    # team_stat
    "team_stat_diff": [
        "What was the difference in {stat} between the two teams?",
        "By how many did one team beat the other in {stat}?",
        "How big was the gap in {stat} between {a} and {b}?",
    ],
    "team_stat_total": [
        "How many {stat} did the two teams have combined?",
        "What was the combined total of {stat} for both teams?",
        "{a} and {b} combined for how many {stat}?",
    ],
    "team_two_pt_pct": [
        "What was {team}'s field goal percentage on two-point shots?",
        "What percentage of its two-point attempts did {team} make?",
        "{team} shot what percentage inside the three-point line?",
    ],
    # period_filtered
    "team_stat_in": [
        "How many {stat} did {team} have in {period}?",
        "What was {team}'s total of {stat} in {period} alone?",
        "{team} recorded how many {stat} during {period}?",
    ],
    "team_points_half": [
        "How many points did {team} score in {half}?",
        "What was {team}'s point total in {half} alone?",
        "{team} scored how many points during {half}?",
    ],
    "margin_after": [
        "By how many points was the leading team ahead after {period}?",
        "What was the margin at the end of {period}?",
        "How many points separated the two teams at the end of {period}?",
    ],
    "player_points_half": [
        "How many points did {player} score in {half}?",
        "What was {player}'s point total in {half} alone?",
        "{player} scored how many points during {half}?",
    ],
    "player_rebounds_in": [
        "How many rebounds did {player} grab in {period}?",
        "What was {player}'s rebound total in {period} alone?",
        "{player} had how many rebounds during {period}?",
    ],
    # leader
    "top_scorer_in_game": [
        "Who scored the most points in {period}?",
        "Which player was the leading scorer of {period}?",
        "Who led all scorers during {period}?",
    ],
    "top_scorer_in_team": [
        "Who was {team}'s leading scorer in {period}?",
        "Which {team} player scored the most points in {period}?",
        "Who led {team} in points during {period}?",
    ],
    "top_scorer_half_team": [
        "Who was {team}'s leading scorer in {half}?",
        "Which {team} player scored the most points in {half}?",
        "Who led {team} in points during {half}?",
    ],
    "top_threes_game": [
        "Which player made the most three-pointers in the game?",
        "Who hit the most threes in this game?",
        "Who led all players in made three-pointers?",
    ],
    # comparison
    "team_more_in": [
        "Which team had more {stat} in {period}?",
        "In {period}, which team recorded more {stat}, {a} or {b}?",
        "Between {a} and {b}, which team had more {stat} during {period}?",
    ],
    "quarter_winner": [
        "Which team scored more points in {period}?",
        "Which team won {period} on the scoreboard?",
        "Between {a} and {b}, which team outscored the other in {period}?",
    ],
    "team_better_contested": [
        "Which team shot a higher percentage on contested shots?",
        "Which team had the better field goal percentage on contested shots, {a} or {b}?",
        "Between {a} and {b}, which team was more accurate on contested field goal attempts?",
    ],
    "player_better_contested": [
        "Who shot a higher percentage on contested shots, {a} or {b}?",
        "Which player had the better field goal percentage on contested shots: {a} or {b}?",
        "Between {a} and {b}, who was more accurate on contested field goal attempts?",
    ],
    "team_more_picks": [
        "Which team ran more pick-and-rolls?",
        "Which team set more on-ball screens (picks) on offense, {a} or {b}?",
        "Between {a} and {b}, which team ran more pick-and-rolls?",
    ],
    # tracking
    "contested_fg_pct_team_in": [
        "What was {team}'s field goal percentage on contested shots in {period}?",
        "In {period}, what percentage of its contested field goal attempts did {team} make?",
        "On contested shots during {period}, {team} shot what percentage from the field?",
    ],
    "uncontested_fg_pct_team": [
        "What was {team}'s field goal percentage on uncontested shots?",
        "What percentage of its uncontested field goal attempts did {team} make?",
        "On uncontested shots, {team} shot what percentage from the field?",
    ],
    "uncontested_fg_pct_player": [
        "What was {player}'s field goal percentage on uncontested shots?",
        "What percentage of his uncontested field goal attempts did {player} make?",
        "On uncontested shots, {player} shot what percentage from the field?",
    ],
    "contested_2p_pct_team": [
        "What was {team}'s field goal percentage on contested two-point shots?",
        "What percentage of its contested two-point attempts did {team} make?",
        "On contested two-pointers, {team} shot what percentage?",
    ],
    "contested_points_team": [
        "How many points did {team} score on contested field goals?",
        "How many of {team}'s points came from contested shots (free throws excluded)?",
        "{team} scored how many points on contested field goal attempts?",
    ],
    "team_picks_in": [
        "How many pick-and-rolls did {team} run in {period}?",
        "How many on-ball screens (picks) did {team} set on offense in {period}?",
        "{team} ran how many pick-and-rolls during {period}?",
    ],
}

SUBTYPES = {
    "player_stat": ["player_twos", "player_fg_pct", "players_combined_points"],
    "team_stat": ["team_stat_diff", "team_stat_total", "team_two_pt_pct"],
    "period_filtered": ["team_stat_in", "team_points_half", "margin_after", "player_points_half", "player_rebounds_in"],
    "leader": ["top_scorer_in_game", "top_scorer_in_team", "top_scorer_half_team", "top_threes_game"],
    "comparison": ["team_more_in", "quarter_winner", "team_better_contested", "player_better_contested", "team_more_picks"],
    "tracking": [
        "contested_fg_pct_team_in",
        "uncontested_fg_pct_team",
        "uncontested_fg_pct_player",
        "contested_2p_pct_team",
        "contested_points_team",
        "team_picks_in",
    ],
}


def flagged(shots, contested: bool):
    """Shots with a contest label equal to `contested` (shots without a label are left out)."""
    return shots[shots["contested"].notna() & (shots["contested"] == contested)]


def candidates(game) -> dict[str, list]:
    """Every askable hard question for a game, as subtype -> [(format kwargs, answer, answer_type)]."""
    out = {key: [] for key in TEMPLATES}
    home_id, away_id = game.teams
    teams = game.teams
    home, away = teams[home_id], teams[away_id]
    both = {"a": home, "b": away}
    periods = sorted(int(p) for p in game.chances["period"].unique())

    # Only players who actually played, and whose full name resolves back to them.
    played = set(game.chance_players["playerId"])
    box = box_score(game)
    box = box[[pid in played and resolve_player(name, game.game_id) == pid for pid, name in box["name"].items()]]
    names = box["name"]
    on_court = set(zip(game.chance_players["playerId"], game.chance_players["period"]))
    team_of = game.players["teamId"]

    shots, picks = game.shots, game.picks
    made, made_fts = shots[shots["outcome"]], game.free_throws[game.free_throws["outcome"]]
    # One row per made shot / made free throw: who scored, in which period, how many points.
    scored = pd.concat(
        [
            made[["shooterId", "period"]].assign(pts=made["three"].map({True: 3, False: 2})),
            made_fts[["shooterId", "period"]].assign(pts=1),
        ]
    ).dropna(subset=["shooterId"])
    rebounds = game.rebounds[game.rebounds["rebounded"]]
    team_period = scoring_events(game).groupby(["offTeamId", "period"])["pts"].sum()

    def points_by_player(in_periods, team_id=None):
        rows = scored[scored["period"].isin(in_periods)]
        if team_id is not None:
            rows = rows[rows["shooterId"].map(team_of) == team_id]
        return rows.groupby("shooterId")["pts"].sum()

    def add_leader(key, kwargs, series):
        if len(series) and (leader := unique_leader(series)) is not None:
            out[key].append((kwargs, game.players.at[leader, "name"], "player"))

    # Full-game player questions.
    for pid, row in box.iterrows():
        out["player_twos"].append(({"player": row["name"]}, int(row["FGM"] - row["3PM"]), "integer"))
        out["player_fg_pct"].append(({"player": row["name"]}, pct(int(row["FGM"]), int(row["FGA"])), "percentage"))
        un_made, un_att = fg_line(flagged(shots[shots["shooterId"] == pid], False))
        if un_att >= MIN_CONTESTED_ATTEMPTS:
            out["uncontested_fg_pct_player"].append(({"player": row["name"]}, pct(un_made, un_att), "percentage"))
    for a, b in itertools.combinations(box.index, 2):
        if box.at[a, "team"] == box.at[b, "team"]:
            out["players_combined_points"].append(({"a": names[a], "b": names[b]}, int(box.at[a, "PTS"] + box.at[b, "PTS"]), "integer"))
    if box["3PM"].max() > 0:
        add_leader("top_threes_game", {}, box["3PM"])

    # Full-game team questions.
    stats = {tid: team_stats(game, tid) for tid in teams}
    for stat in GAME_STATS:
        h, a = stats[home_id][stat], stats[away_id][stat]
        if h != a:
            out["team_stat_diff"].append(({"stat": stat, **both}, abs(h - a), "integer"))
        out["team_stat_total"].append(({"stat": stat, **both}, h + a, "integer"))
    contested_pct = {}
    for tid, name in teams.items():
        mine = shots[shots["offTeamId"] == tid]
        out["team_two_pt_pct"].append(({"team": name}, pct(*fg_line(mine[~mine["three"]])), "percentage"))
        out["uncontested_fg_pct_team"].append(({"team": name}, pct(*fg_line(flagged(mine, False))), "percentage"))
        contested = flagged(mine, True)
        out["contested_2p_pct_team"].append(({"team": name}, pct(*fg_line(contested[~contested["three"]])), "percentage"))
        hit = contested[contested["outcome"]]
        out["contested_points_team"].append(({"team": name}, int(2 * len(hit) + hit["three"].sum()), "integer"))
        contested_pct[tid] = pct(*fg_line(contested))
    if None not in contested_pct.values() and contested_pct[home_id] != contested_pct[away_id]:
        out["team_better_contested"].append((both, teams[max(contested_pct, key=contested_pct.get)], "team"))
    n_picks = {tid: int((picks["offTeamId"] == tid).sum()) for tid in teams}
    if n_picks[home_id] != n_picks[away_id]:
        out["team_more_picks"].append((both, teams[max(n_picks, key=n_picks.get)], "team"))
    player_contested = {}
    for pid in box.index:
        c_made, c_att = fg_line(flagged(shots[shots["shooterId"] == pid], True))
        if c_att >= MIN_CONTESTED_ATTEMPTS:
            player_contested[pid] = pct(c_made, c_att)
    for a, b in itertools.permutations(player_contested, 2):
        if player_contested[a] != player_contested[b]:
            better = a if player_contested[a] > player_contested[b] else b
            out["player_better_contested"].append(({"a": names[a], "b": names[b]}, names[better], "player"))

    # Halves: Q1+Q2 and Q3+Q4 (overtime belongs to neither).
    for half_num, (half, half_periods) in HALVES.items():
        where = {"half": half, "half_num": half_num}
        for tid, name in teams.items():
            pts = int(sum(team_period.get((tid, p), 0) for p in half_periods))
            out["team_points_half"].append(({"team": name, **where}, pts, "integer"))
            add_leader("top_scorer_half_team", {"team": name, **where}, points_by_player(half_periods, tid))
        totals = points_by_player(half_periods)
        for pid, name in names.items():
            if all((pid, p) in on_court for p in half_periods):
                out["player_points_half"].append(({"player": name, **where}, int(totals.get(pid, 0)), "integer"))

    # Single periods.
    for period in periods:
        def in_period(df):
            return df[df["period"] == period]

        counts = {
            tid: {
                "rebounds": int((in_period(game.rebounds)["teamId"] == tid).sum()),
                "turnovers": int((in_period(game.turnovers)["teamId"] == tid).sum()),
                "personal fouls": int((in_period(game.fouls)["foulerTeamId"] == tid).sum()),
            }
            for tid in teams
        }
        period_points = {tid: int(team_period.get((tid, period), 0)) for tid in teams}
        so_far = {tid: int(sum(team_period.get((tid, p), 0) for p in periods if p <= period)) for tid in teams}
        for period_name in PERIOD_NAMES[period]:
            where = {"period": period_name, "period_num": period}
            for tid, name in teams.items():
                for stat in PERIOD_STATS:
                    out["team_stat_in"].append(({"team": name, "stat": stat, **where}, counts[tid][stat], "integer"))
                c_made, c_att = fg_line(flagged(in_period(shots[shots["offTeamId"] == tid]), True))
                if c_att >= MIN_CONTESTED_ATTEMPTS:
                    out["contested_fg_pct_team_in"].append(({"team": name, **where}, pct(c_made, c_att), "percentage"))
                out["team_picks_in"].append(({"team": name, **where}, int((in_period(picks)["offTeamId"] == tid).sum()), "integer"))
                add_leader("top_scorer_in_team", {"team": name, **where}, points_by_player([period], tid))
            add_leader("top_scorer_in_game", where, points_by_player([period]))
            for stat in PERIOD_STATS:
                if counts[home_id][stat] != counts[away_id][stat]:
                    more = max(counts, key=lambda tid: counts[tid][stat])
                    out["team_more_in"].append(({"stat": stat, **both, **where}, teams[more], "team"))
            if period_points[home_id] != period_points[away_id]:
                out["quarter_winner"].append(({**both, **where}, teams[max(period_points, key=period_points.get)], "team"))
            # The margin after the last period is just the final margin.
            if period != periods[-1] and so_far[home_id] != so_far[away_id]:
                out["margin_after"].append((where, abs(so_far[home_id] - so_far[away_id]), "integer"))
            for pid, name in names.items():
                if (pid, period) in on_court:
                    n = int((in_period(rebounds)["rebounderId"] == pid).sum())
                    out["player_rebounds_in"].append(({"player": name, **where}, n, "integer"))

    return {key: [c for c in cands if c[1] is not None] for key, cands in out.items()}


TIER = ("hard-", TEMPLATES, SUBTYPES, candidates)


def generate_train_hard(with_meta: bool = False) -> list[dict]:
    return generate(TRAIN_GAMES, TRAIN_PER_GAME, with_meta=with_meta, tier=TIER)


def generate_eval_hard(with_meta: bool = False) -> list[dict]:
    return generate(EVAL_GAMES, EVAL_PER_GAME, with_meta=with_meta, tier=TIER)


if __name__ == "__main__":
    for name, rows in (("train_hard", generate_train_hard()), ("eval_hard", generate_eval_hard())):
        write_jsonl(OUT_DIR / f"{name}.jsonl", rows)
        print(f"{name}.jsonl: {len(rows)} questions", dict(Counter(r["question_type"] for r in rows)))
