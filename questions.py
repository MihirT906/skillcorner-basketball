"""Generate questions with ground-truth answers computed from the game data (data.py), never from an LLM.

Run `python questions.py` to write train.jsonl and eval.jsonl to datasets/.
"""

import itertools
import json
import random
from collections import Counter
from pathlib import Path

from data import box_score, load_game, resolve_player, scoring_events

SEED = 42
TRAIN_GAMES = [114243, 114234, 114169, 114099, 114086, 179612, 184439, 188630]
EVAL_GAMES = [191313, 178442]
TRAIN_PER_GAME = 30
EVAL_PER_GAME = 20
# eval.jsonl is frozen: it was generated before these templates existed, so they are left out of it.
EVAL_EXCLUDED_TEMPLATES = frozenset({"score_after"})
OUT_DIR = Path(__file__).resolve().parent / "datasets"

# Target share of each question type per game.
MIX = {
    "player_stat": 0.25,
    "team_stat": 0.20,
    "period_filtered": 0.20,
    "leader": 0.15,
    "comparison": 0.10,
    "tracking": 0.10,
}

MIN_CONTESTED_ATTEMPTS = 4  # a player needs this many contested attempts to be asked about them

PERIOD_NAMES = {
    1: ["the first quarter", "the 1st quarter", "Q1"],
    2: ["the second quarter", "the 2nd quarter", "Q2"],
    3: ["the third quarter", "the 3rd quarter", "Q3"],
    4: ["the fourth quarter", "the 4th quarter", "Q4"],
    5: ["overtime", "the overtime period", "OT"],
}

TEMPLATES = {
    # player_stat
    "player_points": [
        "How many points did {player} score?",
        "What was {player}'s point total?",
        "{player} finished with how many points?",
    ],
    "player_threes": [
        "How many three-pointers did {player} make?",
        "How many made threes did {player} have?",
        "{player} hit how many three-point shots?",
    ],
    "player_rebounds": [
        "How many rebounds did {player} grab?",
        "What was {player}'s rebound total?",
        "{player} finished with how many rebounds?",
    ],
    "player_turnovers": [
        "How many turnovers did {player} commit?",
        "What was {player}'s turnover count?",
        "{player} turned the ball over how many times?",
    ],
    # team_stat
    "team_turnovers": [
        "How many turnovers did {team} commit?",
        "What was {team}'s turnover total?",
        "{team} finished with how many turnovers?",
    ],
    "team_rebounds": [
        "How many rebounds did {team} grab?",
        "What was {team}'s total rebound count?",
        "{team} finished with how many rebounds?",
    ],
    "team_fg_pct": [
        "What was {team}'s field goal percentage?",
        "What percentage of its field goal attempts did {team} make?",
        "{team} shot what percentage from the field?",
    ],
    "team_3p_pct": [
        "What was {team}'s three-point percentage?",
        "What percentage of its three-point attempts did {team} make?",
        "{team} shot what percentage from three-point range?",
    ],
    "team_fouls": [
        "How many personal fouls did {team} commit?",
        "What was {team}'s foul total?",
        "{team} was called for how many fouls?",
    ],
    # period_filtered
    "team_points_after": [
        "How many points did {team} have after {period}?",
        "What was {team}'s point total at the end of {period}?",
        "At the end of {period}, how many points had {team} scored in the game so far?",
    ],
    "leader_after": [
        "Which team was leading after {period}?",
        "Which team was ahead at the end of {period}?",
        "At the end of {period}, which team had the lead?",
    ],
    "team_points_in": [
        "How many points did {team} score in {period}?",
        "What was {team}'s point total in {period} alone?",
        "{team} scored how many points during {period}?",
    ],
    "score_after": [
        "What was the score after {period}?",
        "What was the score at the end of {period}?",
        "Give the score at the end of {period}.",
    ],
    "player_points_in": [
        "How many points did {player} score in {period}?",
        "What was {player}'s point total in {period} alone?",
        "{player} scored how many points during {period}?",
    ],
    # leader
    "top_scorer_team": [
        "Who was {team}'s leading scorer?",
        "Which {team} player scored the most points?",
        "Who led {team} in points?",
    ],
    "top_scorer_game": [
        "Who was the game's leading scorer?",
        "Which player scored the most points in the game?",
        "Who led all scorers in this game?",
    ],
    "top_rebounder_team": [
        "Who was {team}'s leading rebounder?",
        "Which {team} player grabbed the most rebounds?",
        "Who led {team} in rebounds?",
    ],
    "top_rebounder_game": [
        "Who was the game's leading rebounder?",
        "Which player grabbed the most rebounds in the game?",
        "Who led all players in rebounds in this game?",
    ],
    # comparison
    "team_more": [
        "Which team had more {stat}?",
        "Which team finished with more {stat}, {a} or {b}?",
        "Between {a} and {b}, which team recorded more {stat}?",
    ],
    "player_more_points": [
        "Who scored more points, {a} or {b}?",
        "Which player had more points: {a} or {b}?",
        "Between {a} and {b}, who was the higher scorer?",
    ],
    # tracking
    "contested_fg_pct_player": [
        "What was {player}'s field goal percentage on contested shots?",
        "What percentage of his contested field goal attempts did {player} make?",
        "On contested shots, {player} shot what percentage from the field?",
    ],
    "contested_fg_pct_team": [
        "What was {team}'s field goal percentage on contested shots?",
        "What percentage of its contested field goal attempts did {team} make?",
        "On contested shots, {team} shot what percentage from the field?",
    ],
    "team_picks": [
        "How many pick-and-rolls did {team} run?",
        "How many on-ball screens (picks) did {team} set on offense?",
        "{team} ran how many pick-and-rolls in this game?",
    ],
}

# Which templates belong to which question type.
SUBTYPES = {
    "player_stat": ["player_points", "player_threes", "player_rebounds", "player_turnovers"],
    "team_stat": ["team_turnovers", "team_rebounds", "team_fg_pct", "team_3p_pct", "team_fouls"],
    "period_filtered": ["team_points_after", "leader_after", "team_points_in", "player_points_in", "score_after"],
    "leader": ["top_scorer_team", "top_scorer_game", "top_rebounder_team", "top_rebounder_game"],
    "comparison": ["team_more", "player_more_points"],
    "tracking": ["contested_fg_pct_player", "contested_fg_pct_team", "team_picks"],
}


def pct(made, attempted):
    return round(100 * made / attempted, 1) if attempted else None


def fg_line(shots):
    """(makes, attempts) with fouled misses excluded from attempts."""
    return int(shots["outcome"].sum()), int((shots["outcome"] | ~shots["fouled"]).sum())


def team_stats(game, team_id) -> dict:
    """Team totals. Rebounds and turnovers include team rebounds / team turnovers."""
    shots = game.shots[game.shots["offTeamId"] == team_id]
    fts = game.free_throws[game.free_throws["offTeamId"] == team_id]
    rebs = game.rebounds[game.rebounds["teamId"] == team_id]
    fgm, fga = fg_line(shots)
    tpm, tpa = fg_line(shots[shots["three"]])
    return {
        "turnovers": int((game.turnovers["teamId"] == team_id).sum()),
        "rebounds": len(rebs),
        "offensive rebounds": int((~rebs["defensive"]).sum()),
        "fg_pct": pct(fgm, fga),
        "3p_pct": pct(tpm, tpa),
        "personal fouls": int((game.fouls["foulerTeamId"] == team_id).sum()),
        "made three-pointers": tpm,
        "made free throws": int(fts["outcome"].sum()),
        "picks": int((game.picks["offTeamId"] == team_id).sum()),
    }


def contested_fg_line(shots):
    """(makes, attempts) on contested shots: contestLevel average / plus / blocked."""
    return fg_line(shots[shots["contested"].notna() & (shots["contested"] == True)])  # noqa: E712


def unique_leader(series):
    """Index label of the max value, or None if the top value is tied."""
    top = series.sort_values(ascending=False)
    if len(top) > 1 and top.iloc[0] == top.iloc[1]:
        return None
    return top.index[0]


def candidates(game) -> dict[str, list]:
    """Every askable question for a game, as subtype -> [(format kwargs, answer, answer_type)]."""
    out = {key: [] for key in TEMPLATES}
    home_id, away_id = game.teams
    teams = game.teams
    periods = sorted(int(p) for p in game.chances["period"].unique())

    # Only players who actually played, and whose full name resolves back to them.
    played = set(game.chance_players["playerId"])
    box = box_score(game)
    box = box[[pid in played and resolve_player(name, game.game_id) == pid for pid, name in box["name"].items()]]
    names = box["name"]

    for pid, row in box.iterrows():
        out["player_points"].append(({"player": row["name"]}, int(row["PTS"]), "integer"))
        out["player_threes"].append(({"player": row["name"]}, int(row["3PM"]), "integer"))
        out["player_rebounds"].append(({"player": row["name"]}, int(row["REB"]), "integer"))
        out["player_turnovers"].append(({"player": row["name"]}, int(row["TOV"]), "integer"))

    stats = {tid: team_stats(game, tid) for tid in teams}
    for tid, name in teams.items():
        s = stats[tid]
        out["team_turnovers"].append(({"team": name}, s["turnovers"], "integer"))
        out["team_rebounds"].append(({"team": name}, s["rebounds"], "integer"))
        out["team_fouls"].append(({"team": name}, s["personal fouls"], "integer"))
        out["team_fg_pct"].append(({"team": name}, s["fg_pct"], "percentage"))
        out["team_3p_pct"].append(({"team": name}, s["3p_pct"], "percentage"))
        out["team_picks"].append(({"team": name}, s["picks"], "integer"))
        made, att = contested_fg_line(game.shots[game.shots["offTeamId"] == tid])
        if att:
            out["contested_fg_pct_team"].append(({"team": name}, pct(made, att), "percentage"))

    # Period-filtered: points by team and by player per period, from shots + free throws.
    points = scoring_events(game)
    team_period = points.groupby(["offTeamId", "period"])["pts"].sum()
    made_shots = game.shots[game.shots["outcome"]]
    made_fts = game.free_throws[game.free_throws["outcome"]]
    on_court = set(zip(game.chance_players["playerId"], game.chance_players["period"]))
    for period in periods:
        for period_name in PERIOD_NAMES[period]:
            for tid, name in teams.items():
                out["team_points_in"].append(
                    ({"team": name, "period": period_name, "period_num": period}, int(team_period.get((tid, period), 0)), "integer")
                )
        # Cumulative score, only for periods before the last (after the last it is just the final score).
        if period != periods[-1]:
            so_far = {tid: int(sum(team_period.get((tid, p), 0) for p in periods if p <= period)) for tid in teams}
            after_names = PERIOD_NAMES[period] + (["the first half"] if period == 2 else [])
            for period_name in after_names:
                for tid, name in teams.items():
                    out["team_points_after"].append(({"team": name, "period": period_name, "period_num": period}, so_far[tid], "integer"))
                # "<home team> <points>-<points> <away team>"
                score = f"{teams[home_id]} {so_far[home_id]}-{so_far[away_id]} {teams[away_id]}"
                out["score_after"].append(({"period": period_name, "period_num": period}, score, "score"))
                if so_far[home_id] != so_far[away_id]:
                    leader = teams[max(so_far, key=so_far.get)]
                    out["leader_after"].append(({"period": period_name, "period_num": period}, leader, "team"))
        for pid, name in names.items():
            if (pid, period) not in on_court:
                continue
            shots = made_shots[(made_shots["shooterId"] == pid) & (made_shots["period"] == period)]
            fts = made_fts[(made_fts["shooterId"] == pid) & (made_fts["period"] == period)]
            pts = int(2 * len(shots) + shots["three"].sum() + len(fts))
            for period_name in PERIOD_NAMES[period]:
                out["player_points_in"].append(({"player": name, "period": period_name, "period_num": period}, pts, "integer"))

    # Leaders (skipped when the top value is tied).
    for stat, key in (("PTS", "top_scorer"), ("REB", "top_rebounder")):
        leader = unique_leader(box[stat])
        if leader is not None:
            out[f"{key}_game"].append(({}, names[leader], "player"))
        for tid, name in teams.items():
            leader = unique_leader(box.loc[box["team"] == name, stat])
            if leader is not None:
                out[f"{key}_team"].append(({"team": name}, names[leader], "player"))

    # Comparisons (skipped when tied).
    home, away = teams[home_id], teams[away_id]
    for stat in ("rebounds", "turnovers", "personal fouls", "made three-pointers", "made free throws", "offensive rebounds"):
        h, a = stats[home_id][stat], stats[away_id][stat]
        if h != a:
            out["team_more"].append(({"stat": stat, "a": home, "b": away}, home if h > a else away, "team"))
    for a, b in itertools.permutations(box.index, 2):
        if box.at[a, "PTS"] != box.at[b, "PTS"]:
            winner = names[a] if box.at[a, "PTS"] > box.at[b, "PTS"] else names[b]
            out["player_more_points"].append(({"a": names[a], "b": names[b]}, winner, "player"))

    for pid, name in names.items():
        made, att = contested_fg_line(game.shots[game.shots["shooterId"] == pid])
        if att >= MIN_CONTESTED_ATTEMPTS:
            out["contested_fg_pct_player"].append(({"player": name}, pct(made, att), "percentage"))

    return {key: [c for c in cands if c[1] is not None] for key, cands in out.items()}


def type_counts(n: int, tie_break: int) -> dict[str, int]:
    """Split n questions across types by MIX (largest remainder; ties alternate by game so they even out)."""
    exact = {t: n * share for t, share in MIX.items()}
    counts = {t: int(x) for t, x in exact.items()}
    types = list(MIX) if tie_break % 2 == 0 else list(MIX)[::-1]
    by_remainder = sorted(types, key=lambda t: -(exact[t] - counts[t]))
    for t in by_remainder[: n - sum(counts.values())]:
        counts[t] += 1
    return counts


def generate_game(game_id: int, n: int, tie_break: int = 0, with_meta: bool = False, exclude=frozenset()) -> list[dict]:
    """Questions for one game. with_meta adds a `meta` dict (template key + the entities asked about);
    exclude lists templates to leave out."""
    game = load_game(game_id)
    rng = random.Random(f"{SEED}-{game_id}")
    home, away = game.teams.values()
    context = f"Game {game.game_id}: {home} vs {away} ({game.meta['date'][:10]}). "
    cands = candidates(game)

    questions = []
    for qtype, count in type_counts(n, tie_break).items():
        pools = {key: rng.sample(cands[key], len(cands[key])) for key in SUBTYPES[qtype] if cands[key] and key not in exclude}
        order = rng.sample(list(pools), len(pools))
        picked, seen, repeats = [], set(), []
        # Round-robin over the subtypes so one game does not ask the same kind of thing repeatedly.
        for key in itertools.cycle(order):
            if len(picked) == count or not any(pools.values()):
                break
            if not pools[key]:
                continue
            kwargs, answer, answer_type = pools[key].pop()
            text = rng.choice(TEMPLATES[key]).format(**kwargs)
            if qtype == "leader":
                # A team's leader who also leads the game is asked about only once per stat.
                fingerprint = (key.rsplit("_", 1)[0], answer)
            else:
                # Different period spellings of the same question count as duplicates.
                fingerprint = (key, answer, tuple(v for k, v in sorted(kwargs.items()) if k not in ("period", "period_num")))
            question = (text, answer, answer_type, {"subtype": key, **kwargs})
            if fingerprint in seen:
                repeats.append(question)
                continue
            seen.add(fingerprint)
            picked.append(question)
        if qtype == "leader":
            # Too few distinct leaders to fill the quota: also ask the team-level and game-level versions.
            picked += repeats[: count - len(picked)]
        for i, (text, answer, answer_type, meta) in enumerate(picked, 1):
            questions.append(
                {
                    "id": f"{game.game_id}-{qtype}-{i:02d}",
                    "game_id": game.game_id,
                    "question": context + text,
                    "answer": answer,
                    "answer_type": answer_type,
                    "question_type": qtype,
                    **({"meta": meta} if with_meta else {}),
                }
            )
    return questions


def generate(game_ids: list[int], per_game: int, with_meta: bool = False, exclude=frozenset()) -> list[dict]:
    return [
        q
        for i, gid in enumerate(game_ids)
        for q in generate_game(gid, per_game, tie_break=i, with_meta=with_meta, exclude=exclude)
    ]


def generate_train(with_meta: bool = False) -> list[dict]:
    return generate(TRAIN_GAMES, TRAIN_PER_GAME, with_meta=with_meta)


def generate_eval(with_meta: bool = False) -> list[dict]:
    return generate(EVAL_GAMES, EVAL_PER_GAME, with_meta=with_meta, exclude=EVAL_EXCLUDED_TEMPLATES)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    for name, rows in (("train", generate_train()), ("eval", generate_eval())):
        write_jsonl(OUT_DIR / f"{name}.jsonl", rows)
        print(f"{name}.jsonl: {len(rows)} questions", dict(Counter(r["question_type"] for r in rows)))
