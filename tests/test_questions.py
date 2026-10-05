import json
from collections import Counter

import tools
from data import load_game
import hashlib
from pathlib import Path

from grade import parse_score
from questions import (EVAL_GAMES, EVAL_PER_GAME, MIX, TEMPLATES, TRAIN_GAMES, TRAIN_PER_GAME, contested_fg_line,
                       generate_eval, generate_train, pct, team_stats)

HERE = Path(__file__).resolve().parent.parent / "datasets"

FIELDS = ["id", "game_id", "question", "answer", "answer_type", "question_type"]


def test_templates_have_three_phrasings():
    assert all(len(set(phrasings)) >= 3 for phrasings in TEMPLATES.values())


def test_generation_is_deterministic_and_well_formed():
    for make, game_ids, per_game in ((generate_train, TRAIN_GAMES, TRAIN_PER_GAME), (generate_eval, EVAL_GAMES, EVAL_PER_GAME)):
        rows = make()
        assert rows == make()
        assert {r["game_id"] for r in rows} == set(game_ids)
        # a game can fall one short when a type runs out of distinct questions
        assert all(per_game - 1 <= n <= per_game for n in Counter(r["game_id"] for r in rows).values())
        assert len({r["id"] for r in rows}) == len({r["question"] for r in rows}) == len(rows)
        json.dumps(rows)
        for r in rows:
            assert list(r) == FIELDS and r["question_type"] in MIX
            game = load_game(r["game_id"])
            home, away = game.teams.values()
            assert r["question"].startswith(f"Game {r['game_id']}: {home} vs {away} ({game.meta['date'][:10]}). ")
            if r["answer_type"] == "integer":
                assert type(r["answer"]) is int
            elif r["answer_type"] == "percentage":
                assert type(r["answer"]) is float and r["answer"] == round(r["answer"], 1)
            elif r["answer_type"] == "team":
                assert r["answer"] in (home, away)
            elif r["answer_type"] == "score":
                left, _, _, right = parse_score(r["answer"])
                assert (left, right) == (home, away)
            else:
                assert r["answer_type"] == "player" and r["answer"] in set(game.players["name"])
        # mix is within 2 points of the target shares
        counts = Counter(r["question_type"] for r in rows)
        for qtype, share in MIX.items():
            assert abs(counts[qtype] / len(rows) - share) <= 0.02


def test_train_and_eval_do_not_share_games():
    assert len(TRAIN_GAMES) == 8 and not set(TRAIN_GAMES) & set(EVAL_GAMES)


def test_eval_set_is_frozen():
    """eval.jsonl must stay the 40 questions the baseline is measured on."""
    on_disk = (HERE / "eval.jsonl").read_bytes()
    assert hashlib.md5(on_disk).hexdigest() == "f5f0dde2a1bfbfbf5ceafee7f7cac6bd"
    rows = [json.loads(line) for line in on_disk.decode().splitlines()]
    assert rows == generate_eval() and len(rows) == 40
    assert "score" not in {r["answer_type"] for r in rows}


def test_files_on_disk_are_current():
    train = [json.loads(line) for line in (HERE / "train.jsonl").read_text().splitlines()]
    assert train == generate_train()
    assert "score" in {r["answer_type"] for r in train}


def test_every_period_points_question_is_answerable_with_the_box_score_tool():
    for row in generate_train(with_meta=True) + generate_eval(with_meta=True):
        meta = row["meta"]
        if meta["subtype"] == "player_points_in":
            result = tools.get_box_score(row["game_id"], player=meta["player"], period=meta["period_num"])
            assert result["teams"][0]["players"][0]["PTS"] == row["answer"], row["id"]


def test_ground_truth_agrees_with_tools():
    for game_id in TRAIN_GAMES + EVAL_GAMES:
        game = load_game(game_id)
        for team_id, name in game.teams.items():
            truth, tool = team_stats(game, team_id), tools.get_team_stats(game_id, name)
            assert (truth["turnovers"], truth["rebounds"], truth["personal fouls"]) == (tool["TOV"], tool["REB"], tool["PF"])
            assert (truth["fg_pct"], truth["3p_pct"]) == (tool["FG%"], tool["3P%"])
            assert truth["picks"] == tools.get_picks(game_id, team=name)["picks"]
            made, attempts = contested_fg_line(game.shots[game.shots["offTeamId"] == team_id])
            assert pct(made, attempts) == tools.get_shots(game_id, team=name, contested=True)["fg_pct"]
