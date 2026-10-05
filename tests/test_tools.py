import json

import pytest

from tools import MAX_ITEMS, TOOL_SCHEMAS, TOOLS, get_box_score, get_picks, get_score_by_period, get_shots, get_team_stats

GAME = "191313"
BREOGAN, ZARAGOZA = "Leche Rio Breogan", "Casademont Zaragoza"


def check_small_json(result):
    """JSON-serializable, no error, and no list longer than MAX_ITEMS anywhere."""
    assert "error" not in result, result
    json.dumps(result)

    def walk(obj):
        if isinstance(obj, list):
            assert len(obj) <= MAX_ITEMS
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value)

    walk(result)
    return result


def test_get_score_by_period():
    r = check_small_json(get_score_by_period(GAME))
    assert r["periods"] == ["Q1", "Q2", "Q3", "Q4"] and r["overtime"] is False
    assert r["home"] == {"team": BREOGAN, "final": 94, "by_period": {"Q1": 26, "Q2": 21, "Q3": 23, "Q4": 24}}
    assert r["away"] == {"team": ZARAGOZA, "final": 95, "by_period": {"Q1": 24, "Q2": 20, "Q3": 23, "Q4": 28}}
    # the one overtime game
    ot = check_small_json(get_score_by_period(178442))
    assert ot["overtime"] is True and ot["periods"][-1] == "OT1"
    assert (ot["home"]["final"], ot["away"]["final"]) == (98, 102)


def test_get_box_score():
    r = check_small_json(get_box_score(GAME))
    assert [t["team"] for t in r["teams"]] == [BREOGAN, ZARAGOZA]
    assert [sum(p["PTS"] for p in t["players"]) for t in r["teams"]] == [94, 95]
    top = r["teams"][0]["players"][0]
    assert (top["player"], top["PTS"], top["FGM"], top["FGA"], top["3PM"], top["3PA"], top["FTM"], top["FTA"]) == (
        "Francisco Alonso", 33, 9, 15, 6, 9, 9, 10)
    assert (top["REB"], top["TOV"], top["PF"]) == (3, 2, 2)
    # fouled misses excluded: 141 shots - 10 fouled misses
    assert sum(p["FGA"] for t in r["teams"] for p in t["players"]) == 131

    one_team = check_small_json(get_box_score(GAME, team="zaragoza"))
    assert [t["team"] for t in one_team["teams"]] == [ZARAGOZA]
    one_player = check_small_json(get_box_score(GAME, player="Spissu"))
    assert one_player["teams"][0]["team"] == ZARAGOZA
    assert [(p["player"], p["PTS"]) for p in one_player["teams"][0]["players"]] == [("Marco Spissu", 20)]
    assert "error" in get_box_score(GAME, team="Breogan", player="Spissu")


def test_get_box_score_period():
    full = get_box_score(GAME)
    assert full["period"] == "full game"
    quarters = [check_small_json(get_box_score(GAME, period=p)) for p in (1, 2, 3, 4)]
    assert [q["period"] for q in quarters] == ["Q1", "Q2", "Q3", "Q4"]
    # team points per quarter match the score by period, so free throws are filtered too
    by_period = get_score_by_period(GAME)
    for i, side in enumerate(("home", "away")):
        points = [sum(p["PTS"] for p in q["teams"][i]["players"]) for q in quarters]
        assert points == list(by_period[side]["by_period"].values())
    # every stat adds up to the full-game line for each player
    for i in (0, 1):
        for row in full["teams"][i]["players"]:
            parts = [p for q in quarters for p in q["teams"][i]["players"] if p["player"] == row["player"]]
            for stat in ("PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "OREB", "DREB", "REB", "TOV", "PF"):
                assert sum(p[stat] for p in parts) == row[stat], (row["player"], stat)
    # a player's points in one quarter, free throws included
    alonso = [get_box_score(GAME, player="Alonso", period=p)["teams"][0]["players"][0] for p in (1, 2, 3, 4)]
    assert sum(q["PTS"] for q in alonso) == 33 and sum(q["FTM"] for q in alonso) == 9
    assert any(q["FTM"] > 0 for q in alonso)
    # the two eval questions that used to be unanswerable
    assert get_box_score(GAME, player="Jaime Fernandez Manzanares", period=1)["teams"][0]["players"][0]["PTS"] == 3
    ot = check_small_json(get_box_score("178442", player="Stefan Lazarevic", period=5))
    assert ot["period"] == "OT1" and ot["teams"][0]["players"][0]["PTS"] == 0
    assert "error" in get_box_score(GAME, period=5)


def test_get_team_stats():
    r = check_small_json(get_team_stats(GAME, "Breogan"))
    box = get_box_score(GAME, team="Breogan")["teams"][0]["players"]
    assert r["team"] == BREOGAN and r["PTS"] == 94
    for stat in ("FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "PF"):
        assert r[stat] == sum(p[stat] for p in box)
    assert r["FG%"] == round(100 * r["FGM"] / r["FGA"], 1)
    assert r["REB"] == r["OREB"] + r["DREB"] >= sum(p["REB"] for p in box)
    assert r["TOV"] >= sum(p["TOV"] for p in box)
    # quarters add up to the game
    quarters = [check_small_json(get_team_stats(GAME, "Breogan", period=p)) for p in (1, 2, 3, 4)]
    assert [q["PTS"] for q in quarters] == [26, 21, 23, 24]
    for stat in ("FGA", "REB", "TOV", "PF"):
        assert sum(q[stat] for q in quarters) == r[stat]
    assert "error" in get_team_stats(GAME, "Breogan", period=5)
    assert "error" in get_team_stats(GAME, "Lakers")


def test_get_shots():
    r = check_small_json(get_shots(GAME))
    assert (r["attempts"], r["makes"], r["points"], r["fouled_misses_excluded"]) == (131, 66, 157, 10)
    assert r["fg_pct"] == 50.4
    threes = check_small_json(get_shots(GAME, three_pt=True))
    twos = check_small_json(get_shots(GAME, three_pt=False))
    assert (threes["attempts"], threes["makes"], threes["points"]) == (57, 25, 75)
    assert (twos["attempts"], twos["makes"], twos["points"]) == (74, 41, 82)

    contested = check_small_json(get_shots(GAME, contested=True))
    uncontested = check_small_json(get_shots(GAME, contested=False))
    assert "contested_rule" in contested
    # one shot in this game has no contest label
    assert contested["no_contest_label_excluded"] == 1
    assert contested["makes"] + uncontested["makes"] == 66
    assert contested["attempts"] + uncontested["attempts"] == 131 - 1

    alonso = check_small_json(get_shots(GAME, player="Alonso", three_pt=True))
    assert alonso["filters"] == {"player": "Francisco Alonso", "three_pt": True}
    assert (alonso["makes"], alonso["attempts"]) == (6, 9)
    by_team = [get_shots(GAME, team=t)["points"] for t in ("Breogan", "Zaragoza")]
    assert sum(by_team) == 157
    assert "error" in get_shots(GAME, player="Smith")


def test_get_picks():
    r = check_small_json(get_picks(GAME))
    assert r["picks"] == 148 and r["possessions"] == 102 and r["direct_picks"] == 84
    assert {c["coverage"]: c["picks"] for c in r["by_ballhandler_coverage"]} == {"over": 92, "switch": 30, "under": 24, "blitz": 2}
    assert sum(c["picks"] for c in r["by_screener_coverage"]) == 148
    assert 0 < r["points"] <= 94 + 95
    teams = [check_small_json(get_picks(GAME, team=t)) for t in ("Breogan", "Zaragoza")]
    assert [t["picks"] for t in teams] == [76, 72]
    assert sum(t["points"] for t in teams) == r["points"]
    quarters = [check_small_json(get_picks(GAME, period=p)) for p in (1, 2, 3, 4)]
    assert sum(q["picks"] for q in quarters) == 148
    assert sum(q["points"] for q in quarters) == r["points"]


def test_schemas_match_tools():
    import inspect

    assert [s["name"] for s in TOOL_SCHEMAS] == list(TOOLS)
    for schema in TOOL_SCHEMAS:
        params = inspect.signature(TOOLS[schema["name"]]).parameters
        props = schema["input_schema"]["properties"]
        assert list(props) == list(params) and list(params)[0] == "game_id"
        required = [n for n, p in params.items() if p.default is inspect.Parameter.empty]
        assert schema["input_schema"]["required"] == required
        assert all(p.get("description") for p in props.values())
    json.dumps(TOOL_SCHEMAS)
