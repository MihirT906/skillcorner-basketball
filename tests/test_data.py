import json

import pytest

from data import (
    DATA_DIR,
    AmbiguousPlayerError,
    PlayerNotFoundError,
    TeamNotFoundError,
    box_score,
    load_game,
    resolve_player,
    resolve_team,
    score_by_period,
    team_score,
)

MATCHES = json.loads((DATA_DIR / "matches.json").read_text())


@pytest.fixture(scope="module")
def game():
    return load_game(191313)


def test_loads_all_event_families(game):
    assert len(game.events) == 20
    assert len(game.players) == 24
    assert len(game.shots) == 141
    assert len(game.free_throws) == 48


def test_load_game_is_cached(game):
    assert load_game(191313) is game
    assert load_game("191313") is game


def test_final_score(game):
    assert team_score(game, "Breogan") == 94
    assert team_score(game, "Zaragoza") == 95
    assert team_score(game, 1322) == 94
    assert team_score(game, 1331) == 95


def test_all_games_match_official_score():
    rows = []
    for m in MATCHES:
        g = load_game(m["id"])
        home = team_score(g, m["home_team"]["id"])
        away = team_score(g, m["away_team"]["id"])
        ok = (home, away) == (m["home_score"], m["away_score"])
        rows.append((m["id"], m["home_team"]["name"], m["away_team"]["name"],
                     f"{m['home_score']}-{m['away_score']}", f"{home}-{away}", "PASS" if ok else "FAIL"))
    print(f"\n{'gameId':<8}{'Home':<25}{'Away':<25}{'Official':<10}{'Computed':<10}Result")
    for r in rows:
        print(f"{r[0]:<8}{r[1]:<25}{r[2]:<25}{r[3]:<10}{r[4]:<10}{r[5]}")
    assert len(rows) == 10
    assert [r[0] for r in rows if r[5] == "FAIL"] == []


@pytest.mark.parametrize("match", MATCHES, ids=lambda m: str(m["id"]))
def test_score_by_period_and_box_score_sum_to_total(match):
    g = load_game(match["id"])
    periods = score_by_period(g)
    box = box_score(g)
    for team_id, team in g.teams.items():
        total = team_score(g, team_id)
        assert periods.loc[team, "total"] == total
        assert sum(team_score(g, team_id, period=p) for p in periods.columns.drop("total")) == total
        assert box.loc[box["team"] == team, "PTS"].sum() == total


def test_score_by_quarter(game):
    table = score_by_period(game)
    print("\n" + table.to_string())
    assert table.loc["Leche Rio Breogan"].tolist() == [26, 21, 23, 24, 94]
    assert table.loc["Casademont Zaragoza"].tolist() == [24, 20, 23, 28, 95]


def test_player_lookup(game):
    walker = game.player(63998)
    assert (walker["name"], walker["team"], walker["jersey"]) == ("Jordan Walker", "Leche Rio Breogan", "2")


def test_resolve_player():
    assert resolve_player("Walker", 191313) == 63998
    assert resolve_player("jordan walker", 191313) == 63998
    assert resolve_player("Spissu", 191313) == 59207
    assert resolve_player("Gonzales", 191313) == 59206  # typo for Gonzalez
    # same player found in a different game
    assert resolve_player("Spissu", 114234) == 59207
    with pytest.raises(PlayerNotFoundError):
        resolve_player("Smith", 191313)
    with pytest.raises(AmbiguousPlayerError):
        resolve_player("an", 191313)


def test_resolve_team():
    assert resolve_team("Zaragoza", 191313) == 1331
    assert resolve_team("breogan", 191313) == 1322
    assert resolve_team("Zaragoza", 114234) == 1331
    assert resolve_team("Real Madrid", 184439) == load_game(184439).meta["homeTeam"]["teamId"]
    with pytest.raises(TeamNotFoundError):
        resolve_team("Lakers", 191313)
    with pytest.raises(TeamNotFoundError):
        resolve_team("Zaragoza", 184439)


def test_every_team_resolves_by_full_name_and_last_word():
    for m in MATCHES:
        for side in ("home_team", "away_team"):
            name = m[side]["name"]
            assert resolve_team(name, m["id"]) == m[side]["id"]
            assert resolve_team(name.split()[-1], m["id"]) == m[side]["id"]


def test_box_score(game):
    box = box_score(game)
    for team_id, team in game.teams.items():
        team_box = box[box["team"] == team]
        print(f"\n{team} ({team_score(game, team_id)})")
        print(team_box.drop(columns="team").to_string(index=False))
        assert team_box["PTS"].sum() == team_score(game, team_id)
    # 141 shots minus 10 fouled misses
    assert box["FGA"].sum() == 131
    assert box["FGM"].sum() == 66
    assert box["FTA"].sum() == 48


def test_box_score_single_player(game):
    by_name = box_score(game, "Walker")
    assert len(by_name) == 1
    assert by_name.equals(box_score(game, 63998))
    assert by_name.equals(box_score(game).loc[[63998]])
