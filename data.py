"""Data layer for the SkillCorner basketball open data: load a game, look up players, compute scores."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process, utils

DATA_DIR = Path(__file__).resolve().parent.parent / "opendata-basketball" / "data"

# resolve_player thresholds (rapidfuzz scores, 0-100)
MIN_SCORE = 75
AMBIGUITY_MARGIN = 5


class PlayerNotFoundError(LookupError):
    pass


class AmbiguousPlayerError(LookupError):
    pass


class TeamNotFoundError(LookupError):
    pass


class AmbiguousTeamError(LookupError):
    pass


@lru_cache(maxsize=None)
def load_aliases() -> dict[int, int]:
    """player_id -> canonical_player_id, for the players who appear under two ids."""
    aliases = pd.read_csv(DATA_DIR / "player_id_aliases.csv")
    return dict(zip(aliases["player_id"], aliases["canonical_player_id"]))


def _player_id_columns(df: pd.DataFrame) -> list[str]:
    # Player ids are the only numeric *Id columns besides gameId and the team ids;
    # chanceId, touchId, shotId etc. are strings.
    return [
        c
        for c in df.columns
        if c.endswith("Id")
        and c != "gameId"
        and "team" not in c.lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]


@dataclass
class Game:
    game_id: int
    meta: dict
    teams: dict[int, str]  # teamId -> teamName, home first
    players: pd.DataFrame  # index: canonical playerId; columns: name, firstName, lastName, jersey, teamId, team
    events: dict[str, pd.DataFrame]  # one DataFrame per event family
    aliases: dict[int, int]

    def __getattr__(self, name):
        # game.shots, game.free_throws, ... as shorthand for game.events[...]
        events = self.__dict__.get("events", {})
        if name in events:
            return events[name]
        raise AttributeError(name)

    def player(self, player_id: int) -> pd.Series:
        """name / team / jersey for a player id (raw or canonical)."""
        return self.players.loc[self.aliases.get(player_id, player_id)]


@lru_cache(maxsize=None)
def _load_game(game_id: int) -> Game:
    game_dir = DATA_DIR / "matches" / str(game_id)
    if not game_dir.is_dir():
        raise FileNotFoundError(f"No data for game {game_id} in {game_dir.parent}")
    meta = json.loads((game_dir / f"{game_id}_game_data.json").read_text())
    raw_events = json.loads((game_dir / f"{game_id}_dynamic_events.json").read_text())
    aliases = load_aliases()

    teams = {meta[side]["teamId"]: meta[side]["teamName"] for side in ("homeTeam", "awayTeam")}
    players = pd.DataFrame(
        [
            {**p, "teamId": meta[side]["teamId"], "team": meta[side]["teamName"]}
            for side in ("homeTeam", "awayTeam")
            for p in meta[side]["players"]
        ]
    )
    players["playerId"] = players["playerId"].replace(aliases)
    players["name"] = players["firstName"] + " " + players["lastName"]
    players = players.set_index("playerId")

    events = {}
    for family, rows in raw_events.items():
        df = pd.DataFrame(rows)
        for col in _player_id_columns(df):
            df[col] = df[col].astype("Int64").replace(aliases)
        for col in ("offPlayerIds", "defPlayerIds"):
            if col in df:
                df[col] = df[col].map(lambda ids: [aliases.get(i, i) for i in ids])
        events[family] = df

    return Game(game_id=game_id, meta=meta, teams=teams, players=players, events=events, aliases=aliases)


def load_game(game_id) -> Game:
    """Load a game (cached: repeated calls return the same Game object)."""
    return _load_game(int(game_id))


def _fuzzy_match(query: str, names: dict[int, str], what: str, game_id: int, error) -> list:
    """Matches as (name, score, id), keeping only those within AMBIGUITY_MARGIN of the best."""
    matches = process.extract(
        query, names, scorer=fuzz.WRatio, processor=utils.default_process, score_cutoff=MIN_SCORE, limit=None
    )
    if not matches:
        raise error(f"No {what} matching {query!r} in game {game_id}")
    return [m for m in matches if matches[0][1] - m[1] <= AMBIGUITY_MARGIN]


def resolve_player(name: str, game_id) -> int:
    """Fuzzy-match a full or partial player name to a player id in that game."""
    game = load_game(game_id)
    close = _fuzzy_match(name, game.players["name"].to_dict(), "player", game.game_id, PlayerNotFoundError)
    if len(close) > 1:
        options = ", ".join(
            f"{n} (#{game.players.at[pid, 'jersey']}, {game.players.at[pid, 'team']})" for n, _, pid in close
        )
        raise AmbiguousPlayerError(f"{name!r} is ambiguous in game {game.game_id}: {options}")
    return int(close[0][2])


def resolve_team(name: str, game_id) -> int:
    """Fuzzy-match a full or partial team name to a team id in that game."""
    game = load_game(game_id)
    close = _fuzzy_match(name, game.teams, "team", game.game_id, TeamNotFoundError)
    if len(close) > 1:
        raise AmbiguousTeamError(f"{name!r} is ambiguous in game {game.game_id}: {', '.join(n for n, _, _ in close)}")
    return int(close[0][2])


def _team_id(game: Game, team) -> int:
    if isinstance(team, str):
        return resolve_team(team, game.game_id)
    if team not in game.teams:
        raise TeamNotFoundError(f"Team id {team} did not play in game {game.game_id}")
    return int(team)


def scoring_events(game: Game) -> pd.DataFrame:
    """One row per made shot / made free throw: offTeamId, period, possessionId, pts."""
    shots, fts = game.shots, game.free_throws
    shots = shots[shots["outcome"]]
    fts = fts[fts["outcome"]]
    cols = ["offTeamId", "period", "possessionId"]
    return pd.concat(
        [
            shots[cols].assign(pts=shots["three"].map({True: 3, False: 2})),
            fts[cols].assign(pts=1),
        ]
    )


def team_score(game: Game, team, period=None) -> int:
    """Points from shots + free_throws only (chances.ptsScored is unreliable, see KNOWN_ISSUES).

    `team` is a team id or a (partial) team name.
    """
    points = scoring_events(game)
    points = points[points["offTeamId"] == _team_id(game, team)]
    if period is not None:
        points = points[points["period"] == period]
    return int(points["pts"].sum())


def score_by_period(game: Game) -> pd.DataFrame:
    """Points per team (rows, home first) per period (columns), plus a `total` column."""
    points = scoring_events(game)
    table = points.pivot_table(index="offTeamId", columns="period", values="pts", aggfunc="sum", fill_value=0)
    table = table.reindex(list(game.teams), fill_value=0).astype(int)
    table.index = pd.Index(game.teams.values(), name="team")
    table["total"] = table.sum(axis=1)
    return table


def box_score(game: Game, player=None) -> pd.DataFrame:
    """Per-player box score, sorted by points. `player` (id or name) limits it to one player.

    FGA / 3PA follow the box-score convention and exclude fouled misses.
    Team rebounds and team turnovers have no player and are not counted.
    """
    shots, fts, rebs = game.shots, game.free_throws, game.rebounds
    made = shots[shots["outcome"]]
    attempts = shots[shots["outcome"] | ~shots["fouled"]]
    rebs = rebs[rebs["rebounded"]]

    def count(df, col):
        return df.groupby(col).size()

    box = game.players[["name", "team", "jersey"]].copy()
    stats = {
        "FGM": count(made, "shooterId"),
        "FGA": count(attempts, "shooterId"),
        "3PM": count(made[made["three"]], "shooterId"),
        "3PA": count(attempts[attempts["three"]], "shooterId"),
        "FTM": count(fts[fts["outcome"]], "shooterId"),
        "FTA": count(fts, "shooterId"),
        "OREB": count(rebs[~rebs["defensive"]], "rebounderId"),
        "DREB": count(rebs[rebs["defensive"]], "rebounderId"),
        "TOV": count(game.turnovers, "turnedOverId"),
        "PF": count(game.fouls, "foulerId"),
    }
    for col, values in stats.items():
        box[col] = values.reindex(box.index, fill_value=0).astype(int)
    box["REB"] = box["OREB"] + box["DREB"]
    box["PTS"] = 2 * box["FGM"] + box["3PM"] + box["FTM"]
    box = box[["name", "team", "jersey", "PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "OREB", "DREB", "REB", "TOV", "PF"]]

    if player is not None:
        player_id = resolve_player(player, game.game_id) if isinstance(player, str) else game.aliases.get(player, player)
        return box.loc[[player_id]]
    return box.sort_values("PTS", ascending=False, kind="stable")
