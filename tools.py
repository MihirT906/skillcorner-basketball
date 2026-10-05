"""LLM-callable tools over the SkillCorner basketball data.

Every tool takes game_id first, speaks in player/team names (not ids), and returns a
small JSON-serializable dict. Bad input (unknown game, team, player, period) comes back
as {"error": "..."} so the model can read it and retry.
"""

import functools

import pandas as pd

from data import load_game, resolve_player, resolve_team, scoring_events

MAX_ITEMS = 20

CONTESTED_RULE = (
    "A shot is 'contested' when SkillCorner's contestLevel is average, plus or blocked, and "
    "'uncontested' when it is open or light (this is the dataset's own `contested` flag). "
    "It is a model judgment from broadcast tracking, not defender distance: about 70% of shots "
    "are flagged contested, it separates 2-point shots well but barely separates 3-point shots, "
    "and about 2% of shots have no contest label and are left out when this filter is used. "
    "Treat it as approximate."
)


def _tool(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (LookupError, FileNotFoundError, ValueError) as e:
            return {"error": str(e.args[0]) if e.args else type(e).__name__}

    return wrapper


def _period_label(period: int) -> str:
    return f"Q{period}" if period <= 4 else f"OT{period - 4}"


def _check_period(game, period):
    if period is None:
        return None
    periods = sorted(int(p) for p in game.chances["period"].unique())
    if isinstance(period, bool) or not isinstance(period, int) or period not in periods:
        raise ValueError(f"period must be one of {periods} for game {game.game_id} (1-4 = quarters, 5+ = overtime)")
    return period


def _pct(made: int, attempted: int):
    return round(100 * made / attempted, 1) if attempted else None


def _shooting(shots: pd.DataFrame, fts: pd.DataFrame) -> dict:
    """Box-score shooting line. FGA / 3PA exclude fouled misses."""
    made = shots[shots["outcome"]]
    attempts = shots[shots["outcome"] | ~shots["fouled"]]
    line = {
        "FGM": len(made),
        "FGA": len(attempts),
        "3PM": int(made["three"].sum()),
        "3PA": int(attempts["three"].sum()),
        "FTM": int(fts["outcome"].sum()),
        "FTA": len(fts),
    }
    return {"PTS": 2 * line["FGM"] + line["3PM"] + line["FTM"], **line}


@_tool
def get_score_by_period(game_id):
    game = load_game(game_id)
    points = scoring_events(game)
    periods = sorted(int(p) for p in game.chances["period"].unique())
    out = {
        "game_id": game.game_id,
        "date": game.meta["date"][:10],
        "periods": [_period_label(p) for p in periods],
        "overtime": max(periods) > 4,
    }
    for side, team_id in zip(("home", "away"), game.teams):
        by_period = points[points["offTeamId"] == team_id].groupby("period")["pts"].sum()
        scores = {_period_label(p): int(by_period.get(p, 0)) for p in periods}
        out[side] = {"team": game.teams[team_id], "final": sum(scores.values()), "by_period": scores}
    return out


@_tool
def get_box_score(game_id, team=None, player=None, period=None):
    game = load_game(game_id)
    period = _check_period(game, period)
    team_ids = [resolve_team(team, game.game_id)] if team else list(game.teams)
    player_id = resolve_player(player, game.game_id) if player else None
    if player_id is not None:
        player_team = int(game.players.at[player_id, "teamId"])
        if player_team not in team_ids:
            raise LookupError(f"{game.players.at[player_id, 'name']} plays for {game.teams[player_team]}, not {game.teams[team_ids[0]]}")
        team_ids = [player_team]

    def in_period(df):
        return df if period is None else df[df["period"] == period]

    # Every stat, free throws included, is limited to the period when one is given.
    shots, fts = in_period(game.shots), in_period(game.free_throws)
    turnovers, fouls = in_period(game.turnovers), in_period(game.fouls)
    played = set(in_period(game.chance_players)["playerId"])
    rebs = in_period(game.rebounds[game.rebounds["rebounded"]])
    teams = []
    for team_id in team_ids:
        roster = game.players[game.players["teamId"] == team_id]
        rows = []
        for pid, info in roster.iterrows():
            if pid not in played or (player_id is not None and pid != player_id):
                continue
            mine = rebs[rebs["rebounderId"] == pid]
            oreb = int((~mine["defensive"]).sum())
            rows.append(
                {
                    "player": info["name"],
                    "number": info["jersey"],
                    **_shooting(shots[shots["shooterId"] == pid], fts[fts["shooterId"] == pid]),
                    "OREB": oreb,
                    "DREB": len(mine) - oreb,
                    "REB": len(mine),
                    "TOV": int((turnovers["turnedOverId"] == pid).sum()),
                    "PF": int((fouls["foulerId"] == pid).sum()),
                }
            )
        rows.sort(key=lambda r: -r["PTS"])
        teams.append({"team": game.teams[team_id], "players": rows[:MAX_ITEMS]})
    return {
        "game_id": game.game_id,
        "period": _period_label(period) if period else "full game",
        "teams": teams,
        "note": "Only players who appeared (in that period, if one is given). FGA and 3PA exclude fouled misses. "
        "Team rebounds and team turnovers are not credited to any player; use get_team_stats for team totals.",
    }


@_tool
def get_team_stats(game_id, team, period=None):
    game = load_game(game_id)
    team_id = resolve_team(team, game.game_id)
    period = _check_period(game, period)

    def rows(family, team_col):
        df = game.events[family]
        df = df[df[team_col] == team_id]
        return df if period is None else df[df["period"] == period]

    line = _shooting(rows("shots", "offTeamId"), rows("free_throws", "offTeamId"))
    rebs = rows("rebounds", "teamId")
    oreb = int((~rebs["defensive"]).sum())
    return {
        "game_id": game.game_id,
        "team": game.teams[team_id],
        "period": _period_label(period) if period else "full game",
        **line,
        "FG%": _pct(line["FGM"], line["FGA"]),
        "3P%": _pct(line["3PM"], line["3PA"]),
        "FT%": _pct(line["FTM"], line["FTA"]),
        "OREB": oreb,
        "DREB": len(rebs) - oreb,
        "REB": len(rebs),
        "TOV": len(rows("turnovers", "teamId")),
        "PF": len(rows("fouls", "foulerTeamId")),
        "note": "FGA and 3PA exclude fouled misses. REB and TOV include team rebounds and team turnovers, "
        "so they can exceed the sum of the player box score.",
    }


@_tool
def get_shots(game_id, team=None, player=None, period=None, three_pt=None, contested=None):
    game = load_game(game_id)
    period = _check_period(game, period)
    shots = game.shots
    filters = {}
    if team:
        team_id = resolve_team(team, game.game_id)
        shots = shots[shots["offTeamId"] == team_id]
        filters["team"] = game.teams[team_id]
    if player:
        player_id = resolve_player(player, game.game_id)
        shots = shots[shots["shooterId"] == player_id]
        filters["player"] = game.players.at[player_id, "name"]
    if period is not None:
        shots = shots[shots["period"] == period]
        filters["period"] = _period_label(period)
    if three_pt is not None:
        shots = shots[shots["three"] == bool(three_pt)]
        filters["three_pt"] = bool(three_pt)

    out = {"game_id": game.game_id, "filters": filters}
    if contested is not None:
        out["no_contest_label_excluded"] = int(shots["contested"].isna().sum())
        shots = shots[shots["contested"].notna() & (shots["contested"] == bool(contested))]
        filters["contested"] = bool(contested)

    made = shots[shots["outcome"]]
    fouled_misses = int((shots["fouled"] & ~shots["outcome"]).sum())
    attempts = len(shots) - fouled_misses
    out.update(
        attempts=attempts,
        makes=len(made),
        fg_pct=_pct(len(made), attempts),
        points=int(2 * len(made) + made["three"].sum()),
        fouled_misses_excluded=fouled_misses,
        note="Field goals only: points exclude free throws, attempts exclude fouled misses.",
    )
    if contested is not None:
        out["contested_rule"] = CONTESTED_RULE
    return out


@_tool
def get_picks(game_id, team=None, period=None):
    game = load_game(game_id)
    period = _check_period(game, period)
    picks = game.picks
    filters = {}
    if team:
        team_id = resolve_team(team, game.game_id)
        picks = picks[picks["offTeamId"] == team_id]
        filters["team"] = game.teams[team_id]
    if period is not None:
        picks = picks[picks["period"] == period]
        filters["period"] = _period_label(period)

    # Points by the offense on each possession, from made shots + free throws.
    points = scoring_events(game).groupby(["possessionId", "offTeamId"])["pts"].sum()

    def summarize(df: pd.DataFrame) -> dict:
        possessions = df[["possessionId", "offTeamId"]].drop_duplicates()
        keys = pd.MultiIndex.from_frame(possessions)
        return {
            "picks": len(df),
            "possessions": len(possessions),
            "points": int(points.reindex(keys, fill_value=0).sum()),
        }

    def breakdown(col: str) -> list:
        rows = [{"coverage": cov, **summarize(df)} for cov, df in picks.groupby(col)]
        return sorted(rows, key=lambda r: -r["picks"])[:MAX_ITEMS]

    total = summarize(picks)
    return {
        "game_id": game.game_id,
        "filters": filters,
        **total,
        "points_per_possession": round(total["points"] / total["possessions"], 2) if total["possessions"] else None,
        "direct_picks": int(picks["direct"].sum()),
        "by_ballhandler_coverage": breakdown("bhrDefType"),
        "by_screener_coverage": breakdown("scrDefType"),
        "note": "A pick is any on-ball screen (pick-and-roll or pick-and-pop). Points are everything the offense "
        "scored on a possession containing at least one pick, whether or not the pick led to the score. "
        "A possession with several picks is counted once in the totals but under each coverage it faced, "
        "so coverage rows can add up to more than the totals. direct_picks led straight to a shot, assist or foul.",
    }


_GAME_ID = {
    "type": "string",
    "description": "SkillCorner game id, e.g. \"191313\". One of the 10 published ACB 2025-2026 games.",
}
_TEAM = "Team name, full or partial (e.g. \"Zaragoza\" or \"Leche Rio Breogan\"). Must be one of the two teams in the game."
_PLAYER = "Player name, full or partial (e.g. \"Spissu\" or \"Marco Spissu\"). Must be on one of the two rosters; an ambiguous name returns an error listing the candidates."
_PERIOD = {
    "type": "integer",
    "minimum": 1,
    "description": "Period to restrict to: 1, 2, 3 or 4 for the quarters, 5 for the first overtime, 6 for a second overtime. Omit for the full game.",
}

TOOL_SCHEMAS = [
    {
        "name": "get_score_by_period",
        "description": "Final score and points per quarter (and overtime, if played) for both teams in a game. "
        "Use this for the result, who won, and how the score developed.",
        "input_schema": {
            "type": "object",
            "properties": {"game_id": _GAME_ID},
            "required": ["game_id"],
        },
    },
    {
        "name": "get_box_score",
        "description": "Per-player box score: PTS, FGM/FGA, 3PM/3PA, FTM/FTA, OREB/DREB/REB, TOV, PF, grouped by team "
        "and sorted by points. Only players who appeared are listed. FGA and 3PA exclude shots missed while fouled. "
        "With `period`, every stat (free throws included) covers that period only, so this is the tool for a "
        "player's points, rebounds or fouls in a single quarter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "game_id": _GAME_ID,
                "team": {"type": "string", "description": _TEAM + " Omit for both teams."},
                "player": {"type": "string", "description": _PLAYER + " Omit for all players."},
                "period": _PERIOD,
            },
            "required": ["game_id"],
        },
    },
    {
        "name": "get_team_stats",
        "description": "Team totals for a game or a single period: PTS, FGM/FGA and FG%, 3PM/3PA and 3P%, FTM/FTA and FT%, "
        "OREB/DREB/REB, TOV, PF. Percentages are 0-100. FGA and 3PA exclude shots missed while fouled. "
        "REB and TOV include team rebounds and team turnovers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "game_id": _GAME_ID,
                "team": {"type": "string", "description": _TEAM},
                "period": _PERIOD,
            },
            "required": ["game_id", "team"],
        },
    },
    {
        "name": "get_shots",
        "description": "Field goal attempts, makes, FG% (0-100) and points for the shots matching the filters. "
        "All filters are optional and combine with AND. Free throws are not included, and attempts exclude shots "
        "missed while fouled. " + CONTESTED_RULE,
        "input_schema": {
            "type": "object",
            "properties": {
                "game_id": _GAME_ID,
                "team": {"type": "string", "description": _TEAM + " Filters to shots taken by that team."},
                "player": {"type": "string", "description": _PLAYER + " Filters to shots taken by that player."},
                "period": _PERIOD,
                "three_pt": {
                    "type": "boolean",
                    "description": "true = 3-point shots only, false = 2-point shots only. Omit for both.",
                },
                "contested": {
                    "type": "boolean",
                    "description": "true = contested shots only (contest level average, plus or blocked), "
                    "false = uncontested only (open or light). Omit for all shots. Approximate; see the tool description.",
                },
            },
            "required": ["game_id"],
        },
    },
    {
        "name": "get_picks",
        "description": "On-ball screens (pick-and-rolls and pick-and-pops): how many, on how many possessions, the points the "
        "offense scored on those possessions, and a breakdown by defensive coverage. Coverage is given twice: on the "
        "ball handler (over, under, switch, ice, blitz) and on the screener (show, soft, switch, ice, blitz, none). "
        "Points count the whole possession, not only scores created by the pick.",
        "input_schema": {
            "type": "object",
            "properties": {
                "game_id": _GAME_ID,
                "team": {"type": "string", "description": _TEAM + " Filters to picks run by that team on offense. Omit for both teams."},
                "period": _PERIOD,
            },
            "required": ["game_id"],
        },
    },
]

TOOLS = {
    "get_score_by_period": get_score_by_period,
    "get_box_score": get_box_score,
    "get_team_stats": get_team_stats,
    "get_shots": get_shots,
    "get_picks": get_picks,
}
