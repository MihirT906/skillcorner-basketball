# Schema notes: SkillCorner basketball, game 191313

Source: `opendata-basketball/docs` (PRIMER, data_dictionary, KNOWN_ISSUES) checked against
`data/matches/191313/191313_dynamic_events.json` and `191313_game_data.json`.
Game: Leche Rio Breogan (home, team 1322) 94, Casademont Zaragoza (away, team 1331) 95. Liga ACB, 2026-05-29.
All counts are for this one game.

## 1. Event families

`dynamic_events.json` is a dict of 20 lists. Every row has `id`, `gameId` (int), `season`, `period`,
`chanceId`, `possessionId` (except `possessions`, whose `id` is the possession id). Nearly every table
has `offTeamId` / `defTeamId`; exceptions are noted in the table.

Clock columns come in two shapes:
- **Point** events: `gameClock`, `frame`, `wallClock`.
- **Span** events: `startGameClock` / `endGameClock`, `startFrame` / `endFrame`, `startWallClock` / `endWallClock`.

| Family | Rows | Clock | Player / team fields | Outcome and other key fields |
|---|---|---|---|---|
| `possessions` | 153 | span | `offPlayerIds`, `defPlayerIds` (can be >5) | `startType`, `outcome`, `ptsScored`, `homeStartScore`, `awayStartScore`, `leftHoop` |
| `chances` | 205 | span | `offPlayerIds`, `defPlayerIds` (always 5 and 5) | `startType`, `outcome`, `ptsScored`, start scores, `startShotClock`, `endShotClock`, `endOfPossession`, `transition`, `lastChanceId`, `qualityIndex`, `usable` |
| `chance_players` | 2050 | none (period only) | `playerId`, `offense`, `startMatchupId`, `endMatchupId`; no team ids | `shotLoc`, `rimLoc`, `reboundLoc`, `minutesPlayedBefore` |
| `matchups` | 1477 | span | `offPlayerId`, `defPlayerId`; no team ids | `shotClock` |
| `shots` | 141 | span (release to end) | `shooterId`, `passerId`, `blockerId`, `closestDefId` | `outcome`, `three`, `fouled`, `foulId`, `assisted`, `assistOpp`, `blocked`, `shotType`, `complexShotType`, `contestLevel`, `distance`, `shotQuality`, `touchId` |
| `free_throws` | 48 | point | `shooterId` | `outcome`, `foulId`, `shotId` |
| `rebounds` | 75 | point | `rebounderId`, `teamId` (no off/def team ids) | `defensive`, `fgReb`, `rebounded`, `shotId`, `nextChanceId`, `touchId` |
| `turnovers` | 24 | point | `turnedOverId`, `stealerId`, `teamId` | `teamTo`, `touchId` |
| `fouls` | 49 | point | `foulerId`, `fouledId`, `foulerTeamId`, `fouledTeamId` | `shooting`, `foulType`, `isBonus`, `isTeam`, `isTechnical`, `touchId`, `shotId` |
| `timeouts` | 7 | point | `teamId` only | none |
| `passes` | 433 | span | `passerId`, `receiverId`, `toReceiverId` | `complete`, `turnover`, `ledToShot`, `assistOpp`, `inbounds`, `backcourt`, `distance`, `touchId` |
| `touches` | 618 | span | `playerId`, `defenderId` | `outcomes` (list), `numDribbles`, `touchTime`, `ledToShot`, `direct`, `regionsIn` |
| `dribbles` | 1353 | point | `ballhandlerId` | `location`, `region`, `touchId` |
| `picks` | 148 | point | `ballhandlerId`, `screenerId`, `ballhandlerDefId`, `screenerDefId` | `direct`, `bhrDefType`, `scrDefType`, `bhrOutcomes`, `scrOutcomes`, `locationType`, `handoffPick`, `touchId` |
| `handoffs` | 29 | point | `setterId`, `receiverId`, `setterDefId`, `receiverDefId` | `direct`, `setterDefType`, `receiverDefType`, `bhrOutcomes`, `handoffPick`, `touchId` |
| `off_ball_screens` | 87 | point | `cutterId`, `screenerId`, `cutterDefId`, `screenerDefId` | `direct`, `cutterDefType`, `screenerDefType`, `outcomes`, `scrOutcomes`, `ledToTouch`, `ledToShot`, `touchId`, `scrTouchId` |
| `drives` | 58 | span | `ballhandlerId`, `ballhandlerDefId` | `direct`, `category`, `endType`, `blowby`, `direction`, `bhrOutcomes`, `touchId` |
| `isolations` | 24 | span | `ballhandlerId`, `ballhandlerDefId` | `direct`, `bhrOutcomes`, `touchId` |
| `posts` | 4 | span | `ballhandlerId`, `ballhandlerDefId` | `direct`, `section`, `dribbledInto`, `bhrOutcomes`, `touchId` |
| `closeouts` | 64 | span, plus `touchGameClock` / `touchFrame` / `touchWallClock` | `ballhandlerId`, `ballhandlerDefId` | `direct`, `bhrAction`, `bhrOutcomes`, `startDistance`, `touchDistance`, `endDistance`, `touchId` |

Most tables also carry `location` ([x, y] in feet, origin at center court, offensive hoop at negative x)
and a `region` string. `shotClock` is on shots, passes, touches, matchups and all action markings except closeouts.

`game_data.json`: game, competition and season ids, plus `homeTeam` / `awayTeam`, each with `teamId`,
`teamName` and `players[]` of `{playerId, firstName, lastName, jersey}`. 12 players per team; no position field.

## 2. Clock convention

- `period` is an integer starting at 1. This game has periods 1 to 4 only; the docs do not say how
  overtime is numbered, and this game cannot confirm it.
- `gameClock` is **seconds remaining in the period**, as a float counting down from 600 (10-minute
  quarters) to 0. It stops on whistles.
- `wallClock` is milliseconds since the start of the video and never stops. `frame` is the 25 fps
  frame index, so `wallClock == frame * 40` (verified). Both match `tracking_data`.
- `shotClock` is seconds remaining on the shot clock.
- The period is also embedded in every id: `shot-191313-4-140` is period 4.

**Last 2 minutes of Q4:** `period == 4 and gameClock <= 120`.
- Point events: filter on `gameClock`.
- Span events: choose `startGameClock` or `endGameClock` deliberately. A chance can start above 120
  and end below it (15 chances start at or below 120; 16 end there).
- `chance_players` has no clock; join to `chances` on `chanceId`.
- In this game the window holds 7 shots, 11 free throws and 6 fouls.

Caveats: clock values are floats with noise (`1.919999980926491`), so compare with `<=`, never `==`.
Free throws in one trip share the same `gameClock`; order them by `frame`.

## 3. Shots: made vs missed, 2PT vs 3PT, fouled misses

All three are booleans on `shots`:
- **Made vs missed:** `outcome` (true = made). 66 made, 75 missed.
- **2PT vs 3PT:** `three` (true = 3PT). 83 twos, 58 threes. Do not derive this from `region`.
- **Fouled miss:** `fouled == true and outcome == false`. 10 rows (9 twos, 1 three).
- **And-one:** `fouled == true and outcome == true`. 3 rows.

`fouled` and `foulId` agree on every row (13 fouled shots, 13 non-null `foulId`, all pointing at a
foul with `shooting == true`).

Things to watch:
- **Box-score FGA** excludes fouled misses only: 141 - 10 = 131. KNOWN_ISSUES says to "exclude shots
  where `fouled` is true", which would also drop the 3 and-ones; the PRIMER wording (misses only) is the right one.
- **`chances.outcome` hides fouled shots.** Every fouled shot, made or missed, sits in a chance with
  `outcome == "FOU"`. `FGM` / `FGX` / `FGM3` / `FGX3` cover unfouled shots only, and here `FGM` / `FGX`
  are always twos despite the docs saying "any". Use `shots` for shooting stats.
- **2 shooting fouls have no `shots` row** (`foul-191313-2-21`, `foul-191313-2-22`). Each has 2 free
  throws with `shotId` null. So `shots.fouled` finds 13 fouled attempts but `fouls.shooting` counts 15.
- Free throws per fouled shot are mostly as expected (1 for an and-one, 2 or 3 for a miss), but one
  fouled missed two has 1 free throw and another has 3.
- `free_throws.outcome` is also a boolean (32 of 48 made).
- Points from `shots` + `free_throws` give 94 and 95, matching the final score. `chances.ptsScored`
  summed by `offTeamId` also matches in this game, but the docs warn it fails in about half of games.

## 4. Joins

- **Ids are strings** shaped `{family}-{gameId}-{period}-{seq}`, e.g. `chance-191313-4-204`.
- **`chanceId`** resolves to `chances.id` for every row in every table. **`possessionId`** resolves to
  `possessions.id`, and a child row's `possessionId` always equals its chance's `possessionId`.
- **Possession to chance** is 1 to many: 111 possessions have 1 chance, 33 have 2, 8 have 3, 1 has 4.
  An offensive rebound starts a new chance in the same possession. `chances.endOfPossession` marks the last one.
- **Chance to shot** is at most 1 to 1: all 141 shots sit in distinct chances.
- **Chance chaining:** `chances.lastChanceId` (previous chance) and `rebounds.nextChanceId`.
- **Player ids** are integers and join to `homeTeam.players[].playerId` / `awayTeam.players[].playerId`.
  Every player id in every table is in the roster (24 rostered, 21 appear). Team ids join to
  `homeTeam.teamId` / `awayTeam.teamId`.
- **On-court lineup:** `chances.offPlayerIds` / `defPlayerIds`, or `chance_players` (10 rows per chance).
- **`touchId`** links shots, passes, dribbles, rebounds, turnovers, fouls and all action markings to `touches.id`.
- **Shot, foul, free throw:** `shots.foulId` to `fouls.id`; `free_throws.foulId` to `fouls.id`;
  `free_throws.shotId` to `shots.id`. Do not use `fouls.shotId` (always null).
- **Rebounds:** `rebounds.shotId` points to a `shots.id` (61) or a `free_throws.id` (11); check the prefix.

## 5. Noisy or mostly-null fields

Always null or constant:
- `fouls.shotId`: null on all 49 rows.
- `fouls.isTeam`, `fouls.isTechnical`: always false. `foulType` is only `shooting` or `other`.
- `drives.dribbleThrough`: always false. `posts.dribbledInto`: always false (4 rows).

Mostly null by design:
- `passes.toReceiverId`: 424 of 433 null. When set it is the **defender who intercepted**, not the intended receiver.
- `shots.blockerId` (136 of 141), `shots.foulId` (128 of 141), `free_throws.shotId` (24 of 48),
  `turnovers.stealerId` (10 of 24).
- `chance_players.reboundLoc` (64% null), `shotLoc` / `rimLoc` (31%), `startMatchupId` / `endMatchupId` (8%).
- `off_ball_screens.scrTouchId` (82 of 87 null), `touchId` (35 of 87), `scrOutcomes` (empty on 77 of 87).

Partly null:
- `shotClock`: null on 20% of touches and 19% of passes.
- `touchId`: null on 19 of 75 rebounds, 11 of 24 turnovers, 8 of 49 fouls, 44 of 1353 dribbles, 6 of 141 shots.
- `passes.receiverLoc` / `receiverRegion` / `distance` (25 null), `receiverId` (16), `endFrame` / `complete` (7).
- `shots`: one row (`shot-191313-3-82`) has null location, region, distance, shot type, contest and
  quality fields. `dribblesBefore` has 6 nulls; `releaseTime` and `catchAndShoot` have 5.
- `rebounds`: `shotId` null on 3, `rebounderId` null on 4 (team rebounds).

Marking noise:
- 14 missed field goals have no `rebounds` row; 10 of those are fouled misses, leaving 4 unexplained.
- 10 passes have `passerId == receiverId`. 2 shots are `assisted` without `assistOpp`.
- 3 chances have zero elapsed game clock, and `chance-191313-4-203` ends before it starts (`endFrame < startFrame`).
- 13 chances have `usable == false` (`qualityIndex < 3`), including 3 made field goals and 5 foul
  chances. Filtering on `usable` drops real points, so only apply it to tracking-derived context.
- One free throw (`free_throw-191313-2-11`) is in a different chance from its foul.
- One pass points at a `touchId` that does not exist (`touch-191313-4-617`).

Docs vs data:
- `closeouts.touchWallClock` is a string; cast before arithmetic.
- Typed as string in the docs but integer in the data: `shots.passerId`, `shots.closestDefId`,
  `turnovers.stealerId`, `timeouts.teamId`, `chances.gameId`.
- `chances.possessionId` is documented as integer but is a string; `possessions.outcome` as a list but is a string.
- Undocumented values: `drives.endType == "stoppage"`, `ASTO` in touch `outcomes`, `handoffs.setterDefType == "show"`.
- Assists and rebounds are tracking-derived judgment calls and differ from the official scorer more than shots do.
- Player ids can change between games for 9 players; use `data/player_id_aliases.csv` for multi-game work.
