## Oracle: examples/jevman_oracle.ts

What it does: draws random legal game states (`randomGame`), asks each Pac-Man decision point for a value per option, and writes the rows. Each option gets an 8 s fallback rollout (the bench's `fallbackDecision` for the rest of the game). Value = score gain minus 5000 per lost life. It uses no rival data.

Bug and fix:
- Bug: the tile filter used `maze.openDirs` only. That checks neighbours, so wall tiles passed. 450 of 750 tiles were walls.
- Fix: `maze.isWalkable(t) && maze.openDirs(t).length`.
- Guard: `randomGame` throws if a placed Pac-Man or open-maze ghost tile is a wall.
- House ghosts are exempt. `HOUSE_TILES` are walls in the grid by design.

Checks:
- `--selftest` passes on the fix (50 random states, all placed tiles walkable).
- With the old filter (temporary copy), `--selftest` fails at the wall check.
- Determinism: two runs of 200 episodes, seed0=1, in 4 shards of 50. `cmp` reports identical output. 773 rows in each run.
