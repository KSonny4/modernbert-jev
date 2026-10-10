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

## Run: ModernBERT Jevman model

Issue: KSonny4/modernbert-jev#1

What it does: trains a ModernBERT-base NLI checkpoint on oracle labels, then runs the bench on the trained model.

Data:
- 8000 random legal episodes, seed0=100000. Generated in 13 shards on the Windows box (RX 7900 XT, WSL). Time: 36 min.
- Rows: 30,505.
- No rival or leaderboard data.

Train:
- Command: `train.py --data jevman` at commit fc5f683 (train-jevman, tied-best targets).
- Start checkpoint: v2 ModernBERT-base NLI (jew-v2-overnight-ckpt).
- Settings: 6000 steps, batch 1, lr 1e-5, premise "instructions", PMI alpha 0.
- Split: 27,428 train rows. 3,077 held-out rows (10% of episodes, split seed 7).
- Time: 859 s on GPU.

Held-out results:
- Top-1 agreement with the teacher: 0.342 before, 0.501 after. Chance is about 0.33 (mean 3.04 options).
- DANGER/TRAP share of picks: 0.382 before, 0.239 after.
- NLI sanity: PASS.

Bench (`jevman npm run bench`, 8 games, parallel 1, quint8 ONNX, 4 threads, endpoint `--premise instructions --no-pmi`):
- Mean score 2726 (range 2220–3770).
- Survived 64.3 s. Pellets 205 (68 per life). Ghosts 1.5. Fruit 14/16.
- Fallbacks 3.2%. Latency 285 ms.
- All 8 games end with 3 deaths on level 1.

Comparison:
- Old model: 413.
- Example rule: 1144.
- Rival-data probe: 1825 (not for submission).

Note: The bench start position is fixed and deterministic. The teacher never used bench recordings. It saw only random positions.

## Jevman oracle: rollout policy and margin flags (issue KSonny4/modernbert-jev#1)

- `examples/jevman_oracle.ts` has three new flags. `--rollout greedy` uses the bench's `scriptedControls` (from `scripts/game.ts`) after the first move. `--horizon <s>` sets the rollout length (default 8). `--margin <points>` skips a row when the best value leads the runner-up by less than the margin. Defaults keep the old output.
- Timing, 100 episodes, seed0 900000, busy Mac. Defaults: 382 rows, 168 s (2.3 rows/s). `--rollout greedy --horizon 12 --margin 300`: 198 rows, 243 s (0.8 rows/s), 179 rows skipped by margin.
- Of 183 aligned rows (same game and frame), 37% changed `best`. The forward play uses the oracle, so the states drift after the first change. Treat this as a rough figure.
- Selftest checks that a greedy rollout returns finite values.
