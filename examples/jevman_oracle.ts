// Offline teacher for modernbert-jev: labels Pac-Man decisions on random game positions by rollout.
// The oracle never plays in the benchmark. Rows have the extractor's format (see jevman_extract.ts) plus `best` and `values`.
// Run from the bench dir:
//   node --import tsx <this file> <bench_dir> <out.jsonl> <n_episodes> <seed0>
//   node --import tsx <this file> --selftest <bench_dir>
// Flags: --rollout fallback|greedy (default fallback), --horizon <seconds> (default 8), --margin <points> (default 0).
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

type J = any;
const DT = 0.0167; // roundDt(1 / 60): the bench's FIXED_STEP, the normal frame dt
const DEATH_COST = 5000;
const MAX_DECISIONS = 5;
const SUBMISSION = 'oracle-rand';

type Rollout = 'fallback' | 'greedy';
const CFG: { rollout: Rollout; horizon: number; margin: number } = { rollout: 'fallback', horizon: 8, margin: 0 };

let B: J; // the bench's modules, loaded once

async function loadBench(dir: string) {
  const at = (p: string) => import(pathToFileURL(join(dir, p)).href);
  const [sim, scheduler, brain, features, maze, layout, types, game] = await Promise.all([
    at('src/sim.ts'), at('src/scheduler.ts'), at('src/brain.ts'), at('src/features.ts'), at('src/maze.ts'), at('src/layout.ts'), at('src/types.ts'),
    at('scripts/game.ts'),
  ]);
  return { sim, Scheduler: scheduler.Scheduler as J, brain, features, maze, layout, types, scriptedControls: game.scriptedControls as () => J };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A copy that keeps Maze's methods; the pellet sets are copied, the walls and distance cache are shared (never changed). */
function cloneState(s: J): J {
  const { maze, ...rest } = s;
  const copy = Object.assign(Object.create(Object.getPrototypeOf(maze)), maze, {
    pellets: new Set(maze.pellets),
    powerPellets: new Set(maze.powerPellets),
  });
  return { ...structuredClone(rest), maze: copy };
}

/**
 * The policy for Pac-Man's decisions after the first move: the bench's fallbackDecision, or its greedy controller
 * (scriptedControls, the one behind `--pacman greedy`). One instance per rollout, as the bench keeps one per game.
 */
function restPolicy(): (point: J, st: J) => string {
  if (CFG.rollout === 'greedy') {
    const scripted = B.scriptedControls();
    return (point, st) => scripted.decide(point, st);
  }
  const memo = new Map<string, string>();
  return (point, st) => {
    if (!memo.has(point.key)) {
      const features = B.features.optionFeatures(st, point);
      memo.set(point.key, B.brain.fallbackDecision(st, { point, features }, 'oracle').choice);
    }
    return memo.get(point.key);
  };
}

/**
 * The bench's rule for Pac-Man and for the scripted ghosts (main.ts). The forced answer holds until Pac-Man's next
 * junction: his escape questions on that corridor get the same direction, else the escape rule would turn him back at once.
 */
function controls(forceKey: string, force: string, rest: (point: J, st: J) => string): J {
  let committed = false;
  return {
    decide(point: J, st: J) {
      if (point.actor !== 'pacman') return B.features.greedyChoice(st, point, B.features.optionFeatures(st, point));
      if (point.key === forceKey) {
        committed = true;
        return force;
      }
      if (committed && point.escape) return force;
      committed = false;
      return rest(point, st);
    },
  };
}

/** Score gained in the rollout, minus DEATH_COST when Pac-Man loses a life (the rollout stops there). */
function rollout(state: J, point: J, option: string): number {
  const s = cloneState(state);
  const score0 = s.score;
  const lives0 = s.lives;
  const ctl = controls(point.key, option, restPolicy());
  let died = false;
  for (let i = 0; i < Math.ceil(CFG.horizon / DT) && !died; i++) {
    B.sim.step(s, DT, ctl);
    died = s.status === 'dying' || s.lives < lives0;
  }
  return s.score - score0 - (died ? DEATH_COST : 0);
}

function valuesAt(state: J, point: J, mode: Rollout = CFG.rollout): Record<string, number> {
  const saved = CFG.rollout;
  CFG.rollout = mode;
  try {
    return Object.fromEntries(point.options.map((o: string) => [o, rollout(state, point, o)]));
  } finally {
    CFG.rollout = saved;
  }
}

/** Spread the bench's random state: pellets, ghosts, headings, modes and lives, with Pac-Man 5+ tiles from every ghost. */
function randomGame(rng: () => number): J {
  const pick = <T>(xs: T[]): T => xs[Math.floor(rng() * xs.length)];
  const state = B.sim.createGame({ pacmanControl: 'jev', ghostsByAI: false });
  const { maze } = state;
  const frac = rng() * 0.9;
  for (const k of [...maze.pellets]) if (rng() < frac) maze.pellets.delete(k);
  for (const k of [...maze.powerPellets]) if (rng() < 0.5) maze.powerPellets.delete(k);
  state.level = 1 + Math.floor(rng() * 3);
  state.lives = 1 + Math.floor(rng() * 3);
  state.status = 'playing';
  state.modeTime = rng() * 60;
  state.mode = B.sim.modeAt(state.modeTime);

  const tiles: J[] = [];
  for (let y = 0; y < maze.height; y++) for (let x = 0; x < maze.width; x++) if (maze.isWalkable({ x, y }) && maze.openDirs({ x, y }).length) tiles.push({ x, y });
  const far = (a: J, b: J) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y) >= 5;
  const ids: string[] = B.types.GHOST_IDS;
  for (let attempt = 0; ; attempt++) {
    if (attempt > 10000) throw new Error('no legal placement');
    const pac = pick(tiles);
    const ghosts = ids.map((id) => {
      const u = rng();
      const st = u < 0.07 ? 'house' : u < 0.14 ? 'eaten' : u < 0.34 ? 'frightened' : 'normal';
      return { id, st, tile: st === 'house' ? B.layout.HOUSE_TILES[id] : pick(tiles) };
    });
    if (!ghosts.every((g) => far(g.tile, pac))) continue;
    // House ghosts sit on HOUSE_TILES, which are walls in the maze grid; only the open-maze placements must be walkable.
    for (const t of [pac, ...ghosts.filter((g) => g.st !== 'house').map((g) => g.tile)]) if (!maze.isWalkable(t)) throw new Error(`placed on a wall: ${JSON.stringify(t)}`);
    state.pacman.tile = { ...pac };
    state.pacman.dir = pick(maze.openDirs(pac));
    state.pacman.progress = 0;
    state.pacman.waiting = false;
    for (const g of ghosts) {
      const ghost = state.ghosts[g.id];
      ghost.state = g.st;
      ghost.tile = { ...g.tile };
      if (g.st !== 'house') ghost.dir = pick(maze.openDirs(g.tile));
      ghost.progress = 0;
      ghost.waiting = false;
    }
    state.frightLeft = ghosts.some((g) => g.st === 'frightened') ? 0.5 + rng() * B.sim.frightSeconds(state.level) : 0;
    return state;
  }
}

/** The request the bench sends for the Pac-Man questions open now (its own Scheduler builds it). */
function requestOf(state: J): J {
  let body: J;
  const sched = new B.Scheduler({
    transport: (b: J) => {
      body = b;
      return new Promise(() => {}); // never answered: only the request matters here
    },
    now: () => 0,
    onEvent: () => {},
    actors: ['pacman'],
    maxInFlight: Infinity,
  });
  sched.update(state);
  const { model: _model, ...request } = body;
  return request;
}

function episode(seed: number, counts: { skipped: number; margin: number }): J[] {
  const rng = mulberry32(seed);
  const state = randomGame(rng);
  const steps = Math.ceil((rng() * 20) / DT);
  const rows: J[] = [];
  const seen = new Set<string>();
  let considered = 0;
  // Forward play: 80% oracle pick, else a random legal option; one answer per key, as the bench keeps it.
  const memo = new Map<string, string>();
  const pick = (xs: string[]) => xs[Math.floor(rng() * xs.length)];
  const forward = {
    decide(point: J, st: J) {
      if (point.actor !== 'pacman') return B.features.greedyChoice(st, point, B.features.optionFeatures(st, point));
      if (!memo.has(point.key)) {
        const values = valuesAt(st, point);
        const top = Math.max(...Object.values(values));
        const oracle = point.options.find((o: string) => values[o] === top);
        memo.set(point.key, rng() < 0.8 ? oracle : pick(point.options));
      }
      return memo.get(point.key);
    },
  };

  for (let frame = 0; frame < steps && state.status !== 'gameover'; frame++) {
    const points = B.sim.decisionPoints(state, 'pacman');
    if (considered < MAX_DECISIONS && points.some((p: J) => !seen.has(p.key))) {
      const body = requestOf(state);
      for (const [name, key] of Object.entries(body.keys as Record<string, string>)) {
        if (seen.has(key) || considered >= MAX_DECISIONS) continue;
        seen.add(key);
        considered += 1;
        const point = points.find((p: J) => p.key === key);
        const question = body.questions[name];
        assert.deepEqual(Object.keys(question.criteria).sort(), [...point.options].sort(), `criteria keys for ${key}`);
        const values = valuesAt(state, point);
        const vals = Object.values(values);
        if (vals.every((v) => v === vals[0])) {
          counts.skipped += 1;
          continue;
        }
        const top = Math.max(...vals);
        const runnerUp = Math.max(...vals.filter((v) => v < top));
        if (top - runnerUp < CFG.margin) {
          counts.margin += 1;
          continue;
        }
        const best = Object.keys(values).filter((o) => values[o] === top);
        const { maze: _maze, ...st } = body.state;
        rows.push({ submission: SUBMISSION, game: seed, frame, game_score: state.score, state: st, question, choice: best[0], best, values });
      }
    }
    B.sim.step(state, DT, forward);
  }
  return rows;
}

function selftest(): void {
  for (let i = 0; i < 50; i++) {
    const g = randomGame(mulberry32(1000 + i));
    const open = Object.values(g.ghosts).filter((gh: J) => gh.state !== 'house');
    for (const t of [g.pacman.tile, ...open.map((gh: J) => gh.tile)]) assert.ok(g.maze.isWalkable(t), `wall tile ${JSON.stringify(t)} in seed ${1000 + i}`);
  }
  const state = B.sim.createGame({ pacmanControl: 'jev', ghostsByAI: false });
  const { maze } = state;
  const { REVERSE } = B.maze;
  // A junction T with a straight corridor of 3 tiles ahead (heading h); a normal ghost at the end, facing Pac-Man.
  let spot: J = null;
  for (let y = 0; y < maze.height && !spot; y++) {
    for (let x = 0; x < maze.width && !spot; x++) {
      const t = { x, y };
      const dirs: string[] = maze.openDirs(t);
      if (dirs.length < 3) continue;
      for (const h of dirs) {
        if (!dirs.includes(REVERSE[h])) continue;
        const line = [t];
        for (let i = 0; i < 3; i++) line.push(maze.neighbor(line.at(-1), h));
        // Plain corridor ahead (no junctions), so the ghost is the only thing that can matter.
        if (line.slice(1).every((c) => maze.isWalkable(c) && maze.openDirs(c).length === 2)) {
          spot = { t, h, ghostAt: line[3] };
          break;
        }
      }
    }
  }
  assert.ok(spot, 'no junction with a straight corridor');
  const { t, h, ghostAt } = spot;
  state.status = 'playing';
  const pac = state.pacman;
  pac.tile = { ...t };
  pac.dir = h;
  pac.progress = 0;
  pac.waiting = true; // at the junction, asking for a direction
  for (const id of ['pinky', 'inky', 'clyde']) state.ghosts[id].releaseAt = Infinity; // stay in the house
  const blinky = state.ghosts.blinky;
  blinky.state = 'normal';
  blinky.tile = { ...ghostAt };
  blinky.dir = REVERSE[h];
  blinky.progress = 0;
  blinky.waiting = false;
  const point = B.sim.nextDecisionPoint(state, 'pacman');
  assert.ok(point && point.distance === 0, 'junction question');
  const values = valuesAt(state, point, 'fallback');
  console.error(`selftest values: ${JSON.stringify(values)} (ahead ${h}, reverse ${REVERSE[h]})`);
  assert.ok(values[h] < values[REVERSE[h]], 'a ghost ahead must lower the value of going ahead');
  const greedy = valuesAt(state, point, 'greedy');
  assert.ok(Object.values(greedy).every(Number.isFinite), 'greedy rollout must return finite values');
}

async function main() {
  const { values: flags, positionals } = parseArgs({
    args: process.argv.slice(2),
    allowPositionals: true,
    options: {
      selftest: { type: 'boolean', default: false },
      rollout: { type: 'string', default: 'fallback' },
      horizon: { type: 'string', default: '8' },
      margin: { type: 'string', default: '0' },
    },
  });
  if (flags.rollout !== 'fallback' && flags.rollout !== 'greedy') throw new Error(`--rollout must be fallback or greedy, got ${flags.rollout}`);
  CFG.rollout = flags.rollout;
  CFG.horizon = Number(flags.horizon);
  CFG.margin = Number(flags.margin);
  if (!(CFG.horizon > 0) || !Number.isFinite(CFG.margin) || CFG.margin < 0) throw new Error('--horizon must be > 0 and --margin >= 0');
  if (flags.selftest) {
    B = await loadBench(positionals[0]);
    selftest();
    console.error('selftest ok');
    return;
  }
  const [benchDir, outPath, n, seed0] = positionals;
  B = await loadBench(benchDir);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, '');
  const t0 = Date.now();
  const counts = { skipped: 0, margin: 0 };
  let rows = 0;
  let options = 0;
  let dangerRows = 0;
  for (let i = 0; i < Number(n); i++) {
    const eps = episode(Number(seed0) + i, counts);
    if (eps.length) appendFileSync(outPath, eps.map((r) => JSON.stringify(r) + '\n').join(''));
    for (const r of eps) {
      rows += 1;
      options += Object.keys(r.question.criteria).length;
      if (r.best.some((k: string) => /DANGER|TRAP/.test(r.question.criteria[k]))) dangerRows += 1;
    }
  }
  console.error(
    `rows ${rows}; mean options ${(options / rows).toFixed(2)}; rows with DANGER/TRAP in best ${((100 * dangerRows) / rows).toFixed(1)}%; ` +
      `no-signal decisions skipped ${counts.skipped}; margin skipped ${counts.margin}; time ${((Date.now() - t0) / 1000).toFixed(1)} s`,
  );
  console.log(`wrote ${rows} rows to ${outPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
