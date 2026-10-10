// Turns jevman-benchmark recordings into training rows: one per Pac-Man decision a model made.
// Row: {submission, game, frame, game_score, state, question, choice}. frame = the frame the request was sent.
// Run from the bench dir:
//   node --import tsx <this file> <bench_dir> <out.jsonl> [submission ids...]
//   node --import tsx <this file> --check <bench_dir> <submission> <game number>
// The bench's own Scheduler builds every request. Its transport answers from the recording, so sends, re-sends and
// the state are the bench's. Requests are sent as endpointDecide sends them: model and keys removed, state.maze removed.
// Rows come only from games whose replay matches the recorded final state. A game that mismatches or throws is skipped
// and counted; its rows are never written.
import { readFileSync, readdirSync, writeFileSync, mkdirSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

type J = any;
const DEFAULT_IDS = ['semif', 'qwen-3-8-flash-next-nvfp4', 'rizzo-flow', 'winnow-12b-nvfp4'];
const push = <K, T>(m: Map<K, T[]>, k: K, v: T) => m.set(k, [...(m.get(k) ?? []), v]);
const sameKeys = (a: string[], b: string[]) => a.length === b.length && b.every((k) => a.includes(k));

async function loadBench(dir: string) {
  const at = (p: string) => import(pathToFileURL(join(dir, p)).href);
  const [sim, sched] = await Promise.all([at('src/sim.ts'), at('src/scheduler.ts')]);
  return { sim, Scheduler: sched.Scheduler as J };
}

function load(bench: string, id: string, n: number): J {
  return JSON.parse(gunzipSync(readFileSync(join(bench, 'submissions', id, `game-${String(n).padStart(2, '0')}.json.gz`))).toString());
}

function gameNumbers(bench: string, id: string): number[] {
  return readdirSync(join(bench, 'submissions', id))
    .map((f) => /^game-(\d+)\.json\.gz$/.exec(f)?.[1])
    .filter((n): n is string => n !== undefined)
    .map(Number)
    .sort((a, b) => a - b);
}

/** Replays one recorded game with the bench's sim and scheduler. Returns the rows and the replay's final state. */
async function replay(bench: J, rec: J, submission: string, game: number) {
  const { sim, Scheduler } = bench;
  // Recorded decision events per key, [frame, decision]; choices per frame|key, in call order.
  const dec = new Map<string, [number, J][]>();
  for (const [f, e] of rec.events) if (e.type === 'decision') push(dec, e.decision.key, [f, e.decision]);
  const choices = new Map<string, string[]>();
  for (const [f, key, dir] of rec.decisions) push(choices, `${f}|${key}`, dir);

  let frame = 0;
  let clock = 0; // ms of game time, the scheduler's clock
  const inbox: { at: number; body: J; resolve: (r: J) => void; reject: (e: Error) => void; failed: boolean }[] = [];
  const answered = new Map<string, { sent: number; body: J; at: number }>(); // key -> request that gave its answer
  const rows: J[] = [];
  let skipped = 0;
  let noAnswer = 0;

  // Each request gets the first unclaimed call event after it was sent, with as many actors as it has questions. The
  // recording logs one call event per answered request, in arrival order; `misfiled` counts arrivals whose decisions
  // name keys the request did not ask.
  // Each call event owns the jev decisions and stale notes logged after it, in the same frame.
  const calls: [number, { n: number; keys: string[] }][] = [];
  for (const [f, e] of rec.events as [number, J][]) {
    if (e.type === 'call') calls.push([f, { n: e.actors.length, keys: [] }]);
    const last = calls.at(-1);
    if (!last || last[0] !== f) continue;
    if (e.type === 'stale') last[1].keys.push(e.key);
    if (e.type === 'decision' && e.decision.source === 'jev') last[1].keys.push(e.decision.key);
  }
  const errors = (rec.events as [number, J][]).filter(([, e]) => e.type === 'decision' && e.decision.source === 'fallback' && e.decision.reason === 'error');
  const claimed = new Set<number>();
  let misfiled = 0;
  const transport = (body: J) =>
    new Promise((resolve, reject) => {
      const keys = Object.values(body.keys) as string[];
      let failed = false;
      // An arrival only fits a request that asked for all of its decisions' keys.
      const i = calls.findIndex(([f, c], j) => f > frame && !claimed.has(j) && c.n === keys.length && sameKeys(c.keys, keys));
      let at = Infinity;
      if (i >= 0) {
        claimed.add(i);
        at = calls[i][0];
        for (const [, d] of (rec.events as [number, J][]).filter(([f, e]) => f === at && e.type === 'decision' && e.decision.source === 'jev')) {
          if (!keys.includes(d.decision.key)) misfiled += 1;
        }
      } else {
        // No answer was logged: the bench got an error, and resolved the request at that frame (empty answers).
        const err = errors.find(([f, e]) => f > frame && keys.includes(e.decision.key));
        if (err) { at = err[0]; failed = true; }
        else noAnswer += 1;
      }
      inbox.push({ at, body: { ...body, sent: frame }, resolve, reject, failed });
    });
  const scheduler = new Scheduler({ transport, now: () => clock, onEvent: () => {}, actors: ['pacman'] });

  const flush = async () => {
    const due = inbox.filter((x) => x.at <= frame);
    if (!due.length) return;
    inbox.splice(0, inbox.length, ...inbox.filter((x) => x.at > frame));
    for (const x of due) {
      if (x.failed) {
        x.reject(new Error('error'));
        continue;
      }
      const answers: J = {};
      for (const [name, key] of Object.entries(x.body.keys) as [string, string][]) {
        const d = dec.get(key)?.find(([f, e]) => f === x.at && e.source === 'jev')?.[1];
        if (!d) continue;
        answers[name] = { type: 'choice', choice: d.choice };
        answered.set(key, { sent: x.body.sent, body: x.body, at: x.at });
      }
      x.resolve({ answers, latencyMs: 0, usage: { input_tokens: 0, output_tokens: 0 }, costUsd: null, traceId: null });
    }
    await new Promise((r) => setImmediate(r)); // let the scheduler's promise handlers run
  };

  const ctl = {
    decide(point: J, s: J) {
      if (point.actor === 'pacman') scheduler.decide(point, s);
      const dir: string | undefined = choices.get(`${frame}|${point.key}`)?.shift();
      if (dir === undefined) return null;
      if (point.actor !== 'pacman') return dir;
      const last = dec.get(point.key)?.filter(([f]) => f <= frame).at(-1);
      if (!last || last[1].source !== 'jev') {
        skipped += 1; // a fallback answer: not a model's choice
        return dir;
      }
      // The model's label is its answer. An escape label names the option it was given (heading or reverse), so the
      // step is mapped through the options the question had when it was sent.
      const snap = answered.get(point.key);
      const label: string = last[1].choice;
      const expected = point.escape ? (label === last[1].options[0] ? point.options[0] : point.options[1]) : label;
      if (!snap || snap.at !== last[0] || expected !== dir) throw new Error(`${submission} game ${game} frame ${frame}: ${point.key} does not match its request (answered at ${snap?.at}, decided at ${last[0]}, label ${label}, dir ${dir}, expected ${expected}, options ${point.options})`);
      const { model: _model, keys, ...request } = snap.body;
      const name = Object.keys(keys).find((n) => keys[n] === point.key)!;
      const question = request.questions[name];
      // The question must list the decision's options, and the label must be one of them.
      const listed = Object.keys(question.criteria).sort().join(',');
      if (listed !== [...last[1].options].sort().join(',') || !Object.hasOwn(question.criteria, label)) {
        throw new Error(`frame ${frame}: question ${name} lists ${listed}, decision had options ${last[1].options}`);
      }
      const { maze: _maze, ...state } = request.state;
      rows.push({ submission, game, frame: snap.sent, game_score: rec.final.score, state, question, choice: label });
      return dir;
    },
  };

  const state = sim.createGame();
  let f = 0;
  while (state.status !== 'gameover' && f < rec.frames.length) {
    frame = f;
    await flush();
    scheduler.update(state);
    sim.step(state, rec.frames[f], ctl);
    clock += rec.frames[f] * 1000;
    f += 1;
  }
  const leftover = [...choices.entries()].filter(([, q]) => q.length).map(([k]) => Number(k.split('|')[0])).sort((a, b) => a - b);
  const final = { score: state.score, lives: state.lives, level: state.level, frames: f };
  const want = rec.final;
  const ok = final.score === want.score && final.lives === want.lives && final.level === want.level && final.frames === want.frames && leftover.length === 0;
  return { ok, final, want, firstUnused: leftover[0], rows, skipped, noAnswer, misfiled };
}

async function main() {
  const check = process.argv[2] === '--check';
  const benchDir = check ? process.argv[3] : process.argv[2];
  const bench = await loadBench(benchDir);
  if (check) {
    const [id, n] = process.argv.slice(4);
    const rec = load(benchDir, id, Number(n));
    const r = await replay(bench, rec, id, Number(n));
    assert.equal(r.final.score, rec.final.score, 'final score');
    assert.equal(r.final.lives, rec.final.lives, 'final lives');
    assert.equal(r.final.frames, rec.final.frames, 'frame count');
    console.log(`ok: ${id} game ${n}: score ${r.final.score}, lives ${r.final.lives}, frames ${r.final.frames}, ${r.rows.length} rows`);
    return;
  }
  const [, outPath, ...given] = process.argv.slice(2);
  const ids = given.length ? given : DEFAULT_IDS;
  const all: J[] = [];
  for (const id of ids) {
    const nums = gameNumbers(benchDir, id);
    let matched = 0;
    let rows = 0;
    for (const n of nums) {
      let r: J;
      try {
        r = await replay(bench, load(benchDir, id, n), id, n);
      } catch (e) {
        console.error(`SKIP ${id} game ${n}: ${(e as Error).message}`);
        continue;
      }
      if (!r.ok) {
        const where = r.firstUnused === undefined ? '' : `; first recorded decision not used at frame ${r.firstUnused}`;
        console.error(`SKIP ${id} game ${n}: replay score ${r.final.score} lives ${r.final.lives} level ${r.final.level} frames ${r.final.frames}; recorded score ${r.want.score} lives ${r.want.lives} level ${r.want.level} frames ${r.want.frames}${where}`);
        continue;
      }
      matched += 1;
      rows += r.rows.length;
      all.push(...r.rows);
    }
    console.log(`${id}: games matched ${matched}/${nums.length}; rows ${rows}`);
  }
  for (const r of all) assert.ok(Object.hasOwn(r.question.criteria, r.choice), `${r.submission} game ${r.game} frame ${r.frame}: choice ${r.choice} is not a criteria key`);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, all.map((r) => JSON.stringify(r)).join('\n') + '\n');
  console.log(`wrote ${all.length} rows to ${outPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
