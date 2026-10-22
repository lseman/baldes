// Compare serial/parallel HiGHS on one exported, fixed CE LP.
// Usage: node scripts/benchmark_ce_lp_threads.mjs MODEL.mps [OUTPUT_DIRECTORY] [THREAD_COUNTS]
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { cpus } from 'node:os';

if (!process.argv[2]) throw Error('Expected an MPS model path');
const model = resolve(process.argv[2]);
const out = resolve(process.argv[3] ?? 'target/ce-lp-threads');
const runs = Number(process.env.CE_LP_RUNS ?? 3);
const seconds = Number(process.env.CE_LP_TIME ?? 20);
if (!Number.isInteger(runs) || runs < 1 || !(seconds > 0)) throw Error('Invalid limits');
mkdirSync(out, { recursive: true });
const threadCounts = (process.argv[4] ?? '1,2,4').split(',').map(Number);
if (threadCounts.some(n => !Number.isInteger(n) || n < 1 || n > 2147483647)) throw Error('Invalid thread counts');
const configs = ['simplex', 'hipo'].flatMap(solver => threadCounts.map(threads => ({ solver, threads })));
const report = {
  model, sha256: createHash('sha256').update(readFileSync(model)).digest('hex'),
  cpu: cpus()[0]?.model, version: spawnSync('highs', ['--version'], { encoding: 'utf8' }).stdout.trim(),
  runs, seconds, unavailableSolvers: [], measurements: [],
};
const unavailable = new Set();
for (let run = 1; run <= runs; ++run) {
  const order = run % 2 ? configs : [...configs].reverse();
  for (const { solver, threads } of order) {
    if (unavailable.has(solver)) continue;
    const name = `${solver}-${threads}t`;
    const options = resolve(out, `${name}.options`);
    const lines = [`solver = ${solver}`, `threads = ${threads}`, `parallel = ${threads > 1 ? 'on' : 'off'}`, `time_limit = ${seconds}`];
    // The installed HiGHS 1.x PAMI implementation supports at most 8 tasks.
    const concurrency = Math.min(threads, 8);
    if (solver === 'simplex') lines.push(`simplex_strategy = ${threads > 1 ? 3 : 1}`, `simplex_max_concurrency = ${concurrency}`);
    writeFileSync(options, lines.join('\n') + '\n');
    const started = performance.now();
    const result = spawnSync('highs', ['--options_file', options, model], { encoding: 'utf8', timeout: (seconds + 15) * 1000, maxBuffer: 16 * 1024 * 1024 });
    const wallSeconds = (performance.now() - started) / 1000;
    const log = `${result.stdout ?? ''}${result.stderr ?? ''}`;
    writeFileSync(resolve(out, `${name}-${run}.log`), log);
    const status = log.match(/Model status\s*:\s*(.+)/)?.[1]?.trim();
    const objective = Number(log.match(/Objective value\s*:\s*([-+\d.eE]+)/)?.[1] ?? NaN);
    const solveSeconds = Number(log.match(/HiGHS run time\s*:\s*([\d.]+)/)?.[1] ?? NaN);
    if (!status && /features are unavailable/.test(log)) {
      unavailable.add(solver);
      report.unavailableSolvers.push({ solver, reason: log.trim() });
      writeFileSync(resolve(out, 'results.json'), JSON.stringify(report, null, 2) + '\n');
      console.log(`${solver}: unavailable in this HiGHS build`);
      continue;
    }
    if (!status || result.error) throw Error(`${name}: ${result.error ?? log}`);
    report.measurements.push({ solver, threads, concurrency, run, status, objective, solveSeconds, wallSeconds, exitCode: result.status });
    writeFileSync(resolve(out, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(`${name} #${run}: ${status} obj=${objective} solve=${solveSeconds}s wall=${wallSeconds.toFixed(3)}s`);
  }
}
