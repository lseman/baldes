// Usage: node scripts/benchmark_column_elimination.mjs name=/path/to/baldes [...]
// CE_BENCH_OUT, CE_BENCH_TIME and CE_BENCH_RUNS override the defaults.
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { cpus } from 'node:os';
import { createHash } from 'node:crypto';

const out = resolve(process.env.CE_BENCH_OUT ?? 'target/ce-reduction-benchmark');
const seconds = Number(process.env.CE_BENCH_TIME ?? 10);
const runs = Number(process.env.CE_BENCH_RUNS ?? 3);
if (!(seconds > 0) || !Number.isInteger(runs) || runs < 1) throw Error('Invalid benchmark limits');
mkdirSync(out, { recursive: true });
const cases = [
  ['C101', 'instances/C101.txt', []],
  ['R101', 'instances/R101.txt', []],
  ['P16', 'instances/cvrplib/P-n16-k8.vrp', []],
  ['P16-10', 'instances/cvrplib/P-n16-k8.vrp', ['--customers', '10']],
  ['A32', 'instances/cvrplib/A-n32-k5.vrp', []],
];
const versions = process.argv.slice(2).map(value => {
  const split = value.indexOf('=');
  if (split < 1) throw Error('Expected name=/path/to/baldes');
  return [value.slice(0, split), resolve(value.slice(split + 1))];
});
if (!versions.length) throw Error('Specify at least one binary');
const reports = new Map();
for (const [name, binary] of versions) {
  if (!/^[\w-]+$/.test(name)) throw Error('Invalid binary label');
  reports.set(name, {
    binary, sha256: createHash('sha256').update(readFileSync(binary)).digest('hex'),
    seconds, runs, cpu: cpus()[0]?.model,
    rustc: spawnSync('rustc', ['--version'], { encoding: 'utf8' }).stdout.trim(),
    profile: 'release', backend: 'Clp', solverThreads: 1,
    measurements: [],
  });
  // Warm up binary and native solver on a small deterministic solve.
  const warmup = spawnSync(binary, ['ce', 'instances/C101.txt', '--customers', '5', '--time', '2'], { encoding: 'utf8', timeout: 15000 });
  if (warmup.status !== 0) throw Error(`Warmup failed: ${warmup.error ?? warmup.stderr}`);
}
for (let run = 1; run <= runs; ++run) {
  for (const [instance, path, extra] of cases) {
    // The larger capacity-state model needs a longer first LP solve.
    const budget = process.env.CE_BENCH_TIME ? seconds : instance === 'A32' ? 30 : seconds;
    const ordered = run % 2 ? versions : [...versions].reverse();
    for (const [name, binary] of ordered) {
      const json = resolve(out, `${name}-${instance}-${run}.json`);
      const start = performance.now();
      const result = spawnSync(binary, ['ce', path, ...extra, '--time', String(budget), '--json', json], {
        encoding: 'utf8', timeout: (budget + 15) * 1000,
      });
      const wallSeconds = (performance.now() - start) / 1000;
      if (result.status !== 0) throw Error(`${name}/${instance}: ${result.error ?? result.stderr}`);
      const data = JSON.parse(readFileSync(json, 'utf8'));
      const report = reports.get(name);
      const measurement = { case: instance, run, budget, wallSeconds, ...data };
      report.measurements.push(measurement);
      writeFileSync(resolve(out, `${name}.json`), JSON.stringify(report, null, 2) + '\n');
      console.log(`${name} ${instance} #${run}: ${data.status} ${data.elapsed_seconds.toFixed(3)}s bound=${data.lower_bound} nodes=${data.nodes} arcs=${data.arcs} LPs=${data.iterations}`);
    }
  }
}
