// Compare exact pricing backends: node scripts/benchmark_pstep_pricing.mjs [BINARY] [OUTPUT]
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { cpus } from 'node:os';
import { createHash } from 'node:crypto';

const binary = resolve(process.argv[2] ?? './target/release/baldes');
const output = resolve(process.argv[3] ?? 'results/pstep-pricing-20261021');
const repetitions = Number(process.env.PSTEP_RUNS ?? 3);
const seconds = Number(process.env.PSTEP_SECONDS ?? 20);
const threads = Number(process.env.PSTEP_THREADS ?? 4);
mkdirSync(output, { recursive: true });
const runs = [];
const sha256 = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const binarySha256 = sha256(binary);
for (const instance of ['P-n16-k8', 'E-n22-k4', 'A-n32-k5']) {
  for (const p of [2, 3, 4]) {
    for (let repetition = 0; repetition < repetitions; repetition++) {
      // Alternate order to reduce systematic warm-cache and scheduling bias.
      for (const method of repetition % 2 ? ['enumeration', 'bucket'] : ['bucket', 'enumeration']) {
        const file = join(output, `${instance}-p${p}-${method}-${repetition}.json`);
        const run = spawnSync(binary, ['pstep', `instances/cvrplib/${instance}.vrp`,
          '--p', String(p), '--threads', String(threads), '--pricing', method,
          '--time', String(seconds), '--labels', '5000000', '--json', file],
        { encoding: 'utf8', timeout: (seconds + 10) * 1000 });
        const result = run.status === 0 ? JSON.parse(readFileSync(file, 'utf8')) : {
          status: /time limit/i.test(run.stderr) ? 'TimeLimit' : run.status === null ? 'ExecutionError' : 'Stopped',
          error: [run.error, run.stderr].filter(Boolean).join('\n').trim(),
          exit_status: run.status, signal: run.signal, lp_bound: null,
        };
        if (run.status !== 0) writeFileSync(file, JSON.stringify(result, null, 2) + '\n');
        runs.push({ instance, p, method, repetition, ...result });
        process.stdout.write(`${instance} p=${p} ${method}: ${result.status} ${result.elapsed_seconds ?? ''}\n`);
      }
    }
    const solved = runs.filter(r => r.instance === instance && r.p === p && r.status === 'Converged');
    if (solved.length && solved.some(r => Math.abs(r.lp_bound - solved[0].lp_bound) > 1e-5)) {
      throw new Error(`Backend bound mismatch: ${instance} p=${p}`);
    }
  }
}
writeFileSync(join(output, 'results.json'), JSON.stringify({ binary, binarySha256,
  rustc: spawnSync('rustc', ['--version'], { encoding: 'utf8' }).stdout.trim(),
  inputSha256: Object.fromEntries(['P-n16-k8', 'E-n22-k4', 'A-n32-k5'].map(name =>
    [name, sha256(`instances/cvrplib/${name}.vrp`)])),
  threads, seconds, repetitions, warmupRuns: 0,
  cpu: cpus()[0]?.model, platform: process.platform, architecture: process.arch, runs }, null, 2) + '\n');
const fields = ['instance', 'p', 'method', 'repetition', 'status', 'lp_bound', 'elapsed_seconds',
  'pricing_seconds', 'lp_seconds', 'iterations', 'columns', 'generated_labels', 'enumerated_extensions'];
writeFileSync(join(output, 'results.csv'), [fields.join(','),
  ...runs.map(r => fields.map(f => r[f] ?? '').join(','))].join('\n') + '\n');
