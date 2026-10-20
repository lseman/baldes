// From the repository root: node scripts/benchmark_pstep.mjs ./target/release/baldes
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { cpus } from 'node:os';

const grid = process.argv.includes('--grid');
const binary = resolve(process.argv.slice(2).find(arg => !arg.startsWith('--')) ?? './target/release/baldes');
const destination = resolve('results/pstep-20261020', grid ? 'p234' : '.');
mkdirSync(destination, { recursive: true });
const cases = grid ? ['P-n16-k8', 'E-n22-k4', 'A-n32-k5'].flatMap(instance =>
  [2, 3, 4].map(p => [instance, p, 4])) : [
  ['P-n16-k8', 1, 4],
  ...Array.from({ length: 5 }, () => [['P-n16-k8', 2, 1], ['P-n16-k8', 2, 4]]).flat(),
  ['P-n16-k8', 3, 4], ['P-n16-k8', 4, 4],
  ['E-n22-k4', 1, 4], ['E-n22-k4', 2, 4], ['E-n22-k4', 3, 4],
  ['A-n32-k5', 1, 4], ['A-n32-k5', 2, 4],
];
const results = [];
for (const [index, [instance, p, threads]] of cases.entries()) {
  const file = `instances/cvrplib/${instance}.vrp`;
  const reportPath = join(destination, `run-${String(index + 1).padStart(2, '0')}.json`);
  const verify = !grid || instance === 'P-n16-k8' || p === 2 || (instance === 'E-n22-k4' && p === 3);
  const args = ['pstep', file, '--p', String(p), '--threads', String(threads),
    ...(verify ? ['--enumerate'] : []), '--labels', grid ? '5000000' : '1000000',
    '--time', grid ? '120' : '60', '--json', reportPath];
  const run = spawnSync(binary, args, { encoding: 'utf8', timeout: grid ? 150_000 : 120_000 });
  if (run.status !== 0) {
    if (!grid) throw new Error(`${instance} p=${p}: ${run.error ?? run.stderr}`);
    const error = String(run.error ?? run.stderr).trim();
    const status = run.error ? 'ExecutionError' : /label limit/i.test(error) ? 'LabelLimit'
      : /time limit/i.test(error) ? 'TimeLimit' : 'PricingStopped';
    const failure = { instance, p, requested_threads: threads, status,
      lp_bound: null, error };
    writeFileSync(reportPath, JSON.stringify(failure, null, 2) + '\n');
    results.push(failure);
    process.stdout.write(`${instance} p=${p}: ${failure.error}\n`);
    continue;
  }
  const result = JSON.parse(readFileSync(reportPath, 'utf8'));
  if (result.status !== 'Converged' || (verify && Math.abs(result.lp_bound - result.enumerated_bound) > 1e-5)) {
    throw new Error(`Nonconverged or mismatching bound: ${reportPath}`);
  }
  results.push({ ...result, sha256: createHash('sha256').update(readFileSync(file)).digest('hex') });
  process.stdout.write(run.stdout);
}
const data = { binary, platform: process.platform, architecture: process.arch,
  logical_cpus: cpus().length, cpu: cpus()[0]?.model,
  time_limit_seconds: grid ? 120 : 60, max_labels_per_job: grid ? 5_000_000 : 1_000_000,
  runs: results };
writeFileSync(join(destination, 'results.json'), JSON.stringify(data, null, 2) + '\n');
const fields = ['instance', 'p', 'requested_threads', 'pricing_workers', 'status', 'lp_bound',
  'enumerated_bound', 'iterations', 'columns', 'pricing_jobs', 'generated_labels',
  'elapsed_seconds', 'pricing_seconds', 'lp_seconds'];
writeFileSync(join(destination, 'results.csv'), [fields.join(','),
  ...results.map(r => fields.map(f => r[f]).join(','))].join('\n') + '\n');
