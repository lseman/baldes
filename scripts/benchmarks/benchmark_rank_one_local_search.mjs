// Compare root solves with the same Rayon budget; run after both builds finish.
// node scripts/benchmarks/benchmark_rank_one_local_search.mjs BASELINE CANDIDATE
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';

const binaries = process.argv.slice(2).map(p => path.resolve(p));
if (binaries.length !== 2) throw Error('Pass baseline and candidate executables');
const threads = Number(process.env.BALDES_LS_THREADS || 4);
const cpus = process.env.BALDES_LS_CPUS || '0-3';
const runs = Number(process.env.BALDES_LS_RUNS || 3);
if (!Number.isSafeInteger(threads) || threads < 1 || !Number.isSafeInteger(runs) || runs < 1 || runs % 2 !== 1) {
  throw Error('Threads must be positive; runs must be positive and odd');
}
const out = path.resolve(process.env.BALDES_LS_OUT || `results/rank-one-local-search/root-${threads}`);
fs.mkdirSync(out, {recursive: true});
if (fs.existsSync(path.join(out, 'metadata.json'))) throw Error('Choose a fresh output directory');
const env = {...process.env, RAYON_NUM_THREADS: String(threads), OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MKL_NUM_THREADS: '1'};
fs.writeFileSync(path.join(out, 'metadata.json'), JSON.stringify({
  date: new Date().toISOString(), cpu: os.cpus()[0]?.model, threads, cpus, runs, warmups: 1,
  rustc: spawnSync('rustc', ['-Vv'], {encoding: 'utf8'}).stdout,
  binaries: binaries.map(binary => ({path: binary, sha256: crypto.createHash('sha256').update(fs.readFileSync(binary)).digest('hex')})),
}, null, 2));
const median = values => [...values].sort((a,b) => a-b)[Math.floor(values.length/2)];
const summaries = [];
for (const instance of ['R103', 'R201']) {
  const args = ['vrptw', `instances/Solomon/${instance}.txt`, '--customers', '100', '--rank1',
    '--master', 'clp', '--reference-ng', '--ng', '8', '--rounding', 'none', '--no-enumeration',
    '--bound-mode', 'numerical', '--time', '120'];
  const samples = [[], []];
  let expected;
  for (let run = 0; run <= runs; run++) {
    for (const variant of run % 2 ? [1, 0] : [0, 1]) {
      const label = variant ? 'parallel' : 'baseline';
      const stem = path.join(out, `${instance}-${label}-${run}`);
      const result = spawnSync('taskset', ['-c', cpus, binaries[variant], ...args, '--json', stem + '.json'], {
        env, encoding: 'utf8', timeout: 180000, maxBuffer: 8 * 1024 * 1024,
      });
      fs.writeFileSync(stem + '.log', (result.stdout || '') + (result.stderr || ''));
      if (result.status !== 0) throw Error(`${stem}: ${result.status} ${result.error || ''}`);
      const data = JSON.parse(fs.readFileSync(stem + '.json'));
      expected ??= data;
      for (const key of ['status', 'cuts', 'cut_rounds', 'iterations', 'pricing_calls', 'simplex_iterations', 'columns', 'route_columns', 'master_columns']) {
        if (data[key] !== expected[key]) throw Error(`${stem}: ${key} differs: ${data[key]} vs ${expected[key]}`);
      }
      if (data.status !== 'Optimal' || !Number.isFinite(data.lp) || !Number.isFinite(data.bound)
          || Math.abs(data.lp - expected.lp) > 1e-6 || Math.abs(data.bound - expected.bound) > 1e-5) {
        throw Error(`${stem}: LP/bound mismatch`);
      }
      if (run) samples[variant].push(data.seconds);
      console.log(`${instance} ${label} run=${run} seconds=${data.seconds} cuts=${data.cuts} lp=${data.lp}`);
    }
  }
  const medians = samples.map(median);
  summaries.push({instance, args, samples, medians, speedup: medians[0]/medians[1], cuts: expected.cuts, lp: expected.lp});
  fs.writeFileSync(path.join(out, 'summary.json'), JSON.stringify(summaries, null, 2));
}
console.log(JSON.stringify(summaries, null, 2));
