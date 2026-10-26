// Alternating, cut-enabled root-solve comparison. Run after compilation ends.
// node scripts/benchmarks/benchmark_vrptw_pair.mjs BEFORE AFTER FRESH_LABEL
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import {spawnSync} from 'node:child_process';
import {isDeepStrictEqual} from 'node:util';

const [before, after, label] = process.argv.slice(2);
if (!before || !after || !label || !/^[\w-]+$/.test(label)) {
  throw Error('usage: node scripts/benchmarks/benchmark_vrptw_pair.mjs BEFORE AFTER FRESH_LABEL');
}
const out = path.resolve(process.env.BALDES_PROFILE_OUT || 'results/vrptw-profile');
fs.mkdirSync(out, {recursive: true});
const output = path.join(out, `${label}.json`);
if (fs.existsSync(output)) throw Error('Choose a fresh label');
const runs = Number(process.env.BALDES_PROFILE_RUNS || 5);
if (!Number.isInteger(runs) || runs < 1) throw Error('Invalid repetition count');
const affinity = process.env.BALDES_PROFILE_CPUS || '0,1';
const env = {...process.env, OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MKL_NUM_THREADS: '1'};
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
function run(command, args) {
  const p = spawnSync(command, args, {env, encoding: 'utf8', timeout: 150000, maxBuffer: 64 * 1024 * 1024});
  if (p.error || p.status !== 0) throw Error(`${command}: ${p.error || p.stderr || p.stdout}`);
  return p.stdout + p.stderr;
}
const binaries = Object.fromEntries(Object.entries({before, after}).map(([variant, binary]) => [variant, {
  path: path.resolve(binary), sha256: sha256(fs.readFileSync(binary)),
  build: process.env[`BALDES_PAIR_${variant.toUpperCase()}_BUILD`] || 'unrecorded',
}]));
const allCuts = process.env.BALDES_PROFILE_ALL_CUTS === '1';
const cases = (process.env.BALDES_PROFILE_CASES || 'R103,R201,R201-Q100').split(',');
if (cases.some(name => !/^[CR][12]\d\d(?:-Q100)?$/.test(name))) throw Error('Invalid Solomon case name');
const args = allCuts ? ['--customers', '100', '--master', 'clp', '--cuts']
  : ['--customers', '100', '--master', 'clp', '--clp-scaling', 'equilibrium',
  '--reference-ng', '--ng', '8', '--rounding', 'none', '--rank1', '--capacity-cuts',
  '--no-enumeration', '--bound-mode', 'numerical', '--time', '30'];
const result = {
  date: new Date().toISOString(), commit: run('git', ['rev-parse', 'HEAD']).trim(),
  rustc: run('rustc', ['-Vv']), cpu: os.cpus()[0].model, affinity, nativeThreads: 1,
  warmups: 1, repetitions: runs, binaries, args, allCuts, cases: [],
};
// Reserve the label before writing raw artifacts; partial failures remain inspectable.
fs.writeFileSync(output, JSON.stringify(result, null, 2));
const median = xs => {
  const sorted = [...xs].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
for (const name of cases) {
  const source = `instances/${name.split('-')[0]}.txt`;
  let contents = fs.readFileSync(source, 'utf8');
  let input = source;
  if (name.endsWith('-Q100')) {
    const header = /(NUMBER\s+CAPACITY\s*\r?\n\s*\d+\s+)(\d+)/;
    if (!header.test(contents)) throw Error('Missing Solomon capacity header');
    contents = contents.replace(header, (_, prefix) => prefix + '100')
      .replace(/^[^\r\n]+/, name);
    input = path.join(out, `${label}-${name}.txt`);
    fs.writeFileSync(input, contents);
  }
  const entry = {name, source, input, syntheticCapacity: name.endsWith('-Q100') ? 100 : null,
    inputSha256: sha256(contents), rows: []};
  result.cases.push(entry);
  let expected;
  for (let i = 0; i <= runs; i++) {
    const pair = {};
    for (const variant of i % 2 ? ['after', 'before'] : ['before', 'after']) {
      const stem = path.join(out, `${label}-${name}-${variant}-${i}`);
      const log = run('taskset', ['-c', affinity, binaries[variant].path,
        'solve', input, ...args, '--json', `${stem}.json`]);
      fs.writeFileSync(`${stem}.log`, log);
      const j = JSON.parse(fs.readFileSync(`${stem}.json`));
      const semantic = Object.fromEntries(Object.entries(j).filter(([key]) =>
        !['seconds', 'pricing_seconds', 'master_seconds'].includes(key)));
      if (j.status !== 'Optimal' || !Number.isFinite(j.lp) || !Array.isArray(j.history) || !Array.isArray(j.routes)) {
        throw Error(`Invalid solver result: ${stem}`);
      }
      expected ??= semantic;
      if (!isDeepStrictEqual(expected, semantic)) throw Error(`Non-timing result mismatch: ${stem}`);
      pair[variant] = Object.fromEntries(['seconds', 'pricing_seconds', 'master_seconds'].map(key => {
        if (!Number.isFinite(j[key]) || j[key] < 0) throw Error(`Invalid ${key}: ${stem}`);
        return [key, j[key]];
      }));
    }
    if (i) entry.rows.push(pair);
    console.log(name, i ? `pair ${i}` : 'warmup', JSON.stringify(pair));
    fs.writeFileSync(output, JSON.stringify(result, null, 2));
  }
  entry.median = Object.fromEntries(['before', 'after'].map(variant => [variant,
    Object.fromEntries(['seconds', 'pricing_seconds', 'master_seconds'].map(key =>
      [key, median(entry.rows.map(row => row[variant][key]))]))]));
  entry.medianPairedReduction = median(entry.rows.map(row => 1 - row.after.seconds / row.before.seconds));
  entry.allNonTimingFieldsMatch = true;
  fs.writeFileSync(output, JSON.stringify(result, null, 2));
}
