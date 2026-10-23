// Reproducible cut-enabled VRPTW timing and optional perf sampling.
// node scripts/profiling/profile_vrptw.mjs BINARY LABEL
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import crypto from 'node:crypto';
import os from 'node:os';
import path from 'node:path';

const [binaryArg, label] = process.argv.slice(2);
if (!binaryArg || !label || !/^[\w-]+$/.test(label)) {
  throw Error('usage: node scripts/profiling/profile_vrptw.mjs BINARY LABEL');
}
const binary = path.resolve(binaryArg);
const out = path.resolve(process.env.BALDES_PROFILE_OUT || 'results/vrptw-profile');
fs.mkdirSync(out, {recursive: true});
const metadataPath = path.join(out, `${label}-metadata.json`);
if (fs.existsSync(metadataPath)) throw Error('Choose a fresh label/output directory');
function positive(name, fallback) {
  const n = Number(process.env[name] ?? fallback);
  if (!Number.isInteger(n) || n <= 0) throw Error(`${name} must be a positive integer`);
  return n;
}
const runs = positive('BALDES_PROFILE_RUNS', 3);
const seconds = positive('BALDES_PROFILE_SECONDS', 30);
const customers = positive('BALDES_PROFILE_CUSTOMERS', 100);
const capacity = process.env.BALDES_PROFILE_CAPACITY === undefined
  ? null : positive('BALDES_PROFILE_CAPACITY');
const cases = (process.env.BALDES_PROFILE_CASES || 'C101,R103,R201').split(',');
if (cases.some(c => !/^[CR][12]\d\d$/.test(c))) throw Error('Invalid Solomon case name');
const inputs = cases.map(name => {
  const source = `instances/${name}.txt`;
  let contents = fs.readFileSync(source, 'utf8');
  let input = source;
  if (capacity !== null) {
    const header = /(NUMBER\s+CAPACITY\s*\r?\n\s*\d+\s+)(\d+)/;
    if (!header.test(contents)) throw Error(`Missing Solomon vehicle capacity: ${source}`);
    contents = contents.replace(header, (_, prefix) => prefix + capacity)
      .replace(/^[^\r\n]+/, `${name}-Q${capacity}`);
    input = path.join(out, `${label}-${name}-Q${capacity}.txt`);
    fs.writeFileSync(input, contents);
  }
  return {name, source, input, sha256: crypto.createHash('sha256').update(contents).digest('hex')};
});
const affinity = process.env.BALDES_PROFILE_CPUS || '0,1';
const clpScaling = process.env.BALDES_PROFILE_CLP_SCALING || 'equilibrium';
if (!['auto', 'equilibrium', 'geometric', 'off'].includes(clpScaling)) throw Error('Invalid Clp scaling policy');
const env = {...process.env, OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MKL_NUM_THREADS: '1'};
function run(command, args) {
  const p = spawnSync(command, args, {env, encoding: 'utf8', timeout: (seconds + 120) * 1000, maxBuffer: 64 * 1024 * 1024});
  if (p.error || p.status !== 0) throw Error(`${command}: ${p.error || p.stderr || p.stdout}`);
  return p.stdout + p.stderr;
}
const write = (name, value) => fs.writeFileSync(path.join(out, name), JSON.stringify(value, null, 2));
write(`${label}-metadata.json`, {
  binary, label, sha256: crypto.createHash('sha256').update(fs.readFileSync(binary)).digest('hex'),
  commit: run('git', ['rev-parse', 'HEAD']).trim(), rustc: run('rustc', ['-Vv']),
  cpu: os.cpus()[0].model, date: new Date().toISOString(), loadavg: os.loadavg(),
  build: process.env.BALDES_PROFILE_BUILD || 'unrecorded', affinity, runs, warmups: 1,
  seconds, customers, cases, capacity, inputs, clpScaling, nativeThreads: 1,
  perf: process.env.BALDES_PROFILE_PERF === '1' ? {event: 'cycles:u', frequency: 499, callGraph: 'dwarf,16384'} : null,
});
const summary = [];
for (const {name, input} of inputs) {
  const args = ['vrptw', input, '--customers', String(customers),
    '--master', 'clp', '--clp-scaling', clpScaling, '--reference-ng', '--ng', '8', '--rounding', 'none',
    '--rank1', '--capacity-cuts', '--no-enumeration', '--bound-mode', 'numerical', '--time', String(seconds)];
  const rows = [];
  for (let i = 0; i <= runs; i++) {
    const stem = `${label}-${name}-${i}`;
    const log = run('taskset', ['-c', affinity, binary, ...args, '--json', path.join(out, `${stem}.json`)]);
    fs.writeFileSync(path.join(out, `${stem}.log`), log);
    const j = JSON.parse(fs.readFileSync(path.join(out, `${stem}.json`)));
    if (!Number.isFinite(j.lp) || !Number.isFinite(j.seconds)) throw Error(`${stem}: invalid result`);
    if (i) rows.push(Object.fromEntries(['status', 'seconds', 'pricing_seconds', 'master_seconds', 'lp', 'bound', 'iterations', 'pricing_calls', 'simplex_iterations', 'cuts', 'cut_rounds', 'lp_before_cuts'].map(k => [k, j[k]])));
    console.log(stem, j.status, j.seconds.toFixed(3), `pricing=${j.pricing_seconds.toFixed(3)}`, `cuts=${j.cuts}`);
  }
  const complete = rows.every(r => r.status === 'Optimal');
  const comparable = complete && rows.every(r => Math.abs(r.lp - rows[0].lp) <= 1e-6 && Math.abs(r.bound - rows[0].bound) <= 1e-5 && r.iterations === rows[0].iterations && r.cuts === rows[0].cuts);
  const median = k => {const values = rows.map(r => r[k]).sort((a,b) => a-b); const m = Math.floor(values.length / 2); return values.length % 2 ? values[m] : (values[m-1] + values[m]) / 2;};
  summary.push({name, args, complete, comparable, rows, median: Object.fromEntries(['seconds', 'pricing_seconds', 'master_seconds'].map(k => [k, median(k)]))});
  write(`${label}-summary.json`, summary);
  if (process.env.BALDES_PROFILE_PERF === '1') {
    const stem = `${label}-${name}-perf`;
    const data = path.join(out, `${stem}.data`);
    fs.writeFileSync(path.join(out, `${stem}.log`), run('perf', ['record', '-q', '-e', 'cycles:u', '-F', '499', '--call-graph', 'dwarf,16384', '-o', data, '--', 'taskset', '-c', affinity, binary, ...args, '--json', path.join(out, `${stem}.json`)]));
    for (const [kind, option] of [['self', '--no-children'], ['inclusive', '--children']]) {
      fs.writeFileSync(path.join(out, `${stem}-${kind}.txt`), run('perf', ['report', '--stdio', '--percent-limit', '0.5', '--sort', 'symbol', '--demangle', '--call-graph', 'none', option, '-i', data]));
    }
  }
}
