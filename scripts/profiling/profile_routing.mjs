// Root timing and perf sampling. Run from the repository root.
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import crypto from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
const [binary, label='baseline'] = process.argv.slice(2);
if (!binary || !/^[\w-]+$/.test(label)) throw Error('usage: node scripts/profiling/profile_routing.mjs BINARY LABEL');
const out = path.resolve(process.env.BALDES_PROFILE_OUT || 'results/routing-profile') + path.sep;
fs.mkdirSync(out, {recursive:true});
if(fs.existsSync(out+label+'-metadata.json')) throw Error('Output exists; choose a fresh label or output directory');
const cases = [
 ['C202','vrptw','instances/Solomon/C202.txt',[]],
 ['R201-rank1','vrptw','instances/Solomon/R201.txt',['--rank1']],
 ['R103-rank1','vrptw','instances/Solomon/R103.txt',['--rank1']],
 ['A-n32-k5','cvrp','instances/cvrplib/A-n32-k5.vrp',[]],
 ['A-n32-k5-capacity','cvrp','instances/cvrplib/A-n32-k5.vrp',['--capacity-cuts']],
 ['E-n22-k4-cuts','cvrp','instances/cvrplib/E-n22-k4.vrp',['--capacity-cuts','--rank1']],
];
const selected=process.env.BALDES_PROFILE_CASES?.split(',');
if(selected?.some(name=>!cases.some(c=>c[0]===name))) throw Error('Unknown case selection');
const env={...process.env,OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1',MKL_NUM_THREADS:'1'};
const run=(cmd,args)=>{const p=spawnSync(cmd,args,{env,encoding:'utf8',timeout:120000,maxBuffer:32*1024*1024});if(p.error||p.status!==0)throw Error(`${cmd}: ${p.error||p.stderr||p.stdout}`);return p.stdout;};
const metadata={binary,label,sha256:crypto.createHash('sha256').update(fs.readFileSync(binary)).digest('hex'),commit:run('git',['rev-parse','HEAD']).trim(),cpu:os.cpus()[0].model,date:new Date().toISOString(),loadavg:os.loadavg(),warmups:1,measuredRuns:3,perf:{event:'cycles:u',frequency:499,callGraph:'dwarf,8192',cvrpRepetitions:20},affinity:'0,1',rustc:run('rustc',['-Vv']),build:process.env.BALDES_PROFILE_BUILD || 'unrecorded; supply BALDES_PROFILE_BUILD',threads:env.OMP_NUM_THREADS};
fs.writeFileSync(out+label+'-metadata.json',JSON.stringify(metadata,null,2));
const summary=[];
for(const [name,type,input,extra] of cases.filter(c => !selected || selected.includes(c[0]))){
 const args=['solve',input,'--master','clp','--reference-ng','--ng','8','--no-enumeration','--bound-mode','numerical','--time','45',...extra,...(type==='vrptw'?['--customers','100','--rounding','none']:['--ignore-time-windows'])];
 const rows=[];
 let expected;
 for(let i=0;i<4;i++){
  const stem=`${label}-${name}-${i}`;
  const log=run('taskset',['-c','0,1',binary,...args,'--json',out+stem+'.json']);
  fs.writeFileSync(out+stem+'.log',log);
  const j=JSON.parse(fs.readFileSync(out+stem+'.json'));
  expected ??= j;
  if(j.status!=='Optimal' || !Number.isFinite(j.lp) || !Number.isFinite(j.bound)) throw Error(`${stem}: invalid result ${j.status}`);
  if(Math.abs(j.lp-expected.lp)>1e-6 || Math.abs(j.bound-expected.bound)>1e-5) throw Error(`${stem}: objective/bound mismatch`);
  if(i)rows.push(Object.fromEntries(['status','seconds','pricing_seconds','master_seconds','lp','bound','iterations','pricing_calls','simplex_iterations','cuts'].map(k=>[k,j[k]])));
  console.log(stem,j.seconds,j.pricing_seconds,j.master_seconds);
 }
 const median=k=>rows.map(r=>r[k]).sort((a,b)=>a-b)[1];
 summary.push({name,args,rows,median:Object.fromEntries(['seconds','pricing_seconds','master_seconds'].map(k=>[k,median(k)]))});
 fs.writeFileSync(out+label+'-summary.json',JSON.stringify(summary,null,2));
 if(label==='baseline'){
  const repeatArgs=type==='cvrp'?['bash','-c','for ((i=0;i<20;i++)); do "$@" > /dev/null || exit; done','--']:[];
  const stem=`${label}-${name}-perf`;
  const log=run('perf',['record','-q','-e','cycles:u','-F','499','--call-graph','dwarf,8192','-o',out+stem+'.data','--',...repeatArgs,'taskset','-c','0,1',binary,...args,'--json',out+stem+'.json']);
  fs.writeFileSync(out+stem+'.log',log);
  for(const [kind,opts] of [['self',['--no-children']],['inclusive',['--children']]]){
   fs.writeFileSync(out+stem+'-'+kind+'.txt',run('perf',['report','--stdio','--percent-limit','0.5','--sort','symbol','--demangle','--call-graph','none','-i',out+stem+'.data',...opts]));
  }
 }
}
