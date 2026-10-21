// From the repository root on Linux, compare builds/options on root LP workloads.
// Times do not measure integer tree solving. Example:
// node scripts/benchmark_solvers.mjs scalar=/tmp/baldes-scalar simd=target/release/baldes
// BALDES_PROFILE_CASES: comma-separated case names; BALDES_PROFILE_RUNS: odd count.
// BALDES_PROFILE_OPTIONS: JSON mapping variant labels to extra CLI arguments.
// BALDES_PROFILE_CPUS: taskset affinity (default 0,1); BALDES_PROFILE_OUT: output directory.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import crypto from 'node:crypto';
import os from 'node:os';
const out=path.resolve(process.env.BALDES_PROFILE_OUT||`results/solver-profile-${new Date().toISOString().replaceAll(':','-')}`)+path.sep;
const variants=process.argv.slice(2).map(x=>{const i=x.indexOf('=');return [x.slice(0,i),x.slice(i+1)]});
if(variants.length<2) throw Error('Pass label=/absolute/binary for at least two variants');
const selected=process.env.BALDES_PROFILE_CASES?.split(',');
const definitions=[
 ['C202','vrptw','instances/C202.txt',[]],
 ['R201-rank1','vrptw','instances/R201.txt',['--rank1']],
 ['R103-rank1','vrptw','instances/R103.txt',['--rank1']],
 ['A-n32-k5','cvrp','instances/cvrplib/A-n32-k5.vrp',[]],
 ['A-n32-k5-capacity','cvrp','instances/cvrplib/A-n32-k5.vrp',['--capacity-cuts']],
 ['E-n22-k4-cuts','cvrp','instances/cvrplib/E-n22-k4.vrp',['--capacity-cuts','--rank1']],
];
const cases=definitions.filter(([name])=>!selected||selected.includes(name)).map(([name,type,input,extra])=>({name,args:[type,input,'--master','clp','--reference-ng','--ng','8','--no-enumeration','--bound-mode','numerical','--time','45',...extra,...(type==='vrptw'?['--customers','100','--rounding','none']:[])]}));
if(!cases.length||selected?.some(name=>!definitions.some(([n])=>n===name))) throw Error('Unknown or empty case selection');
const env={...process.env,OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1',MKL_NUM_THREADS:'1'};
const variantOptions=JSON.parse(process.env.BALDES_PROFILE_OPTIONS||'{}');
const key=variants.map(([n])=>n).join('-');
const runs=Number(process.env.BALDES_PROFILE_RUNS||3);
if(!Number.isSafeInteger(runs)||runs<1||runs%2!==1) throw Error('Run count must be a positive odd integer');
if(variants.some(([name,bin])=>!/^[\w-]+$/.test(name)||!bin)||new Set(variants.map(([name])=>name)).size!==variants.length) throw Error('Variant labels must be unique letters, numbers, underscores or hyphens');
const affinity=process.env.BALDES_PROFILE_CPUS||'0,1';
const metadata={date:new Date().toISOString(),variants:variants.map(([name,path])=>({name,path,options:variantOptions[name]||[],sha256:crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')})),affinity,runs,warmups:1,cpu:os.cpus()[0]?.model,threads:{OMP_NUM_THREADS:1,OPENBLAS_NUM_THREADS:1,MKL_NUM_THREADS:1},rustc:spawnSync('rustc',['-Vv'],{encoding:'utf8'}).stdout};
fs.mkdirSync(out,{recursive:true});
if(fs.existsSync(out+key+'-metadata.json')) throw Error('Results already exist; choose another output directory or variant labels');
fs.writeFileSync(out+key+'-metadata.json',JSON.stringify(metadata,null,2));
const summary=[];
for(const c of cases){
 let expected;
 const rows=Object.fromEntries(variants.map(([name])=>[name,[]]));
 for(let i=0;i<=runs;i++){
  const order=i%2?[...variants].reverse():variants;
  for(const [variant,bin] of order){
   const stem=`${c.name}-${variant}-${i}`;
   const r=spawnSync('taskset',['-c',affinity,bin,...c.args,...(variantOptions[variant]||[]),'--json',out+stem+'.json'],{env,encoding:'utf8',timeout:120000,maxBuffer:8*1024*1024});
   fs.writeFileSync(out+stem+'.log',(r.stdout||'')+(r.stderr||''));
   if(r.status!==0) throw Error(stem+' '+r.status+' '+r.error);
   const j=JSON.parse(fs.readFileSync(out+stem+'.json'));
   expected??=j;
   if(j.status!=='Optimal'||!Number.isFinite(j.lp)||!Number.isFinite(j.bound)||Math.abs(j.lp-expected.lp)>1e-6||Math.abs(j.bound-expected.bound)>1e-5) throw Error('Result mismatch '+stem);
   if(i) rows[variant].push(Object.fromEntries(['seconds','pricing_seconds','master_seconds','lp','bound','iterations','pricing_calls','simplex_iterations'].map(k=>[k,j[k]])));
   console.log(stem,j.seconds);
  }
 }
 const medians=Object.fromEntries(variants.map(([v])=>[v,rows[v].map(r=>r.seconds).sort((a,b)=>a-b)[Math.floor(runs/2)]]));
 summary.push({name:c.name,args:c.args,rows,medians});
 fs.writeFileSync(out+key+'-summary.json',JSON.stringify(summary,null,2));
}
console.log('Results:',out+key+'-summary.json');
