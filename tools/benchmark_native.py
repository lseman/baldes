"""Reproducible Rust/C++/reference-solver root comparison with identical NG inputs.

Run after cargo build --release --features clp. OPENRSCP_ROOT points at the
C++ OpenRSCP checkout (default ../openrscp). Solver timers exclude process
startup/imports. Cases run sequentially; medians exclude one warm-up per side.
Reference cg/pricing_calls count oracle calls, not restricted-master solves.
"""
import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CPP=Path(os.environ.get('OPENRSCP_ROOT',ROOT.parent/'openrscp'))

def clean(x):
    if dataclasses.is_dataclass(x): x=dataclasses.asdict(x)
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def native(instance,n,fixing,out,rank1=False):
    sys.path[:0]=[str(CPP/'python/src'),str(CPP/'python/tools')]
    from openrscp import vrptw
    import differential
    import bapcod_reference
    inst=vrptw.read_solomon(instance).truncated(n)
    sets=differential.ng_sets(inst,8)
    binary_started=time.perf_counter()
    binary=bapcod_reference.solve_root_reference(inst,ng_sets=sets,rounding='none',rank1_cuts=rank1,capacity_cuts=False,params={'RCSPapplyReducedCostFixing':int(fixing)})
    binary=clean(binary)
    binary['seconds']=time.perf_counter()-binary_started
    started=time.perf_counter()
    cpp=differential.openrscp_root(inst,sets,master='clp',src=rank1,reduced_cost_fixing=fixing,time_limit=120.)
    data=clean({'binary':binary,'cpp':cpp,'cpp_wall_seconds':time.perf_counter()-started})
    Path(out).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def audit(record,n):
    coverage=[0.]*n;objective=0.
    for route in record['routes']:
        assert route['nodes'][0]==0 and route['nodes'][-1]==n+1
        for v in route['nodes'][1:-1]:coverage[v-1]+=route['value']
        objective+=route['value']*route['cost']
    return {'coverage_error':max(abs(v-1.) for v in coverage),'objective_error':abs(objective-record['lp'])}

def compare(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for case in args.cases:
        name,n=case.split(':');n=int(n);stem=f'{name}-{n}-fix{int(args.fixing)}'+('-r1' if args.rank1 else '');instance=CPP/'examples/instances'/f'{name}.txt'
        samples=[]
        for repeat in range(args.repeats+1):
            prefix=out/f'{stem}-{repeat}';native_json=str(prefix)+'.native.json';rust_json=str(prefix)+'.rust.json'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--native-case',str(instance),'--customers',str(n),'--native-out',native_json]
            if args.fixing:cmd+=['--fixing']
            if args.rank1:cmd+=['--rank1']
            with open(str(prefix)+'.native.log','w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout,check=True)
            cmd=[str(ROOT/'target/release/baldes-vrptw'),str(instance),'--customers',str(n),'--reference-ng','--json',rust_json,'--time',str(args.timeout-5)]
            if not args.fixing:cmd+=['--no-enumeration']
            if args.rank1:cmd+=['--rank1']
            with open(str(prefix)+'.rust.log','w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout,check=True)
            data=json.loads(Path(native_json).read_text());data['rust']=json.loads(Path(rust_json).read_text());data['rust_audit']=audit(data['rust'],n)
            if repeat:samples.append(data)
        representative=samples[0];cpp=representative['cpp'];rust=representative['rust'];binary=representative['binary']
        summary={'case':case,'fixing':args.fixing,'rank1':args.rank1,'rust_lp':rust['lp'],'cpp_lp':cpp['lp_value'],'binary_lp':binary['root_lp'],'rust_cuts':rust['cuts'],'cpp_cuts':cpp['cuts'],'binary_cuts':binary.get('cuts'),'rust_cut_rounds':rust['cut_rounds'],'cpp_cut_rounds':cpp['cut_rounds'],
            'rust_solves':rust['iterations'],'cpp_solves':cpp['iterations'],'rust_pricing_calls':rust['pricing_calls'],'cpp_pricing_calls':cpp['pricing_calls'],'binary_pricing_calls':binary['pricing_calls'],
            'rust_master_columns':rust['columns'],'cpp_master_columns':cpp['history'][-1]['columns'],'rust_route_columns':rust['route_columns'],'cpp_route_columns':cpp['columns'],'rust_audit':representative['rust_audit'],
            'rust_seconds':statistics.median(s['rust']['seconds'] for s in samples),'cpp_seconds':statistics.median(s['cpp_wall_seconds'] for s in samples),'binary_seconds':statistics.median(s['binary']['seconds'] for s in samples),
            'rust_pricing_seconds':statistics.median(s['rust']['pricing_seconds'] for s in samples),'cpp_pricing_seconds':statistics.median(s['cpp']['pricing_seconds'] for s in samples)}
        keys=['lp','min_reduced_cost','new_columns','columns','pricing_calls','misprices','alpha']
        summary['history_matches']=len(cpp['history'])==len(rust['history']) and all(all(a[k]==b[k] for k in keys) and a['phase_index']==b['phase'] for a,b in zip(cpp['history'],rust['history']))
        summary['objective_matches']=abs(rust['lp']-cpp['lp_value'])<1e-7 and abs(rust['lp']-binary['root_lp'])<1e-7
        reports.append(summary);print(json.dumps(summary),flush=True)
        (out/'summary.json').write_text(json.dumps({'metadata':{'repeats':args.repeats,'warmups':1,'python':sys.version,'platform':platform.platform(),'cuts':args.rank1,'ng':8,'reference_ng':True,'sequential':True},'cases':reports},indent=2)+'\n')
    return reports

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',nargs='*');p.add_argument('--repeats',type=int,default=3);p.add_argument('--timeout',type=float,default=120.);p.add_argument('--out',default='results/native-comparison');p.add_argument('--fixing',action='store_true');p.add_argument('--rank1',action='store_true');p.add_argument('--native-case');p.add_argument('--customers',type=int);p.add_argument('--native-out');args=p.parse_args()
    if args.native_case:return native(args.native_case,args.customers,args.fixing,args.native_out,args.rank1)
    if not args.cases:p.error('supply NAME:customers cases')
    compare(args)
if __name__=='__main__':main()
