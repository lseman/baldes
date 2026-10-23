"""Compare actual VRPTW p-step column generation on C201/C202, no rounding.

Build cuda,clp first. PSTEP_SECONDS, PSTEP_RUNS, PSTEP_ITERATIONS bound work.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

out=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'results/pstep-solomon')
out.mkdir(parents=True,exist_ok=True)
binary=pathlib.Path('target/release/baldes').resolve()
seconds=int(os.environ.get('PSTEP_SECONDS','60'))
repetitions=int(os.environ.get('PSTEP_RUNS','3'))
iterations=int(os.environ.get('PSTEP_ITERATIONS','1000'))
runs=[]
for instance in ['C201','C202']:
    for p in [2,3]:
        bound=None
        for repetition in range(repetitions):
            configs=[('enumeration',16),('cuda',1)]
            if repetition%2: configs.reverse()
            for method,threads in configs:
                path=out/f'{instance}-p{p}-{method}-{repetition}.json'
                cmd=[str(binary),'pstep',f'instances/Solomon/{instance}.txt','--p',str(p),
                     '--pricing',method,'--threads',str(threads),'--rounding','none',
                     '--time',str(seconds),'--iterations',str(iterations),'--labels','5000000','--json',str(path)]
                start=time.perf_counter()
                try:
                    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=seconds+30)
                    wall=time.perf_counter()-start
                    if proc.returncode:
                        data=dict(status='PricingLimit' if 'limit' in proc.stderr else 'ExecutionError',error=proc.stderr.strip(),lp_bound=None)
                    else:
                        data=json.loads(path.read_text())
                        assert data['time_windows_enforced'] and data['customers']==100
                except subprocess.TimeoutExpired:
                    wall=time.perf_counter()-start
                    data=dict(status='ProcessTimeout',lp_bound=None)
                if data['status']=='Converged':
                    if bound is not None: assert abs(bound-data['lp_bound'])<1e-5,(instance,p,bound,data)
                    bound=data['lp_bound']
                record=dict(instance=instance,p=p,method=method,threads=threads,repetition=repetition,
                            process_seconds=wall,**{k:v for k,v in data.items() if k not in ['instance','p']})
                runs.append(record)
                (out/'cg-results.json').write_text(json.dumps(runs,indent=2)+'\n')
                print(instance,p,method,data['status'],data.get('elapsed_seconds',wall),data['lp_bound'],flush=True)
(out/'metadata.json').write_text(json.dumps(dict(binary=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    inputs={name:hashlib.sha256(pathlib.Path(f'instances/Solomon/{name}.txt').read_bytes()).hexdigest() for name in ['C201','C202']},
    rounding='none',time_windows_enforced=True,seconds=seconds,repetitions=repetitions,iterations=iterations,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
    rustc=subprocess.check_output(['rustc','--version'],text=True).strip()),indent=2)+'\n')
