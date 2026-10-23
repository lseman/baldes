"""Matched end-to-end CPU/CUDA CG runs; run after building cuda,clp.

python3 scripts/benchmarks/benchmark_pstep_cuda.py [output-directory]
"""
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'results/pstep-cuda')
out.mkdir(parents=True, exist_ok=True)
binary = pathlib.Path('target/release/baldes').resolve()
runs = []
for instance in ['P-n16-k8', 'E-n22-k4', 'A-n32-k5']:
    for p in [2, 3, 4]:
        bound = None
        for repetition in range(3):
            configs = [('enumeration', 16), ('cuda', 1)]
            if repetition % 2:
                configs.reverse()
            for method, threads in configs:
                path = out / f'{instance}-p{p}-{method}-{repetition}.json'
                cmd = [str(binary), 'pstep', f'instances/cvrplib/{instance}.vrp',
                       '--p', str(p), '--threads', str(threads), '--pricing', method,
                       '--rounding', 'none', '--time', '30', '--labels', '5000000',
                       '--json', str(path)]
                run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if run.returncode:
                    raise RuntimeError(f'{cmd}: {run.stderr}')
                data = json.loads(path.read_text())
                if data['status'] == 'Converged':
                    if bound is not None:
                        assert abs(data['lp_bound'] - bound) < 1e-5, (instance, p, data)
                    bound = data['lp_bound']
                runs.append(dict(instance=instance, p=p, method=method,
                                 threads=threads, repetition=repetition, **data))
                (out / 'cg-results.json').write_text(json.dumps(runs, indent=2) + '\n')
                print(instance, p, method, data['status'], data['elapsed_seconds'], flush=True)
meta = dict(binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            platform=platform.platform(), rounding='none', repetitions=3, warmup_runs=0,
            gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
            cpu=next(line.split(':', 1)[1].strip() for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
            inputs={name: hashlib.sha256(pathlib.Path(f'instances/cvrplib/{name}.vrp').read_bytes()).hexdigest() for name in ['P-n16-k8', 'E-n22-k4', 'A-n32-k5']},
            rustc=subprocess.check_output(['rustc', '--version'], text=True).strip())
(out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
