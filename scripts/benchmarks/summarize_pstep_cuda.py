"""Summarize matched CUDA benchmark artifacts as medians and speed ratios."""
import csv
import json
import pathlib
import statistics
import sys

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'results/pstep-cuda')
rows = list(csv.DictReader((out / 'pricing.csv').open()))
summary = {'pricing': [], 'column_generation': []}
for n, p in sorted({(int(r['customers']), int(r['p'])) for r in rows}):
    group = [r for r in rows if int(r['customers']) == n and int(r['p']) == p]
    record = dict(customers=n, p=p)
    for method, threads in sorted({(r['method'], int(r['threads'])) for r in group}):
        times = [float(r['seconds']) for r in group if r['method'] == method and int(r['threads']) == threads]
        record[f'{method}_{threads}'] = dict(median=statistics.median(times), min=min(times), max=max(times), runs=len(times))
    record['speedup_vs_cpu16'] = record['Enumeration_16']['median'] / record['Cuda_1']['median']
    summary['pricing'].append(record)
if (out / 'cg-results.json').exists():
    runs = json.loads((out / 'cg-results.json').read_text())
    for instance, p in sorted({(r['instance'], r['p']) for r in runs}):
        record = dict(instance=instance, p=p)
        for method in ['enumeration', 'cuda']:
            group = [r for r in runs if r['instance'] == instance and r['p'] == p and r['method'] == method]
            record[method] = dict(seconds=statistics.median(r['elapsed_seconds'] for r in group),
                                  pricing=statistics.median(r['pricing_seconds'] for r in group),
                                  statuses=sorted({r['status'] for r in group}), runs=len(group),
                                  bound=group[0]['lp_bound'])
        record['speedup'] = record['enumeration']['seconds'] / record['cuda']['seconds']
        summary['column_generation'].append(record)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
