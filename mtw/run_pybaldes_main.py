"""Run the paper's 18 ordinary DSN instances with pybaldes pricing.

Build pybaldes with clp,multi-time-windows. From the repository root:
    python mtw/run_pybaldes_main.py --time-limit 300

Uses the original instance generator and keeps the historical result files intact.
The default preserves the Python BCP master/search, including diving and dynamic
NG growth. Rank-one cuts and bucket-arc elimination are off. --backend native
selects the native shared-master tree without those search features.
NG growth is capped at 15 to match native memory capacity. Bounds are numerical.
--backend python-labeling uses the original labeler with the same feature settings.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import dsn_instances
from pybaldes import NetworkProblem, PricingArc, PricingNetwork, PricingResource, SharedRow

HERE = Path(__file__).resolve().parent
GRID = [(per, n, seed) for per in (2, 3) for n in (30, 60, 90) for seed in (1, 2, 3)]


def make_model(data, ng_size=8):
    """Same grouping, windows, setup times, and visit-count objective as bp_mtw."""
    jobs, dist, antennas = dsn_instances.to_jobs(data)
    if any(j.antennas_required != 1 for j in jobs):
        raise ValueError("main benchmark supports ordinary requests only")
    end = len(jobs) - 1
    horizon = data["horizon"]
    groups = {}
    for antenna in range(antennas):
        windows = tuple(tuple((e, l) for e, l in jobs[v].antenna_windows[antenna]
                              if e + jobs[v].duration <= l) for v in range(1, end))
        groups.setdefault(windows, []).append(antenna)
    networks = []
    for gid, (windows, members) in enumerate(groups.items()):
        earliest = {0: 0.}
        latest = {end: float(horizon)}
        starts = [None] * len(jobs)
        for v in range(1, end):
            if windows[v - 1]:
                starts[v] = tuple((e, l - jobs[v].duration) for e, l in windows[v - 1])
                earliest[v] = min(e for e, _ in windows[v - 1]) + jobs[v].duration
                latest[v] = max(l for _, l in windows[v - 1]) - jobs[v].duration
        arcs = []
        for u in sorted(earliest):
            for v in sorted(latest):
                if u == v or (u, v) == (0, end) or earliest[u] + dist[u][v] > latest[v] + 1e-6:
                    continue
                arcs.append(PricingArc(
                    u, v, -1. if v != end else 0.,
                    consumption=(jobs[u].duration + dist[u][v],),
                    entries=((v - 1, 1.),) if v != end else (),
                ))
        ng = [(v,) for v in range(len(jobs))]
        for v in range(1, end):
            if not starts[v]:
                continue
            # Reproduce build_static's nearest jobs by travel + finish-window midpoint.
            center = .5 * (earliest[v] + max(l for _, l in windows[v - 1]))
            def closeness(w):
                other = .5 * (earliest[w] + max(l for _, l in windows[w - 1]))
                return dist[v][w] + abs(other - center)
            near = sorted((w for w in range(1, end) if w != v and starts[w]), key=closeness)
            ng[v] = tuple(sorted([v] + near[:ng_size - 1]))
        networks.append(PricingNetwork(
            gid, len(jobs), (0., len(members)), arcs,
            resources=(PricingResource(initial=0., bounds=((0., horizon),) * len(jobs),
                                       time_windows=starts),), ng=ng,
        ))
    return NetworkProblem([SharedRow("le", 1.) for _ in range(end - 1)], networks)


def validate_witness(data, result):
    """Independently replay returned integer routes against full-service windows."""
    if result.incumbent is None:
        return
    jobs, dist, antennas = dsn_instances.to_jobs(data)
    signatures = {}
    for a in range(antennas):
        signature = tuple(tuple((e, l) for e, l in j.antenna_windows[a]
                                if e + j.duration <= l) for j in jobs[1:-1])
        signatures.setdefault(signature, []).append(a)
    groups = list(signatures.items())
    counts = [0] * len(groups)
    seen = set()
    total_cost = 0.
    for route in result.routes + result.fixed_routes:
        if abs(route.value - round(route.value)) > 1e-6 or route.value < 0:
            raise ValueError("noninteger incumbent route quantity")
        for _ in range(round(route.value)):
            windows, members = groups[route.network]
            counts[route.network] += 1
            if counts[route.network] > len(members):
                raise ValueError("too many routes for an antenna group")
            if route.nodes[0] != 0 or route.nodes[-1] != len(jobs) - 1:
                raise ValueError("invalid incumbent depots")
            finish = 0.
            for u, v in zip(route.nodes, route.nodes[1:]):
                arrival = finish + dist[u][v]
                if v == len(jobs) - 1:
                    if arrival > data["horizon"] + 1e-6:
                        raise ValueError("route finishes after horizon")
                    continue
                if v in seen:
                    raise ValueError("job served more than once")
                seen.add(v)
                start = next((max(arrival, e) for e, l in windows[v - 1]
                              if max(arrival, e) + jobs[v].duration <= l + 1e-6), None)
                if start is None:
                    raise ValueError("service does not fit a window")
                finish = start + jobs[v].duration
            expected = -(len(route.nodes) - 2)
            if abs(route.cost - expected) > 1e-6:
                raise ValueError("route cost differs from visit count")
            total_cost += route.cost
    if abs(total_cost - result.incumbent) > 1e-6 or abs(len(seen) + result.incumbent) > 1e-6:
        raise ValueError("incumbent objective differs from independently checked routes")


def baseline():
    rows = {}
    for line in (HERE / 'results/v2/runs.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['variant'] == 'full' and r.get('rho', 0.) == 0.:
            rows[r['instance']] = r
    return rows


def run_case(per, n, seed, limit, out, backend='python-search'):
    import pybaldes
    data = dsn_instances.generate(n, seed, days=1, antennas_per_complex=per, dur=(120, 300))
    name = data['name']
    model = make_model(data)
    text = model.to_json()
    (out / f'{name}.model.json').write_text(text)
    dsn_instances.save(data, out / f'{name}.instance.json')
    print(f'START {name}', flush=True)
    started = time.perf_counter()
    if backend == 'native':
        result = model.solve(tree=True, time_limit=limit, node_limit=100000,
                             max_iterations=100000, bound_mode='numerical')
        elapsed = time.perf_counter() - started
        validate_witness(data, result)
        (out / f'{name}.solution.json').write_text(result.to_json())
        incumbent = None if result.incumbent is None else round(-result.incumbent)
        bound = None if result.bound is None else -result.bound
        status, nodes, open_nodes = result.status, result.nodes, result.open_nodes
        calls, solver_seconds = result.pricing_calls, result.seconds
    else:
        if backend == 'python-labeling':
            from bp_mtw import solve_bcp
        else:
            from bp_mtw_baldes import solve_bcp
        result = solve_bcp(*dsn_instances.to_jobs(data), verbose=False, time_limit=limit,
                           parallel=False, root_cut_rounds=0, node_cut_rounds=0,
                           arc_elimination=False, ng_max=15)
        elapsed = time.perf_counter() - started
        (out / f'{name}.solution.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        incumbent = result['value']
        bound = result['bound'] if not result['unresolved_nodes'] else None
        status = ('Incomplete' if result['unresolved_nodes'] else
                  'Optimal' if result['optimal'] else 'TimeLimit')
        nodes, open_nodes = result['nodes'], None
        calls, solver_seconds = result.get('baldes_calls'), result['time']
    old = baseline()[name]
    row = dict(instance=name, n=n, antennas=data['antennas'], seed=seed,
               method=f'pybaldes-{backend}', status=status,
               incumbent=incumbent, bound=bound,
               optimal=status == 'Optimal', time=solver_seconds, wall_seconds=elapsed,
               nodes=nodes, open_nodes=open_nodes, pricing_calls=calls,
               time_limit=limit, witness_validated=incumbent is not None,
               model_sha256=hashlib.sha256(text.encode()).hexdigest(),
               baseline_incumbent=old['incumbent'], baseline_time=old['time'],
               baseline_optimal=old['optimal'], pybaldes_version=pybaldes.__version__,
               python=platform.python_version(), bound_mode='numerical',
               cpu_affinity=sorted(os.sched_getaffinity(0)),
               features=dict(rank_one_cuts=False, python_bucket_arc_elimination=False,
                             diving=backend != 'native',
                             dynamic_ng_growth=backend != 'native', ng_size=8,
                             ng_max=15 if backend != 'native' else 8))
    row['matches_baseline'] = row['incumbent'] == old['incumbent']
    if backend == 'python-search':
        row['pricing_adapter'] = 'cached-staged-v2'
        for key in ('baldes_native_calls', 'baldes_preparations', 'baldes_cached_results'):
            row[key] = result[key]
    with (out / 'runs.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row), flush=True)


def report(out):
    rows = {}
    path = out / 'runs.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows[r['instance']] = r
    ordered = sorted(rows.values(), key=lambda r: (r['antennas'], r['n'], r['seed']))
    matched_python = bool(ordered and ordered[0]['method'] == 'pybaldes-python-labeling')
    solver_label = 'Python matched' if matched_python else 'BALDES'
    fields = ['instance', 'baseline_incumbent', 'incumbent', 'bound', 'status',
              'baseline_time', 'time', 'nodes', 'matches_baseline']
    with (out / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ordered)
    tex = [r'\begin{tabular}{lrrrrrr}', r'\hline',
           'Instance & Python full $z$ & ' + solver_label + r' $z$ & bound & Python full s & '
           + solver_label + r' s & nodes \\', r'\hline']
    for r in ordered:
        value = '--' if r['incumbent'] is None else str(r['incumbent'])
        bound = '--' if r['bound'] is None else f"{r['bound']:.2f}"
        tex.append(f"{r['instance'].replace('_', chr(92) + '_')} & {r['baseline_incumbent']} & {value} & {bound} & {r['baseline_time']:.2f} & {r['time']:.2f} & {r['nodes']} " + r'\\')
    tex.extend([r'\hline', r'\end{tabular}'])
    (out / 'comparison.tex').write_text('\n'.join(tex) + '\n')
    grouped = []
    for per in (2, 3):
        for n in (30, 60, 90):
            rs = [r for r in ordered if r['n'] == n and r['antennas'] == per * 3]
            if not rs:
                continue
            complete = all(r['incumbent'] is not None and r['bound'] is not None for r in rs)
            grouped.append(dict(
                n=n, antennas=per * 3, completed=len(rs), expected=3,
                python_optimal=sum(r['baseline_optimal'] for r in rs),
                python_mean_value=statistics.fmean(r['baseline_incumbent'] for r in rs),
                python_mean_seconds=statistics.fmean(r['baseline_time'] for r in rs),
                baldes_optimal=sum(r['optimal'] for r in rs),
                baldes_mean_value=statistics.fmean(r['incumbent'] for r in rs) if complete else None,
                baldes_mean_gap_percent=statistics.fmean(
                    100. * max(0., r['bound'] - r['incumbent']) / r['bound']
                    if r['bound'] > 1e-6 else 0. for r in rs) if complete else None,
                baldes_mean_seconds=statistics.fmean(r['time'] for r in rs),
                baldes_mean_nodes=statistics.fmean(r['nodes'] for r in rs),
            ))
    if grouped:
        exported = [{(k.replace('baldes_', 'matched_python_') if matched_python else k): v
                     for k, v in g.items()} for g in grouped]
        with (out / 'main_table.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(exported[0]))
            writer.writeheader()
            writer.writerows(exported)
        table = [r'% Python baseline: full BCP. BALDES pricing: cuts and elimination off.',
                 r'% Values and times are arithmetic means over the completed seeds.',
                 r'\begin{tabular}{rr rrr rrrrr}', r'\hline',
                 r'$n$ & $|A|$ & Python full opt & Python full $z$ & Python full s & '
                 + solver_label + ' opt & ' + solver_label + r' $z$ & gap (\%) & '
                 + solver_label + r' s & nodes \\',
                 r'\hline']
        for g in grouped:
            value = '--' if g['baldes_mean_value'] is None else f"{g['baldes_mean_value']:.1f}"
            gap = '--' if g['baldes_mean_gap_percent'] is None else f"{g['baldes_mean_gap_percent']:.2f}"
            table.append(
                f"{g['n']} & {g['antennas']} & {g['python_optimal']}/{g['completed']} & "
                f"{g['python_mean_value']:.1f} & {g['python_mean_seconds']:.2f} & "
                f"{g['baldes_optimal']}/{g['completed']} & {value} & {gap} & "
                f"{g['baldes_mean_seconds']:.2f} & {g['baldes_mean_nodes']:.1f} " + r'\\')
        table.extend([r'\hline', r'\end{tabular}'])
        (out / 'main_table.tex').write_text('\n'.join(table) + '\n')
    summary = dict(completed=len(ordered), expected=18,
                   optimal=sum(r['optimal'] for r in ordered),
                   matching_incumbents=sum(r['matches_baseline'] for r in ordered),
                   validated_witnesses=sum(r['witness_validated'] for r in ordered),
                   total_seconds=sum(r['time'] for r in ordered))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    compare_adapters(out, ordered)
    compare_labelers()
    print(json.dumps(summary), flush=True)


def compare_adapters(out, rows):
    previous = HERE / 'results/pybaldes-pricing-main-20261025/runs.jsonl'
    if not previous.exists() or out.resolve() == previous.parent.resolve():
        return
    before = {r['instance']: r for r in map(json.loads, previous.read_text().splitlines())}
    pairs = [(before[r['instance']], r) for r in rows if r['instance'] in before
             and r.get('pricing_adapter') == 'cached-staged-v2']
    for old, new in pairs:
        if old['model_sha256'] != new['model_sha256'] or old['features'] != new['features']:
            raise ValueError('adapter comparison requires identical models and feature settings')
    if not pairs:
        return
    records = [dict(instance=new['instance'], previous_seconds=old['time'],
                    optimized_seconds=new['time'], previous_nodes=old['nodes'],
                    optimized_nodes=new['nodes'], objective=new['incumbent'],
                    matches_previous=new['incumbent'] == old['incumbent'],
                    native_calls=new['baldes_native_calls'],
                    solver_preparations=new['baldes_preparations'],
                    cached_results=new['baldes_cached_results']) for old, new in pairs]
    with (out / 'adapter_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    table = [r'% Mean seconds over three seeds; previous and optimized BALDES use identical feature settings.',
             r'\begin{tabular}{rrrrr}', r'\hline',
             r'$n$ & $|A|$ & Python full BCP & Previous BALDES & Optimized BALDES \\', r'\hline']
    for per in (2, 3):
        for n in (30, 60, 90):
            group = [(a, b) for a, b in pairs if b['n'] == n and b['antennas'] == per * 3]
            if not group:
                continue
            mean = statistics.fmean
            table.append(f"{n} & {per * 3} & {mean(b['baseline_time'] for a, b in group):.2f} & "
                         f"{mean(a['time'] for a, b in group):.2f} & "
                         f"{mean(b['time'] for a, b in group):.2f} " + r'\\')
    table.extend([r'\hline', r'\end{tabular}'])
    (out / 'adapter_comparison.tex').write_text('\n'.join(table) + '\n')


def compare_labelers():
    optimized = HERE / 'results/pybaldes-pricing-reuse-20261025'
    python = HERE / 'results/python-labeling-matched-20261025'
    if not all((p / 'runs.jsonl').exists() for p in (optimized, python)):
        return
    def load(path):
        return {r['instance']: r for r in map(json.loads, (path / 'runs.jsonl').read_text().splitlines())}
    native, reference = load(optimized), load(python)
    pairs = [(reference[name], native[name]) for name in sorted(native.keys() & reference.keys())]
    for a, b in pairs:
        if any(a[key] != b[key] for key in ('model_sha256', 'features', 'cpu_affinity', 'time_limit')):
            raise ValueError('labeler comparison requires matching inputs, settings and CPU affinity')
    records = [dict(instance=b['instance'], python_seconds=a['time'], baldes_seconds=b['time'],
                    python_nodes=a['nodes'], baldes_nodes=b['nodes'],
                    python_value=a['incumbent'], baldes_value=b['incumbent'],
                    python_status=a['status'], baldes_status=b['status'],
                    same_objective=a['incumbent'] == b['incumbent']) for a, b in pairs]
    with (optimized / 'matched_backend_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    table = [r'% Matched feature settings; mean seconds over completed seeds, single runs.',
             r'\begin{tabular}{rrrrrr}', r'\hline',
             r'$n$ & $|A|$ & Python opt & BALDES opt & Python s & BALDES s \\', r'\hline']
    for per in (2, 3):
        for n in (30, 60, 90):
            group = [(a, b) for a, b in pairs if b['n'] == n and b['antennas'] == per * 3]
            if group:
                table.append(f"{n} & {per * 3} & {sum(a['optimal'] for a, b in group)}/{len(group)} & "
                             f"{sum(b['optimal'] for a, b in group)}/{len(group)} & "
                             f"{statistics.fmean(a['time'] for a, b in group):.2f} & "
                             f"{statistics.fmean(b['time'] for a, b in group):.2f} " + r'\\')
    table.extend([r'\hline', r'\end{tabular}'])
    (optimized / 'matched_backend_comparison.tex').write_text('\n'.join(table) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time-limit', type=float, default=300.)
    parser.add_argument('--out', type=Path, default=HERE / 'results/pybaldes-pricing-reuse-20261025')
    parser.add_argument('--backend', choices=['python-search', 'native', 'python-labeling'],
                        default='python-search')
    parser.add_argument('--case', nargs=3, type=int, metavar=('PER', 'N', 'SEED'))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--cpus', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cpus:
        os.sched_setaffinity(0, {int(c) for c in args.cpus.split(',')})
    if args.case:
        run_case(*args.case, args.time_limit, args.out, args.backend)
        return
    done = set()
    if (args.out / 'runs.jsonl').exists():
        previous = [json.loads(s) for s in (args.out / 'runs.jsonl').read_text().splitlines()]
        if any(r['method'] != f'pybaldes-{args.backend}' or r['time_limit'] != args.time_limit
               for r in previous):
            parser.error('existing output uses another backend or time limit; choose a new --out')
        if args.backend == 'python-search' and any(
                r.get('pricing_adapter') != 'cached-staged-v2' for r in previous):
            parser.error('existing output uses another pricing adapter; choose a new --out')
        done = {r['instance'] for r in previous}
    tasks = [(p, n, s) for p, n, s in GRID if f'dsn_n{n}_c3x{p}_d1_s{s}' not in done]
    cpus = sorted(set(os.sched_getaffinity(0)) - {10, 11})
    workers = min(args.workers, len(cpus), len(tasks))
    if args.workers < 1:
        parser.error('--workers must be positive')
    def run_block(index):
        # Disjoint CPU blocks prevent simultaneous instances sharing a core.
        block = cpus[index * len(cpus) // workers:(index + 1) * len(cpus) // workers]
        for per, n, seed in tasks[index::workers]:
            completed = subprocess.run([
                sys.executable, str(Path(__file__).resolve()), '--case', str(per), str(n), str(seed),
                '--time-limit', str(args.time_limit), '--out', str(args.out.resolve()),
                '--backend', args.backend,
                '--cpus', ','.join(map(str, block)),
            ], check=False)
            if completed.returncode:
                raise RuntimeError(f'benchmark case {(per, n, seed)} failed: {completed.returncode}')
    if workers:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_block, i) for i in range(workers)]
            for future in as_completed(futures):
                future.result()
                report(args.out)
    report(args.out)


if __name__ == '__main__':
    main()
