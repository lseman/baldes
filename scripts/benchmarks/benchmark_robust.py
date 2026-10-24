#!/usr/bin/env python3
"""Run the Table 7 solver grid with independent integer-unit route replay.

This screens solver correctness; it does not run the paper's Monte Carlo risk
simulation. Defaults: 1,320 cases, four processes, 60 seconds per case.
"""
import argparse
import concurrent.futures
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {"R101", "R102", "R103", "R104", "RC105"}


def cases(instances):
    result = []
    for instance in instances:
        result.append(dict(key=f"{instance}-nominal", instance=instance, mode="nominal", gamma=0, alpha="0", gq=0, gt=0, aq="0", at="0"))
        for mode in ("q", "t", "qt"):
            for alpha in ("0.1", "0.25", "0.5"):
                for gamma in range(1, 7):
                    result.append(dict(key=f"{instance}-{mode}-a{alpha}-g{gamma}", instance=instance, mode=mode, gamma=gamma, alpha=alpha, gq=gamma if "q" in mode else 0, gt=gamma if "t" in mode else 0, aq=alpha if "q" in mode else "0", at=alpha if "t" in mode else "0"))
    return result


def load_instance(path, customers=25):
    lines = path.read_text().splitlines()
    capacity_line = next(i for i, line in enumerate(lines) if "CAPACITY" in line)
    vehicles, capacity = map(int, next(line for line in lines[capacity_line + 1:] if line.strip()).split())
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) != 7:
            continue
        try:
            row = list(map(int, parts))
        except ValueError:
            continue
        rows.append(row)
    rows = rows[:customers + 1]
    if len(rows) != customers + 1 or [r[0] for r in rows] != list(range(customers + 1)):
        raise ValueError(f"expected sequential Solomon customers: {path}")
    return dict(rows=rows, vehicles=vehicles, capacity=capacity)


def travel(rows, u, v):
    dx, dy = 10 * (rows[u][1] - rows[v][1]), 10 * (rows[u][2] - rows[v][2])
    return math.isqrt(dx * dx + dy * dy)


def validate(report, case, instance):
    """Replay with integer time units; load uses sorted deviations, not a DP."""
    errors = []
    incumbent, bound = report.get("incumbent"), report.get("bound")
    optimal = report.get("status") == "Optimal"
    if incumbent is None:
        return ["Optimal without incumbent"] if optimal else []
    if not isinstance(incumbent, (int, float)) or not math.isfinite(incumbent):
        return ["invalid incumbent"]
    rows = instance["rows"]
    sink = len(rows)
    seen, total = [], 0
    routes = report.get("routes", [])
    if len(routes) > instance["vehicles"]:
        errors.append("fleet limit exceeded")
    for route in routes:
        nodes = route.get("nodes", [])
        if len(nodes) < 3 or nodes[0] != 0 or nodes[-1] != sink or any(type(v) is not int or not 1 <= v < sink for v in nodes[1:-1]):
            errors.append("invalid route vertices")
            continue
        if len(route.get("arcs", [])) != len(nodes) - 1:
            errors.append("arc/vertex count mismatch")
        customers = nodes[1:-1]
        seen.extend(customers)
        aq, at = Fraction(case["aq"]), Fraction(case["at"])
        nominal = sum(rows[v][3] for v in customers)
        deviations = sorted((int(aq * rows[v][3]) for v in customers), reverse=True)
        if nominal + sum(deviations[:case["gq"]]) > instance["capacity"]:
            errors.append("robust capacity violation")
        budget = min(case["gt"], len(nodes) - 1)
        starts = [10 * rows[0][4]] * (budget + 1)
        cost = 0
        for u, v in zip(nodes, nodes[1:]):
            v = 0 if v == sink else v
            duration = travel(rows, u, v)
            cost += duration
            shift = 10 * rows[u][6] + duration
            deviation = int(at * duration)
            previous = starts
            starts = [max(10 * rows[v][4], previous[0] + shift)]
            for gamma in range(1, budget + 1):
                starts.append(max(10 * rows[v][4], previous[gamma] + shift, previous[gamma - 1] + shift + deviation))
            if starts[-1] > 10 * rows[v][5]:
                errors.append("robust time-window violation")
        if route.get("cost") is None or abs(route["cost"] - cost / 10) > 1e-6:
            errors.append("route cost mismatch")
        total += cost
    if sorted(seen) != list(range(1, sink)):
        errors.append("customer cover is not a partition")
    if abs(incumbent - total / 10) > 1e-6:
        errors.append("incumbent cost mismatch")
    if bound is not None:
        if not isinstance(bound, (int, float)) or not math.isfinite(bound):
            errors.append("invalid finite bound")
        elif bound > incumbent + 1e-6:
            errors.append("bound exceeds incumbent")
        elif optimal and incumbent - bound > 1e-6 + 1e-9 * max(abs(incumbent), abs(bound)) + 1e-8:
            errors.append("Optimal with open gap")
    elif optimal:
        errors.append("Optimal without bound")
    return sorted(set(errors))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_case(case, binary, output, seconds, instance):
    base = output / "cases" / case["key"]
    # Keys contain decimal alpha values, so append extensions rather than
    # replacing an apparent suffix in the case identity.
    report_path = Path(str(base) + ".solver.json")
    command = [str(binary), "solve", str(ROOT / "instances/Solomon" / (case["instance"] + ".txt")), "--customers", "25", "--tree", "--cuts", "--rounding", "tenth", "--robust", f"{case['gq']},{case['gt']},{case['aq']},{case['at']}", "--time", str(seconds), "--json", str(report_path)]
    started = time.monotonic()
    result = dict(case, command=command, status="Error", validation_errors=[])
    with Path(str(base) + ".log").open("w") as log:
        try:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=seconds + 15, check=False)
            result["returncode"] = process.returncode
            if process.returncode:
                result["validation_errors"] = [f"solver exited {process.returncode}"]
            elif not report_path.exists():
                result["validation_errors"] = ["missing solver JSON"]
            else:
                report = json.loads(report_path.read_text())
                result.update({key: report.get(key) for key in ("status", "incumbent", "bound", "nodes", "pricing_calls", "seconds")})
                result["validation_errors"] = validate(report, case, instance)
        except subprocess.TimeoutExpired:
            result["status"] = "WatchdogTimeout"
        except (OSError, ValueError, TypeError, KeyError) as error:
            result["validation_errors"] = [f"{type(error).__name__}: {error}"]
    result["wall_seconds"] = time.monotonic() - started
    atomic_json(Path(str(base) + ".result.json"), result)
    return result


def summarize(results, expected, output):
    statuses = Counter(r["status"] for r in results)
    invalid = [r["key"] for r in results if r["validation_errors"]]
    by_instance = defaultdict(list)
    for result in results:
        by_instance[result["instance"]].append(result)
    contradictions = []
    for group in by_instance.values():
        for weaker in group:
            for stronger in group:
                if weaker.get("bound") is None or stronger.get("incumbent") is None:
                    continue
                if all(Fraction(weaker[k]) <= Fraction(stronger[k]) for k in ("gq", "gt", "aq", "at")) and weaker["bound"] > stronger["incumbent"] + 1e-6:
                    contradictions.append([weaker["key"], stronger["key"]])
    summary = dict(completed=len(results), expected=expected, statuses=dict(statuses), invalid_cases=invalid, monotonicity_contradictions=contradictions, all_optimal=len(results) == expected and statuses.get("Optimal", 0) == expected and not invalid and not contradictions)
    atomic_json(output / "summary.json", summary)
    fields = ["key", "instance", "mode", "gamma", "alpha", "status", "incumbent", "bound", "nodes", "pricing_calls", "wall_seconds", "validation_errors"]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["key"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/baldes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--time", type=int, default=60, help="seconds per case; paper used 3600")
    parser.add_argument("--instances", nargs="+", help="optional subset for a smoke sweep")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or args.time <= 0:
        parser.error("jobs must be 1..8 and time positive")
    instances = sorted(args.instances or [p.stem for p in (ROOT / "instances/Solomon").glob("*.txt") if p.stem.startswith(("C1", "R1", "RC1")) and p.stem not in EXCLUDED])
    grid = cases(instances)
    output = (args.output or ROOT / "results" / ("robust-table7-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    binary = output / "baldes"
    shutil.copy2(args.binary.resolve(), binary)
    data = {name: load_instance(ROOT / "instances/Solomon" / (name + ".txt")) for name in instances}
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), instances=instances, cases=len(grid), customers=25, jobs=args.jobs, seconds_per_case=args.time, paper_seconds_per_case=3600, paper="papers/munari_robust.pdf, pp. 24-25, Table 7", excluded=sorted(EXCLUDED), risk_simulation=False, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), tracked_diff_sha256=hashlib.sha256(diff).hexdigest(), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), input_sha256={name: hashlib.sha256((ROOT / "instances/Solomon" / (name + ".txt")).read_bytes()).hexdigest() for name in instances}, grid=grid)
    manifest["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT / "src").rglob("*.rs")}
    manifest["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    shutil.copy2(__file__, output / "benchmark_robust.py.snapshot")
    atomic_json(output / "manifest.json", manifest)
    print(f"OUTPUT={output}\nCASES={len(grid)} JOBS={args.jobs} TIME={args.time}s", flush=True)
    results = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_case, case, binary, output, args.time, data[case["instance"]]) for case in grid]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0 or result["validation_errors"] or len(results) == len(grid):
                summary = summarize(results, len(grid), output)
                print(f"{len(results)}/{len(grid)} elapsed={time.monotonic()-started:.1f}s statuses={summary['statuses']} invalid={len(summary['invalid_cases'])} contradictions={len(summary['monotonicity_contradictions'])}", flush=True)
    summary = summarize(results, len(grid), output)
    summary["wall_seconds"] = time.monotonic() - started
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["all_optimal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
