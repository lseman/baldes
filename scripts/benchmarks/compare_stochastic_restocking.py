#!/usr/bin/env python3
"""Compare Theta and Bellman pricing on local SGJR CVRPSD files.

Preserve the literal cost matrix and the documented independent five-point
PMF. Both methods use optimal restocking without an additional failure penalty.
Build the release binary separately; compile time is excluded from run budgets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(path):
    tokens = path.read_text().split()
    n, fleet, fill, capacity = int(tokens[0]), int(tokens[1]), float(tokens[2]), int(tokens[3])
    if len(tokens) != 4 + n*n + n:
        raise ValueError(f"unexpected SGJR shape: {path}")
    costs = [[float(x) for x in tokens[4+i*n:4+(i+1)*n]] for i in range(n)]
    means = [int(x) for x in tokens[4+n*n:]]
    if means[0] != 0 or any(d not in (3, 8, 13) for d in means[1:]):
        raise ValueError(f"unexpected SGJR demand classes: {path}")
    return {"schema": "baldes.stochastic.generated.v1", "instance_id": path.name,
            "provenance": {"original_benchmark_instance": True, "source_file": str(path),
                           "source_sha256": digest(path), "failure_penalty_applied": 0.,
                           "distance_rounding": "none; literal input matrix"},
            "nodes_including_depot": n, "depot": 0, "vehicles": fleet,
            "capacity": capacity, "target_fill_rate": fill, "costs": costs,
            "demands": [{"customer": i, "support": list(range(d-2, d+3)),
                         "probabilities": [.1, .2, .4, .2, .1]}
                        for i, d in enumerate(means[1:], 1)]}


def validate(data, answer):
    routes = answer.get("routes", [])
    if sorted(j for r in routes for j in r) != list(range(1, data["nodes_including_depot"])):
        return None
    if len(routes) > data["vehicles"] or any(not r for r in routes):
        return None
    q, c = data["capacity"], data["costs"]
    objective = 0.
    for route in routes:
        demands = [data["demands"][j-1] for j in route]
        if sum(sum(d*p for d, p in zip(x["support"], x["probabilities"])) for x in demands) > q+1e-9:
            return None
        values = [c[route[-1]][0]]*(q+1)
        # Independent evaluator: enumerate each demand outcome at each load.
        def expectation(j, load, suffix):
            distribution = data["demands"][j-1]
            total = 0.
            for d, p in zip(distribution["support"], distribution["probabilities"]):
                trips = max(0, (d-load+q-1)//q)
                total += p*(trips*(c[j][0]+c[0][j])+suffix[load+trips*q-d])
            return total
        for i, j in reversed(list(zip(route, route[1:]))):
            refill = c[i][0]+c[0][j]+expectation(j, q, values)
            values = [min(c[i][j]+expectation(j, load, values), refill) for load in range(q+1)]
        objective += c[0][route[0]]+expectation(route[0], q, values)
    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/examples/stochastic_instance"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--hgs-seconds", type=float, default=2.)
    parser.add_argument("--incumbent-search-seconds", type=float, default=0.)
    parser.add_argument("--completion-subset-size", type=int, choices=range(7))
    parser.add_argument("--methods", nargs="+", choices=("theta", "bellman"), default=["theta", "bellman"])
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.hgs_seconds < args.seconds:
        parser.error("require 0 < HGS seconds < finite solve seconds")
    if not math.isfinite(args.incumbent_search_seconds) or args.incumbent_search_seconds < 0:
        parser.error("incumbent search seconds must be finite and nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve(strict=True)
    env = os.environ.copy()
    for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    metadata = {"binary_sha256": digest(binary), "script_sha256": digest(Path(__file__)),
                "seconds": args.seconds, "hgs_seconds": args.hgs_seconds, "hgs_seed": 0,
                "threads": 1, "policy": "OptimalRestocking", "platform": platform.platform(),
                "completion_subset_size": args.completion_subset_size,
                "incumbent_search_seconds": args.incumbent_search_seconds,
                "additional_failure_penalty": 0., "rounding": "none; literal input matrix",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (args.output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    records = []
    for source in args.instances:
        data = convert(source.resolve(strict=True))
        instance = args.output/(source.name+".json")
        instance.write_text(json.dumps(data)+"\n")
        for method in args.methods:
            command = [str(binary), str(instance.resolve()), "--solve", "--seconds", str(args.seconds),
                       "--policy", "p3", "--hgs-seconds", str(args.hgs_seconds), "--hgs-seed", "0",
                       "--tree-threads", "1", "--ng-size", "8"]
            if args.incumbent_search_seconds:
                command.extend(["--incumbent-search-seconds", str(args.incumbent_search_seconds)])
            if args.completion_subset_size is not None:
                command.extend(["--completion-subset-size", str(args.completion_subset_size)])
            if method == "bellman":
                command.append("--restocking")
            started = time.monotonic()
            try:
                process = subprocess.run(command, env=env, text=True, capture_output=True,
                                         timeout=args.seconds+10.)
                (args.output/(source.name+"-"+method+".stderr")).write_text(process.stderr)
                answer = json.loads(process.stdout) if process.returncode == 0 else {
                    "status": "Error", "error": process.stderr[-4000:]}
            except subprocess.TimeoutExpired:
                answer = {"status": "ProcessTimeout"}
            answer.update(instance=source.name, formulation=method, command=command,
                          process_wall_seconds=time.monotonic()-started)
            objective = validate(data, answer)
            answer["independent_objective"] = objective
            answer["solution_validated"] = (objective is not None and answer.get("objective") is not None
                and math.isclose(objective, answer["objective"], rel_tol=1e-8, abs_tol=1e-6))
            (args.output/(source.name+"-"+method+".result.json")).write_text(json.dumps(answer, indent=2)+"\n")
            records.append({key: answer.get(key) for key in ("instance", "formulation", "status", "objective",
                "lower_bound", "lower_bound_source", "bound_mode", "gap", "solve_seconds", "process_wall_seconds", "nodes",
                "pricing_calls", "independent_objective", "solution_validated")})
            with (args.output/"summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
            print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
