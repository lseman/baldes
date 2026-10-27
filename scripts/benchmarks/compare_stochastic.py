#!/usr/bin/env python3
"""Run BC-Theta/Gurobi and BALDES on identical JSON, budgets and thread settings.

Uses a prebuilt Rust binary: build time is never counted as solve time. Both
process wall time and internal build/solve times are retained. A timeout is not
an optimum and a restricted or incomplete result is never a speedup claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], env: dict, timeout: float) -> dict:
    started = perf_counter()
    try:
        process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "ProcessTimeout", "process_wall_seconds": perf_counter() - started,
                "objective": None, "command": command}
    elapsed = perf_counter() - started
    if process.returncode:
        return {"status": "Error", "process_wall_seconds": elapsed, "objective": None,
                "error": process.stderr[-4000:], "command": command}
    try:
        answer = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {"status": "InvalidOutput", "process_wall_seconds": elapsed,
                "objective": None, "stdout": process.stdout[-4000:], "command": command}
    answer["process_wall_seconds"] = elapsed
    answer["command"] = command
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--baldes-bin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", choices=["p1", "p3"], default="p3")
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--ng-size", type=int, default=8,
                        help="NG prepricing neighborhood; 0 disables (default: 8)")
    parser.add_argument("--max-labels", type=int, default=1_000_000,
                        help="stored-label limit per BALDES pricing call (default: 1000000)")
    parser.add_argument("--build-description", default="user-supplied binary; build flags not independently verified")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or min(args.threads, args.repetitions, args.max_labels) < 1:
        parser.error("need positive finite seconds and positive threads/repetitions/max-labels")
    if args.ng_size < 0:
        parser.error("NG size must be nonnegative")
    binary = args.baldes_bin.resolve(strict=True)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty")
    inputs = [p.resolve(strict=True) for p in args.instances]
    hashes = {str(path): digest(path) for path in inputs}
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("stochastic_branch_cut.py").resolve()
    environment = os.environ.copy()
    for name in ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        environment[name] = str(args.threads)
    metadata = {"policy": args.policy, "time_limit_seconds": args.seconds, "threads": args.threads,
                "repetitions": args.repetitions, "max_labels_per_pricing_call": args.max_labels,
                "ng_size": args.ng_size,
                "baldes_binary_sha256": digest(binary), "branch_cut_script_sha256": digest(script),
                "python": sys.version, "build_description": args.build_description, "inputs": hashes,
                "comparison_scope": "new synthetic cases; directed Python BC-Theta with non-CVRPSEP fractional separation",
                "budget_scope": "Gurobi TimeLimit covers optimize only; BALDES limit covers node construction and solves. Report process wall/build/solve times separately.",
                "warm_start_scope": "no external starts; BALDES internally seeds singleton columns and bounded greedy starts"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    all_results = []
    for index, path in enumerate(inputs):
        for repetition in range(args.repetitions):
            if digest(path) != hashes[str(path)]:
                raise RuntimeError("input changed during comparison")
            commands = {
                "gurobi": [sys.executable, str(script), str(path), "--policy", args.policy,
                           "--seconds", str(args.seconds), "--threads", str(args.threads), "--seed", str(repetition)],
                "baldes": [str(binary), str(path), "--solve", "--json", "--policy", args.policy,
                           "--seconds", str(args.seconds), "--max-labels", str(args.max_labels),
                           "--ng-size", str(args.ng_size)]}
            record = {"input": str(path), "input_sha256": hashes[str(path)], "repetition": repetition,
                      "policy": args.policy, "results": {}}
            order = ["gurobi", "baldes"] if (index + repetition) % 2 == 0 else ["baldes", "gurobi"]
            for method in order:
                print(f"{path.name} repetition {repetition + 1}: {method}", flush=True)
                record["results"][method] = run(commands[method], environment, args.seconds + max(args.seconds, 30.))
            a, b = record["results"]["gurobi"], record["results"]["baldes"]
            both_optimal = a["status"] == b["status"] == "Optimal"
            record["both_optimal"] = both_optimal
            record["objective_agreement"] = None
            if both_optimal:
                record["objective_agreement"] = abs(a["objective"] - b["objective"]) <= 1e-5 * (1. + abs(a["objective"]))
            all_results.append(record)
            (args.output / "results.json").write_text(json.dumps(all_results, indent=2, allow_nan=False) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        fields = ["input", "repetition", "method", "status", "objective", "lower_bound", "gap",
                  "process_wall_seconds", "build_seconds", "solve_seconds", "nodes", "both_optimal", "objective_agreement"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for record in all_results:
            for method, result in record["results"].items():
                row = {key: result.get(key) for key in fields}
                row.update({"input": record["input"], "repetition": record["repetition"], "method": method,
                            "both_optimal": record["both_optimal"], "objective_agreement": record["objective_agreement"]})
                writer.writerow(row)
    print(f"Saved comparison to {args.output}")
    if any(r["objective_agreement"] is False for r in all_results):
        raise SystemExit("Objective mismatch: inspect results; do not compare these timings as equivalent solves")


if __name__ == "__main__":
    main()
