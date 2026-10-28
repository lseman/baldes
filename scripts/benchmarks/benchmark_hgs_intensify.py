#!/usr/bin/env python3
"""Compare HGS decomposition workers with fixed batches and fixed-work HGS solves.

Example:
  python3 scripts/benchmarks/benchmark_hgs_intensify.py \
    target/release/baldes instances/XL/XL-n2121-k186.vrp seed.sol \
    --workers 1 2 4 --batch-size 4 --rounds 24 --output results/hgs-workers.json

No network access is performed. The seed must already have been downloaded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("instance", type=Path)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--workers", nargs="+", type=positive, default=[1, 2, 4])
    parser.add_argument("--batch-size", type=positive, default=4)
    parser.add_argument("--rounds", type=positive, default=24)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--nb-iter", type=positive, default=300)
    parser.add_argument("--population-size", type=positive, default=12)
    parser.add_argument("--time", type=positive, default=600,
                        help="Outer safety limit; hitting it invalidates the equal-work comparison")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounding", choices=["none", "nearest", "tenth", "ceil"],
                        help="Explicit instance metric, e.g. tenth for CVRPLib VRPTW BKSs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in [args.binary, args.instance, args.solution]:
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    rows = []
    reference = None
    # Alternate order to limit systematic warm-cache and thermal bias.
    for repeat in range(args.repeats):
        workers_order = args.workers if repeat % 2 == 0 else list(reversed(args.workers))
        for workers in workers_order:
            command = [str(args.binary.resolve()), "hgs", str(args.instance.resolve()),
                       "--initial-solution", str(args.solution.resolve()), "--intensify",
                       "--workers", str(workers), "--batch-size", str(args.batch_size),
                       "--intensify-rounds", str(args.rounds), "--subproblem-time", "0",
                       "--nb-iter", str(args.nb_iter), "--population-size", str(args.population_size),
                       "--time", str(args.time), "--seed", str(args.seed), "--json", "-"]
            if args.rounding:
                command.extend(["--rounding", args.rounding])
            started = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True, timeout=args.time + 120)
            wall = time.perf_counter() - started
            if process.returncode:
                raise RuntimeError(f"worker run failed: {process.stderr}")
            result = json.loads(process.stdout)
            if result["seconds"] >= args.time:
                raise RuntimeError("Outer time limit was reached; increase --time for an equal-work comparison.")
            completed = re.search(r"\brounds=(\d+)\b", process.stderr)
            if completed is None or int(completed.group(1)) != args.rounds:
                raise RuntimeError("The full round budget did not finish; increase --time before comparing speed.")
            signature = (result["cost"], result["generations"], result["routes"])
            if reference is None:
                reference = signature
            if signature != reference:
                raise RuntimeError("Work/results differ across runs; do not claim an equal-work speedup. "
                                   "Check outer time limits and fixed-batch determinism.")
            rows.append({"workers": workers, "repeat": repeat, "wall_seconds": wall,
                         "search_seconds": result["seconds"], "cost": result["cost"],
                         "generations": result["generations"], "summary": process.stderr.strip()})
            print(f"workers={workers} repeat={repeat} seconds={wall:.4f} cost={result['cost']}", flush=True)
    medians = {str(w): statistics.median(r["wall_seconds"] for r in rows if r["workers"] == w)
               for w in args.workers}
    baseline = medians.get("1")
    summary = {"instance": str(args.instance.resolve()), "seed": args.seed,
               "solution_sha256": hashlib.sha256(args.solution.read_bytes()).hexdigest(),
               "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               "batch_size": args.batch_size, "rounds": args.rounds, "nb_iter": args.nb_iter,
               "rounding": args.rounding,
               "population_size": args.population_size, "equal_work_and_routes": True,
               "median_seconds": medians,
               "speedup": {w: baseline / seconds for w, seconds in medians.items()} if baseline else {},
               "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"median_seconds": medians, "speedup": summary["speedup"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
