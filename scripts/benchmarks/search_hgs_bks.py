#!/usr/bin/env python3
"""Search several CVRPLib instances from their current published solutions.

Build with ``cargo build --release --features clp --bin baldes`` first. Each
instance/seed is searched sequentially; workers parallelize its subproblems.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
INITIAL = re.compile(r"\bIntensify:\s+initial=([0-9.eE+-]+)\s+best=")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", type=Path, nargs="+", help="Local .vrp or Solomon instance files")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/baldes")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are not overwritten")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 43, 47])
    parser.add_argument("--time", type=positive, default=300, help="Search seconds per instance and seed; BKS download is additional")
    parser.add_argument("--workers", type=positive, default=4)
    parser.add_argument("--batch-size", type=int, default=0, help="Jobs per snapshot; 0 uses the worker count")
    parser.add_argument("--rounds", type=positive, default=100000)
    parser.add_argument("--subproblem-routes", type=positive, default=12)
    parser.add_argument("--subproblem-customers", type=positive, default=300)
    parser.add_argument("--subproblem-time", type=nonnegative_float, default=1)
    parser.add_argument("--nb-iter", type=positive, default=300)
    parser.add_argument("--population-size", type=positive, default=12)
    parser.add_argument("--threshold", type=nonnegative_float, default=0.003)
    parser.add_argument("--elite-size", type=positive, default=8)
    parser.add_argument("--restart-after", type=positive, default=100)
    parser.add_argument("--std-similarity", action="store_true", help="Use spatial-temporal-demand route similarity")
    args = parser.parse_args(argv)
    if args.batch_size < 0:
        parser.error("--batch-size must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    names = [path.stem.casefold() for path in args.instances]
    if len(set(names)) != len(names):
        parser.error("instance file names must be distinct")
    for path in [args.binary, *args.instances]:
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_for(args, instance, seed, prefix):
    command = [str(args.binary.resolve()), "hgs", str(instance.resolve()), "--get-bks", "--intensify",
               "--time", str(args.time), "--workers", str(args.workers), "--batch-size", str(args.batch_size),
               "--intensify-rounds", str(args.rounds), "--subproblem-routes", str(args.subproblem_routes),
               "--subproblem-customers", str(args.subproblem_customers), "--subproblem-time", str(args.subproblem_time),
               "--nb-iter", str(args.nb_iter), "--population-size", str(args.population_size),
               "--threshold", str(args.threshold), "--elite-size", str(args.elite_size),
               "--restart-after", str(args.restart_after), "--seed", str(seed),
               "--solution-output", str(prefix) + ".sol", "--json", str(prefix) + ".json"]
    if args.std_similarity:
        command.append("--std-similarity")
    return command


def run_one(args, instance, seed, output):
    prefix = output / f"{instance.stem}.seed-{seed}"
    command = command_for(args, instance, seed, prefix)
    log_path = Path(str(prefix) + ".log")
    result_path = Path(str(prefix) + ".json")
    solution_path = Path(str(prefix) + ".sol")
    started = time.monotonic()
    row = {"instance": str(instance.resolve()), "instance_sha256": sha256(instance), "seed": seed,
           "command": command, "log": str(log_path), "solution": str(solution_path), "result": str(result_path)}
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
        row["exit_code"] = process.returncode
        if process.returncode:
            raise ValueError(f"solver exited with status {process.returncode}; see {log_path}")
        match = INITIAL.search(log_path.read_text(encoding="utf-8"))
        if not match:
            raise ValueError(f"missing initial cost in {log_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        initial, best = float(match.group(1)), float(result["cost"])
        if not (math.isfinite(initial) and math.isfinite(best)) or not solution_path.is_file():
            raise ValueError("invalid costs or missing solution checkpoint")
        row.update(status="ok", initial_cost=initial, best_cost=best,
                   improvement=initial - best, candidate_improvement=best < initial - 1e-6,
                   search_seconds=result.get("seconds"), generations=result.get("generations"),
                   solution_sha256=sha256(solution_path))
    except (OSError, ValueError, KeyError, TypeError) as error:
        row.update(status="error", error=str(error))
    row["wall_seconds"] = round(time.monotonic() - started, 3)
    return row


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"ERROR: cannot create output directory: {error}", file=sys.stderr)
        return 1
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "binary": str(args.binary.resolve()), "binary_sha256": sha256(args.binary), "runs": []}
    for instance in args.instances:
        for seed in args.seeds:
            print(f"Searching {instance.name}, seed {seed}...", flush=True)
            row = run_one(args, instance, seed, output)
            summary["runs"].append(row)
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            if row["status"] == "ok":
                label = "candidate improvement" if row["candidate_improvement"] else "no improvement"
                print(f"  initial={row['initial_cost']:g} best={row['best_cost']:g}: {label}", flush=True)
            else:
                print(f"  ERROR: {row['error']}", file=sys.stderr, flush=True)
    return 1 if any(row["status"] != "ok" for row in summary["runs"]) else 0


if __name__ == "__main__":
    sys.exit(main())
