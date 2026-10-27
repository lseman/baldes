#!/usr/bin/env python3
"""Paired-seed, interleaved ablations of robust ILS operations.

Build first: cargo build --release --features clp,multi-time-windows --example robust_pdptw
Each executable run independently replays routes and checks request coverage.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/examples/robust_pdptw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--instances", nargs="+", default=["AA40", "BB40", "AA75", "BB75"])
    parser.add_argument("--methods", nargs="+", default=["ils", "regret", "critical", "exchange", "tails", "ejection", "extended"])
    args = parser.parse_args()
    binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path("src/extensions/robust_pd/heuristic.rs"), Path("src/extensions/robust_pd/heuristic/operators.rs"), Path("src/extensions/robust_pd/mod.rs"), Path("examples/robust_pdptw.rs")]},
                "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "seconds": args.seconds, "seeds": args.seeds, "methods": args.methods,
                "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "rounding": "hundredth", "departure_cost": 10000,
                "settings": {"nominal": [0, 0, 0, 0], "robust": [5, 5, 0.3, 0.3]},
                "instances": {name: hashlib.sha256(Path(f"instances/RobustPDPTW/{name}.txt").read_bytes()).hexdigest() for name in args.instances}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (args.output / "runs.jsonl").open("w") as stream:
        for name in args.instances:
            for setting, parameters in manifest["settings"].items():
                for seed in args.seeds:
                    methods = args.methods.copy()
                    random.Random(f"{name}:{setting}:{seed}").shuffle(methods)
                    for method in methods:
                        command = [str(binary), f"instances/RobustPDPTW/{name}.txt", *map(str, parameters),
                                   str(args.seconds), "hundredth", "--heuristic", method, "--seed", str(seed)]
                        started = time.monotonic()
                        try:
                            result = subprocess.run(command, text=True, capture_output=True, timeout=max(30, args.seconds * 10))
                        except subprocess.TimeoutExpired as error:
                            stdout = error.stdout or b""
                            stderr = error.stderr or b""
                            result = subprocess.CompletedProcess(command, 124,
                                stdout.decode() if isinstance(stdout, bytes) else stdout,
                                (stderr.decode() if isinstance(stderr, bytes) else stderr) + "\nBenchmark process timeout\n")
                        elapsed = time.monotonic() - started
                        stem = f"{name}-{setting}-{seed}-{method}"
                        (args.output / f"{stem}.stdout").write_text(result.stdout)
                        (args.output / f"{stem}.stderr").write_text(result.stderr)
                        match = re.search(r"status=(\S+) bound=\S+ incumbent=(\S+).*seconds=(\S+)", result.stdout)
                        objective = float(match[2]) if match else float("inf")
                        count = sum(line.startswith("quantity=") for line in result.stdout.splitlines())
                        valid = result.returncode == 0 and math.isfinite(objective) and count > 0
                        row = {"instance": name, "setting": setting, "seed": seed, "method": method,
                               "valid": valid, "returncode": result.returncode, "objective": objective if valid else None,
                               "vehicles": count if valid else None, "travel": objective - 10000 * count if valid else None,
                               "wall_seconds": elapsed, "solver_seconds": float(match[3]) if match else None}
                        records.append(row)
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        print(f"{stem}: {objective:.2f}, vehicles={count}, valid={valid}", flush=True)
    baseline = {(r["instance"], r["setting"], r["seed"]): r for r in records if r["method"] == "ils"}
    summary = {}
    for method in args.methods:
        rows = [r for r in records if r["method"] == method]
        pairs = [(r, baseline.get((r["instance"], r["setting"], r["seed"]))) for r in rows]
        pairs = [(r, b) for r, b in pairs if r["valid"] and b and b["valid"]]
        deltas = [r["objective"] - b["objective"] for r, b in pairs]
        summary[method] = {"runs": len(rows), "valid": sum(r["valid"] for r in rows),
                           "paired": len(pairs), "wins": sum(d < -1e-5 for d in deltas),
                           "ties": sum(abs(d) <= 1e-5 for d in deltas), "losses": sum(d > 1e-5 for d in deltas),
                           "mean_objective_delta": sum(deltas) / len(deltas) if deltas else None,
                           "fewer_vehicles": sum(r["vehicles"] < b["vehicles"] for r, b in pairs),
                           "more_vehicles": sum(r["vehicles"] > b["vehicles"] for r, b in pairs)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
