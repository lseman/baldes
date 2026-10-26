"""Compare formulations and native BALDES, with fresh processes and matched data."""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time

from .data import Settings, read_instance
from .models import METHODS, SolveOptions
from .solve import add_instance_options

ROOT = Path(__file__).resolve().parents[1]


def native_result(path: Path, settings: Settings, seconds: float, threads: int, binary: Path) -> dict:
    if not settings.adjust_demands or settings.vehicle_cost != 10_000:
        raise ValueError("native example requires adjusted demands and vehicle cost 10000")
    command = [str(binary), str(path), str(settings.gamma_q), str(settings.gamma_t),
               str(settings.sigma_q), str(settings.sigma_t), str(seconds), settings.rounding]
    env = os.environ.copy()
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    else:
        env.pop("RAYON_NUM_THREADS", None)
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True,
                             timeout=seconds + 30, check=True)
    wall = time.perf_counter() - started
    match = re.search(r"status=(\w+) bound=(\S+) incumbent=(\S+) nodes=(\d+) "
                      r"pricing_calls=(\d+) seconds=(\S+)", process.stdout)
    if match is None:
        raise ValueError(f"unrecognized native output: {process.stdout}")
    finite = lambda x: float(x) if math.isfinite(float(x)) else None
    p = read_instance(path, settings)
    routes = []
    for quantity, nodes in re.findall(r"quantity=(\S+) route=(\[[^\]]*\])", process.stdout):
        if float(quantity) != 1:
            raise ValueError("unit request partition requires unit route quantities")
        routes.append(ast.literal_eval(nodes))
    result = dict(**p.metadata(), method="baldes", solver="baldes-clp", status=match[1],
                  bound=finite(match[2]), objective=finite(match[3]), nodes=int(match[4]),
                  pricing_calls=int(match[5]), algorithm_seconds=float(match[6]),
                  build_seconds=None, solver_seconds=None, separation_seconds=None,
                  environment_seconds=None, wall_seconds=wall, threads=threads, seed=None,
                  valid=None, validation_error=None, routes=routes,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    if result["objective"] is not None:
        try:
            replay = p.validate(routes)
            if abs(replay["objective"] - result["objective"]) > 1e-5:
                raise ValueError("native incumbent objective does not replay")
            result.update(valid=True, replay=replay)
        except ValueError as error:
            result.update(valid=False, validation_error=str(error))
    return result


def run_case(path: Path, settings: Settings, method: str, options: SolveOptions, binary: Path) -> dict:
    if method == "baldes":
        return native_result(path, settings, options.seconds, options.threads, binary)
    command = [sys.executable, "-m", "pd_robust.solve", str(path), "--method", method,
               "--gamma-q", str(settings.gamma_q), "--gamma-t", str(settings.gamma_t),
               "--sigma-q", str(settings.sigma_q), "--sigma-t", str(settings.sigma_t),
               "--rounding", settings.rounding, "--seconds", str(options.seconds),
               "--threads", str(options.threads), "--seed", str(options.seed),
               "--cuts-per-family", str(options.cuts_per_family)]
    if not options.root_user_cuts:
        command.append("--no-root-cuts")
    if not options.warm_start:
        command.append("--no-warm-start")
    if options.literal_big_m:
        command.append("--literal-big-m")
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                             timeout=options.seconds + 60, check=True)
    wall = time.perf_counter() - started
    result = json.loads(process.stdout)
    result["wall_seconds"] = wall
    return result


def _summary(records: list[dict]) -> str:
    lines = ["# Robust PDPTW timing comparison", "",
             "Each cell is a median across fresh processes. Wall time includes startup; algorithm",
             "time excludes startup. Native algorithm time includes graph/master construction;",
             "Gurobi solver time excludes Python model construction. Native algorithm timing is",
             "reported to milliseconds. Only validated Optimal runs count as solved.", "",
             "| Instance / Γq,Γt / σq,σt | Method | Valid Optimal / runs | Wall s | Algorithm s | Solver s | Objective range |",
             "|---|---|---:|---:|---:|---:|---:|"]
    groups: dict[tuple, list] = {}
    for r in records:
        s = r["settings"]
        key = (r["instance"], s["gamma_q"], s["gamma_t"], s["sigma_q"], s["sigma_t"], r["method"])
        groups.setdefault(key, []).append(r)
    for key, runs in sorted(groups.items()):
        name, gq, gt, sq, st, method = key
        def median(field):
            values = [r[field] for r in runs if r.get(field) is not None]
            return f"{statistics.median(values):.6f}" if values else "—"
        objectives = [r["objective"] for r in runs if r.get("valid")]
        obj = f"{min(objectives):.2f}–{max(objectives):.2f}" if objectives else "—"
        solved = sum(r.get("valid") is True and r["status"] == "Optimal" for r in runs)
        lines.append(f"| {name} / {gq},{gt} / {sq:g},{st:g} | {method} | {solved}/{len(runs)} | "
                     f"{median('wall_seconds')} | {median('algorithm_seconds')} | {median('solver_seconds')} | {obj} |")
    invalid = [r for r in records if r.get("valid") is False]
    if invalid:
        lines += ["", f"**{len(invalid)} returned incumbents failed independent replay.**",
                  "These must not be treated as valid solutions or speedups; inspect results.jsonl."]
    # Across-method disagreements remain visible; never silently aggregate unlike optima.
    optima: dict[tuple, list[dict]] = {}
    for r in records:
        if r.get("valid") and r["status"] == "Optimal":
            optima.setdefault((r["instance_sha256"], json.dumps(r["settings"], sort_keys=True)), []).append(r)
    disagreements = [rs for rs in optima.values()
                     if max(r["objective"] for r in rs) - min(r["objective"] for r in rs) > 1e-5]
    if disagreements:
        lines += ["", f"**{len(disagreements)} cases have disagreeing validated optimal objectives.**",
                  "Incumbent replay validates feasibility, not a formulation's completeness."]
    lines += ["", "`compact` reproduces the printed W load model and has a documented counterexample.",
              "`onboard-compact` is the exact alternative from Proposition 1. `branch-cut` uses the",
              "paper's RRCI/RIPEI families. `compact-guarded` also separates robust violations.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="*", type=Path)
    parser.add_argument("--cases", type=Path, help="JSON list of {instance, gamma_q, gamma_t, sigma_q, sigma_t}")
    parser.add_argument("--methods", nargs="+", choices=("baldes",) + METHODS,
                        default=["baldes", "compact", "branch-cut", "onboard-compact"])
    add_instance_options(parser)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-root-cuts", action="store_true")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--literal-big-m", action="store_true")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/examples/robust_pdptw")
    parser.add_argument("--output", type=Path, default=ROOT / "internal/robust-pd/comparison")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    if "baldes" in args.methods and not args.binary.is_file():
        parser.error("native binary missing; build cargo --release --features clp,multi-time-windows --example robust_pdptw")
    defaults = Settings(args.gamma_q, args.gamma_t, args.sigma_q, args.sigma_t, args.rounding)
    cases = [(path.resolve(), defaults) for path in args.instances]
    if args.cases:
        for case in json.loads(args.cases.read_text(encoding="utf-8")):
            case = dict(case)
            path = Path(case.pop("instance"))
            cases.append(((ROOT / path).resolve(), Settings(**{**asdict(defaults), **case})))
    if not cases:
        parser.error("provide instances or --cases")
    options = SolveOptions(seconds=args.seconds, threads=args.threads, seed=args.seed,
                           root_user_cuts=not args.no_root_cuts, warm_start=not args.no_warm_start,
                           literal_big_m=args.literal_big_m)
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                                capture_output=True, check=True).stdout)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), git_commit=revision, dirty=dirty,
                    python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(),
                    cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    initial_load_average=os.getloadavg() if hasattr(os, "getloadavg") else None,
                    source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted((ROOT / "pd_robust").glob("*.py"))},
                    options=asdict(options), methods=args.methods, repeats=args.repeats,
                    native_binary=str(args.binary.resolve()), command=sys.argv,
                    timing_scope="fresh processes; compile excluded; no native warm start; Python starts optional")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    records = []
    rng = random.Random(args.seed)
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for path, settings in cases:
            for repeat in range(args.repeats):
                methods = args.methods.copy()
                rng.shuffle(methods)
                for method in methods:
                    try:
                        result = run_case(path, settings, method, options, args.binary.resolve())
                    except (subprocess.SubprocessError, ValueError, OSError) as error:
                        result = dict(**read_instance(path, settings).metadata(), method=method,
                                      status="Error", valid=None, objective=None, error=str(error))
                        if isinstance(error, subprocess.CalledProcessError):
                            result["stderr"] = error.stderr
                    result["repeat"] = repeat
                    records.append(result)
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    stream.flush()
                    (args.output / "summary.md").write_text(_summary(records), encoding="utf-8")
                    print(f"{path.name} repeat={repeat} {method}: {result['status']} "
                          f"valid={result.get('valid')} objective={result.get('objective')} "
                          f"wall={result.get('wall_seconds')}", flush=True)
    fields = ["instance", "method", "repeat", "status", "valid", "objective", "bound", "gap",
              "wall_seconds", "algorithm_seconds", "build_seconds", "solver_seconds", "separation_seconds", "nodes"]
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ["gamma_q", "gamma_t", "sigma_q", "sigma_t"])
        writer.writeheader()
        for r in records:
            writer.writerow({**{f: r.get(f) for f in fields},
                             **{f: r["settings"][f] for f in ("gamma_q", "gamma_t", "sigma_q", "sigma_t")}})
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["final_load_average"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
