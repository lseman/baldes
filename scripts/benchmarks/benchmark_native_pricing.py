#!/usr/bin/env python3
"""Compare native-pricing against a saved Clp build (Linux, repository root)."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
from datetime import datetime, timezone


TIMINGS = ("seconds", "pricing_seconds", "master_seconds")
CASES = {"C202": [], "R202": [], "R201-rank1": ["--rank1"]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/native-pricing"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--cpus", default="0")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    for path in binaries.values():
        if not path.is_file():
            parser.error(f"missing binary: {path}")
    inputs = {name: Path(f"instances/Solomon/{name.split('-')[0]}.txt") for name in args.cases}
    for path in inputs.values():
        if not path.is_file():
            parser.error(f"missing instance: {path}")
    # Refuse to overwrite prior or partial results.
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    metadata = {
        "date": datetime.now(timezone.utc).isoformat(),
        "binaries": {name: {"path": str(path), "sha256": digest(path)} for name, path in binaries.items()},
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in inputs.items()},
        "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), "unknown"),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "cxx": subprocess.check_output([*shlex.split(os.environ.get("CXX", "c++")), "--version"], text=True),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "candidate_worktree_source_sha256": {str(p): digest(p) for p in [Path("Cargo.toml"), Path("Cargo.lock"), Path("build.rs"),
                          *sorted(Path("src/labeling").rglob("*.rs")),
                          Path("src/labeling/engine/extension_batch.cpp"),
                          Path("src/master/solvers/clp/native/baldes_clp_matrix_kernels.hpp")]},
        "affinity": args.cpus, "runs": args.runs, "warmups": 1,
        "native_threads": 1, "rounding": "none",
        "expected_builds": {"baseline": "release --features clp", "candidate": "release --features native-pricing"},
        "rustflags": env.get("RUSTFLAGS", ""), "cxxflags": env.get("CXXFLAGS", ""),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    summary = []
    for name, source in inputs.items():
        expected = None
        options = ["vrptw", str(source), "--customers", "100", "--master", "clp", "--clp-scaling", "auto",
                   "--reference-ng", "--ng", "8", "--rounding", "none", "--no-enumeration",
                   "--bound-mode", "numerical", "--time", "60", *CASES[name]]
        for repetition in range(args.runs + 1):
            order = list(binaries) if repetition % 2 == 0 else list(reversed(binaries))
            for variant in order:
                output = args.output / f"{name}-{variant}-{repetition}.json"
                command = ["taskset", "-c", args.cpus, str(binaries[variant]), *options, "--json", str(output)]
                rss = output.with_suffix(".rss")
                if Path("/usr/bin/time").is_file():
                    command = ["/usr/bin/time", "-f", "%M", "-o", str(rss), *command]
                process = subprocess.run(
                    command,
                    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
                )
                output.with_suffix(".log").write_text(process.stdout)
                process.check_returncode()
                data = json.loads(output.read_text())
                if data["status"] != "Optimal" or not all(math.isfinite(data[k]) for k in ("lp", "bound", *TIMINGS)):
                    raise RuntimeError(f"incomplete or invalid result: {output}")
                semantic = {key: value for key, value in data.items() if key not in TIMINGS}
                if expected is None:
                    expected = semantic
                if semantic != expected:
                    raise RuntimeError(f"non-timing solver output changed: {output}")
                records.append({"case": name, "variant": variant, "repetition": repetition,
                                "peak_rss_kib": int(rss.read_text().strip()) if rss.is_file() else None,
                                **{key: data[key] for key in TIMINGS}})
                (args.output / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
                print(name, variant, repetition, *(f"{key}={data[key]:.4f}" for key in TIMINGS), flush=True)
        row = {"case": name, "variants": {}}
        for variant in binaries:
            group = [r for r in records if r["case"] == name and r["variant"] == variant and r["repetition"] > 0]
            row["variants"][variant] = {
                key: {"median": statistics.median(r[key] for r in group),
                      "min": min(r[key] for r in group), "max": max(r[key] for r in group)} for key in TIMINGS
            }
            row["variants"][variant]["peak_rss_kib"] = [r["peak_rss_kib"] for r in group]
        summary.append(row)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
