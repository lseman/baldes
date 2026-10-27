#!/usr/bin/env python3
"""Generate NEW symmetric CVRPSD instances from the Salavati-Khoshghalb rules.

The original instance files and random seeds are unavailable. These instances
are synthetic replications of the experimental DESIGN, not the published data.
Standard library only; coordinates/classes are shared across fill rates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
from fractions import Fraction
from pathlib import Path

SCHEMA = "baldes.stochastic.generated.v1"
SOURCE = "https://www.cirrelt.ca/documentstravail/cirrelt-2017-36.pdf"
PROBABILITIES = [0.1, 0.2, 0.4, 0.2, 0.1]
FILL_RATES = ("0.90", "0.92", "0.94", "0.96")
GRID = tuple((n, m) for n in (20, 30, 40, 50, 60)
             for m in ((2,) if n <= 30 else (2, 3, 4)))


def base_instance(nodes: int, vehicles: int, repetition: int, seed: int) -> dict:
    """Independent stable base seed; adding/filtering grid entries changes no others."""
    if nodes < 2 or vehicles < 1 or vehicles >= nodes or repetition < 1:
        raise ValueError("need nodes >= 2, 1 <= vehicles < nodes, repetition >= 1")
    key = f"salavati-style-v1:{seed}:{nodes}:{vehicles}:{repetition}"
    base_seed = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")
    rng = random.Random(base_seed)
    coordinates = [[100.0 * rng.random(), 100.0 * rng.random()] for _ in range(nodes)]
    demands = []
    total_mean = 0
    for customer in range(1, nodes):
        demand_class = rng.randrange(3)
        support = list(range(1 + 5 * demand_class, 6 + 5 * demand_class))
        total_mean += support[2]  # Symmetric PMF has exactly this integer mean.
        demands.append({"customer": customer, "support": support,
                        "probabilities": PROBABILITIES.copy()})
    # Nearest integer, with half-up ties explicitly selected instead of round()
    # (which uses bankers' rounding). No metric closure alters the paper's costs.
    costs = [[math.floor(math.dist(a, b) + 0.5) for b in coordinates] for a in coordinates]
    return {"base_id": f"synthetic-n{nodes}-m{vehicles}-r{repetition:02d}",
            "base_seed": str(base_seed), "nodes_including_depot": nodes,
            "vehicles": vehicles, "depot": 0, "coordinates": coordinates,
            "demands": demands, "costs": costs, "total_expected_demand": total_mean}


def with_fill_rate(base: dict, fill_rate: str) -> dict:
    """Capacity is ceiling(total mean / (fleet * target fill)), recorded explicitly."""
    fill = Fraction(fill_rate)
    if not 0 < fill <= 1:
        raise ValueError("fill rate must be in (0, 1]")
    ideal = Fraction(base["total_expected_demand"], base["vehicles"]) / fill
    capacity = math.ceil(ideal)
    maximum = max(d["support"][-1] for d in base["demands"])
    if capacity < maximum:
        raise ValueError("capacity below maximum demand support; use larger base instances")
    nodes = base["nodes_including_depot"]
    b = sum(base["costs"][j][0] for j in range(1, nodes)) / (nodes - 1)
    return {"schema": SCHEMA, "instance_id": f"{base['base_id']}-fr{fill_rate}",
            "provenance": {
                "kind": "synthetic_paper_style", "original_benchmark_instance": False,
                "source": SOURCE, "source_section": "Section 5, Table 1",
                "generator": "scripts/generate_stochastic_instances.py",
                "capacity_rounding": "ceil(total_expected_demand / (vehicles * target_fill_rate))",
                "distance_rounding": "nearest integer; half-up ties; no metric closure",
                "failure_penalty_applied": 0.0,
                "reference_2017_failure_penalty_not_applied": b,
            },
            **base, "target_fill_rate": fill_rate, "ideal_capacity": float(ideal),
            "capacity": capacity,
            "actual_fill_rate": base["total_expected_demand"] / (base["vehicles"] * capacity)}


def encoded(value: dict) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def generate(output: Path, seed: int, repetitions: int, sizes: list[int]) -> dict:
    if repetitions < 1 or not sizes or len(set(sizes)) != len(sizes) or any(n not in (20, 30, 40, 50, 60) for n in sizes):
        raise ValueError("need positive repetitions and distinct sizes from 20,30,40,50,60")
    artifacts: dict[str, bytes] = {}
    records = []
    for n, m in GRID:
        if n not in sizes:
            continue
        for repetition in range(1, repetitions + 1):
            base = base_instance(n, m, repetition, seed)
            for fill in FILL_RATES:
                instance = with_fill_rate(base, fill)
                filename = instance["instance_id"] + ".json"
                data = encoded(instance)
                artifacts[filename] = data
                records.append({"file": filename, "nodes": n, "customers": n - 1,
                                "vehicles": m, "repetition": repetition,
                                "target_fill_rate": fill, "capacity": instance["capacity"],
                                "actual_fill_rate": instance["actual_fill_rate"],
                                "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"schema": SCHEMA, "kind": "synthetic_paper_style",
                "original_benchmark_instances": False, "master_seed": seed,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "repetitions_per_base_configuration": repetitions,
                "instance_count": len(records), "instances": records}
    artifacts["manifest.json"] = encoded(manifest)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    artifacts["index.csv"] = stream.getvalue().encode()
    # Detect all conflicts before writing anything. An identical rerun is safe;
    # different seed/content requires another destination and never overwrites.
    for filename, content in artifacts.items():
        path = output / filename
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"different content already exists at {path}; choose another output")
    unexpected = {p.name for p in output.glob("synthetic-*.json")} - artifacts.keys()
    if unexpected:
        raise FileExistsError("output contains instances outside the requested grid; choose another output")
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in artifacts.items():
        (output / filename).write_bytes(content)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("instances/stochastic/generated_salavati_style"))
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 30, 40, 50, 60])
    args = parser.parse_args()
    try:
        manifest = generate(args.output, args.seed, args.repetitions, args.sizes)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Generated {manifest['instance_count']} NEW synthetic instances at {args.output}")
    print("These are NOT the original published benchmark instances.")


if __name__ == "__main__":
    main()
