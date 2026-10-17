"""Root-node benchmark of BALDES against the BaPCod binary.

Both solvers get the same instance graph, the same NG memories (size 8,
nearest customers), Clp, no reduced-cost fixing and no enumeration. Each run
is a separate process, run one at a time. Times are the solvers' own root
timers (BaPCod bcTimeRootEval, BALDES `seconds`), so imports and model
building are excluded.

    python tools/benchmark_bapcod.py --out results/bench \
        --cases P-n16-k8 A-n32-k5 C101:50 --configs none rank1 capacity

BAPCOD_RE points at the bapcod-re checkout (default ../bapcod-re); the
reference driver lives in its openrscp/python/tools. Build BALDES first with
`cargo build --release --features clp`.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RE = Path(os.environ.get("BAPCOD_RE", ROOT.parent / "bapcod-re"))
OPENRSCP = RE / "openrscp"
INSTANCES = OPENRSCP / "examples/instances"
BINARY = ROOT / "target/release/baldes"
CONFIGS = {
    "none": {"rank1": False, "capacity": False},
    "rank1": {"rank1": True, "capacity": False},
    "capacity": {"rank1": False, "capacity": True},
    "rank1+capacity": {"rank1": True, "capacity": True},
}


def parse_case(case):
    name, _, n = case.partition(":")
    return name, int(n) if n else None


def reference_worker(name, n, config, limit):
    sys.path[:0] = [str(OPENRSCP / "python/src"), str(OPENRSCP / "python/tools")]
    from openrscp import vrptw
    import bapcod_reference
    import differential

    inst = vrptw.read_solomon(INSTANCES / f"{name}.txt")
    if n is not None:
        inst = inst.truncated(n)
    flags = CONFIGS[config]
    r = bapcod_reference.solve_root_reference(
        inst,
        ng_sets=differential.ng_sets(inst, 8),
        rank1_cuts=flags["rank1"],
        capacity_cuts=flags["capacity"],
        params={"GlobalTimeLimit": int(limit), **json.loads(os.environ.get("BAPCOD_PARAMS", "{}"))},
    )
    print(json.dumps({"lp": r.root_lp, "seconds": r.time_root, "pricing_calls": r.pricing_calls,
                      "cuts": r.cuts, "columns": r.columns,
                      "pricing_seconds": r.pricing_seconds, "master_seconds": r.master_seconds,
                      "separation_seconds": r.separation_seconds}))


def run_reference(name, n, config, limit):
    cmd = [sys.executable, __file__, "--reference-worker", f"{name}:{n or ''}", config, str(limit)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit + 120)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        return {"error": (p.stderr.strip().splitlines() or ["failed"])[-1]}
    return json.loads(lines[-1])


def run_baldes(name, n, config, limit, out):
    flags = CONFIGS[config]
    path = out / f"{name}-{n or 'all'}-{config}.baldes.json"
    cmd = [str(BINARY), "vrptw", str(INSTANCES / f"{name}.txt"), "--reference-ng", "--no-enumeration",
           "--time", str(limit), "--json", str(path)]
    if n is not None:
        cmd += ["--customers", str(n)]
    if flags["rank1"]:
        cmd.append("--rank1")
    if flags["capacity"]:
        cmd.append("--capacity")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=limit + 120)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if p.returncode != 0 or not path.exists():
        return {"error": (p.stderr.strip().splitlines() or ["failed"])[-1]}
    s = json.loads(path.read_text())
    return {k: s[k] for k in ("lp", "seconds", "pricing_calls", "cuts", "columns", "status",
                              "pricing_seconds", "master_seconds")}


def fmt(x, digits=2):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}f}"


def table(rows):
    lines = [
        "| Instance | Cuts | Root LP (BaPCod) | Root LP (BALDES) | Time s (BaPCod) | Time s (BALDES) "
        "| Pricing calls (BaPCod / BALDES) | Cuts added (BaPCod / BALDES) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        b, o = r["bapcod"], r["baldes"]
        def g(d, k):
            return d.get(k) if "error" not in d else None
        lines.append(
            f"| {r['instance']} | {r['config']} | {fmt(g(b, 'lp'), 3)} | {fmt(g(o, 'lp'), 3)} "
            f"| {fmt(g(b, 'seconds'))} | {fmt(g(o, 'seconds'))} "
            f"| {g(b, 'pricing_calls') if g(b, 'pricing_calls') is not None else '—'} / "
            f"{g(o, 'pricing_calls') if g(o, 'pricing_calls') is not None else '—'} "
            f"| {g(b, 'cuts') if g(b, 'cuts') is not None else '—'} / "
            f"{g(o, 'cuts') if g(o, 'cuts') is not None else '—'} |"
        )
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--reference-worker":
        name, n = parse_case(sys.argv[2])
        reference_worker(name, n, sys.argv[3], float(sys.argv[4]))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", required=True, help="NAME or NAME:CUSTOMERS")
    ap.add_argument("--configs", nargs="+", default=["none", "rank1", "capacity"], choices=CONFIGS)
    ap.add_argument("--time", type=float, default=600, help="time limit per solve (s)")
    ap.add_argument("--out", default="results/bench")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in args.cases:
        name, n = parse_case(case)
        label = name if n is None else f"{name} ({n})"
        for config in args.configs:
            row = {"instance": label, "config": config,
                   "bapcod": run_reference(name, n, config, args.time),
                   "baldes": run_baldes(name, n, config, args.time, out)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    md = table(rows)
    (out / "table.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
