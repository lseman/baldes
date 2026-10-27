#!/usr/bin/env python3
"""Directed Python reimplementation of Approach 4 (BC-Theta), not author code.

Tomazella, Munari & Morabito (2026), equations (2)-(6), (27)-(29), (31).
Integer capacity separation is complete. Fractional separation is exhaustive
up to 12 customers, heuristic above that; this does NOT replicate CVRPSEP.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
from time import perf_counter

SOURCE = "https://optimization-online.org/wp-content/uploads/2026/04/Tomazella_Munari_Morabito_2026_Formulations-1.pdf"


def validate(data: dict) -> None:
    n = len(data["demands"])
    q = data["capacity"]
    if (data.get("depot", 0) != 0 or not isinstance(q, int) or q <= 0
            or not isinstance(data["vehicles"], int) or data["vehicles"] < 0
            or data.get("nodes_including_depot", n + 1) != n + 1
            or len(data["costs"]) != n + 1
            or data.get("provenance", {}).get("failure_penalty_applied", 0) != 0):
        raise ValueError("invalid dimensions, capacity, fleet, depot or unsupported failure fee")
    for i, row in enumerate(data["costs"]):
        if len(row) != n + 1 or any(not math.isfinite(c) or c < 0 for c in row) or row[i] != 0:
            raise ValueError("invalid cost matrix")
    for j, demand in enumerate(data["demands"], 1):
        support, probs = demand["support"], demand["probabilities"]
        if (demand["customer"] != j or not support or len(support) != len(probs)
                or any(not isinstance(d, int) or d < 0 or d > q for d in support)
                or any(a >= b for a, b in zip(support, support[1:]))
                or any(not math.isfinite(p) or p < 0 for p in probs)
                or abs(sum(probs) - 1) > 1e-12):
            raise ValueError("invalid customer distribution")


def mean_demands(data: dict) -> list[Fraction]:
    means = []
    for demand in data["demands"]:
        probabilities = [Fraction(str(p)) for p in demand["probabilities"]]
        means.append(sum(d * p for d, p in zip(demand["support"], probabilities)) / sum(probabilities))
    return means


def recourse(data: dict, route: list[int], policy: str) -> float:
    """Independent fixed-order backward Bellman evaluator, equations (9)/(11)."""
    if policy not in ("p1", "p3"):
        raise ValueError("policy must be p1 or p3")
    qmax, costs = data["capacity"], data["costs"]
    suffix = [0.] * (qmax + 1)
    for t in reversed(range(len(route))):
        j, i = route[t], (route[t - 1] if t else 0)
        demand = data["demands"][j - 1]
        support = demand["support"]
        probs = [p / sum(demand["probabilities"]) for p in demand["probabilities"]]
        refill = costs[i][0] + costs[0][j] - costs[i][j] + sum(
            p * suffix[qmax - d] for d, p in zip(support, probs))
        current = []
        for q in range(qmax + 1):
            direct = sum(p * (costs[j][0] + costs[0][j] + suffix[qmax - (d - q)]
                              if d > q else suffix[q - d]) for d, p in zip(support, probs))
            current.append(min(direct, refill) if policy == "p3" else (refill if q == 0 else direct))
        suffix = current
    return suffix[qmax]


def route_cost(data: dict, route: list[int], policy: str) -> float:
    return sum(data["costs"][i][j] for i, j in zip([0] + route, route + [0])) + recourse(data, route, policy)


def split_integer(arcs: list[tuple[int, int]], customers: int) -> tuple[list[list[int]], list[set[int]]]:
    """Decompose a binary degree-feasible candidate into depot routes and cycles."""
    sink = customers + 1
    successor = {i: j for i, j in arcs if i != 0}
    unseen = set(range(1, sink))
    routes, cycles = [], []
    for start in sorted(j for i, j in arcs if i == 0):
        route, at = [], start
        while at != sink:
            if at not in unseen:
                raise ValueError("invalid integer route decomposition")
            unseen.remove(at)
            route.append(at)
            at = successor[at]
        routes.append(route)
    while unseen:
        start = min(unseen)
        cycle, at = set(), start
        while at not in cycle:
            if at not in unseen:
                raise ValueError("invalid integer cycle decomposition")
            cycle.add(at)
            unseen.remove(at)
            at = successor[at]
        cycles.append(cycle)
    return routes, cycles


def fractional_sets(values: dict[tuple[int, int], float], n: int) -> set[frozenset[int]]:
    customers = set(range(1, n + 1))
    if n <= 12:
        return {frozenset(s) for k in range(2, n + 1) for s in itertools.combinations(customers, k)}
    # Cheap connected components at several thresholds; no CVRPSEP parity claim.
    candidates = {frozenset(customers)}
    for threshold in (1e-6, 0.25, 0.5, 0.75):
        unseen = customers.copy()
        while unseen:
            component = {unseen.pop()}
            queue = list(component)
            while queue:
                i = queue.pop()
                neighbors = {j for j in unseen if values.get((i, j), 0.) + values.get((j, i), 0.) > threshold}
                unseen -= neighbors
                component |= neighbors
                queue.extend(neighbors)
            if len(component) >= 2:
                candidates.add(frozenset(component))
    for i, j in itertools.combinations(customers, 2):
        candidates.add(frozenset((i, j)))
    return candidates


def solve(data: dict, policy: str = "p3", seconds: float = 30., threads: int = 1,
          seed: int = 0, fractional: bool = True, log: bool = False) -> dict:
    import gurobipy as gp
    from gurobipy import GRB

    started = perf_counter()
    validate(data)
    if policy not in ("p1", "p3") or not math.isfinite(seconds) or seconds < 0 or threads < 1:
        raise ValueError("invalid policy, time budget or thread count")
    n, qmax = len(data["demands"]), data["capacity"]
    customers, sink = range(1, n + 1), n + 1
    means = mean_demands(data)
    costs = data["costs"]
    arcs = [(i, j) for i in range(n + 1) for j in range(1, sink + 1) if i != j and (i != 0 or j != sink)]
    # Rounded Euclidean costs can violate triangle inequalities. Equation (31)
    # needs nonnegative route charges. A per-route shift B offsets the worst
    # possible negative replenishment detours, then -B is charged on departures.
    # This is zero for metric data and preserves the original objective otherwise.
    deficit = min([0.] + [costs[i][0] + costs[0][j] - costs[i][j]
                              for i in customers for j in customers if i != j])
    shift = -n * deficit * (1. + 1e-12)
    counts = {"lazy_capacity": 0, "fractional_capacity": 0, "theta": 0, "integer_candidates": 0}
    cached = {}
    errors = []
    with gp.Env(params={"OutputFlag": int(log)}) as env, gp.Model("cvrpsd_bc_theta_directed", env=env) as model:
        model.Params.Threads = threads
        model.Params.Seed = seed
        model.Params.LazyConstraints = 1
        model.Params.PreCrush = 1
        model.Params.MIPGap = 0.
        model.Params.MIPGapAbs = 1e-7
        model.Params.FeasibilityTol = 1e-8
        model.Params.IntFeasTol = 1e-8
        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        theta = model.addVars(customers, lb=0., name="theta")
        for i in customers:
            model.addConstr(x.sum(i, "*") == 1, name=f"out[{i}]")
            model.addConstr(x.sum("*", i) == 1, name=f"in[{i}]")
        model.addConstr(x.sum(0, "*") <= data["vehicles"], name="fleet")
        model.setObjective(gp.quicksum(costs[i][0 if j == sink else j] * x[i, j] for i, j in arcs)
                           + theta.sum() - shift * x.sum(0, "*"))

        def capacity_cut(subset, values, lazy):
            required = max(1, math.ceil(sum(means[i - 1] for i in subset) / qmax))
            rhs = len(subset) - required
            internal = [(i, j) for i in subset for j in subset if i != j]
            if sum(values[a] for a in internal) > rhs + 1e-7:
                expression = gp.quicksum(x[a] for a in internal) <= rhs
                (model.cbLazy if lazy else model.cbCut)(expression)
                counts["lazy_capacity" if lazy else "fractional_capacity"] += 1
                return True
            return False

        def callback(current, where):
            try:
                if where == GRB.Callback.MIPNODE and fractional:
                    if current.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                        return
                    values = current.cbGetNodeRel(x)
                    for subset in sorted(fractional_sets(values, n), key=lambda s: (len(s), sorted(s))):
                        capacity_cut(subset, values, False)
                if where != GRB.Callback.MIPSOL:
                    return
                counts["integer_candidates"] += 1
                values = current.cbGetSolution(x)
                selected = [a for a in arcs if values[a] > 0.5]
                routes, cycles = split_integer(selected, n)
                violated = False
                for subset in cycles + [set(r) for r in routes]:
                    violated |= capacity_cut(subset, values, True)
                if violated:
                    return
                theta_values = current.cbGetSolution(theta)
                for route in routes:
                    key = tuple(route)
                    if key not in cached:
                        cached[key] = recourse(data, route, policy)
                    charge = cached[key] + shift
                    if charge < -1e-7:
                        raise ValueError("invalid nonnegative route-shift bound")
                    charge = max(charge, 0.)
                    if theta_values[min(route)] + 1e-7 < charge:
                        edges = list(zip([0] + route, route + [sink]))
                        # Equation (31), directed path version. Re-add violated
                        # cuts even if seen before: cached cuts may not be active.
                        current.cbLazy(theta[min(route)] >= charge * (gp.quicksum(x[a] for a in edges) - len(edges) + 1))
                        counts["theta"] += 1
            except Exception as error:  # Deliberate C callback boundary.
                errors.append(f"{type(error).__name__}: {error}")
                current.terminate()

        model.update()
        build_seconds = perf_counter() - started
        model.Params.TimeLimit = seconds
        model.optimize(callback)
        if errors:
            raise RuntimeError("branch-and-cut callback failed: " + errors[0])
        objective, replay, routes = None, None, None
        if model.SolCount:
            selected = [a for a in arcs if x[a].X > 0.5]
            routes, cycles = split_integer(selected, n)
            if cycles or any(sum(means[i - 1] for i in r) > qmax for r in routes):
                raise RuntimeError("invalid returned route witness")
            objective = float(model.ObjVal)
            replay = sum(route_cost(data, r, policy) for r in routes)
            if abs(objective - replay) > 1e-6 * (1. + abs(replay)):
                raise RuntimeError("Gurobi objective and independent recourse replay disagree")
        status_names = {GRB.OPTIMAL: "Optimal", GRB.INFEASIBLE: "Infeasible", GRB.TIME_LIMIT: "TimeLimit",
                        GRB.INTERRUPTED: "Interrupted", GRB.INF_OR_UNBD: "InfOrUnbounded"}
        bound = float(model.ObjBound) if hasattr(model, "ObjBound") else None
        gap = float(model.MIPGap) if model.SolCount else None
        return {"method": "bc_theta_directed_python", "paper_approach": 4, "source": SOURCE,
                "policy": policy, "instance_id": data.get("instance_id"),
                "status": status_names.get(model.Status, str(model.Status)), "objective": objective,
                "replayed_objective": replay, "lower_bound": bound if bound is not None and math.isfinite(bound) else None,
                "gap": gap if gap is not None and math.isfinite(gap) else None,
                "routes": routes, "nodes": model.NodeCount, "gurobi_version": list(gp.gurobi.version()),
                "threads": threads, "seed": seed, "time_limit_seconds": seconds,
                "build_seconds": build_seconds, "solve_seconds": model.Runtime,
                "total_seconds": perf_counter() - started, "recourse_shift_per_route": shift,
                "fractional_separator": ("exhaustive" if n <= 12 else "threshold_components_and_pairs") if fractional else "none",
                "cuts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--policy", choices=["p1", "p3"], default="p3")
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-fractional-cuts", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    raw = args.instance.read_bytes()
    answer = solve(json.loads(raw), args.policy, args.seconds, args.threads, args.seed,
                   not args.no_fractional_cuts, args.log)
    answer["input_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(answer, allow_nan=False))


if __name__ == "__main__":
    main()
