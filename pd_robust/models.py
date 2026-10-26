"""Gurobi implementations of the paper's compact and cutting-plane models."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

from .data import Arc, Problem, TOLERANCE, extract_routes
from .separators import Cut, separate_ripei, separate_rrci

METHODS = ("compact", "branch-cut", "compact-guarded", "onboard-compact")


@dataclass(frozen=True)
class SolveOptions:
    seconds: float = 60.0
    threads: int = 1
    seed: int = 0
    cuts_per_family: int = 50
    root_user_cuts: bool = True
    warm_start: bool = True
    literal_big_m: bool = False
    log: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or not 0 < self.seconds <= 86400:
            raise ValueError("seconds must be finite and in (0, 86400]")
        if self.threads < 0 or self.seed < 0 or self.cuts_per_family <= 0:
            raise ValueError("threads/seed must be nonnegative and cut limit must be positive")


@dataclass
class BuiltModel:
    model: Any
    x: dict[Arc, Any]
    problem: Problem
    method: str
    options: SolveOptions
    cut_counts: dict[str, int] = field(default_factory=lambda: {
        "rrci_lazy": 0, "ripei_lazy": 0, "rrci_user": 0, "ripei_user": 0,
        "mipsol_calls": 0, "root_node_calls": 0})
    separation_seconds: float = 0.0
    callback_error: Exception | None = None
    user_cut_keys: set[Cut] = field(default_factory=set)


def _times(model, x, p: Problem, *, robust: bool, literal_big_m: bool):
    budget = min(p.settings.gamma_t, p.sink) if robust else 0
    b = {(v, g): model.addVar(lb=p.nodes[v].open, ub=p.nodes[v].close, name=f"B[{v},{g}]")
         for v in range(p.sink + 1) for g in range(budget + 1)}
    for g in range(budget + 1):
        b[0, g].UB = p.nodes[0].open
        for i, j in p.arcs:
            duration = p.nodes[i].service + p.t[i][j]
            big_m = p.nodes[i].close - p.nodes[j].open
            if not literal_big_m:
                big_m = max(0.0, big_m + duration)
            model.addConstr(b[j, g] >= b[i, g] + duration - big_m * (1 - x[i, j]),
                            name=f"time28[{i},{j},{g}]")
            if g:
                uncertain_m = big_m if literal_big_m else max(
                    0.0, p.nodes[i].close + duration + p.dt[i][j] - p.nodes[j].open)
                model.addConstr(b[j, g] >= b[i, g - 1] + duration + p.dt[i][j]
                                - uncertain_m * (1 - x[i, j]), name=f"time29[{i},{j},{g}]")
        for i in p.pickups:
            model.addConstr(b[i + p.n, g] >= b[i, g] + p.nodes[i].service + p.t[i][i + p.n],
                            name=f"precedence30[{i},{g}]")
    return b


def _nominal_load(model, x, p: Problem):
    q = {v: model.addVar(lb=max(0.0, p.q[v]), ub=min(p.capacity, p.capacity + p.q[v]),
                         name=f"nominal_load[{v}]") for v in range(p.sink + 1)}
    q[0].UB = q[p.sink].UB = 0
    for i, j in p.arcs:
        model.addConstr(q[j] >= q[i] + p.q[j] - p.capacity * (1 - x[i, j]), name=f"load9[{i},{j}]")


def _paper_load(model, x, p: Problem):
    """Printed (19)–(26); retain its W inequalities rather than silently tightening them."""
    import gurobipy as gp
    budget = min(p.settings.gamma_q, p.n)
    q = {(v, g): model.addVar(lb=max(0.0, p.q[v]), ub=min(p.capacity, p.capacity + p.q[v]),
                              name=f"Q[{v},{g}]")
         for v in range(p.sink + 1) for g in range(budget + 1)}
    w = {(v, g, r): model.addVar(vtype=gp.GRB.BINARY, ub=0 if v == 0 else 1,
                                name=f"W[{v},{g},{r}]")
         for v in range(p.sink + 1) for g in range(1, budget + 1) for r in p.pickups}
    def marked(v, g, r):
        return w[v, g, r] if g else 0.0
    largest_deviation = max(p.dq[r] for r in p.pickups)
    for g in range(budget + 1):
        q[0, g].UB = q[p.sink, g].UB = 0
        for i, j in p.arcs:
            if j <= p.n:
                model.addConstr(q[j, g] >= q[i, g] + p.q[j] * x[i, j]
                                - p.capacity * (1 - x[i, j]), name=f"load20[{i},{j},{g}]")
                if g:
                    model.addConstr(q[j, g] >= q[i, g - 1] + (p.q[j] + p.dq[j]) * x[i, j]
                                    - p.capacity * (1 - x[i, j]), name=f"load21[{i},{j},{g}]")
                    denominator = p.dq[j] + largest_deviation
                    if denominator:
                        expression = (q[i, g - 1] + p.dq[j] - q[i, g]) / denominator
                        model.addConstr(w[j, g, j] >= expression + x[i, j] - 1,
                                        name=f"marked23[{i},{j},{g}]")
                        model.addConstr(w[j, g, j] >= marked(i, g - 1, j) + expression + x[i, j] - 1,
                                        name=f"marked24[{i},{j},{g}]")
            elif j != p.sink:
                model.addConstr(q[j, g] >= q[i, g] + p.q[j] * x[i, j]
                                + p.dq[j] * marked(i, g, j - p.n)
                                - p.capacity * (1 - x[i, j]), name=f"load22[{i},{j},{g}]")
            if g:
                for r in p.pickups:
                    model.addConstr(w[j, g, r] >= w[i, g, r] + x[i, j] - 1,
                                    name=f"marked25[{i},{j},{g},{r}]")


def _onboard_load(model, x, p: Problem):
    """Exact alternative compact load model derived from Proposition 1.

    Binary O[v,r] tracks on-board requests. Bertsimas–Sim duality represents
    the top-Γ deviations with theta/z; this is not printed model (19)–(26).
    """
    import gurobipy as gp
    onboard = {(v, r): model.addVar(vtype=gp.GRB.BINARY,
                                    lb=1 if v == r else 0,
                                    ub=0 if v in (0, p.sink, r + p.n) else 1,
                                    name=f"O[{v},{r}]")
               for v in range(p.sink + 1) for r in p.pickups}
    for i, j in p.arcs:
        for r in p.pickups:
            change = int(j == r) - int(j == r + p.n)
            expression = onboard[j, r] - onboard[i, r] - change
            model.addConstr(expression <= 1 - x[i, j])
            model.addConstr(expression >= x[i, j] - 1)
    budget = min(p.settings.gamma_q, p.n)
    for v in p.customers:
        nominal = gp.quicksum(p.q[r] * onboard[v, r] for r in p.pickups)
        if budget == 0:
            model.addConstr(nominal <= p.capacity)
        elif budget == p.n:
            model.addConstr(nominal + gp.quicksum(p.dq[r] * onboard[v, r] for r in p.pickups) <= p.capacity)
        else:
            theta = model.addVar(ub=max(p.dq[r] for r in p.pickups), name=f"theta[{v}]")
            z = {r: model.addVar(name=f"z[{v},{r}]") for r in p.pickups}
            for r in p.pickups:
                model.addConstr(z[r] >= p.dq[r] * onboard[v, r] - theta)
            model.addConstr(nominal + budget * theta + gp.quicksum(z.values()) <= p.capacity)


def initial_routes(p: Problem) -> list[list[int]]:
    """§6: insert adjacent pickup/delivery pairs using the hard worst case."""
    from dataclasses import replace
    hard = replace(p, settings=replace(p.settings, gamma_q=p.n, gamma_t=p.sink))
    routes: list[list[int]] = []
    available = set(p.arcs)
    for request in p.pickups:
        candidates = []
        for k, route in enumerate(routes):
            for position in range(1, len(route)):
                candidate = route[:position] + [request, request + p.n] + route[position:]
                if not all(a in available for a in zip(candidate, candidate[1:])):
                    continue
                try:
                    details = hard.replay(candidate)
                except ValueError:
                    continue
                increase = details["cost"] - hard.replay(route)["cost"]
                candidates.append((increase, k, candidate))
        if candidates:
            _, k, routes[k] = min(candidates)
        else:
            candidate = [0, request, request + p.n, p.sink]
            if not all(a in available for a in zip(candidate, candidate[1:])):
                return []
            try:
                hard.replay(candidate)
            except ValueError:
                return []  # no start is preferable to an infeasible start
            routes.append(candidate)
    return routes


def build_model(p: Problem, method: str, options: SolveOptions, *, env=None) -> BuiltModel:
    import gurobipy as gp
    if method not in METHODS:
        raise ValueError(f"unknown formulation {method}")
    m = gp.Model(f"rpdptw_{method}", env=env)
    m.Params.OutputFlag = int(options.log)
    m.Params.TimeLimit = options.seconds
    m.Params.Threads = options.threads
    m.Params.Seed = options.seed
    m.Params.MIPGap = 0
    m.Params.MIPGapAbs = 1e-6
    x = dict(m.addVars(p.arcs, vtype=gp.GRB.BINARY, obj={a: p.cost(a) for a in p.arcs}, name="x"))
    for v in p.customers:
        m.addConstr(gp.quicksum(var for (i, j), var in x.items() if j == v) == 1, name=f"in8[{v}]")
        m.addConstr(gp.quicksum(var for (i, j), var in x.items() if i == v) == 1, name=f"out8[{v}]")
    tags = {v: m.addVar(lb=1, ub=p.n, name=f"v[{v}]") for v in p.customers}
    # Position prevents zero-distance cycles and gives precedence even at coincident nodes.
    order = {v: m.addVar(lb=1, ub=2 * p.n, name=f"position[{v}]") for v in p.customers}
    for r in p.pickups:
        m.addConstr(tags[r] == tags[r + p.n], name=f"pair14[{r}]")
        m.addConstr(order[r + p.n] >= order[r] + 1)
    for v in p.customers:
        first = x.get((0, v), 0)
        m.addConstr(tags[v] >= v * first, name=f"first15lo[{v}]")
        m.addConstr(tags[v] <= v * first + p.n * (1 - first), name=f"first15hi[{v}]")
    for (i, j), var in x.items():
        if i and j != p.sink:
            m.addConstr(tags[j] - tags[i] <= p.n * (1 - var), name=f"pair16hi[{i},{j}]")
            m.addConstr(tags[j] - tags[i] >= -p.n * (1 - var), name=f"pair16lo[{i},{j}]")
            m.addConstr(order[j] >= order[i] + 1 - 2 * p.n * (1 - var))
    _times(m, x, p, robust=method != "branch-cut", literal_big_m=options.literal_big_m)
    if method == "branch-cut":
        _nominal_load(m, x, p)
    elif method == "onboard-compact":
        _onboard_load(m, x, p)
    else:
        _paper_load(m, x, p)
    # A request cannot be split: reject a known impossible individual demand.
    if any(p.q[r] + (p.dq[r] if p.settings.gamma_q else 0) > p.capacity for r in p.pickups):
        m.addConstr(gp.LinExpr() >= 1, name="impossible_individual_demand")
    if method in ("branch-cut", "compact-guarded"):
        m.Params.LazyConstraints = 1
        m.Params.PreCrush = 1
    if options.warm_start:
        routes = initial_routes(p)
        if routes:
            selected = {a for route in routes for a in zip(route, route[1:])}
            for arc, var in x.items():
                var.Start = int(arc in selected)
    m.update()
    return BuiltModel(m, x, p, method, options)


def _callback(built: BuiltModel, model, where: int) -> None:
    import gurobipy as gp
    started = time.perf_counter()
    try:
        integer = where == gp.GRB.Callback.MIPSOL
        if integer:
            values = dict(zip(built.x, model.cbGetSolution(list(built.x.values()))))
            built.cut_counts["mipsol_calls"] += 1
        elif (where == gp.GRB.Callback.MIPNODE and built.options.root_user_cuts
              and model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT) == 0
              and model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL):
            values = dict(zip(built.x, model.cbGetNodeRel(list(built.x.values()))))
            built.cut_counts["root_node_calls"] += 1
        else:
            return
        cuts = separate_rrci(built.problem, values, built.options.cuts_per_family)
        cuts += separate_ripei(built.problem, values, built.options.cuts_per_family, integer=integer)
        for cut in cuts:
            if not integer and cut in built.user_cut_keys:
                continue
            expression = gp.quicksum(built.x[a] for a in cut.arcs)
            constraint = expression >= cut.rhs if cut.kind == "rrci" else expression <= cut.rhs
            if integer:
                # Previously submitted lazy cuts must be resubmitted when violated.
                model.cbLazy(constraint)
            else:
                model.cbCut(constraint)
                built.user_cut_keys.add(cut)
            built.cut_counts[f"{cut.kind}_{'lazy' if integer else 'user'}"] += 1
    except Exception as error:
        # Gurobi callback exceptions otherwise need not abort optimization reliably.
        built.callback_error = error
        model.terminate()
    finally:
        built.separation_seconds += time.perf_counter() - started


def solve(p: Problem, method: str = "branch-cut", options: SolveOptions = SolveOptions()) -> dict:
    import gurobipy as gp
    setup = time.perf_counter()
    with gp.Env(params={"OutputFlag": int(options.log)}) as env:
        environment_seconds = time.perf_counter() - setup
        started = time.perf_counter()
        built = build_model(p, method, options, env=env)
        with built.model as model:
            build_seconds = time.perf_counter() - started
            if method in ("branch-cut", "compact-guarded"):
                model.optimize(lambda m, where: _callback(built, m, where))
            else:
                model.optimize()
            algorithm_seconds = time.perf_counter() - started
            if built.callback_error is not None:
                raise RuntimeError("robust separation callback failed") from built.callback_error
            status_names = {gp.GRB.OPTIMAL: "Optimal", gp.GRB.TIME_LIMIT: "TimeLimit",
                            gp.GRB.INFEASIBLE: "Infeasible", gp.GRB.INTERRUPTED: "Interrupted",
                            gp.GRB.INF_OR_UNBD: "InfeasibleOrUnbounded"}
            result = dict(**p.metadata(), method=method, status=status_names.get(model.Status, str(model.Status)),
                          solver="gurobi", solver_version=".".join(map(str, gp.gurobi.version())),
                          environment_seconds=environment_seconds, build_seconds=build_seconds,
                          solver_seconds=model.Runtime, algorithm_seconds=algorithm_seconds,
                          separation_seconds=built.separation_seconds, cuts=built.cut_counts,
                          nodes=model.NodeCount, variables=model.NumVars, constraints=model.NumConstrs,
                          objective=None, bound=None, gap=None, valid=None, routes=[], validation_error=None,
                          threads=options.threads, seed=options.seed,
                          big_m="printed" if options.literal_big_m else "safe",
                          warm_start=options.warm_start, root_user_cuts=options.root_user_cuts)
            if math.isfinite(model.ObjBound) and abs(model.ObjBound) < gp.GRB.INFINITY:
                result["bound"] = model.ObjBound
            if model.SolCount:
                result["objective"], result["gap"] = model.ObjVal, model.MIPGap
                routes = extract_routes(p, {arc: var.X for arc, var in built.x.items()})
                result["routes"] = routes
                try:
                    replay = p.validate(routes)
                    if abs(replay["objective"] - model.ObjVal) > 1e-5:
                        raise ValueError("replayed objective differs from model objective")
                    result.update(valid=True, replay=replay)
                except ValueError as error:
                    result.update(valid=False, validation_error=str(error))
            return result
