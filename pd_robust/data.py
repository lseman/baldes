"""Shared instance conventions and independent robust route validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

TOLERANCE = 1e-6
Arc = tuple[int, int]


@dataclass(frozen=True)
class Settings:
    gamma_q: int = 1
    gamma_t: int = 1
    sigma_q: float = 0.1
    sigma_t: float = 0.1
    rounding: str = "hundredth"
    adjust_demands: bool = True
    vehicle_cost: float = 10_000.0

    def __post_init__(self) -> None:
        for value in (self.gamma_q, self.gamma_t):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("uncertainty budgets must be nonnegative integers")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.sigma_q, self.sigma_t, self.vehicle_cost)):
            raise ValueError("deviations and vehicle cost must be finite and nonnegative")
        if self.rounding not in ("hundredth", "none"):
            raise ValueError("rounding must be hundredth or none")


@dataclass(frozen=True)
class Node:
    x: float
    y: float
    service: float
    demand: float
    open: float
    close: float


@dataclass(frozen=True)
class Problem:
    name: str
    sha256: str
    n: int
    capacity: float
    nodes: tuple[Node, ...]
    settings: Settings
    q: tuple[float, ...]  # signed nominal demands, indexed by original node
    dq: tuple[float, ...]  # signed deviations
    t: tuple[tuple[float, ...], ...]
    dt: tuple[tuple[float, ...], ...]
    arcs: tuple[Arc, ...]

    @property
    def sink(self) -> int:
        return 2 * self.n + 1

    @property
    def customers(self) -> range:
        return range(1, self.sink)

    @property
    def pickups(self) -> range:
        return range(1, self.n + 1)

    def pair(self, node: int) -> int:
        if node not in self.customers:
            raise ValueError("depot has no request pair")
        return node + self.n if node <= self.n else node - self.n

    def cost(self, arc: Arc) -> float:
        i, j = arc
        return self.t[i][j] + (self.settings.vehicle_cost if i == 0 else 0)

    def load(self, onboard: Sequence[int]) -> float:
        return math.fsum(self.q[i] for i in onboard) + math.fsum(
            sorted((self.dq[i] for i in onboard), reverse=True)[:self.settings.gamma_q])

    def advance(self, starts: Sequence[float], i: int, j: int) -> tuple[float, ...]:
        duration = self.nodes[i].service + self.t[i][j]
        return tuple(max(self.nodes[j].open, starts[g] + duration,
                         starts[g - 1] + duration + self.dt[i][j] if g else -math.inf)
                     for g in range(len(starts)))

    def time_profile(self, path: Sequence[int], *, nominal: bool = False) -> tuple[tuple[float, ...], ...]:
        if not path or len(set(path)) != len(path) or any(v < 0 or v > self.sink for v in path):
            raise ValueError("path must contain distinct valid vertices")
        budget = 0 if nominal else min(self.settings.gamma_t, self.sink)
        profile = [(self.nodes[path[0]].open,) * (budget + 1)]
        for i, j in zip(path, path[1:]):
            profile.append(self.advance(profile[-1], i, j))
        return tuple(profile)

    def time_feasible(self, path: Sequence[int]) -> bool:
        return all(starts[-1] <= self.nodes[v].close + TOLERANCE
                   for v, starts in zip(path, self.time_profile(path)))

    def replay(self, route: Sequence[int]) -> dict:
        if len(route) < 4 or route[0] != 0 or route[-1] != self.sink:
            raise ValueError("route must serve a request between distinct depots")
        if any(v not in self.customers for v in route[1:-1]):
            raise ValueError("interior route node must be a customer")
        if len(set(route)) != len(route):
            raise ValueError("repeated route node")
        onboard: set[int] = set()
        loads = [0.0]
        for v in route[1:]:
            if v <= self.n:
                onboard.add(v)
            elif v != self.sink:
                request = v - self.n
                if request not in onboard:
                    raise ValueError("delivery before its pickup")
                onboard.remove(request)
            elif onboard:
                raise ValueError("route ends with undelivered requests")
            load = self.load(tuple(onboard))
            if load > self.capacity + TOLERANCE:
                raise ValueError(f"robust capacity exceeded at node {v}: {load} > {self.capacity}")
            loads.append(load)
        starts = self.time_profile(route)
        for v, values in zip(route, starts):
            if values[-1] > self.nodes[v].close + TOLERANCE:
                raise ValueError(f"robust window exceeded at node {v}: {values[-1]} > {self.nodes[v].close}")
        return dict(nodes=list(route), cost=math.fsum(self.cost(a) for a in zip(route, route[1:])),
                    worst_load=max(loads), worst_return=starts[-1][-1])

    def validate(self, routes: Sequence[Sequence[int]]) -> dict:
        details = [self.replay(route) for route in routes]
        coverage = [v for route in routes for v in route[1:-1]]
        if sorted(coverage) != list(self.customers):
            raise ValueError("solution must cover every customer exactly once")
        return dict(objective=math.fsum(r["cost"] for r in details), vehicles=len(details), routes=details)

    def metadata(self) -> dict:
        return dict(instance=self.name, instance_sha256=self.sha256, requests=self.n,
                    capacity=self.capacity, settings=asdict(self.settings))


def read_instance(path: Path, settings: Settings = Settings()) -> Problem:
    data = path.read_bytes()
    return parse_instance(data.decode("utf-8"), settings, name=path.name,
                          sha256=hashlib.sha256(data).hexdigest())


def parse_instance(text: str, settings: Settings = Settings(), *, name: str = "inline", sha256: str = "") -> Problem:
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if not lines or len(lines[0]) != 6 or lines[0][::2] != ["n", "Q", "r_mod"]:
        raise ValueError("expected header n N Q CAPACITY r_mod VALUE")
    n, capacity = int(lines[0][1]), float(lines[0][3])
    if n <= 0 or not math.isfinite(capacity) or capacity <= 0 or len(lines) != 2 * n + 3:
        raise ValueError("positive n/capacity and 2n+2 node rows required")
    nodes = []
    for v, fields in enumerate(lines[1:]):
        if len(fields) != 7 or int(fields[0]) != v:
            raise ValueError("consecutive ids and seven node fields required")
        node = Node(*(float(x) for x in fields[1:]))
        if any(not math.isfinite(x) for x in asdict(node).values()) or node.service < 0 or node.open > node.close:
            raise ValueError("nonfinite data, negative service or reversed window")
        nodes.append(node)
    for v in (0, 2 * n + 1):
        if nodes[v].demand != 0 or nodes[v].service != 0:
            raise ValueError("depot demand and service must be zero")
    q, dq = [0.0] * len(nodes), [0.0] * len(nodes)
    for i in range(1, n + 1):
        raw = nodes[i].demand
        if raw <= 0 or nodes[i + n].demand != -raw:
            raise ValueError("pickup i must pair with opposite demand at i+n")
        adjustment = math.ceil(0.3 * raw)
        value = capacity - adjustment if settings.adjust_demands and raw + adjustment > capacity else raw
        if value <= 0:
            raise ValueError("nonpositive adjusted request demand")
        q[i], q[i + n] = value, -value
        deviation = settings.sigma_q * value
        if not math.isfinite(deviation):
            raise ValueError("demand deviation overflow")
        dq[i] = math.ceil(deviation)
        dq[i + n] = -dq[i]
    def distance(a: Node, b: Node) -> float:
        value = math.hypot(a.x - b.x, a.y - b.y)
        if not math.isfinite(value) or (settings.rounding == "hundredth" and not math.isfinite(value * 100)):
            raise ValueError("distance overflow")
        # Match Rust f64::round for nonnegative values, rather than Python tie-to-even.
        return math.floor(value * 100 + 0.5) / 100 if settings.rounding == "hundredth" else value
    t = tuple(tuple(distance(a, b) for b in nodes) for a in nodes)
    def time_deviation(value: float) -> float:
        scaled = settings.sigma_t * value
        if not math.isfinite(scaled) or not math.isfinite(value + math.ceil(scaled) + settings.vehicle_cost):
            raise ValueError("travel time or cost overflow")
        return float(math.ceil(scaled))
    dt = tuple(tuple(time_deviation(x) for x in row) for row in t)
    sink = 2 * n + 1
    arcs = tuple((i, j) for i in range(sink) for j in range(1, sink + 1)
                 if i != j and not (i == 0 and j > n) and not (j == sink and i <= n)
                 and i != j + n
                 and nodes[i].open + nodes[i].service + t[i][j]
                 + (dt[i][j] if settings.gamma_t else 0) <= nodes[j].close)
    return Problem(name, sha256, n, capacity, tuple(nodes), settings, tuple(q), tuple(dq), t, dt, arcs)


def extract_routes(problem: Problem, values: Mapping[Arc, float]) -> list[list[int]]:
    successors: dict[int, int] = {}
    starts = []
    chosen = {(i, j) for (i, j), value in values.items() if value > 0.5}
    for i, j in chosen:
        if i == 0:
            starts.append(j)
        elif i in successors:
            raise ValueError("multiple outgoing arcs in integer solution")
        else:
            successors[i] = j
    routes = []
    used = set()
    for first in sorted(starts):
        route = [0, first]
        while route[-1] != problem.sink:
            tail = route[-1]
            if tail in used or tail not in successors:
                raise ValueError("cycle or disconnected customer in integer solution")
            used.add(tail)
            route.append(successors[tail])
        routes.append(route)
    if used != set(problem.customers) or sum(len(r) - 1 for r in routes) != len(chosen):
        raise ValueError("integer arcs do not form complete depot routes")
    return routes
