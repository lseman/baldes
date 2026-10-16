"""Typed inputs and results for the Rust column-and-cut generation root solver.

Root results describe an LP relaxation, not a complete integer VRP solve.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from os import PathLike
from typing import Mapping, Optional, Tuple, Union

from . import _core


@dataclass(frozen=True)
class Customer:
    """Depot/customer data. VRPTW requires explicit finite due times.

    CVRP ignores ready/due/service. The depot appears first with zero demand.
    """

    x: float
    y: float
    demand: float = 0.0
    ready: float = 0.0
    due: float = math.inf
    service: float = 0.0


@dataclass(frozen=True)
class Instance:
    """Routing data, including the depot at index 0."""

    name: str
    capacity: float
    customers: Tuple[Customer, ...]
    vehicles: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "customers", tuple(self.customers))
        if not all(isinstance(c, Customer) for c in self.customers):
            raise TypeError("customers must contain Customer objects")

    @classmethod
    def from_file(cls, path: Union[str, PathLike]) -> Instance:
        """Read a Solomon-format file, including converted CVRP instances."""
        data = _core._read_instance(path)
        data["customers"] = tuple(Customer(*row) for row in data["customers"])
        return cls(**data)

    def truncated(self, customers: int) -> Instance:
        """Keep the first N customers plus the depot."""
        if not 0 < customers < len(self.customers):
            raise ValueError("customer limit out of range")
        return replace(self, customers=self.customers[:customers + 1])

    def solve_root(self, **options) -> RootResult:
        return solve_root(self, **options)


@dataclass(frozen=True)
class RootRoute:
    """A positive LP column; value can be fractional and nodes may repeat under NG."""

    value: float
    cost: float
    nodes: Tuple[int, ...]
    arcs: Tuple[int, ...]


@dataclass(frozen=True)
class Iteration:
    iteration: int
    lp: float
    minimum_reduced_cost: float
    new_columns: int
    columns: int
    exact: bool
    phase: int
    pricing_calls: int
    alpha: float
    misprices: int


@dataclass(frozen=True)
class RootResult:
    status: str
    lp: float
    bound: float
    lp_before_cuts: float
    incumbent: float
    cuts: int
    cut_rounds: int
    iterations: int
    pricing_calls: int
    columns: int
    fixing_calls: int
    enumerated_routes: int
    pruned_by_bound: bool
    duals: Tuple[float, ...]
    seconds: float
    pricing_seconds: float
    solver_seconds: float
    routes: Tuple[RootRoute, ...]
    history: Tuple[Iteration, ...]

    def to_dict(self) -> dict:
        """Return ordinary Python data, retaining infinite unknown incumbents."""
        return asdict(self)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize to strict JSON; non-finite numbers become null."""
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [clean(v) for v in value]
            return value

        return json.dumps(clean(self.to_dict()), indent=indent, allow_nan=False)


def solve_root(
    instance: Union[Instance, str, PathLike],
    *,
    problem_type: str = "vrptw",
    customers: Optional[int] = None,
    ng: int = 8,
    reference_ng: bool = False,
    rounding: str = "none",
    capacity_cuts: bool = False,
    rank_one_cuts: bool = False,
    stabilize: bool = True,
    reduced_cost_fixing: bool = True,
    upper_bound: float = math.inf,
    time_limit: float = 120.0,
    max_iterations: int = 10_000,
    max_cut_rounds: Optional[int] = None,
    pricing_options: Optional[Mapping[str, float]] = None,
) -> RootResult:
    """Solve a VRPTW/CVRP root relaxation directly in Rust, releasing the GIL.

    Input paths use Solomon format. CVRP ignores time windows and service times.
    Costs are unrounded by default; rounding also accepts 'nearest' and 'tenth'.
    Capacity cuts use complement form; rank-one cuts use limited memory.
    Disabling reduced_cost_fixing also disables route enumeration. Cut counts
    include generated rows that were later retired. These are root relaxations:
    status 'Optimal' does not mean an optimal integer routing solution.
    """
    if not _core.has_clp:
        raise RuntimeError("root solving requires pybaldes built with the clp feature")
    if not isinstance(instance, Instance):
        instance = Instance.from_file(instance)
    if customers is not None:
        instance = instance.truncated(customers)
    rows = [(c.x, c.y, c.demand, c.ready, c.due, c.service) for c in instance.customers]
    data = _core._solve_root(
        instance.name, instance.vehicles, instance.capacity, rows, problem_type,
        ng, reference_ng, rounding, capacity_cuts, rank_one_cuts, stabilize,
        reduced_cost_fixing, upper_bound, time_limit, max_iterations,
        max_cut_rounds, dict(pricing_options or {}),
    )
    data["duals"] = tuple(data["duals"])
    data["routes"] = tuple(
        RootRoute(**{**r, "nodes": tuple(r["nodes"]), "arcs": tuple(r["arcs"])})
        for r in data["routes"]
    )
    data["history"] = tuple(Iteration(**h) for h in data["history"])
    return RootResult(**data)
