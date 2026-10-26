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
    """Routing data, including the depot at index 0.

    ``rounding`` is the distance convention of the source (``"none"`` for
    Solomon files, ``"nearest"`` for CVRPLIB EUC_2D, ``"ceil"`` for CEIL_2D).
    ``distances`` holds explicit CVRPLIB edge weights over the depot and
    customers; when given, coordinates and rounding do not define distances.
    """

    name: str
    capacity: float
    customers: Tuple[Customer, ...]
    vehicles: int = 1
    rounding: str = "none"
    distances: Optional[Tuple[Tuple[float, ...], ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "customers", tuple(self.customers))
        if not all(isinstance(c, Customer) for c in self.customers):
            raise TypeError("customers must contain Customer objects")
        if self.distances is not None:
            object.__setattr__(self, "distances", tuple(tuple(row) for row in self.distances))

    @classmethod
    def from_file(cls, path: Union[str, PathLike]) -> Instance:
        """Read a Solomon-format file or a CVRPLIB ``.vrp`` file.

        CVRPLIB customers have no time windows (infinite due times): solve
        them with ``problem_type="cvrp"``.
        """
        data = _core._read_instance(path)
        data["customers"] = tuple(Customer(*row) for row in data["customers"])
        return cls(**data)

    def truncated(self, customers: int) -> Instance:
        """Keep the first N customers plus the depot."""
        if not 0 < customers < len(self.customers):
            raise ValueError("customer limit out of range")
        distances = self.distances
        if distances is not None:
            distances = tuple(row[:customers + 1] for row in distances[:customers + 1])
        return replace(self, customers=self.customers[:customers + 1], distances=distances)

    def solve(self, **options) -> RootResult:
        from .solve import solve
        return solve(self, **options)


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
class FeasibilityReport:
    """An auxiliary phase-I round, separate from original-objective bounds."""

    status: str
    iterations: int
    pricing_calls: int
    simplex_iterations: int
    ipm_iterations: int
    seconds: float
    artificial_activity: Optional[float]
    lower_bound: Optional[float]
    certified_infeasible: bool


@dataclass(frozen=True)
class RootResult:
    status: str
    lp: float
    bound: float
    bound_mode: str
    numerical_bound: float
    certified_bound: float
    lp_before_cuts: float
    incumbent: float
    cuts: int
    cut_rounds: int
    iterations: int
    pricing_calls: int
    columns: int  # Route columns added over the run
    master_columns: int  # final master size, artificial columns included
    fixing_calls: int
    enumerated_routes: int
    pruned_by_bound: bool
    duals: Tuple[float, ...]
    seconds: float
    pricing_seconds: float
    solver_seconds: float
    routes: Tuple[RootRoute, ...]
    history: Tuple[Iteration, ...]
    simplex_iterations: int = 0
    ipm_iterations: int = 0
    feasibility: Tuple[FeasibilityReport, ...] = ()
    oracle_calls: int = 0

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
