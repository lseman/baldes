"""Shared-master models with multiple pricing networks and resumable nodes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

from . import _core


@dataclass(frozen=True)
class SharedRow:
    sense: str  # "eq", "le", "ge"
    rhs: float


@dataclass(frozen=True)
class PricingResource:
    kind: str = "disposable"
    initial: Optional[float] = None
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None

    def __post_init__(self) -> None:
        if self.bounds is not None:
            object.__setattr__(self, "bounds", tuple(tuple(b) for b in self.bounds))


@dataclass(frozen=True)
class PricingArc:
    tail: int
    head: int
    cost: float
    consumption: Tuple[float, ...] = ()
    bounds: Tuple[Optional[Tuple[float, float]], ...] = ()
    entries: Tuple[Tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumption", tuple(self.consumption))
        object.__setattr__(self, "bounds", tuple(None if b is None else tuple(b) for b in self.bounds))
        object.__setattr__(self, "entries", tuple(tuple(e) for e in self.entries))


@dataclass(frozen=True)
class PricingNetwork:
    id: int
    vertices: int
    multiplicity: Tuple[float, float]
    arcs: Tuple[PricingArc, ...]
    resources: Tuple[PricingResource, ...] = ()
    source: int = 0
    sink: Optional[int] = None
    route_entries: Tuple[Tuple[int, float], ...] = ()
    ng: Optional[Tuple[Tuple[int, ...], ...]] = None
    coupled_resources: Tuple[Mapping[str, Any], ...] = ()
    elementary: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "multiplicity", tuple(self.multiplicity))
        object.__setattr__(self, "arcs", tuple(self.arcs))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "coupled_resources", tuple(dict(r) for r in self.coupled_resources))
        object.__setattr__(self, "route_entries", tuple(tuple(e) for e in self.route_entries))
        if self.ng is not None:
            object.__setattr__(self, "ng", tuple(tuple(n) for n in self.ng))
        if not all(isinstance(a, PricingArc) for a in self.arcs):
            raise TypeError("arcs must contain PricingArc objects")
        if not all(isinstance(r, PricingResource) for r in self.resources):
            raise TypeError("resources must contain PricingResource objects")


@dataclass(frozen=True)
class NetworkProblem:
    rows: Tuple[SharedRow, ...]
    networks: Tuple[PricingNetwork, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "networks", tuple(self.networks))
        if not all(isinstance(r, SharedRow) for r in self.rows):
            raise TypeError("rows must contain SharedRow objects")
        if not all(isinstance(n, PricingNetwork) for n in self.networks):
            raise TypeError("networks must contain PricingNetwork objects")

    @classmethod
    def from_json(cls, text: str) -> NetworkProblem:
        data = json.loads(text)
        data["rows"] = tuple(SharedRow(**row) for row in data["rows"])
        networks = []
        for network in data["networks"]:
            network["arcs"] = tuple(PricingArc(**arc) for arc in network["arcs"])
            network["resources"] = tuple(PricingResource(**r) for r in network.get("resources", ()))
            networks.append(PricingNetwork(**network))
        data["networks"] = tuple(networks)
        return cls(**data)

    @classmethod
    def from_file(cls, path: Union[str, PathLike]) -> NetworkProblem:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """The same model format accepted by ``baldes networks MODEL.json``."""
        return json.dumps(asdict(self), allow_nan=False, indent=indent)

    def solve(self, **options: Any) -> NetworkResult:
        return solve_networks(self, **options)


@dataclass(frozen=True)
class NetworkRowCoefficients:
    network: int
    arcs: Tuple[float, ...]
    per_route: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "arcs", tuple(self.arcs))


@dataclass(frozen=True)
class NetworkResourceGroupMember:
    network: int
    arcs: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arcs", tuple(self.arcs))


@dataclass(frozen=True)
class NetworkAssignment:
    shared_row: int
    network: int
    value: float


@dataclass(frozen=True)
class NetworkRoute:
    network: int
    route_index: Optional[int]
    value: float
    cost: float
    nodes: Tuple[int, ...]
    arcs: Tuple[int, ...]
    resources: Tuple[float, ...]
    resource_profile: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        for name in ("nodes", "arcs", "resources"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "resource_profile", tuple(tuple(r) for r in self.resource_profile))


@dataclass(frozen=True)
class NetworkEnumerationReport:
    attempts: int
    complete: bool
    networks: Tuple[Tuple[int, int], ...]
    routes: int
    transferred: int
    failure: Optional[str]
    coverage_cutoff: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "networks", tuple(tuple(n) for n in self.networks))


@dataclass(frozen=True)
class NetworkNode:
    """Immutable, thread-bound native snapshot. Child operations preserve parents."""
    _native: Any = field(repr=False, compare=False)

    def solve(self, **options: Any) -> NetworkResult:
        return _result(*self._native.solve(_options(options)))

    def branch_assignment(self, shared_row: int, network: int, assigned: bool) -> NetworkNode:
        return NetworkNode(self._native.branch_assignment(shared_row, network, assigned))

    def with_row(self, row: SharedRow, coefficients: Tuple[NetworkRowCoefficients, ...]) -> NetworkNode:
        data = {"row": asdict(row), "coefficients": [asdict(c) for c in coefficients]}
        return NetworkNode(self._native.with_row(json.dumps(data, allow_nan=False)))

    def with_resource_group_branch(
        self, packing_row: int, resource: int, threshold: float, upper: bool,
        members: Tuple[NetworkResourceGroupMember, ...],
    ) -> NetworkNode:
        data = dict(packing_row=packing_row, resource=resource, threshold=threshold,
                    upper=upper, members=[asdict(m) for m in members])
        return NetworkNode(self._native.with_resource_group_branch(json.dumps(data, allow_nan=False)))

    def fix_route(self, route_index: int, quantity: int = 1) -> NetworkNode:
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise TypeError("quantity must be a positive integer")
        if quantity <= 0 or quantity > 9_007_199_254_740_992:
            raise ValueError("quantity must be between 1 and 2**53")
        return NetworkNode(self._native.fix_route(route_index, quantity))

    def solve_enumerated_mip(self, *, time_limit: float = 1.0, node_limit: int = 100000,
                             objective_cutoff: Optional[float] = None) -> Optional[NetworkResult]:
        """Solve a complete transferred copy with HiGHS, or return None if unprepared."""
        text = self._native.solve_enumerated_mip(time_limit, node_limit, objective_cutoff)
        return None if text is None else _result(text, None)

    def fractional_assignments(self, tolerance: float = 1e-6) -> Tuple[NetworkAssignment, ...]:
        return tuple(NetworkAssignment(*c) for c in self._native.fractional_assignments(tolerance))


@dataclass(frozen=True)
class NetworkResult:
    mode: str
    status: str
    bound_mode: str
    lp: Optional[float]
    bound: Optional[float]
    numerical_bound: Optional[float]
    certified_bound: Optional[float]
    incumbent: Optional[float]
    routes: Tuple[NetworkRoute, ...]
    fixed_routes: Tuple[NetworkRoute, ...]
    duals: Tuple[float, ...]
    iterations: int
    pricing_calls: int
    oracle_calls: Tuple[Tuple[int, int], ...]
    nodes: int
    open_nodes: int
    seconds: float
    enumeration_reports: Tuple[NetworkEnumerationReport, ...]
    enumerated_mip_calls: int
    enumerated_mip_nodes: int
    enumerated_mip_failures: int
    snapshot: Optional[NetworkNode] = field(default=None, repr=False, compare=False)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        data = {k: v for k, v in vars(self).items() if k != "snapshot"}
        data["routes"] = [asdict(r) for r in self.routes]
        data["fixed_routes"] = [asdict(r) for r in self.fixed_routes]
        data["enumeration_reports"] = [asdict(r) for r in self.enumeration_reports]
        return json.dumps(data, allow_nan=False, indent=indent)


def _result(text: str, native: Any) -> NetworkResult:
    data = json.loads(text)
    data["routes"] = tuple(NetworkRoute(**r) for r in data["routes"])
    data["fixed_routes"] = tuple(NetworkRoute(**r) for r in data["fixed_routes"])
    data["enumeration_reports"] = tuple(NetworkEnumerationReport(**r) for r in data["enumeration_reports"])
    data["duals"] = tuple(data["duals"])
    data["oracle_calls"] = tuple(tuple(c) for c in data["oracle_calls"])
    return NetworkResult(**data, snapshot=None if native is None else NetworkNode(native))


def _options(options: Mapping[str, Any]) -> str:
    options = dict(options)
    if "pricing_options" in options:
        options["pricing_options"] = {} if options["pricing_options"] is None else dict(options["pricing_options"])
    return json.dumps(options, allow_nan=False)


def solve_networks(
    problem: NetworkProblem,
    *,
    tree: bool = False,
    enumerate: bool = False,
    enumeration_route_limit: int = 10000,
    enumeration_work_limit: int = 10000000,
    enumeration_stored_arc_limit: int = 1000000,
    enumeration_time_limit: float = 1.0,
    enumerated_mip: bool = False,
    mip_time_limit: float = 1.0,
    mip_node_limit: int = 100000,
    mip_objective_cutoff: Optional[float] = None,
    time_limit: float = 120.0,
    max_iterations: int = 100000,
    node_limit: int = 10000,
    stabilize: bool = True,
    bound_mode: str = "numerical",
    artificial_cost: float = 2.76,
    log_interval: Optional[int] = None,
    pricing_options: Optional[Mapping[str, float]] = None,
) -> NetworkResult:
    """Solve a shared LP or assignment tree. Unknown bounds serialize as null.

    Roots retain thread-bound snapshots; these calls keep the GIL. Fresh
    assignment-tree solves release the GIL and return an owned integer witness.
    A tree may return Incomplete when another branching family is required.
    """
    if not isinstance(problem, NetworkProblem):
        raise TypeError("problem must be a NetworkProblem")
    if not _core.has_clp:
        raise RuntimeError("multi-network solving requires a build with Clp")
    options = dict(tree=tree, enumerate=enumerate, enumeration_route_limit=enumeration_route_limit,
                   enumeration_work_limit=enumeration_work_limit, enumeration_stored_arc_limit=enumeration_stored_arc_limit,
                   enumeration_time_limit=enumeration_time_limit, enumerated_mip=enumerated_mip,
                   mip_time_limit=mip_time_limit, mip_node_limit=mip_node_limit,
                   mip_objective_cutoff=mip_objective_cutoff,
                   time_limit=time_limit, max_iterations=max_iterations,
                   node_limit=node_limit, stabilize=stabilize, bound_mode=bound_mode,
                   artificial_cost=artificial_cost, log_interval=log_interval,
                   pricing_options={} if pricing_options is None else dict(pricing_options))
    return _result(*_core._solve_networks(problem.to_json(indent=None), _options(options)))
