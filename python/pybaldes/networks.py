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
    """Scalar resource with optional per-vertex service-start windows.

    ``time_windows`` requires a build with ``multi-time-windows``. Each vertex
    entry is None (ordinary bounds) or a nonempty sequence of (lower, upper)
    intervals. Ordinary bounds intersect the configured windows.
    """

    kind: str = "disposable"
    initial: Optional[float] = None
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None
    time_windows: Optional[Tuple[Optional[Tuple[Tuple[float, float], ...]], ...]] = None

    def __post_init__(self) -> None:
        if self.bounds is not None:
            object.__setattr__(self, "bounds", tuple(tuple(b) for b in self.bounds))
        if self.time_windows is not None:
            object.__setattr__(self, "time_windows", tuple(
                None if windows is None else tuple(tuple(w) for w in windows)
                for windows in self.time_windows
            ))


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
    """One route family with an explicit original resource domain.

    front_program_domain=True selects the computed front program
    for FrontAlgebra resources; the default retains source feasibility.
    """

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
    front_program_domain: bool = False
    route_domain: Mapping[str, Any] = field(default_factory=lambda: {"kind": "physical_simple"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "multiplicity", tuple(self.multiplicity))
        object.__setattr__(self, "arcs", tuple(self.arcs))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "coupled_resources", tuple(dict(r) for r in self.coupled_resources))
        object.__setattr__(self, "route_domain", dict(self.route_domain))
        object.__setattr__(self, "route_entries", tuple(tuple(e) for e in self.route_entries))
        if self.ng is not None:
            object.__setattr__(self, "ng", tuple(tuple(n) for n in self.ng))
        if not all(isinstance(a, PricingArc) for a in self.arcs):
            raise TypeError("arcs must contain PricingArc objects")
        if not all(isinstance(r, PricingResource) for r in self.resources):
            raise TypeError("resources must contain PricingResource objects")


@dataclass(frozen=True)
class SynchronizedServiceMember:
    network: int
    vertex: int


@dataclass(frozen=True)
class SynchronizedService:
    """Optional service by distinct vehicles at one common integer start.

    ``start`` is an inclusive pair of integer resource values. The native
    master adds coverage and a single reward; route costs must exclude that
    reward. Each member identifies the service vertex in one fleet network.
    """

    id: int
    resource: int
    vehicles_required: int
    reward: float
    start: Tuple[int, int]
    members: Tuple[SynchronizedServiceMember, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", tuple(self.start))
        object.__setattr__(self, "members", tuple(self.members))
        if not all(isinstance(m, SynchronizedServiceMember) for m in self.members):
            raise TypeError("members must contain SynchronizedServiceMember objects")


@dataclass(frozen=True)
class NetworkProblem:
    rows: Tuple[SharedRow, ...]
    networks: Union[PricingNetwork, Tuple[PricingNetwork, ...]]
    pricing_variables: Tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pricing_variables", tuple(self.pricing_variables))
        if not all(isinstance(v, Mapping) for v in self.pricing_variables):
            raise TypeError("pricing_variables must contain mappings")
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "networks", (self.networks,) if isinstance(self.networks, PricingNetwork) else tuple(self.networks))
        if not all(isinstance(r, SharedRow) for r in self.rows):
            raise TypeError("rows must contain SharedRow objects")
        if not all(isinstance(n, PricingNetwork) for n in self.networks):
            raise TypeError("networks must contain PricingNetwork objects")

    @classmethod
    def from_json(cls, text: str) -> NetworkProblem:
        data = json.loads(text)
        if isinstance(data, list):
            data = {"networks": data}
        elif "networks" not in data:
            data = {"networks": [data]}
        data.setdefault("rows", [])
        if isinstance(data["networks"], dict):
            data["networks"] = [data["networks"]]
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
        """The same model format accepted by ``baldes solve MODEL.json``."""
        data = asdict(self)
        if not data["pricing_variables"]:
            del data["pricing_variables"]
        for network in data["networks"]:
            for resource in network["resources"]:
                if resource["time_windows"] is None:
                    del resource["time_windows"]
        return json.dumps(data, allow_nan=False, indent=indent)

    def solve(self, **options: Any) -> NetworkResult:
        from .solve import solve
        return solve(self, **options)


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

    def with_row(self, row: SharedRow, coefficients: Tuple[NetworkRowCoefficients, ...], *, dynamic: bool = False) -> NetworkNode:
        data = {"row": asdict(row), "coefficients": [asdict(c) for c in coefficients], "dynamic": dynamic}
        return NetworkNode(self._native.with_row(json.dumps(data, allow_nan=False)))

    def with_resource_group_branch(
        self, packing_row: int, resource: int, threshold: float, upper: bool,
        members: Tuple[NetworkResourceGroupMember, ...],
    ) -> NetworkNode:
        data = dict(packing_row=packing_row, resource=resource, threshold=threshold,
                    upper=upper, members=[asdict(m) for m in members])
        return NetworkNode(self._native.with_resource_group_branch(json.dumps(data, allow_nan=False)))

    def with_synchronized_start(self, id: int, lower: int, upper: int) -> NetworkNode:
        """Fork with an inclusive common integer start domain for one service."""
        return NetworkNode(self._native.with_synchronized_start(id, lower, upper))

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
class MappedBranchEvaluation:
    name: str
    depth: int
    side: int
    stage: int
    kind: str
    parent: float
    estimate: float
    distance: float


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
    tree_processed_nodes: int = 0
    strong_branching_trials: int = 0
    mapped_history_ranked: int = 0
    mapped_branch_evaluations: Tuple[MappedBranchEvaluation, ...] = ()
    original_variables: Tuple[Mapping[str, Any], ...] = ()
    snapshot: Optional[NetworkNode] = field(default=None, repr=False, compare=False)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        data = {k: v for k, v in vars(self).items() if k != "snapshot"}
        data["routes"] = [asdict(r) for r in self.routes]
        data["fixed_routes"] = [asdict(r) for r in self.fixed_routes]
        data["enumeration_reports"] = [asdict(r) for r in self.enumeration_reports]
        data["mapped_branch_evaluations"] = [asdict(e) for e in self.mapped_branch_evaluations]
        return json.dumps(data, allow_nan=False, indent=indent)


def _result(text: str, native: Any) -> NetworkResult:
    data = json.loads(text)
    data.setdefault("tree_processed_nodes", 0)
    data["original_variables"] = tuple(data.get("original_variables", ()))
    data["routes"] = tuple(NetworkRoute(**r) for r in data["routes"])
    data["fixed_routes"] = tuple(NetworkRoute(**r) for r in data["fixed_routes"])
    data["enumeration_reports"] = tuple(NetworkEnumerationReport(**r) for r in data["enumeration_reports"])
    data["mapped_branch_evaluations"] = tuple(
        MappedBranchEvaluation(**e) for e in data.get("mapped_branch_evaluations", ())
    )
    data["duals"] = tuple(data["duals"])
    data["oracle_calls"] = tuple(tuple(c) for c in data["oracle_calls"])
    return NetworkResult(**data, snapshot=None if native is None else NetworkNode(native))


def _options(options: Mapping[str, Any]) -> str:
    options = dict(options)
    if "pricing_options" in options:
        options["pricing_options"] = {} if options["pricing_options"] is None else dict(options["pricing_options"])
    if "synchronized_services" in options:
        options["synchronized_services"] = [asdict(s) if isinstance(s, SynchronizedService) else s
                                            for s in options["synchronized_services"]]
    return json.dumps(options, allow_nan=False)
