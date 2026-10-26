"""One solve entry point; scalar pricing is a collection of length one."""
from __future__ import annotations
import json
import math
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union
from . import _core
from .routing import Instance, RootResult, RootRoute, Iteration, FeasibilityReport
from .networks import NetworkProblem, PricingNetwork, NetworkResult, SharedRow, SynchronizedService, _result, _options


def solve(problem, *, rows=(), time_limit=120.0, max_iterations=100000,
          stabilize=True, stabilization_max_tree_depth=10000,
          bound_mode="numerical", pricing_options=None, **options) -> Union[RootResult, NetworkResult]:
    """Solve routing data or a shared master with one or more pricing networks.

    A PricingNetwork and [PricingNetwork] normalize to the same NetworkProblem.
    Pass shared rows through rows=, or supply a complete NetworkProblem. Paths
    ending in .json contain pricing models; other paths contain routing inputs.
    Model-specific controls are checked by the corresponding model adapter.
    """
    if not _core.has_clp:
        raise RuntimeError("solving requires a build with Clp")
    if isinstance(problem, (str, PathLike)):
        problem = NetworkProblem.from_file(problem) if Path(problem).suffix == ".json" else Instance.from_file(problem)
    if isinstance(problem, PricingNetwork):
        problem = NetworkProblem(tuple(rows), (problem,))
    elif isinstance(problem, (list, tuple)):
        problem = NetworkProblem(tuple(rows), tuple(problem))
    elif rows:
        raise ValueError("rows= belongs to scalar or array pricing inputs; complete models already own their rows")
    options.update(time_limit=time_limit, max_iterations=max_iterations,
                   stabilize=stabilize, stabilization_max_tree_depth=stabilization_max_tree_depth,
                   bound_mode=bound_mode, pricing_options=dict(pricing_options or {}))
    if isinstance(problem, Instance):
        return _routing(problem, **options)
    if isinstance(problem, NetworkProblem):
        return _networks(problem, **options)
    raise TypeError("problem must be an Instance, NetworkProblem, PricingNetwork, or array of PricingNetwork objects")

def _routing(
    instance: Union[Instance, str, PathLike],
    *,
    problem_type: Optional[str] = None,
    customers: Optional[int] = None,
    ng: int = 8,
    reference_ng: bool = False,
    rounding: Optional[str] = None,
    capacity_cuts: bool = False,
    two_path_cuts: bool = False,
    rank_one_cuts: bool = False,
    stabilize: bool = True,
    stabilization_max_tree_depth: int = 10000,
    master_solver: str = "clp",
    master_threads: int = 0,
    bound_mode: str = "numerical",
    clp_scaling: str = "equilibrium",
    clp_keep_work_areas: bool = True,
    verbose: bool = False,
    log_interval: int = 10,
    hgs_time: Optional[float] = None,
    hgs_columns: Union[str, int] = "best",
    hgs_seed: int = 0,
    reduced_cost_fixing: bool = True,
    upper_bound: float = math.inf,
    time_limit: float = 120.0,
    max_iterations: int = 10_000,
    max_cut_rounds: Optional[int] = None,
    pricing_options: Optional[Mapping[str, float]] = None,
) -> RootResult:
    """Solve a VRPTW/CVRP root relaxation directly in Rust, releasing the GIL.

    Input paths use Solomon or CVRPLIB format. CVRP ignores time windows and
    service times. Distances follow the instance's convention by default
    (unrounded for Solomon, TSPLIB nint for CVRPLIB EUC_2D); rounding accepts
    'none', 'nearest', 'tenth' and 'ceil'.
    verbose=True prints a CG progress table to stderr every log_interval iterations.
    clp_scaling selects equilibrium (default), auto, geometric, or off for Clp.
    clp_keep_work_areas retains native work buffers between LP solves by default.
    master_solver='highs' selects HiGHS (requires a build with its feature);
    master_solver='gurobi' selects Gurobi (requires its feature and a license);
    master_threads limits HiGHS/Gurobi workers; 0 uses all available CPUs.
    master_solver='ipm' uses native IPM and disables stabilization. hgs_time
    gives HGS a separate positive search budget before column generation;
    hgs_columns='best' seeds the best feasible plan, 'all' seeds all distinct
    encountered routes that pass target-graph feasibility checks. A positive integer
    selects up to that many cheapest new routes by original route cost.
    Capacity cuts use complement form; rank-one cuts use limited memory.
    two_path_cuts attaches the time resource for Kohl separation and enables
    the capacity family. It requires problem_type='vrptw'.
    Disabling reduced_cost_fixing also disables route enumeration. Cut counts
    include generated rows that were later retired. These are root relaxations:
    status 'Optimal' does not mean an optimal integer routing solution.
    bound_mode='numerical' uses floating-point pricing bounds by default.
    'certified' requires independent bound validation; an unknown certificate
    leaves bound/certified_bound at negative infinity. numerical_bound is
    reported separately and never substitutes for a missing certificate.
    """
    if not isinstance(instance, Instance):
        instance = Instance.from_file(instance)
    if customers is not None:
        instance = instance.truncated(customers)
    if problem_type is None:
        problem_type = "vrptw" if all(math.isfinite(c.due) for c in instance.customers) else "cvrp"
    rows = [(c.x, c.y, c.demand, c.ready, c.due, c.service) for c in instance.customers]
    distances = None if instance.distances is None else [list(row) for row in instance.distances]
    data = _core._solve({"kind": "routing", "data": {'name': instance.name, 'vehicles': instance.vehicles, 'capacity': instance.capacity, 'customers': rows, 'distances': distances}}, {'problem_type': problem_type, 'ng': ng, 'reference_ng': reference_ng, 'rounding': rounding or instance.rounding, 'capacity_cuts': capacity_cuts, 'rank_one_cuts': rank_one_cuts, 'stabilize': stabilize, 'enumerate': reduced_cost_fixing, 'upper_bound': upper_bound, 'time_limit': time_limit, 'max_iterations': max_iterations, 'max_cut_rounds': max_cut_rounds, 'pricing_options': dict(pricing_options or {}), 'two_path_cuts': two_path_cuts, 'master_solver': master_solver, 'hgs_time': hgs_time, 'hgs_columns': str(hgs_columns), 'hgs_seed': hgs_seed, 'verbose': verbose, 'log_interval': log_interval, 'bound_mode': bound_mode, 'clp_scaling': clp_scaling, 'clp_keep_work_areas': clp_keep_work_areas, 'master_threads': master_threads, 'stabilization_max_tree_depth': stabilization_max_tree_depth})
    data["duals"] = tuple(data["duals"])
    data["routes"] = tuple(
        RootRoute(**{**r, "nodes": tuple(r["nodes"]), "arcs": tuple(r["arcs"])})
        for r in data["routes"]
    )
    data["history"] = tuple(Iteration(**h) for h in data["history"])
    data["feasibility"] = tuple(FeasibilityReport(**r) for r in data["feasibility"])
    return RootResult(**data)


def _networks(
    problem: NetworkProblem,
    *,
    tree: bool = False,
    network_pricing_schedule: str = "full-sweep",
    network_unpromising_limit: int = 1,
    network_promising_limit: int = 1000000,
    network_branching: str = "multiplicity-first",
    network_tree_search: str = "default",
    network_strategy2_rounding: str = "unrounded",
    network_open_node_limit: int = 1000,
    network_scoring_gap_tolerance: float = 1e-6,
    network_share_inactive_columns: bool = False,
    network_search_inactive_columns: bool = False,
    network_branch_priorities: Optional[Mapping[str, float]] = None,
    network_branch_directions: Optional[Mapping[str, str]] = None,
    mapped_variables: Tuple[Mapping[str, Any], ...] = (),
    packing: Optional[Mapping[str, Any]] = None,
    variable_boxes: Tuple[Mapping[str, Any], ...] = (),
    original_variable_boxes: Tuple[Mapping[str, Any], ...] = (),
    mapped_strong_branching: bool = True,
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
    stabilization_max_tree_depth: int = 10000,
    proximal: Optional[Mapping[str, Any]] = None,
    bound_mode: str = "numerical",
    artificial_cost: float = 2.76,
    log_interval: Optional[int] = None,
    pricing_options: Optional[Mapping[str, float]] = None,
    synchronized_services: Tuple[SynchronizedService, ...] = (),
) -> NetworkResult:
    """Solve a shared LP or assignment tree. Unknown bounds serialize as null.

    Roots retain thread-bound snapshots; these calls keep the GIL. Fresh
    assignment-tree solves release the GIL and return an owned integer witness.
    A tree may return Incomplete when another branching family is required.
    """
    if not isinstance(problem, NetworkProblem):
        raise TypeError("problem must be a NetworkProblem")
    options = dict(synchronized_services=synchronized_services, tree=tree, network_pricing_schedule=network_pricing_schedule, network_unpromising_limit=network_unpromising_limit, network_promising_limit=network_promising_limit, stabilization_max_tree_depth=stabilization_max_tree_depth, proximal=proximal, network_branching=network_branching, network_tree_search=network_tree_search, network_strategy2_bound_rounding=network_strategy2_rounding, network_open_node_limit=network_open_node_limit, network_scoring_gap_tolerance=network_scoring_gap_tolerance, network_share_inactive_columns=network_share_inactive_columns, network_search_inactive_columns=network_search_inactive_columns, network_branch_priorities=dict(network_branch_priorities or {}), network_branch_directions=dict(network_branch_directions or {}), mapped_variables=mapped_variables, packing=packing, variable_boxes=variable_boxes, original_variable_boxes=original_variable_boxes, mapped_strong_branching=mapped_strong_branching, enumerate=enumerate, enumeration_route_limit=enumeration_route_limit,
                   enumeration_work_limit=enumeration_work_limit, enumeration_stored_arc_limit=enumeration_stored_arc_limit,
                   enumeration_time_limit=enumeration_time_limit, enumerated_mip=enumerated_mip,
                   mip_time_limit=mip_time_limit, mip_node_limit=mip_node_limit,
                   mip_objective_cutoff=mip_objective_cutoff,
                   time_limit=time_limit, max_iterations=max_iterations,
                   node_limit=node_limit, stabilize=stabilize, bound_mode=bound_mode,
                   artificial_cost=artificial_cost, log_interval=log_interval,
                   pricing_options={} if pricing_options is None else dict(pricing_options))
    return _result(*_core._solve({"kind": "networks", "data": json.loads(problem.to_json(indent=None))}, json.loads(_options(options))))
