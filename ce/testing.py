# %%
from collections import defaultdict
from typing import Any, Dict, Set, Tuple, List, Optional
from dataclasses import dataclass
import heapq
import gurobipy as gp
from gurobipy import GRB
import math
from copy import deepcopy
from gurobipy import Model, GRB, quicksum


import math
from typing import List

from reader import *
from auxiliar import *

import numpy as np
from typing import List, Tuple, Dict, Set

import sys

sys.path.append("../build")
import pybaldes as baldes


class ColumnElimination:
    def __init__(
        self,
        nodes: List[Tuple[int, int, int, int, float]],  # x,y,id,start,end,duration
        time_horizon: int,
        bucket_interval: int,
        vehicle_capacity: float,
        n_vehicles: int,
        instance: VRPTWInstance,
    ):
        """Initialize Column Elimination solver for VRPTW"""
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.time_horizon = time_horizon
        self.vehicle_capacity = vehicle_capacity
        self.n_vehicles = n_vehicles
        self.instance = instance

        # Initialize baldes components
        self.vrp_nodes = nodes

        self.graph = baldes.BucketGraph(self.vrp_nodes, time_horizon, bucket_interval)

        print("Initializing conflict tracking")
        self.graph.set_distance_matrix(instance.distances)

        options = baldes.BucketOptions()
        options.bucket_fixing = False
        self.graph.setOptions(options)
        print("Setup bucket graph")
        self.graph.setup()
        print("Setup complete")

        # Initialize conflict tracking
        self.conflicts: Set[Tuple[int, int]] = set()
        self.current_lb = float("-inf")

    def detect_conflict_patterns(self, conflict_path: List[int]):
        """Enhanced conflict pattern detection with more sophisticated patterns"""
        # Check for simple cycles first
        cycle = self._detect_cycle(conflict_path)
        if cycle:
            return {"type": "cycle", "nodes": cycle}

        # Check for capacity violations with subsequence identification
        capacity_violation = self._detect_capacity_violation(conflict_path)
        if capacity_violation:
            return {"type": "capacity", "nodes": capacity_violation}

        # Check time window violations with slack analysis
        time_violation = self._detect_time_window_violation(conflict_path)
        if time_violation:
            return {"type": "time_window", "nodes": time_violation}

        # Check for resource synchronization violations
        sync_violation = self._detect_sync_violation(conflict_path)
        if sync_violation:
            return {"type": "synchronization", "nodes": sync_violation}

        return None

    def _detect_cycle(self, path: List[int]) -> Optional[List[int]]:
        """Detect minimal cycles in the path"""
        node_positions = {}
        for i, node in enumerate(path):
            if node in node_positions and node != 0:  # Allow depot repeats
                return path[node_positions[node] : i + 1]
            node_positions[node] = i
        return None

    def _detect_capacity_violation(self, path: List[int]) -> Optional[List[int]]:
        """Find minimal subsequence causing capacity violation"""
        load = 0
        start_idx = 0

        for i, node in enumerate(path):
            load += self.nodes[node].demand

            # Try to minimize the violating sequence by removing nodes from start
            while (
                start_idx < i
                and load - self.nodes[path[start_idx]].demand > self.vehicle_capacity
            ):
                load -= self.nodes[path[start_idx]].demand
                start_idx += 1

            if load > self.vehicle_capacity:
                return path[start_idx : i + 1]

        return None

    def _detect_time_window_violation(self, path: List[int]) -> Optional[List[int]]:
        """Find minimal subsequence causing time window violation with slack analysis"""
        current_time = 0
        critical_sequence = []
        min_slack = float("inf")
        violation_start = 0

        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            if i > 0:
                current_time += self.nodes[current_node].duration

            travel_time = self.calculate_travel_time(current_node, next_node)
            arrival_time = current_time + travel_time

            # Calculate slack
            slack = self.nodes[next_node].ub[0] - arrival_time
            if slack < min_slack:
                min_slack = slack
                if slack < 0:
                    violation_start = i

            current_time = max(arrival_time, self.nodes[next_node].lb[0])

        if min_slack < 0:
            # Return minimal subsequence that leads to violation
            return path[violation_start : violation_start + 2]

        return None

    def _detect_sync_violation(self, path: List[int]) -> Optional[List[int]]:
        """Detect violations of resource synchronization constraints"""
        # This would be specific to problems with synchronization requirements
        # For basic VRPTW we return None
        return None

    def refine_conflict_by_pattern(self, conflict: Dict):
        """Enhanced conflict refinement based on violation patterns"""
        if conflict["type"] == "cycle":
            cycle_nodes = conflict["nodes"]
            # Remove all arcs that could recreate this cycle pattern
            for i in range(len(cycle_nodes)):
                for j in range(i + 1, len(cycle_nodes)):
                    self.graph.set_deleted_arcs([(cycle_nodes[i], cycle_nodes[j])])

        elif conflict["type"] == "capacity":
            violating_nodes = conflict["nodes"]
            total_load = sum(self.nodes[n].demand for n in violating_nodes)

            # Remove arcs that would create similar load patterns
            for i in range(len(violating_nodes)):
                accumulated_load = sum(
                    self.nodes[n].demand for n in violating_nodes[i:]
                )
                if accumulated_load > self.vehicle_capacity:
                    # Remove arcs that would lead to similar accumulation
                    for j in range(i + 1, len(violating_nodes)):
                        self.graph.set_deleted_arcs(
                            [(violating_nodes[i], violating_nodes[j])]
                        )

        elif conflict["type"] == "time_window":
            violating_sequence = conflict["nodes"]
            # Remove arcs that would create similar timing patterns
            arrival_time = self._calculate_arrival_time(violating_sequence)

            # Remove arcs with similar timing characteristics
            for i in range(len(violating_sequence) - 1):
                node1, node2 = violating_sequence[i], violating_sequence[i + 1]
                # Find similar node pairs that would cause violations
                for n1 in range(self.n_nodes):
                    for n2 in range(self.n_nodes):
                        if self._would_cause_similar_violation(n1, n2, arrival_time):
                            self.graph.set_deleted_arcs([(n1, n2)])

        elif conflict["type"] == "synchronization":
            # Handle synchronization-specific refinements
            pass

    def _calculate_arrival_time(self, sequence: List[int]) -> float:
        """Calculate arrival time at last node in sequence"""
        current_time = 0
        for i in range(len(sequence) - 1):
            if i > 0:
                current_time += self.nodes[sequence[i]].duration
            current_time += self.calculate_travel_time(sequence[i], sequence[i + 1])
            current_time = max(current_time, self.nodes[sequence[i + 1]].lb[0])
        return current_time

    def _would_cause_similar_violation(
        self, node1: int, node2: int, critical_time: float
    ) -> bool:
        """Check if arc would cause similar timing violation"""
        travel_time = self.calculate_travel_time(node1, node2)
        earliest_arrival = (
            self.nodes[node1].lb[0] + self.nodes[node1].duration + travel_time
        )
        return earliest_arrival > self.nodes[node2].ub[0]

    def refine_conflict(self, conflict_path: List[int]):
        """Enhanced conflict refinement with pattern detection"""
        conflict = self.detect_conflict_patterns(conflict_path)
        if conflict:
            self.refine_conflict_by_pattern(conflict)
        else:
            # Fallback to original refinement for other cases
            s_curr = 0
            for j in range(1, len(conflict_path)):
                next_node = conflict_path[j]
                valid_transitions = self.find_valid_transitions_to_state(
                    conflict_path[:j]
                )
                if not any(
                    self.is_subsequence_feasible(trans + [next_node], next_node)
                    for trans in valid_transitions
                ):
                    self.graph.set_deleted_arcs([(conflict_path[j - 1], next_node)])
                    return
                s_curr = next_node

    def is_subsequence_feasible(self, path: List[int], next_node: int) -> bool:
        """
        Check if the path is feasible in the original problem.
        """
        # Check capacity constraints
        current_load = 0
        for i, node_id in enumerate(path):
            current_load += self.nodes[node_id].demand
            if current_load > self.vehicle_capacity:
                return False

        # Check time window constraints
        current_time = 0
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            # Add service time at current node
            if i > 0:
                current_time += self.nodes[current_node].duration

            # Add travel time to next node
            travel_time = self.calculate_travel_time(current_node, next_node)
            current_time += travel_time

            # Check if we arrive before the end of time window
            if current_time > self.nodes[next_node].ub[0]:
                return False

            # Update current_time to start of service at next node
            current_time = max(current_time, self.nodes[next_node].lb[0])

        # Check if we can return to depot within its time window
        if path[-1] != 0:
            final_travel = self.calculate_travel_time(path[-1], 0)
            final_time = current_time + self.nodes[path[-1]].duration + final_travel
            if final_time > self.nodes[0].ub[0]:
                return False

        return True

    def find_valid_transitions_to_state(self, prefix: List[int]) -> List[List[int]]:
        """
        Find all valid transitions from root to the current state that could be
        part of a feasible solution. This implements the S^- set from Algorithm 1.
        """
        valid_transitions = []

        def is_valid_transition(trans):
            # Check capacity
            current_load = 0
            current_time = 0

            for i in range(len(trans) - 1):
                node = trans[i]
                next_node = trans[i + 1]

                current_load += self.nodes[node].demand
                if current_load > self.vehicle_capacity:
                    return False

                travel_time = self.calculate_travel_time(node, next_node)
                service_time = self.nodes[node].duration
                current_time = max(
                    current_time + travel_time + service_time,
                    self.nodes[next_node].lb[0],
                )

                if current_time > self.nodes[next_node].ub[0]:
                    return False

            return True

        # Start with the prefix and try to find other valid paths
        if is_valid_transition(prefix):
            valid_transitions.append(prefix)

        return valid_transitions

    def calculate_travel_time(self, node1: int, node2: int) -> float:
        if node1 == 101:
            node1 = 0
        if node2 == 101:
            node2 = 0
        return self.instance.distances[node1][node2]

    def find_conflict(self, path: List[int]) -> Optional[List[int]]:
        """
        Check if path is feasible in the original problem.
        If not, return the infeasible subsequence.
        Returns None if path is feasible.
        """

        # Check multiple visits
        visits = defaultdict(int)
        for node in path:
            visits[node] += 1
            if visits[node] > 1:
                return path[: path.index(node) + 1]

        # Check capacity constraints
        current_load = 0
        for i, node_id in enumerate(path):
            current_load += self.nodes[node_id].demand
            if current_load > self.vehicle_capacity:
                print(
                    f"Capacity violation at node {node_id}: {current_load} > {self.vehicle_capacity}"
                )
                return path[: i + 1]  # Return subsequence up to violation

        # Check time window constraints
        current_time = 0
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            # Add service time at current node
            if i > 0:  # Skip service time for depot at start
                current_time += self.nodes[current_node].duration

            # Add travel time to next node
            travel_time = self.calculate_travel_time(current_node, next_node)
            current_time += travel_time

            # Check if we arrive before the end of time window
            if current_time > self.nodes[next_node].ub[0]:
                print(f"Time window violation at node {next_node}")
                print(
                    f"Arrival time: {current_time}, Latest allowed: {self.nodes[next_node].ub[0]}"
                )
                return path[: i + 2]  # Return subsequence including violating node

            # Update current_time to start of service at next node
            current_time = max(current_time, self.nodes[next_node].lb[0])

        # Check if we can return to depot within its time window
        if path[-1] != 0:  # If we're not already at depot
            final_travel = self.calculate_travel_time(path[-1], 0)
            final_time = current_time + self.nodes[path[-1]].duration + final_travel
            if final_time > self.nodes[0].ub[0]:
                print(f"Cannot return to depot within time window")
                print(f"Return time: {final_time}, Depot closes: {self.nodes[0].ub[0]}")
                return path  # The whole path is infeasible

        return None

    def _is_feasible_solution(self, paths: List[List[int]]) -> bool:
        """
        Check if paths form a feasible solution:
        - Each customer visited exactly once
        - Paths respect capacity and time windows
        - Number of paths <= n_vehicles
        """
        if not paths or len(paths) > self.n_vehicles:
            return False

        # Check each path is individually feasible
        for path in paths:
            if self.find_conflict(path) is not None:
                return False

        # Check customer coverage
        visits = defaultdict(int)
        for path in paths:
            for node in path[1:-1]:  # Skip depots
                visits[node] += 1

        # Each customer should be visited exactly once
        for node in range(1, self.n_nodes - 1):
            if visits[node] != 1:
                return False

        return True

    def extract_solution(self, x, arc_set):
        """
        Extracts the arcs used in the solution from the Gurobi model.
        """
        selected_arcs = []
        for arc in arc_set:
            if x[arc].x > 0.5:  # Check if the arc is used in the solution
                selected_arcs.append(arc)
        return selected_arcs

    def decompose_paths(self, path_sequences):
        """
        Extract arcs from the given path sequences for the Gurobi model.
        """
        arcs_to_include = set()
        for path in path_sequences:
            for i in range(len(path) - 1):
                arc = (path[i], path[i + 1])
                arcs_to_include.add(arc)
        return arcs_to_include

    def calculate_lagrangian(self, paths, duals):
        """
        Calculate the objective value of the Lagrangian relaxation.
        """
        current_value = 0
        visits = defaultdict(int)

        for path in paths:
            path = path.nodes_covered
            path_cost = sum(
                self.calculate_travel_time(path[i], path[i + 1])
                for i in range(len(path) - 1)
            )
            for node in path[1:-1]:
                visits[node] += 1
                path_cost -= duals[node - 1]

            current_value += path_cost

        return current_value, visits

    def repair_duals_greedy(self, duals):
        """
        Repair infeasible dual values using a greedy method.
        """
        for j in range(1, len(duals) - 1):
            # If dual value is negative or violates a constraint, adjust it greedily
            if duals[j] < 0:
                duals[j] = 0

    def fix_arcs(
        self,
        best_ub: float,
        duals: List[float],
        node_demands: List[float],
        time_windows: List[Tuple[float, float]],
        distances: List[List[float]],
        current_paths: List[List[int]],
    ) -> Set[Tuple[int, int]]:
        """
        Conservative arc fixing for VRPTW using stability checks, critical arc detection,
        and path-based analysis to avoid fixing essential arcs.
        """
        fixed_arcs = set()

        # Calculate path-based statistics and node potentials
        path_stats = {
            "avg_cost": 0,
            "avg_load": 0,
            "avg_service_time": 0,
            "min_service_time": float("inf"),
            "avg_slack": 0,
        }
        total_paths = len(current_paths)
        node_potentials = [0.0] * self.n_nodes

        if total_paths > 0:
            # Calculate path statistics and node potentials
            for path in current_paths:
                for i in range(len(path) - 1):
                    node1, node2 = path[i], path[i + 1]
                    path_stats["avg_cost"] += distances[node1][node2]
                    path_stats["avg_load"] += self.nodes[node1].demand
                    service_time = self.nodes[node1].duration
                    path_stats["avg_service_time"] += service_time
                    path_stats["min_service_time"] = min(
                        path_stats["min_service_time"], service_time
                    )

                    # Calculate node potentials based on path position
                    position_weight = 1.0 - (abs(i - len(path) / 2) / (len(path) / 2))
                    node_potentials[node1] += distances[node1][node2] * position_weight

            path_stats["avg_cost"] /= total_paths
            path_stats["avg_load"] /= total_paths
            path_stats["avg_service_time"] /= total_paths

        # Initialize historical reduced costs if not already present
        if not hasattr(self, "arc_reduced_costs"):
            self.arc_reduced_costs = defaultdict(list)

        # Parameters for arc fixing
        arc_fix_threshold = 0.9
        min_fix_iterations = 5
        max_arcs_to_fix = max(1, int(0.05 * len(distances) * len(distances[0])))

        arc_candidates = []

        for i in range(1, self.n_nodes - 1):
            for j in range(1, self.n_nodes - 1):
                if i != j:
                    # Step 1: Calculate basic reduced cost
                    reduced_cost = distances[i][j] - duals[i - 1] - duals[j - 1]

                    # Enhanced reduced cost calculation
                    early_i, late_i = time_windows[i]
                    early_j, late_j = time_windows[j]
                    travel_time = distances[i][j]
                    service_time = self.nodes[i].duration

                    # Add capacity-based penalty
                    remaining_capacity = (
                        self.vehicle_capacity - node_demands[i] - node_demands[j]
                    )
                    if remaining_capacity < 0:
                        reduced_cost += float("inf")
                    else:
                        reduced_cost += (
                            max(
                                0,
                                (node_demands[i] + node_demands[j]) / remaining_capacity
                                - 1,
                            )
                            * path_stats["avg_load"]
                        )

                    # Add time window compatibility penalty
                    if early_i + service_time + travel_time > late_j:
                        reduced_cost += float("inf")
                    else:
                        time_window_slack = late_j - (
                            early_i + service_time + travel_time
                        )
                        if time_window_slack < 0:
                            reduced_cost += float("inf")
                        else:
                            reduced_cost += (
                                max(
                                    0,
                                    path_stats["avg_slack"] / (time_window_slack + 1e-6)
                                    - 1,
                                )
                                * path_stats["avg_cost"]
                            )

                    # Add node potential penalty
                    reduced_cost += (
                        abs(node_potentials[i] - node_potentials[j])
                        * path_stats["avg_cost"]
                    )

                    # Track historical reduced costs
                    self.arc_reduced_costs[(i, j)].append(reduced_cost)

                    # Use moving average of reduced costs over recent iterations
                    if len(self.arc_reduced_costs[(i, j)]) > min_fix_iterations:
                        avg_rc = (
                            sum(self.arc_reduced_costs[(i, j)][-min_fix_iterations:])
                            / min_fix_iterations
                        )
                    else:
                        avg_rc = reduced_cost

                    # Check if the arc is critical based on paths and time windows
                    is_critical_arc = False
                    for path in current_paths:
                        if (i, j) in zip(path, path[1:]):
                            is_critical_arc = True
                            break
                    if early_i + service_time > late_i or travel_time > late_j:
                        is_critical_arc = True

                    # Add to candidates if it's not critical and has a high average reduced cost
                    if not is_critical_arc and avg_rc >= best_ub * arc_fix_threshold:
                        arc_candidates.append((i, j, avg_rc))

        # Sort arcs by reduced cost in descending order and fix a limited number
        arc_candidates.sort(key=lambda x: x[2], reverse=True)
        for arc in arc_candidates[:max_arcs_to_fix]:
            fixed_arcs.add((arc[0], arc[1]))

        return fixed_arcs

    def calculate_solution_cost(self, paths: List[List[int]]) -> float:
        cost = 0.0
        for path in paths:
            for i in range(len(path) - 1):
                cost += self.calculate_travel_time(path[i], path[i + 1])
        return cost

    def solve_with_subgradient(self, max_iterations: int = 1000):
        """Enhanced subgradient method with adaptive parameters and stabilization"""
        duals = [0.0] * self.n_nodes
        best_lb = float("-inf")
        best_ub = float("inf")
        alpha = 2.0  # Initial step size multiplier
        step_reduction = 0.95  # Step size reduction factor
        non_improving_count = 0
        stabilization_center = duals.copy()
        stability_weight = 1.0

        # Initialize duals using a better heuristic
        for i in range(1, self.n_nodes - 1):
            duals[i - 1] = (
                2 * self.calculate_travel_time(0, i) * (self.nodes[i].demand / 100)
            )

        for iteration in range(max_iterations):
            # Add proximal term to objective for stabilization

            self.graph.set_duals(duals)

            # Solve subproblem using minimum update SSP
            paths = self.graph.phaseFour()
            path_sequences = [
                p.nodes_covered for p in paths if len(p.nodes_covered) > 2
            ]
            print(f"Paths found: {path_sequences}")

            if not path_sequences:
                break

                # Check for conflicts and refine if necessary
            conflict_found = False
            for path in path_sequences:
                conflict = self.find_conflict(path)
                if conflict:
                    print(f"Found conflict in path: {path}")
                    self.refine_conflict(conflict)
                    conflict_found = True
                    break

            # if conflict_found:
            #    print("Conflict found. Refining and continuing.")
            #    continue

            # Calculate current value and subgradient
            current_value, visits = self.calculate_lagrangian(paths, duals)
            subgradient = [1 - visits[j] for j in range(1, self.n_nodes - 1)]
            print(
                f"Iteration {iteration}: Value: {current_value}, Subgradient: {subgradient}"
            )

            # Update bounds
            best_lb = max(best_lb, current_value)

            # Estimate psi* with dynamic adjustment
            if best_ub < float("inf"):
                psi_star = best_lb * (1 + max(0.5, 5 / (100 + iteration)))
            else:
                psi_star = current_value * 1.1

            # Calculate step size with normalization
            subgradient_norm = sum(g * g for g in subgradient)
            if subgradient_norm > 0:
                step_size = alpha * (psi_star - current_value) / subgradient_norm
            else:
                step_size = 0

            # Update duals with projection
            new_duals = []
            for j in range(1, self.n_nodes - 1):
                new_val = max(0, duals[j - 1] - step_size * subgradient[j - 1])
                new_duals.append(new_val)

            # Update stabilization center if improving
            if current_value > best_lb - 0.01 * abs(best_lb):
                stabilization_center = new_duals.copy()
                non_improving_count = 0
            else:
                non_improving_count += 1

            # Adaptive parameter updates
            if non_improving_count > 20:
                alpha *= step_reduction
                stability_weight *= 1.1
                non_improving_count = 0

            duals = new_duals

            # calculate path cost
            path_cost = self.calculate_solution_cost(path_sequences)
            print(f"Path cost: {path_cost}")

            # Perform arc fixing using the current duals
            fixed_arcs = self.fix_arcs(
                best_ub,
                duals,
                [node.demand for node in self.nodes],
                [(node.lb[0], node.ub[0]) for node in self.nodes],
                self.instance.distances,
                path_sequences,
            )
            # print(f"Fixed arcs: {fixed_arcs}")
            self.graph.set_deleted_arcs(list(fixed_arcs))

            # Early stopping condition
            if (
                alpha < 0.01
                or iteration > 50
                and abs(best_ub - best_lb) < 0.001 * abs(best_lb)
            ):
                break

        return best_lb, path_sequences


# %%
# Create instance data
solomon_instance = InstanceData()
solomon_instance.read_instance(
    "../build/instances/C203.txt"
)  # Set mtw=True if using multiple time windows
instance = convert_instance_data(solomon_instance)

nodes = []
for i in range(instance.num_nodes):
    node = baldes.VRPNode()
    node.id = i
    node.demand = instance.demands[i]
    node.start_time = instance.time_windows[i][0]
    node.end_time = instance.time_windows[i][1]
    node.duration = instance.service_times[i]
    node.lb = [instance.time_windows[i][0]]
    node.ub = [instance.time_windows[i][1]]
    node.consumption = [instance.service_times[i]]
    nodes.append(node)

# Initialize solver
solver = ColumnElimination(
    nodes=nodes,
    time_horizon=int(instance.get_time_horizon()),
    bucket_interval=10,
    vehicle_capacity=700,
    n_vehicles=3,
    instance=instance,
)

# Solve
best_cost, best_routes = solver.solve_with_subgradient()
