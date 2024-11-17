# %%
from collections import defaultdict
from typing import Any, Dict, Set, Tuple, List, Optional
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB
from gurobipy import Model, GRB, quicksum


from reader import *
from auxiliar import *

import numpy as np

import sys
import random

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
        self.column_pool = ColumnPool()

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
        """
        Enhanced conflict pattern detection that identifies root causes and minimal subsequences.
        """
        conflict_info = {
            "type": None,
            "subsequence": None,
            "root_cause": None,
            "resources": None,
        }

        # 1. Check for minimal capacity violations
        current_load = 0
        for i, node in enumerate(conflict_path):
            current_load += self.nodes[node].demand
            if current_load > self.vehicle_capacity:
                # Find minimal subsequence causing violation
                min_seq_start = i
                min_load = self.nodes[node].demand
                while (
                    min_seq_start > 0
                    and current_load - min_load > self.vehicle_capacity
                ):
                    min_seq_start -= 1
                    min_load += self.nodes[conflict_path[min_seq_start]].demand

                conflict_info.update(
                    {
                        "type": "capacity",
                        "subsequence": conflict_path[min_seq_start : i + 1],
                        "root_cause": f"Capacity exceeded: {current_load} > {self.vehicle_capacity}",
                        "resources": {"load": current_load},
                    }
                )
                return conflict_info

        # 2. Check for minimal time window violations
        current_time = 0
        min_slack = float("inf")
        critical_node = None

        for i in range(len(conflict_path) - 1):
            current_node = conflict_path[i]
            next_node = conflict_path[i + 1]

            if i > 0:
                current_time += self.nodes[current_node].duration

            travel_time = self.calculate_travel_time(current_node, next_node)
            arrival_time = current_time + travel_time

            # Track minimum slack
            slack = self.nodes[next_node].ub[0] - arrival_time
            if slack < min_slack:
                min_slack = slack
                critical_node = next_node

            if arrival_time > self.nodes[next_node].ub[0]:
                conflict_info.update(
                    {
                        "type": "time_window",
                        "subsequence": [current_node, next_node],
                        "root_cause": (
                            f"Time window violation at node {next_node}: "
                            f"Arrival {arrival_time} > Latest {self.nodes[next_node].ub[0]}"
                        ),
                        "resources": {
                            "arrival_time": arrival_time,
                            "latest_time": self.nodes[next_node].ub[0],
                            "slack": min_slack,
                        },
                    }
                )
                return conflict_info

            current_time = max(arrival_time, self.nodes[next_node].lb[0])

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

    def is_path_feasible(self, path: List[int]) -> bool:
        """Check if a path is feasible in the original problem"""
        current_load = 0
        for i, node_id in enumerate(path):
            current_load += self.nodes[node_id].demand
            if current_load > self.vehicle_capacity:
                return False

        current_time = 0
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            if i > 0:
                current_time += self.nodes[current_node].duration

            travel_time = self.calculate_travel_time(current_node, next_node)
            current_time += travel_time

            if current_time > self.nodes[next_node].ub[0]:
                return False

            current_time = max(current_time, self.nodes[next_node].lb[0])

        if path[-1] != 0:
            final_travel = self.calculate_travel_time(path[-1], 0)
            final_time = current_time + self.nodes[path[-1]].duration + final_travel
            if final_time > self.nodes[0].ub[0]:
                return False

        return True

    def refine_conflict(self, conflict_path: List[int]):
        """
        Enhanced conflict refinement that better handles different types of conflicts
        and their root causes.
        """
        # First try to detect specific conflict patterns
        conflict_info = self.detect_conflict_patterns(conflict_path)
        if conflict_info:
            print(f"Detected conflict: {conflict_info}")
            self.refine_by_conflict_type(conflict_info)
            return

        # Fallback to general refinement algorithm
        for j in range(1, len(conflict_path)):
            prefix = conflict_path[:j]
            next_node = conflict_path[j]

            print(f"Analyzing prefix {prefix} → {next_node}")

            # Find all valid transitions to current state
            valid_transitions = self.find_valid_transitions_to_state(prefix)

            if not valid_transitions:
                print(f"No valid transitions found for prefix {prefix}")
                self.graph.set_deleted_arcs([(prefix[-1], next_node)])
                return

            # Check if any valid transition can be feasibly extended
            is_extension_feasible = False
            for trans in valid_transitions:
                extended = trans + [next_node]
                if self.is_subsequence_feasible(extended, next_node):
                    is_extension_feasible = True
                    break

            if not is_extension_feasible:
                print(f"No feasible extension found from {prefix[-1]} to {next_node}")
                print(f"Valid transitions were: {valid_transitions}")
                self.graph.set_deleted_arcs([(prefix[-1], next_node)])
                return

    def refine_by_conflict_type(self, conflict_info: dict):
        """
        Apply specific refinement strategies based on the type of conflict.
        """
        if conflict_info["type"] == "capacity":
            self._refine_capacity_conflict(conflict_info)
        elif conflict_info["type"] == "time_window":
            self._refine_time_window_conflict(conflict_info)
        # Add other conflict types as needed

    def _refine_capacity_conflict(self, conflict_info: dict):
        """
        Refined strategy for capacity conflicts - remove arcs that would
        lead to similar violations.
        """
        seq = conflict_info["subsequence"]
        total_load = sum(self.nodes[node].demand for node in seq)

        # Remove arcs that would lead to similar overloads
        arcs_to_remove = []
        for i in range(len(seq) - 1):
            accumulated_load = sum(self.nodes[node].demand for node in seq[i:])
            if accumulated_load > self.vehicle_capacity:
                arcs_to_remove.append((seq[i], seq[i + 1]))

        if arcs_to_remove:
            print(
                f"Removing arcs that would cause capacity violations: {arcs_to_remove}"
            )
            self.graph.set_deleted_arcs(arcs_to_remove)

    def _refine_time_window_conflict(self, conflict_info: dict):
        """
        Refined strategy for time window conflicts - remove arcs that would
        lead to similar timing violations.
        """
        seq = conflict_info["subsequence"]
        arrival_time = conflict_info["resources"]["arrival_time"]
        latest_time = conflict_info["resources"]["latest_time"]

        # Remove similar problematic arcs
        arcs_to_remove = []
        for i in range(len(seq) - 1):
            current_node, next_node = seq[i], seq[i + 1]
            travel_time = self.calculate_travel_time(current_node, next_node)
            service_time = self.nodes[current_node].duration

            # If minimal time to reach next node exceeds its time window
            if (
                self.nodes[current_node].lb[0] + service_time + travel_time
                > self.nodes[next_node].ub[0]
            ):
                arcs_to_remove.append((current_node, next_node))

        if arcs_to_remove:
            print(
                f"Removing arcs that would cause time window violations: {arcs_to_remove}"
            )
            self.graph.set_deleted_arcs(arcs_to_remove)

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
        Enhanced implementation of S^- set from Algorithm 1. This method finds all valid
        transitions that could lead to the current state, considering problem constraints.
        """
        valid_transitions = []

        def explore_transitions(current_path, current_load, current_time):
            """
            Recursively explore valid transitions using the bucket graph.
            """
            print(f"Exploring transitions from path: {current_path}")

            # Package current state information
            resources = [current_time]  # Current implementation uses time as resource

            # Get extensions from bucket graph
            new_paths = self.graph.extend_path(current_path, resources)
            if not new_paths:
                print(f"No extensions found from {current_path}")
                return

            # Process and validate each extension
            for path in new_paths:
                path_nodes = path.nodes_covered
                # print(f"Validating potential path: {path_nodes}")

                # Validate complete path
                valid, diagnostics = self.validate_path_with_diagnostics(path_nodes)

                if valid:
                    # print(f"Found valid transition: {path_nodes}")
                    valid_transitions.append(path_nodes)
                    # Note: recursive exploration handled by bucket graph
                else:
                    print(f"Invalid path: {path_nodes}, Reason: {diagnostics}")

        # Initialize from prefix if provided
        if prefix:
            initial_state = self.calculate_initial_state(prefix)
            print(
                f"Starting exploration from prefix {prefix} with state {initial_state}"
            )

            # Add prefix itself if valid
            if self.is_path_feasible(prefix):
                valid_transitions.append(prefix)

            # Explore further transitions
            explore_transitions(prefix, initial_state["load"], initial_state["time"])

        return valid_transitions

    def calculate_initial_state(self, path: List[int]) -> dict:
        """Calculate initial state values for a given path."""
        initial_load = sum(self.nodes[node].demand for node in path)
        initial_time = 0

        # Calculate cumulative time including travel and service
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            if i > 0:
                initial_time += self.nodes[current_node].duration
            travel_time = self.calculate_travel_time(current_node, next_node)
            initial_time += travel_time
            initial_time = max(initial_time, self.nodes[next_node].lb[0])

        return {"load": initial_load, "time": initial_time}

    def validate_path_with_diagnostics(self, path: List[int]) -> Tuple[bool, str]:
        """
        Validate a path and return detailed diagnostics about any violations.
        """
        # Check load constraints
        current_load = sum(self.nodes[node].demand for node in path)
        if current_load > self.vehicle_capacity:
            return False, f"Capacity exceeded: {current_load} > {self.vehicle_capacity}"

        # Check time windows
        current_time = 0
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            if i > 0:
                current_time += self.nodes[current_node].duration

            travel_time = self.calculate_travel_time(current_node, next_node)
            arrival_time = current_time + travel_time

            if arrival_time > self.nodes[next_node].ub[0]:
                return False, (
                    f"Time window violated at node {next_node}: "
                    f"Arrival {arrival_time} > Latest {self.nodes[next_node].ub[0]}"
                )

            current_time = max(arrival_time, self.nodes[next_node].lb[0])

        return True, "Path is valid"

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
        upper_bound: float,
        duals: List[float],
        node_demands: List[float],
        time_windows: List[Tuple[float, float]],
        distances: List[List[float]],
        current_paths: List[List[int]],
    ) -> Set[Tuple[int, int]]:
        """
        Arc fixing based on the paper's approach using:
        - Shortest path calculations with reduced costs
        - Dual values for computing reduced costs
        - Upper and lower bounds for fixing criterion
        """
        fixed_arcs = set()

        # For each arc (v1, v2), calculate:
        # sp↓v1 + rc(v1,v2) + sp↑v2
        # where sp↓v1 is shortest path from root to v1
        # and sp↑v2 is shortest path from v2 to terminal

        # Calculate shortest paths from root to all nodes using reduced costs
        sp_down = self.calculate_shortest_paths_from_root(duals, distances)

        # Calculate shortest paths from all nodes to terminal using reduced costs
        sp_up = self.calculate_shortest_paths_to_terminal(duals, distances)

        # Get current lower bound from dual solution
        v_lambda = sum(duals[i - 1] for i in range(1, self.n_nodes - 1))

        for i in range(1, self.n_nodes - 1):
            for j in range(1, self.n_nodes - 1):
                if i != j:
                    # Calculate reduced cost for arc (i,j)
                    rc = distances[i][j] - duals[i - 1] - duals[j - 1]

                    # Calculate complete path cost through this arc
                    path_cost = v_lambda + sp_down[i] + rc + sp_up[j]

                    # If path cost exceeds upper bound, we can fix this arc to zero
                    if path_cost > upper_bound:
                        fixed_arcs.add((i, j))

        return fixed_arcs

    def calculate_shortest_paths_from_root(
        self, duals: List[float], distances: List[List[float]]
    ) -> List[float]:
        """
        Calculate shortest paths from root to all nodes using reduced costs
        """
        n = self.n_nodes
        dist = [float("inf")] * n
        dist[0] = 0  # root

        # Use Dijkstra's algorithm with reduced costs
        visited = [False] * n
        while True:
            # Find unvisited node with minimum distance
            u = -1
            min_dist = float("inf")
            for i in range(n):
                if not visited[i] and dist[i] < min_dist:
                    min_dist = dist[i]
                    u = i

            if u == -1:
                break

            visited[u] = True

            # Update distances through u
            for v in range(1, n - 1):
                if u != v:
                    # Calculate reduced cost
                    rc = distances[u][v]
                    if u != 0:  # Not from root
                        rc -= duals[u - 1]
                    if v != n - 1:  # Not to terminal
                        rc -= duals[v - 1]

                    if dist[u] + rc < dist[v]:
                        dist[v] = dist[u] + rc

        return dist

    def calculate_shortest_paths_to_terminal(
        self, duals: List[float], distances: List[List[float]]
    ) -> List[float]:
        """
        Calculate shortest paths from all nodes to terminal using reduced costs
        """
        n = self.n_nodes
        dist = [float("inf")] * n
        dist[n - 1] = 0  # terminal

        # Use Dijkstra's algorithm with reduced costs
        visited = [False] * n
        while True:
            # Find unvisited node with minimum distance
            u = -1
            min_dist = float("inf")
            for i in range(n):
                if not visited[i] and dist[i] < min_dist:
                    min_dist = dist[i]
                    u = i

            if u == -1:
                break

            visited[u] = True

            # Update distances through u
            for v in range(1, n - 1):
                if u != v:
                    # Calculate reduced cost
                    rc = distances[v][u]
                    if v != 0:  # Not from root
                        rc -= duals[v - 1]
                    if u != n - 1:  # Not to terminal
                        rc -= duals[u - 1]

                    if dist[u] + rc < dist[v]:
                        dist[v] = dist[u] + rc

        return dist

    def calculate_solution_cost(self, paths: List[List[int]]) -> float:
        cost = 0.0
        for path in paths:
            for i in range(len(path) - 1):
                cost += self.calculate_travel_time(path[i], path[i + 1])
        return cost

    def solve_with_subgradient(self, max_iterations: int = 1000):
        """Modified solve method with column pool"""
        duals = [0.0] * self.n_nodes
        best_lb = float("-inf")
        best_ub = float("inf")
        alpha = 2.0
        step_reduction = 0.95
        non_improving_count = 0

        # Initialize duals as before
        for i in range(1, self.n_nodes - 1):
            duals[i - 1] = 50.0

        for iteration in range(max_iterations):
            self.graph.set_duals(duals)

            # Generate new columns and add to pool
            paths = self.graph.phaseFour()
            path_sequences = [
                p.nodes_covered for p in paths if len(p.nodes_covered) > 2
            ]
            for path in paths:
                if len(path.nodes_covered) > 2:
                    cost = self.calculate_solution_cost([path.nodes_covered])
                    self.column_pool.add_column(path.nodes_covered, cost)

            # Get columns for current iteration
            current_columns = self.column_pool.get_columns_for_rmp()
            # if not current_columns:
            #    break

            # Check conflicts using columns from pool
            conflict_found = False
            for path in current_columns:
                conflict = self.find_conflict(path)
                if conflict:
                    print(f"Found conflict in path: {path}")
                    print(f"Conflict details: {conflict}")
                    self.refine_conflict(conflict)
                    # Verify the conflict was actually eliminated
                    if self.find_conflict(path):
                        print("WARNING: Conflict still exists after refinement!")
                    self.column_pool.remove_column(path)
                    conflict_found = True
                    break

            # Solve RMP using subgradient
            current_value, visits = self.calculate_lagrangian(paths, duals)
            subgradient = [1 - visits[j] for j in range(1, self.n_nodes - 1)]
            print(
                f"Iteration {iteration}: Value: {current_value}, Subgradient: {subgradient}"
            )

            # Update bounds
            best_lb = max(best_lb, current_value)

            # Update step size
            if best_ub < float("inf"):
                # psi_star = best_lb * (1 + max(0.5, 5 / (100 + iteration)))
                psi_star = best_lb * (1 + 5 / (100 + iteration))

            else:
                psi_star = current_value * 1.1

            subgradient_norm = sum(g * g for g in subgradient)
            if subgradient_norm > 0:
                step_size = alpha * (psi_star - current_value) / subgradient_norm
            else:
                step_size = 0

            # Update duals
            new_duals = []
            for j in range(1, self.n_nodes - 1):
                new_val = max(0, duals[j - 1] - step_size * subgradient[j - 1])
                new_duals.append(new_val)

            # Handle convergence
            if current_value > best_lb - 0.01 * abs(best_lb):
                non_improving_count = 0
            else:
                non_improving_count += 1

            if non_improving_count > 20:
                alpha *= step_reduction
                non_improving_count = 0

            duals = new_duals

            # Calculate path cost
            path_cost = self.calculate_solution_cost(path_sequences)
            print(f"Path cost: {path_cost}")

            # Periodic arc fixing when we have good bounds
            if iteration > 0 and iteration % 10 == 0:
                fixed_arcs = self.fix_arcs(
                    best_ub,
                    duals,
                    [node.demand for node in self.nodes],
                    [(node.lb[0], node.ub[0]) for node in self.nodes],
                    self.instance.distances,
                    path_sequences,
                )
                self.graph.set_deleted_arcs(list(fixed_arcs))

            # Termination criteria
            if alpha < 0.01 or (
                iteration > 50 and abs(best_ub - best_lb) < 0.001 * abs(best_lb)
            ):
                break

        return best_lb, path_sequences


# Create instance data
solomon_instance = InstanceData()
solomon_instance.read_instance("../build/instances/R201.txt")
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
    bucket_interval=20,
    vehicle_capacity=700,
    n_vehicles=3,
    instance=instance,
)

# Solve
best_cost, best_routes = solver.solve_with_subgradient()
# best_cost, best_routes = solver.solve_with_column_elimination()
