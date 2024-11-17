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

    def find_conflict(self, path: List[int]) -> Optional[Tuple[str, List[int], Dict]]:
        """
        Enhanced conflict detection that returns type of conflict and relevant state
        Returns: (conflict_type, conflict_sequence, state_at_conflict)
        """
        # Check for cycles first
        visited = set()
        for i, node in enumerate(path):
            if node != 0 and node in visited:  # Allow depot visits
                cycle_start = path.index(node)
                return ("cycle", path[: i + 1], {"cycle_node": node})
            visited.add(node)

        # Track state
        current_state = {"load": 0.0, "time": 0}

        # Check capacity and time window constraints
        for i in range(len(path) - 1):
            prev_node = path[i]
            curr_node = path[i + 1]

            # Update load
            new_load = current_state["load"] + self.nodes[curr_node].demand
            if new_load > self.vehicle_capacity:
                # Find minimal subsequence causing violation
                violation_seq = self._find_minimal_capacity_violation(path[: i + 2])
                print(f"Capacity violation at {violation_seq}")
                return ("capacity", violation_seq, {"load": new_load})

            # Update time
            if i > 0:
                current_state["time"] += self.nodes[prev_node].duration
            travel_time = self.calculate_travel_time(prev_node, curr_node)
            arrival_time = current_state["time"] + travel_time

            if arrival_time > self.nodes[curr_node].ub[0]:
                print(f"Time window violation at {prev_node}->{curr_node}")
                return (
                    "time_window",
                    [prev_node, curr_node],
                    {"time": arrival_time, "latest": self.nodes[curr_node].ub[0]},
                )

            current_state["time"] = max(arrival_time, self.nodes[curr_node].lb[0])
            current_state["load"] = new_load

        return None

    def refine_conflict(
        self, conflict: Tuple[str, List[int], Dict], conflict_path: List[int]
    ):
        """
        Enhanced conflict refinement that properly handles cycles and updates NG relaxation
        """
        conflict_type, sequence, state = conflict
        print(f"Refining {conflict_type} conflict in sequence: {sequence}")

        if conflict_type == "cycle":
            # 1. Update NG neighborhoods for cycle
            cycle_node = state["cycle_node"]
            conflicts = []

            # Find cycle positions
            first_pos = sequence.index(cycle_node)
            second_pos = sequence.index(cycle_node, first_pos + 1)
            cycle = sequence[first_pos : second_pos + 1]

            print(f"Found cycle: {cycle}")

            # Add all pairs in cycle to conflicts
            for i in range(len(cycle)):
                for j in range(i + 1, len(cycle)):
                    if cycle[i] != 0 and cycle[j] != 0:  # Skip depot
                        conflicts.append((cycle[i], cycle[j]))
                        conflicts.append((cycle[j], cycle[i]))  # Both directions

            print(f"Updating NG neighborhoods with conflicts: {conflicts}")
            self.graph.update_ng_neighbors(conflicts)

            # Print updated neighborhoods for debugging
            for node in cycle:
                if node != 0:
                    size = self.graph.get_neighborhood_size(node)
                    neighbors = self.graph.get_neighbors(node)
                    print(f"Node {node} now has neighborhood size {size}: {neighbors}")

            # 2. Also proceed with standard refinement
            print("Proceeding with standard refinement after updating NG neighborhoods")

        # Standard refinement process
        for j in range(1, len(sequence)):
            prefix = sequence[:j]
            next_node = sequence[j]

            print(f"Analyzing prefix {prefix} → {next_node}")
            initial_state = self.calculate_initial_state(prefix)
            print(
                f"Starting exploration from prefix {prefix} with state {initial_state}"
            )

            # Find valid transitions to current state
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
                self.graph.set_deleted_arcs([(prefix[-1], next_node)])
                return

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
        Find all valid transitions that could lead to the current state,
        following Algorithm 1 from the paper (S^- set)
        """
        if not prefix:
            return []

        valid_transitions = []
        current_state = self.calculate_initial_state(prefix)
        print(f"Starting exploration from prefix {prefix} with state {current_state}")

        # Package current state information for bucket graph
        resources = [current_state["time"]]

        # Get extensions from bucket graph
        extensions = self.graph.extend_path(prefix, resources)
        if not extensions:
            return []

        # Validate extensions
        for path in extensions:
            path_nodes = path.nodes_covered
            if self.is_path_feasible(path_nodes):
                valid_transitions.append(path_nodes)

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

    def calculate_travel_time(self, node1: int, node2: int) -> float:
        if node1 == 101:
            node1 = 0
        if node2 == 101:
            node2 = 0
        return self.instance.distances[node1][node2]

    def _find_minimal_capacity_violation(self, path: List[int]) -> List[int]:
        """Find minimal subsequence causing capacity violation"""
        total_load = 0
        min_seq = []

        # Forward pass - find violation
        for node in path:
            total_load += self.nodes[node].demand
            min_seq.append(node)
            if total_load > self.vehicle_capacity:
                break

        # Backward pass - minimize sequence
        while len(min_seq) > 2:  # Keep at least two nodes
            first_node = min_seq[0]
            if total_load - self.nodes[first_node].demand > self.vehicle_capacity:
                total_load -= self.nodes[first_node].demand
                min_seq.pop(0)
            else:
                break

        return min_seq

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
            duals[i - 1] = 100.0

        for iteration in range(max_iterations):
            self.graph.set_duals(duals)

            # Generate new columns and add to pool
            paths = self.graph.phaseFour()
            print(len(paths))
            path_sequences = [
                p.nodes_covered for p in paths if len(p.nodes_covered) > 2
            ]
            print(f"Iteration {iteration}: Found {path_sequences}")
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
                    self.refine_conflict(conflict, path)
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
solomon_instance.read_instance("../build/instances/C201.txt")
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
# best_cost, best_routes = solver.solve_with_column_elimination()
