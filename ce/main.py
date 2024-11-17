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
        self.conflict_manager = ConflictManager(self)

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

    def calculate_travel_time(self, node1: int, node2: int) -> float:
        if node1 == 101:
            node1 = 0
        if node2 == 101:
            node2 = 0
        return self.instance.distances[node1][node2]

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
        """Modified solve method with path selection optimization"""
        duals = [0.0] * self.n_nodes
        best_lb = float("-inf")
        best_ub = float("inf")
        alpha = 2.0
        step_reduction = 0.95
        non_improving_count = 0
        best_solution = None

        # Initialize duals
        for i in range(1, self.n_nodes - 1):
            duals[i - 1] = 100.0

        # Keep track of best feasible paths found
        feasible_paths = set()
        node_coverage = defaultdict(int)

        for iteration in range(max_iterations):
            self.graph.set_duals(duals)

            # Generate new columns and add to pool
            paths = self.graph.phaseFour()
            path_sequences = [
                p.nodes_covered for p in paths if len(p.nodes_covered) > 2
            ]

            # Check feasibility of new paths and add to pool
            for path in paths:
                if len(path.nodes_covered) > 2:
                    conflict = self.conflict_manager.find_conflict(path.nodes_covered)
                    if not conflict:  # Path is feasible
                        cost = self.calculate_solution_cost([path.nodes_covered])
                        self.column_pool.add_column(path.nodes_covered, cost)
                        feasible_paths.add(tuple(path.nodes_covered))

            # Try to construct a complete solution from feasible paths
            if feasible_paths:
                solution = self._select_best_paths(feasible_paths)
                if solution:
                    solution_cost = self.calculate_solution_cost(solution)
                    if solution_cost < best_ub:
                        best_ub = solution_cost
                        best_solution = solution
                        print(f"New best solution found with cost {best_ub}")

            # Get columns for current iteration
            current_columns = self.column_pool.get_columns_for_rmp()

            # Check conflicts using columns from pool
            conflict_found = False
            for path in current_columns:
                conflict = self.conflict_manager.find_conflict(path)
                if conflict:
                    if self.conflict_manager.refine_conflict(conflict):
                        self.column_pool.remove_column(path)
                        if tuple(path) in feasible_paths:
                            feasible_paths.remove(tuple(path))
                        conflict_found = True
                        break

            # Solve RMP using subgradient
            current_value, visits = self.calculate_lagrangian(paths, duals)
            subgradient = [1 - visits[j] for j in range(1, self.n_nodes - 1)]

            # Update bounds
            best_lb = max(best_lb, current_value)

            # Update step size
            if best_ub < float("inf"):
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

            # Print progress
            print(
                f"Iteration {iteration}: LB={best_lb:.2f}, UB={best_ub:.2f}, Gap={((best_ub-best_lb)/best_ub)*100:.2f}%"
            )

            # Termination criteria
            if alpha < 0.01 or (
                iteration > 50 and abs(best_ub - best_lb) < 0.001 * abs(best_lb)
            ):
                break

        return best_lb, best_solution

    def _select_best_paths(
        self, feasible_paths: Set[Tuple[int]]
    ) -> Optional[List[List[int]]]:
        """
        Select best combination of feasible paths to cover all nodes
        Uses a greedy selection strategy with path scoring
        """
        required_nodes = set(range(1, self.n_nodes - 1))  # All nodes except depot
        selected_paths = []
        remaining_nodes = required_nodes.copy()

        # Convert paths to list and sort by score
        paths_list = list(feasible_paths)
        path_scores = [
            (path, self._score_path(path, remaining_nodes)) for path in paths_list
        ]

        while remaining_nodes and path_scores:
            # Sort paths by score (higher is better)
            path_scores.sort(key=lambda x: x[1], reverse=True)

            # Select best path
            best_path, _ = path_scores[0]
            selected_paths.append(list(best_path))

            # Update remaining nodes
            path_nodes = set(best_path) - {0, self.n_nodes - 1}  # Exclude depot
            remaining_nodes -= path_nodes

            # Recalculate scores for remaining paths
            path_scores = [
                (p, self._score_path(p, remaining_nodes))
                for p, _ in path_scores[1:]
                if self._is_path_compatible(p, selected_paths)
            ]

        if not remaining_nodes:
            return selected_paths
        return None

    def _score_path(self, path: Tuple[int], remaining_nodes: Set[int]) -> float:
        """
        Score a path based on various criteria:
        - Coverage of remaining nodes
        - Path cost
        - Load utilization
        - Time window efficiency
        """
        path_nodes = set(path) - {0, self.n_nodes - 1}
        coverage = len(path_nodes & remaining_nodes)
        cost = self.calculate_solution_cost([list(path)])

        # Calculate load utilization
        total_load = sum(self.nodes[i].demand for i in path_nodes)
        load_ratio = total_load / self.vehicle_capacity

        # Time window efficiency (smaller is better)
        time_span = 0
        current_time = 0
        for i in range(len(path) - 1):
            if i > 0:
                current_time += self.nodes[path[i]].duration
            travel = self.calculate_travel_time(path[i], path[i + 1])
            current_time += travel
            time_span = max(time_span, current_time)

        # Combine factors into final score
        # Higher coverage and load utilization are better
        # Lower cost and time span are better
        score = (
            (coverage * 1000) + (load_ratio * 100) - (cost * 0.1) - (time_span * 0.01)
        )
        return score

    def _is_path_compatible(
        self, path: Tuple[int], selected_paths: List[List[int]]
    ) -> bool:
        """Check if a path is compatible with already selected paths"""
        path_nodes = set(path) - {0, self.n_nodes - 1}
        for selected in selected_paths:
            selected_nodes = set(selected) - {0, self.n_nodes - 1}
            if path_nodes & selected_nodes:  # If there's any overlap
                return False
        return True


# Create instance data
solomon_instance = InstanceData()
solomon_instance.read_instance("../build/instances/C203.txt")
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
