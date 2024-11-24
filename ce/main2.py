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
        self.path_decomposer = PathDecomposition(self)
        self.distances = instance.distances
        # Initialize baldes components
        self.vrp_nodes = nodes
        self.path_selector = PathSelector(self)

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
        Calculate the objective value of the Lagrangian relaxation according to the paper.

        L(λ) = min Σ(c_a - Σ g_j(a)λ_j)y_a + Σ λ_j b_j

        Where:
        - c_a is the travel time cost
        - g_j(a) indicates if node j is in arc a
        - λ_j are the dual variables
        - b_j = 1 (each node must be visited exactly once)
        """
        current_value = 0
        visits = defaultdict(int)

        # First term: path costs with dual adjustments
        for path in paths:
            # ath = path.nodes_covered
            # Calculate original path cost
            path_cost = sum(
                self.calculate_travel_time(path[i], path[i + 1])
                for i in range(len(path) - 1)
            )

            # Track visits and adjust cost by duals
            for node in path[1:-1]:  # Skip depots
                visits[node] += 1
                path_cost -= duals[node - 1]

            current_value += path_cost

        # Second term: sum of duals (since b_j = 1 for all j)
        current_value -= sum(duals)

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

    def _is_feasible_solution(self, solution: List[List[int]]) -> bool:
        """Check if a solution is feasible"""
        covered_nodes = set()
        for path in solution:
            path_nodes = set(path) - {0, self.n_nodes - 1}
            if path_nodes & covered_nodes:
                return False
            covered_nodes.update(path_nodes)
        return True

    def solve_with_subgradient(self, max_iterations: int = 10000):
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
        smoothing_factor = 0.7  # For exponential smoothing
        prev_duals = duals.copy()
        max_dual = 500.0  # Maximum allowed dual value
        min_dual = 0.0  # Minimum allowed dual value

        for iteration in range(max_iterations):
            print(duals)
            self.graph.set_duals(duals)

            # Generate new columns and add to pool
            paths = self.graph.solve_min_cost_flow(0, 101)
            print(paths)
            paths = [p.nodes_covered() for p in paths if len(p.nodes_covered()) > 2]
            # print(path_sequences)
            decomposed_paths = self.path_decomposer.decompose_solution(paths)

            # path_sequences = [p for p in decomposed_paths if len(p) > 2]
            # Check feasibility of new paths and add to pool
            for path in paths:
                conflict = self.conflict_manager.find_conflict(path)
                if conflict.type == "empty":
                    cost = self.calculate_solution_cost([path])
                    self.column_pool.add_column(path, cost)
                    feasible_paths.add(tuple(path))

            if feasible_paths:
                # Try all selection strategies
                for strategy_name in ["greedy"]:
                    if strategy_name == "greedy":
                        solution = self.path_selector._greedy_selection(feasible_paths)
                    elif strategy_name == "regret":
                        solution = self.path_selector._regret_based_selection(
                            feasible_paths
                        )
                    else:
                        solution = self.path_selector._load_balanced_selection(
                            feasible_paths
                        )

                    if solution:
                        solution_cost = self.calculate_solution_cost(solution)

                        # Check if solution is feasible
                        if self._is_feasible_solution(solution):
                            if solution_cost < best_ub:
                                best_ub = solution_cost
                                best_solution = solution
                                print(f"New best solution found with cost {best_ub}")
                                print(f"Using strategy: {strategy_name}")
                                print(f"Solution: {solution}")

            # Get columns for current iteration
            current_columns = self.column_pool.get_columns_for_rmp()

            # Check conflicts using columns from pool
            conflict_found = False
            for path in current_columns:
                conflict = self.conflict_manager.find_conflict(path)
                # print(f"Checking conflict for path {path}: {conflict}")
                if conflict.type != "empty":
                    print(f"Conflict found: {conflict}")
                    if self.conflict_manager.refine_conflict(conflict):
                        self.column_pool.remove_column(path)
                        if tuple(path) in feasible_paths:
                            feasible_paths.remove(tuple(path))
                        conflict_found = True
                        break

            # Solve RMP using subgradient
            # lag_paths = [paths[0]] * 3
            # paths = paths[0:3]
            # paths = [paths[0]] * 3
            current_value, visits = self.calculate_lagrangian(paths, duals)

            subgradient = [1 - visits[j] for j in range(1, self.n_nodes - 1)]

            # Update bounds
            best_lb = max(best_lb, current_value)

            # Update step size
            if best_ub < float("inf"):
                psi_star = best_lb * (1 + 5 / (100 + iteration))
            else:
                # psi_star = current_value * 1.1
                psi_star = current_value * 1.1

            subgradient_norm = sum(g * g for g in subgradient)
            if subgradient_norm > 0:
                step_size = alpha * (psi_star - current_value) / subgradient_norm
            # step_size = alpha * (psi_star - current_value) / subgradient_norm
            else:
                step_size = 0

            # Update duals
            # Update duals with smoothing and bounds
            new_duals = []
            for j in range(1, self.n_nodes - 1):
                # Calculate raw update
                raw_update = duals[j - 1] - step_size * subgradient[j - 1]

                new_duals.append(raw_update)

            # Handle convergence
            gap = abs(best_ub - best_lb) / (abs(best_lb) + 1e-10)
            if gap < 0.01:  # 1% gap
                print(f"Converged with gap {gap*100:.2f}%")
                break

            if abs(current_value - best_lb) < 0.01 * abs(best_lb):
                non_improving_count += 1
            else:
                non_improving_count = 0

            if non_improving_count > 20:
                alpha *= step_reduction
                smoothing_factor = min(
                    0.95, smoothing_factor + 0.01
                )  # Increase smoothing
                non_improving_count = 0

            # Store previous duals and update current ones
            prev_duals = duals.copy()
            duals = new_duals

            # Periodic arc fixing when we have good bounds
            if iteration > 0 and iteration % 10 == 0:
                fixed_arcs = self.fix_arcs(
                    best_ub,
                    duals,
                    [node.demand for node in self.nodes],
                    [(node.lb[0], node.ub[0]) for node in self.nodes],
                    self.instance.distances,
                    paths,
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


# Create instance data
solomon_instance = InstanceData()
solomon_instance.read_instance("../build/instances/C202.txt")
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
