from heapq import heappop, heappush
from typing import Dict, Optional, Set, Tuple, List
from collections import defaultdict


class ArcFixingManager:
    """Manages sophisticated arc fixing strategies"""

    def __init__(self, n_nodes: int, vehicle_capacity: float, time_horizon: float):
        self.n_nodes = n_nodes
        self.vehicle_capacity = vehicle_capacity
        self.time_horizon = time_horizon
        self.fixed_arcs: Set[Tuple[int, int]] = set()
        self.arc_scores = defaultdict(float)  # Tracks historical fixing scores
        self.fixing_history: Dict[Tuple[int, int], List[bool]] = defaultdict(list)

    def update_history(self, arc: Tuple[int, int], was_useful: bool):
        """Update historical information about arc fixing decisions"""
        self.fixing_history[arc].append(was_useful)
        if len(self.fixing_history[arc]) > 10:  # Keep rolling window
            self.fixing_history[arc].pop(0)


class ColumnPool:
    def __init__(self):
        self.columns = set()  # Store unique columns as tuples
        self.best_columns = []  # Store best columns found
        self.column_costs = {}  # Map columns to their costs

    def add_column(self, column, cost):
        """
        Add new column if not already present
        column: List of nodes in the path
        cost: Cost of the path
        """
        column_key = tuple(column)
        if column_key not in self.columns:
            self.columns.add(column_key)
            self.column_costs[column_key] = cost
            self.update_best_columns(column_key, cost)

    def update_best_columns(self, column_key, cost):
        """
        Keep track of best columns found
        column_key: Tuple of nodes in the path
        cost: Cost of the path
        """
        self.best_columns.append((column_key, cost))
        self.best_columns.sort(key=lambda x: x[1])  # Sort by cost
        if len(self.best_columns) > 100:  # Keep only top 100
            self.best_columns = self.best_columns[:100]

    def get_columns_for_rmp(self):
        """Get columns for restricted master problem"""
        # Return lists of nodes
        return [list(col) for col, _ in self.best_columns]

    def remove_column(self, column):
        """
        Remove a column from the pool
        column: List of nodes to remove
        """
        column_key = tuple(column)
        if column_key in self.columns:
            self.columns.remove(column_key)
            self.column_costs.pop(column_key, None)
            # Remove from best columns if present
            self.best_columns = [
                (col, c) for col, c in self.best_columns if col != column_key
            ]

    def get_cost(self, column):
        """Get cost of a column if it exists"""
        return self.column_costs.get(tuple(column))

    def __len__(self):
        """Return the number of unique columns in the pool"""
        return len(self.columns)


class Conflict:
    """Class representing a conflict in a path"""

    def __init__(self, type: str, sequence: List[int], details: Dict):
        self.type = type
        self.sequence = sequence
        self.details = details  # Changed from state to details to match usage

    def __str__(self):
        return f"Conflict(type={self.type}, sequence={self.sequence}, details={self.details})"


from typing import List, Dict, Optional, Set
from collections import defaultdict

from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass


from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from copy import deepcopy

# import Union from typing
from typing import Union


class ConflictManager:
    """Enhanced class to manage conflict detection and refinement with proper state management"""

    def __init__(self, solver):
        self.solver = solver
        self.refinement_stats = defaultdict(int)

        # Cache for frequently accessed data
        self._node_demands = {i: node.demand for i, node in enumerate(solver.nodes)}
        self._node_durations = {i: node.duration for i, node in enumerate(solver.nodes)}
        self._node_time_windows = {
            i: (node.lb[0], node.ub[0]) for i, node in enumerate(solver.nodes)
        }
        self._travel_times = {}
        self._feasibility_cache = {}

        # State management for refinement operations
        self._current_state = None
        self._state_history = []

    def calculate_travel_time(self, from_node: int, to_node: int) -> float:
        """Calculate travel time between two nodes"""
        return self.solver.distances[from_node][to_node]
    
    def _get_travel_time(self, from_node: int, to_node: int) -> float:
        """Cached travel time calculation"""
        key = (from_node, to_node)
        if key not in self._travel_times:
            self._travel_times[key] = self.calculate_travel_time(
                from_node, to_node
            )
        return self._travel_times[key]

    def get_valid_transitions(self, prefix: List[int]) -> List[List[int]]:
        """Find all valid transitions that could lead to current state with early exits"""
        if not prefix:
            return []

        # Early capacity check
        current_load = sum(self._node_demands[node] for node in prefix)
        if current_load > self.solver.vehicle_capacity:
            return []

        # Calculate time only if capacity check passes
        current_time = self._calculate_time(prefix)
        resources = [current_time]

        # Get extensions from graph
        extensions = self.solver.graph.extend_path(prefix, resources)

        # Filter extensions in bulk rather than checking each individually
        valid_paths = []
        for path in extensions:
            path_key = tuple(path.nodes_covered())
            if path_key in self._feasibility_cache:
                if self._feasibility_cache[path_key]:
                    valid_paths.append(path.nodes_covered())
            elif self._quick_feasibility_check(path.nodes_covered()):
                valid_paths.append(path.nodes_covered())
                self._feasibility_cache[path_key] = True
            else:
                self._feasibility_cache[path_key] = False

        return valid_paths

    def _quick_feasibility_check(self, path: List[int]) -> bool:
        """Fast preliminary feasibility check with early exits"""
        # Quick capacity check
        total_load = sum(self._node_demands[node] for node in path)
        if total_load > self.solver.vehicle_capacity:
            return False

        # Optimistic time window check
        time = 0
        for i in range(len(path) - 1):
            if i > 0:
                time += self._node_durations[path[i]]
            time += self._get_travel_time(path[i], path[i + 1])

            if time > self._node_time_windows[path[i + 1]][1]:
                return False

            time = max(time, self._node_time_windows[path[i + 1]][0])

        return True

    def _calculate_time(self, path: List[int]) -> float:
        """Optimized time calculation"""
        if not path:
            return 0

        time = 0
        for i in range(1, len(path)):
            if i > 1:
                time += self._node_durations[path[i - 1]]
            time += self._get_travel_time(path[i - 1], path[i])
            time = max(time, self._node_time_windows[path[i]][0])
        return time

    def find_conflict(self, path: List[int]) -> Conflict:
        """Enhanced conflict detection returning Conflict objects"""
        # Check for cycles
        visited = set()
        for i, node in enumerate(path):
            if node != 0 and node in visited:
                cycle_start = path.index(node)
                cycle_sequence = path[cycle_start : i + 1]
                self.refinement_stats["cycle_conflicts"] += 1
                # print(pato)
                novo_conflito = Conflict(
                    type="cycle", sequence=cycle_sequence, details={"cycle_node": node}
                )
                # print(novo_conflito)
                return novo_conflito
            visited.add(node)

        # Track cumulative load and time
        time = 0
        load = 0
        violation_start = 0

        # Check each node in sequence
        for i, node in enumerate(path):
            # Update load
            load += self._node_demands[node]

            # Capacity check
            if load > self.solver.vehicle_capacity:
                # Find minimal subsequence causing violation
                for start in range(violation_start, i + 1):
                    subload = sum(
                        self._node_demands[path[j]] for j in range(start, i + 1)
                    )
                    if subload > self.solver.vehicle_capacity:
                        subsequence = path[start : i + 1]
                        self.refinement_stats["capacity_conflicts"] += 1
                        return Conflict(
                            type="capacity",
                            sequence=subsequence,
                            details={"load": subload},
                        )

            # Time window check
            if i > 0:
                # Add service time at previous location
                time += self._node_durations[path[i - 1]]

                # Add travel time to current location
                travel = self._get_travel_time(path[i - 1], node)
                time += travel

                # Cannot arrive before earliest time
                time = max(time, self._node_time_windows[node][0])

                # Check if we've violated the latest arrival time
                if time > self._node_time_windows[node][1]:
                    # Find minimal subsequence causing violation
                    for start in range(violation_start, i):
                        subtime = self._node_time_windows[path[start]][
                            0
                        ]  # Start at earliest time
                        for j in range(start, i + 1):
                            if j > start:
                                subtime += self._node_durations[path[j - 1]]
                                subtime += self._get_travel_time(path[j - 1], path[j])
                                subtime = max(
                                    subtime, self._node_time_windows[path[j]][0]
                                )

                            if subtime > self._node_time_windows[path[j]][1]:
                                subsequence = path[start : j + 1]
                                self.refinement_stats["time_window_conflicts"] += 1
                                return Conflict(
                                    type="time_window",
                                    sequence=subsequence,
                                    details={
                                        "time": subtime,
                                        "latest": self._node_time_windows[path[j]][1],
                                    },
                                )

            # Only update violation_start if we're still feasible
            if i > 0 and time <= self._node_time_windows[node][1]:
                violation_start = i

        # No conflicts found
        new_empty_conflict = Conflict(type="empty", sequence=[], details={})
        return new_empty_conflict

    def refine_conflict(self, conflict: Conflict) -> bool:
        """Refine conflict with state preservation"""
        try:
            print("Refining conflict")
            sequence = conflict.sequence

            # Save state before refinement
            self._state_history.append(deepcopy(self._current_state))

            # Fast path for cycle conflicts
            if conflict.type == "cycle":
                return self._refine_cycle(conflict)

            # Process sequence positions
            for j in range(1, len(sequence)):
                prefix = sequence[:j]
                next_node = sequence[j]

                # Get valid transitions with caching
                valid_transitions = self.get_valid_transitions(prefix)

                if not valid_transitions:
                    self.solver.graph.set_deleted_arcs([(prefix[-1], next_node)])
                    self.refinement_stats["infeasible_transitions"] += 1
                    return True

                # Check extensions
                can_extend = False
                for trans in valid_transitions:
                    extended = trans + [next_node]
                    path_key = tuple(extended)

                    if path_key in self._feasibility_cache:
                        can_extend = self._feasibility_cache[path_key]
                        if can_extend:
                            break
                    elif self._quick_feasibility_check(extended):
                        can_extend = True
                        self._feasibility_cache[path_key] = True
                        break
                    else:
                        self._feasibility_cache[path_key] = False

                if not can_extend:
                    self.solver.graph.set_deleted_arcs([(prefix[-1], next_node)])
                    self.refinement_stats["infeasible_transitions"] += 1
                    return True

            self.refinement_stats["successful_refinements"] += 1
            return True

        except Exception as e:
            # Restore previous state on error
            if self._state_history:
                self._current_state = self._state_history.pop()
            print(f"Error during conflict refinement: {e}")
            return False

    def _refine_cycle(self, conflict: Conflict) -> bool:
        """Optimized cycle refinement"""
        sequence = conflict.sequence
        conflicts = set()  # Use set for faster lookups

        # Add all pairs in cycle to conflicts
        for i, node_i in enumerate(sequence):
            if node_i != 0:  # Skip depot
                for node_j in sequence[i + 1 :]:
                    if node_j != 0:  # Skip depot
                        conflicts.add((node_i, node_j))
                        conflicts.add((node_j, node_i))

        self.solver.graph.update_ng_neighbors(list(conflicts))
        return True

    def clear_caches(self) -> None:
        """Clear internal caches to free memory"""
        self._travel_times.clear()
        self._feasibility_cache.clear()
        self._state_history.clear()
        self._current_state = None

    def get_stats(self) -> Dict:
        """Get complete statistics"""
        return dict(self.refinement_stats)


import heapq
from collections import defaultdict
from typing import List, Dict

INF = float("inf")


class PathDecomposition:
    def __init__(self, solver, source=0, sink=101):
        self.solver = solver
        self.source = source
        self.sink = sink

    def decompose_solution(self, paths) -> List[List[int]]:
        """Decompose flow using shortest paths in the residual graph."""
        flow_graph = self._build_flow_graph(paths)
        decomposed_paths = []

        while True:
            # Find shortest path in residual graph
            path = self._find_shortest_path(flow_graph)
            if not path or len(path) <= 1:
                break

            # Get minimum flow along path
            min_flow = self._get_min_flow(flow_graph, path)
            if min_flow <= 0.001:
                break

            # Update residual graph
            self._update_residual_graph(flow_graph, path, min_flow)
            decomposed_paths.append(path)

        return decomposed_paths

    def _build_flow_graph(self, paths):
        """Convert paths to a flow graph."""
        flow_graph = defaultdict(lambda: defaultdict(float))
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                flow_graph[u][v] += 1  # Increase flow count along the path
        return flow_graph

    def _find_shortest_path(self, flow_graph):
        """Find the shortest path in the residual graph using Dijkstra's algorithm."""
        distances = {self.source: 0}
        predecessors = {self.source: None}
        pq = [(0, self.source)]  # (distance, node)

        while pq:
            dist, node = heapq.heappop(pq)
            if dist > distances.get(node, INF):
                continue

            for next_node, flow in flow_graph[node].items():
                if flow > 0.001:  # Only consider edges with positive flow
                    new_dist = dist + self.solver.calculate_travel_time(node, next_node)
                    if new_dist < distances.get(next_node, INF):
                        distances[next_node] = new_dist
                        predecessors[next_node] = node
                        heapq.heappush(pq, (new_dist, next_node))

        # Reconstruct the path from source to sink
        if self.sink not in predecessors:
            return None

        path = []
        node = self.sink
        while node is not None:
            path.append(node)
            node = predecessors[node]
        path.reverse()
        return path

    def _get_min_flow(self, flow_graph, path):
        """Get the minimum flow along a path."""
        return min(flow_graph[u][v] for u, v in zip(path[:-1], path[1:]))

    def _update_residual_graph(self, flow_graph, path, flow_value):
        """Update the residual graph by subtracting flow on the forward edges and adding flow on the reverse edges."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Subtract flow from the forward edge
            flow_graph[u][v] -= flow_value
            # Add flow to the reverse edge
            flow_graph[v][u] += flow_value


class PathScore:
    """Class to represent and compare path scores"""

    def __init__(self, path: List[int], score_components: Dict[str, float]):
        self.path = path
        self.components = score_components
        self.total_score = self._calculate_total_score()

    def _calculate_total_score(self) -> float:
        # Weights for different components
        weights = {
            "coverage_ratio": 1000,
            "load_efficiency": 200,
            "time_efficiency": 300,
            "cost_efficiency": 500,
            "route_balance": 150,
        }
        return sum(weights[k] * v for k, v in self.components.items())

    def __lt__(self, other):
        return self.total_score < other.total_score


class PathSelector:
    """Enhanced path selection with multiple strategies"""

    def __init__(self, solver):
        self.solver = solver
        self.best_solution = None
        self.best_cost = float("inf")

    def select_best_paths(
        self, feasible_paths: Set[Tuple[int]]
    ) -> Optional[List[List[int]]]:
        """Select best combination of paths using multiple strategies"""
        if not feasible_paths:
            return None

        solutions = []

        # Try different selection strategies
        strategies = [
            self._greedy_selection,
            self._regret_based_selection,
            self._load_balanced_selection,
        ]

        for strategy in strategies:
            solution = strategy(feasible_paths)
            if solution:
                solutions.append(solution)

        # Return best solution found
        if not solutions:
            return None

        return min(solutions, key=lambda s: self.solver.calculate_solution_cost(s))

    def _greedy_selection(
        self, feasible_paths: Set[Tuple[int]]
    ) -> Optional[List[List[int]]]:
        """Greedy selection based on comprehensive path scoring"""
        required_nodes = set(range(1, self.solver.n_nodes - 1))
        selected_paths = []
        remaining_nodes = required_nodes.copy()

        while remaining_nodes:
            # Score all compatible paths
            scored_paths = []
            for path in feasible_paths:
                if self._is_compatible(path, selected_paths):
                    score = self._score_path(path, remaining_nodes, selected_paths)
                    scored_paths.append((path, score))

            if not scored_paths:
                return None

            # Select best path
            best_path = max(scored_paths, key=lambda x: x[1].total_score)[0]
            selected_paths.append(list(best_path))

            # Update remaining nodes
            path_nodes = set(best_path) - {0, 101}
            remaining_nodes -= path_nodes

            # Early termination if we exceed vehicle limit
            if len(selected_paths) > self.solver.n_vehicles:
                return None

        return selected_paths

    def _regret_based_selection(
        self, feasible_paths: Set[Tuple[int]]
    ) -> Optional[List[List[int]]]:
        """Selection based on regret value - considering cost of not choosing a path"""
        required_nodes = set(range(1, self.solver.n_nodes - 1))
        selected_paths = []
        remaining_nodes = required_nodes.copy()

        while remaining_nodes:
            # Calculate regret values
            regrets = []
            for path in feasible_paths:
                if not self._is_compatible(path, selected_paths):
                    continue

                # Calculate cost with and without this path
                base_score = self._score_path(path, remaining_nodes, selected_paths)
                alternative_scores = []

                for alt_path in feasible_paths:
                    if alt_path != path and self._is_compatible(
                        alt_path, selected_paths
                    ):
                        alt_score = self._score_path(
                            alt_path, remaining_nodes, selected_paths
                        )
                        alternative_scores.append(alt_score)

                if alternative_scores:
                    best_alternative = max(
                        alternative_scores, key=lambda x: x.total_score
                    )
                    regret = base_score.total_score - best_alternative.total_score
                    regrets.append((path, regret))

            if not regrets:
                return None

            # Select path with highest regret
            selected_path = max(regrets, key=lambda x: x[1])[0]
            selected_paths.append(list(selected_path))

            # Update remaining nodes
            path_nodes = set(selected_path) - {0, 101}
            remaining_nodes -= path_nodes

            if len(selected_paths) > self.solver.n_vehicles:
                return None

        return selected_paths

    def _score_path(
        self,
        path: Tuple[int],
        remaining_nodes: Set[int],
        selected_paths: List[List[int]],
    ) -> PathScore:
        """Enhanced path scoring with multiple criteria"""
        path_nodes = set(path) - {0, 101}

        # Coverage efficiency
        coverage_ratio = (
            len(path_nodes & remaining_nodes) / len(remaining_nodes)
            if remaining_nodes
            else 0
        )

        # Load efficiency
        total_load = sum(self.solver.nodes[i].demand for i in path_nodes)
        load_efficiency = total_load / self.solver.vehicle_capacity

        # Time window efficiency
        earliest_possible = (
            min(self.solver.nodes[i].lb[0] for i in path_nodes) if path_nodes else 0
        )
        latest_required = (
            max(self.solver.nodes[i].ub[0] for i in path_nodes) if path_nodes else 0
        )
        time_span = latest_required - earliest_possible
        max_span = self.solver.time_horizon
        time_efficiency = 1 - (time_span / max_span) if max_span > 0 else 0

        # Cost efficiency
        path_cost = self.solver.calculate_solution_cost([list(path)])
        avg_cost_per_node = path_cost / len(path_nodes) if path_nodes else float("inf")
        cost_efficiency = 1 / (1 + avg_cost_per_node)

        # Route balance with existing routes
        if selected_paths:
            avg_route_length = sum(len(p) for p in selected_paths) / len(selected_paths)
            length_diff = abs(len(path) - avg_route_length)
            route_balance = 1 / (1 + length_diff)
        else:
            route_balance = 1.0

        return PathScore(
            list(path),
            {
                "coverage_ratio": coverage_ratio,
                "load_efficiency": load_efficiency,
                "time_efficiency": time_efficiency,
                "cost_efficiency": cost_efficiency,
                "route_balance": route_balance,
            },
        )

    def _is_compatible(
        self, new_path: Tuple[int], selected_paths: List[List[int]]
    ) -> bool:
        """Check if new path is compatible with selected paths"""
        new_nodes = set(new_path) - {0, 101}

        # Check for node overlap
        for path in selected_paths:
            path_nodes = set(path) - {0, 101}
            if new_nodes & path_nodes:
                return False

        # Check vehicle capacity
        if len(selected_paths) + 1 > self.solver.n_vehicles:
            return False

        return True

    def _load_balanced_selection(
        self, feasible_paths: Set[Tuple[int]]
    ) -> Optional[List[List[int]]]:
        """Selection strategy focusing on balanced load distribution"""
        required_nodes = set(range(1, self.solver.n_nodes - 1))
        selected_paths = []
        remaining_nodes = required_nodes.copy()

        # Calculate total demand to aim for balance
        total_demand = sum(self.solver.nodes[i].demand for i in required_nodes)
        target_load_per_vehicle = (
            total_demand / self.solver.n_vehicles
            if self.solver.n_vehicles > 0
            else total_demand
        )

        while remaining_nodes:
            # Score paths based on load balancing criteria
            scored_paths = []
            current_loads = [
                sum(self.solver.nodes[n].demand for n in path[1:-1])
                for path in selected_paths
            ]
            avg_current_load = (
                sum(current_loads) / len(current_loads) if current_loads else 0
            )

            for path in feasible_paths:
                if not self._is_compatible(path, selected_paths):
                    continue

                path_load = sum(
                    self.solver.nodes[i].demand for i in path if i not in {0, 101}
                )

                # Calculate how well this path balances the solution
                if selected_paths:
                    load_deviation = abs(path_load - avg_current_load)
                    target_deviation = abs(path_load - target_load_per_vehicle)
                else:
                    load_deviation = abs(path_load - target_load_per_vehicle)
                    target_deviation = load_deviation

                # Create comprehensive score
                score_components = {
                    "load_balance": 1.0 / (1.0 + load_deviation),
                    "target_balance": 1.0 / (1.0 + target_deviation),
                    "coverage_ratio": len(set(path) & remaining_nodes)
                    / len(remaining_nodes),
                    "capacity_usage": path_load / self.solver.vehicle_capacity,
                    "time_efficiency": self._calculate_time_efficiency(path),
                }

                scored_paths.append((path, PathScore(list(path), score_components)))

            if not scored_paths:
                return None

            # Select best balanced path
            best_path = max(scored_paths, key=lambda x: x[1].total_score)[0]
            selected_paths.append(list(best_path))

            # Update remaining nodes
            path_nodes = set(best_path) - {0, 101}
            remaining_nodes -= path_nodes

            # Check vehicle limit
            if len(selected_paths) > self.solver.n_vehicles:
                return None

        return selected_paths

    def _calculate_time_efficiency(self, path: Tuple[int]) -> float:
        """Calculate time window efficiency for a path"""
        if len(path) <= 2:  # Only depots
            return 0.0

        current_time = 0
        total_waiting = 0
        max_time = 0

        for i in range(len(path) - 1):
            if i > 0:
                current_time += self.solver.nodes[path[i]].duration

            travel = self.solver.calculate_travel_time(path[i], path[i + 1])
            current_time += travel

            if i < len(path) - 1:  # Not for end depot
                earliest = self.solver.nodes[path[i + 1]].lb[0]
                if current_time < earliest:
                    total_waiting += earliest - current_time
                    current_time = earliest

                latest = self.solver.nodes[path[i + 1]].ub[0]
                if current_time > latest:
                    return 0.0  # Infeasible path

            max_time = max(max_time, current_time)

        # Return normalized efficiency score
        time_span = max_time - self.solver.nodes[path[0]].lb[0]
        total_service_time = sum(self.solver.nodes[i].duration for i in path[1:-1])
        total_travel_time = sum(
            self.solver.calculate_travel_time(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

        if time_span <= 0:
            return 1.0

        efficiency = (total_service_time + total_travel_time) / (
            time_span + total_waiting
        )
        return min(1.0, efficiency)


from collections import defaultdict, deque
import heapq
from typing import Dict, List, Set, Tuple, Optional
import heapq
from collections import defaultdict

INF = float("inf")
import heapq
from collections import defaultdict

INF = float("inf")


class muSSP:
    def __init__(self, adj_list, source, sink):
        self.adj_list = adj_list
        self.original_graph = {u: adj_list[u].copy() for u in adj_list}
        self.source = source
        self.sink = sink
        self.flow = defaultdict(int)
        self.costs = {}
        self.zero_tree = {}
        self.initialize_costs()

    def initialize_costs(self):
        """Initialize costs using topological sort + cost conversion"""
        for u in self.adj_list:
            for v, cost, _ in self.adj_list[u]:
                self.costs[(u, v)] = cost

        distances = self._topological_shortest_paths()

        # Convert to reduced costs
        for u in self.adj_list:
            for v, cost, cap in list(self.adj_list[u]):
                reduced_cost = cost + distances[u] - distances[v]
                self.adj_list[u].remove((v, cost, cap))
                self.adj_list[u].append((v, max(0, reduced_cost), cap))

    def _topological_shortest_paths(self):
        """Find shortest paths using topological sort"""
        order = []
        visited = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for v, _, _ in self.adj_list.get(node, []):
                dfs(v)
            order.append(node)

        dfs(self.source)
        order.reverse()

        distances = {v: float("inf") for v in self.adj_list}
        distances[self.source] = 0

        for u in order:
            for v, cost, _ in self.adj_list.get(u, []):
                if distances[u] + cost < distances[v]:
                    distances[v] = distances[u] + cost

        return distances

    def _identify_zero_tree(self, path):
        """Identify 0-tree after path flipping"""
        self.zero_tree = defaultdict(set)
        visited = set()

        def build_tree(node):
            if node in visited:
                return
            visited.add(node)
            for v, cost, _ in self.adj_list.get(node, []):
                if cost == 0 and v not in visited:
                    self.zero_tree[node].add(v)
                    build_tree(v)

        # Start from path's second node
        if len(path) > 1:
            build_tree(path[1])

    def _batch_update_distances(self, distances, node, dist):
        """Update distances for all descendants in 0-tree"""
        queue = deque([(node, dist)])
        updated = {node}

        while queue:
            current, d = queue.popleft()
            distances[current] = d

            for descendant in self.zero_tree[current]:
                if descendant not in updated:
                    queue.append((descendant, d))
                    updated.add(descendant)

    def shortest_path(self):
        self.distances = {node: float("inf") for node in self.adj_list}
        self.parent = {node: None for node in self.adj_list}
        self.distances[self.source] = 0

        queue = [(0, self.source)]
        while queue:
            queue.sort()
            d, u = queue.pop(0)

            if d > self.distances[u]:
                continue

            for v, cost, cap in self.adj_list[u]:
                if self.flow[(u, v)] < cap:
                    new_dist = self.distances[u] + cost
                    if new_dist < self.distances[v]:
                        self.distances[v] = new_dist
                        self.parent[v] = u
                        queue.append((new_dist, v))

        return self.distances[self.sink] != float("inf")

    def solve_from_path(self, initial_path):
        paths = []

        # Add initial path and create reverse edges
        for i in range(len(initial_path) - 1):
            u, v = initial_path[i], initial_path[i + 1]
            # Update flow
            self.flow[(u, v)] = 1
            # Add reverse edge
            if v not in self.adj_list:
                self.adj_list[v] = []
            self.adj_list[v].append([u, -self.costs[(u, v)], 1])

        paths.append(initial_path)

        # Find additional paths
        while self.shortest_path():
            # Reconstruct path
            path = []
            v = self.sink
            while v is not None:
                path.append(v)
                v = self.parent[v]
            path.reverse()

            # Update flow and add reverse edges
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.flow[(u, v)] += 1
                self.flow[(v, u)] -= 1

                # Add reverse edge if needed
                reverse_exists = False
                for edge in self.adj_list[v]:
                    if edge[0] == u:
                        reverse_exists = True
                        break
                if not reverse_exists:
                    self.adj_list[v].append([u, -self.costs[(u, v)], 1])

            paths.append(path)

        # Calculate total cost
        total_cost = 0
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                total_cost += self.costs[(u, v)]

        return paths, total_cost

    def _clip_permanent_edges(self):
        """Remove edges to source or from sink"""
        for u in list(self.adj_list.keys()):
            self.adj_list[u] = [
                (v, c, cap)
                for v, c, cap in self.adj_list[u]
                if v != self.source and u != self.sink
            ]

    def _update_residual_graph(self, path):
        """Update residual graph after pushing flow"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            # Remove forward edge if used up
            self.flow[(u, v)] += 1
            if self.flow[(u, v)] == 1:
                self.adj_list[u] = [
                    (w, c, cap) for w, c, cap in self.adj_list[u] if w != v
                ]

            # Add reverse edge
            orig_cost = next(c for _, c, _ in self.original_graph[u] if _ == v)
            if v not in self.adj_list:
                self.adj_list[v] = []
            self.adj_list[v].append((u, -orig_cost, 1))


from collections import defaultdict, deque
import heapq
from typing import Dict, List, Set, Tuple, Optional


class Graph:
    def __init__(self):
        self.edges: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
        self.nodes: Set[int] = set()
        self.source: Optional[int] = None
        self.sink: Optional[int] = None

    def add_edge(self, u: int, v: int, cost: float, capacity: int = 1):
        self.edges[u].append((v, cost, capacity))
        self.nodes.add(u)
        self.nodes.add(v)

    def set_terminals(self, source: int, sink: int):
        self.source = source
        self.sink = sink

    @classmethod
    def from_arc_dict(
        cls, arc_dict: Dict[int, List[Tuple[int, float, int]]], source: int, sink: int
    ) -> "Graph":
        """
        Create a graph from a dictionary of arcs in the format:
        {from_node: [(to_node, cost, capacity), ...]}
        """
        graph = cls()

        # Add all edges
        for from_node, edges in arc_dict.items():
            for to_node, cost, capacity in edges:
                graph.add_edge(from_node, to_node, cost, capacity)

        # Set source and sink
        graph.set_terminals(source, sink)
        return graph


class muSSP:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.residual_graph = defaultdict(list)
        self.flow = defaultdict(int)
        self.paths = []  # Store all paths found
        self.path_costs = []  # Store the cost of each path

    def _build_residual_graph(self):
        """Initialize residual graph with original edges."""
        for u in self.graph.edges:
            for v, cost, cap in self.graph.edges[u]:
                # Forward edge
                self.residual_graph[u].append((v, cost, cap))
                # Backward edge
                self.residual_graph[v].append((u, -cost, 0))

    def _find_shortest_path(self) -> Tuple[Optional[List[int]], float]:
        """
        Find shortest path using Dijkstra's algorithm.
        Returns (path_nodes, path_cost) where path_nodes is the list of nodes in the path
        """
        distances = {node: float("inf") for node in self.graph.nodes}
        distances[self.graph.source] = 0
        predecessors = {}

        pq = [(0, self.graph.source)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == self.graph.sink:
                break

            for next_node, cost, cap in self.residual_graph[current]:
                if cap > 0 and next_node not in visited:
                    new_dist = current_dist + cost
                    if new_dist < distances[next_node]:
                        distances[next_node] = new_dist
                        predecessors[next_node] = current
                        heapq.heappush(pq, (new_dist, next_node))

        # Reconstruct path if sink was reached
        if self.graph.sink in predecessors:
            path = []
            current = self.graph.sink
            while current != self.graph.source:
                path.append(current)
                current = predecessors[current]
            path.append(self.graph.source)
            path.reverse()
            return path, distances[self.graph.sink]
        return None, float("inf")

    def _augment_flow(self, path: List[int]):
        """Augment flow along the path and update residual graph."""
        # Find the edges and their costs along the path
        path_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for next_node, cost, cap in self.residual_graph[u]:
                if next_node == v:
                    path_edges.append((u, v, cost))
                    break

        # Update residual graph
        for u, v, _ in path_edges:
            # Update forward edge
            for i, (next_node, cost, cap) in enumerate(self.residual_graph[u]):
                if next_node == v:
                    self.residual_graph[u][i] = (next_node, cost, cap - 1)
                    break

            # Update backward edge
            for i, (next_node, cost, cap) in enumerate(self.residual_graph[v]):
                if next_node == u:
                    self.residual_graph[v][i] = (next_node, cost, cap + 1)
                    break

            self.flow[(u, v)] += 1

    def find_all_shortest_paths(self) -> List[Tuple[List[int], float]]:
        """
        Find all shortest paths in order from source to sink.
        Returns list of (path, cost) pairs.
        """
        self._build_residual_graph()

        while True:
            path, cost = self._find_shortest_path()
            if not path:
                break

            # Store the path and its cost
            self.paths.append(path)
            self.path_costs.append(cost)

            # Update residual graph for next iteration
            self._augment_flow(path)

        return list(zip(self.paths, self.path_costs))


import numpy as np
class SubgradientDescent:
    def __init__(self, num_nodes: int, upper_bound: float):
        self.num_nodes = num_nodes
        self.upper_bound = float(upper_bound)
        self.best_lower_bound = float('-inf')
        self.iteration = 0
        self.dual_vars = np.zeros(num_nodes)

    def calculate_lagrangian_value(self, paths: List[Tuple[List[int], float]]) -> Tuple[float, Dict[int, int]]:
        """
        Calculate L(λ) = min ∑a∈A caya + ∑j λj(bj - ∑a∈A gj(a)ya)
        """
        lagrangian_value = 0
        visits = defaultdict(int)
        
        # Path costs minus dual contribution
        for path, cost in paths:
            # Original cost
            lagrangian_value += cost
            
            # Count visits for each node (gj(a) in paper)
            for node in path[1:-1]:  # Skip source/sink
                visits[node] += 1
                # Subtract dual contribution
                lagrangian_value -= self.dual_vars[node - 1]
        
        # Add ∑j λj*bj (all bj = 1 for node covering constraints)
        lagrangian_value += np.sum(self.dual_vars)
        
        return lagrangian_value, visits

    def compute_subgradient(self, visits: Dict[int, int]) -> np.ndarray:
        """
        Compute subgradient γk where γkj = bj - ∑a∈A gj(a)ya
        bj = 1 for all j (each node must be visited once)
        """
        # Start with bj = 1
        subgradient = np.ones(self.num_nodes)
        
        # Subtract actual visits
        for node, count in visits.items():
            if 1 <= node <= self.num_nodes:
                subgradient[node - 1] -= count
                
        return subgradient

    def compute_step_size(self, subgradient: np.ndarray, lagrangian_value: float) -> float:
        """
        Compute Polyak step size: αk = (ψ* - v(λk)) / ||γk||²
        Where ψ* = UB * (1 + 5/(100+k))
        """
        try:
            # Estimate optimal value as per paper
            psi_star = self.upper_bound * (1.0 + 5.0/(100.0 + self.iteration))
            
            # Compute squared L2 norm of subgradient
            subgradient_norm_sq = np.sum(subgradient * subgradient)
            
            if subgradient_norm_sq > 1e-10:
                # Paper's formula
                step_size = (psi_star - lagrangian_value) / subgradient_norm_sq
                return max(0.0, step_size)
                
        except Exception as e:
            print(f"Error in compute_step_size: {e}")
            
        return 0.0

    def update_duals(self, paths: List[Tuple[List[int], float]]) -> np.ndarray:
        """
        Update dual variables using subgradient method
        λk+1 = λk + αk * γk
        """
        # Calculate Lagrangian value and visits
        lagrangian_value, visits = self.calculate_lagrangian_value(paths)
        
        # Compute subgradient
        subgradient = self.compute_subgradient(visits)
        
        # Compute step size
        step_size = self.compute_step_size(subgradient, lagrangian_value)
        
        # Update dual variables
        self.dual_vars = self.dual_vars + step_size * subgradient
        
        # Project to non-negative orthant
        self.dual_vars = np.maximum(0.0, self.dual_vars)
        
        # Update iteration count and best bound
        self.iteration += 1
        self.best_lower_bound = max(self.best_lower_bound, lagrangian_value)
        
        return self.dual_vars