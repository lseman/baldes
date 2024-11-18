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

    def _get_travel_time(self, from_node: int, to_node: int) -> float:
        """Cached travel time calculation"""
        key = (from_node, to_node)
        if key not in self._travel_times:
            self._travel_times[key] = self.solver.calculate_travel_time(
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
            path_key = tuple(path.nodes_covered)
            if path_key in self._feasibility_cache:
                if self._feasibility_cache[path_key]:
                    valid_paths.append(path.nodes_covered)
            elif self._quick_feasibility_check(path.nodes_covered):
                valid_paths.append(path.nodes_covered)
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


class PathDecomposition:
    """Class to handle path decomposition of network flow solutions"""

    def __init__(self, solver):
        self.solver = solver

    def decompose_solution(self, paths) -> List[List[int]]:
        """
        Decompose network flow solution into paths following flow decomposition theorem
        Args:
            paths: Solution from phaseFour() containing flow values
        Returns:
            List of paths representing flow decomposition
        """
        decomposed_paths = []

        # Convert to flow representation if needed
        flow_graph = self._build_flow_graph(paths)

        # While there is remaining flow
        while self._has_remaining_flow(flow_graph):
            # Find a path from source to sink with positive flow
            path = self._find_positive_flow_path(flow_graph)
            if not path:
                break

            # Find minimum flow along path
            min_flow = self._get_min_flow(flow_graph, path)

            # Subtract flow along path
            self._subtract_flow(flow_graph, path, min_flow)

            # Add path to decomposition
            decomposed_paths.append(path)

        return decomposed_paths

    def _build_flow_graph(self, paths):
        """Convert paths to residual flow graph"""
        flow_graph = defaultdict(lambda: defaultdict(float))

        for path in paths:
            sequence = path
            for i in range(len(sequence) - 1):
                u, v = sequence[i], sequence[i + 1]
                flow_graph[u][v] += 1  # Assuming unit flow per path

        return flow_graph

    def _has_remaining_flow(self, flow_graph):
        """Check if graph has any remaining positive flow"""
        return any(
            any(f > 0.001 for f in flows.values()) for flows in flow_graph.values()
        )

    def _find_positive_flow_path(self, flow_graph):
        """Find a path from source (0) to sink (101) with positive flow"""

        def dfs(node, path, visited):
            # Found valid path when we reach end depot (101) and have visited customers
            if node == 101 and len(path) > 2:
                # Verify it starts from depot 0
                if path[0] == 0:
                    return path
                return None

            # Don't add depots to visited set
            if node not in {0, 101}:
                if node in visited:
                    return None
                visited.add(node)

            # Try all possible next nodes with positive flow
            for next_node, flow in flow_graph[node].items():
                if flow > 0.001:
                    # Allow visiting end depot (101) anytime
                    if next_node == 101 or next_node not in visited:
                        new_path = path + [next_node]
                        result = dfs(next_node, new_path, visited.copy())
                        if result:
                            return result

            return None

        # Start DFS from start depot (0)
        initial_path = [0]
        return dfs(0, initial_path, set())

    def _get_min_flow(self, flow_graph, path):
        """Get minimum flow value along path"""
        min_flow = float("inf")
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            min_flow = min(min_flow, flow_graph[u][v])
        return min_flow

    def _subtract_flow(self, flow_graph, path, flow_value):
        """Subtract flow value along path"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            flow_graph[u][v] -= flow_value


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
