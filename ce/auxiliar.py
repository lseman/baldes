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

    def __init__(self, conflict_type: str, sequence: List[int], state: Dict):
        self.type = conflict_type
        self.sequence = sequence
        self.state = state

    def __str__(self):
        return (
            f"Conflict(type={self.type}, sequence={self.sequence}, state={self.state})"
        )


from typing import List, Dict, Optional, Set
from collections import defaultdict

from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Conflict:
    """Data class to represent conflicts"""

    type: str
    sequence: List[int]
    details: Dict


from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from copy import deepcopy


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

    def find_conflict(self, path: List[int]) -> Optional[Dict]:
        """Enhanced conflict detection with state preservation"""
        # Save current state
        self._current_state = {
            "path": path[:],
            "visited": set(),
            "current_load": 0,
            "current_time": 0,
        }

        # Check for cycles using set for O(1) lookup
        for i, node in enumerate(path):
            if node != 0 and node in self._current_state["visited"]:
                cycle_start = path.index(node)
                cycle_sequence = path[cycle_start : i + 1]
                self.refinement_stats["cycle_conflicts"] += 1
                return {
                    "type": "cycle",
                    "sequence": cycle_sequence,
                    "details": {"cycle_node": node},
                }
            self._current_state["visited"].add(node)

        # Progressive resource checking with early exits
        current_load = 0
        time = 0
        violation_start = 0

        for i, node in enumerate(path):
            # Update load
            current_load += self._node_demands[node]

            # Quick capacity check
            if current_load > self.solver.vehicle_capacity:
                subsequence = path[violation_start : i + 1]
                self.refinement_stats["capacity_conflicts"] += 1
                return {
                    "type": "capacity",
                    "sequence": subsequence,
                    "details": {"load": current_load},
                }

            # Update time
            if i > 0:
                time += self._node_durations[path[i - 1]]
                time += self._get_travel_time(path[i - 1], node)
                time = max(time, self._node_time_windows[node][0])

                # Time window check
                if time > self._node_time_windows[node][1]:
                    subsequence = path[violation_start : i + 1]
                    self.refinement_stats["time_window_conflicts"] += 1
                    return {
                        "type": "time_window",
                        "sequence": subsequence,
                        "details": {
                            "time": time,
                            "latest": self._node_time_windows[node][1],
                        },
                    }

            # Reset violation tracking if still feasible
            if i > 0 and time <= self._node_time_windows[node][1]:
                violation_start = i
                current_load = self._node_demands[node]
                time = max(
                    self._node_time_windows[node][0], self._get_travel_time(0, node)
                )

        return None

    def refine_conflict(self, conflict: Dict) -> bool:
        """Refine conflict with state preservation"""
        try:
            sequence = conflict["sequence"]

            # Save state before refinement
            self._state_history.append(deepcopy(self._current_state))

            # Fast path for cycle conflicts
            if conflict["type"] == "cycle":
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

    def _refine_cycle(self, conflict: Dict) -> bool:
        """Optimized cycle refinement"""
        sequence = conflict["sequence"]
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
