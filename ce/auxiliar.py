from typing import Dict, Set, Tuple, List
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
