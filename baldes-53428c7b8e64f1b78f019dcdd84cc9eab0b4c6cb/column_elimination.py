import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import pybaldes
from collections import defaultdict

@dataclass 
class VRPTWInstance:
    """VRPTW problem instance data"""
    num_nodes: int
    demands: np.ndarray  
    time_windows: np.ndarray
    distances: np.ndarray
    capacities: float
    service_times: np.ndarray
    
    def get_time_horizon(self) -> float:
        return np.max(self.time_windows[:, 1])

class SubgradientState:
    """Maintains state for subgradient method"""
    def __init__(self, num_nodes: int):
        self.iteration = 0
        self.step_size = 1.0
        self.best_lb = float('-inf')
        self.duals = np.zeros(num_nodes)
        self.best_duals = np.zeros(num_nodes)
        
    def update(self, routes: List[List[int]], lb: float):
        """Update duals using subgradient method"""
        if lb > self.best_lb:
            self.best_lb = lb
            self.best_duals = self.duals.copy()
            
        # Compute subgradients
        coverage = np.zeros(len(self.duals))
        for route in routes:
            for node in route[1:-1]:  # Exclude depot
                coverage[node] += 1
                
        # Coverage constraints require each node visited exactly once
        subgradients = coverage - 1
        
        # Update step size
        self.step_size *= 0.95  # Geometric decay
        
        # Update duals
        self.duals += self.step_size * subgradients
        self.duals = np.maximum(self.duals, 0)  # Project to nonnegative orthant
        
        self.iteration += 1

class ConflictManager:
    """Manages conflict detection and refinement"""
    def __init__(self, instance: VRPTWInstance):
        self.instance = instance
        self.ng_neighbors: Dict[int, Set[int]] = {}
        self.initialize_ng_neighbors(5)  # Start with small neighborhoods
        
    def initialize_ng_neighbors(self, size: int):
        """Initialize ng-route neighborhoods based on distances"""
        for i in range(self.instance.num_nodes):
            # Get closest nodes based on distances
            distances = self.instance.distances[i]
            nearest = np.argsort(distances)[1:size+1]  # Exclude self
            self.ng_neighbors[i] = set(nearest)
            
    def find_conflicts(self, route: List[int]) -> List[Tuple[int, int]]:
        """Find conflicts in route based on paper's criteria"""
        conflicts = []
        
        # Time window conflicts
        curr_time = self.instance.time_windows[0][0]
        for i in range(len(route)-1):
            curr = route[i]
            next = route[i+1]
            
            curr_time += self.instance.distances[curr][next]
            curr_time = max(curr_time, self.instance.time_windows[next][0])
            
            if curr_time > self.instance.time_windows[next][1]:
                conflicts.append((curr, next))
                
            curr_time += self.instance.service_times[next]
            
        # Capacity conflicts
        load = 0
        for i, node in enumerate(route):
            load += self.instance.demands[node]
            if load > self.instance.capacities:
                conflicts.append((route[i-1], node))
                
        # ng-route conflicts (elementarity)
        for i, node in enumerate(route[1:-1]):  # Skip depot
            for prev_idx in range(max(1, i-len(self.ng_neighbors[node])), i):
                prev = route[prev_idx]
                if prev in self.ng_neighbors[node]:
                    conflicts.append((route[prev_idx-1], prev))
                    
        return conflicts
        
    def refine_ng_neighbors(self, conflicts: List[Tuple[int, int]]):
        """Refine ng-neighborhoods based on conflicts"""
        for (u, v) in conflicts:
            if u != 0:  # Don't add depot to neighborhoods
                self.ng_neighbors[v].add(u)
            if v != 0:
                self.ng_neighbors[u].add(v)

class VariableFixing:
    """Handles reduced cost variable fixing"""
    def __init__(self, instance: VRPTWInstance):
        self.instance = instance
        self.fixed_arcs: Set[Tuple[int, int]] = set()
        
    def compute_shortest_paths(self, duals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute shortest paths with reduced costs"""
        n = self.instance.num_nodes
        forward_dist = np.full((n,), np.inf)
        backward_dist = np.full((n,), np.inf)
        
        # Forward pass from depot
        forward_dist[0] = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    reduced_cost = (self.instance.distances[i][j] - 
                                  (duals[j] if j != 0 else 0))
                    if forward_dist[i] + reduced_cost < forward_dist[j]:
                        forward_dist[j] = forward_dist[i] + reduced_cost
                        
        # Backward pass to depot  
        backward_dist[0] = 0
        for i in range(n-1, -1, -1):
            for j in range(n):
                if i != j:
                    reduced_cost = (self.instance.distances[j][i] - 
                                  (duals[i] if i != 0 else 0))
                    if backward_dist[i] + reduced_cost < backward_dist[j]:
                        backward_dist[j] = backward_dist[i] + reduced_cost
                        
        return forward_dist, backward_dist
        
    def fix_variables(self, duals: np.ndarray, ub: float, lb: float):
        """Fix variables using reduced cost fixing"""
        forward_dist, backward_dist = self.compute_shortest_paths(duals)
        
        gap = ub - lb
        if gap <= 0:
            return
            
        # Check each arc
        for i in range(self.instance.num_nodes):
            for j in range(self.instance.num_nodes):
                if i != j and (i,j) not in self.fixed_arcs:
                    reduced_cost = (self.instance.distances[i][j] - 
                                  (duals[j] if j != 0 else 0))
                    
                    # If reduced cost path through arc exceeds gap, fix to zero
                    path_cost = (forward_dist[i] + reduced_cost + 
                               backward_dist[j])
                    if path_cost > gap:
                        self.fixed_arcs.add((i,j))
                        
    def is_fixed(self, i: int, j: int) -> bool:
        """Check if arc is fixed to zero"""
        return (i,j) in self.fixed_arcs

class BucketGraphAdapter:
    """Adapter for using bucket graph labeling"""
    def __init__(self, instance: VRPTWInstance, bucket_interval: float = 1.0):
        self.instance = instance
        
        # Create VRPNodes
        self.nodes = []
        for i in range(instance.num_nodes):
            node = pybaldes.VRPNode()
            node.id = i
            node.demand = instance.demands[i]
            node.start_time = instance.time_windows[i][0]
            node.end_time = instance.time_windows[i][1]
            node.duration = instance.service_times[i]
            self.nodes.append(node)
            
        # Initialize bucket graph
        self.bucket_graph = pybaldes.BucketGraph(
            self.nodes, 
            instance.get_time_horizon(),
            bucket_interval
        )
        
        # Setup options
        self.options = pybaldes.BucketOptions()
        self.options.depot = 0
        self.options.end_depot = 0
        self.options.resources = 2  # Time and load
        self.options.max_path_size = instance.num_nodes
        self.bucket_graph.setOptions(self.options)
        
        self.bucket_graph.set_distance_matrix(instance.distances)
        
    def solve_relaxed(self, 
                     ng_neighbors: Dict[int, Set[int]],
                     duals: Optional[np.ndarray] = None,
                     fixed_arcs: Optional[Set[Tuple[int, int]]] = None) -> List[List[int]]:
        """Solve relaxed pricing problem"""
        self.bucket_graph.reset_pool()
        
        # Update ng neighborhoods in bucket graph
        # (You'll need to add this functionality to your C++ code)
        self.bucket_graph.update_ng_neighbors(ng_neighbors)
        
        # Set dual values if provided
        if duals is not None:
            self.bucket_graph.set_duals(duals)
            
        # Set fixed arcs if provided
        if fixed_arcs is not None:
            self.bucket_graph.set_fixed_arcs(fixed_arcs)
            
        # Solve and get paths
        paths = self.bucket_graph.solve()
        return paths

class ColumnEliminationVRPTW:
    """Column elimination for VRPTW"""
    def __init__(self, instance: VRPTWInstance):
        self.instance = instance
        self.bucket_adapter = BucketGraphAdapter(instance)
        self.conflict_manager = ConflictManager(instance)
        self.variable_fixing = VariableFixing(instance)
        self.subgradient = SubgradientState(instance.num_nodes)
        
    def solve(self, 
             max_iterations: int = 1000,
             time_limit: float = 3600) -> List[List[int]]:
        """Main column elimination solve loop"""
        best_routes = []
        best_ub = float('inf')
        
        for iter in range(max_iterations):
            # 1. Solve relaxed problem
            routes = self.bucket_adapter.solve_relaxed(
                self.conflict_manager.ng_neighbors,
                self.subgradient.duals,
                self.variable_fixing.fixed_arcs
            )
            
            # 2. Update bounds
            lb = self._compute_lb(routes)
            if lb > self.subgradient.best_lb:
                self.subgradient.best_lb = lb
                
            # Check primal solution
            cost = self._compute_solution_cost(routes)
            if cost < best_ub and self._is_feasible_solution(routes):
                best_routes = routes
                best_ub = cost
                
                # Try variable fixing
                self.variable_fixing.fix_variables(
                    self.subgradient.best_duals,
                    best_ub,
                    self.subgradient.best_lb
                )
                
            # 3. Update subgradient method
            self.subgradient.update(routes, lb)
            
            # 4. Find and refine conflicts
            all_conflicts = []
            for route in routes:
                conflicts = self.conflict_manager.find_conflicts(route)
                all_conflicts.extend(conflicts)
                
            if not all_conflicts:
                break
                
            self.conflict_manager.refine_ng_neighbors(all_conflicts)
            
        return best_routes
    
    def _compute_lb(self, routes: List[List[int]]) -> float:
        """Compute lower bound from relaxed solution"""
        # Reduced cost of routes plus sum of duals
        lb = sum(self._compute_reduced_cost(route) 
                for route in routes)
        lb += sum(self.subgradient.duals)  # Add dual values
        return lb
    
    def _compute_reduced_cost(self, route: List[int]) -> float:
        """Compute reduced cost of a route"""
        cost = self._compute_route_cost(route)
        # Subtract dual values for visited nodes
        for node in route[1:-1]:  # Exclude depot
            cost -= self.subgradient.duals[node]
        return cost
    
    def _compute_route_cost(self, route: List[int]) -> float:
        """Compute cost of a route"""
        cost = 0.0
        for i in range(len(route)-1):
            cost += self.instance.distances[route[i]][route[i+1]]
        return cost
    
    def _compute_solution_cost(self, routes: List[List[int]]) -> float:
        """Compute total solution cost"""
        return sum(self._compute_route_cost(route) for route in routes)
        
    def _is_feasible_solution(self, routes: List[List[int]]) -> bool:
        """Check if solution is feasible"""
        # Check each route
        if any(self.conflict_manager.find_conflicts(route) 
               for route in routes):
            return False
            
        # Check customer coverage
        visited = set()
        for route in routes:
            for node in route[1:-1]:
                if node in visited:
                    return False
                visited.add(node)
                
        return len(visited) == self.instance.num_nodes - 1