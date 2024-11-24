
from typing import List, Dict, Set, Tuple, Optional
import heapq
from collections import defaultdict, namedtuple

class VRPTWLagrangianSolver:
    def __init__(self, diagram, distances, num_vehicles):
        self.diagram = diagram
        self.distances = distances
        self.num_vehicles = num_vehicles
        self.locations = diagram.locations
        self.root = diagram.root
        self.terminal = diagram.terminal
        self.depot_id = diagram.depot_id
        self.extend_state = diagram.extend_state
        self.nodes = diagram.nodes

        # Initialize multipliers conservatively
        self.lambda_multipliers = {
            i: self.distances[(0, i)] * self.diagram.locations[i].demand / self.diagram.capacity 
            for i in range(len(diagram.locations)) 
            if i != diagram.depot_id
        }
        
        # Best bounds and solutions tracking
        self.best_lb = float('inf')
        self.best_ub = float('inf')
        self.iteration = 0
        
        # Subgradient parameters
        self.alpha = 0.1  # Initial step size
        self.beta = 0.98  # Step size reduction factor

    def get_arc_cost(self, from_idx: int, to_idx: int, loc_id: int) -> float:
        """Get modified arc cost including Lagrangian multipliers"""
        from_state = self.diagram.nodes[from_idx].last_visited
        cost = self.distances[(from_state, loc_id)]
        if loc_id != self.diagram.depot_id:
            cost -= self.lambda_multipliers[loc_id]
        return cost

    def find_shortest_path(self, used_arcs: Set[Tuple[int, int, int]] = None) -> Optional[List[Tuple[int, int, int]]]:
        """Find shortest path using modified Dijkstra with path requirements"""
        if used_arcs is None:
            used_arcs = set()
            
        dist = {i: float('inf') for i in range(len(self.diagram.nodes))}
        dist[self.diagram.root] = 0
        pred = {i: None for i in range(len(self.diagram.nodes))}
        visited_customers = {i: set() for i in range(len(self.diagram.nodes))}
        visited_customers[self.diagram.root] = set()
        
        pq = [(0, self.diagram.root)]
        
        while pq:
            d, u = heapq.heappop(pq)
            current_state = self.diagram.nodes[u]
            
            # Don't allow return to depot unless we've visited at least one customer
            if u == self.diagram.terminal and not visited_customers[pred[u]]:
                continue
                
            # If we've reached terminal with some customers, we're done
            if u == self.diagram.terminal and visited_customers[pred[u]]:
                path = []
                curr = u
                while pred[curr] is not None:
                    prev = pred[curr]
                    arc = next((a for a in self.diagram.arcs 
                              if a[0] == prev and a[1] == curr and a not in used_arcs), None)
                    if arc:
                        path.append(arc)
                    curr = prev
                return path[::-1] if path else None
            
            # Examine outgoing arcs
            for from_idx, to_idx, loc_id in self.diagram.arcs:
                if from_idx != u or (from_idx, to_idx, loc_id) in used_arcs:
                    continue
                    
                # Get the next state
                next_state = self.diagram.nodes[to_idx]
                
                # Calculate reduced cost
                cost = self.get_arc_cost(from_idx, to_idx, loc_id)
                
                # Only allow return to depot if we've visited some customers
                if to_idx == self.diagram.terminal and not visited_customers[u]:
                    continue
                
                if dist[u] + cost < dist[to_idx]:
                    dist[to_idx] = dist[u] + cost
                    pred[to_idx] = u
                    # Track visited customers
                    new_visited = visited_customers[u].copy()
                    if loc_id != self.diagram.depot_id:
                        new_visited.add(loc_id)
                    visited_customers[to_idx] = new_visited
                    heapq.heappush(pq, (dist[to_idx], to_idx))
        
        return None

    def solve_subproblem(self) -> Tuple[float, List[List[Tuple[int, int, int]]]]:
        """Solve the subproblem with on-demand diagram construction."""
        paths = []
        total_cost = 0
        used_arcs = set()
        unvisited = set(range(1, len(self.locations)))

        print("\nSolving subproblem")
        print(f"Customers to visit: {unvisited}")

        while unvisited:
            path = []
            current_idx = self.root
            while current_idx != self.terminal:
                print(f"Finding path from {current_idx} to terminal")
                # Find the best feasible transition
                best_next = None
                best_cost = float('inf')
                
                for loc_id in unvisited | {self.depot_id}:
                    next_idx = self.extend_state(current_idx, loc_id)
                    if next_idx == -1:
                        continue  # Infeasible transition
                    
                    arc_cost = self.distances[(self.nodes[current_idx].last_visited, loc_id)]
                    if arc_cost < best_cost:
                        best_cost = arc_cost
                        best_next = (next_idx, loc_id)
                
                if not best_next:
                    break  # No feasible transition
                
                next_idx, loc_id = best_next
                path.append((current_idx, next_idx, loc_id))
                current_idx = next_idx

                # Stop if returning to depot
                if loc_id == self.depot_id:
                    break
            
            # Finalize path
            path_customers = {loc_id for _, _, loc_id in path if loc_id != self.depot_id}
            if path_customers:
                paths.append(path)
                total_cost += sum(self.distances[(self.nodes[from_idx].last_visited, loc_id)]
                                  for from_idx, _, loc_id in path)
                unvisited -= path_customers
        
        return total_cost, paths


    def update_multipliers(self, paths: List[List[Tuple[int, int, int]]], UB: float, LB: float):
        """Update multipliers using Polyak step size."""
        visits = defaultdict(int)
        for path in paths:
            for _, _, loc_id in path:
                if loc_id != self.diagram.depot_id:
                    visits[loc_id] += 1
        
        # Calculate subgradients
        subgradients = {}
        for loc_id in self.lambda_multipliers:
            subgradients[loc_id] = 1 - visits[loc_id]  # Target is one visit per customer
        
        # Compute Polyak step size
        norm_sq = sum(g**2 for g in subgradients.values())
        if norm_sq > 0:
            step_size = (UB - LB) / norm_sq
        else:
            step_size = 0  # Avoid division by zero
        
        # Update multipliers
        for loc_id in self.lambda_multipliers:
            self.lambda_multipliers[loc_id] = max(
                0,  # Multipliers must remain non-negative
                self.lambda_multipliers[loc_id] + step_size * subgradients[loc_id]
            )

