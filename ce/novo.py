import heapq
import numpy as np
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import gurobipy as gp
from gurobipy import GRB

@dataclass
class Location:
    id: int
    demand: float
    is_depot: bool = False

@dataclass
class State:
    """State for the decision diagram, using q-route relaxation"""
    last_visited: int  # Last visited location
    weight: float      # Accumulated weight
    visited: Set[int]  # Set of visited locations

    def __str__(self):
        return f"State(last={self.last_visited}, weight={self.weight}, visited={self.visited})"

class DecisionDiagram:
    def __init__(self, locations: List[Location], capacity: float, num_vehicles: int):
        self.locations = locations
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.nodes = []  # List of nodes in the diagram
        self.arcs = []   # List of arcs in the diagram
        
        # Initialize root and terminal nodes
        self.depot_id = next(loc.id for loc in locations if loc.is_depot)
        
        # Create root node with initial state
        root_state = State(last_visited=self.depot_id, weight=0, visited=set([self.depot_id]))
        self.root = 0  # Index of root node
        self.nodes.append(root_state)
        
        # Create terminal node state
        terminal_state = State(last_visited=self.depot_id, weight=0, visited=set([self.depot_id]))
        self.terminal = 1  # Index of terminal node
        self.nodes.append(terminal_state)
        
        # Build initial diagram
        self._build_initial_diagram()

    def _build_initial_diagram(self):
        current_layer = {self.root}
        next_layer = set()
        
        while current_layer:
            for node_idx in current_layer:
                state = self.nodes[node_idx]
                
                # Try extending paths from current state
                for loc in self.locations:
                    if loc.id == self.depot_id:
                        # Only allow return to depot if we've visited at least one location
                        if len(state.visited) > 1:
                            self.arcs.append((node_idx, self.terminal, self.depot_id))
                        continue
                        
                    # Skip if location was just visited
                    if loc.id == state.last_visited:
                        continue
                        
                    # Check capacity constraint
                    new_weight = state.weight + loc.demand
                    if new_weight <= self.capacity:
                        # Create new state with updated visited set
                        new_visited = state.visited.copy()
                        new_visited.add(loc.id)
                        new_state = State(last_visited=loc.id, weight=new_weight, visited=new_visited)
                        
                        # Check if state already exists
                        existing_idx = next((i for i, s in enumerate(self.nodes) 
                                          if s.last_visited == new_state.last_visited 
                                          and abs(s.weight - new_state.weight) < 1e-6
                                          and s.visited == new_state.visited), None)
                        
                        if existing_idx is None:
                            new_idx = len(self.nodes)
                            self.nodes.append(new_state)
                            next_layer.add(new_idx)
                            # Add arc using loc.id
                            self.arcs.append((node_idx, new_idx, loc.id))
                        else:
                            new_idx = existing_idx
                            next_layer.add(existing_idx)
                            # Add arc using loc.id
                            self.arcs.append((node_idx, new_idx, loc.id))
            
            current_layer = next_layer
            next_layer = set()

    def eliminate_cycle(self, path: List[int]) -> bool:
        """
        Implement Algorithm 1 from the paper for conflict refinement.
        Returns True if a cycle was found and eliminated, False otherwise.
        """
        if not path:
            return False

        # Create copy of current diagram to work on
        new_nodes = self.nodes.copy()
        new_arcs = self.arcs.copy()
        
        # Start from root node
        current_state = self.root
        
        # Process each element in the path
        for j, element in enumerate(path):
            # Find all possible subsequences from root to current state
            subsequences = self._find_subsequences(current_state)
            
            # Get feasible decisions from all possible predecessor states
            feasible_decisions = set()
            for subseq in subsequences:
                # Get state in original diagram P for this subsequence
                p_state = self._get_state_in_P(subseq)
                if p_state is not None:
                    # Add feasible decisions from this state
                    for loc in self.locations:
                        if self._is_feasible_transition(p_state, loc.id):
                            feasible_decisions.add(loc.id)

            # If current element is not feasible from any valid predecessor
            if element not in feasible_decisions:
                # Remove transition with this element
                new_arcs = [(from_idx, to_idx, loc_id) for from_idx, to_idx, loc_id in new_arcs 
                           if not (from_idx == current_state and loc_id == element)]
                return True
            
            # Create new state to isolate this path
            new_state = State(
                last_visited=element,
                weight=sum(self.locations[loc].demand for loc in path[:j+1] if loc != self.depot_id),
                visited=set(path[:j+1])
            )
            
            # Find next state in current diagram
            next_state = None
            for from_idx, to_idx, loc_id in self.arcs:
                if from_idx == current_state and loc_id == element:
                    next_state = to_idx
                    break
                    
            if next_state is None:
                continue
                
            # Copy all transitions from next_state to new_state
            new_state_idx = len(new_nodes)
            new_nodes.append(new_state)
            
            for from_idx, to_idx, loc_id in self.arcs:
                if from_idx == next_state:
                    new_arcs.append((new_state_idx, to_idx, loc_id))
                    
            # Update transition to use new state
            new_arcs = [(from_idx, to_idx if to_idx != next_state else new_state_idx, loc_id) 
                       for from_idx, to_idx, loc_id in new_arcs]
            
            current_state = new_state_idx

        # Update diagram
        self.nodes = new_nodes
        self.arcs = new_arcs
        
        # Clean up unreachable nodes
        self._clean_unreachable_nodes()
        
        return False

    def _find_subsequences(self, state_idx: int) -> List[List[int]]:
        """Find all possible subsequences from root to given state"""
        sequences = []
        
        def dfs(current: int, path: List[int]):
            if current == state_idx:
                sequences.append(path)
                return
                
            for from_idx, to_idx, loc_id in self.arcs:
                if from_idx == current:
                    dfs(to_idx, path + [loc_id])
                    
        dfs(self.root, [])
        return sequences

    def _get_state_in_P(self, sequence: List[int]) -> Optional[State]:
        """
        Get corresponding state in original problem P for a sequence.
        Returns None if sequence is not feasible in P.
        """
        current_state = State(
            last_visited=self.depot_id,
            weight=0,
            visited=set([self.depot_id])
        )
        
        for loc_id in sequence:
            # Check if transition is feasible in P
            if not self._is_feasible_transition(current_state, loc_id):
                return None
                
            # Update state
            current_state.last_visited = loc_id
            current_state.weight += self.locations[loc_id].demand if loc_id != self.depot_id else 0
            current_state.visited.add(loc_id)
            
        return current_state

    def _is_feasible_transition(self, state: State, loc_id: int) -> bool:
        """Check if transition is feasible in original problem P"""
        # Check if location already visited
        if loc_id in state.visited and loc_id != self.depot_id:
            return False
            
        # Check capacity constraint
        if loc_id != self.depot_id:
            new_weight = state.weight + self.locations[loc_id].demand
            if new_weight > self.capacity:
                return False
                
        return True

    def _clean_unreachable_nodes(self):
        """Remove nodes that are no longer reachable after refinement"""
        reachable = set()
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            if node not in reachable:
                reachable.add(node)
                # Add all nodes reachable from current node
                for _, to_idx, _ in self.arcs:
                    if to_idx not in reachable:
                        stack.append(to_idx)

        # Keep only arcs between reachable nodes
        self.arcs = [(from_idx, to_idx, loc_id) for from_idx, to_idx, loc_id in self.arcs 
                     if from_idx in reachable and to_idx in reachable]
        
        # Rebuild node list
        old_to_new = {}
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if i in reachable:
                old_to_new[i] = len(new_nodes)
                new_nodes.append(node)
                
        # Update arc indices
        self.arcs = [(old_to_new[from_idx], old_to_new[to_idx], loc_id) 
                     for from_idx, to_idx, loc_id in self.arcs]
        
        # Update root and terminal indices
        self.root = old_to_new[self.root]
        self.terminal = old_to_new[self.terminal]
        
        self.nodes = new_nodes

    def get_num_nodes(self):
        return len(self.nodes)
    
    def get_num_arcs(self):
        return len(self.arcs) 

class LagrangianSolver:
    def __init__(self, diagram, distances, num_vehicles):
        self.diagram = diagram
        self.distances = distances
        self.num_vehicles = num_vehicles
        # Initialize multipliers for non-depot locations
        self.lambda_multipliers = {i: 0.0 for i in range(len(diagram.locations)) 
                                 if i != diagram.depot_id}
        
    def get_arc_cost(self, from_idx: int, to_idx: int, loc_id: int) -> float:
        """Get modified arc cost using Lagrangian multipliers"""
        from_state = self.diagram.nodes[from_idx].last_visited
        cost = self.distances[(from_state, loc_id)]
        
        # Only apply multiplier for non-depot locations
        if loc_id != self.diagram.depot_id:
            cost -= self.lambda_multipliers[loc_id]
        return cost

    def find_shortest_path(self, used_arcs: Set[Tuple[int, int, int]] = None) -> Optional[List[Tuple[int, int, int]]]:
        """Find shortest path using muSSP algorithm"""
        if used_arcs is None:
            used_arcs = set()
            
        # Initialize distances and predecessors
        dist = {i: float('inf') for i in range(len(self.diagram.nodes))}
        dist[self.diagram.root] = 0
        pred = {i: None for i in range(len(self.diagram.nodes))}
        
        # Priority queue for Dijkstra
        pq = [(0, self.diagram.root)]
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u == self.diagram.terminal:
                # Reconstruct path
                path = []
                curr = u
                while pred[curr] is not None:
                    prev = pred[curr]
                    # Find the arc that was used
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
                    
                cost = self.get_arc_cost(from_idx, to_idx, loc_id)
                if dist[u] + cost < dist[to_idx]:
                    dist[to_idx] = dist[u] + cost
                    pred[to_idx] = u
                    heapq.heappush(pq, (dist[to_idx], to_idx))
        
        return None

    def solve_subproblem(self) -> Tuple[float, List[List[Tuple[int, int, int]]]]:
        """Solve Lagrangian subproblem to find K shortest paths"""
        paths = []
        total_cost = 0
        used_arcs = set()
        
        # Find K shortest paths
        for _ in range(self.num_vehicles):
            path = self.find_shortest_path(used_arcs)
            if not path:
                return float('inf'), []
                
            paths.append(path)
            used_arcs.update(path)
            
            # Add up original costs (without multipliers)
            for from_idx, to_idx, loc_id in path:
                from_state = self.diagram.nodes[from_idx].last_visited
                total_cost += self.distances[(from_state, loc_id)]
        
        return total_cost, paths

    def update_multipliers(self, paths: List[List[Tuple[int, int, int]]], step_size: float):
        """Update Lagrangian multipliers using subgradient method"""
        # Count visits to each location
        visits = defaultdict(int)
        for path in paths:
            for _, _, loc_id in path:
                if loc_id != self.diagram.depot_id:
                    visits[loc_id] += 1
        
        # Update multipliers using subgradient
        for loc_id in self.lambda_multipliers:
            # Subgradient is (1 - number of visits)
            subgradient = 1 - visits[loc_id]
            self.lambda_multipliers[loc_id] += step_size * subgradient


class ColumnElimination:
    def __init__(self, locations: List[Location], distances: Dict[Tuple[int, int], float],
                 capacity: float, num_vehicles: int):
        self.locations = locations
        self.distances = distances
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.diagram = DecisionDiagram(locations, capacity, num_vehicles)

    def extract_routes_from_flows(self, flow_values: Dict[int, float]) -> List[List[int]]:
        """Extract integer routes from fractional flow values"""
        routes = []
        nodes_to_visit = set(range(1, len(self.locations)))  # All non-depot locations
        remaining_flow = flow_values.copy()
        
        while nodes_to_visit and len(routes) < self.num_vehicles:
            route = []
            current = self.diagram.root
            path_flow = 1.0
            visited = set()
            
            while current != self.diagram.terminal:
                # Find best next arc
                best_arc = None
                best_flow = 0
                
                for i, (from_idx, to_idx, loc_id) in enumerate(self.diagram.arcs):
                    if i in remaining_flow and from_idx == current:
                        # Prefer unvisited locations that need to be visited
                        if loc_id in nodes_to_visit and loc_id not in visited:
                            flow = remaining_flow[i]
                            if flow > best_flow:
                                best_flow = flow
                                best_arc = (i, to_idx, loc_id)
                
                # If no unvisited locations, try any location
                if best_arc is None:
                    for i, (from_idx, to_idx, loc_id) in enumerate(self.diagram.arcs):
                        if i in remaining_flow and from_idx == current:
                            flow = remaining_flow[i]
                            if flow > best_flow:
                                best_flow = flow
                                best_arc = (i, to_idx, loc_id)
                
                if best_arc is None:
                    # Try to return to depot if possible
                    for i, (from_idx, to_idx, loc_id) in enumerate(self.diagram.arcs):
                        if from_idx == current and to_idx == self.diagram.terminal:
                            best_arc = (i, to_idx, self.diagram.depot_id)
                            break
                    if best_arc is None:
                        break
                
                arc_idx, next_node, loc_id = best_arc
                
                # Update flows and visited locations
                if arc_idx in remaining_flow:
                    path_flow = min(path_flow, remaining_flow[arc_idx])
                    remaining_flow[arc_idx] -= path_flow
                    if remaining_flow[arc_idx] < 0.01:
                        del remaining_flow[arc_idx]
                
                if loc_id != self.diagram.depot_id:
                    route.append(loc_id)
                    visited.add(loc_id)
                    nodes_to_visit.discard(loc_id)
                
                current = next_node
            
            if route:
                routes.append(route)
        
        return routes

    def extract_routes_from_flows(self, flow_values: Dict[int, float]) -> List[List[int]]:
        """Extract optimal routes from flow values"""
        def get_path_cost(path: List[int]) -> float:
            """Calculate cost of a path"""
            if not path:
                return 0
            cost = 0
            prev = self.diagram.depot_id
            for loc in path:
                cost += self.distances[(prev, loc)]
                prev = loc
            cost += self.distances[(prev, self.diagram.depot_id)]
            return cost

        def get_path_demand(path: List[int]) -> float:
            """Calculate total demand of a path"""
            return sum(self.locations[loc].demand for loc in path)

        def find_best_completion(partial_path: List[int], remaining_nodes: Set[int]) -> List[int]:
            """Find best completion of a partial path respecting capacity"""
            best_path = None
            best_cost = float('inf')
            remaining_capacity = self.capacity - get_path_demand(partial_path)

            # Try all possible completions with remaining nodes
            stack = [(partial_path, remaining_capacity, list(remaining_nodes))]
            while stack:
                current_path, capacity, available = stack.pop()
                
                # If this is a complete path and better than our best, update best
                if not available:
                    path_cost = get_path_cost(current_path)
                    if path_cost < best_cost:
                        best_cost = path_cost
                        best_path = current_path.copy()
                    continue

                # Try adding each available node
                for i, node in enumerate(available):
                    if self.locations[node].demand <= capacity:
                        new_path = current_path + [node]
                        new_capacity = capacity - self.locations[node].demand
                        new_available = available[:i] + available[i+1:]
                        stack.append((new_path, new_capacity, new_available))

            return best_path if best_path else partial_path

        # Main route extraction logic
        routes = []
        nodes_to_visit = set(range(1, len(self.locations)))
        
        # First try to follow high flow values to start routes
        significant_flows = {i: arc for i, arc in enumerate(self.diagram.arcs) 
                           if i in flow_values and flow_values[i] > 0.5}
        
        print("\nSignificant flows:")
        for i, (from_idx, to_idx, loc_id) in significant_flows.items():
            print(f"Arc {i}: {self.diagram.nodes[from_idx].last_visited}->{self.diagram.nodes[to_idx].last_visited} "
                  f"(loc {loc_id}) = {flow_values[i]:.2f}")

        # Build initial route segments from significant flows
        route_segments = []
        visited = set()
        
        for i, (from_idx, to_idx, loc_id) in significant_flows.items():
            if loc_id != self.diagram.depot_id and loc_id not in visited:
                segment = [loc_id]
                visited.add(loc_id)
                current = to_idx
                
                # Follow significant flows
                while True:
                    next_arc = next((
                        (j, arc) for j, arc in significant_flows.items()
                        if arc[0] == current and arc[2] != self.diagram.depot_id
                        and arc[2] not in visited
                    ), None)
                    
                    if not next_arc:
                        break
                        
                    j, (_, next_node, next_loc) = next_arc
                    segment.append(next_loc)
                    visited.add(next_loc)
                    current = next_node
                
                route_segments.append(segment)
        
        print("\nInitial route segments:", route_segments)
        
        # Complete routes from segments
        remaining_nodes = nodes_to_visit - visited
        
        # First, try to extend existing segments
        for segment in route_segments:
            if get_path_demand(segment) <= self.capacity:
                completed_path = find_best_completion(segment, remaining_nodes)
                if completed_path:
                    routes.append(completed_path)
                    remaining_nodes -= set(completed_path)

        # Create new routes for any remaining nodes
        while remaining_nodes:
            best_path = find_best_completion([], remaining_nodes)
            if not best_path:
                break
            routes.append(best_path)
            remaining_nodes -= set(best_path)

        return routes
    
    def solve_network_flow(self) -> Tuple[float, List[List[int]]]:
        """Solve network flow using Lagrangian relaxation"""
        solver = LagrangianSolver(self.diagram, self.distances, self.num_vehicles)
        best_bound = float('inf')
        best_routes = None
        
        # Parameters for subgradient method
        max_iterations = 200
        step_size = 10.0
        step_decay = 0.98
        no_improvement_limit = 30
        no_improvement_count = 0
        
        print("\nStarting Lagrangian relaxation:")
        
        for iteration in range(max_iterations):
            # Solve Lagrangian subproblem
            bound, paths = solver.solve_subproblem()
            
            # Extract routes from paths
            routes = []
            for path in paths:
                route = []
                for _, _, loc_id in path:
                    if loc_id != self.diagram.depot_id:
                        route.append(loc_id)
                if route:
                    routes.append(route)
            
            print(f"\nIteration {iteration}:")
            print(f"Bound: {bound}")
            print(f"Routes: {routes}")
            print(f"Multipliers: {solver.lambda_multipliers}")
            
            if bound < best_bound:
                best_bound = bound
                best_routes = routes
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Update multipliers
            solver.update_multipliers(paths, step_size)
            step_size *= step_decay
            
            # Early stopping
            if no_improvement_count >= no_improvement_limit or step_size < 0.01:
                break
        
        return best_bound, best_routes
        
    def solve(self, max_iterations: int = 100) -> Tuple[float, List[List[int]]]:
        best_bound = float('inf')
        best_routes = None
        
        for iteration in range(max_iterations):
            bound, routes = self.solve_network_flow()
            print(f"Iteration {iteration}: bound = {bound}, routes = {routes}")
            
            if bound >= best_bound:
                break
                
            best_bound = bound
            best_routes = routes
            
            # Check for cycles in routes
            cycles_found = False
            for route in routes:
                if self.diagram.eliminate_cycle(route):
                    cycles_found = True
                    
            if not cycles_found:
                break
                
        return best_bound, best_routes

def create_example():
    # Create simple example from paper
    locations = [
        Location(0, 0, True),   # Depot
        Location(1, 1),
        Location(2, 1),
        Location(3, 1),
        Location(4, 2)
    ]
    
    distances = {
        (0,1): 5,  (1,0): 5,   # Depot <-> 1
        (0,2): 10, (2,0): 10,  # Depot <-> 2
        (0,3): 5,  (3,0): 5,   # Depot <-> 3
        (0,4): 10, (4,0): 10,  # Depot <-> 4
        (1,2): 10, (2,1): 10,  # 1 <-> 2
        (1,3): 10, (3,1): 10,  # 1 <-> 3
        (1,4): 15, (4,1): 15,  # 1 <-> 4
        (2,3): 10, (3,2): 10,  # 2 <-> 3
        (2,4): 15, (4,2): 15,  # 2 <-> 4
        (3,4): 10, (4,3): 10,  # 3 <-> 4
    }
    
    solver = ColumnElimination(
        locations=locations,
        distances=distances,
        capacity=3,
        num_vehicles=2
    )
    
    print("Initial diagram structure:")
    print(f"Nodes ({len(solver.diagram.nodes)}):")
    for i, node in enumerate(solver.diagram.nodes):
        print(f"{i}: {node}")
    
    print(f"\nArcs ({len(solver.diagram.arcs)}):")
    for i, (from_idx, to_idx, loc_id) in enumerate(solver.diagram.arcs):
        from_loc = solver.diagram.nodes[from_idx].last_visited
        to_loc = solver.diagram.nodes[to_idx].last_visited
        print(f"{i}: {from_idx}({from_loc})->{to_idx}({to_loc}) (loc {loc_id})")
    
    bound, routes = solver.solve()
    print(f"\nFinal solution:")
    print(f"Lower bound: {bound}")
    print(f"Routes: {routes}")

if __name__ == "__main__":
    create_example()
