import heapq
import numpy as np
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional

from Aux import Location, State
from DecisionDiagram import VRPTWDecisionDiagram
from LagSolver import VRPTWLagrangianSolver
from reader import *

class ColumnElimination:
    def __init__(self, locations: List[Location], distances: Dict[Tuple[int, int], float],
                 capacity: float, num_vehicles: int, time_windows: List[Tuple[float, float]] = None, service_times: List[float] = None):
        self.locations = locations
        self.distances = distances
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        
        # Create initial decision diagram
        self.diagram = VRPTWDecisionDiagram(
            locations=locations,
            capacity=capacity,
            time_windows=time_windows,
            distances=distances,
            service_times=service_times,
            delta=10,  # Time bucketing parameter
            relax_capacity=False
        )

    def extract_routes_from_flows(self, paths: List[List[Tuple[int, int, int]]]) -> List[List[int]]:
        """Decompose subproblem paths into optimal routes."""
        if not paths:
            print("No paths provided for route extraction.")
            return []
        
        routes = []
        print("\nDecomposing paths into routes:")
        for path in paths:
            route = []
            for _, _, loc_id in path:
                if loc_id != self.diagram.depot_id:  # Exclude depot from route
                    route.append(loc_id)
            if route:
                print(f"Extracted route: {route}")
                routes.append(route)
        
        return routes
 
     
    
    def solve_network_flow(self) -> Tuple[float, List[List[int]]]:
        """Solve network flow using Lagrangian relaxation with Polyak step size."""
        solver = VRPTWLagrangianSolver(self.diagram, self.distances, self.num_vehicles)
        best_ub = float('inf')  # Upper bound (feasible solution cost)
        best_lb = float('-inf')  # Lower bound (Lagrangian relaxation bound)
        best_routes = []

        max_iterations = 200
        no_improvement_limit = 30
        no_improvement_count = 0

        print("\nStarting Lagrangian relaxation:")

        for iteration in range(max_iterations):
            # Solve Lagrangian subproblem
            bound, paths = solver.solve_subproblem()

            # Extract routes from paths
            routes = self.extract_routes_from_flows(paths)

            print(f"\nIteration {iteration}:")
            print(f"Bound (Lower Bound): {bound}")
            print(f"Routes: {routes}")
            print(f"Multipliers: {solver.lambda_multipliers}")

            # Update bounds
            if routes:
                ub = sum(self.distances[(self.diagram.depot_id, route[0])] +
                         sum(self.distances[(route[i], route[i + 1])] for i in range(len(route) - 1)) +
                         self.distances[(route[-1], self.diagram.depot_id)] for route in routes)
                if ub < best_ub:
                    best_ub = ub
                    best_routes = routes

            if bound > best_lb:
                best_lb = bound
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Stop if gap is closed
            if best_ub - best_lb < 1e-3:
                print("Optimality gap closed!")
                break

            # Update multipliers using Polyak step size
            solver.update_multipliers(paths, best_ub, best_lb)

            # Early stopping
            if no_improvement_count >= no_improvement_limit:
                print("Stopping due to no improvement.")
                break

        return best_lb, best_routes



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

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import math

@dataclass
class Location:
    id: int
    demand: float
    service_time: float
    is_depot: bool = False

def create_test_instance():
    """
    Create a small VRPTW instance:
    - Depot (id=0) and 5 customers (id=1-5)
    - Vehicle capacity = 15
    - Each customer has demand between 2-5 units
    - Time windows and service times for each location
    - Manhattan distances between locations
    """
    solomon_instance = InstanceData()
    solomon_instance.read_instance("../build/instances/C203.txt")
    #instance = convert_instance_data(solomon_instance)

    locations = []
    for i in range(solomon_instance.nN-1):
        locations.append(Location(
            id=i,
            demand=solomon_instance.demand[i],
            service_time=solomon_instance.service_time[i]
        ))

    locations[0].is_depot = True
    coordinates = {}
    for i in range(solomon_instance.nN-1):
        coordinates[i] = (solomon_instance.x_coord[i], solomon_instance.y_coord[i])
    
    time_windows = []
    for i in range(solomon_instance.nN-1):
        time_windows.append((solomon_instance.window_open[i], solomon_instance.window_close[i]))

    # Service times
    service_times = [loc.service_time for loc in locations]

    # Problem parameters
    capacity = 700
    num_vehicles = 3
    delta = 10  # Time bucketing parameter

    # from distance_matrix create distances tuple
    distances = {}
    for i in range(solomon_instance.nN-1):
        for j in range(solomon_instance.nN-1):
            if i != j:
                distances[(i, j)] = solomon_instance.travel_cost[i][j]
    return {
        'locations': locations,
        'distances': distances,
        'time_windows': time_windows,
        'service_times': service_times,
        'capacity': capacity,
        'num_vehicles': num_vehicles,
        'delta': delta
    }


def test_feasible_path(instance, path):
    """Test if a path is feasible"""
    load = 0
    time = 0
    prev_loc = 0  # Start at depot
    
    for loc in path:
        # Check capacity
        load += instance['locations'][loc].demand
        if load > instance['capacity']:
            print(f"Capacity violated at location {loc}: {load} > {instance['capacity']}")
            return False
            
        # Check time
        travel_time = instance['distances'][(prev_loc, loc)]
        time += travel_time + instance['service_times'][prev_loc]
        early, late = instance['time_windows'][loc]
        
        # Wait if too early
        time = max(time, early)
        
        if time > late:
            print(f"Time window violated at location {loc}: {time} > {late}")
            return False
            
        prev_loc = loc
    
    # Check return to depot
    time += instance['distances'][(prev_loc, 0)]
    depot_early, depot_late = instance['time_windows'][0]
    if time > depot_late:
        print(f"Depot time window violated: {time} > {depot_late}")
        return False
        
    return True

def test_vrptw():
    """Test the VRPTW implementation"""
    # Create and validate test instance
    instance = create_test_instance()
    
    print("Instance details:")
    print("Distances:")
    for i in range(len(instance['locations'])):
        for j in range(len(instance['locations'])):
            if i != j:
                print(f"{i}->{j}: {instance['distances'][(i,j)]}")
    
    print("\nTime windows:")
    for i, (early, late) in enumerate(instance['time_windows']):
        print(f"Location {i}: [{early}, {late}]")
    
    print("\nDemands:")
    for i, loc in enumerate(instance['locations']):
        print(f"Location {i}: {loc.demand}")

    # Create column elimination solver
    solver = ColumnElimination(
        locations=instance['locations'],
        distances=instance['distances'],
        capacity=instance['capacity'],
        num_vehicles=instance['num_vehicles'],
        time_windows=instance['time_windows'],
        service_times=instance['service_times']
    )
    
    # Print initial diagram structure
    print("\nInitial diagram structure:")
    print(f"Nodes ({len(solver.diagram.nodes)}):")
    for i, node in enumerate(solver.diagram.nodes):
        print(f"{i}: {node}")
    
    print(f"\nArcs ({len(solver.diagram.arcs)}):")
    for i, (from_idx, to_idx, loc_id) in enumerate(solver.diagram.arcs):
        from_loc = solver.diagram.nodes[from_idx].last_visited
        to_loc = solver.diagram.nodes[to_idx].last_visited
        print(f"{i}: {from_idx}({from_loc})->{to_idx}({to_loc}) (loc {loc_id})")
    
    # Solve using column elimination
    bound, routes = solver.solve()
    print(f"\nFinal solution:")
    print(f"Lower bound: {bound}")
    print(f"Routes: {routes}")

    # Verify feasibility of routes
    if routes:
        print("\nVerifying route feasibility:")
        for i, route in enumerate(routes):
            print(f"\nRoute {i+1}: 0 -> {' -> '.join(map(str, route))} -> 0")
            print(f"Feasible: {test_feasible_path(instance, route)}")

if __name__ == "__main__":
    test_vrptw()
