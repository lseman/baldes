# %%
from collections import defaultdict
from typing import Any, Dict, Set, Tuple, List, Optional
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import time

from reader import *
from auxiliar import *

import numpy as np

import sys
import random

sys.path.append("../build")
import pybaldes as baldes

import pandas as pd

# Create instance data
solomon_instance = InstanceData()
solomon_instance.read_instance("../build/instances/C201.txt")
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

instance.nodes = nodes
instance.vehicle_capacity = 700

graph = baldes.BucketGraph(nodes, int(instance.time_windows[0][1]), 20)
duals = [100 for i in range(len(nodes))]
graph.set_distance_matrix(instance.distances)
graph.set_duals(duals)
graph.setup()
instance.graph = graph
from gurobipy import Model, GRB, quicksum

# Input Data
num_nodes = len(instance.demands)  # Number of nodes (including depot)
depot = 0  # Index of the depot
#nodes = range(num_nodes)
distances = instance.distances  # Distance matrix
demands = instance.demands  # Demands for each node
time_windows = instance.time_windows  # (start, end) for each node
service_times = instance.service_times  # Service time for each node
vehicle_capacity = 700  # Vehicle capacity
num_vehicles = 3  # Number of vehicles



from gurobipy import Model, GRB, quicksum

#print("Routing Decision Variables:", x_solution)
#print("Arrival Times:", arrival_times)

# Solve
#best_cost, best_routes = solver.solve_with_subgradient()
# best_cost, best_routes = solver.solve_with_column_elimination()



class ColumnElimination:
    def __init__(self, nodes, distances, demands, time_windows, service_times, vehicle_capacity, depot, num_vehicles, instance):
        """
        Initialize the Column Elimination solver with problem data.
        """
        self.nodes = range(len(nodes))
        self.distances = distances
        self.demands = demands
        self.time_windows = time_windows
        self.service_times = service_times
        self.vehicle_capacity = vehicle_capacity
        self.depot = depot
        self.end_depot = len(nodes) - 1
        self.num_vehicles = num_vehicles
        self.instance = instance
        
        # Model components
        self.relaxed_model = None
        self.refined_model = None
        self.master_variables = None
        self.current_variables = None
        self.cm = ConflictManager(instance)
        
        # Lagrangian components
        self.lambda_values = None
        self.best_lb = float('-inf')
        self.best_ub = sum(
            self.distances[i][j] 
            for i in self.nodes 
            for j in self.nodes 
            if i != j
        )
        
        # Parameters for subgradient optimization
        self.step_size = 2.0
        self.step_reduction = 0.5
        self.non_improvement_limit = 20
        
        # Initialize relaxation parameters
        self.ng_size = 5  # Size of ng-route neighborhoods
        self.ng_neighbors = self._initialize_ng_neighborhoods()

    def _initialize_ng_neighborhoods(self):
        """Initialize ng-route neighborhoods for each node based on distances"""
        ng_neighbors = {}
        for i in self.nodes:
            if i != self.depot:
                # Get closest nodes to i (excluding depot)
                distances_from_i = [(j, self.distances[i][j]) 
                                  for j in self.nodes if j != self.depot and j != i]
                distances_from_i.sort(key=lambda x: x[1])
                ng_neighbors[i] = {x[0] for x in distances_from_i[:self.ng_size]}
        return ng_neighbors

    def build_relaxed_model(self):
        """Build initial relaxed model using ng-route relaxation"""

        model = Model("VRPTW")
        
        # Create indices for variables
        x_indices = [(i, j, k) for i in self.nodes for j in self.nodes 
                    for k in range(self.num_vehicles) if i != j]
        arrival_indices = list(self.nodes)
        
        # Decision Variables
        x = model.addVars(x_indices, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
        arrival_time = model.addVars(arrival_indices, vtype=GRB.CONTINUOUS, name="arrival_time")
        
        # Store variables
        self.master_variables = {
            "x": x,
            "arrival_time": arrival_time
        }
        self.current_variables = self.master_variables
        
        # Objective: Minimize total distance
        model.setObjective(
            quicksum(
                self.distances[i][j] * x[i, j, k]
                for i, j, k in x_indices
            ),
            GRB.MINIMIZE
        )
        
        # Constraints
        # 1. Visit each customer exactly once
        for j in self.nodes:
            if j != self.depot and j != self.end_depot:
                model.addConstr(
                    quicksum(x[i, j, k] 
                            for i in self.nodes if i != j
                            for k in range(self.num_vehicles)) == 1,
                    f"visit_once_{j}"
                )        
        
        # 2. Flow conservation for each vehicle
        for k in range(self.num_vehicles):
            for j in self.nodes:
                if j != self.depot and j != self.end_depot:
                    model.addConstr(
                        quicksum(x[i, j, k] for i in self.nodes if i != j) ==
                        quicksum(x[j, i, k] for i in self.nodes if i != j),
                        f"flow_{j}_{k}"
                    )
        
        # 3. Vehicle capacity constraints
        for k in range(self.num_vehicles):
            model.addConstr(
                quicksum(
                    self.demands[j] * quicksum(x[i, j, k] for i in self.nodes if i != j)
                    for j in self.nodes if j != self.depot and j != self.end_depot
                ) <= self.vehicle_capacity,
                f"capacity_{k}"
            )
        
        # 4. Depot entry and exit constraints for each vehicle
        for k in range(self.num_vehicles):
            model.addConstr(
                quicksum(x[self.depot, j, k] for j in self.nodes if j != self.depot) == 1,
                f"depot_exit_{k}"
            )
            model.addConstr(
                quicksum(x[j, self.end_depot, k] for j in self.nodes if j != self.end_depot) == 1,
                f"end_depot_entry_{k}"
            )
        
        # 5. Time windows with dynamic Big-M
        M = 1e6
        for i, j, k in x_indices:
            model.addConstr(
                arrival_time[j] >= 
                arrival_time[i] + self.service_times[i] + self.distances[i][j] - 
                M * (1 - x[i, j, k]),
                f"time_{i}_{j}_{k}"
            )
        
        # 6. Time window bounds
        for j in self.nodes:
            if j != self.depot and j != self.end_depot:
                model.addConstr(arrival_time[j] >= self.time_windows[j][0], f"time_lb_{j}")
                model.addConstr(arrival_time[j] <= self.time_windows[j][1], f"time_ub_{j}")
        
        # 7. Initialize arrival time at the depot and enforce arrival at end_depot
        model.addConstr(arrival_time[self.depot] == 0, "start_time_depot")
        for k in range(self.num_vehicles):
            model.addConstr(
                quicksum(x[j, self.end_depot, k] for j in self.nodes if j != self.end_depot) == 1,
                f"arrive_at_end_depot_{k}"
            )

        model.update()

        self.relaxed_model = model

    def refine_model(self, violations):
        """Refine the model by adding constraints to address detected violations."""
        if not hasattr(self, 'refined_model') or self.refined_model is None:
            print("Creating new refined model...")
            
            # Create new model
            self.refined_model = Model("Refined_Model")
            
            # Create indices for variables
            x_indices = [(i, j, k) for i in self.nodes for j in self.nodes 
                        for k in range(self.num_vehicles) if i != j]
            
            # Create variables in new model
            x = self.refined_model.addVars(x_indices, lb=0, ub=1, 
                                        vtype=GRB.CONTINUOUS, name="x")
            arrival_time = self.refined_model.addVars(self.nodes, 
                                                    vtype=GRB.CONTINUOUS, 
                                                    name="arrival_time")
            
            # Store the variables
            self.current_variables = {
                "x": x,
                "arrival_time": arrival_time
            }
            
            # Add original constraints
            self._copy_original_constraints()
        
        model = self.refined_model
        vars_x = self.current_variables["x"]
        vars_at = self.current_variables["arrival_time"]
        
        # Track added constraints to prevent duplication
        self.added_constraints = getattr(self, 'added_constraints', set())
        
        try:
            # Add cuts for violations
            for path, violation_type in violations:
                if (tuple(path), violation_type) in self.added_constraints:
                    continue  # Skip already addressed violations
                self.added_constraints.add((tuple(path), violation_type))
                
                if violation_type == "capacity":
                    # Add capacity constraint
                    total_demand = quicksum(
                        self.demands[node2] * vars_x.select(node1, node2, k)[0]
                        for node1 in path for node2 in path if node1 != node2
                        for k in range(self.num_vehicles)
                        if vars_x.select(node1, node2, k)
                    )
                    model.addConstr(total_demand <= self.vehicle_capacity, f"capacity_violation_{model.NumConstrs}")
                
                elif violation_type.startswith("time_window"):
                    # Add time precedence constraints
                    for i in range(len(path) - 1):
                        node1, node2 = path[i], path[i+1]
                        if node1 != self.depot and node2 != self.depot:
                            M = self.time_windows[node2][1] - self.time_windows[node1][0] + \
                                self.service_times[node1] + self.distances[node1][node2]
                            for k in range(self.num_vehicles):
                                if vars_x.select(node1, node2, k):
                                    model.addConstr(
                                        vars_at[node2] >= vars_at[node1] + self.service_times[node1] + 
                                        self.distances[node1][node2] - M * (1 - vars_x.select(node1, node2, k)[0]),
                                        f"time_window_violation_{model.NumConstrs}"
                                    )
            
            model.update()
            print(f"Added constraints for {len(violations)} violations")
        
        except Exception as e:
            print(f"Error adding constraints: {str(e)}")
            raise


    def _copy_original_constraints(self):
        """Helper method to copy original constraints to refined model."""
        model = self.refined_model
        vars_x = self.current_variables["x"]
        vars_at = self.current_variables["arrival_time"]
        
        # Objective
        model.setObjective(
            quicksum(self.distances[i][j] * vars_x[i,j,k]
                    for (i,j,k) in vars_x),
            GRB.MINIMIZE
        )
        
        # Visit once constraints
        for j in self.nodes:
            if j != self.depot:
                model.addConstr(
                    quicksum(vars_x[i,j,k] 
                            for i in self.nodes if i != j
                            for k in range(self.num_vehicles)
                            if (i,j,k) in vars_x) == 1,
                    f"visit_once_{j}"
                )
        
        # Flow conservation
        for k in range(self.num_vehicles):
            for j in self.nodes:
                if j != self.depot:
                    model.addConstr(
                        quicksum(vars_x[i,j,k] for i in self.nodes if i != j if (i,j,k) in vars_x) ==
                        quicksum(vars_x[j,i,k] for i in self.nodes if i != j if (j,i,k) in vars_x),
                        f"flow_{j}_{k}"
                    )
        
        # Capacity
        for k in range(self.num_vehicles):
            model.addConstr(
                quicksum(self.demands[j] * quicksum(vars_x[i,j,k] 
                        for i in self.nodes if i != j if (i,j,k) in vars_x)
                        for j in self.nodes if j != self.depot) <= self.vehicle_capacity,
                f"capacity_{k}"
            )
        
        # Time windows
        M = max(tw[1] for tw in self.time_windows)
        for (i,j,k) in vars_x:
            model.addConstr(
                vars_at[j] >= 
                vars_at[i] + self.service_times[i] + 
                self.distances[i][j] - M * (1 - vars_x[i,j,k]),
                f"time_{i}_{j}_{k}"
            )
        
        # Time window bounds
        for j in self.nodes:
            model.addConstr(vars_at[j] >= self.time_windows[j][0])
            model.addConstr(vars_at[j] <= self.time_windows[j][1])
        
        model.update()

    def _extract_paths(self, model):
        """Extract ordered nodes for each vehicle from the current solution."""
        ordered_nodes = {k: [] for k in range(self.num_vehicles)}

        # Extract variable values for arcs and arrival times
        x_vals = {(i, j, k): var.X for (i, j, k), var in self.current_variables["x"].items()}
        arrival_times = {i: var.X for i, var in self.current_variables["arrival_time"].items()}

        for k in range(self.num_vehicles):
            # Find all arcs for this vehicle
            arcs_for_vehicle = [(i, j) for (i, j, v), value in x_vals.items() if v == k and value > 1e-6]

            # Flatten the nodes involved in arcs and sort them by arrival time
            nodes_for_vehicle = list({node for arc in arcs_for_vehicle for node in arc})
            nodes_sorted_by_time = sorted(
                nodes_for_vehicle,
                key=lambda node: arrival_times.get(node, float('inf'))  # Use a large value if arrival time is missing
            )

            # Add to the ordered list for this vehicle
            ordered_nodes[k] = nodes_sorted_by_time

        paths = []
        for k in ordered_nodes:
            paths.append(ordered_nodes[k])
        print(paths)
        return paths


    def detect_violations(self, paths):
        """
        Detect violations in the current solution paths.

        Parameters:
        - paths: A list of paths, where each path is a list of nodes visited by a single vehicle.

        Returns:
        - violations: A list of tuples, where each tuple contains a path and the type of violation.
        """
        violations = []

        for path in paths:
            # Check capacity violations
            total_demand = sum(self.demands[j] for j in path[1:-1])  # Exclude the depot nodes
            if total_demand > self.vehicle_capacity:
                violations.append((path, "capacity"))
            
            # Check time window violations
            current_time = 0
            prev_node = path[0]  # Start from the depot
            
            for node in path[1:]:  # Iterate through the path, skipping the depot at the end
                travel_time = self.distances[prev_node][node]
                arrival = current_time + travel_time
                
                if node != self.depot and node != self.end_depot:  # Check for customer nodes
                    if arrival < self.time_windows[node][0]:
                        violations.append((path, f"time_window_early: node {node}"))
                    elif arrival > self.time_windows[node][1]:
                        violations.append((path, f"time_window_late: node {node}"))
                    
                    # Update the current time considering service time
                    current_time = max(arrival, self.time_windows[node][0]) + self.service_times[node]
                elif node == self.end_depot:
                    # If there are time constraints on the end_depot, check them explicitly
                    if arrival < self.time_windows[self.end_depot][0]:
                        violations.append((path, f"time_window_early: end_depot"))
                    elif arrival > self.time_windows[self.end_depot][1]:
                        violations.append((path, f"time_window_late: end_depot"))
                    current_time = arrival
                
                prev_node = node  # Update the previous node for the next iteration
            
        return violations


    def solve(self, max_iterations=100, time_limit=3600):
        """Main column elimination algorithm with Lagrangian relaxation."""
        start_time = time.time()
        
        # Initialize
        self.build_relaxed_model()
        self.lambda_values = {j: 0.0 for j in self.nodes if j != self.depot}
        non_improvement_count = 0
        best_solution_paths = None
        
        print("Starting column elimination...")

        current_model = self.relaxed_model
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}")
            
            current_model.optimize()

            # if infeasible
            if current_model.status == GRB.INFEASIBLE:
                # save ilp
                current_model.computeIIS()
                current_model.write('model.ilp')
            solution_value = current_model.objVal

            paths = self._extract_paths(current_model)

            for path in paths:
                conflicts = self.cm.find_conflict(path)
                print(conflicts)
                self.cm.refine_conflict(conflicts)
            
            labels = self.instance.graph.phaseFour()
            new_paths = [label.nodes_covered() for label in labels]
            print(new_paths)
            # Update bounds
            prev_lb = self.best_lb
            self.best_lb = max(self.best_lb, solution_value)
            
            # Check for violations
            violations = self.detect_violations(paths)

            
            if not violations:
                # Found feasible solution - update upper bound
                current_ub = sum(
                    sum(self.distances[path[i]][path[i+1]] 
                        for i in range(len(path)-1))
                    for path in paths
                )
                if current_ub > 0 and current_ub < self.best_ub:
                    self.best_ub = current_ub
                    best_solution_paths = paths
                    print(f"Found improved feasible solution with value: {current_ub:.2f}")
                    
                    # Try variable fixing after finding a better solution
                    self.apply_variable_fixing()
            else:
                # Refine model based on violations
                print(f"Found {len(violations)} violations. Refining model...")
                self.refine_model(violations)
                current_model = self.refined_model
            
            # Print current bounds
            if self.best_ub < float('inf'):
                gap = (self.best_ub - self.best_lb) / self.best_ub
                print(f"Bounds - LB: {self.best_lb:.2f}, UB: {self.best_ub:.2f}, Gap: {gap:.2%}")
            else:
                print(f"Bounds - LB: {self.best_lb:.2f}, UB: inf")
            
            # Update Lagrangian multipliers
            #self.update_lagrangian_multipliers(solution_value, paths)
            
            # Check convergence
            if self.best_ub < float('inf'):
                gap = (self.best_ub - self.best_lb) / self.best_ub
                if gap < 0.01:  # 1% optimality gap
                    print("Reached optimality gap threshold")
                    break
            
            # Check for step size reduction
            if solution_value <= prev_lb:  # Changed to use prev_lb
                non_improvement_count += 1
                if non_improvement_count >= self.non_improvement_limit:
                    self.step_size *= self.step_reduction
                    non_improvement_count = 0
                    print(f"Reducing step size to {self.step_size}")
            else:
                non_improvement_count = 0
            
            # Check time limit
            if time.time() - start_time > time_limit:
                print("Time limit reached")
                break
        
        # Return final results
        return self.best_lb, self.best_ub, best_solution_paths

    def apply_variable_fixing(self):
        """Apply reduced cost variable fixing to eliminate variables."""
        # First check if we have a model to work with
        model_to_use = self.refined_model if self.refined_model is not None else self.relaxed_model
        if model_to_use is None or self.best_lb == float('-inf'):
            return

        fixed_count = 0
        
        try:
            # Get dual values from visit_once constraints
            pi = {}
            for j in self.nodes:
                if j != self.depot:
                    constr = model_to_use.getConstrByName(f"visit_once_{j}")
                    if constr is not None:  # Make sure constraint exists
                        pi[j] = constr.Pi

            # Only proceed if we got some dual values
            if pi:
                # Calculate reduced costs and fix variables
                for (i,j,k), var in self.current_variables["x"].items():
                    if var.UB > 0:  # Only check unfixed variables
                        # Calculate reduced cost
                        rc = self.distances[i][j]
                        if j != self.depot and j in pi:
                            rc -= pi[j]
                        
                        # Fix variable if reduced cost is too high
                        if rc > self.best_ub - self.best_lb:
                            model_to_use.addConstr(var == 0, name=f"fixing_{i}_{j}_{k}")
                            fixed_count += 1

            if fixed_count > 0:
                print(f"Fixed {fixed_count} variables using reduced cost criterion")
                
        except Exception as e:
            print(f"Warning: Variable fixing failed with error: {str(e)}")
            return  # Continue with the algorithm even if variable fixing fails
        
    def check_solution_quality(self, paths):
        """Verify solution quality and feasibility."""
        metrics = {
            'total_distance': 0,
            'total_demand': 0,
            'time_window_violations': 0,
            'capacity_violations': 0,
            'route_count': len(paths)
        }
        
        for path in paths:
            # Route distance
            route_distance = sum(self.distances[path[i]][path[i+1]] 
                               for i in range(len(path)-1))
            metrics['total_distance'] += route_distance
            
            # Capacity check
            route_demand = sum(self.demands[j] for j in path[1:-1])
            metrics['total_demand'] += route_demand
            if route_demand > self.vehicle_capacity:
                metrics['capacity_violations'] += 1
            
            # Time windows check
            current_time = 0
            prev_node = path[0]
            for node in path[1:]:
                arrival = current_time + self.distances[prev_node][node]
                if node != self.depot:
                    if (arrival < self.time_windows[node][0] or 
                        arrival > self.time_windows[node][1]):
                        metrics['time_window_violations'] += 1
                    current_time = max(arrival, self.time_windows[node][0]) + \
                                 self.service_times[node]
                prev_node = node
        
        metrics['is_feasible'] = (metrics['capacity_violations'] == 0 and 
                                 metrics['time_window_violations'] == 0)
        return metrics

    def print_solution_summary(self, paths, metrics):
        """Print detailed summary of the solution."""
        print("\nSolution Summary:")
        print(f"Number of routes: {metrics['route_count']}")
        print(f"Total distance: {metrics['total_distance']:.2f}")
        print(f"Total demand: {metrics['total_demand']}")
        print(f"Feasible: {metrics['is_feasible']}")
        
        if not metrics['is_feasible']:
            print(f"Capacity violations: {metrics['capacity_violations']}")
            print(f"Time window violations: {metrics['time_window_violations']}")
        
        print("\nDetailed Routes:")
        for i, path in enumerate(paths):
            print(f"\nRoute {i+1}:")
            print(f"Sequence: {path}")
            route_demand = sum(self.demands[j] for j in path[1:-1])
            print(f"Route demand: {route_demand}/{self.vehicle_capacity}")
            route_distance = sum(self.distances[path[i]][path[i+1]] 
                               for i in range(len(path)-1))
            print(f"Route distance: {route_distance:.2f}")
# Create solver instance
solver = ColumnElimination(
    nodes=nodes,
    distances=instance.distances,
    demands=instance.demands,
    time_windows=instance.time_windows,
    service_times=instance.service_times,
    vehicle_capacity=vehicle_capacity,
    depot=0,
    num_vehicles=num_vehicles,
    instance = instance
)

# Solve the problem
best_lb, best_ub, paths = solver.solve()

# Get solution quality metrics
metrics = solver.check_solution_quality(paths)

# Print detailed solution
solver.print_solution_summary(paths, metrics)

# Get dual values and variable values if needed
x_values, arrival_times, duals = solver.get_solution()
