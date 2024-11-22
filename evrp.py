# %%
import math
import pandas as pd
import re
from typing import List
import numpy as np
from typing import Tuple

from evrp.evrp_aux import *

# %%
import math
import numpy as np
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random
from typing import List, Tuple, Dict, Optional


def check_battery_feasibility(from_node: int, to_node: int, remaining_battery: float, evrp) -> bool:
    battery_needed = evrp.dist_matrix[from_node][to_node] * evrp.fuel_consumption
    return battery_needed <= remaining_battery


def calculate_route_cost(route: List[int], evrp) -> Tuple[float, bool]:
    total_cost = 0
    remaining_battery = evrp.battery_capacity

    for i in range(len(route) - 1):
        from_node, to_node = route[i], route[i + 1]
        distance = evrp.dist_matrix[from_node][to_node]
        battery_needed = distance * evrp.fuel_consumption

        if battery_needed > remaining_battery:
            return float("inf"), False

        remaining_battery -= battery_needed
        total_cost += distance

        if to_node > evrp.num_customers:  # Recharge at station
            remaining_battery = evrp.battery_capacity

    return total_cost, True


def nearest_neighbor_initial_solution(evrp) -> List[List[int]]:
    routes = []
    unvisited = set(range(1, evrp.num_customers + 1))

    while unvisited:
        route = [0]
        current = 0
        remaining_battery = evrp.battery_capacity

        while unvisited:
            best_next = None
            best_cost = float("inf")

            for customer in unvisited:
                distance = evrp.dist_matrix[current][customer]
                battery_needed = distance * evrp.fuel_consumption

                if battery_needed > remaining_battery:
                    continue

                if distance < best_cost:
                    best_next = customer
                    best_cost = distance

            if best_next is None:
                break

            route.append(best_next)
            remaining_battery -= evrp.dist_matrix[current][best_next] * evrp.fuel_consumption
            unvisited.remove(best_next)
            current = best_next

        route.append(0)
        routes.append(route)

    return routes


def two_opt(route: List[int], evrp) -> List[int]:
    best_route = route[:]
    best_cost, _ = calculate_route_cost(route, evrp)

    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                new_cost, feasible = calculate_route_cost(new_route, evrp)

                if feasible and new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True

        route = best_route[:]

    return best_route


def relocate(routes: List[List[int]], evrp) -> List[List[int]]:
    best_routes = [route[:] for route in routes]
    best_cost = sum(calculate_route_cost(route, evrp)[0] for route in routes)

    improved = True
    while improved:
        improved = False

        for r1 in range(len(routes)):
            for r2 in range(len(routes)):
                if r1 == r2:
                    continue

                for i in range(1, len(routes[r1]) - 1):
                    for j in range(1, len(routes[r2]) - 1):
                        new_routes = [route[:] for route in routes]
                        node = new_routes[r1].pop(i)
                        new_routes[r2].insert(j, node)

                        new_cost = sum(calculate_route_cost(route, evrp)[0] for route in new_routes)
                        if new_cost < best_cost:
                            best_routes = new_routes
                            best_cost = new_cost
                            improved = True

        routes = [route[:] for route in best_routes]

    return best_routes


def station_insertion_removal(routes: List[List[int]], evrp) -> List[List[int]]:
    best_routes = [route[:] for route in routes]
    best_cost = sum(calculate_route_cost(route, evrp)[0] for route in routes)

    for r_idx, route in enumerate(routes):
        for i in range(1, len(route) - 1):
            for station in range(evrp.num_customers + 1, evrp.num_customers + evrp.num_stations + 1):
                new_routes = [route[:] for route in routes]
                new_routes[r_idx].insert(i, station)

                new_cost = sum(calculate_route_cost(route, evrp)[0] for route in new_routes)
                if new_cost < best_cost:
                    best_routes = new_routes
                    best_cost = new_cost

    return best_routes


def optimize_routes(evrp, initial_routes: List[List[int]], max_iterations: int = 500) -> List[List[int]]:
    routes = initial_routes[:]
    best_routes = routes[:]
    best_cost = sum(calculate_route_cost(route, evrp)[0] for route in routes)

    for _ in range(max_iterations):
        routes = [two_opt(route, evrp) for route in routes]
        routes = relocate(routes, evrp)
        routes = station_insertion_removal(routes, evrp)

        current_cost = sum(calculate_route_cost(route, evrp)[0] for route in routes)
        if current_cost < best_cost:
            best_routes = routes[:]
            best_cost = current_cost

    return best_routes


# %%
# Main program
def main():
    # Sample data for testing
    instance = InstanceData()
    if EVRP_read_instance("build/evrp_instances/c101C5.txt", instance):
        evrp = convert_instance_to_evrp(instance)
        print("EVRP data initialized from InstanceData.")
    print("Customers", evrp.num_customers)
    print("Station", evrp.num_stations)
    initial_routes = nearest_neighbor_initial_solution(evrp)
    print("initial routes", initial_routes)
    # Optimize using local search
    best_solution, time_info = optimize_routes(evrp, initial_routes)
    print("Best route", best_solution)
    # compute cost of best solution
    cost, _, _ = calculate_route_cost(best_solution[0], evrp)

    # call print solution
    print_solution(best_solution, time_info, evrp)
    print("Cost of best solution:", cost)


if __name__ == "__main__":
    main()



