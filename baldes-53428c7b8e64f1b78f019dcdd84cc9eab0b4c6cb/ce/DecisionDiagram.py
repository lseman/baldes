
import heapq
import numpy as np
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import math

from Aux import *

class VRPTWDecisionDiagram:
    def __init__(self, locations: List[Location], capacity: float, time_windows: List[Tuple[float, float]], 
                 distances: Dict[Tuple[int, int], float], service_times: List[float], 
                 delta: float = 1.0, relax_capacity: bool = False):
        self.locations = locations
        self.capacity = capacity
        self.time_windows = time_windows
        self.distances = distances
        self.service_times = service_times
        self.delta = delta
        self.relax_capacity = relax_capacity
        self.nodes = []
        self.arcs = []
        
        # Initialize depot related variables
        self.depot_id = next(loc.id for loc in locations if loc.is_depot)
        
        # Add self-loops with zero distance for depot
        self.distances[(self.depot_id, self.depot_id)] = 0
        
        # Calculate maximum route length and other bounds
        self.max_route_length = self._calculate_max_route_length()
        self.max_locations_per_route = self._calculate_max_locations_per_route()
        
       # Root and terminal node
        root_state = State(self.depot_id, 0, set([self.depot_id]), 0, 0)
        self.root = 0
        self.nodes.append(root_state)

        terminal_state = State(self.depot_id, 0, set([self.depot_id]), 0, 0)
        self.terminal = 101
        self.nodes.append(terminal_state)
        
        # Cache for created states
        self.state_cache = {self._state_key(root_state): self.root}

    def _state_key(self, state: State) -> Tuple:
        """Generate a unique key for a state."""
        return (state.last_visited, state.weight, state.time, frozenset(state.visited))

    def extend_state(self, current_idx: int, loc_id: int) -> int:
        """Extend the diagram by transitioning to a new state."""
        current_state = self.nodes[current_idx]
        
        # Skip infeasible transitions
        if not self._is_feasible_transition(current_state, loc_id):
            return -1  # Invalid transition
        
        # Compute new state
        travel_time = self.distances[(current_state.last_visited, loc_id)]
        service_time = self.service_times[current_state.last_visited]
        arrival_time = current_state.time + travel_time + service_time
        early, _ = self.time_windows[loc_id]
        actual_time = max(arrival_time, early)
        bucketed_time = self._get_bucketed_time(actual_time)
        
        new_state = State(
            last_visited=loc_id,
            weight=current_state.weight + (0 if self.relax_capacity else self.locations[loc_id].demand),
            visited=current_state.visited | {loc_id},
            time=bucketed_time,
            count=current_state.count + 1
        )
        
        # Check if this state already exists
        state_key = self._state_key(new_state)
        if state_key in self.state_cache:
            new_idx = self.state_cache[state_key]
        else:
            new_idx = len(self.nodes)
            self.nodes.append(new_state)
            self.state_cache[state_key] = new_idx

        # Add arc to new state
        self.arcs.append((current_idx, new_idx, loc_id))
        
        return new_idx


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
                    if self._is_feasible_transition(p_state, element):
                        feasible_decisions.add(element)

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
                visited=set(path[:j+1]),
                time=self._get_feasible_time(path[:j+1]),
                count=j+1
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
        """Get corresponding state in original problem P for a sequence"""
        current_state = State(
            last_visited=self.depot_id,
            weight=0,
            visited=set([self.depot_id]),
            time=0,
            count=0
        )
        
        for loc_id in sequence:
            # Check if transition is feasible in P
            if not self._is_feasible_transition(current_state, loc_id):
                return None
                
            # Update state
            current_state.last_visited = loc_id
            current_state.weight += self.locations[loc_id].demand if loc_id != self.depot_id else 0
            current_state.visited.add(loc_id)
            # Update time considering time windows
            travel_time = self.distances[(current_state.last_visited, loc_id)]
            service_time = self.service_times[current_state.last_visited]
            arrival_time = current_state.time + travel_time + service_time
            early, _ = self.time_windows[loc_id]
            current_state.time = max(arrival_time, early)
            current_state.count += 1
            
        return current_state

    def _get_feasible_time(self, path: List[int]) -> float:
        """Calculate feasible time for a path considering time windows"""
        time = 0
        prev_loc = self.depot_id
        
        for loc_id in path:
            travel_time = self.distances[(prev_loc, loc_id)]
            service_time = self.service_times[prev_loc]
            arrival_time = time + travel_time + service_time
            early, _ = self.time_windows[loc_id]
            time = max(arrival_time, early)
            prev_loc = loc_id
            
        return time

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

    def _calculate_max_route_length(self) -> float:
        """Calculate maximum possible route duration based on depot time window"""
        depot_early, depot_late = self.time_windows[self.depot_id]
        return depot_late - depot_early

    def _calculate_max_locations_per_route(self) -> int:
        """Calculate maximum number of locations that can be visited in one route"""
        if self.relax_capacity:
            # If capacity is relaxed, limit by time windows
            total_time = self.max_route_length
            min_visit_time = float('inf')
            
            # Find minimum time needed to visit a location
            for i in range(len(self.locations)):
                if i == self.depot_id:
                    continue
                # Minimum time = minimum travel time + service time
                min_travel = min(self.distances.get((j, i), float('inf')) 
                               for j in range(len(self.locations)) if j != i)
                min_visit_time = min(min_visit_time, min_travel + self.service_times[i])
            
            return int(total_time / min_visit_time) if min_visit_time < float('inf') else len(self.locations)
        else:
            # If capacity is enforced, also consider capacity constraint
            total_capacity = self.capacity
            min_demand = min(loc.demand for loc in self.locations if not loc.is_depot)
            print(f"Min demand: {min_demand}")
            capacity_limit = int(total_capacity / min_demand)
            
            time_limit = self._calculate_max_route_length()
            min_visit_time = min(self.service_times[i] + min(self.distances.get((j, i), float('inf')) 
                               for j in range(len(self.locations)) if j != i)
                               for i in range(len(self.locations)) if i != self.depot_id)
            time_based_limit = int(time_limit / min_visit_time) if min_visit_time < float('inf') else len(self.locations)
            
            return min(capacity_limit, time_based_limit)

    def _get_bucketed_time(self, time: float) -> float:
        """Round time down to nearest multiple of delta"""
        return math.floor(time / self.delta) * self.delta

    def _is_feasible_transition(self, state: State, loc_id: int) -> bool:
        """Check if transition is feasible considering time windows and capacity"""
        # Don't allow self-loops except at depot
        if state.last_visited == loc_id and loc_id != self.depot_id:
            return False
            
        # Check if location was already visited (except depot)
        if loc_id in state.visited and loc_id != self.depot_id:
            return False
            
        # Check if maximum locations per route exceeded
        if state.count >= self.max_locations_per_route:
            return False
            
        # Check capacity constraint if not relaxed
        if not self.relax_capacity and loc_id != self.depot_id:
            new_weight = state.weight + self.locations[loc_id].demand
            if new_weight > self.capacity:
                return False
        
        # Get travel time and service time
        travel_time = self.distances.get((state.last_visited, loc_id), float('inf'))
        service_time = self.service_times[state.last_visited]
        
        # Calculate arrival time
        arrival_time = state.time + travel_time + service_time
        
        # Get time window
        early, late = self.time_windows[loc_id]
        
        # Check if we can reach the location within its time window
        if arrival_time > late:
            return False
            
        # If returning to depot, check if we can reach it before depot's closing time
        if loc_id == self.depot_id:
            depot_early, depot_late = self.time_windows[self.depot_id]
            if arrival_time > depot_late:
                return False
        
        return True

    def _build_initial_diagram(self):
        """Build initial relaxed diagram for VRPTW"""
        current_layer = {self.root}
        next_layer = set()
        
        print("\nBuilding initial diagram:")
        
        while current_layer:
            #print(f"\nCurrent layer: {current_layer}")
            for node_idx in current_layer:
                state = self.nodes[node_idx]
                
                # Try extending paths from current state
                for loc in self.locations:
                    # Skip infeasible transitions
                    if not self._is_feasible_transition(state, loc.id):
                        continue
                                      
                    # Calculate arrival time
                    travel_time = self.distances[(state.last_visited, loc.id)]
                    service_time = self.service_times[state.last_visited]
                    arrival_time = state.time + travel_time + service_time
                    early, _ = self.time_windows[loc.id]
                    
                    actual_time = max(arrival_time, early)
                    bucketed_time = self._get_bucketed_time(actual_time)
                    
                    if loc.id == self.depot_id:
                        self.arcs.append((node_idx, self.terminal, loc.id))
                        continue
                    
                    new_state = State(
                        last_visited=loc.id,
                        weight=state.weight + (0 if self.relax_capacity else loc.demand),
                        visited=state.visited | {loc.id},
                        time=bucketed_time,
                        count=state.count + 1
                    )
                    
                    existing_idx = next((i for i, s in enumerate(self.nodes) 
                                      if s.last_visited == new_state.last_visited 
                                      and abs(s.weight - new_state.weight) < 1e-6
                                      and abs(s.time - new_state.time) < 1e-6
                                      and s.visited == new_state.visited), None)
                    
                    if existing_idx is None:
                        new_idx = len(self.nodes)
                        self.nodes.append(new_state)
                        next_layer.add(new_idx)
                        self.arcs.append((node_idx, new_idx, loc.id))
                    else:
                        next_layer.add(existing_idx)
                        self.arcs.append((node_idx, existing_idx, loc.id))
            
            current_layer = next_layer
            next_layer = set()
        print(f"Diagram built with {len(self.nodes)} nodes and {len(self.arcs)} arcs")