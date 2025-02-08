from dataclasses import dataclass
from typing import Set

@dataclass
class Location:
    id: int
    demand: float
    is_depot: bool = False

@dataclass
class State:
    """State for VRPTW using time window relaxation"""
    last_visited: int      # Last visited location
    weight: float         # Accumulated weight/load
    visited: Set[int]     # Set of visited locations
    time: float          # Current time
    count: int           # Counter to maintain acyclic graph

def __str__(self):
    return f"State(last={self.last_visited}, weight={self.weight}, time={self.time}, count={self.count}, visited={self.visited})"
