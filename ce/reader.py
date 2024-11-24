# %%
from typing import Any, Dict, Set, Tuple, List, Optional
from dataclasses import dataclass
import math


class InstanceData:
    def __init__(self):
        self.distance: List[List[float]] = []
        self.travel_cost: List[List[float]] = []
        self.demand: List[int] = []
        self.window_open: List[float] = []
        self.window_close: List[float] = []
        self.n_tw: List[int] = []
        self.service_time: List[float] = []
        self.demand_additional: List[float] = []
        self.q = 0
        self.nC = 0
        self.nV = 0
        self.nN = 0
        self.nR = 0
        self.T_max = 0.0
        self.T_avg = 0.0
        self.demand_sum = 0.0
        self.nV_min = 0
        self.nV_max = 0
        self.nD_min = 0
        self.nD_max = 0
        self.deleted_arcs: List[int] = []
        self.deleted_arcs_n = 0
        self.deleted_arcs_n_max = 0
        self.x_coord: List[float] = []
        self.y_coord: List[float] = []
        self.problem_type = "vrptw"

    def read_instance(self, file_name: str):
        with open(file_name, "r") as file:
            # Skip the initial 4 lines
            for _ in range(7):
                next(file)

            # Read nV and q values
            line = file.readline().strip()

            # self.nV, self.q = map(int, line.split())
            self.nN = 102  # Assuming N_SIZE = 25

            # Initialize arrays based on nN
            self.x_coord = [0.0] * self.nN
            self.y_coord = [0.0] * self.nN
            self.demand = [0] * self.nN
            self.window_open = [0.0] * self.nN
            self.window_close = [0.0] * self.nN
            self.service_time = [0.0] * self.nN
            self.n_tw = [0] * self.nN

            # Read each line to populate coordinates, demand, and time windows
            i = 0
            for line in file:
                values = line.split()

                if len(values) < 7:
                    continue  # Skip any incomplete lines
                self.x_coord[i] = float(values[1])
                self.y_coord[i] = float(values[2])
                self.demand[i] = int(values[3])
                self.window_open[i] = float(values[4])
                self.window_close[i] = float(values[5])
                self.service_time[i] = float(values[6])
                self.n_tw[i] = 0
                i += 1

            # Add end depot = start depot
            self.x_coord[self.nN - 1] = self.x_coord[0]
            self.y_coord[self.nN - 1] = self.y_coord[0]
            self.demand[self.nN - 1] = self.demand[0]
            self.window_open[self.nN - 1] = self.window_open[0]
            self.window_close[self.nN - 1] = self.window_close[0]
            self.service_time[self.nN - 1] = self.service_time[0]
            self.n_tw[self.nN - 1] = self.n_tw[0]

            # Populate distance and travel_cost matrices
            self.distance = [[0.0] * self.nN for _ in range(self.nN)]
            for i in range(self.nN):
                for j in range(self.nN):
                    dx = self.x_coord[i] - self.x_coord[j]
                    dy = self.y_coord[i] - self.y_coord[j]
                    dist = int(10 * math.sqrt(dx * dx + dy * dy))
                    self.distance[i][j] = 1.0 * dist
                self.service_time[i] *= 10
                self.window_open[i] *= 10
                self.window_close[i] *= 10
            self.travel_cost = self.distance

            self.q = 700

            # Calculate T_max, T_avg, and demand_sum
            self.T_max = max(self.window_close) if self.window_close else 0.0
            self.T_avg = sum(
                self.window_close[i] - self.window_open[i]
                for i in range(1, self.nN - 1)
            ) / (self.nN - 2)
            self.demand_sum = sum(self.demand[1 : self.nN - 1])

import numpy as np

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


def convert_instance_data(instance_data: InstanceData) -> VRPTWInstance:
    """Convert Solomon format InstanceData to VRPTWInstance"""
    return VRPTWInstance(
        num_nodes=instance_data.nN,
        demands=np.array(instance_data.demand),
        time_windows=np.column_stack(
            [instance_data.window_open, instance_data.window_close]
        ),
        distances=np.array(instance_data.distance),
        capacities=instance_data.q,
        service_times=np.array(instance_data.service_time),
    )
