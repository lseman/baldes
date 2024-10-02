import os
import random
import sys

# Add the build directory to the Python path
sys.path.append(os.path.abspath('../../build'))

# Now you can import the BALDES module
import baldes

# Define nodes
nodes = [baldes.VRPNode() for _ in range(102)]
num_intervals = 1

# Set random bounds for each node
id = 0
for node in nodes:
    node.lb = [random.randint(0, 9000) for _ in range(num_intervals)]  # Set random lower bounds
    node.ub = [random.randint(lb + 1, 10000) for lb in node.lb]  # Set random upper bounds greater than lb
    node.duration = random.randint(1, 100)  # Set random duration
    node.cost = random.randint(1, 100)  # Set random cost
    node.start_time = random.randint(0, 10000)  # Set random start time
    node.end_time = random.randint(node.start_time, 10000)  # Set random end time greater than start time
    node.demand = random.randint(1, 100)  # Set random demand
    node.consumption = [random.randint(1, 100) for _ in range(num_intervals)]  # Set random consumption
    node.set_location(random.randint(0, 100), random.randint(0, 100))  # Set random location
    node.id = id
    id += 1

# Create fake distance matrix with size equal to the number of nodes
distances = [[random.randint(1, 100) for _ in range(len(nodes))] for _ in range(len(nodes))]

# Initialize BucketGraph using these nodes
bg = baldes.BucketGraph(nodes, 12000, 1)

# Create random duals with size equal to the number of nodes
duals = [random.random() for _ in range(len(nodes))]

# Set the distance matrix, adjacency list, and duals
bg.set_distance_matrix(distances)
bg.set_adjacency_list()
bg.set_duals(duals)
bg.setup()

# Call the solve method
labels = bg.solve()