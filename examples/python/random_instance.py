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
    node.lb = [random.randint(0, 9000) for _ in range(num_intervals)]  # Set random integer lower bounds
    node.ub = [random.randint(int(lb) + 1, 10000) for lb in node.lb]  # Ensure ub > lb (integer bounds)
    node.duration = random.randint(1, 100)  # Set random integer duration
    node.cost = random.uniform(1.0, 100.0)  # Set random float cost
    node.start_time = random.randint(0, 10000)  # Set random integer start time
    node.end_time = random.randint(node.start_time + 1, 10000)  # Ensure end_time > start_time (integer)
    node.demand = random.uniform(1.0, 100.0)  # Set random float demand
    node.consumption = [random.uniform(1.0, 100.0) for _ in range(num_intervals)]  # Set random float consumption
    node.set_location(random.uniform(0, 100), random.uniform(0, 100))  # Set random float location
    node.id = id
    id += 1

# Create symmetric fake distance matrix with size equal to the number of nodes
distances = [[0] * len(nodes) for _ in range(len(nodes))]
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        dist = random.uniform(1.0, 100.0)  # Use uniform for float distances
        distances[i][j] = dist
        distances[j][i] = dist

# Initialize BucketGraph using these nodes
bg = baldes.BucketGraph(nodes, 12000, 1)

# Create random duals with size equal to the number of nodes (floats)
duals = [random.uniform(0.0, 1.0) for _ in range(len(nodes))]

# Set the distance matrix, adjacency list, and duals
bg.set_distance_matrix(distances)
bg.set_adjacency_list()
bg.set_duals(duals)

# Perform the setup
bg.setup()

# Output to check if everything initialized properly
print("Nodes and BucketGraph initialized successfully!")

# Call the solve method
labels = bg.solve()
print(labels)
