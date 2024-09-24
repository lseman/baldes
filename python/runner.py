import os
import random
import sys

# Add the build directory to the Python path
sys.path.append(os.path.abspath('../build'))

# Now you can import the BALDES module
import baldes

# Define jobs
jobs = [baldes.VRPJob() for _ in range(102)]
num_intervals = 1

# Set random bounds for each job
id = 0
for job in jobs:
    job.lb = [random.randint(0, 9000) for _ in range(num_intervals)]  # Set random lower bounds
    job.ub = [random.randint(lb + 1, 10000) for lb in job.lb]  # Set random upper bounds greater than lb
    job.duration = random.randint(1, 100)  # Set random duration
    job.cost = random.randint(1, 100)  # Set random cost
    job.start_time = random.randint(0, 10000)  # Set random start time
    job.end_time = random.randint(job.start_time, 10000)  # Set random end time greater than start time
    job.demand = random.randint(1, 100)  # Set random demand
    job.consumption = [random.randint(1, 100) for _ in range(num_intervals)]  # Set random consumption
    job.set_location(random.randint(0, 100), random.randint(0, 100))  # Set random location
    job.id = id
    id += 1

# Create fake distance matrix with size equal to the number of jobs
distances = [[random.randint(1, 100) for _ in range(len(jobs))] for _ in range(len(jobs))]

# Initialize BucketGraph using these jobs
bg = baldes.BucketGraph(jobs, 12000, 1)

# Create random duals with size equal to the number of jobs
duals = [random.random() for _ in range(len(jobs))]

# Set the distance matrix, adjacency list, and duals
bg.set_distance_matrix(distances)
bg.set_adjacency_list()
bg.set_duals(duals)
bg.setup()

# Call the solve method
labels = bg.solve()