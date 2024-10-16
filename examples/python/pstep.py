# %% [markdown]
# # ATSP

# %%
import os
import sys
import numpy as np
import random
# Add the build directory to the Python path
sys.path.append(os.path.abspath("../../build"))

# %%


class ATSPInstance:
    def __init__(self, filename):
        self.name = ""
        self.dimension = 0
        self.distance_matrix = None
        self.parse_instance(filename)

    def parse_instance(self, filename):
        edge_weights = []
        reading_edge_weights = False

        with open(filename, "r") as file:
            for line in file:
                line = line.strip()

                if "NAME:" in line:
                    self.name = line.split()[1]
                elif "DIMENSION:" in line:
                    self.dimension = int(line.split()[1])
                elif line == "EDGE_WEIGHT_SECTION":
                    reading_edge_weights = True
                elif reading_edge_weights:
                    if line == "EOF":
                        break
                    edge_weights.extend(map(int, line.split()))

        self.distance_matrix = np.array(edge_weights).reshape(
            (self.dimension, self.dimension)
        )

    def print_instance(self):
        print(f"ATSP Instance: {self.name}")
        print(f"Dimension: {self.dimension}")
        print("Distance Matrix:")
        print(self.distance_matrix)


# %%
def solve_atsp(instance):
    try:
        # Assume que instance.dimension e instance.distance_matrix estão disponíveis
        n = instance["dimension"]
        dist = instance["distance_matrix"]

        # Cria o ambiente e o modelo
        model = gp.Model("ATSP")

        # Cria as variáveis x[i][j] binárias
        x = [
            [
                model.addVar(vtype=GRB.BINARY, obj=dist[i][j]) if i != j else None
                for j in range(n)
            ]
            for i in range(n)
        ]

        # Cria as variáveis contínuas u[i] para a formulação MTZ
        u = [
            model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY) for i in range(n)
        ]

        # Define o sentido de minimização no modelo
        model.modelSense = GRB.MINIMIZE

        # Restrições de grau: Cada cidade deve ter exatamente um arco de entrada e um de saída
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i][j] for j in range(n) if i != j) == 1, name=f"out_{i}"
            )
            model.addConstr(
                gp.quicksum(x[j][i] for j in range(n) if i != j) == 1, name=f"in_{i}"
            )

        # Restrições MTZ para eliminação de subciclos
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(u[i] - u[j] + (n - 1) * x[i][j] <= n - 2)

        # Restrições de limites para u[i]
        for i in range(1, n):
            model.addConstr(u[i] >= 1)
            model.addConstr(u[i] <= n - 1)

        # Fixar u[0] a 0
        model.addConstr(u[0] == 0)

        # Otimizar o modelo
        model.optimize()

        # Verificar se uma solução ótima foi encontrada
        if model.status == GRB.OPTIMAL:
            print(f"Comprimento ótimo da rota: {model.objVal}")
            print("Rota ótima:")

            # Imprimir a rota a partir do vértice 0
            current_vertex = 0
            visited = [False] * n
            for count in range(n):
                visited[current_vertex] = True
                for j in range(n):
                    if current_vertex != j and x[current_vertex][j].x > 0.5:
                        print(f"{current_vertex} -> {j}", end=" ")
                        current_vertex = j
                        break
            print()

            # Retorna a solução na forma de uma matriz 2D
            solution = [
                [x[i][j].x if i != j else None for j in range(n)] for i in range(n)
            ]
            return solution
        else:
            print("Nenhuma solução ótima encontrada.")

    except gp.GurobiError as e:
        print(f"Erro no Gurobi: {e.errno}, {e.message}")
    except Exception as e:
        print(f"Erro durante a otimização: {e}")


# %%


def generate_tsp_instance(num_nodes):
    # Create a random symmetric cost matrix with values between 1 and 100
    cost_matrix = np.random.randint(1, 101, size=(num_nodes, num_nodes))

    # Make the cost matrix symmetric by mirroring the upper triangular part
    cost_matrix = (cost_matrix + cost_matrix.T) // 2

    # Set diagonal to zero (no cost to travel from a node to itself)
    np.fill_diagonal(cost_matrix, 0)

    return cost_matrix


# Generate TSP instance with 11 nodes
num_nodes = 11
cost_matrix = generate_tsp_instance(num_nodes)

# Print the cost matrix
print("Cost Matrix:")
print(cost_matrix)

# create new ATSPInstance
instancia = {}
instancia["dimension"] = num_nodes
instancia["distance_matrix"] = cost_matrix

# %%
# instancia.print_instance()
# x = solve_atsp(instancia)


# %%
import numpy as np


def generate_initial_paths(
        n, p1, p2, paths, path_costs, firsts, lasts, cost_matrix, num_paths
):
    for i in range(num_paths):
        p = p1 if i % 2 == 0 else p2  # Alternate between p1 and p2
        start = (
            0 if i < num_paths / 2 else random.randint(1, n - 1)
        )  # Half paths start at 0, others at random nodes
        path = [start]
        current = start
        total_cost = 0.0

        for step in range(1, min(p, n)):
            candidates = [
                (node, cost_matrix[current][node])
                for node in range(n)
                if node not in path
            ]

            if candidates:
                # Sort candidates by cost
                candidates.sort(key=lambda x: x[1])

                # Select from the top 3 candidates to introduce randomness
                k = min(3, len(candidates))
                chosen_idx = random.randint(0, k - 1)

                next_node = candidates[chosen_idx][0]
                min_cost = candidates[chosen_idx][1]

                path.append(next_node)
                current = next_node
                total_cost += min_cost
            else:
                break  # No more nodes to visit

        paths.append(path)
        path_costs.append(total_cost)
        firsts.append(start)
        lasts.append(current)


# %%
import gurobipy as gp
from gurobipy import GRB


def solve_tsp(paths, nodes, path_costs, firsts, lasts, cost_matrix, first_time=False):
    n = 11  # Number of nodes
    p1 = 6  # Number of steps in some paths
    p2 = 6  # Number of steps in other paths
    num_paths = 5000  # Number of initial paths to generate

    if first_time:
        generate_initial_paths(
            n, p1, p2, paths, path_costs, firsts, lasts, cost_matrix, num_paths
        )

    #print(paths)
    # Create an environment and model
    env = gp.Env(empty=True)
    env.setParam("LogFile", "tsp.log")
    env.start()
    model = gp.Model(env=env)

    R = len(paths)  # Number of p-steps
    x = {}  # Binary variables for each p-step
    u = {}  # Continuous variables for each node

    # Create binary variables x_r for each p-step r
    for i in range(R):
        varname = f"x_{i}"
        cost = path_costs[i]
        x[i] = model.addVar(vtype=GRB.CONTINUOUS, obj=cost, name=varname)

    # Create continuous variables u_i for each node i
    for i in range(n):
        name = f"u_{i}"
        u[i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=name)

    # Objective function: Minimize the total cost of the selected p-steps
    model.setObjective(
        gp.quicksum(path_costs[i] * x[i] for i in range(R)), GRB.MINIMIZE
    )

    # Constraints (3.3) and (3.4)
    three_two = {}
    three_three = {}
    for i in range(1, n):
        in_constraint = gp.LinExpr()
        out_constraint = gp.LinExpr()

        for r in range(R):
            if firsts[r] == i:
                in_constraint += x[r]
            if lasts[r] == i:
                in_constraint -= x[r]
            if i in paths[r] and i != lasts[r]:
                out_constraint += x[r]

        three_two[i] = model.addConstr(in_constraint == 0, name=f"33_{i}")
        three_three[i] = model.addConstr(out_constraint == 1, name=f"34_{i}")

    # Constraints (3.5): Avoid sub-tours using MTZ constraints
    three_five_constraints_matrix = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                r_sum = gp.LinExpr()
                for r in range(R):
                    for k in range(len(paths[r]) - 1):
                        if paths[r][k] == i and paths[r][k + 1] == j:
                            r_sum += x[r]

                three_five_constraints_matrix[(i, j)] = model.addConstr(
                    u[j] >= u[i] + n * r_sum - (n - 1), name=f"mtz_{i}_{j}"
                )

    # Optimize the model
    model.optimize()

    # Capture dual variables
    three_two_duals = [0.0] * n
    three_three_duals = [0.0] * n
    three_five_duals = [[0.0] * n for _ in range(n)]

    if model.status == GRB.OPTIMAL:
        print("Optimal solution found!")
        print(f"Objective value: {model.objVal}")
        for i in range(R):
            if x[i].X > 1e-4:
                print(f"x[{i}] = {x[i].X}")

        # Get dual variables for the constraints
        for i in range(1, n):
            three_two_duals[i] = three_two[i].Pi
            three_three_duals[i] = three_three[i].Pi

        for i in range(n):
            for j in range(n):
                if i != j:
                    three_five_duals[i][j] = three_five_constraints_matrix[(i, j)].Pi
    else:
        print("No optimal solution found.")

    # Write the model to a file
    model.write("tsp.lp")

    return three_two_duals, three_three_duals, three_five_duals


# %%
# Example usage
paths = []
nodes = []
path_costs = []
firsts = []
lasts = []
cost_matrix = instancia["distance_matrix"]
three_two, three_three, three_five = solve_tsp(
    paths, nodes, path_costs, firsts, lasts, cost_matrix, first_time=True
)

# %%
import os
import random
import sys

# Add the build directory to the Python path
sys.path.append(os.path.abspath("../build"))

import baldes

pstep_duals = baldes.PSTEPDuals()
three_two_tuples = [(i, value) for i, value in enumerate(three_two)]
pstep_duals.set_threetwo_dual_values(three_two_tuples)

three_three_tuples = [(i, value) for i, value in enumerate(three_three)]
pstep_duals.set_threethree_dual_values(three_three_tuples)

arc_duals_tuples = []
for i in range(len(three_five)):
    for j in range(len(three_five[i])):
        if i != j:  # Skip diagonal or add specific condition if needed
            arc_duals_tuples.append(((i, j), three_five[i][j]))

pstep_duals.set_arc_dual_values(arc_duals_tuples)

options = baldes.BucketOptions()
options.depot = 3
options.end_depot = 11
options.max_path_size = 5

# %%
nodes = [baldes.VRPNode() for _ in range(11)]
num_intervals = 1

# Set random bounds for each node
id = 0
for node in nodes:
    node.lb = [0]  # Set random lower bounds
    node.ub = [100]  # Set random upper bounds greater than lb
    node.duration = 0  # Set random duration
    node.cost = 0
    node.demand = 0
    node.consumption = [0]  # Set random consumption
    node.set_location(
        random.randint(0, 100), random.randint(0, 100)
    )  # Set random location
    node.id = id
    id += 1

# %%
print(nodes)
print(len(nodes))
# Initialize BucketGraph using these nodes
bg = baldes.BucketGraph(nodes, 100, 1)
bg.setOptions(options)

# Create random duals with size equal to the number of nodes
duals = [0 for _ in range(len(nodes) + 2)]
print(duals)

# Set the distance matrix, adjacency list, and duals
print("Setting distance matrix")
bg.set_distance_matrix(instancia["distance_matrix"])
bg.set_adjacency_list()

print("Setting duals")
bg.set_duals(duals)

print("Setup")
bg.setup()

# %%

print("Solving")
paths = bg.solvePSTEP()

print("Number of paths:", len(paths))

for path in paths:
    print("Path:", path.nodes_covered)
