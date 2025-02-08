# %% [markdown]
# # ATSP

# %%
import os
import sys
import numpy as np
import random
# Add the build directory to the Python path
#sys.path.append(os.path.abspath("../../build"))

# %%

# define random seed

class ATSPInstance:
    def __init__(self, filename):
        self.name = ""
        self.dimension = 0
        self.distance_matrix = None
        self.parse_instance(filename)

    # parse tsp file
    # this need to be a padron "tsp file"
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
        #print(self.distance_matrix)
        for line in self.distance_matrix:
            for col in line:
                print(f"{col:5}", end=" ")
            print()


# %%
def solve_atsp(instance):
    try:
        # Get the number of nodes and the distance matrix
        n = instance["dimension"] - 1
        dist = instance["distance_matrix"]

        # Create the model
        model = gp.Model("ATSP")
        # Set mute
        model.setParam('OutputFlag', 0)

        # Create binary variables x[i][j] for each pair of nodes (i != j)
        x = [
            [
                model.addVar(vtype=GRB.BINARY, obj=dist[i][j]) if i != j else None
                for j in range(n)
            ]
            for i in range(n)
        ]

        # Create continuous variables u[i] for MTZ formulation
        u = [
            model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=n - 1) for i in range(n)
        ]

        # Set the model to minimize the objective
        model.modelSense = GRB.MINIMIZE

        # Degree constraints: Each city has one incoming and one outgoing arc
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i][j] for j in range(n) if i != j) == 1, name=f"out_{i}"
            )
            model.addConstr(
                gp.quicksum(x[j][i] for j in range(n) if i != j) == 1, name=f"in_{i}"
            )

        # MTZ constraints for eliminating sub-tours
        for i in range(0, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(u[i] - u[j] + n * x[i][j] <= n - 1)

        # Fix u[0] to 0 (starting node for MTZ constraints)
        #model.addConstr(u[0] == 0)

        # Optimize the model
        model.optimize()
        print("Objective value: ", model.objVal)
                
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


import numpy as np

import numpy as np

def generate_tsp_instance(num_nodes: int):
    # Create a random symmetric cost matrix with values between 1 and 100
    cost_matrix = np.random.randint(1, 101, size=(num_nodes, num_nodes))

    # Make the cost matrix symmetric by mirroring the upper triangular part
    cost_matrix = (cost_matrix + cost_matrix.T) // 2

    # Set diagonal to zero (no cost to travel from a node to itself)
    np.fill_diagonal(cost_matrix, 0)

    # Add an extra row and column to the matrix for the end depot
    cost_matrix = np.vstack((cost_matrix, cost_matrix[0:1]))  # Copy first row to the last row
    cost_matrix = np.hstack((cost_matrix, cost_matrix[:, 0:1]))  # Copy first column to the last column

    return cost_matrix


def generate_tsp_instance2(num_nodes):
    my_pos = { i : ( random.randint(0,1000), random.randint(0,1000)) for i in range(num_nodes) }

    cost_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            cost_matrix[i][j] = np.sqrt( (my_pos[i][0] - my_pos[j][0])**2 + (my_pos[i][1] - my_pos[j][1])**2 )
        
    return cost_matrix, my_pos

# %%
import numpy as np


# PSTEP
import random

def generate_initial_paths(
        n, p1, p2, paths, path_costs, firsts, lasts, cost_matrix, num_paths
):
    for i in range(num_paths):
        p = p1 if i % 2 == 0 else p2  # Alternate between p1 and p2
        start = (
            0 if i < num_paths / 2 else random.randint(1, n - 2)
        )  # Half paths start at 0, others at random nodes (excluding n-1)
        path = [start]
        current = start
        total_cost = 0.0

        for step in range(1, min(p, n - 1)):
            # Determine candidates based on the starting node
            if start == 0:
                # Paths starting at 0 should exclude n-1
                candidates = [
                    (node, cost_matrix[current][node])
                    for node in range(n - 1)  # Exclude n-1 from candidates
                    if node not in path
                ]
            else:
                # Paths not starting at 0 will include n-1 as the last node, so we exclude it here
                candidates = [
                    (node, cost_matrix[current][node])
                    for node in range(n - 1)  # Exclude n-1 for now, will add it as last
                    if node not in path
                ]

            if candidates:
                # Sort candidates by cost
                candidates.sort(key=lambda x: x[1])

                # Select from the top 3 candidates to introduce randomness
                k = min(3, len(candidates))
                chosen_idx = random.randint(0, k - 1)

                next_node = candidates[chosen_idx][0]
                # if next_node == 0, retry selection
                while next_node == 0:
                    chosen_idx = random.randint(0, k - 1)
                    next_node = candidates[chosen_idx][0]
                min_cost = candidates[chosen_idx][1]

                path.append(next_node)
                current = next_node
                total_cost += min_cost
            else:
                break  # No more nodes to visit

        # If path does not start at 0, make sure it ends with n-1
        if start != 0:
            # remove last cost from total
            total_cost -= cost_matrix[path[-2]][path[-1]]
            # remove last element from path
            path.pop()
            path.append(n - 1)
            total_cost += cost_matrix[path[-2]][path[-1]]

        paths.append(path)
        path_costs.append(total_cost)
        firsts.append(start)
        lasts.append(path[-1])


# %%
import gurobipy as gp
from gurobipy import GRB

# Create an environment and model
env = gp.Env(empty=True)
#env.setParam("LogFile", "tsp.log")
env.start()

# arrumar o imput correto!
def solve_tsp(paths:list, nodes:list[int], path_costs:list, firsts:list[int], lasts:list[int], 
              cost_matrix, first_time=False, num_paths=250):
    #n = 11  # Number of nodes
    #p1 = 6  # Number of steps in some paths
    #p2 = 6  # Number of steps in other paths
    #num_paths = 5000  # Number of initial paths to generate
    
    n =  len(nodes)

    if first_time:
        if (n) % 2 == 0:
            p1 = 4
            p2 = 4
        else:
            p1 = 4
            p2 = 4
        generate_initial_paths(
            n, p1, p2, paths, path_costs, firsts, lasts, cost_matrix, num_paths
        )


    

    model = gp.Model(env=env)
    model.setParam('OutputFlag', 0)
    # set method
    model.setParam('Method', 2)
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

    model.update()
    # Objective function: Minimize the total cost of the selected p-steps
    model.setObjective(
        gp.quicksum(path_costs[i] * x[i] for i in range(R)), GRB.MINIMIZE
    )

    # Constraints (3.3) and (3.4)
    three_two = {}
    three_three = {}
    for i in range(1, n-1):
        in_constraint = gp.LinExpr()
        out_constraint = gp.LinExpr()

        for r in range(R):
            if firsts[r] == i:
                in_constraint += 1.0*x[r]
            if lasts[r] == i:
                in_constraint += -1.0*x[r]
            if i in paths[r] and i != lasts[r]:
                out_constraint += x[r]

        #a_r_r
        three_two[i] = model.addConstr(in_constraint == 0, name=f"33_{i}")
        #e_r_i
        three_three[i] = model.addConstr(out_constraint == 1, name=f"34_{i}")

    # Constraints (3.5): Avoid sub-tours using MTZ constraints
    three_five_constraints_matrix = {}
    for i in range(0, n-1):
        for j in range(1, n):
            if i != j:
                r_sum = gp.LinExpr()
                for r in range(R):
                    for k in range(len(paths[r]) - 1):
                        if paths[r][k] == i and paths[r][k + 1] == j:
                            r_sum += x[r]

                three_five_constraints_matrix[(i, j)] = model.addConstr(
                    u[i] - u[j] + (n - 1) * r_sum <= (n - 2), name=f"mtz_{i}_{j}"
                )

    for i in range(1, n):
       model.addConstr(u[i] >= 0)
       model.addConstr(u[i] <= n - 1)

    # Fixar u[0] a 0
    model.addConstr(u[0] == 0)
    model.addConstr(u[n-1] == n-1)
    # define u[11] = 11
    

    # Optimize the model
    model.optimize()
    """
    pegar duais
    pegar dual que indica e ver se eu to no meio da rota ou no fim da rota
    """
    # Capture dual variables
    three_two_duals = [0.0] * n
    three_three_duals = [0.0] * n
    three_five_duals = [[0.0] * n for _ in range(n)]

    if model.status == GRB.OPTIMAL:
        # print("######## Optimal solution found! ########")
        print(f"Objective value: {model.objVal}")
        # for i in range(r):
        #     if x[i].X > 1e-4:
        #         print(f"x[{i}] = {x[i].X}")
        #         print(f"Path {i} =  {paths[i]}")

        # Get dual variables for the constraints
        for i in range(1, n-1):
            three_two_duals[i] = three_two[i].Pi
            three_three_duals[i] = -three_three[i].Pi

        
        for i in range(0,n-1):
            for j in range(1,n):
                if i != j:
                    three_five_duals[i][j] = (n-1) * three_five_constraints_matrix[(i, j)].Pi
        
        objetivo = model.ObjVal
        # recover integrality on x variables
        for i in range(R):
            x[i].vtype = GRB.BINARY
        model.update()
        model.optimize()
        try:
            print(f"Integer Objective value: {model.objVal}")
            #print("######## Integer solution found! ########")
            full_paths = []
            # Step 1: Extract the paths with x[i] = 1.0
            for i in range(r):
                if x[i].X > 1e-4:
                    print(f"x[{i}] = {x[i].X}")
                    print(f"Path {i} = {paths[i]}")
                    full_paths.append(paths[i])

            # Step 2: Form the final path starting from the path that contains node 0
            final_path = []
            for i, path in enumerate(full_paths):
                if path[0] == 0:  # Find the path that starts with node 0
                    final_path = path
                    full_paths.pop(i)
                    break

            # Step 3: Connect paths based on overlapping nodes
            while full_paths:
                last_node = final_path[-1]
                connected = False
                for i, path in enumerate(full_paths):
                    if path[0] == last_node:  # Check if the next path starts with the last node
                        final_path += path[1:]  # Append the rest of the path, excluding the first node
                        full_paths.pop(i)  # Remove the used path from the list
                        connected = True
                        break

                if not connected:
                    print("Error: Could not connect the paths!")
                    break

            #print("Final Path:", final_path)
            # compute cost
            cost = 0
            for i in range(len(final_path)-1):
                cost += cost_matrix[final_path[i]][final_path[i+1]]
            # print("Final Path Cost:", cost)

            
        except:
            # print("######## No optimal solution found. ########")
            pass

        # relax again
        for i in range(R):
            x[i].vtype = GRB.CONTINUOUS
        model.update()


        
        """
        # Solução ótima foi encontrada
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
        """

        return three_two_duals, three_three_duals, three_five_duals, objetivo
    else:
        print("######## No relaxation solution found. ########")

    # Write the model to a file
    ################
    #model.write("tsp.lp")
    ################

    #print('Three_two Dual',three_two[1].Pi)
    #print('Three_three Dual',three_three[1].Pi)
    #print(three_five[1].Pi)
    

    return three_two_duals, three_three_duals, three_five_duals, 0

# %%
import os
import random
import sys

# Add the build directory to the Python path
#sys.path.append(os.path.abspath("build"))

import pybaldes as baldes
# %%
# paro quando o custo positivo nao ser menor que o custo...
# geração de colulas....

def tsp_baldes(num_nodes=12):
    # Generate TSP instance with 11 nodes
    #cost_matrix, nodes_pos = generate_tsp_instance2(num_nodes)
    cost_matrix = generate_tsp_instance(num_nodes-1)

    # Print the cost matrix
    print("Cost Matrix:")
    print(cost_matrix)

    # create new ATSPInstance
    instancia = {}
    instancia["dimension"] = num_nodes
    instancia["distance_matrix"] = cost_matrix
    paths = []
    nodes = [_ for _ in range(num_nodes)]
    path_costs = []
    firsts = []
    lasts = []
    cost_matrix = instancia["distance_matrix"]
    
    print("Distancia TSP: ")
    solve_atsp(instancia)
    #baldes config
    options = baldes.BucketOptions()

    options.depot = 0
    options.end_depot = num_nodes
    options.max_path_size = 5


    nodes_baldes = [baldes.VRPNode() for _ in range(num_nodes)]
    num_intervals = 1

    # Set random bounds for each node
    id = 0
    for node in nodes_baldes:
        node.lb = [0]  # Set random lower bounds
        node.ub = [1000]  # Set random upper bounds greater than    my_pos = { i : ( random.random(), random.random() ) for i in range(num_nodes) } lb
        node.duration = 0  # Set random duration
        node.cost = 0
        node.demand = 0
        node.consumption = [0]  # Set random consumption
        node.set_location(random.randint(0, 100), random.randint(0, 100))  # Set random location
        node.id = id
        id += 1


    # Initialize BucketGraph using these nodes
    bg = baldes.BucketGraph(nodes_baldes, 10000, 1)

    bg.setOptions(options)
    # Create random duals with size equal to the number of nodes
    duals = [0 for _ in range(len(nodes_baldes))]
    bg.set_distance_matrix(instancia["distance_matrix"])
    bg.set_duals(duals)

    #bg.set_adjacency_list()
    bg.setup()   

    nodes = [_ for _ in range(num_nodes)]
    
    # -----------------------------
    # first_time 
    three_two, three_three, three_five, obj = solve_tsp(
        paths, nodes, path_costs, firsts, lasts, cost_matrix, first_time=True
    )
    best = [obj]
    
    for z in range(3):
        print("------------------------------------------")
        print("Iteração: ", z)


        #baldes
        pstep_duals = baldes.PSTEPDuals()
        three_two_tuples = [(i, value) for i, value in enumerate(three_two)]
        pstep_duals.set_threetwo_dual_values(three_two_tuples)
        # print("Three two tuples", three_two_tuples)

        three_three_tuples = [(i, value) for i, value in enumerate(three_three)]
        pstep_duals.set_threethree_dual_values(three_three_tuples)
        # print("Three three tuples", three_three_tuples)

        arc_duals_tuples = []
        for i in range(len(three_five)):
            for j in range(len(three_five[i])):
                if i != j:  # Skip diagonal or add specific condition if needed
                    arc_duals_tuples.append(((i, j), three_five[i][j]))

        pstep_duals.set_arc_dual_values(arc_duals_tuples)
        
        ## **************LOOP*************
        for i in range(num_nodes-1): #not the last
            for j in range(1,num_nodes): #not de firs
                for z in [4]:
                    if i == j:
                        continue
                    


                    # alterar inicio e fim conforme o while avança 
                    # saber inicio e fim de caminho

                    options.depot = i
                    options.end_depot = j
                    options.max_path_size = z


                    #print("------------------------------------------")
                    #print("Solving for depot", i, "end depot", j)
                    bg.setPSTEPDuals(pstep_duals)
                    bg.setOptions(options)
                    bg.setup()

                    paths_baldes = bg.solvePSTEP()
                    #print(paths_baldes)

                    #print("Number of paths:", len(paths_baldes))
                    if len(paths_baldes) == 0:
                        #print("No paths found")
                        break
                    
                    for path in paths_baldes:
                        #print("Path:", path.nodes_covered)
                        #print("Path:", path)
                        #print("Path:", path.cost)
                        #print("Path first:", path.nodes_covered[0])
                        #print("Path last:", path.nodes_covered[-1])
                        #print(path.nodes_covered)

                        paths.append(path.nodes_covered)
                        path_costs.append(path.real_cost)
                        firsts.append(path.nodes_covered[0])
                        lasts.append(path.nodes_covered[-1])


        #print("Paths size", len(paths))
        # tsp solve
        three_two, three_three, three_five, obj = solve_tsp(
            paths, nodes, path_costs, firsts, lasts, cost_matrix, first_time=False)

        best.append(obj)

        #distancia TSP 

    


        
# %%

#br17 = ATSPInstance("br17.atsp")
#br17.print_instance()
# %%
tsp_baldes(10)
#print(generate_tsp_instance2(11)[0])
# %%
