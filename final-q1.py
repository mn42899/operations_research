import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

delivery = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/delivery.csv')
delivery

print("="*60 + "\n")
print("1E")
print("="*60 + "\n")

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# Load the data
url = 'https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/delivery.csv'
delivery = pd.read_csv(url)
print("Data loaded successfully:")
print(delivery.head())

# Extract the necessary data
num_customers = delivery.shape[0]  # 15 customers
weights = delivery['Size'].values  # Package weights
depot_distances = delivery['Depot'].values  # Distances from depot

# Create distance matrix between customers
distances = np.zeros((num_customers, num_customers))
for i in range(num_customers):
    for j in range(num_customers):
        if i != j:
            distances[i, j] = delivery[f'Customer_{j+1}'].iloc[i]

# Problem parameters
num_vans = 3
max_weight_per_van = 15000  # 15,000 lbs
max_distance_per_van = 254  # 254 km

# Display information about the data
print(f"\nNumber of customers: {num_customers}")
print(f"Total weight to deliver: {sum(weights)} lbs")
print(f"Package weights (lbs): {weights}")
print(f"Number of vans available: {num_vans}")
print(f"Maximum weight per van: {max_weight_per_van} lbs")
print(f"Maximum distance per van: {max_distance_per_van} km")

# Create the Gurobi model
model = gp.Model("FedEx_VRP")

# Decision variables
# x[i,v] = 1 if customer i is assigned to van v
x = model.addVars(num_customers, num_vans, vtype=GRB.BINARY, name="x")

# y[v] = total weight assigned to van v
y = model.addVars(num_vans, vtype=GRB.CONTINUOUS, name="y")

# z = maximum difference in load between any two vans
z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

# Constraint: Each customer must be assigned to exactly one van
for i in range(num_customers):
    model.addConstr(gp.quicksum(x[i,v] for v in range(num_vans)) == 1, f"Customer_{i+1}_Assignment")

# Constraint: Calculate total weight for each van
for v in range(num_vans):
    model.addConstr(y[v] == gp.quicksum(weights[i] * x[i,v] for i in range(num_customers)), f"Van_{v+1}_Weight")

# Constraint: Maximum weight difference between any two vans
for v1 in range(num_vans):
    for v2 in range(v1+1, num_vans):
        model.addConstr(y[v1] - y[v2] <= z, f"Weight_Diff_{v1+1}_{v2+1}_Plus")
        model.addConstr(y[v2] - y[v1] <= z, f"Weight_Diff_{v1+1}_{v2+1}_Minus")

# Constraint: Maximum weight per van is 15,000 lbs
for v in range(num_vans):
    model.addConstr(y[v] <= max_weight_per_van, f"Max_Weight_Van_{v+1}")

# Constraint: No more than two of customers 7-9 can be assigned to the same van
for v in range(num_vans):
    model.addConstr(x[6,v] + x[7,v] + x[8,v] <= 2, f"Constraint_Customers_7_to_9_Van_{v+1}")

# Constraint: Customers 10-12 must all be assigned to the same van
for v in range(num_vans):
    model.addConstr(x[9,v] == x[10,v], f"Constraint_Customers_10_11_Same_Van_{v+1}")
    model.addConstr(x[10,v] == x[11,v], f"Constraint_Customers_11_12_Same_Van_{v+1}")

# Constraint: If customer 1 is assigned to a van, at least one of 13 or 14 must also be assigned
for v in range(num_vans):
    model.addConstr(x[0,v] <= x[12,v] + x[13,v], f"Constraint_Customer_1_13_14_Van_{v+1}")

# Constraint: Customer 2 cannot be with customers 3, 4, and 5
for v in range(num_vans):
    model.addConstr(x[1,v] + x[2,v] + x[3,v] + x[4,v] <= 3, f"Constraint_Customer_2_345_Van_{v+1}")

# Constraint: No van can make more than 5 deliveries
for v in range(num_vans):
    model.addConstr(gp.quicksum(x[i,v] for i in range(num_customers)) <= 5, f"Max_Deliveries_Van_{v+1}")

# Constraint: Battery range - simplified approach without subtour elimination
for v in range(num_vans):
    model.addConstr(
        gp.quicksum(2 * depot_distances[i] * x[i,v] for i in range(num_customers)) <= max_distance_per_van,
        f"Max_Distance_Van_{v+1}"
    )

# Set the objective function: Minimize the maximum weight difference
model.setObjective(z, GRB.MINIMIZE)

# Print model statistics
print(f"\nModel statistics:")
print(f"Number of variables: {model.numVars}")
print(f"Number of binary variables: {sum(1 for v in model.getVars() if v.vtype == GRB.BINARY)}")
print(f"Number of constraints: {model.numConstrs}")

# Optimize the model
print("\nSolving the model...")
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("\nOptimal solution found!")
    print(f"Optimal objective value (minimum maximum weight difference): {model.objVal} lbs")
    
    # Print the assignment of customers to vans
    for v in range(num_vans):
        assigned_customers = [i+1 for i in range(num_customers) if x[i,v].X > 0.5]
        van_weight = sum(weights[i] for i in range(num_customers) if x[i,v].X > 0.5)
        approx_distance = sum(2 * depot_distances[i] for i in range(num_customers) if x[i,v].X > 0.5)
        
        print(f"\nVan {v+1}:")
        print(f"  Customers: {assigned_customers}")
        print(f"  Total Weight: {van_weight} lbs")
        print(f"  Approx. Distance: {approx_distance} km")
        print(f"  Number of Deliveries: {len(assigned_customers)}")
    
    # Calculate the total weight to verify the solution
    total_weight = sum(weights)
    print(f"\nTotal weight of all deliveries: {total_weight} lbs")
    print(f"Average weight per van: {total_weight/num_vans:.2f} lbs")
    
    # Calculate the weight difference between each pair of vans to verify the objective
    van_weights = [sum(weights[i] for i in range(num_customers) if x[i,v].X > 0.5) for v in range(num_vans)]
    max_diff = max(abs(van_weights[i] - van_weights[j]) for i in range(num_vans) for j in range(i+1, num_vans))
    print(f"Maximum weight difference (verification): {max_diff} lbs")
    

print("="*60 + "\n")
print("1G")
print("="*60 + "\n")

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# Load the data
url = 'https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/delivery.csv'
delivery = pd.read_csv(url)
print("Data loaded successfully:")
print(delivery.head())

# Extract the necessary data
num_customers = delivery.shape[0]  # 15 customers
weights = delivery['Size'].values  # Package weights
depot_distances = delivery['Depot'].values  # Distances from depot

# Create distance matrix between customers
distances = np.zeros((num_customers, num_customers))
for i in range(num_customers):
    for j in range(num_customers):
        if i != j:
            distances[i, j] = delivery[f'Customer_{j+1}'].iloc[i]

# Problem parameters
num_vans = 3
max_weight_per_van = 15000  # 15,000 lbs
max_distance_per_van = 254  # 254 km

# Display information about the data
print(f"\nNumber of customers: {num_customers}")
print(f"Total weight to deliver: {sum(weights)} lbs")
print(f"Package weights (lbs): {weights}")
print(f"Number of vans available: {num_vans}")
print(f"Maximum weight per van: {max_weight_per_van} lbs")
print(f"Maximum distance per van: {max_distance_per_van} km")

# Create the Gurobi model
model = gp.Model("FedEx_VRP")

# Decision variables
# x[i,v] = 1 if customer i is assigned to van v
x = model.addVars(num_customers, num_vans, vtype=GRB.BINARY, name="x")

# y[v] = total weight assigned to van v
y = model.addVars(num_vans, vtype=GRB.CONTINUOUS, name="y")

# z = maximum difference in load between any two vans
z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

# Constraint: Each customer must be assigned to exactly one van
for i in range(num_customers):
    model.addConstr(gp.quicksum(x[i,v] for v in range(num_vans)) == 1, f"Customer_{i+1}_Assignment")

# Constraint: Calculate total weight for each van
for v in range(num_vans):
    model.addConstr(y[v] == gp.quicksum(weights[i] * x[i,v] for i in range(num_customers)), f"Van_{v+1}_Weight")

# Constraint: Maximum weight difference between any two vans
for v1 in range(num_vans):
    for v2 in range(v1+1, num_vans):
        model.addConstr(y[v1] - y[v2] <= z, f"Weight_Diff_{v1+1}_{v2+1}_Plus")
        model.addConstr(y[v2] - y[v1] <= z, f"Weight_Diff_{v1+1}_{v2+1}_Minus")

# Constraint: Maximum weight per van is 15,000 lbs
for v in range(num_vans):
    model.addConstr(y[v] <= max_weight_per_van, f"Max_Weight_Van_{v+1}")

# Constraint: No more than two of customers 7-9 can be assigned to the same van
for v in range(num_vans):
    model.addConstr(x[6,v] + x[7,v] + x[8,v] <= 2, f"Constraint_Customers_7_to_9_Van_{v+1}")

# Constraint: Customers 10-12 must all be assigned to the same van
for v in range(num_vans):
    model.addConstr(x[9,v] == x[10,v], f"Constraint_Customers_10_11_Same_Van_{v+1}")
    model.addConstr(x[10,v] == x[11,v], f"Constraint_Customers_11_12_Same_Van_{v+1}")

# Constraint: If customer 1 is assigned to a van, at least one of 13 or 14 must also be assigned
for v in range(num_vans):
    model.addConstr(x[0,v] <= x[12,v] + x[13,v], f"Constraint_Customer_1_13_14_Van_{v+1}")

# Constraint: Customer 2 cannot be with customers 3, 4, and 5
for v in range(num_vans):
    model.addConstr(x[1,v] + x[2,v] + x[3,v] + x[4,v] <= 3, f"Constraint_Customer_2_345_Van_{v+1}")

# Constraint: No van can make more than 5 deliveries
for v in range(num_vans):
    model.addConstr(gp.quicksum(x[i,v] for i in range(num_customers)) <= 5, f"Max_Deliveries_Van_{v+1}")

# Constraint: Battery range - simplified approach without subtour elimination
for v in range(num_vans):
    model.addConstr(
        gp.quicksum(2 * depot_distances[i] * x[i,v] for i in range(num_customers)) <= max_distance_per_van,
        f"Max_Distance_Van_{v+1}"
    )

# Set the objective function: Minimize the maximum weight difference
model.setObjective(z, GRB.MINIMIZE)

# Print model statistics
print(f"\nModel statistics:")
print(f"Number of variables: {model.numVars}")
print(f"Number of binary variables: {sum(1 for v in model.getVars() if v.vtype == GRB.BINARY)}")
print(f"Number of constraints: {model.numConstrs}")

# Optimize the model
print("\nSolving the model...")
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("\nOptimal solution found!")
    print(f"Optimal objective value (minimum maximum weight difference): {model.objVal} lbs")
    
    # Print the assignment of customers to vans
    for v in range(num_vans):
        assigned_customers = [i+1 for i in range(num_customers) if x[i,v].X > 0.5]
        van_weight = sum(weights[i] for i in range(num_customers) if x[i,v].X > 0.5)
        approx_distance = sum(2 * depot_distances[i] for i in range(num_customers) if x[i,v].X > 0.5)
        
        print(f"\nVan {v+1}:")
        print(f"  Customers: {assigned_customers}")
        print(f"  Total Weight: {van_weight} lbs")
        print(f"  Approx. Distance: {approx_distance} km")
        print(f"  Number of Deliveries: {len(assigned_customers)}")
    
    # Calculate the total weight to verify the solution
    total_weight = sum(weights)
    print(f"\nTotal weight of all deliveries: {total_weight} lbs")
    print(f"Average weight per van: {total_weight/num_vans:.2f} lbs")
    
    # Calculate the weight difference between each pair of vans to verify the objective
    van_weights = [sum(weights[i] for i in range(num_customers) if x[i,v].X > 0.5) for v in range(num_vans)]
    max_diff = max(abs(van_weights[i] - van_weights[j]) for i in range(num_vans) for j in range(i+1, num_vans))
    print(f"Maximum weight difference (verification): {max_diff} lbs")
    
else:
    print(f"Optimization failed with status {model.status}")
