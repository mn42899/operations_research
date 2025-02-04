# -*- coding: utf-8 -*-
"""
@author: Adam Diamant (2025)
"""

import gurobipy as gp
from gurobipy import GRB

# Create a new Gurobi model
model = gp.Model("Quadratic_Optimization")

# Decision variables: p1 and p2 (continuous variables)
p1 = model.addVar(vtype=GRB.CONTINUOUS, name="p1", lb=0)
p2 = model.addVar(vtype=GRB.CONTINUOUS, name="p2", lb=0)

# Define the objective function
objective = p1 * (35234 - 26*p1) + p2 * (27790 - 9*p2)
model.setObjective(objective, GRB.MAXIMIZE)

# Add the demand constraints
model.addConstr(35234 - 26*p1 >= 0, name="Demand_Constraint_1")
model.addConstr(27790 - 9*p2 >= 0, name="Demand_Constraint_2")

# Add the pricing constraint
model.addConstr(p2 >= 550 + p1, name="Pricing_Constraint")

# Optimize the model
model.optimize()

# Output the results
if model.status == GRB.OPTIMAL:
    print(f"Optimal value of p1: {p1.X}")
    print(f"Optimal value of p2: {p2.X}")
    print(f"Optimal objective value (Z): {model.objVal}")
else:
    print("No optimal solution found.")
