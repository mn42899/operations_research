"""
@author: Adam Diamant (2025)
"""

from gurobipy import GRB
import gurobipy as gb

# Create the dual model
dual_model = gb.Model("Tower Research Dual")

# Add dual variables
y = dual_model.addVars(4, vtype=GRB.CONTINUOUS, name="y")
z = dual_model.addVars(3, lb=0, vtype=GRB.CONTINUOUS, name="z")

# Objective function
dual_obj = 3800 * y[0] - 400 * y[1] + 1588 * y[2] + 2000 * y[3] + 3000 * z[0] + 3000 * z[1] + 3000 * z[2]
# dual_obj = 3800 * y[0] + 400 * y[1] - 1588 * y[2] - 2000 * y[3] + 3000 * z[0] + 3000 * z[1] + 3000 * z[2]
dual_model.setObjective(dual_obj, GRB.MINIMIZE)

# Constraints linking periods
dual_model.addConstr(y[0] - y[1] >= 0, "C1")
dual_model.addConstr(y[1] - y[2] >= 0, "C2")
dual_model.addConstr(y[2] - y[3] >= 0, "C3")
dual_model.addConstr(y[3] >= 1, "C3")

# Borrowing constraints across periods
dual_model.addConstr(-y[0] + 1.03 * y[1] + z[0] >= 0, "B1")
dual_model.addConstr(-y[1] + 1.03 * y[2] + z[1] >= 0, "B2")
dual_model.addConstr(-y[2] + 1.03 * y[3] + z[2] >= 0, "B3")

# Investment growth constraints
dual_model.addConstr(y[0] - 1.02 * y[2] >= 0, "I1")
dual_model.addConstr(y[1] - 1.02 * y[3] >= 0, "I2")

# Solve the dual model
dual_model.optimize()

# Output the solution
print("Optimal value (dual):", dual_model.objVal)
