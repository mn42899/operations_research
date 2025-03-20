import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Load data
try:
    sustainability = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/sustainability.csv')
    heat = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/heat.csv')
    lubricant_materials = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/lubricant_materials.csv')
    lubricant_products = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/lubricant_products.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Model dimensions
num_lubricants = len(lubricant_products)
num_materials = len(lubricant_materials)

def create_linear_model(sustainability, heat, lubricant_materials, lubricant_products, with_regulatory_constraints=True):
    """
    Create the linear programming model for KryptoFlow
    
    Parameters:
    sustainability, heat, lubricant_materials, lubricant_products: DataFrames containing problem data
    with_regulatory_constraints (bool): Whether to include regulatory constraints
    
    Returns:
    model: Gurobi model object
    x: Dictionary of decision variables
    production: Dictionary of production quantity variables
    """
    # Create a new model
    model = gp.Model("KryptoFlow_Production")

    # Decision Variables: x[i,j] = amount of material j used in lubricant i
    x = model.addVars(num_lubricants, num_materials, lb=0, vtype=GRB.CONTINUOUS, name="x")
    production = model.addVars(num_lubricants, lb=0, vtype=GRB.CONTINUOUS, name="production")

    # Objective: Maximize total revenue
    model.setObjective(
        gp.quicksum(lubricant_products.loc[i, 'revenue_per_unit'] * production[i] for i in range(num_lubricants)),
        GRB.MAXIMIZE
    )

    # Constraint 1: Demand constraints
    for i in range(num_lubricants):
        model.addConstr(
            production[i] <= lubricant_products.loc[i, 'total_demand'],
            name=f"demand_{i}"
        )

    # Constraint 2: Material availability
    for j in range(num_materials):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_lubricants)) <= lubricant_materials.loc[j, 'max_availability'],
            name=f"material_avail_{j}"
        )

    # Constraint 3: Maximum proportion of each material in each lubricant
    for i in range(num_lubricants):
        for j in range(num_materials):
            model.addConstr(
                x[i, j] <= lubricant_materials.loc[j, 'p_max'] * production[i],
                name=f"max_prop_{i}_{j}"
            )

    if with_regulatory_constraints:
        # Constraint 4: Sustainability requirements for each lubricant
        for i in range(num_lubricants):
            model.addConstr(
                gp.quicksum(sustainability.loc[i, f'Material_{j+1}'] * x[i, j] for j in range(num_materials)) >= 
                lubricant_products.loc[i, 'min_sustainability'] * production[i],
                name=f"min_sustainability_{i}"
            )

        # Constraint 5: Heat limitations for each lubricant
        for i in range(num_lubricants):
            model.addConstr(
                gp.quicksum(heat.loc[i, f'Material_{j+1}'] * x[i, j] for j in range(num_materials)) <= 
                lubricant_products.loc[i, 'max_heat'] * production[i],
                name=f"max_heat_{i}"
            )

        # Constraint 6: Total sustainability across all products
        model.addConstr(
            gp.quicksum(sustainability.loc[i, f'Material_{j+1}'] * x[i, j] for i in range(num_lubricants) for j in range(num_materials)) >= 
            2 * gp.quicksum(lubricant_products.loc[i, 'min_sustainability'] * production[i] for i in range(num_lubricants)),
            name="total_sustainability"
        )

    return model, x, production

def solve_kryptoflow_problem():
    """Solve the KryptoFlow optimization problem and answer all exam questions"""
    
    # (a) Decision Variables
    print("\n(a) Decision Variables:")
    print("The decision variables x[i,j] represent the amount of material j used in lubricant i.")
    print(f"Total number of decision variables: {num_lubricants} lubricants × {num_materials} materials = {num_lubricants * num_materials}")
    
    # Build and solve the linear model
    linear_model, x_linear, production_linear = create_linear_model(
        sustainability, heat, lubricant_materials, lubricant_products
    )
    linear_model.optimize()
    
    # (b) Optimal Revenue
    print("\n(b) Optimal Revenue:")
    if linear_model.status == GRB.OPTIMAL:
        print(f"The optimal revenue is ${linear_model.objVal:.2f}")
    else:
        print("Linear model could not be solved to optimality.")

    # (c) Are there any lubricant products that are not produced?
    print("\n(c) Production Analysis:")
    not_produced = []
    
    for i in range(num_lubricants):
        if production_linear[i].X < 1e-5:
            not_produced.append(i+1)  # 1-indexed for reporting
    
    if not_produced:
        print(f"{len(not_produced)} lubricants are not produced: {not_produced}")
    else:
        print("All lubricant products are produced.")

    # (f) Solve model without regulatory constraints
    print("\n(f) Model without Regulatory Constraints:")
    unreg_model, _, _ = create_linear_model(sustainability, heat, lubricant_materials, lubricant_products, with_regulatory_constraints=False)
    unreg_model.optimize()
    
    if linear_model.status == GRB.OPTIMAL and unreg_model.status == GRB.OPTIMAL:
        diff = unreg_model.objVal - linear_model.objVal
        
        print(f"Optimal revenue without regulatory constraints: ${unreg_model.objVal:.2f}")
        
        if linear_model.objVal != 0:  # Prevent division by zero
            percentage_change = (diff / linear_model.objVal) * 100
            print(f"Difference from regulated model: ${diff:.2f} (+{percentage_change:.2f}%)")
        else:
            print("Difference from regulated model: ${diff:.2f} (Linear model objective value is 0, percentage change not computed)")
    else:
        print("One or both models could not be solved to optimality.")

if __name__ == "__main__":
    solve_kryptoflow_problem()

