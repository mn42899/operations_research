print("\n" + "="*60)
print("  QUESTION 1 - BioAgri Solutions ")
print("="*60 + "\n")

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Question 1
centers = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/centers.csv')
farms = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/farms.csv')
processing = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/processing.csv')
updated_gym_data = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/updated_gym_data.csv')

# Exploring Dataset
centers.head()
farms.head()
processing.head()



""" 1. b) Using Gurobi, what is the minimum cost of the transportation and procurement plan?"""

print("\n" + "="*60)
print("  Running Optimization: Question 1 Part B (Minimum Cost)  ")
print("="*60 + "\n")

# Parameters
farm_ids = farms['Farm_ID']
processing_ids = processing['Processing_Plant_ID']
center_ids = centers['Center_ID']

farm_capacity = farms.set_index('Farm_ID')['Bio_Material_Capacity_Tons']
farm_cost = farms.set_index('Farm_ID')['Cost_Per_Ton']

# Extract transport costs from farm to processing
farm_to_processing_cost = farms.set_index('Farm_ID').filter(like="Transport_Cost_To_Plant")

processing_capacity = processing.set_index('Processing_Plant_ID')['Capacity_Tons']
processing_cost = processing.set_index('Processing_Plant_ID')['Processing_Cost_Per_Ton']

# Extract transport costs from processing to centers
processing_to_center_cost = processing.set_index('Processing_Plant_ID').filter(like="Transport_Cost_To_Center")

center_demand = centers.set_index('Center_ID')['Requested_Demand_Tons']

# Create model
model = gp.Model('Transportation_and_Procurement')

# Decision variables
x_fp = model.addVars(farm_ids, processing_ids, name="x_fp", lb=0)  # Raw material from farms to processing
x_pc = model.addVars(processing_ids, center_ids, name="x_pc", lb=0)  # Fertilizer from processing to centers

# Objective function: Minimize total cost
model.setObjective(
    gp.quicksum(
        x_fp[f, p] * (farm_cost[f] + farm_to_processing_cost.loc[f, f'Transport_Cost_To_Plant_{p.split("_")[-1]}'])
        for f in farm_ids for p in processing_ids
    ) +
    gp.quicksum(
        x_pc[p, c] * (processing_cost[p] + processing_to_center_cost.loc[p, f'Transport_Cost_To_Center_{c.split("_")[-1]}'])
        for p in processing_ids for c in center_ids
    ),
    GRB.MINIMIZE
)

# Constraints
# Farm capacity constraints
for f in farm_ids:
    model.addConstr(gp.quicksum(x_fp[f, p] for p in processing_ids) <= farm_capacity[f], f"FarmCapacity_{f}")

# Processing facility capacity constraints
for p in processing_ids:
    model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) <= processing_capacity[p], f"ProcessingCapacity_{p}")

# Center demand constraints
for c in center_ids:
    model.addConstr(gp.quicksum(x_pc[p, c] for p in processing_ids) == center_demand[c], f"CenterDemand_{c}")

# Flow balance constraints: Input to processing equals output
for p in processing_ids:
    model.addConstr(
        gp.quicksum(x_fp[f, p] for f in farm_ids) == gp.quicksum(x_pc[p, c] for c in center_ids),
        f"FlowBalance_{p}"
    )

# Solve the model
model.optimize()

# Output the results
if model.status == GRB.OPTIMAL:
    # Output the minimum cost
    print("=" * 50)
    print(f"Minimum Transportation and Procurement Cost: ${model.objVal:,.2f}")
    print("=" * 50)

    # Output detailed solution
    print("\nFarm to Processing Assignments:")
    for f in farm_ids:
        for p in processing_ids:
            if x_fp[f, p].x > 0:
                print(f"Farm {f} -> Processing {p}: {x_fp[f, p].x:.2f} tons")

    print("\nProcessing to Center Assignments:")
    for p in processing_ids:
        for c in center_ids:
            if x_pc[p, c].x > 0:
                print(f"Processing {p} -> Center {c}: {x_pc[p, c].x:.2f} tons")

else:
    print("No optimal solution found.")

print("\n" + "-"*60 + "\n")





""" 1. c) If the processing plants of the raw material are restricted to only send fertilizer to
home centers within the same region of the US, what is the optimal cost? """

print("\n" + "="*60)
print("    Running Optimization: Question 1 Part C (Regional Constraints)    ")
print("="*60 + "\n")

# Parameters
farm_ids = farms['Farm_ID']
processing_ids = processing['Processing_Plant_ID']
center_ids = centers['Center_ID']

farm_capacity = farms.set_index('Farm_ID')['Bio_Material_Capacity_Tons']
farm_cost = farms.set_index('Farm_ID')['Cost_Per_Ton']

# Extract transport costs from farm to processing
farm_to_processing_cost = farms.set_index('Farm_ID').filter(like="Transport_Cost_To_Plant")

processing_capacity = processing.set_index('Processing_Plant_ID')['Capacity_Tons']
processing_cost = processing.set_index('Processing_Plant_ID')['Processing_Cost_Per_Ton']

# Extract transport costs from processing to centers
processing_to_center_cost = processing.set_index('Processing_Plant_ID').filter(like="Transport_Cost_To_Center")

center_demand = centers.set_index('Center_ID')['Requested_Demand_Tons']

# Extract regions
processing_regions = processing.set_index('Processing_Plant_ID')['Region']
center_regions = centers.set_index('Center_ID')['Region']

# Create model
model = gp.Model('Transportation_and_Procurement_Regional')

# Decision variables
x_fp = model.addVars(farm_ids, processing_ids, name="x_fp", lb=0)  # Raw material from farms to processing
x_pc = model.addVars(processing_ids, center_ids, name="x_pc", lb=0)  # Fertilizer from processing to centers

# Objective function: Minimize total cost
model.setObjective(
    gp.quicksum(
        x_fp[f, p] * (farm_cost[f] + farm_to_processing_cost.loc[f, f'Transport_Cost_To_Plant_{p.split("_")[-1]}'])
        for f in farm_ids for p in processing_ids
    ) +
    gp.quicksum(
        x_pc[p, c] * (processing_cost[p] + processing_to_center_cost.loc[p, f'Transport_Cost_To_Center_{c.split("_")[-1]}'])
        for p in processing_ids for c in center_ids
    ),
    GRB.MINIMIZE
)

# Constraints
# Farm capacity constraints
for f in farm_ids:
    model.addConstr(gp.quicksum(x_fp[f, p] for p in processing_ids) <= farm_capacity[f], f"FarmCapacity_{f}")

# Processing facility capacity constraints
for p in processing_ids:
    model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) <= processing_capacity[p], f"ProcessingCapacity_{p}")

# Center demand constraints
for c in center_ids:
    model.addConstr(gp.quicksum(x_pc[p, c] for p in processing_ids) == center_demand[c], f"CenterDemand_{c}")

# Flow balance constraints: Input to processing equals output
for p in processing_ids:
    model.addConstr(
        gp.quicksum(x_fp[f, p] for f in farm_ids) == gp.quicksum(x_pc[p, c] for c in center_ids),
        f"FlowBalance_{p}"
    )

# Regional constraints: Processing plants can only send to centers in the same region
for p in processing_ids:
    for c in center_ids:
        if processing_regions[p] != center_regions[c]:
            model.addConstr(x_pc[p, c] == 0, f"RegionalConstraint_{p}_{c}")

# Solve the model
model.optimize()

# Output the results
if model.status == GRB.OPTIMAL:
    # Output the minimum cost
    print("=" * 50)
    print(f"Optimal Transportation and Procurement Cost (Regional): ${model.objVal:,.2f}")
    print("=" * 50)

    # Output detailed solution
    print("\nFarm to Processing Assignments:")
    for f in farm_ids:
        for p in processing_ids:
            if x_fp[f, p].x > 0:
                print(f"Farm {f} -> Processing {p}: {x_fp[f, p].x:.2f} tons")

    print("\nProcessing to Center Assignments:")
    for p in processing_ids:
        for c in center_ids:
            if x_pc[p, c].x > 0:
                print(f"Processing {p} -> Center {c}: {x_pc[p, c].x:.2f} tons")
else:
    print("No optimal solution found.")

print("\n" + "-"*60 + "\n")





""" 1. d) If only the highest quality raw material (i.e., levels 3 and 4) is sourced from farms to make fertilizer, what is the optimal cost?"""

print("\n" + "="*60)
print("  Running Optimization: Question 1 Part D (High-Quality Materials)  ")
print("="*60 + "\n")


# Parameters
farm_ids = farms['Farm_ID']
processing_ids = processing['Processing_Plant_ID']
center_ids = centers['Center_ID']

farm_capacity = farms.set_index('Farm_ID')['Bio_Material_Capacity_Tons']
farm_cost = farms.set_index('Farm_ID')['Cost_Per_Ton']
farm_quality = farms.set_index('Farm_ID')['Quality']  # Add quality column

# Extract transport costs from farm to processing
farm_to_processing_cost = farms.set_index('Farm_ID').filter(like="Transport_Cost_To_Plant")

processing_capacity = processing.set_index('Processing_Plant_ID')['Capacity_Tons']
processing_cost = processing.set_index('Processing_Plant_ID')['Processing_Cost_Per_Ton']

# Extract transport costs from processing to centers
processing_to_center_cost = processing.set_index('Processing_Plant_ID').filter(like="Transport_Cost_To_Center")

center_demand = centers.set_index('Center_ID')['Requested_Demand_Tons']

# Create model
model = gp.Model('Transportation_and_Procurement_HighQuality')

# Decision variables
x_fp = model.addVars(farm_ids, processing_ids, name="x_fp", lb=0)  # Raw material from farms to processing
x_pc = model.addVars(processing_ids, center_ids, name="x_pc", lb=0)  # Fertilizer from processing to centers

# Objective function: Minimize total cost
model.setObjective(
    gp.quicksum(
        x_fp[f, p] * (farm_cost[f] + farm_to_processing_cost.loc[f, f'Transport_Cost_To_Plant_{p.split("_")[-1]}'])
        for f in farm_ids for p in processing_ids
    ) +
    gp.quicksum(
        x_pc[p, c] * (processing_cost[p] + processing_to_center_cost.loc[p, f'Transport_Cost_To_Center_{c.split("_")[-1]}'])
        for p in processing_ids for c in center_ids
    ),
    GRB.MINIMIZE
)

# Constraints
# Farm capacity constraints
for f in farm_ids:
    model.addConstr(gp.quicksum(x_fp[f, p] for p in processing_ids) <= farm_capacity[f], f"FarmCapacity_{f}")

# Processing facility capacity constraints
for p in processing_ids:
    model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) <= processing_capacity[p], f"ProcessingCapacity_{p}")

# Center demand constraints
for c in center_ids:
    model.addConstr(gp.quicksum(x_pc[p, c] for p in processing_ids) == center_demand[c], f"CenterDemand_{c}")

# Flow balance constraints: Input to processing equals output
for p in processing_ids:
    model.addConstr(
        gp.quicksum(x_fp[f, p] for f in farm_ids) == gp.quicksum(x_pc[p, c] for c in center_ids),
        f"FlowBalance_{p}"
    )

# Quality-based constraints: Only farms with quality levels 3 or 4 can supply raw materials
for f in farm_ids:
    if farm_quality[f] < 3:  # Exclude farms with quality less than 3
        for p in processing_ids:
            model.addConstr(x_fp[f, p] == 0, f"QualityConstraint_{f}")

# Solve the model
model.optimize()

# Output the results
if model.status == GRB.OPTIMAL:
    # Output the minimum cost
    print("=" * 50)
    print(f"Optimal Transportation and Procurement Cost (High Quality): ${model.objVal:,.2f}")
    print("=" * 50)

    # Output detailed solution
    print("\nFarm to Processing Assignments:")
    for f in farm_ids:
        for p in processing_ids:
            if x_fp[f, p].x > 0:
                print(f"Farm {f} -> Processing {p}: {x_fp[f, p].x:.2f} tons")

    print("\nProcessing to Center Assignments:")
    for p in processing_ids:
        for c in center_ids:
            if x_pc[p, c].x > 0:
                print(f"Processing {p} -> Center {c}: {x_pc[p, c].x:.2f} tons")
else:
    print("No optimal solution found.")

print("\n" + "-"*60 + "\n")




"""1. e) If each facility is limited to processing no more than 3% of all raw material sourced from farms 
(as a sourcing risk mitigation measure), what is the optimal cost? Alternatively, if a production facility 
is limited to supplying no more than 50% of all fertilizer to a single home center 
(as a supply risk mitigation measure), what is the optimal cost?"""

print("\n" + "="*60)
print(" Running Optimization: Question 1 Part E (Risk Mitigation Constraints) ")
print("="*60 + "\n")

 # Parameters
farm_ids = farms['Farm_ID']
processing_ids = processing['Processing_Plant_ID']
center_ids = centers['Center_ID']

farm_capacity = farms.set_index('Farm_ID')['Bio_Material_Capacity_Tons']
farm_cost = farms.set_index('Farm_ID')['Cost_Per_Ton']

# Extract transport costs from farm to processing
farm_to_processing_cost = farms.set_index('Farm_ID').filter(like="Transport_Cost_To_Plant")

processing_capacity = processing.set_index('Processing_Plant_ID')['Capacity_Tons']
processing_cost = processing.set_index('Processing_Plant_ID')['Processing_Cost_Per_Ton']

# Extract transport costs from processing to centers
processing_to_center_cost = processing.set_index('Processing_Plant_ID').filter(like="Transport_Cost_To_Center")

center_demand = centers.set_index('Center_ID')['Requested_Demand_Tons']

# Total raw material sourced from farms
total_raw_material = farm_capacity.sum()

# Function to solve the optimization problem with specified constraints
def solve_model(apply_3_percent_constraint=False, apply_50_percent_constraint=False):
    # Create model
    model = gp.Model('Transportation_and_Procurement')

    # Decision variables
    x_fp = model.addVars(farm_ids, processing_ids, name="x_fp", lb=0)  # Raw material from farms to processing
    x_pc = model.addVars(processing_ids, center_ids, name="x_pc", lb=0)  # Fertilizer from processing to centers

    # Objective function: Minimize total cost
    model.setObjective(
        gp.quicksum(
            x_fp[f, p] * (farm_cost[f] + farm_to_processing_cost.loc[f, f'Transport_Cost_To_Plant_{p.split("_")[-1]}'])
            for f in farm_ids for p in processing_ids
        ) +
        gp.quicksum(
            x_pc[p, c] * (processing_cost[p] + processing_to_center_cost.loc[p, f'Transport_Cost_To_Center_{c.split("_")[-1]}'])
            for p in processing_ids for c in center_ids
        ),
        GRB.MINIMIZE
    )

    # Constraints
    # Farm capacity constraints
    for f in farm_ids:
        model.addConstr(gp.quicksum(x_fp[f, p] for p in processing_ids) <= farm_capacity[f], f"FarmCapacity_{f}")

    # Processing facility capacity constraints
    for p in processing_ids:
        model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) <= processing_capacity[p], f"ProcessingCapacity_{p}")

    # Center demand constraints
    for c in center_ids:
        model.addConstr(gp.quicksum(x_pc[p, c] for p in processing_ids) == center_demand[c], f"CenterDemand_{c}")

    # Flow balance constraints: Input to processing equals output
    for p in processing_ids:
        model.addConstr(
            gp.quicksum(x_fp[f, p] for f in farm_ids) == gp.quicksum(x_pc[p, c] for c in center_ids),
            f"FlowBalance_{p}"
        )

    # Apply 3% processing constraint (if enabled)
    if apply_3_percent_constraint:
        for p in processing_ids:
            model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) <= 0.03 * total_raw_material, f"MaxProcessing_{p}")

    # Apply 50% supply constraint (if enabled)
    if apply_50_percent_constraint:
        for p in processing_ids:
            for c in center_ids:
                model.addConstr(
                    x_pc[p, c] <= 0.5 * gp.quicksum(x_pc[p, cc] for cc in center_ids),
                    f"MaxSupply_{p}_{c}"
                )

    # Solve the model
    model.optimize()

    # Output the results
    if model.status == GRB.OPTIMAL:
        print("=" * 50)
        if apply_3_percent_constraint:
            print("Optimal Cost with 3% Processing Constraint:")
        elif apply_50_percent_constraint:
            print("Optimal Cost with 50% Supply Constraint:")
        print(f"Optimal Transportation and Procurement Cost: ${model.objVal:,.2f}")
        print("=" * 50)
    else:
        print("No optimal solution found.")

# Solve with the 3% processing constraint
solve_model(apply_3_percent_constraint=True, apply_50_percent_constraint=False)

# Solve with the 50% supply constraint
solve_model(apply_3_percent_constraint=False, apply_50_percent_constraint=True)
print("\n" + "-"*60 + "\n")



""" 1. f) Four options were evaluated to understand how changes to the supply chain impacted cost, i.e., see parts (c) through (e). 
Which of these options (or multiple) are financially defensible, and why? 
What is the optimal cost when you implement all of the defensible options together?"""

print("\n" + "="*60)
print("    Running Optimization: Question 1 Part F (Defensible Options)    ")
print("="*60 + "\n")

farm_capacity = farms.set_index('Farm_ID')['Bio_Material_Capacity_Tons']
farm_cost = farms.set_index('Farm_ID')['Cost_Per_Ton']
farm_quality = farms.set_index('Farm_ID')['Quality']

# Extract transport costs from farm to processing
farm_to_processing_cost = farms.set_index('Farm_ID').filter(like="Transport_Cost_To_Plant")
processing_capacity = processing.set_index('Processing_Plant_ID')['Capacity_Tons']
processing_cost = processing.set_index('Processing_Plant_ID')['Processing_Cost_Per_Ton']

# Extract transport costs from processing to centers
processing_to_center_cost = processing.set_index('Processing_Plant_ID').filter(like="Transport_Cost_To_Center")
center_demand = centers.set_index('Center_ID')['Requested_Demand_Tons']

# Extract region data for processing plants and centers
processing_regions = processing.set_index('Processing_Plant_ID')['Region']
center_regions = centers.set_index('Center_ID')['Region']

# Calculate total raw material sourced
total_raw_material = farm_capacity.sum()

# Create optimization model
model = gp.Model('Transportation_and_Procurement_Defensible_Options')

# Decision variables
x_fp = model.addVars(farm_ids, processing_ids, name="x_fp", lb=0)  # Farm to processing
x_pc = model.addVars(processing_ids, center_ids, name="x_pc", lb=0)  # Processing to centers

# Objective function: Minimize total cost
model.setObjective(
    gp.quicksum(
        x_fp[f, p] * (farm_cost[f] + farm_to_processing_cost.loc[f, f'Transport_Cost_To_Plant_{p.split("_")[-1]}'])
        for f in farm_ids for p in processing_ids
    ) +
    gp.quicksum(
        x_pc[p, c] * (processing_cost[p] + processing_to_center_cost.loc[p, f'Transport_Cost_To_Center_{c.split("_")[-1]}'])
        for p in processing_ids for c in center_ids
    ),
    GRB.MINIMIZE
)

# Constraints
# 1. Farm capacity constraints
for f in farm_ids:
    model.addConstr(gp.quicksum(x_fp[f, p] for p in processing_ids) <= farm_capacity[f], f"FarmCapacity_{f}")

# 2. Processing plant capacity constraints
for p in processing_ids:
    model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) <= processing_capacity[p], f"ProcessingCapacity_{p}")

# 3. Center demand constraints
for c in center_ids:
    model.addConstr(gp.quicksum(x_pc[p, c] for p in processing_ids) == center_demand[c], f"CenterDemand_{c}")

# 4. Flow balance constraints: Input to processing equals output
for p in processing_ids:
    model.addConstr(
        gp.quicksum(x_fp[f, p] for f in farm_ids) == gp.quicksum(x_pc[p, c] for c in center_ids),
        f"FlowBalance_{p}"
    )

# 5. Regional constraints: Processing plants can only supply home centers within the same region
for p in processing_ids:
    for c in center_ids:
        if processing_regions[p] != center_regions[c]:
            model.addConstr(x_pc[p, c] == 0, f"RegionalConstraint_{p}_{c}")

# 6. Quality constraints: Use only raw materials with quality level 3 or 4
for f in farm_ids:
    if farm_quality[f] < 3:
        for p in processing_ids:
            model.addConstr(x_fp[f, p] == 0, f"QualityConstraint_{f}")

# 7. Risk mitigation: No processing plant should process more than 3% of total raw material
for p in processing_ids:
    model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) <= 0.03 * total_raw_material, f"MaxProcessing_{p}")

# 8. Risk mitigation: A processing plant cannot supply more than 50% of a center’s demand
for p in processing_ids:
    for c in center_ids:
        model.addConstr(x_pc[p, c] <= 0.5 * gp.quicksum(x_pc[p, cc] for cc in center_ids), f"MaxSupply_{p}_{c}")

# Solve the model
model.optimize()

# Store results dynamically
cost_results = {}
if model.status == GRB.OPTIMAL:
    cost_results['Part F'] = model.objVal
    print("=" * 50)
    print(f"Optimal Cost with Defensible Options: ${model.objVal:,.2f}")
    print("=" * 50)

    # Output detailed solution
    print("\nFarm to Processing Assignments:")
    for f in farm_ids:
        for p in processing_ids:
            if x_fp[f, p].x > 0:
                print(f"Farm {f} -> Processing {p}: {x_fp[f, p].x:.2f} tons")

    print("\nProcessing to Center Assignments:")
    for p in processing_ids:
        for c in center_ids:
            if x_pc[p, c].x > 0:
                print(f"Processing {p} -> Center {c}: {x_pc[p, c].x:.2f} tons")
else:
    print("No optimal solution found.")

print("\n" + "-"*60 + "\n")



"""1. g) While implementing all of the defensible options together incurs a higher cost as compared to the original system, 
it may still represent a strong business decision. How would you concisely defend the implementation of all of 
the defensible options to management? """

print("\n" + "="*60)
print("    Question 1 Part G - Justification of Defensible Options    ")
print("="*60 + "\n")

# Given cost values from parts (b) and (f)
initial_cost = 2297089.97  # From part (b)
defensible_cost = 5744578.24  # From part (f)

# Calculate cost increase and percentage change
cost_increase = defensible_cost - initial_cost
percentage_increase = (cost_increase / initial_cost) * 100

# Generate the business justification report
print("Cost Comparison Overview")
print(f"Initial Cost without Constraints: ${initial_cost:,.2f}")
print(f"New Cost with Defensible Options: ${defensible_cost:,.2f}")
print(f"Cost Increase: ${cost_increase:,.2f} ({percentage_increase:.2f}%)")

print("\n" + "-"*60 + "\n")


"""1. h) The supply chain network has a limited capacity for risk mitigation. 
To see this, when implementing all of the defensible options from part (f), 
at what value (to the nearest tenth of a percent) does the model become infeasible 
when reducing the sourcing risk mitigation percentage from the value given in part (e) of 3%? 
What is the managerial interpretation of this result, and what are the implications for 
managing supply chain risk?"""

print("\n" + "="*60)
print("    Question 1 Part H - Feasibility Analysis    ")
print("="*60 + "\n")


# Parameters
total_raw_material = farm_capacity.sum()

# Function to determine the lowest feasible sourcing percentage
def find_feasible_limit():
    sourcing_limit = 0.03  # Start at 3% (current value)
    step_size = 0.001  # Reducing by 0.1% each iteration
    feasible = True

    while feasible and sourcing_limit > 0:
        model = gp.Model('Transportation_and_Procurement_Risk_Test')
        x_fp = model.addVars(farm_ids, processing_ids, name="x_fp", lb=0)
        x_pc = model.addVars(processing_ids, center_ids, name="x_pc", lb=0)

        # Objective function to minimize total cost
        model.setObjective(
            gp.quicksum(
                x_fp[f, p] * (farm_cost[f] + farm_to_processing_cost.loc[f, f'Transport_Cost_To_Plant_{p.split("_")[-1]}'])
                for f in farm_ids for p in processing_ids
            ) +
            gp.quicksum(
                x_pc[p, c] * (processing_cost[p] + processing_to_center_cost.loc[p, f'Transport_Cost_To_Center_{c.split("_")[-1]}'])
                for p in processing_ids for c in center_ids
            ),
            GRB.MINIMIZE
        )

        # Constraints

        # Farm capacity
        for f in farm_ids:
            model.addConstr(gp.quicksum(x_fp[f, p] for p in processing_ids) <= farm_capacity[f], f"FarmCapacity_{f}")

        # Processing facility capacity
        for p in processing_ids:
            model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) <= processing_capacity[p], f"ProcessingCapacity_{p}")

        # Center demand constraints
        for c in center_ids:
            model.addConstr(gp.quicksum(x_pc[p, c] for p in processing_ids) == center_demand[c], f"CenterDemand_{c}")

        # Flow balance constraints
        for p in processing_ids:
            model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) == gp.quicksum(x_pc[p, c] for c in center_ids),
                            f"FlowBalance_{p}")

        # Regional constraints
        for p in processing_ids:
            for c in center_ids:
                if processing_regions[p] != center_regions[c]:
                    model.addConstr(x_pc[p, c] == 0, f"RegionalConstraint_{p}_{c}")

        # Quality constraints (only levels 3 and 4)
        for f in farm_ids:
            if farm_quality[f] < 3:
                for p in processing_ids:
                    model.addConstr(x_fp[f, p] == 0, f"QualityConstraint_{f}")

        # Sourcing risk mitigation constraint: No more than sourcing_limit
        for p in processing_ids:
            model.addConstr(gp.quicksum(x_fp[f, p] for f in farm_ids) <= sourcing_limit * total_raw_material,
                            f"SourcingRisk_{p}")

        # Supply risk mitigation constraint: No more than 50% per center
        for p in processing_ids:
            for c in center_ids:
                model.addConstr(x_pc[p, c] <= 0.5 * gp.quicksum(x_pc[p, cc] for cc in center_ids),
                                f"SupplyRisk_{p}_{c}")

        # Optimize the model
        model.optimize()

        if model.status == GRB.INFEASIBLE:
            feasible = False
        else:
            sourcing_limit -= step_size  # Reduce sourcing limit by 0.1%

    return round((sourcing_limit + step_size) * 100, 1)

# Run the feasibility test
min_feasible_sourcing = find_feasible_limit()
print(f"Minimum feasible sourcing risk mitigation percentage: {min_feasible_sourcing}%")
