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

print("\n" + "-"*60 + "\n")

print("\n" + "-"*60 + "\n")




print("\n" + "="*60)
print("  QUESTION 2 - RP Strength ")
print("="*60 + "\n")

import numpy as np
import pandas as pd
import seaborn as sns
from gurobipy import GRB, quicksum
import gurobipy as gb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/updated_gym_data.csv')


# EDA
df.head(10)
df.columns

df['BodyPart'].unique()

# getting data from data set
exercise_ids = df.index
hypertrophy_ratings = df['Hypertrophy Rating']
sfr_values = df['Stimulus-to-Fatigue']
body_parts = df['BodyPart']
categories = df['Category']
equipment_types = df['Equipment']
Difficulty_rating = df['Difficulty']


""" 2. c) Using Gurobi, what is the optimal hypertrophy rating using all constraints? """

print("\n" + "="*60)
print("  Running Optimization: Question 2 Part C (Optimal Hypertrophy Rating)  ")
print("="*60 + "\n")

# Create a new optimization model
model = gb.Model("Hypertrophy Optimization")

# Decision variables
num_exercises = len(df)
x = model.addVars(num_exercises, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="ExerciseProportion")

# Objective function: Maximizes the total hypertrophy rating across all exercises based on their proportions.
model.setObjective(
        quicksum(df.loc[i, 'Hypertrophy Rating'] * x[i] for i in range(num_exercises)),
        GRB.MAXIMIZE
    )

# Adding the constraints
## 1. Limit the proportion per exercise to a maximum of 5%.
for i in range(num_exercises):
    model.addConstr(x[i] <= 0.05, name=f"MaxProportion_{i}")

# 2. Enforce minimum allocations for specific and general body parts.
general_min_allocation = 0.025  # 2.5%
specific_min_allocations = {
    'Traps': 0.005,      # 0.5%
    'Neck': 0.005,       # 0.5%
    'Forearms': 0.005,   # 0.5%
    'Abdominals': 0.04   # 4%
}

# General and specific allocation constraints for body parts
unique_body_parts = df['BodyPart'].unique()
for part in unique_body_parts:
    if part in specific_min_allocations:
        model.addConstr(
            gb.quicksum(x[i] for i in exercise_ids if body_parts[i] == part) >= specific_min_allocations[part],
            f"{part}Min"
        )
    else:
        model.addConstr(
            gb.quicksum(x[i] for i in exercise_ids if body_parts[i] == part) >= general_min_allocation,
            f"{part}GeneralMin"
        )

# 3. Ensure leg muscles allocation is at least 2.6 times the upper body allocation.
leg_muscles = ['Adductors', 'Abductors', 'Calves', 'Glutes', 'Hamstrings', 'Quadriceps']
upper_body_muscles = ['Chest', 'Lower Back', 'Middle Back', 'Biceps', 'Traps', 'Triceps', 
                      'Shoulders', 'Abdominals', 'Forearms', 'Neck', 'Lats']

leg_upper_body_ratio_constraint = model.addConstr(
    gb.quicksum(x[i] for i in exercise_ids if body_parts[i] in leg_muscles) >= 
    2.6 * gb.quicksum(x[i] for i in exercise_ids if body_parts[i] in upper_body_muscles),
    "LegUpperBodyRatio"
)

# 4. Balance biceps and triceps allocations with chest, lower back, and middle back allocations.
muscle_group_balance_constraint = model.addConstr(
    gb.quicksum(x[i] for i in exercise_ids if body_parts[i] in ['Biceps', 'Triceps']) ==
    gb.quicksum(x[i] for i in exercise_ids if body_parts[i] in ['Chest', 'Lower Back', 'Middle Back']),
    "MuscleGroupBalance"
)

# 5. Restrict the overall Stimulus-to-Fatigue Ratio (SFR) to a maximum of 0.55
sfr_constraint = model.addConstr(
    gb.quicksum(sfr_values[i] * x[i] for i in exercise_ids) <= 0.55,
    "SFRConstraint"
)

# 6. Maintain beginner ≥ 1.4 × intermediate ≥ advanced difficulty ratios.
beginner_intermediate_ratio_constraint = model.addConstr(
    gb.quicksum(x[i] for i in exercise_ids if Difficulty_rating[i] == 'Beginner') >= 
    1.4 * gb.quicksum(x[i] for i in exercise_ids if Difficulty_rating[i] == 'Intermediate'),
    "BeginnerIntermediateRatio"
)

intermediate_advanced_ratio_constraint = model.addConstr(
    gb.quicksum(x[i] for i in exercise_ids if Difficulty_rating[i] == 'Intermediate') >= 
    1.1 * gb.quicksum(x[i] for i in exercise_ids if Difficulty_rating[i] == 'Advanced'),
    "IntermediateAdvancedRatio"
)

# 7. Set minimum and maximum allocations for Strongman, Powerlifting, and Olympic Weightlifting exercises.
# Strongman exercises ≤ 8%
strongman_constraint = model.addConstr(
    gb.quicksum(x[i] for i in exercise_ids if categories[i] == 'Strongman') <= 0.08,
    "StrongmanMax"
)

# Powerlifting exercises ≥ 9%
powerlifting_constraint = model.addConstr(
    gb.quicksum(x[i] for i in exercise_ids if categories[i] == 'Powerlifting') >= 0.09,
    "PowerliftingMin"
)

# Olympic Weightlifting exercises ≥ 10%
olympic_weightlifting_constraint = model.addConstr(
    gb.quicksum(x[i] for i in exercise_ids if categories[i] == 'Olympic Weightlifting') >= 0.10,
    "OlympicWeightliftingMin"
)

# 8. Ensure at least 60% of exercises involve essential equipment types.
essential_equipment = ['Barbell', 'Dumbbell', 'Machine', 'Cable', 'E-Z Curl Bar', 'Bands']
equipment_constraint = model.addConstr(
    gb.quicksum(x[i] for i in exercise_ids if equipment_types[i] in essential_equipment) >= 0.6,
    "EssentialEquipmentMin"
)

# Ensure total proportion is 1
model.addConstr(gb.quicksum(x[i] for i in range(len(df))) == 1, "Total_Proportion")

# Solve the optimization model to find the optimal proportions for each exercise while adhering to all constraints.
# The model successfully converges to an optimal solution with an objective value of 0.7672, maximizing hypertrophy.
model.optimize()

if model.status == GRB.INFEASIBLE:
    print("Model is infeasible. Computing IIS...")
    model.computeIIS()
    model.write("infeasible_constraints.ilp")
    
# Check the optimization status
if model.status == GRB.OPTIMAL:
    print("The optimal solution: ", model.objVal)
    # Optionally, display the decision variables
    for i in exercise_ids:
        if x[i].x > 0:
            print(f"Exercise {df.loc[i, 'Exercise']}: Proportion = {x[i].x:.4f}")
elif model.status == GRB.INFEASIBLE:
    print("The model is infeasible.")
elif model.status == GRB.UNBOUNDED:
    print("The model is unbounded.")
else:
    print("Optimization was stopped with status", model.status)




from gurobipy import Model, GRB, quicksum

# Initialize the Gurobi model
model = Model("Hypertrophy_Optimization")

exercise_ids = range(len(df))
body_parts = df['BodyPart']
categories = df['Category']
equipment_types = df['Equipment']
sfr_values = df['Stimulus-to-Fatigue']
difficulty_ratings = df['Difficulty']

# Decision variables
x = model.addVars(len(df), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="ExerciseProportion")

# Objective function: Maximize hypertrophy rating
model.setObjective(
    quicksum(df.loc[i, 'Hypertrophy Rating'] * x[i] for i in exercise_ids),
    GRB.MAXIMIZE
)

# Constraint 1: Total proportion must equal 1
model.addConstr(
    quicksum(x[i] for i in exercise_ids) == 1, 
    name="TotalProportionConstraint"
)

# Constraint 2: Limit the proportion per exercise to a maximum of 5%
for i in exercise_ids:
    model.addConstr(x[i] <= 0.05, name=f"MaxProportion_{i}")

# Constraint 3: Enforce minimum allocations for specific and general body parts
general_min_allocation = 0.025  # 2.5%
specific_min_allocations = {
    'Traps': 0.005,      # 0.5%
    'Neck': 0.005,       # 0.5%
    'Forearms': 0.005,   # 0.5%
    'Abdominals': 0.04   # 4%
}

for part in df['BodyPart'].unique():
    if part in specific_min_allocations:
        model.addConstr(
            quicksum(x[i] for i in exercise_ids if body_parts[i] == part) >= specific_min_allocations[part],
            f"{part}Min"
        )
    else:
        model.addConstr(
            quicksum(x[i] for i in exercise_ids if body_parts[i] == part) >= general_min_allocation,
            f"{part}GeneralMin"
        )

# Constraint 4: Leg muscles allocation ≥ 2.6 × upper body allocation
leg_muscles = ['Adductors', 'Abductors', 'Calves', 'Glutes', 'Hamstrings', 'Quadriceps']
upper_body_muscles = ['Chest', 'Lower Back', 'Middle Back', 'Biceps', 'Traps', 'Triceps', 
                      'Shoulders', 'Abdominals', 'Forearms', 'Neck', 'Lats']

model.addConstr(
    quicksum(x[i] for i in exercise_ids if body_parts[i] in leg_muscles) >= 
    2.6 * quicksum(x[i] for i in exercise_ids if body_parts[i] in upper_body_muscles),
    "LegUpperBodyRatio"
)

# Constraint 5: Biceps and triceps allocations balance with chest, lower back, and middle back
model.addConstr(
    quicksum(x[i] for i in exercise_ids if body_parts[i] in ['Biceps', 'Triceps']) ==
    quicksum(x[i] for i in exercise_ids if body_parts[i] in ['Chest', 'Lower Back', 'Middle Back']),
    "MuscleGroupBalance"
)

# Constraint 6: Stimulus-to-Fatigue Ratio (SFR) ≤ 0.55
model.addConstr(
    quicksum(sfr_values[i] * x[i] for i in exercise_ids) <= 0.55,
    "SFRConstraint"
)

# Constraint 7: Beginner ≥ 1.4 × intermediate ≥ advanced difficulty ratios
model.addConstr(
    quicksum(x[i] for i in exercise_ids if difficulty_ratings[i] == 'Beginner') >= 
    1.4 * quicksum(x[i] for i in exercise_ids if difficulty_ratings[i] == 'Intermediate'),
    "BeginnerIntermediateRatio"
)

model.addConstr(
    quicksum(x[i] for i in exercise_ids if difficulty_ratings[i] == 'Intermediate') >= 
    1.1 * quicksum(x[i] for i in exercise_ids if difficulty_ratings[i] == 'Advanced'),
    "IntermediateAdvancedRatio"
)

# Constraint 8: Strongman, Powerlifting, and Olympic Weightlifting allocations
model.addConstr(
    quicksum(x[i] for i in exercise_ids if categories[i] == 'Strongman') <= 0.08,
    "StrongmanMax"
)
model.addConstr(
    quicksum(x[i] for i in exercise_ids if categories[i] == 'Powerlifting') >= 0.09,
    "PowerliftingMin"
)
model.addConstr(
    quicksum(x[i] for i in exercise_ids if categories[i] == 'Olympic Weightlifting') >= 0.10,
    "OlympicWeightliftingMin"
)

# Constraint 9: At least 60% of exercises involve essential equipment
essential_equipment = ['Barbell', 'Dumbbell', 'Machine', 'Cable', 'E-Z Curl Bar', 'Bands']
model.addConstr(
    quicksum(x[i] for i in exercise_ids if equipment_types[i] in essential_equipment) >= 0.6,
    "EssentialEquipmentMin"
)

# Solve the model
model.optimize()

# Debug and display results
if model.Status == GRB.OPTIMAL:
    total_proportion = sum(x[i].x for i in exercise_ids)
    print(f"Total Proportion: {total_proportion:.4f}")
    print(f"The optimal solution: {model.ObjVal:.4f}")
    for i in exercise_ids:
        if x[i].x > 0:  # Display only non-zero proportions
            print(f"Exercise {df.loc[i, 'Exercise']}: Proportion = {x[i].x:.4f}")
else:
    print(f"Optimization status: {model.Status}")
    if model.Status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("infeasible_constraints.ilp")



#geting the shadow prices of the model 
# Loop through all constraints and print their shadow prices
for constr in model.getConstrs():
    print(f"Constraint {constr.ConstrName}: Shadow Price = {constr.Pi}")

print("\n" + "-"*60 + "\n")




"""2. d) If the SFR requirement (i.e., constraint 5) were relaxed by 0.001, 
how much would the hypertrophy rating of the workout program improve by? Is this estimate valid?"""

print("\n" + "="*60)
print("  Running Optimization: Question 2 Part D (Relaxed SFR Constraint Impact)  ")
print("="*60 + "\n")

from gurobipy import Model, GRB, quicksum

# Initialize the Gurobi model
model_s = Model("Hypertrophy SFR")

exercise_ids = range(len(df))
body_parts = df['BodyPart']
categories = df['Category']
equipment_types = df['Equipment']
sfr_values = df['Stimulus-to-Fatigue']
difficulty_ratings = df['Difficulty']

# Decision variables
x = model_s.addVars(len(df), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="ExerciseProportion")

# Objective function: Maximize hypertrophy rating
model_s.setObjective(
    quicksum(df.loc[i, 'Hypertrophy Rating'] * x[i] for i in exercise_ids),
    GRB.MAXIMIZE
)

# Constraint 1: Total proportion must equal 1
model_s.addConstr(
    quicksum(x[i] for i in exercise_ids) == 1, 
    name="TotalProportionConstraint"
)

# Constraint 2: Limit the proportion per exercise to a maximum of 5%
for i in exercise_ids:
    model_s.addConstr(x[i] <= 0.05, name=f"MaxProportion_{i}")

# Constraint 3: Enforce minimum allocations for specific and general body parts
general_min_allocation = 0.025  # 2.5%
specific_min_allocations = {
    'Traps': 0.005,      # 0.5%
    'Neck': 0.005,       # 0.5%
    'Forearms': 0.005,   # 0.5%
    'Abdominals': 0.04   # 4%
}

for part in df['BodyPart'].unique():
    if part in specific_min_allocations:
        model_s.addConstr(
            quicksum(x[i] for i in exercise_ids if body_parts[i] == part) >= specific_min_allocations[part],
            f"{part}Min"
        )
    else:
        model_s.addConstr(
            quicksum(x[i] for i in exercise_ids if body_parts[i] == part) >= general_min_allocation,
            f"{part}GeneralMin"
        )

# Constraint 4: Leg muscles allocation ≥ 2.6 × upper body allocation
leg_muscles = ['Adductors', 'Abductors', 'Calves', 'Glutes', 'Hamstrings', 'Quadriceps']
upper_body_muscles = ['Chest', 'Lower Back', 'Middle Back', 'Biceps', 'Traps', 'Triceps', 
                      'Shoulders', 'Abdominals', 'Forearms', 'Neck', 'Lats']

model_s.addConstr(
    quicksum(x[i] for i in exercise_ids if body_parts[i] in leg_muscles) >= 
    2.6 * quicksum(x[i] for i in exercise_ids if body_parts[i] in upper_body_muscles),
    "LegUpperBodyRatio"
)

# Constraint 5: Biceps and triceps allocations balance with chest, lower back, and middle back
model_s.addConstr(
    quicksum(x[i] for i in exercise_ids if body_parts[i] in ['Biceps', 'Triceps']) ==
    quicksum(x[i] for i in exercise_ids if body_parts[i] in ['Chest', 'Lower Back', 'Middle Back']),
    "MuscleGroupBalance"
)

# Constraint 6: Stimulus-to-Fatigue Ratio (SFR) ≤ 0.55
model_s.addConstr(
    quicksum(sfr_values[i] * x[i] for i in exercise_ids) <= 0.551,
    "SFRConstraint"
)

# Constraint 7: Beginner ≥ 1.4 × intermediate ≥ advanced difficulty ratios
model_s.addConstr(
    quicksum(x[i] for i in exercise_ids if difficulty_ratings[i] == 'Beginner') >= 
    1.4 * quicksum(x[i] for i in exercise_ids if difficulty_ratings[i] == 'Intermediate'),
    "BeginnerIntermediateRatio"
)

model_s.addConstr(
    quicksum(x[i] for i in exercise_ids if difficulty_ratings[i] == 'Intermediate') >= 
    1.1 * quicksum(x[i] for i in exercise_ids if difficulty_ratings[i] == 'Advanced'),
    "IntermediateAdvancedRatio"
)

# Constraint 8: Strongman, Powerlifting, and Olympic Weightlifting allocations
model_s.addConstr(
    quicksum(x[i] for i in exercise_ids if categories[i] == 'Strongman') <= 0.08,
    "StrongmanMax"
)
model_s.addConstr(
    quicksum(x[i] for i in exercise_ids if categories[i] == 'Powerlifting') >= 0.09,
    "PowerliftingMin"
)
model_s.addConstr(
    quicksum(x[i] for i in exercise_ids if categories[i] == 'Olympic Weightlifting') >= 0.10,
    "OlympicWeightliftingMin"
)

# Constraint 9: At least 60% of exercises involve essential equipment
essential_equipment = ['Barbell', 'Dumbbell', 'Machine', 'Cable', 'E-Z Curl Bar', 'Bands']
model_s.addConstr(
    quicksum(x[i] for i in exercise_ids if equipment_types[i] in essential_equipment) >= 0.6,
    "EssentialEquipmentMin"
)

# Solve the model
model_s.optimize()

# Debug and display results
if model_s.Status == GRB.OPTIMAL:
    total_proportion = sum(x[i].x for i in exercise_ids)
    print(f"Total Proportion: {total_proportion:.4f}")
    print(f"The optimal solution: {model_s.ObjVal:.4f}")
    for i in exercise_ids:
        if x[i].x > 0:  # Display only non-zero proportions
            print(f"Exercise {df.loc[i, 'Exercise']}: Proportion = {x[i].x:.4f}")
else:
    print(f"Optimization status: {model_s.Status}")
    if model_s.Status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model_s.computeIIS()
        model_s.write("infeasible_constraints.ilp")


print("\n" + "-"*60 + "\n")



""" 2. f) Barbell Back Squats are currently not included in the workout program.
 By how much would their hypertrophy rating need to increase for them to be included?"""

print("\n" + "="*60)
print("  Running Optimization: Question 2 Part F (Barbell Back Squat Inclusion)  ")
print("="*60 + "\n")

exercise_name = "Barbell Back Squats"
exercise_index = df[df['Exercise'] == exercise_name].index[0]

variable = x[exercise_index]

if model.status == GRB.OPTIMAL:
    print(f"Sensitivity Information for '{exercise_name}':")
    print(f"  Current Hypertrophy Rating: {df['Hypertrophy Rating'][exercise_index]:.4f}")
    print(f"  Increase Limit (SAObjUp): {variable.SAObjUp:.4f}")
    print(f"  Decrease Limit (SAObjLow): {variable.SAObjLow:.4f}")
else:
    print("Model is not optimized.")


print("\n" + "-"*60 + "\n")



""" 2. h) Suppose that all of the common constraints are removed except {1, 2, 8} from the list above.
 What is the optimal hypertrophy rating, and why is it higher than in the original solution?"""

print("\n" + "="*60)
print("  Running Optimization: Question 2 Part H (Reduced Constraints Optimization)  ")
print("="*60 + "\n")

# Create a new optimization model
model_p = gb.Model("Primal Model")

# Set tighter solver tolerances
model_p.Params.OptimalityTol = 1e-9
model_p.Params.FeasibilityTol = 1e-9
model_p.Params.IntFeasTol = 1e-9

# Decision variables
num_exercises = len(df)
x = model_p.addVars(num_exercises, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="ExerciseProportion")

# Objective function: Maximizes the total hypertrophy rating across all exercises based on their proportions.
model_p.setObjective(
        quicksum(df.loc[i, 'Hypertrophy Rating'] * x[i] for i in range(num_exercises)),
        GRB.MAXIMIZE
    )

# Adding the constraints
## 1. Limit the proportion per exercise to a maximum of 5%.
for i in range(num_exercises):
    model_p.addConstr(x[i] <= 0.05, name=f"MaxProportion_{i}")

# 2. Enforce minimum allocations for specific and general body parts.
general_min_allocation = 0.025  # 2.5%
specific_min_allocations = {
    'Traps': 0.005,      # 0.5%
    'Neck': 0.005,       # 0.5%
    'Forearms': 0.005,   # 0.5%
    'Abdominals': 0.04   # 4%
}

# General and specific allocation constraints for body parts
unique_body_parts = df['BodyPart'].unique()
for part in unique_body_parts:
    if part in specific_min_allocations:
        model_p.addConstr(
            gb.quicksum(x[i] for i in exercise_ids if body_parts[i] == part) >= specific_min_allocations[part],
            f"{part}Min"
        )
    else:
        model_p.addConstr(
            gb.quicksum(x[i] for i in exercise_ids if body_parts[i] == part) >= general_min_allocation,
            f"{part}GeneralMin"
        )

# 8. Ensure at least 60% of exercises involve essential equipment types.
essential_equipment = ['Barbell', 'Dumbbell', 'Machine', 'Cable', 'E-Z Curl Bar', 'Bands']
equipment_constraint = model_p.addConstr(
    gb.quicksum(x[i] for i in exercise_ids if equipment_types[i] in essential_equipment) >= 0.6,
    "EssentialEquipmentMin"
)

# Ensure total proportion is 1
model_p.addConstr(gb.quicksum(x[i] for i in range(len(df))) == 1, "Total_Proportion")

# Solve the optimization model to find the optimal proportions for each exercise while adhering to all constraints.
# The model successfully converges to an optimal solution with an objective value of 0.7672, maximizing hypertrophy.
model_p.optimize()

if model_p.status == GRB.INFEASIBLE:
    print("Model is infeasible. Computing IIS...")
    model_p.computeIIS()
    model_p.write("infeasible_constraints.ilp")
    
# Check the optimization status
if model_p.status == GRB.OPTIMAL:
    print("The optimal solution: ", model_p.objVal)
    # Optionally, display the decision variables
    for i in exercise_ids:
        if x[i].x > 0:
            print(f"Exercise {df.loc[i, 'Exercise']}: Proportion = {x[i].x:.4f}")
elif model_p.status == GRB.INFEASIBLE:
    print("The model is infeasible.")
elif model_p.status == GRB.UNBOUNDED:
    print("The model is unbounded.")
else:
    print("Optimization was stopped with status", model_p.status)


print("Primal Decision Variable Values:")
for var in model_p.getVars():
    print(f"{var.varName}: {var.x:.4f}")

print(f"Primal Objective Value: {model_p.objVal:.4f}")


print("\n" + "-"*60 + "\n")



"""2. i) Formulate and solve the dual linear program for model in part (h) demonstrating that the 
model you create is, indeed, the correct dual problem of the primal formulation."""

print("\n" + "="*60)
print("  Running Optimization: Question 2 Part I (Dual Linear Program Formulation)  ")
print("="*60 + "\n")

from gurobipy import Model, GRB, quicksum

def solve_dual(df):
    """
    Solve the dual problem for the primal optimization.
    """
    # Create the dual model
    dual_model = Model("Dual_Problem")

    # Primal data
    num_exercises = len(df)
    muscle_groups = {"Traps", "Neck", "Forearms", "Abdominals"}
    equipment_types = ["Barbell", "Dumbbell", "Machine", "Cable", "E-Z Curl Bar", "Bands"]

    # Define muscle group thresholds from the primal
    muscle_thresholds = {
        "Traps": 0.005,
        "Neck": 0.005,
        "Forearms": 0.005,
        "Abdominals": 0.04
    }
    default_body_part_weight = 0.025

    # Dual variables
    mu = dual_model.addVars(num_exercises, lb=0, vtype=GRB.CONTINUOUS, name="mu")
    nu = dual_model.addVars(muscle_groups, lb=0, vtype=GRB.CONTINUOUS, name="nu")
    lambda_var = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="lambda")
    sigma = dual_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="sigma")

    # Objective function: Minimize
    dual_model.setObjective(
        quicksum(0.05 * mu[i] for i in range(num_exercises)) +
        quicksum(nu[g] * muscle_thresholds[g] for g in muscle_groups) +
        lambda_var * 0.60 +
        sigma,
        GRB.MINIMIZE
    )

    # Dual constraints
    for i in range(num_exercises):
        body_part = df['BodyPart'].iloc[i]
        equipment = df['Equipment'].iloc[i]
        hypertrophy_rating = df['Hypertrophy Rating'].iloc[i]

        dual_model.addConstr(
            mu[i] +
            (nu[body_part] if body_part in muscle_groups else 0) +
            (lambda_var if equipment in equipment_types else 0) +
            sigma >= hypertrophy_rating,
            name=f"Dual_Constraint_{i}"
        )

    # Optimize the dual problem
    dual_model.optimize()

    # Output results
    if dual_model.status == GRB.OPTIMAL:
        print("Optimization completed successfully.")
        print(f"Optimal Dual Objective Value: {dual_model.objVal:.4f}")

        # Dual Variables
        print("\nDual Variables:")
        for v in dual_model.getVars():
            print(f"{v.varName}: {v.x:.4f}")

        # Constraint Slackness
        print("\nConstraint Slackness:")
        for constr in dual_model.getConstrs():
            print(f"{constr.constrName}: Slack = {constr.slack:.4f}")

    return dual_model.objVal

# Run the dual problem solver
dual_score = solve_dual(df)
