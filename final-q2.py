import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

cost = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/cost.csv')
time = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/time.csv')

print("="*60 + "\n")
print("2E")
print("="*60 + "\n")

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import time

# Load the cost and time data
cost = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/cost.csv')
time_data = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/time.csv')

# Display to confirm data loaded correctly
print("Cost data shape:", cost.shape)
print("Time data shape:", time_data.shape)

# Extract customer information
customers = cost.columns[1:].tolist()
n_customers = len(customers)
print(f"Number of customers: {n_customers}")

# Set Monte Carlo parameters
n_trials = 50
n_scenarios = 100

# Cost parameters
disposal_cost = 5  # $5 per unit disposal cost
shortage_penalty = 2000  # $2000 per unit shortage penalty
max_conversion_time = 724  # Maximum conversion time in hours

# Function to generate demand scenarios based on customer distributions
def generate_demand_scenarios(n_scenarios):
    scenarios = np.zeros((n_scenarios, n_customers))
    
    # Customer 1: Normal(loc=15000, scale=1000)
    scenarios[:, 0] = np.random.normal(loc=15000, scale=1000, size=n_scenarios)
    
    # Customer 2: Exponential(scale=15000)
    scenarios[:, 1] = np.random.exponential(scale=15000, size=n_scenarios)
    
    # Customer 3: Uniform(low=10000, high=20000)
    scenarios[:, 2] = np.random.uniform(low=10000, high=20000, size=n_scenarios)
    
    # Customer 4: Normal(loc=15000, scale=500)
    scenarios[:, 3] = np.random.normal(loc=15000, scale=500, size=n_scenarios)
    
    # Customer 5: Exponential(scale=7500)
    scenarios[:, 4] = np.random.exponential(scale=7500, size=n_scenarios)
    
    # Customer 6: Uniform(low=12000, high=18000)
    scenarios[:, 5] = np.random.uniform(low=12000, high=18000, size=n_scenarios)
    
    # Customer 7: Normal(loc=15000, scale=2000)
    scenarios[:, 6] = np.random.normal(loc=15000, scale=2000, size=n_scenarios)
    
    # Round to integers and ensure non-negative
    scenarios = np.maximum(np.round(scenarios), 0)
    
    return scenarios

# Convert DataFrame to numpy array for easier access
cost_matrix = cost.iloc[:, 1:].values
time_matrix = time_data.iloc[:, 1:].values

# Function to solve one SAA problem with given scenarios
def solve_saa_instance(scenarios):
    model = gp.Model("Dye_Production_SAA")
    model.setParam('OutputFlag', 0)  # Turn off output for cleaner execution
    
    # First-stage variables: amount of dye to produce for each customer type
    x = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="produce")
    
    # Second-stage variables (for each scenario)
    # Amount to convert from type i to type j
    y = {}
    # Amount of shortage for each customer type
    shortage = {}
    # Amount to dispose for each customer type
    dispose = {}
    
    for s in range(n_scenarios):
        y[s] = model.addVars(n_customers, n_customers, lb=0, vtype=GRB.CONTINUOUS, name=f"convert_{s}")
        shortage[s] = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name=f"shortage_{s}")
        dispose[s] = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name=f"dispose_{s}")
    
    # Objective function: minimize expected cost
    # First calculate the second-stage costs for each scenario
    scenario_costs = []
    
    for s in range(n_scenarios):
        # Conversion costs
        conversion_cost = gp.quicksum(y[s][i, j] * cost_matrix[i, j] 
                                   for i in range(n_customers) 
                                   for j in range(n_customers) if i != j)
        
        # Disposal costs
        disposal_total = gp.quicksum(dispose[s][i] * disposal_cost for i in range(n_customers))
        
        # Shortage penalty costs
        shortage_total = gp.quicksum(shortage[s][i] * shortage_penalty for i in range(n_customers))
        
        # Total cost for this scenario
        scenario_cost = conversion_cost + disposal_total + shortage_total
        scenario_costs.append(scenario_cost)
    
    # Expected cost is the average of all scenario costs
    expected_cost = (1.0 / n_scenarios) * gp.quicksum(scenario_costs)
    model.setObjective(expected_cost, GRB.MINIMIZE)
    
    # Constraints
    # For each scenario
    for s in range(n_scenarios):
        # Flow balance constraints
        for i in range(n_customers):
            # Initial production + incoming conversions - outgoing conversions - disposal = demand - shortage
            model.addConstr(
                x[i] + 
                gp.quicksum(y[s][j, i] for j in range(n_customers) if j != i) - 
                gp.quicksum(y[s][i, j] for j in range(n_customers) if i != j) - 
                dispose[s][i] == 
                scenarios[s, i] - shortage[s][i],
                f"flow_balance_{s}_{i}"
            )
        
        # Conversion time constraint
        model.addConstr(
            gp.quicksum(y[s][i, j] * time_matrix[i, j] for i in range(n_customers) for j in range(n_customers) if i != j) <= max_conversion_time,
            f"conversion_time_{s}"
        )
    
    # Solve the model
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        # Extract the optimal first-stage decisions
        production = [x[i].x for i in range(n_customers)]
        
        # Calculate the objective value
        obj_value = model.objVal
        
        return obj_value, production
    else:
        print(f"Model status: {model.status}")
        return None, None

# Run multiple trials of SAA
start_time = time.time()
trial_results = []
production_decisions = []

print(f"Starting Monte Carlo simulation with {n_trials} trials, each with {n_scenarios} scenarios...")

for trial in range(n_trials):
    print(f"Running trial {trial+1}/{n_trials}...")
    
    # Generate scenarios for this trial
    scenarios = generate_demand_scenarios(n_scenarios)
    
    # Solve the SAA instance
    obj_value, production = solve_saa_instance(scenarios)
    
    if obj_value is not None:
        trial_results.append(obj_value)
        production_decisions.append(production)
        print(f"  Trial {trial+1} completed. Objective value: {obj_value:.2f}")
    else:
        print(f"  Trial {trial+1} failed to find an optimal solution.")

# Calculate the optimal expected cost and confidence interval
if trial_results:
    optimal_expected_cost = np.mean(trial_results)
    std_dev = np.std(trial_results)
    
    # Calculate 95% confidence interval
    n = len(trial_results)
    margin_of_error = 1.96 * (std_dev / np.sqrt(n))  # 1.96 for 95% confidence
    ci_lower = optimal_expected_cost - margin_of_error
    ci_upper = optimal_expected_cost + margin_of_error
    
    # Results
    print("\nResults from SAA with Monte Carlo simulation:")
    print(f"Number of trials: {n_trials}")
    print(f"Number of scenarios per trial: {n_scenarios}")
    print(f"Optimal expected cost: ${optimal_expected_cost:.2f}")
    print(f"Standard deviation: ${std_dev:.2f}")
    print(f"95% confidence interval: (${ci_lower:.2f}, ${ci_upper:.2f})")
    
    # Calculate average optimal production decisions
    avg_production = np.mean(production_decisions, axis=0)
    print("\nAverage optimal production quantities:")
    for i, customer in enumerate(customers):
        print(f"{customer}: {avg_production[i]:.2f} units")
else:
    print("No successful trials to analyze.")

print(f"\nTotal execution time: {(time.time() - start_time):.2f} seconds")

print("="*60 + "\n")
print("2F")
print("="*60 + "\n")

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import time

# Load the cost and time data
cost = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/cost.csv')
time_data = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/time.csv')

# Display to confirm data loaded correctly
print("Cost data shape:", cost.shape)
print("Time data shape:", time_data.shape)

# Extract customer information
customers = cost.columns[1:].tolist()
n_customers = len(customers)
print(f"Number of customers: {n_customers}")

# Set Monte Carlo parameters
n_trials = 50
n_scenarios = 100

# Cost parameters
disposal_cost = 5      # $5 per unit disposal cost
shortage_penalty = 2000  # $2000 per unit shortage penalty
max_conversion_time = 724  # Maximum conversion time in hours

# Function to generate demand scenarios based on customer distributions
def generate_demand_scenarios(n_scenarios):
    scenarios = np.zeros((n_scenarios, n_customers))
    
    # Customer 1: Normal(loc=15000, scale=1000)
    scenarios[:, 0] = np.random.normal(loc=15000, scale=1000, size=n_scenarios)
    # Customer 2: Exponential(scale=15000)
    scenarios[:, 1] = np.random.exponential(scale=15000, size=n_scenarios)
    # Customer 3: Uniform(low=10000, high=20000)
    scenarios[:, 2] = np.random.uniform(low=10000, high=20000, size=n_scenarios)
    # Customer 4: Normal(loc=15000, scale=500)
    scenarios[:, 3] = np.random.normal(loc=15000, scale=500, size=n_scenarios)
    # Customer 5: Exponential(scale=7500)
    scenarios[:, 4] = np.random.exponential(scale=7500, size=n_scenarios)
    # Customer 6: Uniform(low=12000, high=18000)
    scenarios[:, 5] = np.random.uniform(low=12000, high=18000, size=n_scenarios)
    # Customer 7: Normal(loc=15000, scale=2000)
    scenarios[:, 6] = np.random.normal(loc=15000, scale=2000, size=n_scenarios)
    
    # Round to integers and ensure non-negative
    scenarios = np.maximum(np.round(scenarios), 0)
    
    return scenarios

# Convert DataFrame to numpy array for easier access
cost_matrix = cost.iloc[:, 1:].values
time_matrix = time_data.iloc[:, 1:].values

# =============================
# Existing function for SAA solution:
def solve_saa_instance(scenarios):
    model = gp.Model("Dye_Production_SAA")
    model.setParam('OutputFlag', 0)  # Turn off output for cleaner execution
    
    # First-stage variables: amount of dye to produce for each customer type
    x = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="produce")
    
    # Second-stage variables (for each scenario)
    y = {}
    shortage = {}
    dispose = {}
    
    for s in range(n_scenarios):
        y[s] = model.addVars(n_customers, n_customers, lb=0, vtype=GRB.CONTINUOUS, name=f"convert_{s}")
        shortage[s] = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name=f"shortage_{s}")
        dispose[s] = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name=f"dispose_{s}")
    
    # Objective function: minimize expected cost for the SAA model
    scenario_costs = []
    for s in range(n_scenarios):
        conversion_cost = gp.quicksum(y[s][i, j] * cost_matrix[i, j] 
                                        for i in range(n_customers) 
                                        for j in range(n_customers) if i != j)
        disposal_total = gp.quicksum(dispose[s][i] * disposal_cost for i in range(n_customers))
        shortage_total = gp.quicksum(shortage[s][i] * shortage_penalty for i in range(n_customers))
        scenario_cost = conversion_cost + disposal_total + shortage_total
        scenario_costs.append(scenario_cost)
    
    expected_cost = (1.0 / n_scenarios) * gp.quicksum(scenario_costs)
    model.setObjective(expected_cost, GRB.MINIMIZE)
    
    # Constraints: For each scenario s, flow balance and conversion time constraint
    for s in range(n_scenarios):
        for i in range(n_customers):
            model.addConstr(
                x[i] + gp.quicksum(y[s][j, i] for j in range(n_customers) if j != i) - 
                gp.quicksum(y[s][i, j] for j in range(n_customers) if i != j) - 
                dispose[s][i] == scenarios[s, i] - shortage[s][i],
                f"flow_balance_{s}_{i}"
            )
        model.addConstr(
            gp.quicksum(y[s][i, j] * time_matrix[i, j] for i in range(n_customers) for j in range(n_customers) if i != j) <= max_conversion_time,
            f"conversion_time_{s}"
        )
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        production = [x[i].x for i in range(n_customers)]
        obj_value = model.objVal
        return obj_value, production
    else:
        print(f"Model status: {model.status}")
        return None, None

# =============================
# New function for Perfect Foresight (WS) calculation:
def solve_ws_instance(demand_vector):
    """Solve the deterministic model (with perfect information) for a given demand_vector.
    In this model, all decisions (including production) are scenario-specific."""
    model = gp.Model("Dye_Production_WS")
    model.setParam('OutputFlag', 0)
    
    # Decision variables for this scenario (now production is flexible)
    x = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="produce")
    y = model.addVars(n_customers, n_customers, lb=0, vtype=GRB.CONTINUOUS, name="convert")
    shortage = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="shortage")
    dispose = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="dispose")
    
    # Objective function: minimize deterministic cost (conversion + disposal + shortage)
    conversion_cost = gp.quicksum(y[i, j] * cost_matrix[i, j] 
                                   for i in range(n_customers) 
                                   for j in range(n_customers) if i != j)
    disposal_total = gp.quicksum(dispose[i] * disposal_cost for i in range(n_customers))
    shortage_total = gp.quicksum(shortage[i] * shortage_penalty for i in range(n_customers))
    total_cost = conversion_cost + disposal_total + shortage_total
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Flow balance constraints for each customer i
    for i in range(n_customers):
        model.addConstr(
            x[i] + gp.quicksum(y[j, i] for j in range(n_customers) if j != i) -
            gp.quicksum(y[i, j] for j in range(n_customers) if j != i) - dispose[i] ==
            demand_vector[i] - shortage[i],
            name=f"flow_balance_{i}"
        )
    
    # Conversion time constraint for this scenario
    model.addConstr(
        gp.quicksum(y[i, j] * time_matrix[i, j] for i in range(n_customers) for j in range(n_customers) if i != j)
        <= max_conversion_time,
        "conversion_time"
    )
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        print("WS model did not solve optimally.")
        return None

# =============================
# Main simulation: Run multiple SAA trials and, additionally, compute WS costs.
start_time = time.time()
trial_results = []
production_decisions = []
ws_cost_trials = []  # To store WS cost for each trial

print(f"Starting Monte Carlo simulation with {n_trials} trials, each with {n_scenarios} scenarios...")

for trial in range(n_trials):
    print(f"Running trial {trial+1}/{n_trials}...")
    
    # Generate scenarios for this trial
    scenarios = generate_demand_scenarios(n_scenarios)
    
    # Solve the SAA (two-stage stochastic) instance
    obj_value, production = solve_saa_instance(scenarios)
    
    # Compute Wait-and-See cost for this trial by solving the WS model for each scenario
    ws_costs = []
    for s in range(n_scenarios):
        # Extract the demand vector for scenario s
        demand_vector = scenarios[s, :]
        ws_obj = solve_ws_instance(demand_vector)
        if ws_obj is not None:
            ws_costs.append(ws_obj)
    if ws_costs:
        ws_trial = np.mean(ws_costs)
    else:
        ws_trial = None
    
    if obj_value is not None:
        trial_results.append(obj_value)
        production_decisions.append(production)
        ws_cost_trials.append(ws_trial)
        print(f"  Trial {trial+1} completed. Objective value: {obj_value:.2f}, WS cost: {ws_trial:.2f}")
    else:
        print(f"  Trial {trial+1} failed to find an optimal solution.")

# Calculate the optimal expected cost and confidence interval from SAA trials
if trial_results:
    optimal_expected_cost = np.mean(trial_results)
    std_dev = np.std(trial_results)
    
    n = len(trial_results)
    margin_of_error = 1.96 * (std_dev / np.sqrt(n))  # 1.96 for 95% confidence
    ci_lower = optimal_expected_cost - margin_of_error
    ci_upper = optimal_expected_cost + margin_of_error
    
    # Average WS cost across trials (ignoring trials where ws_cost_trial is None)
    ws_cost_trials = [ws for ws in ws_cost_trials if ws is not None]
    overall_ws_cost = np.mean(ws_cost_trials) if ws_cost_trials else None
    
    print("\nResults from SAA with Monte Carlo simulation:")
    print(f"Number of trials: {n_trials}")
    print(f"Number of scenarios per trial: {n_scenarios}")
    print(f"Optimal expected cost: ${optimal_expected_cost:.2f}")
    print(f"Standard deviation: ${std_dev:.2f}")
    print(f"95% confidence interval: (${ci_lower:.2f}, ${ci_upper:.2f})")
    
    if overall_ws_cost is not None:
        print(f"Average Wait-and-See (WS) cost: ${overall_ws_cost:.2f}")
        EVPI = optimal_expected_cost - overall_ws_cost
        print(f"Expected Value of Perfect Information (EVPI): ${EVPI:.2f}")
    else:
        print("WS cost could not be computed for all trials.")
    
    # Compute average production decisions across trials
    avg_production = np.mean(production_decisions, axis=0)
    print("\nAverage optimal production quantities:")
    for i, customer in enumerate(customers):
        print(f"{customer}: {avg_production[i]:.2f} units")
else:
    print("No successful trials to analyze.")

print(f"\nTotal execution time: {(time.time() - start_time):.2f} seconds")

print("="*60 + "\n")
print("2G")
print("="*60 + "\n")

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import time

# ---------------------------
# Load Data
# ---------------------------
cost = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/cost.csv')
time_data = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/time.csv')

print("Cost data shape:", cost.shape)
print("Time data shape:", time_data.shape)

# Extract customer names (assume first column is a label)
customers = cost.columns[1:].tolist()
n_customers = len(customers)
print(f"Number of customers: {n_customers}")

# ---------------------------
# Parameters
# ---------------------------
n_trials = 50         # Number of Monte Carlo trials
n_scenarios = 100     # Scenarios per trial

# Cost parameters
disposal_cost = 5      # $5 per unit disposal cost
shortage_penalty = 2000  # $2000 per unit shortage penalty
max_conversion_time = 724  # Maximum conversion time in hours

# Convert cost and time data to numpy arrays (skip first column which is label)
cost_matrix = cost.iloc[:, 1:].values
time_matrix = time_data.iloc[:, 1:].values

# ---------------------------
# Demand Scenario Generator
# ---------------------------
def generate_demand_scenarios(n_scenarios):
    scenarios = np.zeros((n_scenarios, n_customers))
    
    # Use the given distributions:
    # Customer 1: Normal(loc=15000, scale=1000)
    scenarios[:, 0] = np.random.normal(loc=15000, scale=1000, size=n_scenarios)
    # Customer 2: Exponential(scale=15000) → mean=15000
    scenarios[:, 1] = np.random.exponential(scale=15000, size=n_scenarios)
    # Customer 3: Uniform(low=10000, high=20000) → mean=15000
    scenarios[:, 2] = np.random.uniform(low=10000, high=20000, size=n_scenarios)
    # Customer 4: Normal(loc=15000, scale=500)
    scenarios[:, 3] = np.random.normal(loc=15000, scale=500, size=n_scenarios)
    # Customer 5: Exponential(scale=7500) → mean=7500
    scenarios[:, 4] = np.random.exponential(scale=7500, size=n_scenarios)
    # Customer 6: Uniform(low=12000, high=18000) → mean=15000
    scenarios[:, 5] = np.random.uniform(low=12000, high=18000, size=n_scenarios)
    # Customer 7: Normal(loc=15000, scale=2000)
    scenarios[:, 6] = np.random.normal(loc=15000, scale=2000, size=n_scenarios)
    
    # Round and ensure nonnegative values
    scenarios = np.maximum(np.round(scenarios), 0)
    return scenarios

# ---------------------------
# SAA Two-Stage Model Function
# ---------------------------
def solve_saa_instance(scenarios):
    model = gp.Model("Dye_Production_SAA")
    model.setParam('OutputFlag', 0)
    
    # First-stage decision: production amounts (here-and-now)
    x = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="produce")
    
    # Second-stage decisions (per scenario)
    y = {}
    shortage = {}
    dispose = {}
    for s in range(n_scenarios):
        y[s] = model.addVars(n_customers, n_customers, lb=0, vtype=GRB.CONTINUOUS, name=f"convert_{s}")
        shortage[s] = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name=f"shortage_{s}")
        dispose[s] = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name=f"dispose_{s}")
    
    # Build objective: expected second-stage recourse cost
    scenario_costs = []
    for s in range(n_scenarios):
        conversion_cost = gp.quicksum(y[s][i, j] * cost_matrix[i, j]
                                        for i in range(n_customers)
                                        for j in range(n_customers) if i != j)
        disposal_total = gp.quicksum(dispose[s][i] * disposal_cost for i in range(n_customers))
        shortage_total = gp.quicksum(shortage[s][i] * shortage_penalty for i in range(n_customers))
        scenario_cost = conversion_cost + disposal_total + shortage_total
        scenario_costs.append(scenario_cost)
    
    expected_cost = (1.0 / n_scenarios) * gp.quicksum(scenario_costs)
    model.setObjective(expected_cost, GRB.MINIMIZE)
    
    # Constraints (applied for each scenario)
    for s in range(n_scenarios):
        for i in range(n_customers):
            model.addConstr(
                x[i] + gp.quicksum(y[s][j, i] for j in range(n_customers) if j != i) -
                gp.quicksum(y[s][i, j] for j in range(n_customers) if j != i) - dispose[s][i] ==
                scenarios[s, i] - shortage[s][i],
                name=f"flow_balance_{s}_{i}"
            )
        model.addConstr(
            gp.quicksum(y[s][i, j] * time_matrix[i, j]
                        for i in range(n_customers) for j in range(n_customers) if i != j)
            <= max_conversion_time,
            name=f"conversion_time_{s}"
        )
    
    model.optimize()
    if model.status == GRB.OPTIMAL:
        production = [x[i].x for i in range(n_customers)]
        return model.objVal, production
    else:
        print("SAA model status:", model.status)
        return None, None

# ---------------------------
# Mean-Value Problem (MVP) Function
# ---------------------------
def solve_mvp_instance(mean_demand):
    model = gp.Model("Dye_Production_MVP")
    model.setParam('OutputFlag', 0)
    
    # Decision variables (all decisions are flexible)
    x = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="produce")
    y = model.addVars(n_customers, n_customers, lb=0, vtype=GRB.CONTINUOUS, name="convert")
    shortage = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="shortage")
    dispose = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="dispose")
    
    conversion_cost = gp.quicksum(y[i, j] * cost_matrix[i, j]
                                  for i in range(n_customers)
                                  for j in range(n_customers) if i != j)
    disposal_total = gp.quicksum(dispose[i] * disposal_cost for i in range(n_customers))
    shortage_total = gp.quicksum(shortage[i] * shortage_penalty for i in range(n_customers))
    total_cost = conversion_cost + disposal_total + shortage_total
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    for i in range(n_customers):
        model.addConstr(
            x[i] + gp.quicksum(y[j, i] for j in range(n_customers) if j != i) -
            gp.quicksum(y[i, j] for j in range(n_customers) if j != i) - dispose[i] ==
            mean_demand[i] - shortage[i],
            name=f"flow_balance_{i}"
        )
    model.addConstr(
        gp.quicksum(y[i, j] * time_matrix[i, j]
                    for i in range(n_customers) for j in range(n_customers) if i != j)
        <= max_conversion_time,
        name="conversion_time"
    )
    
    model.optimize()
    if model.status == GRB.OPTIMAL:
        production = [x[i].x for i in range(n_customers)]
        return model.objVal, production
    else:
        print("MVP model status:", model.status)
        return None, None

# ---------------------------
# Evaluate Fixed Production Decision (for EEV)
# ---------------------------
def evaluate_fixed_production(production_fixed, scenarios):
    recourse_costs = []
    for s in range(scenarios.shape[0]):
        demand_vector = scenarios[s, :]
        model = gp.Model("Evaluate_Fixed_Production")
        model.setParam('OutputFlag', 0)
        
        # In this model, production is fixed at production_fixed
        y = model.addVars(n_customers, n_customers, lb=0, vtype=GRB.CONTINUOUS, name="convert")
        shortage = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="shortage")
        dispose = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="dispose")
        
        conversion_cost = gp.quicksum(y[i, j] * cost_matrix[i, j]
                                      for i in range(n_customers)
                                      for j in range(n_customers) if i != j)
        disposal_total = gp.quicksum(dispose[i] * disposal_cost for i in range(n_customers))
        shortage_total = gp.quicksum(shortage[i] * shortage_penalty for i in range(n_customers))
        total_cost = conversion_cost + disposal_total + shortage_total
        model.setObjective(total_cost, GRB.MINIMIZE)
        
        for i in range(n_customers):
            model.addConstr(
                production_fixed[i] +
                gp.quicksum(y[j, i] for j in range(n_customers) if j != i) -
                gp.quicksum(y[i, j] for j in range(n_customers) if j != i) - dispose[i] ==
                demand_vector[i] - shortage[i],
                name=f"flow_balance_{s}_{i}"
            )
        model.addConstr(
            gp.quicksum(y[i, j] * time_matrix[i, j]
                        for i in range(n_customers)
                        for j in range(n_customers) if i != j)
            <= max_conversion_time,
            name=f"conversion_time_{s}"
        )
        
        model.optimize()
        if model.status == GRB.OPTIMAL:
            recourse_costs.append(model.objVal)
        else:
            recourse_costs.append(np.nan)
    return np.nanmean(recourse_costs)

# ---------------------------
# Main Monte Carlo Simulation and Evaluation
# ---------------------------
start_time = time.time()
trial_results = []         # SAA two-stage objective values
production_decisions = []  # First-stage production from SAA trials

print(f"Starting Monte Carlo simulation with {n_trials} trials, each with {n_scenarios} scenarios...\n")
for trial in range(n_trials):
    print(f"Running trial {trial+1}/{n_trials}...")
    scenarios_trial = generate_demand_scenarios(n_scenarios)
    obj_value, production = solve_saa_instance(scenarios_trial)
    if obj_value is not None:
        trial_results.append(obj_value)
        production_decisions.append(production)
        print(f"  Trial {trial+1} completed. Objective value: {obj_value:.2f}")
    else:
        print(f"  Trial {trial+1} failed to solve optimally.")

if trial_results:
    two_stage_expected_cost = np.mean(trial_results)
    std_dev = np.std(trial_results)
    n = len(trial_results)
    margin_of_error = 1.96 * (std_dev / np.sqrt(n))
    ci_lower = two_stage_expected_cost - margin_of_error
    ci_upper = two_stage_expected_cost + margin_of_error

    print("\nResults from SAA (Two-stage stochastic solution):")
    print(f"Number of trials: {n_trials}")
    print(f"Number of scenarios per trial: {n_scenarios}")
    print(f"Optimal expected cost (two-stage): ${two_stage_expected_cost:.2f}")
    print(f"Standard deviation: ${std_dev:.2f}")
    print(f"95% confidence interval: (${ci_lower:.2f}, ${ci_upper:.2f})")
else:
    print("No successful trials from the two-stage model.")

# ---------------------------
# Solve the Mean-Value (MVP) Problem and Evaluate (EEV)
# ---------------------------
# Define the mean demand based on our distributions:
# Customer 1: 15000, Customer 2: 15000, Customer 3: 15000, Customer 4: 15000,
# Customer 5: 7500,  Customer 6: 15000, Customer 7: 15000.
mean_demand = np.array([15000, 15000, 15000, 15000, 7500, 15000, 15000])
mvp_obj, mvp_production = solve_mvp_instance(mean_demand)
if mvp_obj is not None:
    print(f"\nMean-Value Problem (MVP) solution objective: ${mvp_obj:.2f}")
    print("MVP production decision:")
    for i, prod in enumerate(mvp_production):
        print(f"  {customers[i]}: {prod:.2f} units")
else:
    print("MVP did not solve optimally.")

# Evaluate the fixed production decision from MVP over new scenarios to obtain the EEV
scenarios_for_eev = generate_demand_scenarios(n_scenarios)
EEV = evaluate_fixed_production(mvp_production, scenarios_for_eev)
if EEV is not None:
    print(f"\nExpected cost using MVP production decision (EEV): ${EEV:.2f}")
else:
    print("Failed to compute EEV.")

# ---------------------------
# Compute VSS (Value of the Stochastic Solution)
# ---------------------------
if EEV is not None and trial_results:
    VSS = EEV - two_stage_expected_cost
    print(f"Value of the Stochastic Solution (VSS): ${VSS:.2f}")
else:
    print("Insufficient data to compute VSS.")

# ---------------------------
# Average Production Decisions from SAA Trials
# ---------------------------
if production_decisions:
    avg_production = np.mean(production_decisions, axis=0)
    print("\nAverage optimal production quantities from SAA trials:")
    for i, cust in enumerate(customers):
        print(f"{cust}: {avg_production[i]:.2f} units")
    
print(f"\nTotal execution time: {(time.time() - start_time):.2f} seconds")

print("="*60 + "\n")
print("2I")
print("="*60 + "\n")

import gurobipy as gp
from gurobipy import GRB

# Define the number of customers and their names
customers = ["Customer 1", "Customer 2", "Customer 3", "Customer 4", "Customer 5", "Customer 6", "Customer 7"]
n_customers = len(customers)

# Create a new Gurobi model
model = gp.Model("Robust_Dye_Production_No_Recourse")

# Decision variables: production amounts for each customer.
# Since there is no recourse (conversion, shortage, or disposal actions),
# these are the only decisions.
x = model.addVars(n_customers, lb=0, vtype=GRB.CONTINUOUS, name="produce")

# In a box-uncertainty setting with no conversion, each customer’s demand 
# can be anywhere between 10,000 and 20,000. To guarantee feasibility
# under every possible demand realization, you must produce at least 20,000 units.
for i in range(n_customers):
    model.addConstr(x[i] >= 20000, name=f"robust_min_production_{i}")

# OPTIONAL: If you have production cost per unit for each customer, include it in the objective.
# For illustration, assume a simple production cost of $1 per unit for every customer.
production_cost = [1] * n_customers  # cost per unit for each customer

# Objective: Minimize total production cost
model.setObjective(gp.quicksum(production_cost[i] * x[i] for i in range(n_customers)), GRB.MINIMIZE)

# Optimize the model.
model.optimize()

# After optimization, print the production decision for each customer.
print("\nRobust Production Decisions:")
for i in range(n_customers):
    print(f"{customers[i]}: Production = {x[i].x:.2f} units")
    
print("="*60 + "\n")
print("2J")
print("="*60 + "\n")

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Define the matrix P from the question (7 x 7)
P_list = [
    [2250000, 200000,      0,       0,       0,       0,       0      ],
    [200000, 2250000, 300000,     0,       0,       0,       0      ],
    [0,      300000, 2250000, 150000,       0,       0,       0      ],
    [0,      0,      150000, 2250000, 400000,     0,    0      ],
    [0,      0,      0,      400000, 2250000, 250000,     0      ],
    [0,      0,      0,      0,      250000, 2250000, 200000    ],
    [0,      0,      0,      0,      0,      200000, 2250000]
]
P = np.array(P_list, dtype=float)

# Number of customers
n = P.shape[0]

# Invert P to handle the ellipsoidal bound
P_inv = np.linalg.inv(P)

# Compute worst-case coordinate shift for each i:
# alpha_i = sqrt(e_i^T P_inv e_i )
# e_i is the i-th unit vector
worst_case_shifts = []
for i in range(n):
    # e_i
    e_i = np.zeros(n)
    e_i[i] = 1.0
    # alpha_i
    alpha_i = np.sqrt(e_i @ P_inv @ e_i)
    worst_case_shifts.append(alpha_i)

# Build Gurobi model: x_i >= 17500 + alpha_i
model = gp.Model("Ellipsoidal_Robust_Production")
model.setParam('OutputFlag', 1)  # turn on solver output for diagnostics

# Decision variables: x_i
x = model.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")

# Constraints: x_i >= 17500 + alpha_i
for i in range(n):
    model.addConstr(x[i] >= 17500 + worst_case_shifts[i], f"robust_constr_{i}")

# Total production constraint
model.addConstr(gp.quicksum(x[i] for i in range(n)) >= 17500, "total_production")

# Objective: minimize sum(x_i)
model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MINIMIZE)

# Solve
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!\n")
    # Production amounts
    production = [x[i].X for i in range(n)]
    total_cost = model.objVal
    
    for i in range(n):
        print(f"Customer {i+1} production: {production[i]:.2f}")
    print(f"\nTotal Production Cost (Objective): {total_cost:.2f}")
    print(f"Total Production: {sum(production):.2f}")
else:
    print(f"Model not solved to optimality. Status code: {model.status}")