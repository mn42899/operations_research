import gurobipy as gb
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


####### PART E #######


# Read data
costs_df = pd.read_csv("https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/costs.csv", index_col=0)
randomness_df = pd.read_csv("https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/randomness.csv", index_col=0)


# Parameters
excess_capacity_penalty = 0.09  # $ per liter
insufficient_capacity_penalty = 0.13  # $ per liter

# Extract stations (excluding depot)
nodes = list(costs_df.index)
depot = 'Station_0'
stations = [station for station in nodes if station != depot]
n = len(stations)

# Extract travel costs between all nodes
travel_costs = {}
for i in nodes:
    for j in nodes:
        if i != j:
            travel_costs[i, j] = costs_df.loc[i, j]

# Extract demand probabilities and distributions
probabilities = {}
demand_means = {}
demand_stddevs = {}

for station in stations:
    probabilities[station] = randomness_df.loc[station, 'Probability']
    demand_means[station] = randomness_df.loc[station, 'Mean_Demand']
    demand_stddevs[station] = randomness_df.loc[station, 'Std_Dev_Demand']

# The number of trials to perform
trials = 20
    
# The number of scenarios per trial
scenarios = 10

# Range of truck capacities to evaluate
min_capacity = 500
max_capacity = 3000
capacity_step = 100
capacities = list(range(min_capacity, max_capacity + capacity_step, capacity_step))

# Store results
results = {}

for capacity in capacities:
    print(f"Evaluating truck capacity: {capacity} liters")
    total_cost = 0
    
    for trial in range(trials):
        # Create a new model for this trial
        model = gb.Model("FuelFlow Logistics Truck Sizing")
        model.setParam('OutputFlag', 0)
        
        # Generate scenarios for this trial
        D = {}  # Dictionary to store demand for each station and scenario
        active_stations = {}  # Dictionary to track which stations are active in each scenario
        
        for k in range(scenarios):
            D[k] = {}
            active_stations[k] = []
            
            for station in stations:
                # Determine if station needs refueling
                if np.random.random() < probabilities[station]:
                    # Generate demand from normal distribution
                    demand = max(0, int(np.random.normal(demand_means[station], demand_stddevs[station])))
                    D[k][station] = demand
                    
                    if demand > 0:
                        active_stations[k].append(station)
                else:
                    D[k][station] = 0
        
        # Decision variables
        # x[i,j,k] = 1 if truck travels from node i to node j in scenario k
        x = model.addVars([(i, j, k) for i in nodes for j in nodes for k in range(scenarios) 
                          if i != j], vtype=GRB.BINARY, name="x")
        
        # u[i,k] = Position of node i in the route for scenario k (for subtour elimination)
        u = model.addVars([(i, k) for i in stations for k in range(scenarios)], 
                         vtype=GRB.INTEGER, lb=0, name="u")
        
        # over[k] = excess capacity in scenario k
        over = model.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name="over")
        
        # under[k] = insufficient capacity in scenario k
        under = model.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name="under")
        
        # Objective function: minimize expected total cost
        obj = (1.0/scenarios) * (
            # Travel costs
            gb.quicksum(travel_costs[i, j] * x[i, j, k] 
                      for i in nodes for j in nodes for k in range(scenarios) 
                      if i != j) +
            # Capacity mismatch penalties
            gb.quicksum(excess_capacity_penalty * over[k] for k in range(scenarios)) +
            gb.quicksum(insufficient_capacity_penalty * under[k] for k in range(scenarios))
        )
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        for k in range(scenarios):
            # Calculate total demand for this scenario
            total_demand_k = sum(D[k][station] for station in stations)
            
            # Capacity constraints
            model.addConstr(over[k] >= capacity - total_demand_k, name=f"over_{k}")
            model.addConstr(under[k] >= total_demand_k - capacity, name=f"under_{k}")
            
            # Each active station must be visited exactly once
            for j in active_stations[k]:
                model.addConstr(gb.quicksum(x[i, j, k] for i in nodes if i != j) == 1, 
                              name=f"visit_{j}_{k}")
            
            # Vehicle must leave each visited station
            for i in active_stations[k]:
                model.addConstr(gb.quicksum(x[i, j, k] for j in nodes if i != j) == 1, 
                              name=f"leave_{i}_{k}")
            
            # Vehicle must leave the depot
            if active_stations[k]:  # Only if there are active stations
                model.addConstr(gb.quicksum(x[depot, j, k] for j in nodes if j != depot) == 1, 
                              name=f"depot_out_{k}")
                
                # Vehicle must return to the depot
                model.addConstr(gb.quicksum(x[i, depot, k] for i in nodes if i != depot) == 1, 
                              name=f"depot_in_{k}")
                
                # Subtour elimination constraints (MTZ formulation)
                for i in active_stations[k]:
                    model.addConstr(u[i, k] >= 2, name=f"mtz_lb_{i}_{k}")
                    model.addConstr(u[i, k] <= len(active_stations[k]) + 1, name=f"mtz_ub_{i}_{k}")
                
                # If we travel from i to j, ensure u[j] = u[i] + 1
                M = len(active_stations[k]) + 1  # Big-M value
                for i in active_stations[k]:
                    for j in active_stations[k]:
                        if i != j:
                            model.addConstr(u[j, k] >= u[i, k] + 1 - M * (1 - x[i, j, k]), 
                                         name=f"mtz_{i}_{j}_{k}")
            else:
                # If no active stations, no travel cost
                for i in nodes:
                    for j in nodes:
                        if i != j:
                            model.addConstr(x[i, j, k] == 0, name=f"no_travel_{i}_{j}_{k}")
        
        # Optimize the model
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # Add this trial's objective to the total
            total_cost += model.objVal
        else:
            print(f"Warning: Model for capacity {capacity}, trial {trial} failed to solve optimally")
    
    # Store average cost across all trials
    results[capacity] = total_cost / trials

# Find optimal capacity
optimal_capacity = min(results, key=results.get)
optimal_cost = results[optimal_capacity]

print(f"\nOptimal truck capacity: {optimal_capacity} liters")
print(f"Expected daily cost: ${optimal_cost:.2f}")

# Plot results
capacities_list = list(results.keys())
costs_list = list(results.values())

plt.figure(figsize=(10, 6))
plt.plot(capacities_list, costs_list, 'o-')
plt.axvline(x=optimal_capacity, color='r', linestyle='--', label=f'Optimal capacity: {optimal_capacity}L')
plt.xlabel('Truck Capacity (liters)')
plt.ylabel('Expected Daily Cost ($)')
plt.title('Expected Cost vs. Truck Capacity')
plt.grid(True)
plt.legend()
plt.show()







######## PART F ##########


# Function to solve the VRP for a specific scenario (perfect information)
def solve_vrp_for_scenario(scenario_demands, active_stations_list):
    model = gb.Model("VRP_Scenario")
    model.setParam('OutputFlag', 0)
    
    # Calculate total demand for this scenario
    total_demand = sum(scenario_demands.values())
    
    # Decision variables
    # x[i,j] = 1 if truck travels from node i to node j
    x = model.addVars([(i, j) for i in nodes for j in nodes 
                       if i != j], vtype=GRB.BINARY, name="x")
    
    # u[i] = Position of node i in the route (for subtour elimination)
    u = model.addVars(stations, vtype=GRB.INTEGER, lb=0, name="u")
    
    # Objective function: travel costs only
    obj = gb.quicksum(travel_costs[i, j] * x[i, j] 
                   for i in nodes for j in nodes if i != j)
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # Each active station must be visited exactly once
    for j in active_stations_list:
        model.addConstr(gb.quicksum(x[i, j] for i in nodes if i != j) == 1, 
                       name=f"visit_{j}")
    
    # Vehicle must leave each visited station
    for i in active_stations_list:
        model.addConstr(gb.quicksum(x[i, j] for j in nodes if i != j) == 1, 
                       name=f"leave_{i}")
    
    # Vehicle must leave the depot
    if active_stations_list:  # Only if there are active stations
        model.addConstr(gb.quicksum(x[depot, j] for j in nodes if j != depot) == 1, 
                       name="depot_out")
        
        # Vehicle must return to the depot
        model.addConstr(gb.quicksum(x[i, depot] for i in nodes if i != depot) == 1, 
                       name="depot_in")
        
        # Subtour elimination constraints (MTZ formulation)
        for i in active_stations_list:
            model.addConstr(u[i] >= 2, name=f"mtz_lb_{i}")
            model.addConstr(u[i] <= len(active_stations_list) + 1, name=f"mtz_ub_{i}")
        
        # If we travel from i to j, ensure u[j] = u[i] + 1
        M = len(active_stations_list) + 1  # Big-M value
        for i in active_stations_list:
            for j in active_stations_list:
                if i != j:
                    model.addConstr(u[j] >= u[i] + 1 - M * (1 - x[i, j]), 
                                   name=f"mtz_{i}_{j}")
    else:
        # If no active stations, no travel cost
        for i in nodes:
            for j in nodes:
                if i != j:
                    model.addConstr(x[i, j] == 0, name=f"no_travel_{i}_{j}")
    
    # Optimize the model
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        travel_cost = model.objVal
        return travel_cost, total_demand
    else:
        print("Warning: VRP model failed to solve optimally")
        return float('inf'), total_demand

# Function to get the optimal capacity for a specific demand
def get_optimal_capacity_for_demand(total_demand):
    # Calculate costs for different capacities
    capacity_costs = {}
    
    for capacity in capacities:
        # Calculate capacity mismatch penalties
        if total_demand > capacity:
            # Insufficient capacity penalty
            penalty = insufficient_capacity_penalty * (total_demand - capacity)
        else:
            # Excess capacity penalty
            penalty = excess_capacity_penalty * (capacity - total_demand)
        
        capacity_costs[capacity] = penalty
    
    # Return capacity with minimum cost
    return min(capacity_costs, key=capacity_costs.get)

# Store results
results = {}
results_ws = {}  # Wait-and-See results

print("Part 1: Computing Here-and-Now Solution (RP)")
for capacity in capacities:
    print(f"Evaluating truck capacity: {capacity} liters")
    total_cost = 0
    
    for trial in range(trials):
        # Create a new model for this trial
        model = gb.Model("FuelFlow Logistics Truck Sizing")
        model.setParam('OutputFlag', 0)
        
        # Generate scenarios for this trial
        D = {}  # Dictionary to store demand for each station and scenario
        active_stations = {}  # Dictionary to track which stations are active in each scenario
        
        for k in range(scenarios):
            D[k] = {}
            active_stations[k] = []
            
            for station in stations:
                # Determine if station needs refueling
                if np.random.random() < probabilities[station]:
                    # Generate demand from normal distribution
                    demand = max(0, int(np.random.normal(demand_means[station], demand_stddevs[station])))
                    D[k][station] = demand
                    
                    if demand > 0:
                        active_stations[k].append(station)
                else:
                    D[k][station] = 0
        
        # Decision variables
        # x[i,j,k] = 1 if truck travels from node i to node j in scenario k
        x = model.addVars([(i, j, k) for i in nodes for j in nodes for k in range(scenarios) 
                          if i != j], vtype=GRB.BINARY, name="x")
        
        # u[i,k] = Position of node i in the route for scenario k (for subtour elimination)
        u = model.addVars([(i, k) for i in stations for k in range(scenarios)], 
                         vtype=GRB.INTEGER, lb=0, name="u")
        
        # over[k] = excess capacity in scenario k
        over = model.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name="over")
        
        # under[k] = insufficient capacity in scenario k
        under = model.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name="under")
        
        # Objective function: minimize expected total cost
        obj = (1.0/scenarios) * (
            # Travel costs
            gb.quicksum(travel_costs[i, j] * x[i, j, k] 
                      for i in nodes for j in nodes for k in range(scenarios) 
                      if i != j) +
            # Capacity mismatch penalties
            gb.quicksum(excess_capacity_penalty * over[k] for k in range(scenarios)) +
            gb.quicksum(insufficient_capacity_penalty * under[k] for k in range(scenarios))
        )
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        for k in range(scenarios):
            # Calculate total demand for this scenario
            total_demand_k = sum(D[k][station] for station in stations)
            
            # Capacity constraints
            model.addConstr(over[k] >= capacity - total_demand_k, name=f"over_{k}")
            model.addConstr(under[k] >= total_demand_k - capacity, name=f"under_{k}")
            
            # Each active station must be visited exactly once
            for j in active_stations[k]:
                model.addConstr(gb.quicksum(x[i, j, k] for i in nodes if i != j) == 1, 
                              name=f"visit_{j}_{k}")
            
            # Vehicle must leave each visited station
            for i in active_stations[k]:
                model.addConstr(gb.quicksum(x[i, j, k] for j in nodes if i != j) == 1, 
                              name=f"leave_{i}_{k}")
            
            # Vehicle must leave the depot
            if active_stations[k]:  # Only if there are active stations
                model.addConstr(gb.quicksum(x[depot, j, k] for j in nodes if j != depot) == 1, 
                              name=f"depot_out_{k}")
                
                # Vehicle must return to the depot
                model.addConstr(gb.quicksum(x[i, depot, k] for i in nodes if i != depot) == 1, 
                              name=f"depot_in_{k}")
                
                # Subtour elimination constraints (MTZ formulation)
                for i in active_stations[k]:
                    model.addConstr(u[i, k] >= 2, name=f"mtz_lb_{i}_{k}")
                    model.addConstr(u[i, k] <= len(active_stations[k]) + 1, name=f"mtz_ub_{i}_{k}")
                
                # If we travel from i to j, ensure u[j] = u[i] + 1
                M = len(active_stations[k]) + 1  # Big-M value
                for i in active_stations[k]:
                    for j in active_stations[k]:
                        if i != j:
                            model.addConstr(u[j, k] >= u[i, k] + 1 - M * (1 - x[i, j, k]), 
                                         name=f"mtz_{i}_{j}_{k}")
            else:
                # If no active stations, no travel cost
                for i in nodes:
                    for j in nodes:
                        if i != j:
                            model.addConstr(x[i, j, k] == 0, name=f"no_travel_{i}_{j}_{k}")
        
        # Optimize the model
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # Add this trial's objective to the total
            total_cost += model.objVal
            
            # Compute Wait-and-See solution for this trial
            ws_cost_trial = 0
            
            for k in range(scenarios):
                # Solve the VRP for this scenario
                travel_cost, total_demand = solve_vrp_for_scenario(D[k], active_stations[k])
                
                # Find the optimal capacity for this scenario
                optimal_capacity = get_optimal_capacity_for_demand(total_demand)
                
                # Calculate capacity mismatch penalties
                if total_demand > optimal_capacity:
                    # Insufficient capacity penalty
                    penalty = insufficient_capacity_penalty * (total_demand - optimal_capacity)
                else:
                    # Excess capacity penalty
                    penalty = excess_capacity_penalty * (optimal_capacity - total_demand)
                
                # Total cost for this scenario with perfect information
                scenario_cost = travel_cost + penalty
                ws_cost_trial += scenario_cost / scenarios
            
            # Add to Wait-and-See results
            if capacity not in results_ws:
                results_ws[capacity] = 0
            results_ws[capacity] += ws_cost_trial / trials
                
        else:
            print(f"Warning: Model for capacity {capacity}, trial {trial} failed to solve optimally")
    
    # Store average cost across all trials
    results[capacity] = total_cost / trials

# Find optimal capacity for Here-and-Now solution
optimal_capacity = min(results, key=results.get)
optimal_cost = results[optimal_capacity]  # This is RP (Recourse Problem solution)

# Calculate Wait-and-See solution (WS)
ws_value = min(results_ws.values())  # This is the expected cost with perfect information

# Calculate EVPI
evpi = optimal_cost - ws_value

print("\n--- Results ---")
print(f"Optimal truck capacity (Here-and-Now): {optimal_capacity} liters")
print(f"Expected cost with recourse (RP): ${optimal_cost:.2f}")
print(f"Expected cost with perfect information (WS): ${ws_value:.2f}")
print(f"Expected Value of Perfect Information (EVPI): ${evpi:.2f}")
print(f"EVPI as percentage of RP: {(evpi/optimal_cost)*100:.2f}%")

# Plot results
plt.figure(figsize=(12, 8))

# Plot Here-and-Now results
plt.subplot(2, 1, 1)
capacities_list = list(results.keys())
costs_list = list(results.values())
plt.plot(capacities_list, costs_list, 'o-b', label='Here-and-Now (RP)')
plt.axvline(x=optimal_capacity, color='r', linestyle='--', label=f'Optimal capacity: {optimal_capacity}L')
plt.axhline(y=optimal_cost, color='r', linestyle=':')
plt.xlabel('Truck Capacity (liters)')
plt.ylabel('Expected Daily Cost ($)')
plt.title('Here-and-Now Solution (RP)')
plt.grid(True)
plt.legend()

# Plot Wait-and-See vs Here-and-Now
plt.subplot(2, 1, 2)
capacities_list = list(results.keys())
rp_costs = list(results.values())
ws_costs = [results_ws[c] for c in capacities_list]
evpi_values = [rp_costs[i] - ws_costs[i] for i in range(len(capacities_list))]

plt.plot(capacities_list, rp_costs, 'o-b', label='Here-and-Now (RP)')
plt.plot(capacities_list, ws_costs, 'o-g', label='Wait-and-See (WS)')
plt.fill_between(capacities_list, rp_costs, ws_costs, alpha=0.2, color='red', label='EVPI')
plt.axvline(x=optimal_capacity, color='r', linestyle='--', label=f'Optimal capacity: {optimal_capacity}L')
plt.xlabel('Truck Capacity (liters)')
plt.ylabel('Expected Daily Cost ($)')
plt.title('Comparison of RP and WS Solutions (EVPI)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Plot EVPI values
plt.figure(figsize=(10, 6))
plt.bar(capacities_list, evpi_values)
plt.xlabel('Truck Capacity (liters)')
plt.ylabel('EVPI Value ($)')
plt.title('Expected Value of Perfect Information by Capacity')
plt.grid(True, axis='y')
plt.show()







####### PART G #######

# Function to solve VRP for a given scenario
def solve_vrp_for_scenario(scenario_demands, active_stations_list):
    model = gb.Model("VRP_Scenario")
    model.setParam('OutputFlag', 0)
    
    # Calculate total demand for this scenario
    total_demand = sum(scenario_demands.values())
    
    # Decision variables
    # x[i,j] = 1 if truck travels from node i to node j
    x = model.addVars([(i, j) for i in nodes for j in nodes 
                       if i != j], vtype=GRB.BINARY, name="x")
    
    # u[i] = Position of node i in the route (for subtour elimination)
    u = model.addVars(stations, vtype=GRB.INTEGER, lb=0, name="u")
    
    # Objective function: travel costs only
    obj = gb.quicksum(travel_costs[i, j] * x[i, j] 
                   for i in nodes for j in nodes if i != j)
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # Each active station must be visited exactly once
    for j in active_stations_list:
        model.addConstr(gb.quicksum(x[i, j] for i in nodes if i != j) == 1, 
                       name=f"visit_{j}")
    
    # Vehicle must leave each visited station
    for i in active_stations_list:
        model.addConstr(gb.quicksum(x[i, j] for j in nodes if i != j) == 1, 
                       name=f"leave_{i}")
    
    # Vehicle must leave the depot
    if active_stations_list:  # Only if there are active stations
        model.addConstr(gb.quicksum(x[depot, j] for j in nodes if j != depot) == 1, 
                       name="depot_out")
        
        # Vehicle must return to the depot
        model.addConstr(gb.quicksum(x[i, depot] for i in nodes if i != depot) == 1, 
                       name="depot_in")
        
        # Subtour elimination constraints (MTZ formulation)
        for i in active_stations_list:
            model.addConstr(u[i] >= 2, name=f"mtz_lb_{i}")
            model.addConstr(u[i] <= len(active_stations_list) + 1, name=f"mtz_ub_{i}")
        
        # If we travel from i to j, ensure u[j] = u[i] + 1
        M = len(active_stations_list) + 1  # Big-M value
        for i in active_stations_list:
            for j in active_stations_list:
                if i != j:
                    model.addConstr(u[j] >= u[i] + 1 - M * (1 - x[i, j]), 
                                   name=f"mtz_{i}_{j}")
    else:
        # If no active stations, no travel cost
        for i in nodes:
            for j in nodes:
                if i != j:
                    model.addConstr(x[i, j] == 0, name=f"no_travel_{i}_{j}")
    
    # Optimize the model
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        travel_cost = model.objVal
        return travel_cost, total_demand
    else:
        print("Warning: VRP model failed to solve optimally")
        return float('inf'), total_demand

# STEP 1: Solve the Mean Value Problem (EV) to get the mean value solution
print("STEP 1: Solving the Mean Value Problem (EV)")

# Calculate the expected demand for each station
expected_demands = {}
expected_active_stations = []

for station in stations:
    # Expected demand = probability * mean demand
    expected_demands[station] = probabilities[station] * demand_means[station]
    if expected_demands[station] > 0:
        expected_active_stations.append(station)

# Solve the VRP for the expected demand
travel_cost_mean, total_expected_demand = solve_vrp_for_scenario(expected_demands, expected_active_stations)

# Find the optimal capacity for the mean value problem
mean_value_costs = {}
for capacity in capacities:
    # Calculate capacity mismatch penalties
    if total_expected_demand > capacity:
        # Insufficient capacity penalty
        penalty = insufficient_capacity_penalty * (total_expected_demand - capacity)
    else:
        # Excess capacity penalty
        penalty = excess_capacity_penalty * (capacity - total_expected_demand)
    
    # Total cost for this capacity
    mean_value_costs[capacity] = travel_cost_mean + penalty

# Find the optimal capacity for the mean value problem
ev_optimal_capacity = min(mean_value_costs, key=mean_value_costs.get)
ev_optimal_cost = mean_value_costs[ev_optimal_capacity]

print(f"Mean Value Problem (EV) results:")
print(f"Optimal truck capacity: {ev_optimal_capacity} liters")
print(f"Expected total demand: {total_expected_demand:.2f} liters")
print(f"Optimal cost: ${ev_optimal_cost:.2f}")

# STEP 2: Calculate the Expected result of using the EV solution (EEV)
print("\nSTEP 2: Calculating EEV - Expected cost when using mean value solution")

# We'll use the capacity from the mean value problem (ev_optimal_capacity)
# and evaluate it on random scenarios
eev_total_cost = 0

for trial in range(trials):
    eev_trial_cost = 0
    
    # Generate scenarios for this trial
    for k in range(scenarios):
        # Generate a random scenario
        scenario_demands = {}
        active_stations_list = []
        
        for station in stations:
            # Determine if station needs refueling
            if np.random.random() < probabilities[station]:
                # Generate demand from normal distribution
                demand = max(0, int(np.random.normal(demand_means[station], demand_stddevs[station])))
                scenario_demands[station] = demand
                
                if demand > 0:
                    active_stations_list.append(station)
            else:
                scenario_demands[station] = 0
        
        # Solve the VRP for this scenario
        travel_cost, total_demand = solve_vrp_for_scenario(scenario_demands, active_stations_list)
        
        # Calculate capacity mismatch penalties using the EV solution's capacity
        if total_demand > ev_optimal_capacity:
            # Insufficient capacity penalty
            penalty = insufficient_capacity_penalty * (total_demand - ev_optimal_capacity)
        else:
            # Excess capacity penalty
            penalty = excess_capacity_penalty * (ev_optimal_capacity - total_demand)
        
        # Total cost for this scenario
        scenario_cost = travel_cost + penalty
        eev_trial_cost += scenario_cost / scenarios
    
    # Add this trial's cost to the total
    eev_total_cost += eev_trial_cost / trials

print(f"Expected value of the EV solution (EEV): ${eev_total_cost:.2f}")

# STEP 3: Solve the Recourse Problem (RP) - stochastic solution
print("\nSTEP 3: Solving the Recourse Problem (RP)")

# Store results
rp_results = {}

for capacity in capacities:
    print(f"Evaluating truck capacity: {capacity} liters")
    total_cost = 0
    
    for trial in range(trials):
        # Create a new model for this trial
        model = gb.Model("FuelFlow Logistics Truck Sizing")
        model.setParam('OutputFlag', 0)
        
        # Generate scenarios for this trial
        D = {}  # Dictionary to store demand for each station and scenario
        active_stations = {}  # Dictionary to track which stations are active in each scenario
        
        for k in range(scenarios):
            D[k] = {}
            active_stations[k] = []
            
            for station in stations:
                # Determine if station needs refueling
                if np.random.random() < probabilities[station]:
                    # Generate demand from normal distribution
                    demand = max(0, int(np.random.normal(demand_means[station], demand_stddevs[station])))
                    D[k][station] = demand
                    
                    if demand > 0:
                        active_stations[k].append(station)
                else:
                    D[k][station] = 0
        
        # Decision variables
        # x[i,j,k] = 1 if truck travels from node i to node j in scenario k
        x = model.addVars([(i, j, k) for i in nodes for j in nodes for k in range(scenarios) 
                          if i != j], vtype=GRB.BINARY, name="x")
        
        # u[i,k] = Position of node i in the route for scenario k (for subtour elimination)
        u = model.addVars([(i, k) for i in stations for k in range(scenarios)], 
                         vtype=GRB.INTEGER, lb=0, name="u")
        
        # over[k] = excess capacity in scenario k
        over = model.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name="over")
        
        # under[k] = insufficient capacity in scenario k
        under = model.addVars(scenarios, vtype=GRB.CONTINUOUS, lb=0, name="under")
        
        # Objective function: minimize expected total cost
        obj = (1.0/scenarios) * (
            # Travel costs
            gb.quicksum(travel_costs[i, j] * x[i, j, k] 
                      for i in nodes for j in nodes for k in range(scenarios) 
                      if i != j) +
            # Capacity mismatch penalties
            gb.quicksum(excess_capacity_penalty * over[k] for k in range(scenarios)) +
            gb.quicksum(insufficient_capacity_penalty * under[k] for k in range(scenarios))
        )
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        for k in range(scenarios):
            # Calculate total demand for this scenario
            total_demand_k = sum(D[k][station] for station in stations)
            
            # Capacity constraints
            model.addConstr(over[k] >= capacity - total_demand_k, name=f"over_{k}")
            model.addConstr(under[k] >= total_demand_k - capacity, name=f"under_{k}")
            
            # Each active station must be visited exactly once
            for j in active_stations[k]:
                model.addConstr(gb.quicksum(x[i, j, k] for i in nodes if i != j) == 1, 
                              name=f"visit_{j}_{k}")
            
            # Vehicle must leave each visited station
            for i in active_stations[k]:
                model.addConstr(gb.quicksum(x[i, j, k] for j in nodes if i != j) == 1, 
                              name=f"leave_{i}_{k}")
            
            # Vehicle must leave the depot
            if active_stations[k]:  # Only if there are active stations
                model.addConstr(gb.quicksum(x[depot, j, k] for j in nodes if j != depot) == 1, 
                              name=f"depot_out_{k}")
                
                # Vehicle must return to the depot
                model.addConstr(gb.quicksum(x[i, depot, k] for i in nodes if i != depot) == 1, 
                              name=f"depot_in_{k}")
                
                # Subtour elimination constraints (MTZ formulation)
                for i in active_stations[k]:
                    model.addConstr(u[i, k] >= 2, name=f"mtz_lb_{i}_{k}")
                    model.addConstr(u[i, k] <= len(active_stations[k]) + 1, name=f"mtz_ub_{i}_{k}")
                
                # If we travel from i to j, ensure u[j] = u[i] + 1
                M = len(active_stations[k]) + 1  # Big-M value
                for i in active_stations[k]:
                    for j in active_stations[k]:
                        if i != j:
                            model.addConstr(u[j, k] >= u[i, k] + 1 - M * (1 - x[i, j, k]), 
                                         name=f"mtz_{i}_{j}_{k}")
            else:
                # If no active stations, no travel cost
                for i in nodes:
                    for j in nodes:
                        if i != j:
                            model.addConstr(x[i, j, k] == 0, name=f"no_travel_{i}_{j}_{k}")
        
        # Optimize the model
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # Add this trial's objective to the total
            total_cost += model.objVal
        else:
            print(f"Warning: Model for capacity {capacity}, trial {trial} failed to solve optimally")
    
    # Store average cost across all trials
    rp_results[capacity] = total_cost / trials

# Find optimal capacity for RP
rp_optimal_capacity = min(rp_results, key=rp_results.get)
rp_optimal_cost = rp_results[rp_optimal_capacity]

print(f"Recourse Problem (RP) results:")
print(f"Optimal truck capacity: {rp_optimal_capacity} liters")
print(f"Expected cost: ${rp_optimal_cost:.2f}")

# STEP 4: Calculate VSS
vss = eev_total_cost - rp_optimal_cost

print("\n--- Final Results ---")
print(f"Mean Value Problem (EV) optimal capacity: {ev_optimal_capacity} liters")
print(f"Mean Value Problem (EV) cost: ${ev_optimal_cost:.2f}")
print(f"Expected value of the EV solution (EEV): ${eev_total_cost:.2f}")
print(f"Recourse Problem (RP) optimal capacity: {rp_optimal_capacity} liters")
print(f"Recourse Problem (RP) cost: ${rp_optimal_cost:.2f}")
print(f"Value of Stochastic Solution (VSS = EEV - RP): ${vss:.2f}")
print(f"VSS as percentage of RP: {(vss/rp_optimal_cost)*100:.2f}%")

# Plot results
plt.figure(figsize=(15, 10))

# Plot RP vs EEV results
plt.subplot(2, 1, 1)
capacities_list = list(rp_results.keys())
rp_costs = list(rp_results.values())

plt.plot(capacities_list, rp_costs, 'o-b', label='Stochastic Solution (RP)')
plt.axhline(y=eev_total_cost, color='g', linestyle='-', label=f'Mean Value Solution (EEV): ${eev_total_cost:.2f}')
plt.axvline(x=rp_optimal_capacity, color='b', linestyle='--', label=f'RP Optimal Capacity: {rp_optimal_capacity}L')
plt.axvline(x=ev_optimal_capacity, color='g', linestyle='--', label=f'EV Optimal Capacity: {ev_optimal_capacity}L')
plt.axhline(y=rp_optimal_cost, color='b', linestyle=':')
plt.fill_between(capacities_list, 
                [eev_total_cost] * len(capacities_list), 
                [min(eev_total_cost, rp_cost) for rp_cost in rp_costs], 
                where=[eev_total_cost > rp_cost for rp_cost in rp_costs],
                alpha=0.2, color='red', label='VSS')
plt.xlabel('Truck Capacity (liters)')
plt.ylabel('Expected Daily Cost ($)')
plt.title('Comparison of Stochastic (RP) and Mean Value (EEV) Solutions')
plt.grid(True)
plt.legend()

# Plot EV, EEV, and RP comparison
plt.subplot(2, 1, 2)
solutions = ['EV', 'EEV', 'RP']
costs = [ev_optimal_cost, eev_total_cost, rp_optimal_cost]
capacities = [ev_optimal_capacity, ev_optimal_capacity, rp_optimal_capacity]  # EEV uses EV capacity

plt.bar(solutions, costs, color=['green', 'orange', 'blue'])
plt.text(0, ev_optimal_cost + 1, f"${ev_optimal_cost:.2f}\n{ev_optimal_capacity}L", ha='center')
plt.text(1, eev_total_cost + 1, f"${eev_total_cost:.2f}\n{ev_optimal_capacity}L", ha='center')
plt.text(2, rp_optimal_cost + 1, f"${rp_optimal_cost:.2f}\n{rp_optimal_capacity}L", ha='center')

# Add VSS annotation
plt.annotate(f"VSS = ${vss:.2f}", 
             xy=(1.5, (eev_total_cost + rp_optimal_cost)/2),
             xytext=(1.7, (eev_total_cost + rp_optimal_cost)/2 + 5),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=12)

plt.ylabel('Expected Daily Cost ($)')
plt.title('Cost Comparison of Different Solution Approaches')
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()