print("\n" + "="*100)
print("QUESTION 2: FUELFLOW")
print("="*100 + "\n")

print("\n" + "="*100)
print("2. e) Sample Average Approximation (SAA)")
print("="*100 + "\n")

# ========================================================
# Assignment Part (e) – Sample Average Approximation (SAA)
# FuelFlow Logistics – OMIS 6000
# ========================================================

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# -----------------------------
# Load Input Data
# -----------------------------
costs_df = pd.read_csv("https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/costs.csv", index_col=0)
randomness_df = pd.read_csv("https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/randomness.csv", index_col=0)

# Station IDs: 1 through 14 (excluding Station_0, the depot)
stations = list(range(1, len(randomness_df) + 1))
probabilities = randomness_df["Probability"].to_numpy()
means = randomness_df["Mean_Demand"].to_numpy()
stds = randomness_df["Std_Dev_Demand"].to_numpy()
cost_matrix = costs_df.values  # Shape: (15 x 15)

# -----------------------------
# Scenario Generator
# -----------------------------
def generate_scenarios(num_scenarios, rng=None):
    """
    Generate a list of scenarios, where each scenario is a dict:
    {
        'active_stations': list of station IDs needing fuel,
        'demands': {station_id: demand}
    }
    """
    if rng is None:
        rng = np.random.default_rng()

    scenarios = []
    for _ in range(num_scenarios):
        active = []
        demands = {}
        for i, s in enumerate(stations):
            if rng.random() < probabilities[i]:
                d = max(0.0, rng.normal(loc=means[i], scale=stds[i]))
                demands[s] = d
                active.append(s)
            else:
                demands[s] = 0.0
        scenarios.append({'active_stations': active, 'demands': demands})
    return scenarios

# -----------------------------
# Build and Solve SAA Model
# -----------------------------
def solve_saa_model(scenarios):
    """
    Solves the SAA model by building a single Gurobi MIP with:
    - A shared truck size variable (first-stage)
    - Scenario-specific routing and penalty variables (second-stage)
    - Subtour elimination via MTZ constraints
    """
    m = gp.Model("FuelFlow_SAA")
    m.setParam("OutputFlag", 0)

    # First-stage variable
    truck_size = m.addVar(lb=0, name="TruckSize")

    scenario_costs = []

    for s_idx, scenario in enumerate(scenarios):
        s_name = f"s{s_idx}"
        active = scenario["active_stations"]
        demands = scenario["demands"]
        total_demand = sum(demands[s] for s in active)

        # Penalty variables
        over = m.addVar(lb=0, name=f"over_{s_name}")
        under = m.addVar(lb=0, name=f"under_{s_name}")

        # Capacity mismatch constraints
        m.addConstr(over >= truck_size - total_demand, name=f"OverCap_{s_name}")
        m.addConstr(under >= total_demand - truck_size, name=f"UnderCap_{s_name}")

        if active:
            nodes = [0] + active  # Depot is node 0
            x = m.addVars(nodes, nodes, vtype=GRB.BINARY, name=f"x_{s_name}")
            u = m.addVars(active, lb=1, ub=len(active), name=f"u_{s_name}")

            # Routing constraints
            m.addConstr(gp.quicksum(x[0, j] for j in active) == 1, name=f"DepotOut_{s_name}")
            m.addConstr(gp.quicksum(x[j, 0] for j in active) == 1, name=f"DepotIn_{s_name}")
            for i in active:
                m.addConstr(gp.quicksum(x[k, i] for k in nodes if k != i) == 1, name=f"In_{s_name}_{i}")
                m.addConstr(gp.quicksum(x[i, k] for k in nodes if k != i) == 1, name=f"Out_{s_name}_{i}")

            # Subtour elimination (MTZ)
            for i in active:
                for j in active:
                    if i != j:
                        m.addConstr(
                            u[i] - u[j] + len(active)*x[i, j] <= len(active) - 1,
                            name=f"MTZ_{s_name}_{i}_{j}"
                        )

            # Travel cost
            travel = gp.quicksum(cost_matrix[i, j] * x[i, j] for i in nodes for j in nodes if i != j)
        else:
            travel = 0

        penalty = 0.09 * over + 0.13 * under
        scenario_costs.append(travel + penalty)

    # Objective: average cost across all scenarios
    m.setObjective((1.0 / len(scenarios)) * gp.quicksum(scenario_costs), GRB.MINIMIZE)
    m.optimize()

    return m.objVal

# -----------------------------
# SAA Experiment Runner
# -----------------------------
def run_saa(trials=20, scenarios_per_trial=10, seed=2025):
    """
    Runs the full SAA experiment with fixed random seed.
    Each trial involves solving one SAA model on new scenarios.
    """
    rng = np.random.default_rng(seed)
    trial_costs = []

    for t in range(trials):
        scens = generate_scenarios(scenarios_per_trial, rng)
        obj = solve_saa_model(scens)
        trial_costs.append(obj)
        print(f"Trial {t+1}: Cost = {obj:.2f}")

    avg = np.mean(trial_costs)
    print(f"\nFinal SAA Estimate of Optimal Expected Cost: ${avg:.2f}")
    return avg

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    run_saa()


print("\n" + "="*100)
print("2. f) – Wait-and-See (WS) and EVPI")
print("="*100 + "\n")
# ========================================================
# Assignment Part (f) – Wait-and-See (WS) and EVPI
# FuelFlow Logistics – OMIS 6000
# ========================================================

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np


stations = list(range(1, len(randomness_df) + 1))  # Station IDs: 1–14 (Station_0 is depot)
probabilities = randomness_df["Probability"].to_numpy()
means = randomness_df["Mean_Demand"].to_numpy()
stds = randomness_df["Std_Dev_Demand"].to_numpy()
cost_matrix = costs_df.values  # 15x15 matrix

# -----------------------------
# Scenario Generator (Monte Carlo)
# -----------------------------
def generate_scenario(rng):
    """
    Generate a single scenario based on Bernoulli(p) for activation
    and Normal(mean, std) for demand when active.
    """
    active = []
    demands = {}
    for i, s in enumerate(stations):
        if rng.random() < probabilities[i]:
            d = max(0.0, rng.normal(loc=means[i], scale=stds[i]))
            demands[s] = d
            active.append(s)
        else:
            demands[s] = 0.0
    return active, demands

# -----------------------------
# Solve TSP with Perfect Foresight (WS)
# -----------------------------
def solve_perfect_information(active_stations, demands):
    """
    Solve VRP with known active stations and demand.
    No penalties for over/under capacity – perfect foresight.
    Only routing costs are minimized.
    """
    m = gp.Model("WS_TSP")
    m.setParam("OutputFlag", 0)

    if not active_stations:
        return 0.0  # No delivery required

    nodes = [0] + active_stations  # 0 is depot
    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    u = m.addVars(active_stations, lb=1, ub=len(active_stations), name="u")

    # Objective: Minimize travel cost
    m.setObjective(gp.quicksum(cost_matrix[i, j] * x[i, j] for i, j in arcs), GRB.MINIMIZE)

    # Depot constraints
    m.addConstr(gp.quicksum(x[0, j] for j in active_stations) == 1)
    m.addConstr(gp.quicksum(x[j, 0] for j in active_stations) == 1)

    # Each station has one in/out arc
    for i in active_stations:
        m.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) == 1)
        m.addConstr(gp.quicksum(x[j, i] for j in nodes if j != i) == 1)

    # Subtour elimination (MTZ constraints)
    for i in active_stations:
        for j in active_stations:
            if i != j:
                m.addConstr(u[i] - u[j] + len(active_stations) * x[i, j] <= len(active_stations) - 1)

    m.optimize()
    return m.objVal

# -----------------------------
# Run WS Trials and Compute EVPI
# -----------------------------
def run_ws_and_evpi(saa_cost, trials=20, scenarios_per_trial=10, seed=2025):
    """
    Runs Monte Carlo WS estimation and computes EVPI vs SAA.
    """
    rng = np.random.default_rng(seed)
    trial_costs = []

    for t in range(trials):
        scenario_costs = []
        for _ in range(scenarios_per_trial):
            active, demands = generate_scenario(rng)
            cost = solve_perfect_information(active, demands)
            scenario_costs.append(cost)
        trial_avg = np.mean(scenario_costs)
        trial_costs.append(trial_avg)
        print(f"Trial {t+1}: WS Cost = {trial_avg:.2f}")

    ws_estimate = np.mean(trial_costs)
    evpi = saa_cost - ws_estimate
    print(f"\nWait-and-See (WS) Expected Cost: ${ws_estimate:.2f}")
    print(f"EVPI (Expected Value of Perfect Information): ${evpi:.2f}")
    return ws_estimate, evpi

run_ws_and_evpi(saa_cost=158.94)


print("\n" + "="*100)
print("2. g) EEV & VSS")
print("="*100 + "\n")

# ========================================================
# Assignment Part (g) – EEV and VSS 
# FuelFlow Logistics – OMIS 6000
# ========================================================

import gurobipy as gp
from gurobipy import GRB
import numpy as np

stations = list(range(1, len(randomness_df) + 1))
probabilities = randomness_df["Probability"].to_numpy()
means = randomness_df["Mean_Demand"].to_numpy()
stds = randomness_df["Std_Dev_Demand"].to_numpy()
cost_matrix = costs_df.values

# -----------------------------
# Generate a Scenario
# -----------------------------
def generate_scenario(rng):
    active = []
    demands = {}
    for i, s in enumerate(stations):
        if rng.random() < probabilities[i]:
            d = max(0.0, rng.normal(loc=means[i], scale=stds[i]))
            demands[s] = d
            active.append(s)
        else:
            demands[s] = 0.0
    return active, demands

# -----------------------------
# Solve Mean Value Problem (EVP)
# -----------------------------
def solve_mean_value_model():
    m = gp.Model("MeanValueVRP")
    m.setParam("OutputFlag", 0)

    expected_demands = probabilities * means
    active = [s for s in stations if expected_demands[s - 1] > 0]
    total_demand = sum(expected_demands[s - 1] for s in active)

    truck_size = m.addVar(lb=0, name="TruckSize")
    over = m.addVar(lb=0, name="Over")
    under = m.addVar(lb=0, name="Under")

    nodes = [0] + active
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    u = m.addVars(active, lb=1, ub=len(active), name="u")

    travel_cost = gp.quicksum(cost_matrix[i, j] * x[i, j] for i, j in arcs)
    penalty = 0.09 * over + 0.13 * under
    m.setObjective(travel_cost + penalty, GRB.MINIMIZE)

    m.addConstr(over >= truck_size - total_demand)
    m.addConstr(under >= total_demand - truck_size)

    m.addConstr(gp.quicksum(x[0, j] for j in active) == 1)
    m.addConstr(gp.quicksum(x[j, 0] for j in active) == 1)
    for i in active:
        m.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) == 1)
        m.addConstr(gp.quicksum(x[j, i] for j in nodes if j != i) == 1)
    for i in active:
        for j in active:
            if i != j:
                m.addConstr(u[i] - u[j] + len(active)*x[i, j] <= len(active) - 1)

    m.optimize()

    # Reconstruct the route from x values
    route = [0]
    visited = {0}
    current = 0
    while len(visited) < len(nodes):
        for j in nodes:
            if j != current and x[current, j].X > 0.5:
                route.append(j)
                visited.add(j)
                current = j
                break
    route.append(0)  # Return to depot
    return truck_size.X, route

# -----------------------------
# Evaluate EVP Solution on Stochastic Scenarios
# -----------------------------
def evaluate_eev(truck_size_val, ev_route, trials=20, scenarios_per_trial=10, seed=2025):
    rng = np.random.default_rng(seed)
    trial_costs = []

    for t in range(trials):
        scen_costs = []
        for _ in range(scenarios_per_trial):
            _, demands = generate_scenario(rng)

            # Total demand along fixed route
            total_demand = sum(demands[s] for s in ev_route if s != 0)
            over = max(0, truck_size_val - total_demand)
            under = max(0, total_demand - truck_size_val)

            # Fixed travel cost from route
            travel = sum(cost_matrix[ev_route[i], ev_route[i+1]] for i in range(len(ev_route) - 1))
            penalty = 0.09 * over + 0.13 * under
            scen_costs.append(travel + penalty)

        trial_avg = np.mean(scen_costs)
        trial_costs.append(trial_avg)
        print(f"Trial {t+1}: EEV Cost = {trial_avg:.2f}")

    eev = np.mean(trial_costs)
    print(f"\nEEV (Expected cost using EV solution): ${eev:.2f}")
    return eev

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    saa_cost = 158.94  # Replace with actual SAA cost
    ev_truck_size, ev_route = solve_mean_value_model()
    eev = evaluate_eev(ev_truck_size, ev_route)
    vss = eev - saa_cost
    print(f"VSS (Value of the Stochastic Solution): ${vss:.2f}")
