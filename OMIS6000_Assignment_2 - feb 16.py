print("\n" + "="*100)
print("  QUESTION 1 - DYNAMIC PRICING FOR GADGETMARKET INC.")
print("="*100 + "\n")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

df = pd.read_csv('https://raw.githubusercontent.com/kristxna/Datasets/refs/heads/main/price_response.csv')
df.head()


print("\n" + "-"*100)
print("Question 1 - Part A")
print("-"*100 + "\n")

from scipy.optimize import minimize

# Revenue function for Week 1
def revenue_week1(P):
    return -(1000 * P - 5 * P**2)  # Negative for minimization

# Revenue function for Week 2
def revenue_week2(P):
    return -(950 * P - 4.5 * P**2)  # Negative for minimization

# Constraints: P >= 0
bounds = [(0, None)]

# Solve Week 1
result_week1 = minimize(revenue_week1, x0=[50], bounds=bounds)
optimal_price_week1 = result_week1.x[0]

# Solve Week 2
result_week2 = minimize(revenue_week2, x0=[50], bounds=bounds)
optimal_price_week2 = result_week2.x[0]


print(f"Optimal Price for Week 1: {optimal_price_week1}")
print(f"Optimal Price for Week 2: {optimal_price_week2}")


print("\n" + "-"*100)
print("Question 1 - Part B")
print("-"*100 + "\n")

#Revenue function for Week 1 and 2 
def revenue_week1and2(P): 
    return -(1950*P - 9.5 * P**2)

# Constraints: P >= 0
bounds = [(0, None)]

# Solve Week 1
result_week1and2 = minimize(revenue_week1and2, x0=[50], bounds=bounds)
optimal_price_week1and2 = result_week1and2.x[0]
optimal_revenue_2 = -result_week1and2.fun  # Reverse negation to get max revenue


print(f"Optimal Price: {optimal_price_week1and2}")



print("\n" + "-"*100)
print("Question 1 - Part C")
print("-"*100 + "\n")

# comparing the Maxium revenues
optimal_revenue_df = -result_week1.fun - result_week2.fun
optimal_revenue_2 = -result_week1and2.fun  # Reverse negation to get max revenue
print(f"Optimal Revenue with Equal Prices:{optimal_revenue_2} ")
print(f"Optimal Revenue with Different Prices:{optimal_revenue_df} ")



print("\n" + "-"*100)
print("Question 1 - Part D")
print("-"*100 + "\n")

df.head()
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/kristxna/Datasets/refs/heads/main/price_response.csv')

# Filter for Week 1 and Week 2 only
df_filtered = df[df["Week"].isin([1, 2])]

# Extract coefficients for TechFit Smartwatch
techfit_params = df_filtered[df_filtered["Product"] == "TechFit Smartwatch"].select_dtypes(include=[np.number]).mean()
intercept_techfit = techfit_params["Intercept"]
own_price_coeff_techfit = techfit_params["Own_Price_Coefficient"]
cross_price_coeff_techfit = techfit_params["Cross_Price_Coefficient"]

# Extract coefficients for PowerSound Earbuds
powersound_params = df_filtered[df_filtered["Product"] == "PowerSound Earbuds"].select_dtypes(include=[np.number]).mean()
intercept_powersound = powersound_params["Intercept"]
own_price_coeff_powersound = powersound_params["Own_Price_Coefficient"]
cross_price_coeff_powersound = powersound_params["Cross_Price_Coefficient"]

# Define hyperparameters
eta = 0.001  # Step size
tolerance = 1e-6  # Stopping criterion
max_iterations = 10000  # Max number of iterations

# Initialize prices
Pwat, Pear = 0.0, 0.0  # Using new variable names Pwat for Smartwatch, Pear for Earbuds

# Projected Gradient Descent Algorithm with Cross-Price Effects
for _ in range(max_iterations):
    # Compute gradients 
    grad_Pwat = intercept_techfit + 2 * own_price_coeff_techfit * Pwat + cross_price_coeff_techfit * Pear
    grad_Pear = intercept_powersound + 2 * own_price_coeff_powersound * Pear + cross_price_coeff_powersound * Pwat

    # Gradient step
    new_Pwat = Pwat + eta * grad_Pwat
    new_Pear = Pear + eta * grad_Pear

    # Projection step (ensure non-negativity)
    new_Pwat = max(0, new_Pwat)
    new_Pear = max(0, new_Pear)

    # Check stopping condition
    if abs(new_Pwat - Pwat) < tolerance and abs(new_Pear - Pear) < tolerance:
        break

    # Update prices
    Pwat, Pear = new_Pwat, new_Pear

# Output the optimal prices
print(f"Optimal Price for TechFit Smartwatch: ${Pwat:.6f}")
print(f"Optimal Price for PowerSound Earbuds: ${Pear:.6f}")



print("\n" + "-"*100)
print("Question 1 - Part F")
print("-"*100 + "\n")

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/kristxna/Datasets/refs/heads/main/price_response.csv')
df.set_index(['Week', 'Product'], inplace=True)

# Extract coefficients
demand_coefficients = df[['Intercept', 'Own_Price_Coefficient', 'Cross_Price_Coefficient']].to_dict(orient='index')

# Create model
model = gp.Model("Optimal_Pricing")

weeks = range(1, 18)
products = ["TechFit Smartwatch", "PowerSound Earbuds"]
M = 10000  # Big-M constant

# Decision Variables
price_vars = model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=20, ub=500, name="Price")
demand_vars = model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=0, name="Demand")
revenue_vars = model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=0, name="Revenue")
aux_vars = model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=0, name="Aux")  # Auxiliary variable

# Binary Variables for Pricing Constraints
binary_vars = model.addVars(weeks, products, vtype=GRB.BINARY, name="Binary")

# Demand Constraints (Fully Linear)
for w in weeks:
    for p in products:
        other_p = "TechFit Smartwatch" if p == "PowerSound Earbuds" else "PowerSound Earbuds"
        model.addConstr(
            demand_vars[w, p] == demand_coefficients[(w, p)]['Intercept'] +
                                demand_coefficients[(w, p)]['Own_Price_Coefficient'] * price_vars[w, p] +
                                demand_coefficients[(w, p)]['Cross_Price_Coefficient'] * price_vars[w, other_p]
        )

# Revenue Linearization Using Big-M
for w in weeks:
    for p in products:
        # Define lower and upper bounds for demand and price
        demand_lb = 0
        demand_ub = 10000  # Adjust based on realistic estimates
        price_lb = 20
        price_ub = 500

        # Enforce linear constraints for auxiliary variable
        model.addConstr(aux_vars[w, p] >= price_lb * demand_vars[w, p] + price_vars[w, p] * demand_lb - price_lb * demand_lb)
        model.addConstr(aux_vars[w, p] >= price_ub * demand_vars[w, p] + price_vars[w, p] * demand_ub - price_ub * demand_ub)
        model.addConstr(aux_vars[w, p] <= price_ub * demand_vars[w, p] + price_vars[w, p] * demand_lb - price_ub * demand_lb)
        model.addConstr(aux_vars[w, p] <= price_lb * demand_vars[w, p] + price_vars[w, p] * demand_ub - price_lb * demand_ub)

        # Revenue is approximated using aux_vars
        model.addConstr(revenue_vars[w, p] == aux_vars[w, p])

# Objective Function (Maximize Total Revenue)
model.setObjective(gp.quicksum(revenue_vars[w, p] for w in weeks for p in products), GRB.MAXIMIZE)

# Pricing Constraints
for p in products:
    # Weeks 1-4: Static Pricing
    for w in range(2, 5):
        model.addConstr(price_vars[w, p] == price_vars[1, p])

    # Weeks 5-8: Discounted Pricing
    for w in range(6, 9):
        model.addConstr(price_vars[w, p] <= price_vars[1, p] - 10)
        model.addConstr(price_vars[w, p] >= price_vars[1, p] - 20)

    # Weeks 9-11: Increased Pricing
    for w in range(10, 12):
        model.addConstr(price_vars[w, p] == price_vars[9, p])
    model.addConstr(price_vars[9, p] >= price_vars[1, p] + 20)

    # Black Friday (Week 12) - Should be the lowest by $5
    for w in weeks:
        if w != 12:
            model.addConstr(price_vars[12, p] <= price_vars[w, p] - 5)

    # Weeks 13-15: Moderate Pricing
    for w in range(14, 16):
        model.addConstr(price_vars[w, p] == price_vars[13, p])

    # Week 16: Special Constraint (At least $4 higher than Black Friday but $6 lower than any other week)
    model.addConstr(price_vars[16, p] >= price_vars[12, p] + 4)
    for w in weeks:
        if w != 12 and w != 16:
            model.addConstr(price_vars[16, p] <= price_vars[w, p] - 6)

    # Week 17: Peak Pricing (Highest by $15)
    for w in weeks:
        if w != 17:
            model.addConstr(price_vars[17, p] >= price_vars[w, p] + 15)

# Solve Model
model.optimize()

# Display Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal Revenue: {model.objVal}")
    for w in weeks:
        print(f"Week {w}: " + ", ".join([f"{p} = {price_vars[w, p].x:.2f}" for p in products]))
else:
    print("No optimal solution found.")




print("\n" + "-"*100)
print("Question 1 - Part G")
print("-"*100 + "\n")

import matplotlib.pyplot as plt
import seaborn as sns

# Extract optimal price results
days = list(weeks)
optimal_prices_smartwatch = [price_vars[w, "TechFit Smartwatch"].X for w in weeks]
optimal_prices_earbuds = [price_vars[w, "PowerSound Earbuds"].X for w in weeks]

# Plot price trends
sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))

# Plot price dynamics
plt.plot(days, optimal_prices_smartwatch, marker='o', color="#FF9999", label="TechFit Smartwatch")
plt.plot(days, optimal_prices_earbuds, marker='s', color="#7FB3D5", label="PowerSound Earbuds")

# Annotate Black Friday and Boxing Day
plt.axvline(x=12, color='purple', linestyle='--', alpha=0.6, label="Black Friday (Week 12)")
plt.axvline(x=16, color='blue', linestyle='--', alpha=0.6, label="Boxing Day (Week 16)")

# Labels and legend
plt.xlabel("Week")
plt.ylabel("Optimal Price ($)")
plt.title("Price Dynamics Over 17 Weeks")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()



print("\n" + "-"*100)
print("Question 1 - Part H")
print("-"*100 + "\n")

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/kristxna/Datasets/refs/heads/main/price_response.csv')
df.set_index(['Week', 'Product'], inplace=True)

# Extract coefficients
demand_coefficients = df[['Intercept', 'Own_Price_Coefficient', 'Cross_Price_Coefficient']].to_dict(orient='index')

weeks = range(1, 18)
products = ["TechFit Smartwatch", "PowerSound Earbuds"]
M = 10000  # Big-M constant

# ============================
# SCENARIO 1: Fully Dynamic Pricing (No Constraints)
# ============================
dynamic_model = gp.Model("Fully_Dynamic_Pricing")

# Decision Variables
price_dynamic = dynamic_model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=20, ub=500, name="Price_Dynamic")
demand_dynamic = dynamic_model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=0, name="Demand_Dynamic")
revenue_dynamic = dynamic_model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=0, name="Revenue_Dynamic")
aux_dynamic = dynamic_model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=0, name="Aux_Dynamic")

# Demand Constraints (Fully Linear)
for w in weeks:
    for p in products:
        other_p = "TechFit Smartwatch" if p == "PowerSound Earbuds" else "PowerSound Earbuds"
        dynamic_model.addConstr(
            demand_dynamic[w, p] == demand_coefficients[(w, p)]['Intercept'] +
                                    demand_coefficients[(w, p)]['Own_Price_Coefficient'] * price_dynamic[w, p] +
                                    demand_coefficients[(w, p)]['Cross_Price_Coefficient'] * price_dynamic[w, other_p]
        )

# Revenue Linearization Using Big-M
for w in weeks:
    for p in products:
        price_lb, price_ub = 20, 500
        demand_lb, demand_ub = 0, 10000

        dynamic_model.addConstr(aux_dynamic[w, p] >= price_lb * demand_dynamic[w, p] + price_dynamic[w, p] * demand_lb - price_lb * demand_lb)
        dynamic_model.addConstr(aux_dynamic[w, p] >= price_ub * demand_dynamic[w, p] + price_dynamic[w, p] * demand_ub - price_ub * demand_ub)
        dynamic_model.addConstr(aux_dynamic[w, p] <= price_ub * demand_dynamic[w, p] + price_dynamic[w, p] * demand_lb - price_ub * demand_lb)
        dynamic_model.addConstr(aux_dynamic[w, p] <= price_lb * demand_dynamic[w, p] + price_dynamic[w, p] * demand_ub - price_lb * demand_ub)

        dynamic_model.addConstr(revenue_dynamic[w, p] == aux_dynamic[w, p])

# Objective Function: Maximize Total Revenue
dynamic_model.setObjective(gp.quicksum(revenue_dynamic[w, p] for w in weeks for p in products), GRB.MAXIMIZE)

# Solve Model
dynamic_model.optimize()

# Store revenue from the optimal solution
revenue_dynamic_value = dynamic_model.objVal if dynamic_model.status == GRB.OPTIMAL else None

# ============================
# SCENARIO 2: Fully Static Pricing (Same Price for All Weeks)
# ============================
static_model = gp.Model("Fully_Static_Pricing")

# Decision Variables
price_static = static_model.addVars(products, vtype=GRB.CONTINUOUS, lb=20, ub=500, name="Price_Static")
demand_static = static_model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=0, name="Demand_Static")
revenue_static = static_model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=0, name="Revenue_Static")
aux_static = static_model.addVars(weeks, products, vtype=GRB.CONTINUOUS, lb=0, name="Aux_Static")

# Demand Constraints (Fully Linear)
for w in weeks:
    for p in products:
        other_p = "TechFit Smartwatch" if p == "PowerSound Earbuds" else "PowerSound Earbuds"
        static_model.addConstr(
            demand_static[w, p] == demand_coefficients[(w, p)]['Intercept'] +
                                    demand_coefficients[(w, p)]['Own_Price_Coefficient'] * price_static[p] +
                                    demand_coefficients[(w, p)]['Cross_Price_Coefficient'] * price_static[other_p]
        )

# Revenue Linearization Using Big-M
for w in weeks:
    for p in products:
        price_lb, price_ub = 20, 500
        demand_lb, demand_ub = 0, 10000

        static_model.addConstr(aux_static[w, p] >= price_lb * demand_static[w, p] + price_static[p] * demand_lb - price_lb * demand_lb)
        static_model.addConstr(aux_static[w, p] >= price_ub * demand_static[w, p] + price_static[p] * demand_ub - price_ub * demand_ub)
        static_model.addConstr(aux_static[w, p] <= price_ub * demand_static[w, p] + price_static[p] * demand_lb - price_ub * demand_lb)
        static_model.addConstr(aux_static[w, p] <= price_lb * demand_static[w, p] + price_static[p] * demand_ub - price_lb * demand_ub)

        static_model.addConstr(revenue_static[w, p] == aux_static[w, p])

# Objective Function: Maximize Total Revenue
static_model.setObjective(gp.quicksum(revenue_static[w, p] for w in weeks for p in products), GRB.MAXIMIZE)

# Solve Model
static_model.optimize()

# Store revenue from the optimal solution
revenue_static_value = static_model.objVal if static_model.status == GRB.OPTIMAL else None

# ============================
# Benchmarking Revenue Across Pricing Strategies
# ============================
revenue_values = {
    "Fully Dynamic Pricing": revenue_dynamic_value if revenue_dynamic_value else 0,
    "Fully Static Pricing": revenue_static_value if revenue_static_value else 0
}

# Print benchmark results
print("\nBenchmarking Revenue Across Pricing Strategies")
for strategy, rev in revenue_values.items():
    print(f"{strategy}: ${rev:.2f}")


print("\n" + "-"*100 + "\n")


print("\n" + "="*100)
print("  QUESTION 2 - HOTEL STAFFING OPTIMIZATION")
print("="*100 + "\n")

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Load the dataset
hotels_df = pd.read_csv('https://raw.githubusercontent.com/kristxna/Datasets/refs/heads/main/hotels.csv')
hotels_df.columns
hotels_df.head()

print("\n" + "-"*100)
print("Question 2 - Part E")
print("Covers parts a, b, c and e.")
print("-"*100 + "\n")

# Covers Parts (a), (b), (c) and (e)

# Part (e) Build Optimal Staffing and Cost Solution
print("\n=================== Binary Optimization Model ===================")

# Extract relevant data
floors = hotels_df['Floor'].unique()
rooms = hotels_df['Room_ID'].unique()
num_attendants = 8
max_sqft_per_day = 3500
base_wage = 25
extra_floor_cost = 75
shift_hours = 8
ot_wage = base_wage * 1.5  # Overtime wage (1.5x base)
floor_limit_wage = base_wage * 2  # Wage doubles if over 3500 sqft

# Function to structure room data by floor
def get_room_data():
    room_data = {}
    for _, row in hotels_df.iterrows():
        floor, room, sqft, time = row['Floor'], row['Room_ID'], row['Square_Feet'], row['Cleaning_Time_Hours']
        if floor not in room_data:
            room_data[floor] = []
        room_data[floor].append((room, sqft, time))
    return room_data

room_data = get_room_data()

# Initialize model 
model = gp.Model("Hotel Cleaning Scheduling")

# Decision Variables
x = model.addVars(num_attendants, floors, rooms, vtype=GRB.BINARY, name="Assign")
f = model.addVars(num_attendants, floors, vtype=GRB.BINARY, name="FloorAssign")
over_sqft = model.addVars(num_attendants, vtype=GRB.BINARY, name="OverSqft")

# Binary Overtime Variables (1 or 2 extra hours)
overtime_1h = model.addVars(num_attendants, vtype=GRB.BINARY, name="Overtime_1h")
overtime_2h = model.addVars(num_attendants, vtype=GRB.BINARY, name="Overtime_2h")

# Binary Floor Bonus and Square Footage Penalty Variables
floor_bonus = model.addVars(num_attendants, vtype=GRB.BINARY, name="FloorBonus")
sqft_penalty = model.addVars(num_attendants, vtype=GRB.BINARY, name="SqftPenalty")

# Objective Function: Minimize cost
model.setObjective(
    gp.quicksum(
        (base_wage * shift_hours) +  # Base wage
        (ot_wage * overtime_1h[i]) +  # 1 hour overtime
        (2 * ot_wage * overtime_2h[i]) +  # 2 hours overtime
        (extra_floor_cost * floor_bonus[i]) +  # Extra floor bonus
        (2 * base_wage * shift_hours * sqft_penalty[i])  # Wage doubles if over 3500 sqft
        for i in range(num_attendants)
    ), GRB.MINIMIZE
)

# Constraints

# Ensure all rooms are assigned to one attendant
for k in floors:
    for j in room_data[k]:
        model.addConstr(gp.quicksum(x[i, k, j[0]] for i in range(num_attendants)) == 1, f"RoomAssignment_{k}_{j[0]}")

# Floor assignment consistency
for i in range(num_attendants):
    for k in floors:
        for j in room_data[k]:
            model.addConstr(x[i, k, j[0]] <= f[i, k], f"FloorConsistency_{i}_{k}_{j[0]}")

# Each attendant cleans between 2 to 4 floors
for i in range(num_attendants):
    model.addConstr(gp.quicksum(f[i, k] for k in floors) >= 2, f"MinFloors_{i}")
    model.addConstr(gp.quicksum(f[i, k] for k in floors) <= 4, f"MaxFloors_{i}")

# Track extra floor bonus (Binary encoding using Big-M method)
M = len(floors)  # A sufficiently large number

for i in range(num_attendants):
    model.addConstr(floor_bonus[i] * M >= gp.quicksum(f[i, k] for k in floors) - 2, f"FloorBonus_{i}")

# Ensure attendants do not exceed 3500 sqft before extra pay applies
for i in range(num_attendants):
    model.addConstr(
        gp.quicksum(x[i, k, j[0]] * j[1] for k in floors for j in room_data[k]) <= max_sqft_per_day + sqft_penalty[i] * 1000,
        f"SqFtLimit_{i}"
    )

# Overtime constraint (Binary encoding)
for i in range(num_attendants):
    work_hours = gp.quicksum(x[i, k, j[0]] * j[2] for k in floors for j in room_data[k])
    model.addConstr(work_hours <= 8 + overtime_1h[i] + 2 * overtime_2h[i], f"WorkHours_{i}")

    # Ensure attendants cannot have both 1h and 2h overtime simultaneously
    model.addConstr(overtime_1h[i] + overtime_2h[i] <= 1, f"OvertimeBinary_{i}")

# Solve the model
model.optimize()

# Display Results
if model.status == GRB.OPTIMAL:
    total_overtime_hours = sum(overtime_1h[i].x + 2 * overtime_2h[i].x for i in range(num_attendants))
    total_floor_bonuses = sum(floor_bonus[i].x for i in range(num_attendants))
    print(f"Optimal Staffing Cost for the Day: ${model.objVal:.2f}")
    print(f"Total Overtime Hours: {total_overtime_hours}")
    print(f"Total Floor Bonuses: {total_floor_bonuses}")

    print("\nAttendant Workload and Wage Breakdown:")
    for i in range(num_attendants):
        assigned_floors = [k for k in floors if f[i, k].x > 0.5]
        num_floors = len(assigned_floors)
        total_sqft_cleaned = sum(x[i, k, j[0]].x * j[1] for k in floors for j in room_data[k])
        overtime_hours = int(overtime_1h[i].x + 2 * overtime_2h[i].x)
        floor_bonus_active = int(floor_bonus[i].x)
        sqft_penalty_active = int(sqft_penalty[i].x)

        # Calculate total hours worked (8 + overtime if applicable)
        total_hours_worked = min(8 + overtime_hours, 10)  # Capped at 10 hours

        extra_sqft_pay = (2 * base_wage * shift_hours) * sqft_penalty_active
        extra_floor_pay = extra_floor_cost * floor_bonus_active
        total_wage = (base_wage * shift_hours) + (ot_wage * overtime_hours) + extra_sqft_pay + extra_floor_pay

        print(f"Attendant {i+1}: Floors [{', '.join(map(str, assigned_floors))}], "
              f"Total Hours Worked: {total_hours_worked}, Overtime: {overtime_hours} hours, "
              f"Total Sqft Cleaned: {total_sqft_cleaned:.2f}, Floor Bonus: {'Yes' if floor_bonus_active else 'No'}, "
              f"Sqft Penalty: {'Yes' if sqft_penalty_active else 'No'}, "
              f"Total Wage: ${total_wage:.2f}")
else:
    print("No optimal solution found.")



# Verify all rooms are assigned
all_rooms_cleaned = True
uncleaned_rooms = []

for k in floors:
    for j in room_data[k]:
        assigned = sum(x[i, k, j[0]].x for i in range(num_attendants))  # Summing assignments
        if assigned != 1:
            all_rooms_cleaned = False
            uncleaned_rooms.append((j[0], k))  # Collect uncleaned rooms

if all_rooms_cleaned:
    print("All rooms were assigned and cleaned.")
else:
    print("Some rooms were not cleaned!")
    print(f"Uncleaned rooms: {uncleaned_rooms}")


print("\n" + "-"*100)
print("Question 2 - Part F")
print("-"*100 + "\n")

# Part (f) Model relaxation 
print("\n=================== Relaxed Model ===================")
relaxed_model = model.relax() 
relaxed_model.optimize()

if relaxed_model.status == GRB.OPTIMAL:
    print(f"(f) Relaxed Model Cost: ${relaxed_model.objVal:.2f}")

    # Workload and Wage Breakdown for Relaxed Model
    print("\nAttendant Workload and Wage Breakdown for Relaxed Model:")
    for i in range(num_attendants):
        overtime_hours = relaxed_model.getVarByName(f"Overtime[{i}]").x if relaxed_model.getVarByName(f"Overtime[{i}]") else 0
        floor_violations = relaxed_model.getVarByName(f"FloorViolation[{i}]").x if relaxed_model.getVarByName(f"FloorViolation[{i}]") else 0
        total_sqft_cleaned = sum(x[i, k, j[0]].x * j[1] for k in floors for j in room_data[k])

        # Ensure no negative zero values appear
        overtime_hours = 0 if abs(overtime_hours) < 1e-5 else round(overtime_hours, 2)
        floor_violations = 0 if abs(floor_violations) < 1e-5 else round(floor_violations, 2)
        total_sqft_cleaned = 0 if abs(total_sqft_cleaned) < 1e-5 else round(total_sqft_cleaned, 2)

        # Calculate total hours worked (8 + overtime if applicable)
        total_hours_worked = min(8 + overtime_hours, 10)  # Capped at 10 hours

        # Wage calculations
        base_pay = base_wage * 8
        overtime_pay = 1.5 * base_wage * overtime_hours
        extra_sqft_pay = 2 * base_wage * (total_sqft_cleaned > max_sqft_per_day)
        extra_floor_pay = extra_floor_cost * floor_violations
        total_wage = base_pay + overtime_pay + extra_sqft_pay + extra_floor_pay

        total_wage = 0 if abs(total_wage) < 1e-5 else round(total_wage, 2)  # Ensure no negative zeros

        assigned_floors = [k for k in floors if f[i, k].x > 0.5]

        print(f"Attendant {i+1}: Floors [{', '.join(map(str, assigned_floors))}], "
              f"Total Hours Worked: {total_hours_worked}, Overtime: {overtime_hours:.2f} hour(s), "
              f"Total Sqft Cleaned: {total_sqft_cleaned:.2f}, Floor Violations: {floor_violations}, "
              f"Total Wage: ${total_wage:.2f}")
else:
    print("No optimal solution found for the relaxed model.")


print("\n" + "-"*100)
print("Question 2 - Part G")
print("-"*100 + "\n")

# Part (g) Convert binary to continuous and solve manually relaxed problem
print("\n=================== Manually Relaxed Model ===================")

# Copy the original model for manual relaxation
manual_relax_model = model.copy()

# Convert all binary variables to continuous
for v in manual_relax_model.getVars():
    if v.vType in [GRB.BINARY]:  
        v.vtype = GRB.CONTINUOUS  

# Remove constraints that force integer-like behavior
for constr in manual_relax_model.getConstrs():
    if any(keyword in constr.constrName for keyword in ["FloorViolation", "WorkHours", "SqFtLimit"]):
        manual_relax_model.remove(constr)

# Adjust the objective function to encourage continuous solutions
manual_relax_model.setObjective(
    gp.quicksum(
        base_wage * 8 +
        1.5 * base_wage * (manual_relax_model.getVarByName(f"Overtime_1h[{i}]") or 0) +  
        2 * 1.5 * base_wage * (manual_relax_model.getVarByName(f"Overtime_2h[{i}]") or 0) +
        extra_floor_cost * (manual_relax_model.getVarByName(f"FloorBonus[{i}]") or 0) +
        2 * base_wage * shift_hours * (manual_relax_model.getVarByName(f"SqftPenalty[{i}]") or 0)
        for i in range(num_attendants)
    ),
    GRB.MINIMIZE
)

# Solve the manually relaxed model
manual_relax_model.optimize()

if manual_relax_model.status == GRB.OPTIMAL:
    print(f"(g) Manually Relaxed Solution Cost: ${manual_relax_model.objVal:.2f}")

    # Workload and Wage Breakdown for Manually Relaxed Model
    print("\nAttendant Workload and Wage Breakdown for Manually Relaxed Model:")
    for i in range(num_attendants):
        overtime_var = manual_relax_model.getVarByName(f"Overtime[{i}]")
        floor_violation_var = manual_relax_model.getVarByName(f"FloorViolation[{i}]")

        overtime_hours = overtime_var.x if overtime_var else 0
        floor_violations = floor_violation_var.x if floor_violation_var else 0
        total_sqft_cleaned = sum(x[i, k, j[0]].x * j[1] for k in floors for j in room_data[k])

        # Ensure no negative zero values appear
        overtime_hours = 0 if abs(overtime_hours) < 1e-5 else round(overtime_hours, 2)
        floor_violations = 0 if abs(floor_violations) < 1e-5 else round(floor_violations, 2)
        total_sqft_cleaned = 0 if abs(total_sqft_cleaned) < 1e-5 else round(total_sqft_cleaned, 2)

        # Calculate total hours worked (8 + overtime if applicable)
        total_hours_worked = min(8 + overtime_hours, 10)  # Capped at 10 hours

        # Wage calculations
        base_pay = base_wage * 8
        overtime_pay = 1.5 * base_wage * overtime_hours
        extra_sqft_pay = 2 * base_wage * (total_sqft_cleaned > max_sqft_per_day)
        extra_floor_pay = extra_floor_cost * floor_violations
        total_wage = base_pay + overtime_pay + extra_sqft_pay + extra_floor_pay

        total_wage = 0 if abs(total_wage) < 1e-5 else round(total_wage, 2)  # Ensure no negative zeros

        assigned_floors = [k for k in floors if f[i, k].x > 0.5]

        print(f"Attendant {i+1}: Floors [{', '.join(map(str, assigned_floors))}], "
              f"Total Hours Worked: {total_hours_worked}, Overtime: {overtime_hours:.2f} hours, "
              f"Total Sqft Cleaned: {total_sqft_cleaned:.2f}, Floor Violations: {floor_violations}, "
              f"Total Wage: ${total_wage:.2f}")
else:
    print("No optimal solution found for the manually relaxed model.")



print("-"*100 + "\n")

print("\n=================== Cost Comparisons ===================")
print(f"Binary Model Cost: ${model.objVal:.2f}")
print(f"Relaxed Model Cost: ${relaxed_model.objVal:.2f}")
print(f"Manually Relaxed Model Cost: ${manual_relax_model.objVal:.2f}")


print("\n" + "-"*100)
print("Question 2 - Part I")
print("-"*100 + "\n")


# Part (i) Evaluating the Impact of 2× Overtime Pay on Costs and Violations
print("\n=================== Binary Optimization Model with 2x Overtime Pay ===================")

overtime_multiplier = 2.0
model_i = model.copy()  # Copy the original model

# Set solver parameters to speed up the process
model_i.setParam(GRB.Param.TimeLimit, 60)  # Set a 60-second limit
model_i.setParam(GRB.Param.MIPFocus, 1)  # Prioritize feasibility
model_i.setParam(GRB.Param.Symmetry, 2)  # Reduce unnecessary computations
model_i.setParam(GRB.Param.Heuristics, 0.2)  # Use heuristics for faster solutions

# Use warm start to accelerate solution
for v in model.getVars():
    model_i.getVarByName(v.VarName).Start = v.X

# Helper function to prevent negative zero
def safe_round(value):
    """Ensure no negative zero values appear by rounding small values to exactly zero."""
    return 0.00 if abs(value) < 1e-5 else round(value, 2)

# Update Objective Function
model_i.setObjective(
    gp.quicksum(
        base_wage * 8 * (1 + (model_i.getVarByName(f"SqftPenalty[{i}]") or 0)) +
        base_wage * overtime_multiplier * ((model_i.getVarByName(f"Overtime_1h[{i}]") or 0) +
                                           2 * (model_i.getVarByName(f"Overtime_2h[{i}]") or 0))
        for i in range(num_attendants)
    ) +
    gp.quicksum(extra_floor_cost * (model_i.getVarByName(f"FloorBonus[{i}]") or 0) for i in range(num_attendants)), 
    GRB.MINIMIZE
)

model_i.optimize()

# Display Results
if model_i.status == GRB.OPTIMAL or model_i.status == GRB.TIME_LIMIT:
    print(f"(i) Updated Optimal Cost with 2x Overtime: ${model_i.objVal:.2f}")

    # Workload and Wage Breakdown for 2x Overtime Model
    print("\n Attendant Workload and Wage Breakdown for Model with 2x Overtime Pay:")
    for i in range(num_attendants):
        assigned_floors = [k for k in floors if f[i, k].x > 0.5]
        num_floors = len(assigned_floors)

        # Floor Violation Calculation
        floor_violations = max(0, num_floors - 2) if num_floors > 2 else 0

        # Overtime Calculation
        overtime_var_1h = model_i.getVarByName(f"Overtime_1h[{i}]")
        overtime_var_2h = model_i.getVarByName(f"Overtime_2h[{i}]")
        overtime_hours = safe_round(
            (overtime_var_1h.x if overtime_var_1h else 0) + 
            (2 * (overtime_var_2h.x if overtime_var_2h else 0))
        )

        # Square Footage Calculation
        total_sqft_cleaned = sum(x[i, k, j[0]].x * j[1] for k in floors for j in room_data[k])
        total_sqft_cleaned = safe_round(total_sqft_cleaned)

        # Wage calculations
        base_pay = base_wage * 8
        overtime_pay = overtime_multiplier * base_wage * overtime_hours
        extra_sqft_pay = (2 * base_wage * 8) if total_sqft_cleaned > max_sqft_per_day else 0  
        extra_floor_pay = extra_floor_cost * floor_violations
        total_wage = safe_round(base_pay + overtime_pay + extra_sqft_pay + extra_floor_pay)

        print(f"Attendant {i+1}: Floors [{', '.join(map(str, assigned_floors))}], "
              f"Total Hours Worked: {8 + overtime_hours:.2f}, Overtime: {overtime_hours:.2f} hours, "
              f"Total Sqft Cleaned: {total_sqft_cleaned:.2f}, Floor Violations: {int(floor_violations)}, "
              f"Total Wage: ${total_wage:.2f}")
else:
    print("No optimal solution found for the updated model with 2x overtime pay.")



print("-"*100 + "\n")

print("\n=================== Binary Model Cost Comparison: Before vs After 2x Overtime ===================")
print(f"{'Metric':<25} {'Before 2x Overtime':<25} {'After 2x Overtime'}")
print("-" * 75)

# Optimal Cost Comparison
print(f"{'Optimal Cost':<25} ${model.objVal:,.2f} {' ' * 10} ${model_i.objVal:,.2f}")

# Overtime Hours Calculation Before & After
total_overtime = sum(
    (model.getVarByName(f"Overtime_1h[{i}]").x if model.getVarByName(f"Overtime_1h[{i}]") else 0) +
    (2 * model.getVarByName(f"Overtime_2h[{i}]").x if model.getVarByName(f"Overtime_2h[{i}]") else 0)
    for i in range(num_attendants)
)

total_overtime_after = sum(
    (model_i.getVarByName(f"Overtime_1h[{i}]").x if model_i.getVarByName(f"Overtime_1h[{i}]") else 0) +
    (2 * model_i.getVarByName(f"Overtime_2h[{i}]").x if model_i.getVarByName(f"Overtime_2h[{i}]") else 0)
    for i in range(num_attendants)
)

print(f"{'Total Overtime Hours':<25} {total_overtime:.2f} {' ' * 20} {total_overtime_after:.2f}")

# Floor Violation Calculation
total_floor_violations = sum(
    max(0, len([k for k in floors if f[i, k].x > 0.5]) - 2) for i in range(num_attendants)
)

total_floor_violations_after = sum(
    max(0, len([k for k in floors if f[i, k].x > 0.5]) - 2) for i in range(num_attendants)
)

# Floor Violations Comparison
print(f"{'Total Floor Violations':<25} {int(total_floor_violations)} {' ' * 23} {int(total_floor_violations_after)}")
