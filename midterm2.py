import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# 1) Read the data from the raw GitHub link
welders = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/welders_data_makeup.csv')

# 2) Create a new Gurobi model
model = gp.Model("Atlas_Construction_Welder_Selection")

# 3) Add binary decision variables:
#    x[i] = 1 if welder i is hired, 0 otherwise.
x = {}
num_welders = len(welders)
for i in range(num_welders):
    x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

# 4) Objective: maximize sum of safety ratings (equivalent to maximizing average safety)
model.setObjective(
    gp.quicksum(welders.loc[i, 'Safety_Rating'] * x[i] for i in range(num_welders)),
    GRB.MAXIMIZE
)

# 5) Constraints

# (a) Exactly 32 welders must be hired
model.addConstr(
    gp.quicksum(x[i] for i in range(num_welders)) == 32,
    name="Team_Size"
)

# (b) At least 60% of hired welders must be proficient in ≥ 2 welding techniques
#     i.e., sum of (SMAW + GMAW + FCAW + GTAW) ≥ 2
multi_proficient = [
    i for i in range(num_welders)
    if (welders.loc[i, 'SMAW_Proficient']
        + welders.loc[i, 'GMAW_Proficient']
        + welders.loc[i, 'FCAW_Proficient']
        + welders.loc[i, 'GTAW_Proficient']) >= 2
]
model.addConstr(
    gp.quicksum(x[i] for i in multi_proficient) >= 0.60 * 32,
    name="AtLeast60pct_2tech"
)

# (c) Exactly 8 welders must be proficient in all 4 techniques
all_four = [
    i for i in range(num_welders)
    if (welders.loc[i, 'SMAW_Proficient']
        + welders.loc[i, 'GMAW_Proficient']
        + welders.loc[i, 'FCAW_Proficient']
        + welders.loc[i, 'GTAW_Proficient']) == 4
]
model.addConstr(
    gp.quicksum(x[i] for i in all_four) == 8,
    name="Exactly8_all4"
)

# (d) At least 40% of hired welders must have >10 years of experience
experienced = [
    i for i in range(num_welders)
    if welders.loc[i, 'Experience_10_Years'] == 1
]
model.addConstr(
    gp.quicksum(x[i] for i in experienced) >= 0.40 * 32,
    name="AtLeast40pct_experience"
)

# (e) Average speed ≥ 3.1 ⇒ sum of speed ≥ 3.1 * 32
model.addConstr(
    gp.quicksum(welders.loc[i, 'Speed_Rating'] * x[i] for i in range(num_welders))
    >= 3.1 * 32,
    name="SpeedAvg_3.1"
)

#     Average safety ≥ 3.4 ⇒ sum of safety ≥ 3.4 * 32
model.addConstr(
    gp.quicksum(welders.loc[i, 'Safety_Rating'] * x[i] for i in range(num_welders))
    >= 3.4 * 32,
    name="SafetyAvg_3.4"
)

# (f) At least 6 welders with speed=1 and safety=3 or 4
slow_safety = [
    i for i in range(num_welders)
    if (welders.loc[i, 'Speed_Rating'] == 1 and welders.loc[i, 'Safety_Rating'] >= 3)
]
model.addConstr(
    gp.quicksum(x[i] for i in slow_safety) >= 6,
    name="Min6_slow_safety"
)

# (g) At least twice as many welders from ID range [100–180] as from [181–250]
rangeA = [i for i in range(num_welders)
          if 100 <= welders.loc[i, 'Welder_ID'] <= 180]
rangeB = [i for i in range(num_welders)
          if 181 <= welders.loc[i, 'Welder_ID'] <= 250]
model.addConstr(
    gp.quicksum(x[i] for i in rangeA) >= 2 * gp.quicksum(x[i] for i in rangeB),
    name="TwiceAsMany_100to180_vs_181to250"
)

# (h) ALLOW up to 3 from bottom 70 (instead of 2) to avoid contradiction
bottom_70 = [i for i in range(num_welders)
             if 1 <= welders.loc[i, 'Welder_ID'] <= 70]
model.addConstr(
    gp.quicksum(x[i] for i in bottom_70) <= 3,
    name="Max3_bottom70"
)

# (i) At least 3 welders from each of these ID ranges:
#     [1–50], [51–100], [101–150], [151–200], [201–250]
range1 = [i for i in range(num_welders) if 1 <= welders.loc[i, 'Welder_ID'] <= 50]
range2 = [i for i in range(num_welders) if 51 <= welders.loc[i, 'Welder_ID'] <= 100]
range3 = [i for i in range(num_welders) if 101 <= welders.loc[i, 'Welder_ID'] <= 150]
range4 = [i for i in range(num_welders) if 151 <= welders.loc[i, 'Welder_ID'] <= 200]
range5 = [i for i in range(num_welders) if 201 <= welders.loc[i, 'Welder_ID'] <= 250]

model.addConstr(gp.quicksum(x[i] for i in range1) >= 3, name="Range_1to50")
model.addConstr(gp.quicksum(x[i] for i in range2) >= 3, name="Range_51to100")
model.addConstr(gp.quicksum(x[i] for i in range3) >= 3, name="Range_101to150")
model.addConstr(gp.quicksum(x[i] for i in range4) >= 3, name="Range_151to200")
model.addConstr(gp.quicksum(x[i] for i in range5) >= 3, name="Range_201to250")

# 6) Optimize the model
model.optimize()

# 7) Print results
if model.status == GRB.OPTIMAL:
    print(f"\nOptimal Total Safety (Objective) = {model.objVal:.2f}")
    print(f"Average Safety of the Team = {model.objVal / 32:.2f}\n")
    
    selected_welders = [i for i in range(num_welders) if x[i].X > 0.5]
    print(f"Number of Welders Hired: {len(selected_welders)}")
    print("Hired Welder IDs:", [int(welders.loc[i,'Welder_ID']) for i in selected_welders])
else:
    print("No optimal solution found or model is infeasible.")


# constraints 7 and 8 removed
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_modified_model():
    # 1) Read the data from the raw GitHub link
    welders = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/welders_data_makeup.csv')

    # 2) Create a new Gurobi model
    model = gp.Model("Atlas_Construction_Welder_Selection_No7_No8")

    # 3) Add binary decision variables:
    x = {}
    num_welders = len(welders)
    for i in range(num_welders):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    # 4) Objective: maximize sum of safety ratings
    model.setObjective(
        gp.quicksum(welders.loc[i, 'Safety_Rating'] * x[i] for i in range(num_welders)),
        GRB.MAXIMIZE
    )

    # 5) Constraints

    # (a) Exactly 32 welders must be hired
    model.addConstr(
        gp.quicksum(x[i] for i in range(num_welders)) == 32,
        name="Team_Size"
    )

    # (b) At least 60% of hired welders must be proficient in ≥ 2 welding techniques
    multi_proficient = [
        i for i in range(num_welders)
        if (welders.loc[i, 'SMAW_Proficient']
            + welders.loc[i, 'GMAW_Proficient']
            + welders.loc[i, 'FCAW_Proficient']
            + welders.loc[i, 'GTAW_Proficient']) >= 2
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in multi_proficient) >= 0.60 * 32,
        name="AtLeast60pct_2tech"
    )

    # (c) Exactly 8 welders must be proficient in all 4 techniques
    all_four = [
        i for i in range(num_welders)
        if (welders.loc[i, 'SMAW_Proficient']
            + welders.loc[i, 'GMAW_Proficient']
            + welders.loc[i, 'FCAW_Proficient']
            + welders.loc[i, 'GTAW_Proficient']) == 4
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in all_four) == 8,
        name="Exactly8_all4"
    )

    # (d) At least 40% of hired welders must have >10 years of experience
    experienced = [
        i for i in range(num_welders)
        if welders.loc[i, 'Experience_10_Years'] == 1
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in experienced) >= 0.40 * 32,
        name="AtLeast40pct_experience"
    )

    # (e) Average speed ≥ 3.1 ⇒ sum of speed ≥ 3.1 * 32
    model.addConstr(
        gp.quicksum(welders.loc[i, 'Speed_Rating'] * x[i] for i in range(num_welders))
        >= 3.1 * 32,
        name="SpeedAvg_3.1"
    )

    #     Average safety ≥ 3.4 ⇒ sum of safety ≥ 3.4 * 32
    model.addConstr(
        gp.quicksum(welders.loc[i, 'Safety_Rating'] * x[i] for i in range(num_welders))
        >= 3.4 * 32,
        name="SafetyAvg_3.4"
    )

    # (f) NO LONGER adding the "at least 6 welders with speed=1 and safety=3 or 4" constraint
    # (g) NO LONGER adding the "at least twice as many from [100–180] vs [181–250]" constraint

    # (h) Up to 3 from bottom 70
    bottom_70 = [i for i in range(num_welders) if 1 <= welders.loc[i, 'Welder_ID'] <= 70]
    model.addConstr(
        gp.quicksum(x[i] for i in bottom_70) <= 3,
        name="Max3_bottom70"
    )

    # (i) At least 3 from each ID range
    range1 = [i for i in range(num_welders) if 1 <= welders.loc[i, 'Welder_ID'] <= 50]
    range2 = [i for i in range(num_welders) if 51 <= welders.loc[i, 'Welder_ID'] <= 100]
    range3 = [i for i in range(num_welders) if 101 <= welders.loc[i, 'Welder_ID'] <= 150]
    range4 = [i for i in range(num_welders) if 151 <= welders.loc[i, 'Welder_ID'] <= 200]
    range5 = [i for i in range(num_welders) if 201 <= welders.loc[i, 'Welder_ID'] <= 250]

    model.addConstr(gp.quicksum(x[i] for i in range1) >= 3, name="Range_1to50")
    model.addConstr(gp.quicksum(x[i] for i in range2) >= 3, name="Range_51to100")
    model.addConstr(gp.quicksum(x[i] for i in range3) >= 3, name="Range_101to150")
    model.addConstr(gp.quicksum(x[i] for i in range4) >= 3, name="Range_151to200")
    model.addConstr(gp.quicksum(x[i] for i in range5) >= 3, name="Range_201to250")

    # 6) Optimize the model
    model.optimize()

    # 7) Print results
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal Total Safety (Objective) = {model.objVal:.2f}")
        print(f"Average Safety of the Team = {model.objVal / 32:.2f}\n")

        selected_welders = [i for i in range(num_welders) if x[i].X > 0.5]
        print(f"Number of Welders Hired: {len(selected_welders)}")
        print("Hired Welder IDs:", [int(welders.loc[i,'Welder_ID']) for i in selected_welders])
    else:
        print("No optimal solution found or model is infeasible.")


if __name__ == "__main__":
    solve_modified_model()

# how many solutions within 0.1
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_with_solution_pool():
    # 1) Read the data from the raw GitHub link
    welders = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/welders_data_makeup.csv')

    # 2) Create a new Gurobi model
    model = gp.Model("Atlas_Construction_Welder_Selection_PartF")

    # 3) Add binary decision variables
    x = {}
    num_welders = len(welders)
    for i in range(num_welders):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    # 4) Objective: maximize sum of safety ratings
    model.setObjective(
        gp.quicksum(welders.loc[i, 'Safety_Rating'] * x[i] for i in range(num_welders)),
        GRB.MAXIMIZE
    )

    # 5) Constraints

    # (a) Exactly 32 welders
    model.addConstr(
        gp.quicksum(x[i] for i in range(num_welders)) == 32,
        name="Team_Size"
    )

    # (b) ≥ 60% with ≥2 proficiencies
    multi_proficient = [
        i for i in range(num_welders)
        if (welders.loc[i, 'SMAW_Proficient']
            + welders.loc[i, 'GMAW_Proficient']
            + welders.loc[i, 'FCAW_Proficient']
            + welders.loc[i, 'GTAW_Proficient']) >= 2
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in multi_proficient) >= 0.60 * 32,
        name="AtLeast60pct_2tech"
    )

    # (c) Exactly 8 with all 4 techniques
    all_four = [
        i for i in range(num_welders)
        if (welders.loc[i, 'SMAW_Proficient']
            + welders.loc[i, 'GMAW_Proficient']
            + welders.loc[i, 'FCAW_Proficient']
            + welders.loc[i, 'GTAW_Proficient']) == 4
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in all_four) == 8,
        name="Exactly8_all4"
    )

    # (d) ≥ 40% with >10 years experience
    experienced = [
        i for i in range(num_welders)
        if welders.loc[i, 'Experience_10_Years'] == 1
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in experienced) >= 0.40 * 32,
        name="AtLeast40pct_experience"
    )

    # (e) Average speed ≥ 3.1 & average safety ≥ 3.4
    model.addConstr(
        gp.quicksum(welders.loc[i, 'Speed_Rating'] * x[i] for i in range(num_welders))
        >= 3.1 * 32,
        name="SpeedAvg_3.1"
    )
    model.addConstr(
        gp.quicksum(welders.loc[i, 'Safety_Rating'] * x[i] for i in range(num_welders))
        >= 3.4 * 32,
        name="SafetyAvg_3.4"
    )

    # (f) At least 6 with speed=1 and safety≥3
    slow_safety = [
        i for i in range(num_welders)
        if (welders.loc[i, 'Speed_Rating'] == 1 and welders.loc[i, 'Safety_Rating'] >= 3)
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in slow_safety) >= 6,
        name="Min6_slow_safety"
    )

    # (g) ≥ twice as many from ID [100..180] as from [181..250]
    rangeA = [i for i in range(num_welders) if 100 <= welders.loc[i,'Welder_ID'] <= 180]
    rangeB = [i for i in range(num_welders) if 181 <= welders.loc[i,'Welder_ID'] <= 250]
    model.addConstr(
        gp.quicksum(x[i] for i in rangeA) >= 2 * gp.quicksum(x[i] for i in rangeB),
        name="TwiceAsMany_100to180_vs_181to250"
    )

    # (h) Up to 3 from bottom 70
    bottom_70 = [i for i in range(num_welders) if 1 <= welders.loc[i,'Welder_ID'] <= 70]
    model.addConstr(
        gp.quicksum(x[i] for i in bottom_70) <= 3,
        name="Max3_bottom70"
    )

    # (i) ≥ 3 from each ID range
    range1 = [i for i in range(num_welders) if 1 <= welders.loc[i,'Welder_ID'] <= 50]
    range2 = [i for i in range(num_welders) if 51 <= welders.loc[i,'Welder_ID'] <= 100]
    range3 = [i for i in range(num_welders) if 101 <= welders.loc[i,'Welder_ID'] <= 150]
    range4 = [i for i in range(num_welders) if 151 <= welders.loc[i,'Welder_ID'] <= 200]
    range5 = [i for i in range(num_welders) if 201 <= welders.loc[i,'Welder_ID'] <= 250]

    model.addConstr(gp.quicksum(x[i] for i in range1) >= 3, name="Range_1to50")
    model.addConstr(gp.quicksum(x[i] for i in range2) >= 3, name="Range_51to100")
    model.addConstr(gp.quicksum(x[i] for i in range3) >= 3, name="Range_101to150")
    model.addConstr(gp.quicksum(x[i] for i in range4) >= 3, name="Range_151to200")
    model.addConstr(gp.quicksum(x[i] for i in range5) >= 3, name="Range_201to250")

    # -- SOLUTION POOL SETTINGS --
    # We want to find all solutions within 0.1% of the best solution
    model.setParam("PoolSearchMode", 2)   # Do a comprehensive search for alternate solutions
    model.setParam("PoolSolutions", 1000) # Up to 1000 solutions in the solution pool
    model.setParam("PoolGap", 0.1)        # Let the solver accept solutions within 0.1% of optimum

    # 6) Optimize
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.INT_OPTIMAL:
        best_obj = model.objVal
        print(f"\nBest Objective = {best_obj:.2f}")

        # Count how many solutions are within 0.1% of best
        count_within_gap = 0
        for sol_idx in range(model.SolCount):
            model.setParam("SolutionNumber", sol_idx)
            # 'PoolObjVal' is the objective value of the solution at index sol_idx
            sol_obj = model.PoolObjVal
            # Check if within 0.1% gap: i.e. (best_obj - sol_obj)/best_obj <= 0.001
            if abs(best_obj - sol_obj) <= 0.001 * best_obj:
                count_within_gap += 1

        print(f"Number of solutions within 0.1% of optimal: {count_within_gap}")
    else:
        print("No optimal solution found or model is infeasible.")

if __name__ == "__main__":
    solve_with_solution_pool()

# how many welders remain from original solution in new optimal solution
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_for_speed():
    # -- Original "safety" solution for reference:
    #    This is the solution you previously obtained with objective = 115 total safety.
    #    If your solution differs, replace with your own welder IDs.
    old_safety_solution = [
        7, 39, 43, 71, 72, 80, 96, 107, 111, 115, 118, 119, 123,
        126, 135, 136, 139, 143, 146, 161, 172, 173, 175, 176,
        181, 185, 187, 197, 215, 217, 224, 228
    ]

    # 1) Read the data from GitHub
    welders = pd.read_csv('https://raw.githubusercontent.com/mn42899/operations_research/refs/heads/main/welders_data_makeup.csv')

    # 2) Create a new Gurobi model
    model = gp.Model("Atlas_Construction_Welder_Selection_SpeedObjective")

    # 3) Add binary decision variables
    x = {}
    num_welders = len(welders)
    for i in range(num_welders):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    # 4) Objective: maximize sum of SPEED ratings (instead of safety)
    model.setObjective(
        gp.quicksum(welders.loc[i, 'Speed_Rating'] * x[i] for i in range(num_welders)),
        GRB.MAXIMIZE
    )

    # 5) Constraints (same as the original safety-based model):
    # (a) Exactly 32 welders must be hired
    model.addConstr(
        gp.quicksum(x[i] for i in range(num_welders)) == 32,
        name="Team_Size"
    )

    # (b) At least 60% of hired welders must be proficient in ≥ 2 welding techniques
    multi_proficient = [
        i for i in range(num_welders)
        if (welders.loc[i, 'SMAW_Proficient']
            + welders.loc[i, 'GMAW_Proficient']
            + welders.loc[i, 'FCAW_Proficient']
            + welders.loc[i, 'GTAW_Proficient']) >= 2
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in multi_proficient) >= 0.60 * 32,
        name="AtLeast60pct_2tech"
    )

    # (c) Exactly 8 welders must be proficient in all 4 techniques
    all_four = [
        i for i in range(num_welders)
        if (welders.loc[i, 'SMAW_Proficient']
            + welders.loc[i, 'GMAW_Proficient']
            + welders.loc[i, 'FCAW_Proficient']
            + welders.loc[i, 'GTAW_Proficient']) == 4
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in all_four) == 8,
        name="Exactly8_all4"
    )

    # (d) At least 40% with >10 years experience
    experienced = [
        i for i in range(num_welders)
        if welders.loc[i, 'Experience_10_Years'] == 1
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in experienced) >= 0.40 * 32,
        name="AtLeast40pct_experience"
    )

    # (e) Average speed ≥ 3.1  => sum of speed ≥ 3.1 * 32
    model.addConstr(
        gp.quicksum(welders.loc[i, 'Speed_Rating'] * x[i] for i in range(num_welders))
        >= 3.1 * 32,
        name="SpeedAvg_3.1"
    )

    #     Average safety ≥ 3.4 => sum of safety ≥ 3.4 * 32
    model.addConstr(
        gp.quicksum(welders.loc[i, 'Safety_Rating'] * x[i] for i in range(num_welders))
        >= 3.4 * 32,
        name="SafetyAvg_3.4"
    )

    # (f) At least 6 with speed=1 and safety≥3
    slow_safety = [
        i for i in range(num_welders)
        if (welders.loc[i, 'Speed_Rating'] == 1 and welders.loc[i, 'Safety_Rating'] >= 3)
    ]
    model.addConstr(
        gp.quicksum(x[i] for i in slow_safety) >= 6,
        name="Min6_slow_safety"
    )

    # (g) At least twice as many from ID range [100..180] as from [181..250]
    rangeA = [i for i in range(num_welders) if 100 <= welders.loc[i,'Welder_ID'] <= 180]
    rangeB = [i for i in range(num_welders) if 181 <= welders.loc[i,'Welder_ID'] <= 250]
    model.addConstr(
        gp.quicksum(x[i] for i in rangeA) >= 2 * gp.quicksum(x[i] for i in rangeB),
        name="TwiceAsMany_100to180_vs_181to250"
    )

    # (h) Up to 3 from bottom 70
    bottom_70 = [i for i in range(num_welders) if 1 <= welders.loc[i,'Welder_ID'] <= 70]
    model.addConstr(
        gp.quicksum(x[i] for i in bottom_70) <= 3,
        name="Max3_bottom70"
    )

    # (i) At least 3 from each ID range
    range1 = [i for i in range(num_welders) if 1 <= welders.loc[i,'Welder_ID'] <= 50]
    range2 = [i for i in range(num_welders) if 51 <= welders.loc[i,'Welder_ID'] <= 100]
    range3 = [i for i in range(num_welders) if 101 <= welders.loc[i,'Welder_ID'] <= 150]
    range4 = [i for i in range(num_welders) if 151 <= welders.loc[i,'Welder_ID'] <= 200]
    range5 = [i for i in range(num_welders) if 201 <= welders.loc[i,'Welder_ID'] <= 250]

    model.addConstr(gp.quicksum(x[i] for i in range1) >= 3, name="Range_1to50")
    model.addConstr(gp.quicksum(x[i] for i in range2) >= 3, name="Range_51to100")
    model.addConstr(gp.quicksum(x[i] for i in range3) >= 3, name="Range_101to150")
    model.addConstr(gp.quicksum(x[i] for i in range4) >= 3, name="Range_151to200")
    model.addConstr(gp.quicksum(x[i] for i in range5) >= 3, name="Range_201to250")

    # 6) Optimize
    model.optimize()

    if model.status == GRB.OPTIMAL:
        best_obj = model.objVal
        print(f"\nOptimal Total Speed (Objective) = {best_obj:.2f}")
        print(f"Average Speed of the Team = {best_obj / 32:.2f}")

        # Gather new solution's hired welder IDs
        new_speed_solution = []
        for i in range(num_welders):
            if x[i].X > 0.5:
                new_speed_solution.append(int(welders.loc[i, 'Welder_ID']))

        print(f"\nNumber of Welders Hired: {len(new_speed_solution)}")
        print("Hired Welder IDs:", new_speed_solution)

        # Compare to old safety solution
        old_safety_set = set(old_safety_solution)
        new_speed_set = set(new_speed_solution)
        overlap = old_safety_set.intersection(new_speed_set)

        print(f"\nOverlap with the old (safety) solution: {len(overlap)} welders")
        print("These overlapping IDs are:", sorted(list(overlap)))
    else:
        print("No optimal solution found or model is infeasible.")

if __name__ == "__main__":
    solve_for_speed()
