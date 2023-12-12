# models/packaging.py
import numpy as np
import gurobipy as gp
from config.settings import GlulamConfig
import pandas as pd


def optimize_packaging(pat):
    # Initialize the model
    cutmodel = gp.Model("Cutting final model")

    # Decision variable, how often should we cut each pattern
    x = cutmodel.addVars(pat.J, lb=0, vtype=gp.GRB.CONTINUOUS, name="x")  # try also vtype = GRB.INTEGER

    # Objective: Minimize the total number of patterns used
    cutmodel.setObjective(gp.quicksum(x[j] for j in pat.J), gp.GRB.MINIMIZE)

    # Constraint: Ensure all orders are satisfied
    cutmodel.addConstrs(gp.quicksum(pat.A[i, j] * x[j] for j in pat.J) >= pat.data.orders[i] for i in pat.I)

    # Solve the model
    cutmodel.optimize()

    # Extract the integer solution via flooring
    xfloor = np.floor([x[j].X for j in pat.J])

    # TODO clean this up
    print("Patterns used:")
    results = []
    press_id = 0
    press_height = 0
    for j in pat.J:
        if xfloor[j] > 0:
            for _ in range(int(xfloor[j])):
                tot_length = np.sum(pat.A[:, j] * pat.data.widths)
                height = pat.data.heights[list(np.where(np.floor(pat.A[:, j]) > 0))[0][0]] / GlulamConfig.COUNT_HEIGHT
                if press_height + height > GlulamConfig.MAX_HEIGHT:
                    press_height = 0
                    press_id = press_id + 1
                press_height = press_height + height
                results.append(np.hstack((j, height, tot_length, press_id, press_height, pat.A[:, j].T)).tolist())

    df = pd.DataFrame(np.array(results).astype('int'))
    missingorder = (pat.data.orders - pat.A @ xfloor.transpose()).astype('int')

    # print("missingorders = ", missingorder)
    for i in range(len(missingorder)):
        if missingorder[i] > 0:
            print(missingorder[i], "/", pat.data.orders[i], "missing from order", i)
    print("each roll has length = ", pat.roll_width, " missing material in rollwidth:",
          np.ceil(missingorder @ pat.data.widths.transpose() / pat.roll_width))
    print("the total number of rolls used is at least ",
          int(np.sum(xfloor)) + np.ceil(missingorder @ pat.data.widths.transpose() / pat.roll_width))
    pontun_nafn = pat.data._filtered_data.index
    df.columns = ['MynsturID'] + ['haed', 'lengd', 'lota', 'pressuhaed'] + pontun_nafn.tolist()
    df.to_excel('lausn2.xlsx', index=False)
    df

