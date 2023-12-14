import gurobipy as gp
from gurobipy import GRB
import numpy as np
from config.settings import GlulamConfig


def pack_n_press(A, b, H, L, wr, debug=True):
    # parameters
    bigM = 100000000  # a big number
    nump = len(wr)
    print("number of presses:", np)

    # sets
    I = range(A.shape[0])  # items
    J = range(A.shape[1])  # cutting patterns
    K = range(nump)  # presses
    R = range(2)  # regions

    # debug:
    if debug == True:
        for i in I:
            print("item", i, "width", L[i], "height", H[i], "demand", b[i])

    # model and solve parameters
    pmodel = gp.Model("pack_n_press")  # the packing model
    pmodel.setParam('OutputFlag', GlulamConfig.GUROBI_OUTPUT_FLAG)
    pmodel.setParam('TimeLimit', GlulamConfig.GUROBI_TIME_LIMIT)

    # decision variables
    x = pmodel.addVars(J, K, R, vtype=GRB.INTEGER)  # number of times pattern j is used in press k and region r
    x1 = pmodel.addVars(J, K, R, vtype=GRB.BINARY)  # is pattern j used in press k and region r

    omega = pmodel.addVars(K, R)  # the total waste in the press (\omega)
    delta = pmodel.addVars(I)  # the difference between demand and supply
    h = pmodel.addVars(K, R)  # what is the height of the region
    z = pmodel.addVars(K, vtype=GRB.BINARY)  # is press k used (not sure if this is needed)

    # compute height of each region
    pmodel.addConstrs(gp.quicksum(H[j] * x[j, k, r] for j in J) == h[k, r] for k in K for r in R)
    # the total height of the region must be less than the maximum height of the press
    pmodel.addConstrs(
        gp.quicksum(h[k, r] for r in R) <= GlulamConfig.MAX_HEIGHT_LAYERS * GlulamConfig.LAYER_HEIGHT for k in K)

    # Note, we must make sure that the height of each region is at least the minimum height - ignoring the last press
    # Here we use bigM to make sure that if z[k] = 0 then h[k,r] >= min_height.
    # Min height depends on the region configuration.

    # h[k,0] is the height of the R0 in press k, and we must make sure that it is at least the minimum height
    pmodel.addConstrs(
        h[k, 0] >=
        GlulamConfig.MIN_HEIGHT_LAYER_REGION[0] * GlulamConfig.LAYER_HEIGHT - (1 - z[k]) * bigM for k in K[:-1])

    # If we have two regions (i.e. R0 and R1 - never just R1) then we must make sure that the combined height of R0
    # and R1 is at least the minimum height for R1.
    pmodel.addConstrs(
        gp.quicksum(h[k, r] for r in R) >=
        GlulamConfig.MIN_HEIGHT_LAYER_REGION[1] * GlulamConfig.LAYER_HEIGHT - (1 - z[k]) * bigM for k in K[:-1])

    # we want to make sure that wr[k][r]-L[j] >= 0 if j used in region k,r
    pmodel.addConstrs(x1[j, k, r] * bigM >= x[j, k, r] for j in J for k in K for r in R)
    pmodel.addConstrs(wr[k][r] >= L[j] - bigM * (1 - x1[j, k, r]) for j in J for k in K for r in R)

    # if the press is not used the x must be zero
    pmodel.addConstrs(x[j, k, r] <= bigM * z[k] for j in J for k in K for r in R)

    # now we must fulfill the demand for each item
    pmodel.addConstrs(gp.quicksum(A[i, j] * x[j, k, r] for j in J for k in K for r in R) >= b[i] for i in I)
    pmodel.addConstrs(gp.quicksum(A[i, j] * x[j, k, r] for j in J for k in K for r in R) <= b[i] + delta[i] for i in I)

    # we must compute the waste in each press
    pmodel.addConstrs(
        omega[k, r] == gp.quicksum(H[j] * (wr[k][r] - L[j]) * x[j, k, r] for j in J) for k in K for r in R)

    # now we add the objective function as the sum of waste for all presses and the difference between demand and supply
    pmodel.setObjective(
        1000 * gp.quicksum(delta[i] for i in I) +  # the difference between demand and supply in pieces
        gp.quicksum(omega[k, r] for k in K for r in R) / 1000.0 / 1000.0  # the waste in m^2
        , GRB.MINIMIZE)

    # solve the model
    pmodel.optimize()

    # see if model is infeasible
    if pmodel.status == GRB.INFEASIBLE:
        print("The model is infeasible; quitting.")
        return None, None

    # extract the solution
    Lp_ = np.zeros((len(K), len(R)))  # the length of the press
    Waste_ = np.zeros((len(K), len(R)))
    for k in K:
        for r in R:
            for j in J:
                if x[j, k, r].X > 0.1:
                    Lp_[k, r] = max(Lp_[k, r], L[j])
                    Waste_[k, r] += H[j] * (Lp_[k, r] - L[j]) * x[j, k, r].X / 1000 / 1000  # convert to m^2

    # print the solution
    if debug == True:
        print_press_results(K, R, Lp_, h, omega, Waste_)
        print_item_results(A, b, K, R, I, J, H, L, x, wr, delta)

    # return all omega values
    return Waste_, Lp_


def print_press_results(K, R, Lp_, h, omega, Waste_):
    """ Print the information about the presses. """
    print("\n\nTable: Press Information\n")
    row_format = "{:<5} {:>6} {:>6} {:>6} {:>6}"
    header = ['Press', 'Width', 'Height', 'Waste', 'TotalWaste']
    header = row_format.format(*header)
    seperator_minor = '-' * len(header)
    seperator_major = '=' * len(header)

    print(seperator_major)
    print(header)

    for k in K:
        print(seperator_major if k == 0 else seperator_minor)
        for r in R:
            press_info = [f'{k}.{r}', int(Lp_[k, r]), int(h[k, r].X / GlulamConfig.LAYER_HEIGHT),
                          f"{Waste_[k, r]:.2f}", f"{omega[k, r].X / 1000 / 1000:.2f}"]
            print(row_format.format(*press_info))
    print(seperator_major)


def print_item_results(A, b, K, R, I, J, H, L, x, wr, delta):
    """ Print the information about the items pressed. """
    print("\n\nTable 2: Item Information\n")
    row_format = "{:<5} {:>4} {:>8} {:>3} {:>7} {:>4} {:>3} {:>5}"
    header = ['Press', 'Item', 'Waste', 'Pat', 'Width', 'Used', 'Qty', 'Delta']
    subheader = ['k.r', 'i', 'H(wr-L)x', 'j', 'L', 'xA', 'b', 'delta']
    header = row_format.format(*header)
    seperator_minor = '-' * len(header)
    seperator_major = '=' * len(header)

    print(seperator_major)
    print(header)
    print(row_format.format(*subheader))

    def single_press_info(k, r):
        if any([x[j, k, r].X > 0.1 for j in J]):
            print(seperator_major if k == 0 and r == 0 else seperator_minor)
        else:
            return

        # Initialize variables for total values
        tot_item_used = tot_item_waste = 0
        tot_items = set()
        tot_patterns = set()

        for j in J:
            if x[j, k, r].X > 0.1:
                for i in I:
                    if A[i, j] > 0.1:
                        item_waste = H[j] * (wr[k][r] - L[j]) * x[j, k, r].X / 1000 / 1000
                        item_used = x[j, k, r].X * A[i, j]
                        item_info = [f"{k}.{r}", i, f"{item_waste:.2f}", j, int(L[j]),
                                     int(item_used), b[i], f"{delta[i].X:.0f}"]
                        print(row_format.format(*item_info))
                        # Keep track of total values
                        tot_items.add(i)
                        tot_patterns.add(j)
                        tot_item_used += item_used
                        tot_item_waste += item_waste

        # Print total values
        item_info = ["==>", f"#{len(tot_items)}", f"={tot_item_waste:.2f}", f"#{len(tot_patterns)}", '-',
                     f"#{tot_item_used:.0f}", '-', '-']
        print(row_format.format(*item_info))

    for k in K:
        for r in R:
            single_press_info(k, r)

    print(seperator_major)
