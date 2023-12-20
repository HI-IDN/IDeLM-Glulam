import gurobipy as gp
from gurobipy import GRB
import numpy as np
from config.settings import GlulamConfig


def pack_n_press(merged, number_of_presses, time_limit=GlulamConfig.GUROBI_TIME_LIMIT, debug=True):
  """
  Given a set of cutting patterns, pack them into presses such demand is fulfilled and the objective is
  1) to minimize the waste and 2) to minimize the difference between demand and supply.

  Parameters:
  - merged (ExtendedGlulamPatternProcessor): The glulam pattern data, merged into a single object for a set of
                                              several press configurations.
  - np (int): The number of presses to use.

  Returns:
  - omega (np.array): The waste in each press and each region.
  - Waste_ (np.array): The total waste in each press and each region.
  - z (np.array): Whether each press is used or not.
  - Lp_ (np.array): The length of each press.

  """
  H = merged.H
  L = merged.W
  A = merged.A
  b = merged.b
  RW = merged.RW

  # Assuming merged.O is your array of order IDs
  Oid = merged.O
  Ounique, inverse = np.unique(Oid, return_inverse=True)
  O = np.zeros((len(Oid), len(Ounique)))
  O[np.arange(len(Oid)), inverse] = 1
  print(O.shape)
  print(O)

  # parameters
  bigM = 100000000  # a big number
  delta = 100
  print("number of presses:", number_of_presses)

  # sets
  I = range(A.shape[0])  # items
  J = range(A.shape[1])  # cutting patterns
  K = range(number_of_presses)  # presses
  R = range(len(GlulamConfig.MIN_HEIGHT_LAYER_REGION))  # regions

  # model and solve parameters
  print("pack'n'press...")
  pmodel = gp.Model("pack_n_press")  # the packing model
  pmodel.setParam('OutputFlag', GlulamConfig.GUROBI_OUTPUT_FLAG)
  pmodel.setParam('TimeLimit', time_limit)

  # decision variables
  x = pmodel.addVars(J, K, R, vtype=GRB.INTEGER)  # number of times pattern j is used in press k and region r
  x1 = pmodel.addVars(J, K, R, vtype=GRB.BINARY)  # is pattern j used in press k and region r
  h = pmodel.addVars(K, R)  # what is the height of the region
  Lp = pmodel.addVars(K, R)  # what is the length of the region
  z = pmodel.addVars(K, R, vtype=GRB.BINARY)  # is press k used in region r
  h1 = pmodel.addVars(K, vtype=GRB.BINARY)  # is height in press k region 0 less than 24 layers
  F = pmodel.addVars(J, K, R)
  Cp = pmodel.addVars(K,I)
  Ci = pmodel.addVars(K,I, vtype=GRB.BINARY)

  # indicate if a pattern is used or not in press k region r
  pmodel.addConstrs(x1[j, k, r] * bigM >= x[j, k, r] for j in J for k in K for r in R)

  # compute height of each region
  pmodel.addConstrs(gp.quicksum((H[j] / 45.0) * x[j, k, r] for j in J) == h[k, r] for k in K for r in R)
  # the total height of the region must be less than the maximum height of the press
  pmodel.addConstrs(
      gp.quicksum(h[k, r] for r in R) <= GlulamConfig.MAX_HEIGHT_LAYERS for k in K)

  # h[k,0] is the height of the R0 in press k, and we must make sure that it is at least the minimum height
  pmodel.addConstrs(
      h[k, 0] >=
      GlulamConfig.MIN_HEIGHT_LAYER_REGION[0] - (1 - z[k, 0]) * bigM for k in K[:-1])

  # If we have two regions (i.e. R0 and R1 - never just R1) then we must make sure that the combined height of R0
  # and R1 is at least the minimum height for R1.
  pmodel.addConstrs(
      gp.quicksum(h[k, r] for r in R) >=
      GlulamConfig.MIN_HEIGHT_LAYER_REGION[1] - (1 - z[k, 0]) * bigM for k in K[:-1])
  pmodel.addConstrs(
      gp.quicksum(x[j, k, 0] for j in J) >= z[k, 1] for k in K)  # if region 1 is used then region 0 is used

  # is customer with product in this press and then how many?
  # pmodel.addConstrs(Cp[k,c] == gp.quicksum(x[j, k, r]*A[i,j]*O[i,c] for j in J for r in R) for k in K for c in C)
  # indicator if Cp is larger than zero
  # pmodel.addConstrs(Ci[k,c]*bigM >= Cp[k,c] for k in K for c in C)

  # if the press is not used the x must be zero
  pmodel.addConstrs(x[j, k, r] <= bigM * z[k, r] for j in J for k in K for r in R)

  # now we must fulfill the demand for each item exactly
  pmodel.addConstrs(gp.quicksum(A[i, j] * x[j, k, r] for j in J for k in K for r in R) == b[i] for i in I)

  # now there is the condition that is region 0 is below 24 then region 1 must have length less than 16m
  # h1[k] will indicate that the height of region 0 is less than 24 layers
  pmodel.addConstrs(h1[k] <= (24 - h[k, 0]) / 24 for k in K[:-1])
  pmodel.addConstrs(Lp[k, r] >= 16000 - h1[k] * bigM - (1 - z[k, 1]) * bigM for r in R for k in K[:-1])
  pmodel.addConstrs(Lp[k, r] >= x1[j, k, r] * L[j] for j in J for r in R for k in K)

  # make sure that all pattern length in region 1 are smaller than those in region 0
  # pmodel.addConstrs(x1[i, k, 0] * L[i] >= x1[j, k, 1] * L[j] - (1-x1[i, k, 0])*bigM - (1-x1[j, k, 1])*bigM for i in I for j in J for k in K)
  # make sure that the length of region 0 is longer than region 1
  pmodel.addConstrs(Lp[k, 0] >= Lp[k, 1] - (1 - z[k, 1]) * bigM for k in K)
  pmodel.addConstrs(
      (Lp[k, r] - L[j]) * H[j] <= F[j, k, r] + (1 - x1[j, k, r]) * bigM for j in J for k in K for r in R)
  # now we add the objective function as the sum of waste for all presses and the difference between demand and supply
  pmodel.setObjective(
      # gp.quicksum(Lp[k,r] for k in K for r in R
      # gp.quicksum((Lp[k,r] - L[j])*H[j]*x[j,k,r] for j in J for k in K for r in R)/1000./1000.
      gp.quicksum(F[j, k, r] for j in J for k in K for r in R)
      # + gp.quicksum(delta[k,r] for k in K for r in R)
      , GRB.MINIMIZE)

  # solve the model
  pmodel.optimize()

  # see if model is infeasible
  if pmodel.status == GRB.INFEASIBLE:
      print("The model is infeasible; quitting, increase number of presses")
      return False, None, None, None, None, None
  # extract rollwidths used
  RW_used = [RW[j] for j in J for k in K for r in R if x[j, k, r].X > 0.1]
  RW_used, RW_counts = np.unique(RW_used, return_counts=True)
  obj_value = pmodel.ObjVal

  # extract the solution
  Lp_ = np.zeros((len(K), len(R)))  # the length of the press
  Waste_ = np.zeros((len(K), len(R)))
  for k in K:
      for r in R:
          for j in J:
              if x[j, k, r].X > 0.1:
                  Lp_[k, r] = int(max(Lp_[k, r], L[j]))
                  Waste_[k, r] += H[j] * (Lp[k, r].X - L[j]) * x[j, k, r].X / 1000 / 1000  # convert to m^2
  # print the solution
  if debug == True:
      print_press_results(K, R, Lp_, h, Waste_)
      print_item_results(A, b, K, R, I, J, H, L, x, Lp, RW)

  # return all omega values
  return True, Waste_, Lp_, RW_used, RW_counts, obj_value


def print_press_results(K, R, Lp_, h, Waste_):
  """ Print the information about the presses. """
  print("\n\nTable: Press Information\n")
  row_format = "{:<5} {:>6} {:>6} {:>6}"
  header = ['Press', 'Width', 'Height', 'TrueWaste']
  header = row_format.format(*header)
  seperator_minor = '-' * len(header)
  seperator_major = '=' * len(header)

  print(seperator_major)
  print(header)

  for k in K:
    print(seperator_major if k == 0 else seperator_minor)
    for r in R:
      press_info = [f'{k}.{r}', np.round(Lp_[k, r]), np.round(h[k, r].X),
                    f"{Waste_[k, r]:.2f}"]
      print(row_format.format(*press_info))
  print(seperator_major)

def print_item_results(A, b, K, R, I, J, H, L, x, Lp, RW):
  """ Print the information about the items pressed. """
  print("\n\nTable 2: Item Information\n")
  row_format = "{:<5} {:>4} {:>4} {:>8} {:>4} {:>7} {:>4} {:>7} {:>5} {:>5} {:>9}"
  header = ['Press', 'Item', 'Order', 'Waste', 'Pat', 'Width', 'Height', '#Pat', 'Used', 'Len', 'Rollwidth']
  subheader = ['k.r', 'i', 'b[i]', 'H(Lp-L)x', 'j', 'L', 'H (rep)', '#j', '#j x A', 'Lp', 'Rw']
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
      tot_item_used = 0
      tot_item_waste = 0
      tot_press_height = 0
      tot_items = set()
      tot_patterns = set()

      for j in J:
          if x[j, k, r].X > 0.1:
              tot_press_height += np.round(x[j, k, r].X) * H[j] / 45.0
              for i in I:
                  if A[i, j] > 0.1:
                      item_waste = H[j] * (Lp[k, r].X - L[j]) * np.round(x[j, k, r].X) / 1000 / 1000
                      item_used = np.round(x[j, k, r].X) * A[i, j]
                      pattern_used = np.round(x[j, k, r].X)
                      item_info = [f"{k}.{r}", i, b[i], f"{item_waste:.2f}", j, L[j],
                                    np.round(H[j] / 45.0),
                                    pattern_used, item_used, np.round(Lp[k, r].X), f"{RW[j]:.0f}"]
                      print(row_format.format(*item_info))
                      # Keep track of total values
                      tot_items.add(i)
                      tot_patterns.add(j)
                      tot_item_used += item_used
                      tot_item_waste += item_waste

      # Print total values
      item_info = ["==>", f"#{len(tot_items)}", '-', f"={tot_item_waste:.2f}", f"#{len(tot_patterns)}", '-',
                    f"#{tot_press_height:.0f}", '-', '-', '-', '-']
      print(row_format.format(*item_info))

  for k in K:
      for r in R:
          single_press_info(k, r)

  print(seperator_major)


def test_pack_n_press(merged, number_of_presses, time_limit):
  success, Waste_, Lp_, RW_used, RW_counts, obj_value = pack_n_press(merged, number_of_presses, time_limit)
  return success, obj_value
