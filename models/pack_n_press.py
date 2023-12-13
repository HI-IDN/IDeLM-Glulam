import gurobipy as gp
from gurobipy import GRB
import numpy as np

def pack_n_press(A, b, H, L, lr, np = 7, flag = 1):
  
  # parameters
  bigM = 100000000 # a big number

  # sets
  I = range(A.shape[0]) # items
  J = range(A.shape[1]) # cutting patterns
  K = range(np) # presses
  R = range(2) # regions

  # model and solve parameters
  pmodel = gp.Model("pack_n_press") # the packing model
  pmodel.setParam('OutputFlag', flag) # 0: silent, 1: summary, 2: detailed, 3: verbose
  pmodel.setParam('TimeLimit', 7 * 60) # 7 minutes

  # decision variables
  x = pmodel.addVars(J, K, R, vtype = GRB.INTEGER) # number of times pattern j is used in press k and region r
  Lp = pmodel.addVars(K,R)  # the maximum length of a region in the press
  omega = pmodel.addVars(K) # the total waste in the press (\omega)
  h = pmodel.addVars(K, R)  # what is the height of the region

  # the objective is to minimize the waste produced
  pmodel.setObjective(gp.quicksum(omega[p] for p in press), GRB.MINIMIZE)
  # compute heigth of each region
  pmodel.addConstrs(gp.quicksum(H[j]*x[j,k,r] for j in J) == h[k,r] for k in K for r in R)
  # the total height of the region must be less than the maximum height of the press
  pmodel.addConstrs(gp.quicksum(h[k,r] for r in R) <= 26 for k in K)
  pmodel.addConstrs(gp.quicksum(h[k,r] for r in R) >= 24 for k in K[:-1]) # the last press is ignored
  pmodel.addConstrs(h[k,0] >= 11 for k in K[:-1]) #  the last press is ignored
  # note that the length in each region is defined but the heights a free to vary
  pmodel.addConstrs(Lp[k,r] == gp.quicksum(L[j]*x[j,k,r] for j in J) for k in K for r in R)
  # the length of each region must be less than the maximum length of the press
  pmodel.addConstrs(Lp[k,r] <= lr[k,r] for k in K for r in R)

  # now we must fulfill the demand for each item
  pmodel.addConstrs(gp.quicksum(A[i,j]*x[j,k,r] for j in J for r in R) >= b[i] for i in I)

  # we must compute the waste in each press
  pmodel.addConstrs(omega[k] == gp.quicksum(H[j]*(Lp[k,r]-L[j]*x[j,k,r]) for j in J for r in R) for k in K)

  # now we add the objective function as the sum of waste for all presses
  pmodel.setObjective(gp.quicksum(omega[p] for p in K), GRB.MINIMIZE)
  
  # solve the model
  pmodel.optimize()

  # return all omega values
  return [omega[k].x for k in K]