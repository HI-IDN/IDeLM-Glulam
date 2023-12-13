import gurobipy as gp
from gurobipy import GRB
import numpy as np

def pack_n_press(A, b, H, L, wr, flag = 1, debug = True):
  
  # parameters
  bigM = 100000000 # a big number
  np = len(wr)
  print("number of presses:",np)

  # sets
  I = range(A.shape[0]) # items
  J = range(A.shape[1]) # cutting patterns
  K = range(np) # presses
  R = range(2) # regions

  # debug:
  if debug == True:
    for i in I:
      print("item",i,"width",L[i],"height",H[i],"demand",b[i])

  # model and solve parameters
  pmodel = gp.Model("pack_n_press") # the packing model
  pmodel.setParam('OutputFlag', flag) # 0: silent, 1: summary, 2: detailed, 3: verbose
  pmodel.setParam('TimeLimit', 7 * 60) # 7 minutes

  # decision variables
  x = pmodel.addVars(J, K, R, vtype = GRB.INTEGER) # number of times pattern j is used in press k and region r
  Lp = pmodel.addVars(K,R)  # the maximum length of a region in the press
  omega = pmodel.addVars(K, R) # the total waste in the press (\omega)
  delta = pmodel.addVars(I) # the difference between demand and supply
  h = pmodel.addVars(K, R)  # what is the height of the region
  z = pmodel.addVars(K, vtype = GRB.BINARY) # is press k used?

  # compute heigth of each region
  pmodel.addConstrs(gp.quicksum(H[j]*x[j,k,r] for j in J) == h[k,r] for k in K for r in R)
  # the total height of the region must be less than the maximum height of the press
  pmodel.addConstrs(gp.quicksum(h[k,r] for r in R) <= 26*45.0 for k in K)
  pmodel.addConstrs(gp.quicksum(h[k,r] for r in R) >= 24*45.0 - z[k]*bigM for k in K[:-1]) # the last press is ignored
  pmodel.addConstrs(h[k,0] >= 11*45.0 for k in K[:-1]) #  the last press is ignored
  # note that the length in each region is defined but the heights a free to vary
  pmodel.addConstrs(Lp[k,r] >= L[j]*x[j,k,r] for j in J for k in K for r in R)
  # the length of each region must be less than the maximum length of the press
  pmodel.addConstrs(Lp[k,r] <= wr[k][r] + z[k]*bigM  for k in K for r in R if wr[k][r] > 0.1)

  # if the press is not used the x must be zero
  pmodel.addConstrs(x[j,k,r] <= bigM*z[k] for j in J for k in K for r in R)

  # now we must fulfill the demand for each item
  pmodel.addConstrs(gp.quicksum(A[i,j]*x[j,k,r] for j in J for k in K for r in R) >= b[i] for i in I)
  pmodel.addConstrs(gp.quicksum(A[i,j]*x[j,k,r] for j in J for k in K for r in R) <= b[i] + delta[i] for i in I)

  # we must compute the waste in each press
  pmodel.addConstrs(omega[k,r] == gp.quicksum(H[j]*(Lp[k,r]-L[j]*x[j,k,r]) for j in J) for k in K for r in R)

  # now we add the objective function as the sum of waste for all presses
  pmodel.setObjective(1000*gp.quicksum(delta[i] for i in I) + gp.quicksum(omega[k,r]/1000.0/1000.0 for k in K for r in R), GRB.MINIMIZE)
  
  # solve the model
  pmodel.optimize()

  # see if model is infeasible
  if pmodel.status == GRB.INFEASIBLE:
    print("The model is infeasible; quitting.")
    return None
  
  if debug == True:
    for k in K:
      for r in R:
        print("press",k,"region",r,"length",Lp[k,r].X,"height",h[k,r].X/45.0,"waste",omega[k,r].X/1000/1000,'m^2')
        for j in J:
          if x[j,k,r].X > 0.1:
            for i in I:
              if A[i,j] > 0.1:
                print("item",i," in pattern", j, "of width", L[j], "used", A[i,j], "times and order is: ",b[i], "/", b[i], "delta =",delta[i].X)

  # return all omega values
  return [omega[k,r].X for k in K for r in R]