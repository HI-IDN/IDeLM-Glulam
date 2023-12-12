def packNpress(df_repeated, np, env, flag = 1):
  # parameters:
  bigM = 100000000
  H = df_repeated['height']
  T = df_repeated['totlen']

  #sets:
  beams = range(len(df_repeated)) # all the beams needed to be stacked in presses in the different regions
  press = range(np)   # we need to experiment with the number of presses
  region = [1,2,3]   # the three regions discussed

  pressmodel = gp.Model("CuttingL", env=env)
  pressmodel.setParam('OutputFlag', flag)
  pressmodel.setParam('TimeLimit', 7*60)

  # Decision variables
  Lp = pressmodel.addVars(press, region, ub = 25000) # the maximum length of a region in the press
  z = pressmodel.addVars(press, region, beams, vtype = gp.GRB.BINARY) # should beam be in press and region
  y = pressmodel.addVars(press, vtype = gp.GRB.BINARY) # is region 1 above 28 units?
  y_ = pressmodel.addVars(press, vtype = gp.GRB.BINARY) # is region 1+3 above 28 units?
  waste = pressmodel.addVars(press) # the total waste in the press (\omega)
  s = pressmodel.addVars(press, lb = -9000, ub = 16000) # this slack variable will align region 2 to minimize waste
  Lp = pressmodel.addVars(press, [12,3], ub = 25000) # the maximum length of the RHS of press 1+2 and region 3
  w =  pressmodel.addVars(press, region, I) # the waste produced per beam
  x =  pressmodel.addVars(press, region, vtype = gp.GRB.BINARY) # is the region used?
  h =  pressmodel.addVars(press, region) # what is the height of the region

  # the objective is to minimize the waste produced
  pressmodel.setObjective(gp.quicksum(waste[p] for p in press), gp.GRB.MINIMIZE)
  # each beam must be packed within some press in some region
  pressmodel.addConstrs(gp.quicksum(z[p,r,i] for p in press for r in region) == 1 for i in beams )

  # TODO TASK 3 !!!! add the missing constraints!

  # solve the model
  pressmodel.optimize()
