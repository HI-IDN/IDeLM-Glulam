import gurobipy as gp
from gurobipy import GRB
import numpy as np
from config.settings import GlulamConfig


class GlulamPackagingProcessor:
    def __init__(self, pattern_processor, number_of_presses=GlulamConfig.MAX_PRESSES):
        self.patterns = pattern_processor
        """ Pattern data set from GlulamPatternProcessor. """

        self._number_of_presses = number_of_presses
        """ The number of presses. """

        self._number_of_regions = GlulamConfig.REGIONS
        """ The number of regions. """

        self.Waste = None
        """ Waste in each press and each region. """

        self.Lp = None
        """ Length of each press. """

        self.RW_used = None
        """ Roll widths used. """

        self.RW_counts = None
        """ Roll width counts. """

        self.solved = False
        """ Boolean if model has been packaged and pressed successfully. """

    def update_number_of_presses(self, number_of_presses):
        """ Update the number of presses that can be used. """
        self._number_of_presses = number_of_presses

    @property
    def TotalWaste(self):
        """ Total waste in all presses. """
        return None if self.Waste is None else np.sum(self.Waste)

    @property
    def number_of_presses(self):
        """ Number of presses. """
        return self._number_of_presses

    @property
    def H(self):
        return self.patterns.H

    @property
    def L(self):
        return self.patterns.W

    @property
    def A(self):
        return self.patterns.A

    @property
    def b(self):
        return self.patterns.b

    @property
    def RW(self):
        return self.patterns.RW

    @property
    def I(self):
        return self.patterns.I

    @property
    def J(self):
        return self.patterns.J

    @property
    def K(self):
        """ The set of presses. """
        return range(self._number_of_presses)

    @property
    def R(self):
        """ The set of regions. """
        return range(self._number_of_regions)

    def pack_n_press(self, time_limit=GlulamConfig.GUROBI_TIME_LIMIT):
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

        H = self.H
        L = self.L
        A = self.A
        b = self.b
        RW = self.RW

        # Assuming merged.O is your array of order IDs
        Oid = self.patterns.O
        Ounique, inverse = np.unique(Oid, return_inverse=True)
        O = np.zeros((len(Oid), len(Ounique)))
        O[np.arange(len(Oid)), inverse] = 1
        print(O.shape)
        print(O)

        # parameters
        bigM = 1e8  # a big number
        delta = 100
        print("number of presses:", self._number_of_presses)

        # sets
        I = self.I
        J = self.J
        K = self.K
        R = range(self._number_of_regions)  # regions

        # model and solve parameters
        print("pack'n'press...")
        pmodel = gp.Model("pack_n_press")  # the packing model
        pmodel.setParam('OutputFlag', GlulamConfig.GUROBI_OUTPUT_FLAG)
        pmodel.setParam('TimeLimit', time_limit)

        # decision variables
        x = pmodel.addVars(J, K, R, vtype=GRB.INTEGER)
        """ The number of times pattern j is used in press k and region r. """

        x1 = pmodel.addVars(J, K, R, vtype=GRB.BINARY)
        """ Whether pattern j is used in press k and region r."""

        h = pmodel.addVars(K, R)
        """ The height of each region. """

        Lp = pmodel.addVars(K, R)
        """ The length of each region. """

        z = pmodel.addVars(K, R, vtype=GRB.BINARY)
        """ Whether press k is used in region r. """

        h1 = pmodel.addVars(K, vtype=GRB.BINARY)
        """ Whether the height of region 0 in press k is less than MIN_HEIGHT_LAYER_REGION layers (i.e. 24 layers)."""

        F = pmodel.addVars(J, K, R)
        """ The surplus of pattern j in press k and region r. """

        Cp = pmodel.addVars(K, I)
        Ci = pmodel.addVars(K, I, vtype=GRB.BINARY)

        # indicate if a pattern is used or not in press k region r
        pmodel.addConstrs(x1[j, k, r] * bigM >= x[j, k, r] for j in J for k in K for r in R)

        # compute height of each region
        pmodel.addConstrs(
            gp.quicksum((H[j] / GlulamConfig.LAYER_HEIGHT) * x[j, k, r] for j in J) == h[k, r] for k in K for r in R)
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
        pmodel.addConstrs(
            h1[k] <= (GlulamConfig.MIN_HEIGHT_LAYER_REGION[1] - h[k, 0]) / GlulamConfig.MIN_HEIGHT_LAYER_REGION[1]
            for k in K[:-1])
        pmodel.addConstrs(
            Lp[k, r] >= GlulamConfig.MAX_ROLL_WIDTH_REGION[1] - h1[k] * bigM - (1 - z[k, 1]) * bigM
            for r in R for k in K[:-1])
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
            self.solved = False
            self.RW_counts = False
            self.RW_used = False
            self.obj_value = None
            self.Lp = None
            self.Waste = None
            self.x = None
            self.h = None
            return None, None, None, None, None

        # extract roll widths used
        self.RW_used, self.RW_counts = np.unique([RW[j] for j in J for k in K for r in R if x[j, k, r].X > 0.1],
                                                 return_counts=True)
        self.obj_value = pmodel.ObjVal
        self.x = np.array([[[x[j, k, r].X > 0.1 for r in R] for k in K] for j in J], dtype=bool)
        self.h = np.array([[h[k, r].X for r in R] for k in K], dtype=float)
        self.Lp = np.array([[Lp[k, r].X for r in R] for k in K], dtype=int)

        # extract the solution
        Lp_ = np.zeros((len(K), len(R)))  # the length of the press
        Waste_ = np.zeros((len(K), len(R)))
        for k in K:
            for r in R:
                for j in J:
                    if self.x[j, k, r]:
                        Lp_[k, r] = int(max(Lp_[k, r], L[j]))
                        Waste_[k, r] += H[j] * (Lp[k, r].X - L[j]) * self.x[j, k, r] / 1000 / 1000  # convert to m^2
        self.Lp_ = Lp_
        self.Waste = Waste_

        # return all omega values
        self.solved = True
        return Waste_, Lp_, self.RW_used, self.RW_counts, self.obj_value

    def print_results(self):
        self.print_press_results()
        self.print_item_results()

    def print_press_results(self):
        if not self.solved:
            return

        """ Print the information about the presses. """
        print("\n\nTable: Press Information\n")
        row_format = "{:<5} {:>8} {:>6} {:>8}"
        header = ['Press', 'Width', 'Height', 'TrueWaste']
        header = row_format.format(*header)
        seperator_minor = '-' * len(header)
        seperator_major = '=' * len(header)

        print(seperator_major)
        print(header)

        for k in self.K:
            print(seperator_major if k == 0 else seperator_minor)
            for r in self.R:
                press_info = [f'{k}.{r}', np.round(self.Lp[k, r]), np.round(self.h[k, r]),
                              f"{self.Waste[k, r]:.2f}"]
                print(row_format.format(*press_info))
        print(seperator_major)

    def print_item_results(self):
        if not self.solved:
            return

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
            if any([self.x[j, k, r] for j in self.J]):
                print(seperator_major if k == 0 and r == 0 else seperator_minor)
            else:
                return

            # Initialize variables for total values
            tot_item_used = 0
            tot_item_waste = 0
            tot_press_height = 0
            tot_items = set()
            tot_patterns = set()

            for j in self.J:
                if self.x[j, k, r]:
                    tot_press_height += self.x[j, k, r] * self.H[j] / GlulamConfig.LAYER_HEIGHT
                    for i in self.I:
                        if self.A[i, j] > 0:
                            item_waste = self.H[j] * (self.Lp[k, r] - self.L[j]) * self.x[j, k, r] / 1000 / 1000
                            item_used = self.x[j, k, r] * self.A[i, j]
                            pattern_used = self.x[j, k, r]
                            item_info = [f"{k}.{r}", i, self.b[i], f"{item_waste:.2f}", j, self.L[j],
                                         np.round(self.H[j] / GlulamConfig.LAYER_HEIGHT),
                                         pattern_used, item_used, self.Lp[k, r], f"{self.RW[j]:.0f}"]
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

        for k in self.K:
            for r in self.R:
                single_press_info(k, r)

        print(seperator_major)

    def table_press_info(self):
        """ Table per press and true waste. """
        # Tafla 1: press and true waste.
        pass

    def table_press_region_info(self):
        """ Table per press and region - with patterns lenght heigh and max length. """
        # Tafla 2: press + region, patterns, length, height, og max length Lp(ekkert pseudo - area)
        pass

    def table_press_item_order(self):
        """ Table per item - what press they are in and what order. """
        pass
