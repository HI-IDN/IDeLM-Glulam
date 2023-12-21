import logging
import pandas as pd
import gurobipy as gp
import numpy as np
from config.settings import GlulamConfig
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('GlulamPackagingProc')


class GlulamPackagingProcessor:
    def __init__(self, pattern_processor, number_of_presses=GlulamConfig.MAX_PRESSES):
        """
        Initializes the GlulamPackagingProcessor with pattern data and the number of presses.

        Parameters:
        - pattern_processor (ExtendedGlulamPatternProcessor): Processor containing glulam pattern data.
        - number_of_presses (int): The number of presses available for packaging. Defaults to the maximum number
                                   specified in GlulamConfig.
        """
        self.patterns = pattern_processor
        """ Pattern data set from GlulamPatternProcessor. """

        self._number_of_presses = number_of_presses
        """ The number of presses. """

        self._number_of_regions = GlulamConfig.REGIONS
        """ The number of regions. """

        self.Waste = None
        """ Waste in each press and each region, given in m^2. """

        self.Lp = None
        """ The length of each region, given in mm. """

        self.Lp_ = None
        """ The actual length of each press, given in mm. """

        self.RW_used = None
        """ Roll widths used in all presses. """

        self.RW_counts = None
        """ Roll width frequencies in all presses. """

        self.solved = False
        """ Boolean, True if model has been packaged and pressed successfully. """

        self.ObjectiveValue = None
        """ Value of the objective function. """

    def update_number_of_presses(self, number_of_presses):
        """
        Update the number of presses that can be used.

        Parameters:
        - number_of_presses (int): The new number of presses to be used.
        """
        assert number_of_presses <= GlulamConfig.MAX_PRESSES, f'Cannot surpass {GlulamConfig.MAX_PRESSES} presses. Check data.'
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
        """ A matrix with patterns. """
        return self.patterns.A

    @property
    def b(self):
        return self.patterns.data.quantity

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

    # Define a function to process Gurobi logs

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

        # Use logger within these modules
        logger.info(f"Starting Pack'n'Press model for {self.number_of_presses} presses.")
        logger.info(f'Time limit for Gurobi: {time_limit} seconds.')

        self.solved = False
        self.RW_counts = False
        self.RW_used = False
        self.ObjectiveValue = None
        self.Lp = None
        self.Lp_ = None
        self.Waste = None
        self.x = None
        self.h = None

        # parameters
        bigM = 1e8  # a big number

        # model and solve parameters
        pmodel = gp.Model("Pack'N'Press")  # the packing model
        pmodel.setParam('OutputFlag', GlulamConfig.GUROBI_OUTPUT_FLAG)

        # Set time limit
        pmodel.setParam('TimeLimit', time_limit)

        # decision variables
        x = pmodel.addVars(self.J, self.K, self.R, vtype=gp.GRB.INTEGER)
        """ The number of times pattern j is used in press k and region r. """

        x1 = pmodel.addVars(self.J, self.K, self.R, vtype=gp.GRB.BINARY)
        """ Whether pattern j is used in press k and region r."""

        h = pmodel.addVars(self.K, self.R)
        """ The height of each region. """

        Lp = pmodel.addVars(self.K, self.R)
        """ The length of each region. """

        z = pmodel.addVars(self.K, self.R, vtype=gp.GRB.BINARY)
        """ Whether press k is used in region r. """

        h1 = pmodel.addVars(self.K, vtype=gp.GRB.BINARY)
        """ Whether the height of region 0 in press k is less than MIN_HEIGHT_LAYER_REGION layers (i.e. 24 layers)."""

        F = pmodel.addVars(self.J, self.K, self.R)
        """ The surplus of pattern j in press k and region r. """

        # indicate if a pattern is used or not in press k region r
        pmodel.addConstrs(x1[j, k, r] * bigM >= x[j, k, r] for j in self.J for k in self.K for r in self.R)

        # compute height of each region
        pmodel.addConstrs(
            gp.quicksum((self.H[j] / GlulamConfig.LAYER_HEIGHT) * x[j, k, r] for j in self.J) == h[k, r]
            for k in self.K for r in self.R)

        # the total height of the region must be less than the maximum height of the press
        pmodel.addConstrs(
            gp.quicksum(h[k, r] for r in self.R) <= GlulamConfig.MAX_HEIGHT_LAYERS for k in self.K)

        # h[k,0] is the height of the R0 in press k, and we must make sure that it is at least the minimum height
        pmodel.addConstrs(
            h[k, 0] >=
            GlulamConfig.MIN_HEIGHT_LAYER_REGION[0] - (1 - z[k, 0]) * bigM for k in self.K[:-1])

        # If we have two regions (i.e. R0 and R1 - never just R1) then we must make sure that the combined height of R0
        # and R1 is at least the minimum height for R1.
        pmodel.addConstrs(
            gp.quicksum(h[k, r] for r in self.R) >=
            GlulamConfig.MIN_HEIGHT_LAYER_REGION[1] - (1 - z[k, 0]) * bigM for k in self.K[:-1])
        pmodel.addConstrs(
            gp.quicksum(x[j, k, 0] for j in self.J) >= z[k, 1] for k in
            self.K)  # if region 1 is used then region 0 is used

        # if the press is not used the x must be zero
        pmodel.addConstrs(x[j, k, r] <= bigM * z[k, r] for j in self.J for k in self.K for r in self.R)

        # now we must fulfill the demand for each item exactly
        pmodel.addConstrs(
            gp.quicksum(self.A[i, j] * x[j, k, r] for j in self.J for k in self.K for r in self.R)
            == self.b[i] for i in self.I)

        # now there is the condition that is region 0 is below 24 then region 1 must have length less than 16m
        # h1[k] will indicate that the height of region 0 is less than 24 layers
        pmodel.addConstrs(
            h1[k] <= (GlulamConfig.MIN_HEIGHT_LAYER_REGION[1] - h[k, 0]) / GlulamConfig.MIN_HEIGHT_LAYER_REGION[1]
            for k in self.K[:-1])
        pmodel.addConstrs(
            Lp[k, r] >= GlulamConfig.MAX_ROLL_WIDTH_REGION[1] - h1[k] * bigM - (1 - z[k, 1]) * bigM
            for r in self.R for k in self.K[:-1])
        pmodel.addConstrs(Lp[k, r] >= x1[j, k, r] * self.L[j] for j in self.J for r in self.R for k in self.K)

        # make sure that all pattern length in region 1 are smaller than those in region 0
        # make sure that the length of region 0 is longer than region 1
        pmodel.addConstrs(Lp[k, 0] >= Lp[k, 1] - (1 - z[k, 1]) * bigM for k in self.K)
        pmodel.addConstrs(
            (Lp[k, r] - self.L[j]) * self.H[j] <= F[j, k, r] + (1 - x1[j, k, r]) * bigM
            for j in self.J for k in self.K for r in self.R)

        # now we add the objective function as the sum of waste for all presses and the difference between demand and supply
        pmodel.setObjective(
            gp.quicksum(F[j, k, r] for j in self.J for k in self.K for r in self.R)
            , gp.GRB.MINIMIZE)

        # solve the model
        pmodel.optimize()

        # see if model is infeasible
        if pmodel.status == gp.GRB.INFEASIBLE:
            logger.info(f"Pack'n'Press model for {self.number_of_presses} presses is infeasible; quitting.")
            return

        # Extract the results
        self.solved = True
        self.x = np.array([[[x[j, k, r].X > 0.1 for r in self.R] for k in self.K] for j in self.J], dtype=bool)
        self.RW_used, self.RW_counts = np.unique(
            [self.RW[j] for j in self.J for k in self.K for r in self.R if self.x[j, k, r]],
            return_counts=True)
        row_format = " ".join(["{:>5}"] * len(self.RW_used))
        logger.debug(f'RW_used:\n{row_format.format(*[str(x) for x in self.RW_used])}')
        logger.debug(f'RW_counts:\n{row_format.format(*[str(x) for x in self.RW_counts])}')
        self.ObjectiveValue = pmodel.ObjVal
        logger.debug(f'Objective value: {self.ObjectiveValue:.2f}')
        self.h = np.array([[h[k, r].X for r in self.R] for k in self.K], dtype=float)
        logger.debug(f'h:\n{self.h}')
        self.Lp = np.array([[Lp[k, r].X for r in self.R] for k in self.K], dtype=int)
        logger.debug(f'Lp:\n{self.Lp}')

        # Compute the waste
        self.Lp_ = np.max(self.L[:, None, None] * self.x, axis=0).astype(int)
        logger.debug(f'Lp_:\n{self.Lp_}')
        self.Waste = np.sum(self.H[:, None, None] * (self.Lp[None, :, :] - self.L[:, None, None]) * self.x / 1000000,
                            axis=0)  # Waste in m^2
        logger.info(f'Total waste: {self.TotalWaste:.3f} m^2')
        logger.debug(f'Waste:\n{self.Waste}')

    def print_results(self):
        if not self.solved:
            return False
        self.table_set_I()
        self.table_set_J()
        self.table_set_K()
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
                press_info = [f'{k}.{r}', self.Lp[k, r], self.h[k, r],
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

    def table_set_I(self):
        """ Table pertaining to set I (of all items). """
        logger.info(f'Item information: (m={self.patterns.m})')

        df = pd.DataFrame(
            columns=['Order', 'D', 'W', 'H', 'h', 'b', 'Ax', 'Ax-b'] + [f'P{k}' for k in self.K])

        df['Order'] = self.patterns.data.order
        df['D'] = self.patterns.data.depth
        df['W'] = self.patterns.data.widths
        df['H'] = self.patterns.data.heights
        df['h'] = self.patterns.data.layers
        df['b'] = self.patterns.data.quantity
        df['Ax'] = np.dot(self.A, np.sum(self.x, axis=(1, 2)))
        for k in self.K:
            df[f'P{k}'] = np.dot(self.A, np.sum(self.x[:, k, :], axis=1))  # A times sum of x over all regions
        df['Ax-b'] = df['Ax'] - df['b']
        if df['Ax-b'].sum() > 0:
            logger.warning(f"Surplus is {df['Ax-b'].sum()} pieces; check model.")
        if df['Ax-b'].sum() < 0:
            logger.error(f"Deficit of {-df['Ax-b'].sum()} pieces; check model.")

        print("\n\nTable: Item Information\n")
        print(df)
        row_format = "{:<17} {:>4} {:>4}"
        logger.info(row_format.format('Total Items', 'm', str(self.patterns.m)))
        logger.info(row_format.format('Total Production', 'Ax', str(np.sum(df['Ax']))))
        logger.info(row_format.format('Total Demand', 'b', str(np.sum(df['b']))))
        logger.info(row_format.format('Total Surplus', 'Ax-b', str(np.sum(df['Ax-b']))))

    def table_set_J(self):
        """ Table pertaining to set J (of all patterns). """
        logger.info(f'Pattern information: (n={self.patterns.n})')

        df = pd.DataFrame(
            columns=['h', 'H', 'L', 'Area', 'PatFr', 'ItCr', 'TotIt'] + [f'P{k}' for k in self.K])
        df['H'] = self.H
        df['h'] = (self.H / GlulamConfig.LAYER_HEIGHT).astype(int)
        df['L'] = self.L
        df['Area'] = self.H * self.L / 1000000
        df['PatFr'] = np.sum(self.x, axis=(1, 2))
        df['ItCr'] = np.sum(self.A, axis=0)
        df['TotIt'] = df['PatFr'] * df['ItCr']

        for k in self.K:
            df[f'P{k}'] = np.sum(self.x[:, k, :], axis=1)

        print("\n\nTable: Pattern Information\n")
        print(df.sort_values(by=['TotIt', 'PatFr']))
        print(f'Total patterns: n={self.patterns.n}')
        print(f'Total Area of all Pattern (Σ HxL * PatFr) {np.sum(df["Area"] * df["PatFr"]):.2f}m²')
        print(f'Pattern Usage Frequency (Σ PatFr) {np.sum(df["PatFr"])}')
        print(f'Items Created by Pattern (Σ ItCr * PatFr) {np.sum(df["ItCr"] * df["PatFr"])}')
        print(f'Total Items Produced (Σ TotIt) {np.sum(df["TotIt"])}')

    def table_set_K(self):
        """ Table pertaining to set K (of all presses). """
        logger.info(f'Press information: ({self.number_of_presses} number of presses)')
        df = pd.DataFrame(
            columns=['Press', 'Region', 'h', 'H', 'L', 'Lp', 'Lp_', 'HxL', 'HxLp', 'Waste', 'Patterns', 'Items'])
        df['Press'] = [f'{k}' for k in self.K for r in self.R]
        df['Region'] = [f'{r}' for k in self.K for r in self.R]
        df['Lp'] = [self.Lp[k, r] for k in self.K for r in self.R]
        df['Lp_'] = [self.Lp_[k, r] for k in self.K for r in self.R]

        print("\n\nTable: Press Information\n")
        print(df)
