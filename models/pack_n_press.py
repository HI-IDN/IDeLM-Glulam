import pandas as pd
import gurobipy as gp
import numpy as np
from config.settings import GlulamConfig
from utils.logger import setup_logger
import time
import pickle
from utils.plotter import get_press_layout, plot_rectangles

# Setup logger
logger = setup_logger('GlulamPackagingProc')

# Set pandas options to display precision to 3 significant digits
pd.set_option('display.precision', 3)


def cb(model, where):
    """
    Gurobi callback function for logging and termination conditions.

    Logs when the first feasible solution is found and logs termination due to no
    improvement in GUROBI_NO_IMPROVEMENT_TIME_LIMIT seconds after finding at least one feasible solution.

    Parameters:
    - model (gurobipy.Model): The Gurobi model.
    - where (int): Callback code indicating the current Gurobi state.
    """
    if where == gp.GRB.Callback.MIP:
        # Check if a feasible solution is found
        if model.cbGet(gp.GRB.Callback.MIP_SOLCNT) > 0 and not model._feasible:
            model._feasible = time.time()
            model._time = time.time()
            logger.info(
                f"First feasible solution found after {(model._feasible - model._start_time) / 60:.2f} minutes.")

    if where == gp.GRB.Callback.MIPNODE:
        # Get model objective
        obj = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBST)

        # Has objective changed?
        if abs(obj - model._cur_obj) > 1e-8:
            # Update incumbent and time
            model._cur_obj = obj
            model._time = time.time()

    # Terminate if a feasible solution has been found and no improvement for over a minute
    if model._feasible:
        if time.time() - model._time > GlulamConfig.GUROBI_NO_IMPROVEMENT_TIME_LIMIT:
            model.terminate()
            if not model._terminated:
                model._terminated = time.time()
                logger.warning(f"Terminating optimization: no improvement for over "
                               f"{GlulamConfig.GUROBI_NO_IMPROVEMENT_TIME_LIMIT / 60:.0f} minute. "
                               f"Elapsed time since first feasible solution: "
                               f"{(model._terminated - model._feasible) / 60:.2f} minutes.")


class GlulamPackagingProcessor:
    def __init__(self, pattern_processor, number_of_presses, number_of_regions=GlulamConfig.REGIONS):
        """
        Initializes the GlulamPackagingProcessor with pattern data and the number of presses.

        Parameters:
        - pattern_processor (ExtendedGlulamPatternProcessor): Processor containing glulam pattern data.
        - number_of_presses (int): The number of presses available for packaging.
        - number_of_regions (int): Number of regions in the press that can be used.
        """
        self.patterns = pattern_processor
        """ Pattern data set from GlulamPatternProcessor. """

        self._number_of_presses = number_of_presses
        """ The number of presses. """

        self._number_of_regions = number_of_regions
        """ The number of regions. """

        self.Waste = None
        """ Waste in each press and each region, given in m^2. """

        self.Lp_estimated = None
        """ The length of each region, given in mm. """

        self.Lp_actual = None
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
    def buffer_item(self):
        """
        A buffer item that is used to fill the last layer of the press.

        It is a dummy item with no demand, a single layer height and its width is determined by its region (i.e. Lp).
        """
        return {'order': 'buffer', 'quantity': 0, 'priority': False,
                'depth': self.patterns.data.depth, 'width': GlulamConfig.MAX_ROLL_WIDTH,
                'height': GlulamConfig.LAYER_HEIGHT, 'layers': 1}

    @property
    def I_priority(self):
        """
        The set of orders that are prioritized. I_priority ⫋ I.
        They that cannot be in the last press (which can be a partial press).
        """
        return self.patterns.data.priority_items

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
        logger.debug(f"Starting Pack'n'Press model for {self.number_of_presses} presses.")
        logger.debug(f'Time limit for Gurobi: {time_limit} seconds.')

        self.solved = False
        self.presses_in_use = np.full(self.number_of_presses, False)
        self.RW_counts = False
        self.RW_used = False
        self.ObjectiveValue = None
        self.Lp_estimated = None
        self.Lp_actual = None
        self.Waste = None
        self.x = None
        self.xn = None
        self.h = None
        self.buffer = None
        self.run_summary = None
        self.press_size = None

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

        B = pmodel.addVars(self.K)
        """ The number of buffer items in each press. """

        # indicate if a pattern is used or not in press k region r
        pmodel.addConstrs((x1[j, k, r] * bigM >= x[j, k, r] for j in self.J for k in self.K for r in self.R),
                          name="x1_definition")

        # compute height of each region
        pmodel.addConstrs(
            (gp.quicksum(self.H[j] * x[j, k, r] for j in self.J) == h[k, r]
             for k in self.K for r in self.R), name="height")

        # the total height of the region must be less than the maximum height of the press
        pmodel.addConstrs(
            (gp.quicksum(h[k, r] for r in self.R) + B[k] <= GlulamConfig.MAX_HEIGHT_LAYERS for k in self.K),
            name="max_height")

        # h[k,0] is the height of the R0 in press k, and we must make sure that it is at least the minimum height
        # not including the buffer item (because it's implied they go on top of all other items)
        pmodel.addConstrs(
            (h[k, 0] >=
             GlulamConfig.MIN_HEIGHT_LAYER_REGION[0] - (1 - z[k, 0]) * bigM for k in self.K[:-1]), name="min_height_R0")

        # now we must fulfill the demand for each item exactly (unless it is the buffer item)
        pmodel.addConstrs(
            (gp.quicksum(self.A[i, j] * x[j, k, r] for j in self.J for k in self.K for r in self.R)
             == self.b[i] for i in self.I), name="demand_equal_supply")

        # Limit the number of buffer items in each press
        pmodel.addConstrs((B[k] <= GlulamConfig.BUFFER_LIMIT for k in self.K), name="buffer_items_limit")

        # If we have two regions (i.e. R0 and R1 - never just R1) then we must make sure that the combined height of R0
        # and R1 is at least the minimum height for R1.
        pmodel.addConstrs(
            (gp.quicksum(h[k, r] for r in self.R) + B[k] >=
             GlulamConfig.MIN_HEIGHT_LAYER_REGION[1] - (1 - z[k, 0]) * bigM for k in self.K[:-1]),
            name="min_height_combined")

        # if the press is not used the x must be zero
        pmodel.addConstrs((x[j, k, r] <= bigM * z[k, r] for j in self.J for k in self.K for r in self.R),
                          name="if_press_is_not_used_then_x_is_zero")

        if self._number_of_regions > 1:
            # if region 1 is used then region 0 must be used
            pmodel.addConstrs((z[k, 0] >= z[k, 1] for k in self.K), name="if_region1_then_region0")

            # h1[k] will indicate that we have 2 regions, and region 1 is less than 16 meters
            pmodel.addConstrs(
                (h1[k] <= z[k, 1] for k in self.K[:-1]), name="h1_region1_used")
            pmodel.addConstrs(
                (Lp[k, 1] + bigM * (1 - h1[k]) >= 16000 for k in self.K[:-1]), name="Lp1_is_less_than_16m")
            pmodel.addConstrs(
                (h[k, 0] >= GlulamConfig.MIN_HEIGHT_LAYER_REGION[1] - bigM * (1 - h1[k]) for k in self.K[:-1]),
                name="h0_must_meet_min_height_24_if_region1_is_less_than_16m")

            # make sure that all pattern length in region 1 are sufficiently smaller than those in region 0
            pmodel.addConstrs(
                (Lp[k, 0] >= Lp[k, 1] + GlulamConfig.MINIMUM_REGION_DIFFERENCE - (1 - z[k, 1]) * bigM for k in self.K),
                name="Lp0_greater_than_Lp1")

        # make sure that the length of the region is at least the length of the longest pattern in use
        pmodel.addConstrs(Lp[k, r] >= x1[j, k, r] * self.L[j] for j in self.J for r in self.R for k in self.K)

        # Ensure priority items are not produced in the last press
        for j in self.J:
            if any(self.A[i, j] > 0 for i in self.I_priority):  # This pattern produces a priority item
                pmodel.addConstrs(
                    (x1[j, self.K[-1], r] == 0 for r in self.R), name="no_priority_in_last_press")

        # define the surplus of each pattern in each press and region as the difference between the length of the
        # pattern and the length of the region
        pmodel.addConstrs(
            ((Lp[k, r] - self.L[j]) * self.H[j] <= F[j, k, r] + (1 - x1[j, k, r]) * bigM
             for j in self.J for k in self.K for r in self.R), name="F_surplus_definition")

        # the objective function as the sum of waste for all presses and the difference between demand and supply
        pmodel.setObjective(
            gp.quicksum(F[j, k, r] for j in self.J for k in self.K for r in self.R)  # total waste
            + gp.quicksum(B[k] for k in self.K)  # number of buffer item used
            , gp.GRB.MINIMIZE)

        # Last updated objective and time
        pmodel._cur_obj = 1e100
        pmodel._start_time = time.time()  # Initialize the start time before optimization
        pmodel._feasible = None
        pmodel._terminated = None

        # solve the model
        pmodel.optimize(callback=cb)

        # see if model is infeasible
        if pmodel.status == gp.GRB.INFEASIBLE:
            logger.info(f"Pack'n'Press model for {self.number_of_presses} presses is infeasible; quitting.")
            return False

        try:
            assert x[0, 0, 0].X >= 0
        except:
            logger.info(f"Pack'n'Press model for {self.number_of_presses} presses is infeasible; quitting.")
            return False

        # Extract the results
        self.solved = True
        self.presses_in_use = (np.array([any(z[k, r].X for r in self.R) for k in self.K], dtype=bool))
        logger.info(f'Presses in use: {np.sum(self.presses_in_use)} out of {self.number_of_presses}')
        self.x = np.array([[[x[j, k, r].X > 0.1 for r in self.R] for k in self.K] for j in self.J], dtype=bool)
        self.xn = np.array([[[x[j, k, r].X for r in self.R] for k in self.K] for j in self.J], dtype=int)
        self.RW_used, self.RW_counts = np.unique(
            [self.RW[j] for j in self.J for k in self.K for r in self.R if self.x[j, k, r]],
            return_counts=True)
        row_format = " ".join(["{:>5}"] * len(self.RW_used))
        logger.debug(f'RW_used:\n{row_format.format(*[str(x) for x in self.RW_used])}')
        logger.debug(f'RW_counts:\n{row_format.format(*[str(x) for x in self.RW_counts])}')
        self.ObjectiveValue = pmodel.ObjVal
        logger.debug(f'Objective value: {self.ObjectiveValue:.2f}')
        self.h = np.array([[h[k, r].X for r in self.R] for k in self.K], dtype=int)
        logger.debug(f'h:\n{self.h}')
        self.Lp_estimated = np.array([[Lp[k, r].X for r in self.R] for k in self.K], dtype=int)
        logger.debug(f'Lp estimated:\n{self.Lp_estimated}')
        self.press_size = [(k, r, h[k, r].X, int(Lp[k, r].X)) for k in self.K for r in self.R if z[k, r].X > 0.1]
        self.buffer = np.array([B[k].X for k in self.K], dtype=int)
        logger.debug(f'B:\t{", ".join([str(x) for x in self.buffer])}')

        # Compute the waste
        self.Lp_actual = np.max(self.L[:, None, None] * self.x, axis=0).astype(int)
        logger.debug(f'Lp actual:\n{self.Lp_actual}')
        self.Waste = np.sum(
            self.H[:, None, None] * (
                    self.Lp_estimated[None, :, :] - self.L[:, None, None]) * self.x * GlulamConfig.LAYER_HEIGHT / 1e6,
            axis=0)  # Waste in m^2
        logger.info(f'Total waste: {self.TotalWaste:.3f} m^2')
        logger.debug(f'Waste:\n{self.Waste}')

        # Maintain a summary of the run
        self.run_summary = {
            'time': int(time.time() - pmodel._start_time),
            'first_feasible': int(pmodel._feasible - pmodel._start_time) if pmodel._feasible else None,
            'terminated_early': int(pmodel._terminated - pmodel._feasible) if pmodel._terminated else None,
            'nconstrs': pmodel.NumConstrs,
            'nvars': pmodel.NumVars,
            'status': pmodel.status,
            'npatterns': self.patterns.n,
            'norders': self.patterns.m,
            'npresses': self.number_of_presses,
            'npresses_used': np.sum(self.presses_in_use),
            'objective': self.ObjectiveValue,
            'total_waste': self.TotalWaste,
        }

        return True

    def print_results(self):
        if not self.solved:
            return False
        self.table_set_I()
        self.table_set_J()
        self.table_set_K()
        # self.print_item_results()

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
                    tot_press_height += self.xn[j, k, r] * self.H[j]
                    for i in self.I:
                        if self.A[i, j] > 0:
                            item_waste = self.H[j] * (self.Lp_estimated[k, r] - self.L[j]) * self.x[
                                j, k, r] * GlulamConfig.LAYER_HEIGHT / 1e6
                            item_used = self.xn[j, k, r] * self.A[i, j]
                            pattern_used = self.x[j, k, r]
                            item_info = [f"{k}.{r}", i, self.b[i], f"{item_waste:.2f}", j, self.L[j], self.H[j],
                                         pattern_used, item_used, self.Lp_estimated[k, r], f"{self.RW[j]:.0f}"]
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
        df['D'] = self.patterns.data.depths
        df['W'] = self.patterns.data.widths
        df['H'] = self.patterns.data.layers
        df['h'] = self.patterns.data.heights
        df['b'] = self.patterns.data.quantity
        # xj = np.array([np.sum([x[j,k,r].X for k in self.K for r in self.R]) for j in self.J])
        df['Ax'] = np.dot(self.A, np.sum(self.xn, axis=(1, 2)))
        for k in self.K:
            df[f'P{k}'] = np.dot(self.A, np.sum(self.xn[:, k, :], axis=1))  # A times sum of x over all regions
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
            columns=['H', 'h', 'L', 'Area', 'PatFr', 'ItCr', 'TotIt', 'RW'] + [f'P{k}' for k in self.K])
        df['H'] = self.H
        df['h'] = (self.H / GlulamConfig.LAYER_HEIGHT).astype(int)
        df['L'] = self.L
        df['Area'] = df['h'] * self.L / 1e6
        df['PatFr'] = np.sum(self.xn, axis=(1, 2))
        df['ItCr'] = np.sum(self.A, axis=0)
        df['TotIt'] = df['PatFr'] * df['ItCr']
        df['RW'] = self.RW

        for k in self.K:
            df[f'P{k}'] = np.sum(self.x[:, k, :], axis=1)

        print("\n\nTable: Pattern Information\n")
        print(df.sort_values(by=['TotIt', 'PatFr']))
        logger.info(f'Total patterns: n={self.patterns.n}')
        logger.info(f'Total Area of all Pattern (Σ HxL * PatFr) {np.sum(df["Area"] * df["PatFr"]):.2f}m²')
        logger.info(f'Pattern Usage Frequency (Σ PatFr) {np.sum(df["PatFr"])}')
        logger.info(f'Items Created by Pattern (Σ ItCr * PatFr) {np.sum(df["ItCr"] * df["PatFr"])}')
        logger.info(f'Total Items Produced (Σ TotIt) {np.sum(df["TotIt"])}')

    def table_set_K(self):
        """ Table pertaining to set K (of all presses). """
        logger.info(f'Press information: ({self.number_of_presses} number of presses)')
        df = pd.DataFrame(
            columns=['P', 'R', 'h', 'H', 'Lp', 'Lp¹', 'HxLp', 'Area', 'Waste', 'Pat', 'Its'])
        df['P'] = [k for k in self.K for r in self.R]
        df['R'] = [r for k in self.K for r in self.R]
        df['Lp'] = [self.Lp_estimated[k, r] for k in self.K for r in self.R]
        df['Lp¹'] = [self.Lp_actual[k, r] for k in self.K for r in self.R]
        df['h'] = [self.h[k, r] for k in self.K for r in self.R]
        df['H'] = (df['h'] * GlulamConfig.LAYER_HEIGHT).astype(int)
        df['HxLp'] = (df['H'] * df['Lp']) / 1e6
        df['Area'] = [np.sum(self.H[:, None, None] * (self.L[:, None, None]) * self.x / 1e6, axis=0)[k, r]
                      for k in self.K for r in self.R]
        df['Waste'] = [self.Waste[k, r] for k in self.K for r in self.R]
        df['Pat'] = [np.sum(self.xn[:, k, r]) for k in self.K for r in self.R]
        df['Its'] = [np.sum(self.A[:, :, np.newaxis, np.newaxis] * self.xn, axis=(1, 0))[k, r]
                     for k in self.K for r in self.R]
        df['Buf'] = [self.buffer[k] if r == 0 else 0 for k in self.K for r in self.R]

        print("\n\nTable: Press & Region Information\n")
        print(df)

        df_ = df.groupby(['P']).agg({'Area': 'sum', 'Waste': 'sum', 'Pat': 'sum', 'Its': 'sum', 'Buf': 'sum'})
        print("\n\nTable: Press Information\n")
        print(df_)

        logger.debug('Lp (mm): Estimated length of press')
        logger.debug('Lp¹ (mm): Actual length of press')
        logger.debug('HxLp (m²): Estimated area of press')
        logger.debug('HxLp¹ (m²): Actual area of press')
        logger.debug('Waste (m²): Waste in press HxLp - ΣHxLj (pattern j in press)')
        logger.info(f'Total area of all presses: {np.sum(df["Area"]):.2f}m²')
        logger.info(f'Total waste in all presses: {np.sum(df["Waste"]):.2f}m²')
        logger.info(f'Number of patterns in press: {np.sum(df["Pat"])}')
        logger.info(f'Number of items in press: {np.sum(df["Its"])}')
        logger.info(f'Number of buffer items in press: {np.sum(self.buffer)}')

    def save(self, filename, filename_png=None):
        """ Save results to csv and png. """
        assert self.solved, "Model must be solved before saving results."
        assert filename.endswith('.csv'), "Filename must end with .csv"

        if GlulamConfig.SAVE_PRESS_TO_PICKLE:
            pkl_filename = filename.replace('.csv', '.pkl')
            with open(pkl_filename, 'wb') as f:
                pickle.dump(self, f)
            logger.debug(f"Saving results to {pkl_filename}.")

        rects = get_press_layout(self, filename)
        logger.info(f"Saved rectangle results to {filename}.")

        if filename_png is not None:
            assert filename_png.endswith('.png'), "Filename must end with .png"
            plot_rectangles(rects, filename_png)
            logger.info(f"Plotted press to {filename_png}.")
