# models/cutting_pattern.py
# Import necessary libraries for Gurobi MIP modelling
from config.settings import GlulamConfig
import numpy as np
import gurobipy as gp
from utils.logger import setup_logger

# Setup logger
single_logger = setup_logger('GlulamPatternProcessor')
merged_logger = setup_logger('ExtendedGlulamPatternProcessor')


class GlulamPatternProcessor:
    def __init__(self, data, roll_width=None):
        """
        Initializes the GlulamPatternProcessor with the necessary data.

        Parameters:
        - data (GlulamDataProcessor): An instance of the GlulamDataProcessor class, which contains the glulam data.
        - roll_width (float, optional): The maximum width available on the roll. Defaults to the value in GlulamConfig.
        """
        self.data = data

        if roll_width is None:
            self.roll_width = GlulamConfig.MAX_ROLL_WIDTH
        else:
            self.roll_width = roll_width
            assert roll_width <= GlulamConfig.MAX_ROLL_WIDTH, (f"Roll width {roll_width} mm exceeds the maximum roll "
                                                               f"width {GlulamConfig.MAX_ROLL_WIDTH} mm.")

        # Use logger within these modules
        single_logger.info(f"Initialising Glulam Pattern Processor instance with {self.roll_width} roll width.")

        # The starting cutting patterns for each order
        self._A = np.zeros((self.data.m, self.data.m * 2), dtype=int)
        self._H = np.zeros(self.data.m * 2, dtype=int)
        self._W = np.zeros(self.data.m * 2, dtype=int)
        self._RW = np.zeros(self.data.m * 2, dtype=int)

        # Create one piece of each item
        for i in range(self.data.m):
            self._A[i, i] = 1
            self._H[i] = self.data.heights[i]
            self._W[i] = self.data.widths[i] * self._A[i, i]
            self._RW[i] = self._W[i]  # or rounded to the nearest multiple of GlulamConfig.ROLL_WIDTH_TOLERANCE

        # Create as many pieces as possible of each item
        for i in range(self.data.m):
            copies = np.floor(self.roll_width / self.data.widths[i])  # How many copies of the pattern can be made
            copies = min(copies, self.data.quantity[i])  # Never make more copies of the pattern than the demand
            copies = max(copies, 1)  # Make at least one copy of the pattern, even if the demand is 0
            self._A[i, self.data.m + i] = copies
            self._H[self.data.m + i] = self.data.heights[i]
            self._W[self.data.m + i] = self.data.widths[i] * self._A[i, self.data.m + i]
            self._RW[self.data.m + i] = self._W[self.data.m + i]

        self._remove_duplicate_patterns()

        #single_logger.info(f"Added {self.n} unique initial patterns.")

    @property
    def A(self):
        """ Pattern matrix, a matrix of size m x n, where m is the number of orders and n is the number of patterns. """
        return self._A

    @property
    def b(self):
        """ The demand for each order, a vector of size m, such that Ax = b.

        This vector represents the quantities for each order type, filtered such that only those orders are
        considered where the item width does not exceed the roll width.
        In other words, this method ignores the quantities for item widths larger than the roll width.
        """
        return [q if w <= self.roll_width else 0 for q, w in zip(self.data.quantity, self.data.widths)]

    @property
    def H(self):
        """ Pattern height """
        return self._H

    @property
    def W(self):
        """ Pattern total width """
        return self._W

    @property
    def RW(self):
        """ Roll width """
        return self._RW

    @property
    def m(self):
        """ Number of orders """
        return self.data.m

    @property
    def I(self):
        """ Set of orders, i.e. the rows in the pattern matrix 0 <= i < m """
        return range(self.m)

    @property
    def n(self):
        """ Number of patterns."""
        return self._A.shape[1]

    @property
    def J(self):
        """ Set of patterns, i.e. the columns in the pattern matrix 0 <= j < n """
        return range(self.n)

    @property
    def H(self):
        """ Pattern height """
        return self._H

    @property
    def W(self):
        """ Pattern total width """
        return self._W

    def cutting_stock_column_generation(self):
        """
        Solves the cutting stock problem using column generation.

        Returns:
        - A (matrix): Final pattern matrix.
        - x (dict): Quantities for each pattern to be cut.
        """

        bailout = False
        while not bailout:
            # Create the cutting model
            cut_model = gp.Model("Cutting")
            cut_model.setParam('OutputFlag', 0)

            # Decision variables: how often to cut each pattern
            x = cut_model.addVars(self.J)  # Note this a continuous variable in order to get shadow prices later

            # Objective: Minimize the total number of patterns used
            cut_model.setObjective(gp.quicksum(x[j] for j in self.J), gp.GRB.MINIMIZE)

            # Constraints: Ensure all orders are satisfied
            ci = 0
            cut_model.addConstrs(gp.quicksum(self._A[i, j] * x[j] for j in self.J) >= self.b[i]
                                 for i in self.I)
            # Constraints: Ensure no more than MAX_SURPLUS_QUANTITY pieces are left over
            if GlulamConfig.SURPLUS_LIMIT > 0:
                cut_model.addConstrs(
                    -gp.quicksum(self._A[i, j] * x[j] for j in self.J) >= -self.b[i] - GlulamConfig.SURPLUS_LIMIT
                    for i in self.I)

            # Solve the master problem
            cut_model.optimize()

            # Retrieve the dual prices from the constraints
            pi = [c.Pi for c in cut_model.getConstrs()]
            if GlulamConfig.SURPLUS_LIMIT > 0:  # Adjust the dual prices if surplus limit is used
                pi = [pi[i] - pi[i + self.data.m] for i in range(self.data.m)]

            # Solve the column generation sub problem
            bailout = self._column_generation_subproblem(pi)

        # Return the cutting frequencies
        return {j: x[j].X for j in self.J}

    def _column_generation_subproblem(self, pi):
        """
        Solves the column generation subproblem for the cutting stock problem.

        Parameters:
        - pi (array): Dual prices from the master problem's constraints.

        Returns:
        - (bool) True if no more patterns with negative reduced cost are found, False otherwise (i.e. if a new
        pattern has been added to the pattern matrix).
        """
        # A large number for big-M method in MIP
        bigM = 1e6

        # Initialize the knapsack model
        knap_model = gp.Model("Knapsack")
        knap_model.setParam('OutputFlag', 0)  # Suppress output for cleaner execution

        # Decision variables for the knapsack problem
        use = knap_model.addVars(self.I, lb=0, vtype=gp.GRB.INTEGER)
        """ Pattern usage variables: How many times each pattern is used. """

        h = knap_model.addVar()
        """ Height of the roll. """

        z = knap_model.addVars(self.I, vtype=gp.GRB.BINARY)
        """ Indicator variables for height constraints. """

        # Objective: Minimize reduced cost
        knap_model.setObjective(1.0 - gp.quicksum(pi[i] * use[i] for i in self.I), gp.GRB.MINIMIZE)

        # Width constraint: Total width used must not exceed roll width
        knap_model.addConstr(
            gp.quicksum(self.data.widths[i] * use[i] for i in self.I) <= self.roll_width)  # Width limit
        knap_model.addConstr(
            gp.quicksum(self.data.widths[i] * use[i] for i in
                        self.I) >= self.roll_width - GlulamConfig.ROLL_WIDTH_TOLERANCE)  # Width limit

        # Indicator constraints for height limits
        knap_model.addConstrs(z[i] * bigM >= use[i] for i in self.I)  # If z[i] = 0, then use[i] = 0 (Indicator constr.)
        knap_model.addConstrs(h >= self.data.heights[i] - bigM * (1 - z[i]) for i in self.I)  # Height limit low
        knap_model.addConstrs(h <= self.data.heights[i] + bigM * (1 - z[i]) for i in self.I)  # Height limit high

        # Solve the knapsack problem
        knap_model.optimize()

        # Bailout if not feasible solution found
        if knap_model.status != gp.GRB.OPTIMAL:
            single_logger.info(f"Cannot find more patterns; quitting the process with n={self.n}.")
            return True

        # Check if a new pattern with negative reduced cost is found
        if knap_model.objval < -0.00000001:

            # Generate a new pattern based on the solution of the sub problem
            new_pattern = np.array([[use[i].X] for i in self.I], dtype=int)

            # Append the new pattern to the existing pattern matrix A: This horizontally stacks the new pattern to
            # the end of the matrix
            self._A = np.hstack((self._A, new_pattern))

            # Append the height of the new pattern to the H array
            self._H = np.concatenate((self._H, np.array([h.X], dtype=int)))

            # Calculate the total width of the new pattern: This is done by summing the width of each item in the
            # pattern multiplied by its usage (use[i].X). Then, flatten the array to ensure it's a 1D array
            W = np.array(np.sum([use[i].X * self.data.widths[i] for i in self.I]), dtype=int).flatten()
            # Append this total width of the new pattern to the W array of existing widths
            self._W = np.concatenate((self._W, W))

            # Append the roll width to the R array
            self._RW = np.concatenate((self._RW, np.array([self.roll_width], dtype=int)))

            #single_logger.info(f"Added a new pattern; continuing the process (n={self.n}).")
            return False
        else:
            #single_logger.info(
            #    f"No more patterns with negative reduced cost found; quitting the process with n={self.n}.")
            return True

    def _remove_duplicate_patterns(self):
        """
        Removes duplicate patterns from the pattern matrix A, and corresponding entries in H and W arrays.
        """

        # Find unique patterns in A - keep indices and update corresponding arrays H, W and R
        self._A, unique_indices = np.unique(self._A, axis=1, return_index=True)

        self._H = self._H[unique_indices]
        self._W = self._W[unique_indices]
        self._RW = self._RW[unique_indices]


class ExtendedGlulamPatternProcessor(GlulamPatternProcessor):
    def __init__(self, data):
        """
        Initializes the ExtendedGlulamPatternProcessor with the necessary data.

        Parameters:
        - data (GlulamDataProcessor): An instance of the GlulamDataProcessor class, which contains the glulam data.
        """
        super().__init__(data, roll_width=None)  # Initialize the base class
        self._roll_widths = set()
        #merged_logger.info(f"Initialising Extended Glulam Pattern Processor instance with {self.n} patterns.")

    @property
    def roll_widths(self):
        return self._roll_widths

    def add_roll_width(self, roll_width):
        """
        Generates cutting patterns for the given roll width and adds them to the existing patterns.
        """
        if roll_width in self._roll_widths:
            merged_logger.info(f"Roll width {roll_width} already exists in the existing patterns, n={self.n}.")
            return

        pattern = GlulamPatternProcessor(self.data, roll_width)
        pattern.cutting_stock_column_generation()
        self._A = np.hstack((self._A, pattern.A))
        self._H = np.concatenate((self._H, pattern.H))
        self._W = np.concatenate((self._W, pattern.W))
        self._RW = np.concatenate((self._RW, pattern.RW))
        self._remove_duplicate_patterns()
        self._roll_widths.add(roll_width)
        #merged_logger.info(
        #    f"Added {pattern.n} patterns for roll width {roll_width} to the existing patterns, now n={self.n}.")

    def remove_roll_width(self, roll_width):
        """
        Removes cutting patterns for the given roll width from the existing patterns.
        """
        if roll_width not in self.roll_widths:
            merged_logger.info(f"Roll width {roll_width} not found in the existing patterns, n={self.n}.")
            return

        ix = np.where(self._RW == roll_width)
        self._A = np.delete(self._A, ix, axis=1)
        self._H = np.delete(self._H, ix)
        self._W = np.delete(self._W, ix)
        self._RW = np.delete(self._RW, ix)
        self._roll_widths.remove(roll_width)
        merged_logger.info(
            f"Removed {len(ix)} patterns for roll width {roll_width} from the existing patterns, now n={self.n}.")
