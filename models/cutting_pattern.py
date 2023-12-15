# models/cutting_pattern.py
# Import necessary libraries for Gurobi MIP modelling
from config.settings import GlulamConfig
import numpy as np
import gurobipy as gp


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

        # The starting cutting patterns for each order
        self._A = np.zeros((self.data.m, self.data.m * 2))
        self._H = np.zeros(self.data.m * 2)
        self._W = np.zeros(self.data.m * 2)
        self._R = np.zeros(self.data.m * 2)
        for i in range(self.data.m):
            self._A[i, i] = 1
            self._H[i] = self.data.heights[i]
            self._W[i] = self.data.widths[i] * self._A[i, i]
            self._R[i] = self.data.widths[i]
        for i in range(self.data.m):
            copies = np.floor(self.roll_width / self.data.widths[i])  # How many copies of the pattern can be made
            copies = min(copies, self.data.quantity[i])  # Never make more copies of the pattern than the demand
            copies = max(copies, 1)  # Make at least one copy of the pattern, even if the demand is 0
            self._A[i, self.data.m + i] = copies
            self._H[self.data.m + i] = self.data.heights[i]
            self._W[self.data.m + i] = self.data.widths[i] * self._A[i, self.data.m + i]
            self._R[i] = self.data.widths[i] * copies

        # Final check to ensure all diagonal elements in A are greater than 0
        if not np.all(np.diag(self._A) > 0):
            raise ValueError("Invalid pattern matrix: Some diagonal elements in A are not greater than 0.")

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
    def R(self):
        """ Roll width """
        return self._R

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
        # return self.data.widths @ self._A
        return self._W

    def cutting_stock_column_generation(self):
        """
        Solves the cutting stock problem using column generation.

        Returns:
        - A (matrix): Final pattern matrix.
        - x (dict): Quantities for each pattern to be cut.
        """

        def filter_unused_patterns(x):
            """
            Removes unused patterns from the pattern matrix A, and corresponding entries in H and W arrays.
            A pattern is considered unused if its usage value x[j] is close to zero.

            Parameters:
                - x (gurobi.Var): Quantities for each pattern to be cut, a decision variable in cut_model problem.
            """
            # Create a filter for indices of patterns that are used (x[j] > 0) and not lose the identity patterns
            used_patterns_filter = [j for j in self.J if x[j].X > 0.0000001 or j <= self.data.m]

            # Apply the filter to H, W, and A
            self._H = self._H[used_patterns_filter]
            self._W = self._W[used_patterns_filter]
            self._A = self._A[:, used_patterns_filter]
            self._R = self._R[used_patterns_filter]

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

            # Remove columns in A[:,:] corresponding to x[j] = 0
            filter_unused_patterns(x)

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
        bigM = 1000000

        # Initialize the knapsack model
        knapmodel = gp.Model("Knapsack")
        knapmodel.setParam('OutputFlag', 0)  # Suppress output for cleaner execution

        # Decision variables for the knapsack problem
        use = knapmodel.addVars(self.I, lb=0, vtype=gp.GRB.INTEGER)  # Pattern usage variables
        h = knapmodel.addVar()  # Height of the roll
        z = knapmodel.addVars(self.I, vtype=gp.GRB.BINARY)  # Binary variables for height constraints

        # Objective: Minimize reduced cost
        knapmodel.setObjective(1.0 - gp.quicksum(pi[i] * use[i] for i in self.I), gp.GRB.MINIMIZE)

        # Width constraint: Total width used must not exceed roll width
        knapmodel.addConstr(gp.quicksum(self.data.widths[i] * use[i] for i in self.I) <= self.roll_width)  # Width limit
        knapmodel.addConstr(
            gp.quicksum(self.data.widths[i] * use[i] for i in
                        self.I) >= self.roll_width - GlulamConfig.ROLL_WIDTH_TOLERANCE)  # Width limit

        # Indicator constraints for height limits
        knapmodel.addConstrs(z[i] * bigM >= use[i] for i in self.I)  # If z[i] = 0, then use[i] = 0 (Indicator constr.)
        knapmodel.addConstrs(h >= self.data.heights[i] - bigM * (1 - z[i]) for i in self.I)  # Height limit low
        knapmodel.addConstrs(h <= self.data.heights[i] + bigM * (1 - z[i]) for i in self.I)  # Height limit high

        # Solve the knapsack problem
        knapmodel.optimize()

        # Bailout if not feasible solution found
        if knapmodel.status != gp.GRB.OPTIMAL:
            return True

        # Check if a new pattern with negative reduced cost is found
        if knapmodel.objval < -0.0000001:

            # Generate a new pattern based on the solution of the sub problem
            new_pattern = np.array([[int(use[i].X)] for i in self.I])

            # Append the new pattern to the existing pattern matrix A: This horizontally stacks the new pattern to
            # the end of the matrix
            self._A = np.hstack((self._A, new_pattern))

            # Append the height of the new pattern to the H array
            self._H = np.concatenate((self._H, np.array([h.X])))

            # Calculate the total width of the new pattern: This is done by summing the width of each item in the
            # pattern multiplied by its usage (use[i].X). Then, flatten the array to ensure it's a 1D array
            W = np.array(np.sum([use[i].X * self.data.widths[i] for i in self.I])).flatten()
            # Append this total width to the W array. This adds the total width of the new pattern to the existing widths
            self._W = np.concatenate((self._W, W))

            # Append the roll width to the R array
            self._R = np.concatenate((self._R, np.array([self.roll_width])))

            return False
        else:
            # No more patterns with negative reduced cost, stop the process
            return True

    def _remove_duplicate_patterns(self):
        """
        Removes duplicate patterns from the pattern matrix A, and corresponding entries in H and W arrays.
        """
        old_shape = self._A.shape

        # Find unique patterns in A - keep indices and update corresponding arrays H, W and R
        self._A, unique_indices = np.unique(self._A, axis=1, return_index=True)

        self._H = self._H[unique_indices]
        self._W = self._W[unique_indices]
        self._R = self._R[unique_indices]

        print(f'=> Combined A is {self.A.shape} matrix after removing duplicates (from {old_shape})')


class ExtendedGlulamPatternProcessor(GlulamPatternProcessor):
    def __init__(self, data, roll_widths):
        """
        Initializes the ExtendedGlulamPatternProcessor with the necessary data and multiple roll widths.

        Parameters:
        - data (GlulamDataProcessor): An instance of the GlulamDataProcessor class, which contains the glulam data.
        - roll_widths (list): A list of roll widths to generate patterns for.
        """
        super().__init__(data, roll_width=None)  # Initialize the base class
        self.roll_widths = roll_widths
        self._generate_and_combine_patterns()

    def _generate_and_combine_patterns(self):
        """
        Generates and combines patterns for each roll width.
        """
        print(f'Generating patterns for roll widths: {self.roll_widths}')
        for i, roll_width in enumerate(self.roll_widths):
            pattern = GlulamPatternProcessor(self.data, roll_width)
            pattern.cutting_stock_column_generation()
            print(f'#{i} {roll_width}mm: A is {pattern.A.shape} matrix')

            self._A = np.hstack((self._A, pattern.A))
            self._H = np.concatenate((self._H, pattern.H))
            self._W = np.concatenate((self._W, pattern.W))
            self._R = np.concatenate((self._R, pattern.R))

        self._remove_duplicate_patterns()
        print(f'=> Combined A is {self.A.shape} matrix')
