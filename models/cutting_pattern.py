# models/cutting_pattern.py
# Import necessary libraries for Gurobi MIP modelling
from config.settings import GlulamConfig
import numpy as np
import gurobipy as gp
from tqdm import tqdm  # For progress bar

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

        self._A = None
        self._H = None
        self._W = None

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
        return [q for q, w in zip(self.data.quantity, self.data.widths) if w <= self.roll_width]

    @property
    def H(self):
        """ Pattern height """
        return self._H

    @property
    def W(self):
        """ Pattern total width """
        return self._W

    @property
    def I(self):
        """ Number of orders, i.e. the number of rows in the pattern matrix 0 <= i < m """
        return range(self.data.m)

    @property
    def J(self):
        """ Number of patterns, i.e. the number of columns in the pattern matrix 0 <= j < n """
        return range(self._A.shape[1])

    def cutting_stock_column_generation(self):
        """
        Solves the cutting stock problem using column generation.

        Returns:
        - A (matrix): Final pattern matrix.
        - x (dict): Quantities for each pattern to be cut.
        """

        def initial_pattern():
            """ Generates initial cutting patterns for each order. """
            A = np.zeros((self.data.m, self.data.m))
            H = np.zeros(self.data.m)
            W = np.zeros(self.data.m)
            for i in range(self.data.m):
                A[i, i] = np.floor(self.roll_width / self.data.widths[i])
                H[i] = self.data.heights[i]
                W[i] = self.data.widths[i] * A[i, i]

            # Final check to ensure all diagonal elements in A are greater than 0
            if not np.all(np.diag(A) > 0):
                raise ValueError("Invalid pattern matrix: Some diagonal elements in A are not greater than 0.")

            return A, H, W

        # Initialize the pattern matrix and bailout flag
        self._A, self._H, self._W = initial_pattern()
        bailout = False

        with tqdm(total=100, leave=False) as pbar:  # total is set to 100 for cycling.
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
                self._H = self._H[[j for j in self.J if x[j].X > 0.0000001]]
                self._W = self._W[[j for j in self.J if x[j].X > 0.0000001]]
                self._A = self._A[:, [j for j in self.J if x[j].X > 0.0000001]]

                # Solve the column generation sub problem
                self._A, self._H, self._W, bailout = self._column_generation_subproblem(pi)

                # Update the progress bar
                pbar.update(1)
                if pbar.n >= 100:
                    pbar.reset()

        pbar.close()

        # Return the cutting frequencies
        return {j: x[j].X for j in self.J}

    def _column_generation_subproblem(self, pi):
        """
        Solves the column generation subproblem for the cutting stock problem.

        Parameters:
        - pi (array): Dual prices from the master problem's constraints.

        Returns:
        - Updated A and bailout flag.
        """
        # A large number for big-M method in MIP
        bigM = 1000000
        Delta = 50

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
        knapmodel.addConstr(gp.quicksum(self.data.widths[i] * use[i] for i in self.I) >= self.roll_width - Delta)  # Width limit

        # Indicator constraints for height limits
        knapmodel.addConstrs(z[i] * bigM >= use[i] for i in self.I)  # If z[i] = 0, then use[i] = 0 (Indicator constr.)
        knapmodel.addConstrs(h >= self.data.heights[i] - bigM * (1 - z[i]) for i in self.I)  # Height limit low
        knapmodel.addConstrs(h <= self.data.heights[i] + bigM * (1 - z[i]) for i in self.I)  # Height limit high

        # Solve the knapsack problem
        knapmodel.optimize()

        # Check if a new pattern with negative reduced cost is found
        if knapmodel.objval < -0.0000001:
            # Add the new pattern to the matrix A
            new_pattern = np.array([[int(use[i].X)] for i in self.I])
            A = np.hstack((self._A, new_pattern))
            H = np.concatenate((self._H, np.array([h.X])))
            tmp = np.array(np.sum([use[i].X*self.data.widths[i] for i in self.I])).flatten()
            # print("debug: tmp=", tmp)
            W = np.concatenate((self._W, tmp))
            return A, H, W, False
        else:
            # No more patterns with negative reduced cost, stop the process
            return self._A, self._H, self._W, True


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
        self._A, self._H, self._W = None, None, None

        print(f'Generating patterns for roll widths: {self.roll_widths}')
        for i, roll_width in enumerate(self.roll_widths):
            pattern = GlulamPatternProcessor(self.data, roll_width)
            pattern.cutting_stock_column_generation()
            print(f'#{i} {roll_width}mm: A is {pattern.A.shape} matrix')

            if self._A is None:
                self._A, self._H, self._W = pattern.A, pattern.H, pattern.W
            else:
                self._A = np.hstack((self._A, pattern.A))
                self._H = np.concatenate((self._H, pattern.H))
                self._W = np.concatenate((self._W, pattern.W))

        print(f'=> Combined A is {self.A.shape} matrix')
