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
            self.roll_width = GlulamConfig.ROLL_WIDTH
        else:
            self.roll_width = roll_width
            assert roll_width <= GlulamConfig.ROLL_WIDTH, f"Roll width {roll_width} mm exceeds the maximum roll width {GlulamConfig.ROLL_WIDTH} mm."

        self._A = None

    @property
    def A(self):
        """ Pattern matrix """
        return self._A

    @property
    def I(self):
        """ Number of orders """
        return range(self.data.m)

    @property
    def J(self):
        """ Number of patterns """
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
            for i in range(self.data.m):
                A[i, i] = np.floor(self.roll_width / self.data.widths[i])
            return A

        # Initialize the pattern matrix and bailout flag
        self._A = initial_pattern()
        bailout = False

        with tqdm(total=100, leave=False) as pbar:  # total is set to 100 for cycling
            while not bailout:
                # Create the cutting model
                cutmodel = gp.Model("Cutting")
                cutmodel.setParam('OutputFlag', 0)

                # Decision variables: how often to cut each pattern
                x = cutmodel.addVars(self.J)  # Note this a continuous variable in order to get shadow prices later

                # Objective: Minimize the total number of patterns used
                cutmodel.setObjective(gp.quicksum(x[j] for j in self.J), gp.GRB.MINIMIZE)

                # Constraints: Ensure all orders are satisfied
                cutmodel.addConstrs(gp.quicksum(self._A[i, j] * x[j] for j in self.J) >= self.data.orders[i]
                                    for i in self.I)

                # Solve the master problem
                cutmodel.optimize()

                # Retrieve the dual prices from the constraints
                pi = [c.Pi for c in cutmodel.getConstrs()]

                # Solve the column generation subproblem
                self._A, bailout = self._column_generation_subproblem(pi)

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

        # Initialize the knapsack model
        knapmodel = gp.Model("Knapsack")
        knapmodel.setParam('OutputFlag', 0)  # Suppress output for cleaner execution

        # Decision variables for the knapsack problem
        use = knapmodel.addVars(self.I, lb=0, vtype=gp.GRB.INTEGER)  # Pattern usage variables
        y = knapmodel.addVar()  # Height of the roll
        z = knapmodel.addVars(self.I, vtype=gp.GRB.BINARY)  # Binary variables for height constraints

        # Objective: Minimize reduced cost
        knapmodel.setObjective(1.0 - gp.quicksum(pi[i] * use[i] for i in self.I), gp.GRB.MINIMIZE)

        # Width constraint: Total width used must not exceed roll width
        knapmodel.addConstr(gp.quicksum(self.data.widths[i] * use[i] for i in self.I) <= self.roll_width)  # Width limit

        # Indicator constraints for height limits
        knapmodel.addConstrs(z[i] * bigM >= use[i] for i in self.I)  # If z[i] = 0, then use[i] = 0 (Indicator constr.)
        knapmodel.addConstrs(y >= self.data.heights[i] - bigM * (1 - z[i]) for i in self.I)  # Height limit low
        knapmodel.addConstrs(y <= self.data.heights[i] + bigM * (1 - z[i]) for i in self.I)  # Height limit high

        # Solve the knapsack problem
        knapmodel.optimize()

        # Check if a new pattern with negative reduced cost is found
        if knapmodel.objval < -0.0000001:
            # Add the new pattern to the matrix A
            new_pattern = np.array([[use[i].X] for i in self.I])
            A = np.hstack((self._A, new_pattern))
            return A, False
        else:
            # No more patterns with negative reduced cost, stop the process
            return self._A, True
