import pandas as pd
import numpy as np
import gurobipy as gp
from config.settings import GlulamConfig


class GlulamPressProcessor:
    """
    Optimizes the packaging of the glulam orders.
    """

    def __init__(self, pat):
        self.pat = pat
        self.results_df = None
        self.missing_order_info = None

    def optimize_packaging(self):
        """
        Optimizes the packaging of the glulam orders.
        """
        cut_model = gp.Model("Cutting final model")
        x = cut_model.addVars(self.pat.J, lb=0, vtype=gp.GRB.CONTINUOUS, name="x")
        cut_model.setObjective(gp.quicksum(x[j] for j in self.pat.J), gp.GRB.MINIMIZE)
        cut_model.addConstrs(
            gp.quicksum(self.pat.A[i, j] * x[j] for j in self.pat.J) >= self.pat.b[i] for i in self.pat.I)
        cut_model.optimize()

        # Extract the integer solution via flooring
        self.x = np.floor([x[j].X for j in self.pat.J]).astype(int)
        """ x is the number of times each pattern is cut (integer). """

        # Generate the resulting patterns and calculate the missing orders
        self.results_df = self._generate_patterns()
        self.missing_order_info = self._calculate_missing_orders()

    def _generate_patterns(self):
        """
        Generates patterns based on the cutting patterns and the integer solution from the cutting stock problem.
        """
        assert self.x is not None, "Please run optimize_packaging() first."

        # Initialise results list and press ID and height
        results = []
        press_id = press_height = 0

        # Loop through the patterns
        for j in self.pat.J:
            for _ in range(self.x[j]):
                pattern = self.pat.A[:, j]
                tot_length = np.sum(pattern * self.pat.data.widths)
                height_idx = np.argmax(pattern > 0)
                height = self.pat.data.heights[height_idx] / GlulamConfig.LAYER_HEIGHT

                if press_height + height > GlulamConfig.MAX_HEIGHT:
                    press_height = 0
                    press_id += 1

                press_height += height
                row = [j, height, tot_length, press_id, press_height] + pattern.tolist()
                results.append(row)

        # Convert the results to a DataFrame
        column_names = ['PatternID', 'Height', 'TotalLength', 'PressID', 'PressHeight'] + \
                       ['Order' + str(i) for i in range(self.pat.data.m)]
        df = pd.DataFrame(results, columns=column_names).astype(int)
        return df

    def _calculate_missing_orders(self):
        """
        Calculates the missing orders and the number of rolls used.
        """
        assert self.x is not None
        missing_order = (self.pat.b - self.pat.A @ self.x).astype('int')
        missing_per_roll = np.ceil(missing_order @ self.pat.data.widths / self.pat.roll_width).astype('int')
        total_rolls_used = int(np.sum(self.x)) + missing_per_roll

        return {'Orders': missing_order.tolist(),
                'MissingPerRoll': missing_per_roll,
                'Rolls': total_rolls_used
                }

    def save_results_to_csv(self, filepath):
        """ Saves the results to a CSV file. """
        self.results_df.to_csv(filepath, index=False)

    def print_results(self):
        """ Prints the results and missing order information. """
        assert self.results_df is not None, "Please run optimize_packaging() first."
        print("Patterns:")
        print(self.results_df)

        print("Missing orders:")
        print("\n".join(
            f"\tOrder #{i}: {self.missing_order_info['Orders'][i]} / {self.pat.b[i]}"
            for i in range(self.pat.data.m) if self.missing_order_info['Orders'][i] > 0)
        )

        print(f"At least {self.missing_order_info['MissingPerRoll']} rolls are needed to satisfy the missing orders.")
        print(f"Grand total of {self.missing_order_info['Rolls']} rolls are needed.")
