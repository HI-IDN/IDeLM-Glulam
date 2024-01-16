import os
import sys

# Append project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import gurobipy as gp
from config.settings import GlulamConfig
from utils.data_processor import GlulamDataProcessor
import numpy as np
import argparse


def main(file_path, depth, presses_per_file, subset_folder):
    data = GlulamDataProcessor(file_path, depth)
    area_per_press = GlulamConfig.MAX_ROLL_WIDTH * GlulamConfig.MAX_HEIGHT_LAYERS * GlulamConfig.LAYER_HEIGHT / 1e6
    num_files = max(np.floor(data.area / (presses_per_file * area_per_press)).astype(int), 1)
    items = range(data.m)

    name = os.path.basename(file_path).split('.')[0]

    def save_subset(subset_data, filename):
        # Group by order, width, height, and layers and sum the quantity
        subset_data = subset_data.groupby(['order', 'depth', 'width', 'height', 'layers']).sum().reset_index()
        subset_data.to_csv(filename, index=False)
        print(f"Saved subset to {filename}")

    if num_files == 1:
        # Save the subset to a file
        filename = f'{subset_folder}/{name}_d{depth}.csv'
        save_subset(data._filtered_data, filename)

    else:
        files = range(1, num_files + 1)

        # Initialize model
        m = gp.Model("glulam_balancing")

        # Decision Variables
        x = m.addVars(items, files, vtype=gp.GRB.INTEGER, name="items")
        y = m.addVars(data.orders, files, vtype=gp.GRB.BINARY, name="order")
        total_area = m.addVars(files, vtype=gp.GRB.CONTINUOUS, name="total_area")
        order_count = m.addVars(files, vtype=gp.GRB.INTEGER, name="order_count")

        # Constraints
        # Ensure each item is assigned
        m.addConstrs((gp.quicksum(x[i, f] for f in files) == data.quantity[i] for i in items), name="item_assignment")
        # Balance the area per file
        m.addConstrs(
            (total_area[f] == gp.quicksum(x[i, f] * data.heights[i] * data.widths[i] for i in items) / 1e6 for f in
             files),
            name="area_balance")
        # Minimize orders per file
        m.addConstrs((order_count[f] == gp.quicksum(y[o, f] for o in data.orders) for f in files), name="order_count")

        # Map order to items
        M = 1e100
        for order in data.orders:
            m.addConstrs(
                (gp.quicksum(x[i, f] for i in items if data.order[i] == order) <= M * y[order, f] for f in files),
                name=f"order_{order}")

        # New decision variables
        max_area = m.addVar(vtype=gp.GRB.CONTINUOUS, name="max_area")
        min_area = m.addVar(vtype=gp.GRB.CONTINUOUS, name="min_area")

        # Update constraints
        # Ensuring max_area and min_area correctly represent the maximum and minimum total areas
        m.addConstrs((max_area >= total_area[f] for f in files), name="max_area_constr")
        m.addConstrs((min_area <= total_area[f] for f in files), name="min_area_constr")

        # Minimize the difference between max_area and min_area along as few orders per file as possible
        m.setObjective(
            gp.quicksum(order_count[f] for f in files) +
            (max_area - min_area), gp.GRB.MINIMIZE)

        # Solve
        m.optimize()

        if m.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible. Change number of presses to plan.")
            return

        # Print solution
        print(f"Total area: {sum(total_area[f].x for f in files)} for {data.area}")
        print(f"Objective: {m.objVal}")
        print("Solution:")
        for f in files:
            print(f"File {f}:")
            print(f"  Total area: {total_area[f].x:.2f} m^2")
            print(f"  Number of orders: {order_count[f].x:0f}")
            print(f"  Orders: {[o for o in data.orders if y[o, f].x > 0.1]}")
            filename = f'{subset_folder}/{name}_d{depth}_p{f}.csv'
            subset_items = [i for i in items if x[i, f].x > 0.01]
            subset_data = data._filtered_data.iloc[subset_items].copy()
            # Update the quantity based on the x values
            subset_data['quantity'] = [int(x[i, f].x) for i in subset_items]
            save_subset(subset_data, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process glulam data.")
    parser.add_argument("--input_file", type=str, default="data/glulam.csv",
                        help="Path to the input CSV file (default: %(default)s)")
    parser.add_argument("depth", type=int, help="Depth parameter for GlulamDataProcessor")
    parser.add_argument("--presses_per_file", type=int, default=5,
                        help="Number of presses per file (default: %(default)s)")
    parser.add_argument("--folder", type=str, default="data/subset",
                        help="Folder to save the output files (default: %(default)s)")

    args = parser.parse_args()
    os.makedirs(args.folder, exist_ok=True)
    main(args.input_file, args.depth, args.presses_per_file, args.folder)
