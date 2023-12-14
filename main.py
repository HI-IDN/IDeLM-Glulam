# main.py
import argparse
from utils.data_processor import GlulamDataProcessor
from models.cutting_pattern import ExtendedGlulamPatternProcessor
from strategies.evolution_strategy import optimize_press_configuration
from models.pack_n_press import pack_n_press
from config.settings import GlulamConfig
import numpy as np


def main(file_path, depth):
    # Load and process data
    data = GlulamDataProcessor(file_path, depth)

    # Pack the patterns into presses that all look like this:
    wr = [(25000, 16000) for _ in range(7)]
    # Generate cutting patterns
    roll_widths = list(set(roll_width for configuration in wr for roll_width in configuration))
    merged = ExtendedGlulamPatternProcessor(data, roll_widths)
    H = merged.H
    W = merged.W
    A = merged.A
    waste, Lp = pack_n_press(A, data.quantity, H, W, wr)
    print(waste)

    # Optimize the press configuration
    # press_config = optimize_press_configuration()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glulam Production Optimizer")
    parser.add_argument(
        "--file", type=str, default="data/glulam.csv",
        help="Path to the data file (default: %(default)s)"
    )
    parser.add_argument(
        "--depth", type=int, default=GlulamConfig.DEFAULT_DEPTH,
        help="Depth to consider in mm (default: %(default)s)"
    )
    args = parser.parse_args()

    main(args.file, args.depth)
