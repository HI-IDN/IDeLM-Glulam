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
    wr = [[GlulamConfig.MAX_ROLL_WIDTH_REGION[0]-np.random.choice(range(0,101,GlulamConfig.ROLL_WIDTH_TOLERANCE)),
            GlulamConfig.MAX_ROLL_WIDTH_REGION[1]-np.random.choice(range(0,101,GlulamConfig.ROLL_WIDTH_TOLERANCE))] 
            for _ in range(GlulamConfig.MAX_PRESSES)]
    wr_ = [GlulamConfig.MAX_ROLL_WIDTH_REGION for _ in range(GlulamConfig.MAX_PRESSES)]
    
    print(wr)
    # Generate cutting patterns
    roll_widths = list(set(roll_width for configuration in wr for roll_width in configuration))
    merged = ExtendedGlulamPatternProcessor(data, roll_widths)
    waste, true_waste, number_of_presses, Lp, delta = pack_n_press(merged, wr)
    #print(true_waste)
    print("total waste = ", np.sum(true_waste))
    waste, true_waste, number_of_presses, Lp, delta = pack_n_press(merged, wr_)
    #print(true_waste)
    print("total waste = ", np.sum(true_waste))


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
