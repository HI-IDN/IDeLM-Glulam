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
    #wr_ = [GlulamConfig.MAX_ROLL_WIDTH_REGION for _ in range(GlulamConfig.MAX_PRESSES)]

    #wr = [[GlulamConfig.MAX_ROLL_WIDTH_REGION[0]-i*GlulamConfig.ROLL_WIDTH_TOLERANCE] 
    #        for i in range(len(GlulamConfig.MAX_ROLL_WIDTH_REGION)*GlulamConfig.MAX_PRESSES)]
    wr = [[np.floor((16000 + np.random.randint(0, 9000))/GlulamConfig.ROLL_WIDTH_TOLERANCE)*GlulamConfig.ROLL_WIDTH_TOLERANCE]
                for i in range(10)]
    # randomly generate we between 16000 and 25000 in the discretization of ROLL_WIDTH_TOLERANCE

    print(wr)


    # Generate cutting patterns
    merged = ExtendedGlulamPatternProcessor(data)
    roll_widths = list(set(roll_width for configuration in wr for roll_width in configuration))
    for roll_width in roll_widths[:2]:
        merged.add_roll_width(roll_width)

    number_of_presses = int(GlulamConfig.MAX_PRESSES) - 1
    success = False
    while success == False:
        number_of_presses += 1
        success, waste, Lp = pack_n_press(merged, number_of_presses)

    print(waste)
    print("total waste = ", np.sum(waste))
    #waste, true_waste, number_of_presses, Lp, delta = pack_n_press(merged, wr_)
    #print(true_waste)
    #print("total waste = ", np.sum(true_waste))

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
