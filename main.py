# main.py
import argparse
from utils.data_processor import GlulamDataProcessor
from models.cutting_pattern import GlulamPatternProcessor
from strategies.evolution_strategy import optimize_press_configuration
from models.packaging import GlulamPressProcessor
from models.pack_n_press import pack_n_press
from config.settings import GlulamConfig
import numpy as np

def main(file_path, depth):
    # Load and process data
    data = GlulamDataProcessor(file_path, depth)

    # Generate cutting patterns
    cutting_patterns1 = GlulamPatternProcessor(data, roll_width=25000)
    cutting_patterns1.cutting_stock_column_generation()
    A1 = cutting_patterns1.A
    H1 = cutting_patterns1.H
    W1 = cutting_patterns1.W
    cutting_patterns2 = GlulamPatternProcessor(data, roll_width=24000)
    cutting_patterns2.cutting_stock_column_generation()
    A2 = cutting_patterns2.A
    H2 = cutting_patterns2.H
    W2 = cutting_patterns2.W

    H = np.concatenate((H1,H2))
    W = np.concatenate((W1,W2))
    A = np.hstack((A1,A2))
    print(A1.shape, A2.shape, A.shape,H1.shape,H2.shape,W1.shape,W2.shape)

    # Pack the patterns into presses that all look like this:
    wr = [(25000, 16000) for _ in range(8)]
    waste = pack_n_press(A, data.orders, H, W, wr)
    print(waste)
    # Pack the patterns
    #press_processor = GlulamPressProcessor(cutting_patterns)
    #press_processor.optimize_packaging()
    #press_processor.print_results()
    #press_processor.save_results_to_csv('data/packaged_patterns.csv')

    # Optimize the press configuration
    #press_config = optimize_press_configuration()


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
