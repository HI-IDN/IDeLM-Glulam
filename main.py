# main.py
import argparse
from utils.data_processor import GlulamDataProcessor
from models.cutting_pattern import ExtendedGlulamPatternProcessor
from strategies.evolution_strategy import optimize_press_configuration
from models.pack_n_press import GlulamPackagingProcessor
from config.settings import GlulamConfig
import numpy as np


def main(file_path, depth):
    # Load and process data
    data = GlulamDataProcessor(file_path, depth)

    # generate initial roll widths, say ten different configurations
    wr = [25000, 23600, 24500, 23800, 22600]

    for i in range(0):
        wr.append(np.floor((16000 + np.random.randint(0,
                                                      9000)) / GlulamConfig.ROLL_WIDTH_TOLERANCE) * GlulamConfig.ROLL_WIDTH_TOLERANCE)

    # Generate cutting patterns
    merged = ExtendedGlulamPatternProcessor(data)

    print(np.sort(merged.RW))

    roll_widths = [int(wr_) for wr_ in wr]
    for roll_width in roll_widths:
        merged.add_roll_width(roll_width)

    press = GlulamPackagingProcessor(merged, 0)
    while not press.solved and press.number_of_presses < GlulamConfig.MAX_PRESSES:
        press.update_number_of_presses(press.number_of_presses + 1)
        press.pack_n_press()
        press.print_results()

    # summarize how many and which rolls are used
    print("A.shape=", merged.A.shape)
    print(roll_widths)
    for i in range(len(roll_widths)):
        rw = roll_widths[i]
        if rw not in press.RW_used:
            print(f"rollwidth {rw} is not used, remove it from the list of roll widths")
            merged.remove_roll_width(rw)
            print("A.shape=", merged.A.shape)
            # removing rollwidths from the list of rollwidths
            roll_widths[i] = -roll_widths[i]
    print(roll_widths)
    print(press.RW_used)
    print(press.RW_counts)

    print(press.Waste)
    print("total waste = ", press.TotalWaste)


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
