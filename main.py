# main.py
import argparse
from utils.data_processor import GlulamDataProcessor
from models.cutting_pattern import GlulamPatternProcessor
from strategies.evolution_strategy import optimize_press_configuration
from models.packaging import optimize_packaging
from config.settings import GlulamConfig


def main(file_path, depth):
    # Load and process data
    data = GlulamDataProcessor(file_path, depth)

    # Generate cutting patterns
    cutting_patterns = GlulamPatternProcessor(data)
    cutting_patterns.cutting_stock_column_generation()

    # Optimize press configuration
    press_config = optimize_packaging(cutting_patterns)

    # Optimize packaging based on press configuration
    packaging = optimize_press_configuration()


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
