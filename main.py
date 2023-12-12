# main.py
import argparse
from utils.data_processor import filter_glulam_data
from models.cutting_pattern import generate_cutting_pattern
from strategies.evolution_strategy import optimize_press_configuration
from models.packaging import optimize_packaging
from config.settings import GlulamConfig


def main(file_path, depth):
    # Load and process data
    processed_data = filter_glulam_data(file_path, depth)

    # Generate cutting patterns
    cutting_pattern = generate_cutting_pattern()

    # Optimize press configuration
    press_config = optimize_press_configuration()

    # Optimize packaging based on press configuration
    packaging = optimize_packaging()

    # Additional logic as needed


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
