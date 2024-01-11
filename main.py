# main.py
import argparse
import pickle

from utils.data_processor import GlulamDataProcessor
from strategies.evolution_strategy import Search
from config.settings import GlulamConfig
from utils.logger import setup_logger
import os


def main(file_path, depth, name, run, mode, overwrite):
    logger = setup_logger('IDeLM-Glulam')
    logger.info("Starting the Glulam Production Optimizer")

    # Load and process data
    data = GlulamDataProcessor(file_path, depth)
    logger.info(f"Data loaded for depth: {depth}")
    logger.info(f"Number of items: {data.m}")

    # File to save the solution
    if run is None:
        filename = f'data/{name}/soln_{mode}_d{depth}.pkl'
    else:
        filename = f'data/{name}/soln_{mode}_d{depth}_{run}.pkl'  # Save the solution to a json file
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create the directory if it does not exist

    # Check if file exists and overwrite flag is not set
    if not overwrite and os.path.exists(filename):
        logger.info(f"File {filename} already exists. Use --overwrite to overwrite.")
        return
    else:
        logger.info(f"Running {mode} mode. Results will be saved in {filename}.")

    if mode == "ES":
        # Evolutionary Search mode
        results = Search(data, x=None, max_generations=GlulamConfig.ES_MAX_GENERATIONS)

    elif mode == "single":
        wr = [22800, 23000, 23500, 23600, 23700, 24900]
        logger.info(f"Running a single run mode with width: {wr} roll widths")
        results = Search(data, x=wr, max_generations=1)
    else:
        logger.error(f"Unknown mode: {mode}")
        return

    # Save the solution
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved the solution to {filename}")


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
    parser.add_argument(
        "--name", type=str, default="tmp",
        help="name of experiment (default: %(default)s)"
    )
    parser.add_argument(
        "--run", type=int, default=None,
        help="name number of experiment (default: %(default)s)"
    )
    parser.add_argument(
        "--mode", type=str, default="ES", choices=["ES", "single"],
        help="Mode of operation (default: %(default)s)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing files (default: %(default)s)"
    )

    args = parser.parse_args()

    main(args.file, args.depth, args.name, args.run, args.mode, args.overwrite)
