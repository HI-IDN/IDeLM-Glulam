# main.py
import argparse
from utils.data_processor import GlulamDataProcessor
from models.cutting_pattern import ExtendedGlulamPatternProcessor
from strategies.evolution_strategy import Search
from models.pack_n_press import GlulamPackagingProcessor
from config.settings import GlulamConfig
import numpy as np
from utils.logger import setup_logger
import pickle


def main(file_path, depth, name, run):
    logger = setup_logger('IDeLM-Glulam')
    logger.info("Starting the Glulam Production Optimizer")

    # Load and process data
    data = GlulamDataProcessor(file_path, depth)
    logger.info(f"Data loaded for depth: {depth}")
    logger.info(f"Number of items: {data.m}")

    # generate initial roll widths, say ten different configurations
    wr = [25000, 23600, 24500, 23800, 22600]
    wr = np.array([22800, 23000, 23500, 23600, 23700, 24900])
    xstar, sstar, STATS = Search(data, x=None, max_generations=GlulamConfig.ES_MAX_GENERATIONS)

    filename = name + '_' + str(depth) + '/soln_' + str(run) + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump((xstar, sstar, STATS), f)

    with open(filename, 'rb') as f:
        (xstar, sstar, STATS) = pickle.load(f)


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
        "--run", type=int, default=0,
        help="name number of experiment (default: %(default)s)"
    )
    args = parser.parse_args()

    main(args.file, args.depth, args.name, args.run)
