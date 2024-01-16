# main.py
import argparse

from utils.data_processor import GlulamDataProcessor
from strategies.evolution_strategy import EvolutionStrategy
from models.cutting_pattern import ExtendedGlulamPatternProcessor
from models.pack_n_press import GlulamPackagingProcessor
from config.settings import GlulamConfig
from utils.logger import setup_logger
import os


def main(file_path, depth, name, run, mode, overwrite, roll_widths, max_presses):
    """
    Main function to run the optimizer
    Args:
        file_path: Path to the data file
        depth: Depth to consider in mm
        name: Name of the experiment
        run: Run number of the experiment
        mode: Mode of operation
        overwrite: Overwrite existing files
        roll_widths: Roll widths to start the run
        max_presses: Maximum number of presses
    """
    logger = setup_logger('IDeLM-Glulam')
    logger.info("Starting the Glulam Production Optimizer")

    # File to save the solution
    file_ending = 'json' if mode == 'ES' else 'csv'
    if run is None:
        filename = f'data/{name}/soln_{mode}_d{depth}.{file_ending}'
    else:
        filename = f'data/{name}/soln_{mode}_d{depth}_{run}.{file_ending}'
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create the directory if it does not exist

    # Check if file exists and overwrite flag is not set
    if not overwrite and os.path.exists(filename):
        logger.info(f"File {filename} already exists. Use --overwrite to overwrite.")
        return
    else:
        logger.info(f"Running {mode} mode. Results will be saved in {filename}.")

    # Load and process data
    data = GlulamDataProcessor(file_path, depth)
    logger.info(f"Data loaded for depth: {depth}")
    logger.info(f"Number of items: {data.m}")

    if mode == "ES":
        # Evolutionary Search mode
        evolution_strategy = EvolutionStrategy(data, max_generations=GlulamConfig.ES_MAX_GENERATIONS)
        evolution_strategy.Search(filename, x=roll_widths)

    elif mode == "single":
        assert roll_widths is not None and len(roll_widths) > 0, "Roll widths must be provided in single run mode"
        logger.info(f"Running a single run mode with width: {roll_widths} roll widths")
        pattern = ExtendedGlulamPatternProcessor(data, max_presses)
        for roll_width in roll_widths:
            pattern.add_roll_width(roll_width)
        press = GlulamPackagingProcessor(pattern, max_presses)
        press.pack_n_press()
        press.print_results()
        press.save(filename, filename.replace('.csv', '.png'))
    else:
        logger.error(f"Unknown mode: {mode}")
        return


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
    parser.add_argument(
        '--roll_widths', nargs='+', type=int, default=None,
        help="Roll widths to start the run (default: %(default)s)"
    )
    parser.add_argument(
        "--max_presses", type=int, default=None,
        help="Maximum number of presses (default: %(default)s)"
    )
    args = parser.parse_args()

    main(args.file, args.depth, args.name, args.run, args.mode, args.overwrite, args.roll_widths, args.max_presses)
