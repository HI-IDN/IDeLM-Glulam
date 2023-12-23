# main.py
import argparse
from utils.data_processor import GlulamDataProcessor
from models.cutting_pattern import ExtendedGlulamPatternProcessor
from strategies.evolution_strategy import optimize_press_configuration
from models.pack_n_press import GlulamPackagingProcessor
from config.settings import GlulamConfig
import numpy as np
from utils.logger import setup_logger
import pickle

def main(file_path, depth):
    logger = setup_logger('IDeLM-Glulam')
    logger.info("Starting the Glulam Production Optimizer")

    # Load and process data
    data = GlulamDataProcessor(file_path, depth)
    logger.info(f"Data loaded for depth: {depth}")
    logger.info(f"Number of items: {data.m}")

    # generate initial roll widths, say ten different configurations
    wr = [25000, 23600, 24500, 23800, 22600]
    roll_widths = []
    num_roll_widths = 8
    for i in range(num_roll_widths):
        roll_widths.append(np.random.randint(20000, 25000))
    best_solution = (100000,100000)
    num_generations = 10
    WR = np.zeros((num_generations+1, num_roll_widths))
    WR[0,:] = roll_widths
    for gen in range(1,num_generations+1):
        # Generate cutting patterns
        merged = ExtendedGlulamPatternProcessor(data)
        #logger.debug(f"Initial patterns have roll width {np.sort(merged.RW)} (n={merged.n})")
        for roll_width in roll_widths:
            #logger.info(f"Generating cutting patterns for roll width: {roll_width}")
            merged.add_roll_width(roll_width)
            #logger.info(f"Number of patterns: {merged.n}")

        press = GlulamPackagingProcessor(merged, 4)
        while not press.solved and press.number_of_presses < GlulamConfig.MAX_PRESSES:
            press.update_number_of_presses(press.number_of_presses + 1)
            press.pack_n_press()
            press.print_results()
            if press.solved:
                logger.info(f"Optimization completed successfully for {press.number_of_presses} presses.")
            else:
                logger.warning(f"Optimization did not reach a solution within {press.number_of_presses} presses.")
        if ((best_solution[0] > press.TotalWaste) and (best_solution[1] <= press.number_of_presses)) or (best_solution[1] > press.number_of_presses):
            best_solution = (press.TotalWaste, press.number_of_presses)
            logger.info(f"Number of patterns (current best): {merged.n}")
            logger.info(f"Best total waste so far: {best_solution[0]} with {best_solution[1]} presses.")
            # dump using pickle the current best solution, that is merged and press
            with open('best_solution.pkl'+str(gen), 'wb') as f:
                pickle.dump((merged, press), f)
        # summarize how many and which rolls are used
        for i in range(len(roll_widths)):
            rw = roll_widths[i]
            if rw not in press.RW_used:
                print(f"rollwidth {rw} is not used, remove it from the list of roll widths")
                merged.remove_roll_width(rw)
                print("A.shape=", merged.A.shape)
                WR[gen-1,i] = -WR[gen-1,i]
                roll_widths[i] = np.random.randint(10000, 25000) # need to play around with this search operator
        # now find one roll_width that is being used and replace it with a new one
        i = int(np.where(WR[gen-1,:] > 0)[0])
        roll_widths[i] = np.random.randint(10000, 25000)
        WR[gen,:] = roll_widths
            #print(roll_widths)
            #print(press.RW_used)
            #print(press.RW_counts)

    print(press.Waste)
    print("total waste = ", press.TotalWaste)
    # save the matrix WR using pickle
    with open('WR.pkl', 'wb') as f:
        pickle.dump(WR, f)

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
