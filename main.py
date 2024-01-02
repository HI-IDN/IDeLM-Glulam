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

# ML distribution as define by Gunter Rudolph
def mernd(S, n=None):
    # Calculate probability p for the geometric distributions
    p = 1 - (S) / (np.sqrt(1 + (S)**2) + 1)
    # Generate two sets of geometrically distributed random numbers and subtract
    G1 = np.floor(np.log(np.random.rand()) / np.log(1 - p))
    G2 = np.floor(np.log(np.random.rand()) / np.log(1 - p))
    Z = G1 - G2
    return Z, p

# the mutation operator should create a new vector x_ and a new vector s_
# the values in x_ should be unique note that we may re-create a parent in this case
# the Search algorithm should delete the parent and replace it with the child !?
def Mutate(x, s, tau, nmin, nmax, dn):
    x_ = np.zeros(len(x))
    s_ = s.copy()
    for i in range(len(x)):
        s_[i] = s[i]*np.exp(tau*np.random.randn())
        retry = 0
        while x_[i] == 0 or x_[i] < nmin or x_[i] > nmax or x_[i] in x_[:i] or x_[i] in x:
            x_[i] = x[i] + dn*mernd(s_[i])[0] #                                ^^^^^^^^^^^^^
            retry += 1
            if retry > 100:
                print("mutation failed for :", x[i], s_[i])
                break # note that this will return x_[i] as 0 !!!
    return x_, s_

def Selection(x, s, press):
    # extract the roll widths used in the press
    I = range(len(x))
    rw_used = np.array([x[i] for i in I if x[i] in press.RW_used])
    sa_used = np.array([s[i] for i in I if x[i] in press.RW_used])

    return rw_used, sa_used

# The objective function assumes the MIN_PRESSES is to small to fit all orders
# if the number of presses are too few or the patterns just do not fit then the
# objective function will return (None, None)
def Objective(merged):
    press = GlulamPackagingProcessor(merged, GlulamConfig.MIN_PRESSES)
    objective = (None, None)
    while not press.solved and press.number_of_presses < GlulamConfig.MAX_PRESSES:
        press.update_number_of_presses(press.number_of_presses + 1)
        press.pack_n_press()
        if press.solved:
            press.print_results()
            objective = (press.TotalWaste, press.number_of_presses)
    return objective, press

# The Search algorithm is a simple (1+1)-ES using self-adaptive mutation
# note that there is one problem with this approach, namely that the
# step size may not adapt if the parent is not killed.
def Search(data, max_generations=100, tau=1.0/(np.sqrt(2)), sigma0=5, lamba=10, n_max=GlulamConfig.MAX_ROLL_WIDTH-GlulamConfig.ROLL_WIDTH_TOLERANCE, dn=GlulamConfig.ROLL_WIDTH_TOLERANCE, nmin=0):
    # generate packing patterns from input data
    merged = ExtendedGlulamPatternProcessor(data)
    # generate initial unique roll widths, say lamba different configurations
    # this is by default the best solution found so far
    x = np.random.choice(range(0, n_max, dn),size=lamba, replace=False)
    sigma = sigma0*np.ones(lamba)
    for roll_width in x:
        merged.add_roll_width(roll_width)
    (waste, npresses), press = Objective(merged)
    if waste is None:
        print("The ES intialization failed to find a feasible solution, retry?")
        return None
    xstar, sstar = Selection(x, sigma, press)
    # now lets start the search, for max max_generations
    STATS = [(xstar,sstar,waste,npresses,x,sigma,press,merged,0)]
    for gen in range(1,max_generations): 
        lamba = len(xstar) # each parent generates one child (could be more in theory)
        x_, s_ = Mutate(xstar, sstar, tau, nmin, n_max, dn)
        # here we need to check if x_ is zero and if to se should reset this parameter
        for i in range(len(x_)):
            if x_[i] == 0:
                s_[i] = sigma
                while x_[i] == 0 or x_[i] < nmin or x_[i] > n_max or x_[i] in x_[:i] or x_[i] in x:
                    x_[i] = np.random.choice(range(0, n_max, dn), size = 1, replace = False)
        # remove roll widths that are not used in the press
        for i in range(len(x)):
            if x[i] not in xstar:
                merged.remove_roll_width(x[i])
        # add the new roll widths to the press
        x = np.concatenate((xstar, x_))
        sigma = np.concatenate((sstar, s_))
        for i in range(len(x)):
            if x[i] not in xstar:
                merged.add_roll_width(x[i])
        # evaluate the new roll widths
        (waste_, npresses_), press = Objective(merged)
        if waste_ is not None:
            if (waste_ <= waste and npresses_ <= npresses) or (npresses_ < npresses):
                xstar, sstar = Selection(x, sigma, press)
                waste, npresses = waste_, npresses_
                print("Generation", gen)
                print("new best solution found with waste =", waste, "and npresses =", npresses)
                print("the roll widths are", xstar)
                print("the step sizes are", sstar)
                print("the number of roll widths is", len(xstar))
                print("the number of patterns is", merged.n)
                print("the number of presses is", npresses)
                print("the total waste is", waste)
        STATS.append((xstar,sstar,waste,npresses,x,sigma,press,merged,gen))

    return xstar, sstar, STATS         
def main(file_path, depth):
    logger = setup_logger('IDeLM-Glulam')
    logger.info("Starting the Glulam Production Optimizer")

    # Load and process data
    data = GlulamDataProcessor(file_path, depth)
    logger.info(f"Data loaded for depth: {depth}")
    logger.info(f"Number of items: {data.m}")

    # generate initial roll widths, say ten different configurations
    wr = [25000, 23600, 24500, 23800, 22600]
    xstar, sstar, STATS = Search(data, max_generations = 100)
    with open('best_solution.pkl', 'wb') as f:
        pickle.dump((xstar, sstar, STATS), f)
    
    with open('best_solution.pkl', 'rb') as f:
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
    args = parser.parse_args()

    main(args.file, args.depth)
