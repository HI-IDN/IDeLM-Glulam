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
def Mutate(x, s, tau, tau_, nmin, nmax, dn):
    x_ = np.zeros(len(x))
    s_ = s.copy()
    eta1 = tau_*np.random.randn() # global steps size
    for i in range(len(x)):
        s_[i] = s[i]*np.exp(eta1+tau*np.random.randn())
        retry = 0
        while x_[i] == 0 or x_[i] < nmin or x_[i] > nmax or x_[i] in x_[:i] or x_[i] in x:
            x_[i] = x[i] + dn*mernd(s_[i])[0] #                                ^^^^^^^^^^^^^
            retry += 1
            if retry > 100:
                print("mutation failed for :", x[i], s_[i])
                x_[i] = 0
                break # note that this will return x_[i] as 0 !!!
    return x_, s_

def Selection(x, s, success, press):
    # extract the roll widths used in the press
    I = range(len(x))
    rw_used = np.array([x[i] for i in I if x[i] in press.RW_used])
    sa_used = np.array([s[i] for i in I if x[i] in press.RW_used])
    success_used = np.array([success[i] for i in I if x[i] in press.RW_used])
    return rw_used, sa_used, success_used

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
def Search(data, x = None, max_generations = 100, alpha = 0.1, sigma0=5, lamba=10, n_max=GlulamConfig.MAX_ROLL_WIDTH-GlulamConfig.ROLL_WIDTH_TOLERANCE, dn=GlulamConfig.ROLL_WIDTH_TOLERANCE, nmin=0):
    # generate packing patterns from input data
    merged = ExtendedGlulamPatternProcessor(data)
    # generate initial unique roll widths, say lamba different configurations
    # this is by default the best solution found so far
    if x is None:
        x = np.random.choice(range(0, n_max, dn),size=lamba, replace=False)
    sigma = sigma0*np.ones(lamba)
    success = np.ones(lamba)
    for roll_width in x:
        merged.add_roll_width(roll_width)
    (waste, npresses), press = Objective(merged)
    if waste is None:
        print("The ES intialization failed to find a feasible solution, retry?")
        print("Try increasing maximum number of presses", GlulamConfig.MAX_PRESSES)
        return None
    xstar, sstar, sucstar = Selection(x, sigma, success, press)
    # now lets start the search, for max max_generations
    STATS = [(xstar,sstar,waste,npresses,x,sigma,press,merged,0)]
    for gen in range(1,max_generations): 
        print("Generation: ", gen, "/", max_generations)
        lamba = len(xstar) # each parent generates one child (could be more in theory)
        tau_ = 1/np.sqrt(2*len(xstar))
        tau = 1/np.sqrt(2*np.sqrt(len(xstar)))
        # use the one-fith rule to adapt the step size
        for i in range(len(xstar)):
            if sucstar[i] > 5:
                sstar[i] = sstar[i]*0.8
                sucstar[i] = 1        
        x_, s_ = Mutate(xstar, sstar, tau, tau_, nmin, n_max, dn)
        # here we need to check if x_ is zero and if to se should reset this parameter
        iremove = []
        for i in range(len(x_)):
            if x_[i] in xstar:
                # find the index of the roll width in xstar
                #j = np.where(xstar == x_[i])[0][0]
                #sstar[j] = sstar[j] + alpha*(s_[i] - sstar[j]) # facilitate self-adaptation
                # remove entry i from x_ and s_
                iremove.append(i)
            if x_[i] == 0: # reset the variable using initialization procedure
                s_[i] = sigma0
                while x_[i] == 0 or x_[i] < nmin or x_[i] > n_max or x_[i] in x_[:i] or x_[i] in x:
                    x_[i] = np.random.choice(range(0, n_max, dn), size = 1, replace = False)
        # remove roll widths that are not used in the press
        for i in range(len(x)):
            if x[i] not in xstar:
                merged.remove_roll_width(x[i])
        # remove i entries stored in iremove from x_ and s_
        x_ = np.delete(x_, iremove)
        s_ = np.delete(s_, iremove)
        # concatenate the parents and children
        x = np.concatenate((xstar, x_))
        sigma = np.concatenate((sstar, s_))
        sucstar = sucstar + 1
        success = np.concatenate((sucstar, np.ones(len(x_)))) # increment success parameter for parents
        for i in range(len(x)):
            if x[i] not in xstar:
                merged.add_roll_width(x[i])
        # evaluate the new roll widths
        (waste_, npresses_), press = Objective(merged)
        if waste_ is not None:
            if (waste_ <= waste and npresses_ <= npresses) or (npresses_ < npresses):
                xstar, sstar, sucstar = Selection(x, sigma, success, press)
                waste, npresses = waste_, npresses_
                print("new best solution found with waste =", waste, "and npresses =", npresses)
                print("the roll widths are", xstar)
                print("the step sizes are", sstar)
                print("the successes are", sucstar)
                print("the number of roll widths is", len(xstar))
                print("the number of patterns is", merged.n)
                print("the number of presses is", npresses)
                print("the total waste is", waste)
        STATS.append((xstar,sstar,sucstar,waste,npresses,x,sigma,press,merged,gen))

    return xstar, sstar, STATS         
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
    xstar, sstar, STATS = Search(data, x = None, max_generations = 100)
    with open(name + '_' + str(depth) + '/soln_' + str(run) + '.pkl', 'wb') as f:
        pickle.dump((xstar, sstar, STATS), f)
    
    with open(name + '_' + str(depth) + '/soln_' + str(run) + '.pkl', 'rb') as f:
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
