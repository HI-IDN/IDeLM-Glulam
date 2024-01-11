# evolution_strategy.py
import logging

import numpy as np
from models.cutting_pattern import ExtendedGlulamPatternProcessor
from models.pack_n_press import GlulamPackagingProcessor
from config.settings import GlulamConfig

from utils.logger import setup_logger
from utils.data_processor import convert_numpy_to_json

# Setup logger
logger = setup_logger('GlulamES')


class EvolutionStrategy:
    """ Evolution Strategy class based on the (1+1)-ES algorithm """

    def __init__(self, data, max_generations=None, alpha=0.1, sigma0=5, lamda=10, n_max=None, dn=None, n_min=0):
        self.max_generations = max_generations or GlulamConfig.ES_MAX_GENERATIONS
        """ Maximum number of generations to be used in the search. """

        self.alpha = alpha
        """ Step size adaptation parameter. """

        self.sigma0 = sigma0
        """ Initial step size for the (1+1)-ES algorithm. """

        self.lamda = lamda
        """ Lambda parameter for the (1+1)-ES algorithm. """

        self.n_max = n_max or GlulamConfig.MAX_ROLL_WIDTH - GlulamConfig.ROLL_WIDTH_TOLERANCE
        """ Maximum roll width to be used in the search. """

        self.dn = dn or GlulamConfig.ROLL_WIDTH_TOLERANCE
        """ Roll width step size. """

        self.n_min = n_min
        """ Minimum roll width to be used in the search. """

        self.stats = None
        """ Statistics of the search. """

        # generate packing patterns from input data
        self.merged = ExtendedGlulamPatternProcessor(data)
        """ The merged pattern processor. """

        area_press = GlulamConfig.MAX_ROLL_WIDTH * GlulamConfig.MAX_HEIGHT_LAYERS * GlulamConfig.LAYER_HEIGHT / 1e6
        self.min_presses = np.ceil(self.merged.data.area / area_press).astype(int)
        """ The minimum number of presses needed to pack all orders. """

        self.npresses = None
        """ The number of presses used in the best solution found so far. """

        self.waste = None
        """ The total waste in the best solution found so far. """

        self.xstar = None
        """ The roll widths used in the best solution found so far. """

        self.sstar = None
        """ The step sizes used in the best solution found so far. """

        self.sucstar = None
        """ The number of successes used in the best solution found so far. """

    def mernd(self, S):
        """
        This function generates two geometrically distributed random numbers and subtracts them to obtain a random
        number from the ML distribution as defined by Gunter Rudolph

        Args:
            S (float): the step size parameter
        Returns:
            Z (float): a random number from the ML distribution
            p (float): the probability of success p for the geometric distribution
        """

        # Calculate probability p for the geometric distributions
        p = 1 - (S) / (np.sqrt(1 + (S) ** 2) + 1)
        # Generate two sets of geometrically distributed random numbers and subtract
        G1 = np.floor(np.log(np.random.rand()) / np.log(1 - p))
        G2 = np.floor(np.log(np.random.rand()) / np.log(1 - p))
        Z = G1 - G2
        return Z, p

    def Mutate(self, x, s, tau, tau_):
        """
        The mutation operator should create a new vector x_ and a new vector s_
        the values in x_ should be unique note that we may re-create a parent in this case
        the Search algorithm should delete the parent and replace it with the child !?

        Args:
            x (list): the roll widths
            s (list): the step sizes
            tau (float): the step size parameter
            tau_ (float): the step size parameter
        Returns:
            x_ (list): the mutated roll widths
            s_ (list): the mutated step sizes
        """
        x_ = np.zeros(len(x))
        s_ = s.copy()
        eta1 = tau_ * np.random.randn()  # global steps size
        for i in range(len(x)):
            s_[i] = s[i] * np.exp(eta1 + tau * np.random.randn())
            retry = 0
            while x_[i] == 0 or x_[i] < self.n_min or x_[i] > self.n_max or x_[i] in x_[:i] or x_[i] in x:
                x_[i] = x[i] + self.dn * self.mernd(s_[i])[0]
                retry += 1
                if retry > 100:
                    logger.error(f"Mutation failed for : x={x[i]} s={s[i]}")
                    x_[i] = 0
                    break  # note that this will return x_[i] as 0 !!!
        return x_, s_

    def Selection(self, x, s, success, press):
        """
        The selection operator selects the best roll widths from the population by selecting the roll widths used in
        the successful press. And updates the values for xstar, sstar, and sucstar.

        Args:
            x (list): the roll widths
            s (list): the step sizes
            success (list): the number of successes
            press (GlulamPackagingProcessor): the press object

        """
        logger.info(f"Selection - Updating xstar, sstar, and sucstar based on the roll widths used in the press.")

        # extract the roll widths used in the press
        I = range(len(x))
        rw_used = np.array([x[i] for i in I if x[i] in press.RW_used])
        sa_used = np.array([s[i] for i in I if x[i] in press.RW_used])
        success_used = np.array([success[i] for i in I if x[i] in press.RW_used])

        self.xstar = rw_used
        self.sstar = sa_used
        self.sucstar = success_used

    def Objective(self):
        """
        The objective function assumes the min presses is too small to fit all orders
        if the number of presses are too few or the patterns just do not fit then the
        objective function will return (None, None)

        Returns:
            objective (tuple): (waste, npresses) where
            - waste is the total waste and
            - npresses is the minimum number of presses needed for a feasible solution
            press (GlulamPackagingProcessor): the press object
        """
        logger.info(f"Objective - Finding the minimum number of presses needed for a feasible solution.")
        press = GlulamPackagingProcessor(self.merged, self.min_presses - 1)
        objective = (None, None)
        while not press.solved:
            press.update_number_of_presses(press.number_of_presses + 1)
            press.pack_n_press()
            if press.solved:
                logger.info(f"Objective - Found a feasible solution with {press.number_of_presses} presses "
                            f"and total waste {press.TotalWaste}.")
                press.print_results()
                objective = (press.TotalWaste, press.number_of_presses)

        return objective, press

    def _add_stats(self, x, sigma, gen):
        if self.stats is None:
            self.stats = {'xstar': [], 'sstar': [], 'sucstar': [], 'waste': [], 'npresses': [], 'x': [], 'sigma': [],
                          'gen': []}
        self.stats['xstar'].append(self.xstar)
        self.stats['sstar'].append(self.sstar)
        self.stats['sucstar'].append(self.sucstar)
        self.stats['waste'].append(self.waste)
        self.stats['npresses'].append(self.npresses)
        self.stats['x'].append(x)
        self.stats['sigma'].append(sigma)
        self.stats['gen'].append(gen)
        logger.info(
            f"Stats - Generation {gen} - waste = {self.waste}, npresses = {self.npresses}, "
            f"xstar = {self.xstar} (#{len(self.xstar)})")

    def Search(self, x=None):
        """
        The Search algorithm is a simple (1+1)-ES using self-adaptive mutation
        note that there is one problem with this approach, namely that the
        step size may not adapt if the parent is not killed.
        """
        logger.info(f"Initialising the Evolutionary Search")

        # generate initial unique roll widths, say Objective different configurations
        # this is by default the best solution found so far
        if x is None:
            logger.info(f"Randomly generating {self.lamda} roll widths")
            x = np.random.choice(range(0, self.n_max, self.dn), size=self.lamda, replace=False)
        elif isinstance(x, list):
            logger.info(f"Using {len(x)} roll widths from input")
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            logger.error(f"Unknown type for x: {type(x)}")
            return None
        else:
            logger.info(f"Using {len(x)} roll widths from input")

        logger.info(f"Initialising the search parameters")
        sigma = self.sigma0 * np.ones(self.lamda)
        success = np.ones(self.lamda)
        for roll_width in x:
            self.merged.add_roll_width(roll_width)
        (self.waste, self.npresses), press = self.Objective()

        if self.waste is None:
            logger.error("The ES initialization failed to find a feasible solution, retry?")
            return None

        # Initialise the algorithm
        self.Selection(x, sigma, success, press)

        # now lets start the search, for max max_generations
        self._add_stats(x, sigma, 0)
        for gen in range(1, self.max_generations):
            logger.info(f"Generation: {gen}/{self.max_generations}")

            self.lamda = len(self.xstar)  # each parent generates one child (could be more in theory)
            tau_ = 1 / np.sqrt(2 * len(self.xstar))
            tau = 1 / np.sqrt(2 * np.sqrt(len(self.xstar)))
            # use the one-fifth rule to adapt the step size
            for i in range(len(self.xstar)):
                if self.sucstar[i] > 5:
                    self.sstar[i] = self.sstar[i] * 0.8
                    self.sucstar[i] = 1
            x_, s_ = self.Mutate(self.xstar, self.sstar, tau, tau_)
            # here we need to check if x_ is zero and if to se should reset this parameter
            iremove = []
            for i in range(len(x_)):
                if x_[i] in self.xstar:
                    # find the index of the roll width in xstar
                    # j = np.where(self.xstar == x_[i])[0][0]
                    # self.sstar[j] = self.sstar[j] + self.alpha*(s_[i] - self.sstar[j]) # facilitate self-adaptation
                    # remove entry i from x_ and s_
                    iremove.append(i)
                if x_[i] == 0:  # reset the variable using initialization procedure
                    s_[i] = self.sigma0
                    while x_[i] == 0 or x_[i] < self.n_min or x_[i] > self.n_max or x_[i] in x_[:i] or x_[i] in x:
                        x_[i] = np.random.choice(range(0, self.n_max, self.dn), size=1, replace=False)

            # remove roll widths that are not used in the press
            for i in range(len(x)):
                if x[i] not in self.xstar:
                    self.merged.remove_roll_width(x[i])

            # remove i entries stored in iremove from x_ and s_
            x_ = np.delete(x_, iremove)
            s_ = np.delete(s_, iremove)

            # concatenate the parents and children
            x = np.concatenate((self.xstar, x_))
            sigma = np.concatenate((self.sstar, s_))
            self.sucstar = self.sucstar + 1
            success = np.concatenate((self.sucstar, np.ones(len(x_))))  # increment success parameter for parents
            for i in range(len(x)):
                if x[i] not in self.xstar:
                    self.merged.add_roll_width(x[i])

            # evaluate the new roll widths
            (waste_, npresses_), press = self.Objective()
            # Update the best solution found so far if the new solution is better
            if waste_ is not None:
                if (waste_ <= self.waste and npresses_ <= self.npresses) or (npresses_ < self.npresses):
                    self.Selection(x, sigma, success, press)
                    self.waste, self.npresses = waste_, npresses_
                    logger.info(f"NEW BEST: solution found with waste = {self.waste} and npresses = {self.npresses}")
                    logger.info(f"NEW BEST: the roll widths are {self.xstar} (# = {len(self.xstar)})")
                    logger.info(f"NEW BEST: the step sizes are {self.sstar}")
                    logger.info(f"NEW BEST: the successes are {self.sucstar}")
                    logger.info(f"NEW BEST: the number of patterns is {self.merged.n}")

            self._add_stats(x, sigma, gen)

        logger.info(f"Search - Finished the search after {self.max_generations} generations.")

        results = {'roll_widths': self.xstar, 'presses': self.npresses, 'waste': self.waste, 'stats': self.stats,
                   'depth': self.merged.data.depth, 'n': self.merged.n, 'm': self.merged.m}
        return convert_numpy_to_json(results)
