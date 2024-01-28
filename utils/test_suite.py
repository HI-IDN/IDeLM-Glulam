import sys
import os
import numpy as np

# Append project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import unittest
from config.settings import GlulamConfig
from utils.data_processor import GlulamDataProcessor
from models.cutting_pattern import GlulamPatternProcessor, ExtendedGlulamPatternProcessor
from models.pack_n_press import GlulamPackagingProcessor
from utils.logger import setup_logger

# Setup logger
logger = setup_logger("UnitTest")


# Your test cases
class TestGlulamDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Setting up TestGlulamDataProcessor')
        cls.data = GlulamDataProcessor('data/glulam.csv', 115)

    def test_height_mismatch(self):
        # Sanity check: height should be a multiple of layer height
        self.assertTrue((GlulamConfig.LAYER_HEIGHT * self.data.layers == self.data.heights).all(),
                        "Height mismatch. Check input data.")

    def test_width_mismatch(self):
        # Sanity check: width should be less than ROLL_WIDTH
        self.assertTrue((GlulamConfig.MAX_ROLL_WIDTH >= self.data.widths).all(),
                        "Width mismatch. Check input data.")

    def test_m(self):
        self.assertTrue(self.data.m == 20, "There should be 20 items.")

    def test_orders(self):
        self.assertTrue((self.data.orders == ['SP703563', 'SP707443', 'SP708587', 'SP708588']),
                        "Orders don't match.")


class TestGlulamExtendedPatternProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Setting up TestGlulamPatternProcessor')
        cls.data = GlulamDataProcessor('data/glulam.csv', 115)
        cls.pattern = ExtendedGlulamPatternProcessor(cls.data)
        cls.pattern_smaller = GlulamPatternProcessor(cls.data, 16000)

    def check_number_of_patterns(self, pattern_processor, expected_num_patterns):
        self.assertEqual(pattern_processor.n, expected_num_patterns, f"Incorrect number of patterns.")

    def test_n(self):
        self.check_number_of_patterns(self.pattern, 34)
        self.check_number_of_patterns(self.pattern_smaller, 33)

    def test_W(self):
        self.assertTrue((self.data.widths @ self.pattern.A == self.pattern.W).all())
        self.assertTrue((self.data.widths @ self.pattern_smaller.A == self.pattern_smaller.W).all())

    def test_demand(self):
        def check_demand(pattern):
            for width, quantity, demand in zip(pattern.data.widths, pattern.data.quantity, pattern.b):
                if width > pattern.roll_width:
                    self.assertEqual(demand, 0, "b should be 0 when width is greater than roll width")
                else:
                    self.assertEqual(demand, quantity, "b should equal quantity when width is within roll width")

        self.assertTrue(self.pattern.roll_width == GlulamConfig.MAX_ROLL_WIDTH,
                        f"Merged pattern should have 25M roll width not {self.pattern.roll_width}.")
        check_demand(self.pattern)
        check_demand(self.pattern_smaller)

    def check_AHWR(self, pattern):
        self.assertTrue(pattern.A.shape == (pattern.m, pattern.n), "A should be a m x n matrix.")
        self.assertTrue(pattern.H.shape == (pattern.n,), "H should be a n x 1 vector.")
        self.assertTrue(pattern.W.shape == (pattern.n,), "W should be a n x 1 vector.")
        self.assertTrue(pattern.RW.shape == (pattern.n,), "RW should be a n x 1 vector.")

    def test_A(self):
        self.check_AHWR(self.pattern)
        self.check_AHWR(self.pattern_smaller)

    def test_add_rollwidth(self):
        merged = ExtendedGlulamPatternProcessor(self.data)
        self.check_number_of_patterns(merged, 34)
        merged.add_roll_width(24000)
        self.check_AHWR(merged)
        self.check_number_of_patterns(merged, 45)
        merged.remove_roll_width(24000)
        self.assertFalse(24000 in merged.roll_widths)
        self.check_AHWR(merged)
        self.check_number_of_patterns(merged, 35)


class TestGlulamPackagingProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Setting up GlulamPackagingProcessor')
        data = GlulamDataProcessor('data/glulam.csv', 115)
        pattern = ExtendedGlulamPatternProcessor(data)
        for roll_width in [25000, 23600, 24500, 23800, 22600]:
            pattern.add_roll_width(roll_width)
        cls.press = GlulamPackagingProcessor(pattern)

    def test_presses(self):
        def check_constraints():
            """
            Helper function to verify constraints in the GlulamPackagingProcessor tests.
            This method is NOT automatically invoked as a test.

            This function checks:
            1. If the demand is met by ensuring the product of A and x equals b.
            2. If the maximum height is not exceeded.
            3. If the minimum height is met for each region.
            4. If the maximum roll width is not exceeded.
            5. If the minimum roll width is met for each region.

            """
            # Check if demand is met
            self.assertTrue(np.dot(self.press.A, np.sum(self.press.x, axis=(1, 2)) == self.press.b).all(),
                            "Demand is not met.")

            # Check if height is within bounds
            self.assertTrue((self.press.h <= GlulamConfig.MAX_ROLL_WIDTH).flatten().all(), "Max height is exceeded.")
            for region in self.press.R:
                self.assertTrue(
                    (np.sum(self.press.h[:, range(0, region + 1)], axis=1)[:-1] >=
                     GlulamConfig.MIN_HEIGHT_LAYERS[region]).all(), f"Min height is not met in region {region}.")

            # Check if roll width is within bounds
            self.assertTrue((self.press.L_estimated <= GlulamConfig.MAX_ROLL_WIDTH).flatten().all(),
                            "Max roll width is exceeded.")
            self.assertTrue((self.press.L_actual <= GlulamConfig.MAX_ROLL_WIDTH).flatten().all(),
                            "Max roll width is exceeded.")
            for region in self.press.R:
                self.assertTrue(
                    (self.press.L_estimated[region] <= GlulamConfig.MAX_ROLL_WIDTH_REGION[region]).all(),
                    f"Max roll width is not met in region {region}, is {self.press.L_estimated[region]} "
                    f">{GlulamConfig.MAX_ROLL_WIDTH_REGION[region]}.")
                self.assertTrue(
                    (self.press.L_actual[region] <= GlulamConfig.MAX_ROLL_WIDTH_REGION[region]).all(),
                    f"Max roll width is not met in region {region}, is {self.press.L_actual[region]} "
                    f">{GlulamConfig.MAX_ROLL_WIDTH_REGION[region]}.")
                if region > 0:
                    self.assertTrue(
                        (self.press.L_estimated[:, region] < self.press.L_estimated[:, region - 1]).all(),
                        f"Roll width is not decreasing in region {region}.")

        self.press.update_number_of_presses(5)
        self.press.pack_n_press(10)
        self.assertFalse(self.press.solved, "Five presses are not enough")

        self.press.update_number_of_presses(7)
        self.press.pack_n_press(420)
        self.assertTrue(self.press.solved, "Seven presses are enough")
        self.assertTrue(round(self.press.ObjectiveValue) == 33030, f'Objective was {self.press.ObjectiveValue}.')
        self.press.print_results()
        check_constraints()

        self.press.update_number_of_presses(6)
        self.press.pack_n_press(420)
        self.assertTrue(self.press.solved, "Six presses are enough")
        self.assertTrue(round(self.press.ObjectiveValue) == 52470, f'Objective was {self.press.ObjectiveValue}.')
        self.press.print_results()
        check_constraints()


# This allows running the tests directly from this script
if __name__ == '__main__':
    unittest.main()
