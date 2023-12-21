import sys
import os

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

    def test_press(self):
        self.press.update_number_of_presses(5)
        self.press.pack_n_press(10)
        self.assertFalse(self.press.solved, "Five presses are not enough")

        self.press.update_number_of_presses(7)
        self.press.pack_n_press(420)
        self.assertTrue(self.press.solved, "Seven presses are enough")
        self.assertTrue(round(self.press.ObjectiveValue) == 33030, f'Objective was {self.press.ObjectiveValue}.')
        self.press.print_results()

        self.press.update_number_of_presses(6)
        self.press.pack_n_press(420)
        self.assertTrue(self.press.solved, "Six presses are enough")
        self.assertTrue(round(self.press.ObjectiveValue) == 52470, f'Objective was {self.press.ObjectiveValue}.')
        self.press.print_results()


# This allows running the tests directly from this script
if __name__ == '__main__':
    unittest.main()
