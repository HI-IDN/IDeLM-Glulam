# utils/data_processor.py
import pandas as pd
import numpy as np
from config.settings import GlulamConfig


class GlulamDataProcessor:
    def __init__(self, file_path, depth):
        raw_data = pd.read_csv(file_path)

        # Check if the depth is available in the data
        available_depths = raw_data['depth'].unique()
        assert depth in available_depths, f"Depth {depth} mm not found. Available depths are: {available_depths}"

        # Filter and update the filtered data
        self._filtered_data = raw_data[raw_data['depth'] == depth]

        # Sanity check: height should be a multiple of layer height
        assert (GlulamConfig.LAYER_HEIGHT * self._filtered_data['layers'] == self._filtered_data['height']).all(), \
            "Height mismatch. Check input data."

        # Sanity check: width should be less than ROLL_WIDTH
        assert (GlulamConfig.MAX_ROLL_WIDTH >= self._filtered_data['width']).all(), \
            "Width mismatch. Check input data."

        # Reset index
        self._filtered_data.reset_index(drop=True, inplace=True)

    @property
    def widths(self):
        return np.array(self._filtered_data['width'].tolist())

    @property
    def heights(self):
        return np.array(self._filtered_data['height'].tolist())

    @property
    def quantity(self):
        return np.array(self._filtered_data['quantity'].tolist())

    @property
    def m(self):
        """ Number of orders. """
        return len(self._filtered_data)

    @property
    def order(self):
        """ Name of the order. """
        return np.array(self._filtered_data['order'].tolist())

    @property
    def orders(self):
        """ Set of orders. """
        return sorted(list(set(self.order.tolist())))
