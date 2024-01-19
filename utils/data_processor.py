# utils/data_processor.py
import pandas as pd
import numpy as np
from config.settings import GlulamConfig
import logging


class GlulamDataProcessor:
    def __init__(self, file_path, depth):
        raw_data = pd.read_csv(file_path)

        # Check if the depth is available in the data
        available_depths = raw_data['depth'].unique()
        assert depth in available_depths, f"Depth {depth} mm not found. Available depths are: {available_depths}"

        # Filter and update the filtered data
        self._filtered_data = raw_data[raw_data['depth'] == depth]
        self.depth = depth

        assert all(self._filtered_data['layers'] * GlulamConfig.LAYER_HEIGHT == self._filtered_data['height']), \
            "Height mismatch: layers * layer_height != height"

        # Check if there are any beams with height greater than the maximum allowed height
        # Warn the user and set the height to the maximum allowed height
        too_tall = self._filtered_data['layers'] > GlulamConfig.MAX_HEIGHT_LAYERS
        if too_tall.any():
            logging.warning(f"The following beams are too tall and will be truncated to "
                            f"{GlulamConfig.MAX_HEIGHT_LAYERS} layers:\n{self._filtered_data[too_tall]}")
            self._filtered_data.loc[too_tall, 'layers'] = GlulamConfig.MAX_HEIGHT_LAYERS

        # Reset index
        self._filtered_data.reset_index(drop=True, inplace=True)

        # Priority items are those defined in the file, but if they are all the same order, then there is no priority
        # it is the index for the items with column 'priority' set to True (but only if there are some items with False)
        self._priority_items = []
        priority_items = self._filtered_data['priority']
        if priority_items.any() and not priority_items.all():
            self._priority_items = np.where(priority_items)[0]

    @property
    def area(self):
        """ Compute the minimum area needed to fit all orders in square meters. """
        return np.sum(self.quantity * self.heights * self.widths) / 1e6

    @property
    def depths(self):
        return np.array(self._filtered_data['depth'].tolist(), dtype=int)

    @property
    def widths(self):
        return np.array(self._filtered_data['width'].tolist(), dtype=int)

    @property
    def heights(self):
        return np.array(self._filtered_data['height'].tolist(), dtype=int)

    @property
    def layers(self):
        return np.array(self._filtered_data['layers'].tolist(), dtype=int)

    @property
    def quantity(self):
        return np.array(self._filtered_data['quantity'].tolist(), dtype=int)

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

    @property
    def priority_items(self):
        """ Set of priority items. """
        return self._priority_items


def convert_numpy_to_json(items):
    """ Convert numpy types to json compatible types. """
    if isinstance(items, np.ndarray):
        return convert_numpy_to_json(items.tolist())
    if isinstance(items, np.int64):
        return int(items)
    if isinstance(items, list):
        return [convert_numpy_to_json(x) for x in items]
    if isinstance(items, dict):
        return {k: convert_numpy_to_json(v) for k, v in items.items()}
    if isinstance(items, int) or isinstance(items, float) or isinstance(items, str):
        return items
