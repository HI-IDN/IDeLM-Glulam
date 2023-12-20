import sys
import os

# Append project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import logging
from config.settings import GlulamConfig


# Configure logger for the entire package
def setup_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if GlulamConfig.VERBOSE_LOGGING:
        logger.setLevel(logging.DEBUG)

    # Create a console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
