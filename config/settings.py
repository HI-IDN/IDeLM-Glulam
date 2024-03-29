# config/settings.py
class GlulamConfig:
    """ Glulam factory settings, given in mm """

    # Press settings
    MAX_ROLL_WIDTH = 25000  # 25M roll width
    OLD_NEW_SPLIT_WIDTH = 16000  # 16M roll width
    REGIONS = 2
    MINIMUM_REGION_DIFFERENCE = 2000  # in mm
    LAYER_HEIGHT = 45.0  # in mm
    MAX_HEIGHT_LAYERS = 26  # number of layers
    MIN_HEIGHT_OLD_PRESS = 11
    MIN_HEIGHT_NEW_PRESS = 24

    # Cutting pattern settings
    BUFFER_LIMIT = MAX_HEIGHT_LAYERS - MIN_HEIGHT_NEW_PRESS
    ROLL_WIDTH_TOLERANCE = 100  # in mm

    # Other settings
    DEFAULT_DEPTH = 90  # in mm

    # Algorithm settings
    GUROBI_TIME_LIMIT = 20 * 60  # in seconds
    GUROBI_NO_IMPROVEMENT_TIME_LIMIT = 1 * 60  # in seconds
    GUROBI_OUTPUT_FLAG = 0  # 0: silent, 1: summary, 2: detailed, 3: verbose
    VERBOSE_LOGGING = False
    ES_MAX_GENERATIONS = 100
    SAVE_PRESS_TO_PICKLE = False
