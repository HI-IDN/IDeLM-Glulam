# config/settings.py
class GlulamConfig:
    """ Glulam factory settings, given in mm """

    # Press settings
    MAX_ROLL_WIDTH = 25000  # 25M roll width
    REGIONS = 2
    MINIMUM_REGION_DIFFERENCE = 0  # in mm
    MAX_ROLL_WIDTH_REGION = [25000, 16000]  # Region 0 can be 25M, Region 1 can be 16M
    LAYER_HEIGHT = 45.0  # in mm
    MAX_HEIGHT_LAYERS = 26  # number of layers
    MIN_HEIGHT_LAYER_REGION = [11, 26]  # Region 0 must be at least 11 layers, Region 1 must be at least 24 layers

    # Cutting pattern settings
    SURPLUS_LIMIT = .10  # in pieces
    BUFFER_LIMIT = 2  # in pieces
    ROLL_WIDTH_TOLERANCE = 100  # in mm

    # Other settings
    DEFAULT_DEPTH = 90  # in mm

    # Algorithm settings
    GUROBI_TIME_LIMIT = 20 * 60  # in seconds
    GUROBI_NO_IMPROVEMENT_TIME_LIMIT = 1 * 60  # in seconds
    GUROBI_OUTPUT_FLAG = 0  # 0: silent, 1: summary, 2: detailed, 3: verbose
    VERBOSE_LOGGING = False
    ES_MAX_GENERATIONS = 200
