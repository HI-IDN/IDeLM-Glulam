# config/settings.py
class GlulamConfig:
    """ Glulam factory settings, given in mm """

    # Press settings
    MAX_ROLL_WIDTH = 25000  # 25M roll width
    REGIONS = 2
    MAX_ROLL_WIDTH_REGION = [25000, 16000]  # Region 0 can be 25M, Region 1 can be 16M
    LAYER_HEIGHT = 45.0  # in mm
    MAX_HEIGHT_LAYERS = 26  # number of layers
    MIN_HEIGHT_LAYER_REGION = [11, 24]  # Region 0 must be at least 11 layers, Region 1 must be at least 24 layers
    MAX_PRESSES = 10  # maximum number of presses
    MIN_PRESSES = 5  # minimum number of presses

    # Cutting pattern settings
    SURPLUS_LIMIT = .10  # in pieces
    ROLL_WIDTH_TOLERANCE = 100  # in mm

    # Other settings
    DEFAULT_DEPTH = 115  # in mm

    # Algorithm settings
    GUROBI_TIME_LIMIT = 4*60  # in seconds, 2 minutes
    GUROBI_OUTPUT_FLAG = 0  # 0: silent, 1: summary, 2: detailed, 3: verbose
    VERBOSE_LOGGING = False
